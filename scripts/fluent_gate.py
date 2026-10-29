"""Verify, gate and summarise the fluency-constrained GPU search (docs/prereg/fluent_search.md).

1. Every text the kernel saved (generated, final and the pool of accepted / highest-cos states) is re-embedded with the
   local ONNX encoder (cache keyed by text and encoder fingerprint, as scripts/bounds/verify_support.py does) and
   scored against the target; the ONNX score is the only score reported.
2. Validity: exact domain format (DOMAINS[d].render of the parsed fields gives the text back) and the corpus-p99
   bounds of latent_gap.coherent; gibberish flag of latent_gap.coherent.
3. Coherence gate per target and lambda, as scripts/coherent_search.py: distinct valid non-gibberish candidates in
   descending ONNX score; judge 1 (claude-haiku-5-5), then judge 2 (codex:gpt-6-luna) if judge 1 accepts; stop at
   the first that both rate >= 7; at most 15 candidates. Subscription CLIs only (latent_gap.cli_judges), no gateway.
4. Tables and pre-registered verdicts.

Usage: uv run python scripts/fluent_gate.py --run full [--workers 4] [--no-judge] [--minutes 60]
Input: runs/fluent-search/<run>/kaggle-out/fluent_gpu*.jsonl. Output in runs/fluent-search/<run>/.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.cli_judges import RateLimitedError
from latent_gap.coherent import bounds_violation, gibberish_rule, gibberish_stats, judge
from latent_gap.domains import DOMAINS
from latent_gap.embedder import MODEL_DIR, LocalQwen

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
ROOT = DATA / "runs/fluent-search"
JUDGE1, JUDGE2, COH_PASS, CAP = "claude:claude-haiku-5-5", "codex:gpt-6-luna", 7, 15
# Pre-registered thresholds (docs/prereg/fluent_search.md)
REAL_SHARE, COVERAGE = 0.80, 0.90
FAR_WITHIN, FAR_WITHIN_SHARE, FAR_ESCAPE, FAR_ESCAPE_SHARE = 0.03, 0.80, 0.05, 0.20
GAP_MARGIN, GAP_SHARE = 0.02, 0.50


def parse(domain: str, text: str) -> dict | None:
    if domain == "steam":
        if not text.startswith("Genres: ") or ". Tags: " not in text:
            return None
        g, rest = text[len("Genres: "):].split(". Tags: ", 1)
        if ". About: " not in rest:
            return None
        tg, about = rest.split(". About: ", 1)
        item = {"name": "", "genres": g.split(","), "tags": tg.split(","), "about": about}
    else:
        if not text.startswith("Title: ") or ". Abstract: " not in text:
            return None
        title, ab = text[len("Title: "):].split(". Abstract: ", 1)
        item = {"title": title, "abstract": ab}
    return item if DOMAINS[domain].render(item) == text else None


def md(df: pd.DataFrame) -> str:
    df = df.fillna("-")
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns),
                      *["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]])


def fmt(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--minutes", type=float, default=600, help="wall-clock budget for ONNX embedding (resumable)")
    args = ap.parse_args()
    run = ROOT / args.run
    rows = [json.loads(line) for f in sorted((run / "kaggle-out").glob("fluent_gpu*.jsonl"))
            for line in f.read_text().splitlines() if line.strip()]
    print(f"{len(rows)} target x lambda records", flush=True)
    refs = pd.read_json(ROOT / "refs.jsonl", lines=True)
    refs_d = {(r.domain, r.arm, r.k): r for r in refs.itertuples()}
    jobs = {(j["domain"], j["arm"], j["k"]): j for j in json.loads((ROOT / "inputs/fluent_inputs.json").read_text())}

    # ---------------- 1-2. candidates, ONNX scores, validity, gibberish
    model = LocalQwen()
    sig = hashlib.sha256((inspect.getsource(LocalQwen) + str([
        (str(p.relative_to(MODEL_DIR)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(MODEL_DIR.rglob("*")) if p.is_file()])).encode()).hexdigest()[:16]
    cache = ROOT / "verified_cache" / sig
    cache.mkdir(parents=True, exist_ok=True)
    corpus = {}
    for d in {r["domain"] for r in rows}:
        meta = pd.read_parquet(DATA / "prepared" / d / "corpus.parquet", columns=["split", "adult"])
        E = np.load(DATA / "prepared" / d / "emb_qwen3e06.npy").astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        corpus[d] = E[np.flatnonzero((~meta.adult & (meta.split == "past")).to_numpy())]
        del E
    deadline = time.monotonic() + 60 * args.minutes
    cands = []
    n_new = 0
    for r in rows:
        key = (r["domain"], r["arm"], r["k"])
        t = np.asarray(jobs[key]["target"], dtype=np.float32)
        t /= np.linalg.norm(t)
        hidden = refs_d[key].hidden_pos
        texts = {}
        for c in ([{"text": r["generated"], "cos": None, "nll": None, "src": "generated"}]
                  + [dict(c, src="side" if c.get("side") else "accepted") for c in r["pool"]]):
            texts.setdefault(c["text"], c)
        for text, c in texts.items():
            sha = hashlib.sha256(text.encode()).hexdigest()
            f = cache / f"{sha}.npy"
            if f.exists():
                e = np.load(f)
            else:
                if time.monotonic() > deadline:
                    raise SystemExit("ONNX budget used up; rerun to resume from the cache")
                e = model.embed_one(text)
                np.save(f, e)
                n_new += 1
                if n_new % 500 == 0:
                    print(f"  embedded {n_new} new texts", flush=True)
            sims = corpus[r["domain"]] @ e
            if len(hidden):
                sims[hidden] = -np.inf
            item = parse(r["domain"], text)
            n_tok = len(model.tok.encode(text).ids)
            bad = "format" if item is None else bounds_violation(r["domain"], item, n_tok)
            st = gibberish_stats(text)
            cands.append({"domain": r["domain"], "arm": r["arm"], "k": r["k"], "lam": r["lam"], "text": text,
                          "sha256": sha, "src": c["src"], "gpu_cos": c["cos"], "nll": c["nll"],
                          "score": float(e @ t), "corpus_max": float(sims.max()), "n_tokens": n_tok,
                          "valid": not bad, "status": bad or "ok", "gib_rule": gibberish_rule(st), **st})
    C = pd.DataFrame(cands)
    C["gibberish"] = C.gib_rule != ""
    print(f"{len(C)} candidates ({n_new} newly embedded); valid {C.valid.mean():.3f}, gibberish {C.gibberish.mean():.3f}; "
          f"max |ONNX - GPU| {(C.score - C.gpu_cos).abs().max():.4f}", flush=True)

    # ---------------- 3. coherence gate
    jpath = run / "judgements.jsonl"
    jcache: dict[tuple[str, str], dict] = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            h = json.loads(line)
            jcache[h["judge"], h["sha256"]] = h
    lock = threading.Lock()

    def rate(spec: str, domain: str, sha: str, text: str) -> int | None:
        with lock:
            hit = jcache.get((spec, sha))
        if hit is None:
            hit = judge(spec, DOMAINS[domain].noun, text) | {"judge": spec, "sha256": sha}
            if hit.get("coherence") is not None:
                with lock:
                    jcache[spec, sha] = hit
                    with jpath.open("a") as f:
                        f.write(json.dumps(hit) + "\n")
        return hit.get("coherence")

    def walk(g: pd.DataFrame) -> dict:
        d, arm, k, lam = g.iloc[0][["domain", "arm", "k", "lam"]]
        out = {"domain": d, "arm": arm, "k": k, "lam": lam, "h_coh": np.nan, "coh_text": None, "n_judged": 0,
               "coh_corpus_max": np.nan, "coh_nll": np.nan}
        if args.no_judge:
            return out
        ok = g[g.valid & ~g.gibberish].drop_duplicates("sha256").sort_values("score", ascending=False).head(CAP)
        for c in ok.itertuples():
            out["n_judged"] += 1
            try:
                a = rate(JUDGE1, d, c.sha256, c.text)
                b = rate(JUDGE2, d, c.sha256, c.text) if (a or 0) >= COH_PASS else None
            except RateLimitedError as e:
                print(f"usage limit outlasted the waits on {d} {arm} {k} lam={lam}: {e}; left undone", flush=True)
                out["undone"] = True
                return out
            if (a or 0) >= COH_PASS and (b or 0) >= COH_PASS:
                out.update(h_coh=c.score, coh_text=c.text, coh_corpus_max=c.corpus_max, coh_nll=c.nll,
                           coh_j1=a, coh_j2=b)
                break
        print(f"  gate {d} {arm} {k} lam={lam}: h_coh {fmt(out['h_coh'])} after {out['n_judged']} candidates",
              flush=True)
        return out

    groups = [g for _, g in C.groupby(["domain", "arm", "k", "lam"])]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        G = pd.DataFrame(list(ex.map(walk, groups)))
    print(f"gate done in {(time.time() - t0) / 60:.1f} min", flush=True)
    C.drop(columns=["text"]).to_csv(run / "candidates.csv", index=False)
    G.to_csv(run / "gate.csv", index=False)

    # ---------------- 4. per target table
    per_lam = C.groupby(["domain", "arm", "k", "lam"]).agg(
        h_any=("score", "max"), med_nll=("nll", "median")).reset_index().merge(G, on=["domain", "arm", "k", "lam"])
    best = per_lam.groupby(["domain", "arm", "k"]).agg(h_coh=("h_coh", "max")).reset_index()
    any0 = per_lam[per_lam.lam == 0].groupby(["domain", "arm", "k"]).h_any.max().rename("h_any_l0").reset_index()
    T = refs.drop(columns="hidden_pos").merge(best, on=["domain", "arm", "k"]).merge(any0, on=["domain", "arm", "k"],
                                                                                      how="left")
    sb = DATA / "runs/support-search/support_bounds.csv"
    if sb.exists():
        T = T.merge(pd.read_csv(sb)[["domain", "arm", "k", "h_any_lower"]], on=["domain", "arm", "k"], how="left")
    cs = DATA / "runs/coherent_search_targets.csv"
    if cs.exists():
        c = pd.read_csv(cs)
        T = T.merge(c[c.run == "claude-sonnet-5-5"][["domain", "arm", "k", "h_coh", "landing"]].rename(
            columns={"h_coh": "h_coh_blind_sonnet"}), on=["domain", "arm", "k"], how="left")
    T["ceiling_star"] = np.maximum(T.ceiling, T.heldout_ceiling)
    T.to_csv(run / "targets.csv", index=False)
    lam_wide = per_lam.pivot_table(index=["domain", "arm", "k"], columns="lam", values="h_coh").add_prefix("h_coh_l")
    lam_wide.to_csv(run / "h_coh_by_lambda.csv")

    lines = [f"# fluent search summary, run {args.run}", ""]
    tab = []
    for (d, arm), g in T.groupby(["domain", "arm"], sort=False):
        pl = per_lam[(per_lam.domain == d) & (per_lam.arm == arm)]
        row = {"domain": d, "arm": arm, "n": len(g), "ceiling": fmt(g.ceiling.median()),
               "corpus q99": fmt(g.corpus_q99.median()),
               "anchored landing": fmt(g.landing.median()) if "landing" in g else "-",
               "blind h_coh (Sonnet)": fmt(g.h_coh_blind_sonnet.median()) if "h_coh_blind_sonnet" in g else "-",
               "h_any support-search": fmt(g.h_any_lower.median()) if "h_any_lower" in g else "-",
               "h_any (lam=0, this format)": fmt(g.h_any_l0.median())}
        for lam, gl in pl.groupby("lam"):
            row[f"h_coh lam={lam:g}"] = f"{fmt(gl.h_coh.median())} ({gl.h_coh.notna().sum()}/{len(gl)})"
        row["h_coh (max over lam)"] = f"{fmt(g.h_coh.median())} ({g.h_coh.notna().sum()}/{len(g)})"
        row["h_coh - ceiling"] = fmt((g.h_coh - g.ceiling).median())
        tab.append(row)
    lines += ["Medians over targets (h_coh: best gate-passing ONNX score; (measured/targets))", "", md(pd.DataFrame(tab)),
              ""]
    fr = per_lam.groupby(["domain", "arm", "lam"]).agg(h_any=("h_any", "median"), nll=("med_nll", "median"),
                                                        h_coh=("h_coh", "median"),
                                                        measured=("h_coh", lambda s: s.notna().mean())).reset_index()
    lines += ["Frontier (medians over targets per lambda; nll = median LM NLL of the run's states, nats/token)", "",
              md(fr.round(3)), ""]
    nr = run / "kaggle-out/nll_reference.jsonl"
    if nr.exists():
        ref = pd.read_json(nr, lines=True)
        lines += ["Corpus NLL reference: " + ", ".join(f"{d} median {g.nll.median():.3f} [p10 {g.nll.quantile(.1):.3f},"
                                                     f" p90 {g.nll.quantile(.9):.3f}]" for d, g in ref.groupby("domain")), ""]

    verdicts = []
    for d, g in T.groupby("domain"):
        real, far, gap = g[g.arm == "REAL"], g[g.arm == "FAR"], g[g.arm == "GAP"]
        share = float((real.h_coh >= real.corpus_q99).mean()) if len(real) else np.nan
        powered = bool(len(real) and share >= REAL_SHARE)
        verdicts.append(f"- {d} C (REAL control): {'POWERED' if powered else 'UNDERPOWERED'}: h_coh >= corpus q99 on "
                        f"{share:.0%} of {len(real)} (needs >= {REAL_SHARE:.0%}); h_coh >= ceiling on "
                        f"{(real.h_coh >= real.ceiling).mean():.0%}")
        if len(far):
            cov = far.h_coh.notna().mean()
            esc = (far.h_coh >= far.ceiling_star + FAR_ESCAPE).mean()
            within = (far.h_coh <= far.ceiling_star + FAR_WITHIN).mean()  # NaN -> False
            if esc >= FAR_ESCAPE_SHARE:
                v = f"objection REFUTED in its strong form: coherent text >= ceiling* + {FAR_ESCAPE} on {esc:.0%}"
            elif cov < COVERAGE:
                v = f"insufficient coverage ({cov:.0%} < {COVERAGE:.0%}): no verdict"
            elif not powered:
                v = "uninformative: REAL control failed (optimiser underpowered)"
            elif within >= FAR_WITHIN_SHARE:
                v = (f"objection SUPPORTED: a powered fluent search stays within ceiling* + {FAR_WITHIN} on "
                     f"{within:.0%} of FAR targets")
            else:
                v = f"intermediate: within +{FAR_WITHIN} on {within:.0%}, escape +{FAR_ESCAPE} on {esc:.0%}"
            verdicts.append(f"- {d} P1 FAR: {v} (coverage {cov:.0%}; median h_coh - ceiling* "
                            f"{(far.h_coh - far.ceiling_star).median():+.3f})")
        if len(gap):
            cov = gap.h_coh.notna().mean()
            reached = (gap.h_coh >= gap.ceiling - GAP_MARGIN).mean()
            esc = (gap.h_coh >= gap.ceiling_star + FAR_ESCAPE).mean()
            if cov < COVERAGE:
                v = f"insufficient coverage ({cov:.0%})"
            elif reached > GAP_SHARE:
                v = f"gaps REACHABLE: h_coh >= ceiling - {GAP_MARGIN} on {reached:.0%}"
            elif powered:
                v = f"not reached by a powered fluent search (reached on {reached:.0%})"
            else:
                v = "inconclusive: optimiser underpowered"
            verdicts.append(f"- {d} P2 GAP: {v}; beyond ceiling* + {FAR_ESCAPE} on {esc:.0%}")
    lines += ["Verdicts (pre-registered rules)", "", *verdicts, ""]
    (run / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
