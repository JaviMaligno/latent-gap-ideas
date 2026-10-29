"""Seed-and-rewrite: an LLM rewriter turns the fluent GPU search's best (incoherent) texts into coherent pages, then
moves them towards the target with cosine feedback (a number only). Pre-registration: docs/prereg/fluent_search.md
(revision 3).

Seeds: per target, the top-S distinct texts by ONNX score from the GPU search's lambda in {0, 0.05} runs (the kernel
never saw a corpus text). The rewriter never sees a corpus text or neighbour: only the seed, its own previous best
rewrite and that rewrite's cosine. Rewriters: subscription CLIs only (claude-sonnet-5-5 via Claude Code, gpt-6-luna
via Codex; latent_gap.cli_judges), never the gateway. Scoring: local ONNX encoder (LocalQwen). Gate: as
fluent_search (distinct valid non-gibberish candidates in descending score; claude-haiku-5-5, then codex:gpt-6-luna if
the first accepts; both >= 7; at most 15).

Usage:
  uv run python scripts/seed_rewrite.py --label tune-v1 --set tune --writer claude:claude-sonnet-5-5 --domain steam
  uv run python scripts/seed_rewrite.py --label test --set test --writer codex:gpt-6-luna --domain arxiv
  uv run python scripts/seed_rewrite.py --label test --summary      # tables and verdicts over every run of a label
Sets: tune = REAL k=11..20 (seeds from the tune-seeds-<d> kernel runs); test = FAR/GAP k=1..20 and REAL k=1..10
(seeds from runs/fluent-search/full for arXiv and steam4b for Steam). Resumable: one checkpoint per target.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.cli_judges import CliChat, RateLimitedError, RefusedError
from latent_gap.coherent import bounds_violation, gibberish_rule, gibberish_stats, judge
from latent_gap.domains import DOMAINS
from latent_gap.embedder import MODEL_DIR, LocalQwen

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
FS = DATA / "runs/fluent-search"
OUT = DATA / "runs/seed-rewrite"
JUDGE1, JUDGE2, COH_PASS, CAP = "claude:claude-haiku-5-5", "codex:gpt-6-luna", 7, 15
SEED_RUNS = {"test": {"steam": "steam4b", "arxiv": "full"}, "tune": {"steam": "tune-seeds-steam",
                                                                     "arxiv": "tune-seeds-arxiv"}}
SETS = {"test": {"FAR": range(1, 21), "GAP": range(1, 21), "REAL": range(1, 11)}, "tune": {"REAL": range(11, 21)}}
SEED_LAMBDAS = (0.0, 0.05)
# Pre-registered thresholds (as fluent_search)
REAL_SHARE, COVERAGE = 0.80, 0.90
FAR_WITHIN, FAR_WITHIN_SHARE, FAR_ESCAPE, FAR_ESCAPE_SHARE = 0.03, 0.80, 0.05, 0.20
GAP_MARGIN, GAP_SHARE = 0.02, 0.50

PROMPT_VERSION = "v2"  # v1 (tune-v1): no length cap, no polish round; arXiv failed the tuning control
SYSTEM = ("You write {noun} descriptions. You always answer with JSON and nothing else. Each {noun} is one JSON object "
          "with exactly these fields: {spec}")
FIRST = ("Below is a rough, partly garbled draft of a {noun} description, produced by an automatic search procedure. "
         "Rewrite it into a real, sensible {noun} description in normal English, keeping as much of its specific "
         "content as possible: the same subject, setting, {what}, named things{lists} and distinctive words and "
         "phrases. Drop only what cannot be made sensible, and do not add unrelated material. Every named method, "
         "dataset, benchmark, mechanic or term must make sense together with the rest; leave out names that do not fit. "
         "No lists of random words, no keyword stuffing, no other languages. Keep the {body} under 170 words."
         "\n\n<draft>\n{seed}\n</draft>\n\nAnswer with the JSON object only.")
AGAIN = ("We want a {noun} description as close as possible to a hidden target point in an embedding space. The draft "
         "below came from a search procedure steered towards that target. Your previous rewrite of it and its cosine "
         "similarity to the target (1.0 = exactly on the target) are shown. Write a new rewrite that should score "
         "higher: bring back specific content of the draft that the previous rewrite dropped, softened or generalised, "
         "while keeping it a real, sensible {noun} description in normal English in which every named method, dataset, "
         "benchmark, mechanic or term fits the rest (no keyword stuffing, no lists of random words, no other languages). "
         "Keep the {body} under 170 words.\n\n<draft>\n{seed}\n</draft>\n\n<previous_rewrite cosine=\"{score:.4f}\">"
         "\n{prev}\n</previous_rewrite>\n\nAnswer with the JSON object only.")
WHAT = {"steam": ("gameplay and mechanics", ", genres and tags"), "arxiv": ("problem and methods", "")}
BODY = {"steam": "About text", "arxiv": "abstract"}
POLISH = ("Below is a {noun} description. Edit it lightly so that it reads as a real, coherent {noun} description: "
          "make sure every named method, dataset, benchmark, mechanic or term fits the rest, remove or replace anything "
          "contradictory or out of place, and keep everything else (topic, specific content, wording) as it is. Keep "
          "the {body} under 170 words.\n\n<description>\n{text}\n</description>\n\nAnswer with the JSON object only.")


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


class Scorer:
    """ONNX embedding with the same cache as scripts/fluent_gate.py, cosine to the target, corpus max (REAL masked),
    validity and the gibberish flag."""

    def __init__(self, domains):
        self.model = LocalQwen()
        sig = hashlib.sha256((inspect.getsource(LocalQwen) + str([
            (str(p.relative_to(MODEL_DIR)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(MODEL_DIR.rglob("*")) if p.is_file()])).encode()).hexdigest()[:16]
        self.cache = FS / "verified_cache" / sig
        self.cache.mkdir(parents=True, exist_ok=True)
        self.P = {}
        for d in domains:
            meta = pd.read_parquet(DATA / "prepared" / d / "corpus.parquet", columns=["split", "adult"])
            E = np.load(DATA / "prepared" / d / "emb_qwen3e06.npy").astype(np.float32)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            self.P[d] = E[np.flatnonzero((~meta.adult & (meta.split == "past")).to_numpy())]
        jobs = json.loads((FS / "inputs/fluent_inputs.json").read_text())
        self.t = {(j["domain"], j["arm"], j["k"]): np.asarray(j["target"], np.float32) for j in jobs}
        refs = pd.read_json(FS / "refs.jsonl", lines=True)
        self.refs = {(r.domain, r.arm, r.k): r for r in refs.itertuples()}

    def emb(self, text: str) -> np.ndarray:
        f = self.cache / f"{hashlib.sha256(text.encode()).hexdigest()}.npy"
        if f.exists():
            return np.load(f)
        e = self.model.embed_one(text)
        np.save(f, e)
        return e

    def score(self, key, text: str) -> dict:
        d = key[0]
        t = self.t[key] / np.linalg.norm(self.t[key])
        e = self.emb(text)
        sims = self.P[d] @ e
        hidden = self.refs[key].hidden_pos
        if len(hidden):
            sims[hidden] = -np.inf
        item = parse(d, text)
        bad = "format" if item is None else bounds_violation(d, item, len(self.model.tok.encode(text).ids))
        st = gibberish_stats(text)
        rule = gibberish_rule(st)
        return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "score": float(e @ t),
                "corpus_max": float(sims.max()), "valid": not bad, "status": bad or "ok", "gib_rule": rule,
                "gibberish": rule != ""}


def seeds_for(scorer: Scorer, set_name: str, key, S: int) -> list[dict]:
    d = key[0]
    rows = [json.loads(line) for f in sorted((FS / SEED_RUNS[set_name][d] / "kaggle-out").glob("fluent_gpu*.jsonl"))
            for line in f.read_text().splitlines() if line.strip()]
    texts = set()
    for r in rows:
        if (r["domain"], r["arm"], r["k"]) == key and r["lam"] in SEED_LAMBDAS:
            texts.add(r["generated"])
            texts.update(c["text"] for c in r["pool"])
    scored = sorted((scorer.score(key, t) for t in texts), key=lambda c: -c["score"])
    return scored[:S]


def render_item(domain: str, o: dict) -> str | None:
    if not isinstance(o, dict) or not all(f in o for f in DOMAINS[domain].schema["required"]):
        return None
    return DOMAINS[domain].render(o)


def run_target(key, args, scorer, writer, out: Path, log) -> dict:
    ck = out / f"{key[1]}-{key[2]}.json"
    cfg = {"S": args.S, "R": args.R, "prompt": PROMPT_VERSION, "writer": args.writer}
    if ck.exists():
        rec = json.loads(ck.read_text())
        if rec["config"] == cfg and rec.get("done"):
            return rec
    d = key[0]
    dom = DOMAINS[d]
    seeds = seeds_for(scorer, args.set, key, args.S)
    sysmsg = {"role": "system", "content": SYSTEM.format(noun=dom.noun, spec=dom.json_spec)}
    calls = {"n": 0, "errors": 0, "seconds": 0.0}
    lock = threading.Lock()

    def call(user: str) -> str | None:
        t0 = time.time()
        try:
            raw = writer.chat([sysmsg, {"role": "user", "content": user}])
            o = json.loads(raw)
            return render_item(d, o)
        except RateLimitedError:
            raise
        except (RefusedError, RuntimeError, ValueError, TypeError) as e:
            with lock:
                calls["errors"] += 1
            log(f"writer error: {type(e).__name__}: {str(e)[:150]}")
            return None
        finally:
            with lock:
                calls["n"] += 1
                calls["seconds"] += time.time() - t0

    def chain(si: int, seed: dict) -> list[dict]:
        out_c = []
        best = None
        what, lists = WHAT[d]
        for r in range(args.R + 1):
            if best is None:
                text = call(FIRST.format(noun=dom.noun, what=what, lists=lists, seed=seed["text"], body=BODY[d]))
            else:
                text = call(AGAIN.format(noun=dom.noun, seed=seed["text"], score=best["score"], prev=best["text"],
                                         body=BODY[d]))
            if text is None:
                continue
            c = scorer.score(key, text) | {"seed": si, "round": r}
            out_c.append(c)
            if c["valid"] and (best is None or c["score"] > best["score"]):
                best = c
        if best is not None:  # v2: one coherence-polish round on the chain's best text (the seed is not shown)
            text = call(POLISH.format(noun=dom.noun, text=best["text"], body=BODY[d]))
            if text is not None:
                out_c.append(scorer.score(key, text) | {"seed": si, "round": args.R + 1, "polish": True})
        return out_c

    with ThreadPoolExecutor(len(seeds) or 1) as ex:
        cands = [c for cl in ex.map(lambda a: chain(*a), enumerate(seeds)) for c in cl]
    rec = {"domain": d, "arm": key[1], "k": key[2], "config": cfg, "seeds": seeds, "candidates": cands,
           "calls": calls, "done": True}
    ck.write_text(json.dumps(rec))
    best = max((c["score"] for c in cands if c["valid"]), default=None)
    log(f"rewritten: {len(cands)} candidates, best {best}, seed best {seeds[0]['score'] if seeds else None:.4f}, "
        f"{calls['n']} calls ({calls['errors']} errors, {calls['seconds']:.0f} s)")
    return rec


class Gate:
    def __init__(self, path: Path):
        self.path, self.lock, self.cache = path, threading.Lock(), {}
        if path.exists():
            for line in path.read_text().splitlines():
                h = json.loads(line)
                self.cache[h["judge"], h["sha256"]] = h

    def rate(self, spec, domain, c) -> int | None:
        with self.lock:
            hit = self.cache.get((spec, c["sha256"]))
        if hit is None:
            hit = judge(spec, DOMAINS[domain].noun, c["text"]) | {"judge": spec, "sha256": c["sha256"]}
            if hit.get("coherence") is not None:
                with self.lock:
                    self.cache[spec, c["sha256"]] = hit
                    with self.path.open("a") as f:
                        f.write(json.dumps(hit) + "\n")
        return hit.get("coherence")

    def walk(self, rec: dict) -> dict:
        d = rec["domain"]
        seen, ok = set(), []
        for c in sorted(rec["candidates"], key=lambda c: -c["score"]):
            if c["valid"] and not c["gibberish"] and c["sha256"] not in seen:
                seen.add(c["sha256"])
                ok.append(c)
        out = {"h_coh": None, "n_judged": 0}
        for c in ok[:CAP]:
            out["n_judged"] += 1
            a = self.rate(JUDGE1, d, c)
            b = self.rate(JUDGE2, d, c) if (a or 0) >= COH_PASS else None
            if (a or 0) >= COH_PASS and (b or 0) >= COH_PASS:
                out.update(h_coh=c["score"], coh_text=c["text"], coh_corpus_max=c["corpus_max"], j1=a, j2=b,
                           coh_round=c["round"], coh_seed=c["seed"])
                break
        return out


def verdicts(T: pd.DataFrame, who: str) -> list[str]:
    v = []
    real, far, gap = T[T.arm == "REAL"], T[T.arm == "FAR"], T[T.arm == "GAP"]
    share = float((real.h_coh >= real.corpus_q99).mean()) if len(real) else float("nan")
    powered = bool(len(real) and share >= REAL_SHARE)
    v.append(f"- {who} C (REAL control): {'POWERED' if powered else 'UNDERPOWERED'}: h_coh >= q99 on "
             f"{(real.h_coh >= real.corpus_q99).sum()}/{len(real)} (needs >= {REAL_SHARE:.0%}); > ceiling on "
             f"{(real.h_coh > real.ceiling).sum()}/{len(real)}")
    if len(far):
        cov = far.h_coh.notna().mean()
        esc = (far.h_coh >= far.ceiling_star + FAR_ESCAPE).mean()
        within = (far.h_coh <= far.ceiling_star + FAR_WITHIN).mean()
        if esc >= FAR_ESCAPE_SHARE:
            s = f"objection REFUTED (strong form): >= ceiling* + {FAR_ESCAPE} on {esc:.0%}"
        elif cov < COVERAGE:
            s = f"insufficient coverage ({cov:.0%})"
        elif not powered:
            s = "uninformative: REAL control failed"
        elif within >= FAR_WITHIN_SHARE:
            s = f"objection SUPPORTED: within ceiling* + {FAR_WITHIN} on {within:.0%}"
        else:
            s = f"intermediate: within {within:.0%}, escape {esc:.0%}"
        v.append(f"- {who} P1 FAR: {s} (coverage {cov:.0%}, within {within:.0%}, escape {esc:.0%}, "
                 f"max h_coh - ceiling* {(far.h_coh - far.ceiling_star).max():+.3f})")
    if len(gap):
        cov = gap.h_coh.notna().mean()
        reached = (gap.h_coh >= gap.ceiling - GAP_MARGIN).mean()
        if cov < COVERAGE:
            s = f"insufficient coverage ({cov:.0%})"
        elif reached > GAP_SHARE:
            s = f"REACHABLE: >= ceiling - {GAP_MARGIN} on {reached:.0%}"
        elif powered:
            s = f"not reached by a powered search ({reached:.0%})"
        else:
            s = "inconclusive: control failed"
        v.append(f"- {who} P2 GAP: {s}; >= ceiling* + {FAR_ESCAPE} on "
                 f"{(gap.h_coh >= gap.ceiling_star + FAR_ESCAPE).mean():.0%}")
    return v


def summary(label: str) -> None:
    rows = []
    for ck in sorted((OUT / label).glob("*/*/*.json")):
        rec = json.loads(ck.read_text())
        g = rec.get("gate", {})
        ok = [c for c in rec["candidates"] if c["valid"]]
        rows.append({"domain": rec["domain"], "writer": rec["config"]["writer"].split(":")[1], "arm": rec["arm"],
                     "k": rec["k"], "h_coh": g.get("h_coh"), "h_all": max((c["score"] for c in ok), default=None),
                     "seed_best": rec["seeds"][0]["score"] if rec["seeds"] else None,
                     "calls": rec["calls"]["n"], "call_s": rec["calls"]["seconds"],
                     "judged": g.get("n_judged", 0)})
    D = pd.DataFrame(rows)
    refs = pd.read_csv(FS / "refs.csv")
    refs["ceiling_star"] = np.maximum(refs.ceiling, refs.heldout_ceiling)
    D = D.merge(refs, on=["domain", "arm", "k"])
    D["h_coh"] = D.h_coh.astype(float)
    comb = D.groupby(["domain", "arm", "k"], as_index=False).agg(
        h_coh=("h_coh", "max"), h_all=("h_all", "max"), seed_best=("seed_best", "first"),
        ceiling=("ceiling", "first"), corpus_q99=("corpus_q99", "first"), ceiling_star=("ceiling_star", "first"),
        calls=("calls", "sum"), call_s=("call_s", "sum"), judged=("judged", "sum")).assign(writer="max over writers")
    D = pd.concat([D, comb])
    D.to_csv(OUT / label / "targets.csv", index=False)
    lines, verd = [], []
    tab = []
    for (d, w), g in D.groupby(["domain", "writer"], sort=False):
        for arm, h in g.groupby("arm"):
            tab.append({"domain": d, "writer": w, "arm": arm, "n": len(h), "ceiling": f"{h.ceiling.median():.3f}",
                        "corpus q99": f"{h.corpus_q99.median():.3f}", "seed (GPU) best": f"{h.seed_best.median():.3f}",
                        "h_all": f"{h.h_all.median():.3f}",
                        "h_coh": f"{h.h_coh.median():.3f}" if h.h_coh.notna().any() else "-",
                        "coverage": f"{h.h_coh.notna().sum()}/{len(h)}",
                        ">= q99": f"{(h.h_coh >= h.corpus_q99).sum()}/{len(h)}",
                        "calls": int(h.calls.sum()), "writer h": f"{h.call_s.sum() / 3600:.1f}"})
        verd += verdicts(g, f"{d} {w}")
    t = pd.DataFrame(tab)
    lines.append("| " + " | ".join(t.columns) + " |")
    lines.append("|" + "---|" * len(t.columns))
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    lines += ["", *verd]
    (OUT / label / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--set", choices=sorted(SETS))
    ap.add_argument("--writer", help="claude:<model> or codex:<model> (subscription CLIs)")
    ap.add_argument("--domain", choices=["steam", "arxiv"])
    ap.add_argument("--S", type=int, default=2, help="seeds per target")
    ap.add_argument("--R", type=int, default=3, help="feedback rounds after the first rewrite")
    ap.add_argument("--workers", type=int, default=4, help="targets in parallel")
    ap.add_argument("--arms", default="")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.label)
        return
    assert args.writer.split(":")[0] in ("claude", "codex"), "subscription CLIs only"
    out = OUT / args.label / args.domain / args.writer.split(":")[1]
    out.mkdir(parents=True, exist_ok=True)
    scorer = Scorer([args.domain])
    dom = DOMAINS[args.domain]
    writer = CliChat(args.writer, dom.schema)
    gate = Gate(OUT / args.label / "judgements.jsonl")
    arms = args.arms.split(",") if args.arms else list(SETS[args.set])
    keys = [(args.domain, a, k) for a in arms for k in SETS[args.set][a]]
    t0 = time.time()

    def one(key):
        log = lambda m: print(f"[{key[1]} {key[2]}] {m}", flush=True)
        try:
            rec = run_target(key, args, scorer, writer, out, log)
            if "gate" not in rec:
                rec["gate"] = gate.walk(rec)
                (out / f"{key[1]}-{key[2]}.json").write_text(json.dumps(rec))
            g = rec["gate"]
            ref = scorer.refs[key]
            log(f"gate: h_coh {g['h_coh']} after {g['n_judged']} (ceiling {ref.ceiling:.3f}, q99 {ref.corpus_q99:.3f})")
            return rec
        except Exception as e:  # noqa: BLE001  left undone; a rerun resumes
            log(f"FAILED, left undone: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
            return None

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, keys))
    print(f"{sum(r is not None for r in res)}/{len(keys)} targets done in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
