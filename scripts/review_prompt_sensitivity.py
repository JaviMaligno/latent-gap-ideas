"""Study 6a of the review round: prompt sensitivity of the one-shot step (docs/prereg/review_prompt_sensitivity.md).
PAID (gpt-4o through an API gateway, hard budget shared with review_rewrite_nocontext.py: 30 USD).

  uv run python scripts/review_prompt_sensitivity.py dry-run                     # prompts + call counts, no call
  uv run python scripts/review_prompt_sensitivity.py smoke --confirm-paid        # 5 calls (3 here + 2 no-context)
  uv run python scripts/review_prompt_sensitivity.py estimate                    # projected cost of every part
  uv run python scripts/review_prompt_sensitivity.py generate --study disc|rep --confirm-paid   (resumable)
  OMP_NUM_THREADS=6 uv run python scripts/review_prompt_sensitivity.py embed-qwen --study disc|rep
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow \
      python scripts/review_prompt_sensitivity.py embed-bge --study disc|rep
  OMP_NUM_THREADS=4 uv run python scripts/review_prompt_sensitivity.py analyse   # -> docs/prereg/review_prompt_sensitivity.json
  uv run python scripts/review_prompt_sensitivity.py spend                       # USD per tag and per day (ledger)

Ideas: runs/review-prompt-sensitivity/{disc,rep}/ideas.jsonl (ctx, variant, k, title, abstract, via, attempts).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import continue_from_context as CFC  # noqa: E402

DATA = CFC.DATA
LEDGER = DATA / "ledger.jsonl"
OUTDIR = DATA / "runs/review-prompt-sensitivity"
OUT = ROOT / "docs/prereg/review_prompt_sensitivity.json"
ALIAS = "gpt-4o-0806"
SEED = 20261115
N_BOOT = 10000
MAX_TOKENS, TEMPERATURE, ATTEMPTS = 600, 1.0, 3  # as continue_from_context.generate (up to 3 attempts)
N_IDEAS = {"disc": 5, "rep": 9}  # = ideas per context of the original arm (DISC CTX 5; REP round 0, 3 arms x 3)
TAGS = {"smoke": "review-promptsens-smoke", "disc": "review-promptsens-disc", "rep": "review-promptsens-rep"}
BUDGET_TAGS = ("review-promptsens-smoke", "review-promptsens-disc", "review-promptsens-rep",
               "review-rewrite-noctx-disc", "review-rewrite-noctx-rep")
BUDGET_USD, STOP_AT_USD = 30.0, 29.5  # no new call once the five tags have spent 29.5 USD (in-flight calls < 0.5)
PRICE_IN, PRICE_OUT = 2.50, 10.0

_TAIL = " Give its title and an arXiv-style abstract of 120-180 words."
ORIGINAL = ("Here is a paper:\n\n{ctx}\n\nPropose the paper that should come next, building on this one: "
            "an original follow-up a research group could write." + _TAIL)
VARIANTS = {
    "V1": ("Here is a paper:\n\n{ctx}\n\nPropose a paper that will cite this one, written about three years from now: "
           "an original paper a research group could write." + _TAIL),
    "V2": ("Here is a paper:\n\n{ctx}\n\nPropose a research idea that builds on this paper: "
           "an original idea a research group could pursue." + _TAIL),
}


def log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ------------------------------------------------------------------------------------------------ budget and ledger
def ledger_rows(tags=BUDGET_TAGS) -> list[dict]:
    out = []
    for l in LEDGER.read_text().splitlines():
        r = json.loads(l)
        if r["tag"].split("|")[0] in tags:
            out.append(r)
    return out


def usd(r: dict) -> float:
    return 0.0 if r["model"] == "codex-cli" else (r["in"] * PRICE_IN + r["out"] * PRICE_OUT) / 1e6


def spent(tags=BUDGET_TAGS) -> float:
    return sum(usd(r) for r in ledger_rows(tags))


class BudgetGateway:
    """Gateway with (a) a hard stop on the shared budget, counted from the ledger rows of the five tags, and (b) the
    route of each call (azure / openrouter / codex) returned with the text. The gateway itself is not modified."""

    def __init__(self):
        from latent_gap.gateway import Gateway
        self.gw = Gateway(ledger=LEDGER, model=ALIAS)
        self._spent = spent()
        self._lock = threading.Lock()
        self._tl = threading.local()
        orig_log = self.gw._log

        def _log(model, usage, tag):
            orig_log(model, usage, tag)
            self._tl.tag = tag
            with self._lock:
                self._spent += usd({"model": model, "in": usage.get("prompt_tokens", 0),
                                    "out": usage.get("completion_tokens", 0) or 0})
        self.gw._log = _log
        log(f"budget: {self._spent:.4f} USD already spent on the five tags (cap {BUDGET_USD}, stop at {STOP_AT_USD})")

    def chat(self, system: str, user: str, tag: str, max_tokens: int, temperature: float) -> tuple[str, str]:
        with self._lock:
            if self._spent >= STOP_AT_USD:
                raise BudgetStop(f"spent {self._spent:.3f} USD >= {STOP_AT_USD}")
        self._tl.tag = ""
        raw = self.gw.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           max_tokens=max_tokens, json_mode=True, tag=tag, extra={"temperature": temperature})
        t = self._tl.tag
        via = "codex" if "codex-fallback" in t else "openrouter" if "openrouter-fallback" in t else "azure"
        return raw, via


class BudgetStop(RuntimeError):
    pass


def parse(raw: str) -> dict | None:
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and d.get("title") and d.get("abstract"):
        return {"title": d["title"], "abstract": d["abstract"]}
    return None


def run_items(items: list[dict], out: Path, tag: str, workers: int, attempts: int) -> dict:
    """items: id, prompt, + metadata. One output row per item; parse failure after `attempts` -> empty (dropped)."""
    bg = BudgetGateway()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    lock = threading.Lock()
    st = {"calls": 0, "parse_fail": 0, "dropped": 0, "via_openrouter": 0, "via_codex": 0, "stopped": 0}
    stop = threading.Event()

    def one(it: dict) -> None:
        if stop.is_set():
            return
        rec, via, att = {"title": "", "abstract": ""}, "", 0
        try:
            for att in range(1, attempts + 1):
                raw, via = bg.chat(CFC.SYSTEM, it["prompt"], tag, MAX_TOKENS, TEMPERATURE)
                p = parse(raw)
                with lock:
                    st["calls"] += 1
                    st["parse_fail"] += p is None
                    st["via_openrouter"] += via == "openrouter"
                    st["via_codex"] += via == "codex"
                if p:
                    rec = p
                    break
        except BudgetStop as e:
            stop.set()
            with lock:
                st["stopped"] += 1
            log(f"BUDGET STOP: {e}")
            return
        with lock, out.open("a") as fh:
            st["dropped"] += not rec["abstract"]
            fh.write(json.dumps({k: v for k, v in it.items() if k != "prompt"} | rec | {"via": via, "attempts": att}) + "\n")

    log(f"{tag}: {len(todo)} items to call ({len(done)} done) -> {out}")
    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            if (n + 1) % 100 == 0:
                log(f"  {n + 1}/{len(todo)} {st} spent {bg._spent:.3f} USD")
    log(f"{tag}: done {st} spent {bg._spent:.3f} USD")
    return st


# ------------------------------------------------------------------------------------------------ items
def contexts(study: str) -> list[dict]:
    if study == "disc":
        return CFC.contexts()
    import strengthen_critique_replication as SCR
    return SCR.ctx_rows()


def ctx_blocks(study: str) -> list[str]:
    import pandas as pd
    ctx = contexts(study)
    t = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"])
    t = t[t["id"].isin({c["arxiv_id"] for c in ctx})].set_index("id")
    return [CFC.paper_block(t.loc[c["arxiv_id"], "name"], t.loc[c["arxiv_id"], "abstract"]) for c in ctx]


def build_items(study: str) -> list[dict]:
    cbs = ctx_blocks(study)
    return [{"id": f"{v}/{i}/{k}", "ctx": i, "variant": v, "k": k, "prompt": VARIANTS[v].format(ctx=cb)}
            for i, cb in enumerate(cbs) for v in VARIANTS for k in range(N_IDEAS[study])]


def dry_run() -> None:
    print(f"=== DRY RUN: no gateway client, no call ===\nsystem: {CFC.SYSTEM}\nmodel {ALIAS} temperature {TEMPERATURE} "
          f"max_tokens {MAX_TOKENS} json_mode attempts {ATTEMPTS}")
    cb = ctx_blocks("disc")[0]
    print(f"\n--- ORIGINAL (not regenerated) ---\n{ORIGINAL.format(ctx=cb)}")
    for v, p in VARIANTS.items():
        print(f"\n--- {v} ---\n{p.format(ctx=cb)}")
    for s in ("disc", "rep"):
        print(f"{s}: {len(contexts(s))} contexts x {len(VARIANTS)} variants x {N_IDEAS[s]} ideas = "
              f"{len(contexts(s)) * len(VARIANTS) * N_IDEAS[s]} items")


# ------------------------------------------------------------------------------------------------ smoke and estimate
SMOKE_OUT = OUTDIR / "smoke.jsonl"


def smoke() -> None:
    """5 calls in all, kept out of every analysis: V1 and V2 on DISC context 0, V1 on REP context 0 (this study), and
    the no-context rewrite of two DISC follow-ups (review_rewrite_nocontext.py). Tag review-promptsens-smoke."""
    import review_rewrite_nocontext as RNC
    it = build_items("disc")
    pick = [x for x in it if x["id"] in ("V1/0/0", "V2/0/0")]
    rep = [x for x in build_items("rep") if x["id"] == "V1/0/0"]
    items = [x | {"id": "disc/" + x["id"]} for x in pick] + [x | {"id": "rep/" + x["id"]} for x in rep]
    nc = RNC.build_items("disc")[:2]
    items += [{k: v for k, v in x.items()} | {"id": "noctx/" + x["id"]} for x in nc]
    assert len(items) == 5
    if SMOKE_OUT.exists():
        raise SystemExit(f"{SMOKE_OUT} exists: the smoke runs once")
    run_items(items, SMOKE_OUT, TAGS["smoke"], workers=5, attempts=1)
    for r in map(json.loads, SMOKE_OUT.read_text().splitlines()):
        print(r["id"], r["via"], "|", r["title"][:90])
    rows = ledger_rows(("review-promptsens-smoke",))
    print(json.dumps({"calls": len(rows), "in": [r["in"] for r in rows], "out": [r["out"] for r in rows],
                      "usd": round(sum(map(usd, rows)), 5)}))


def estimate() -> dict:
    """Projected cost per part: input tokens = characters of every prompt (+ system) x the smoke's tokens per
    character (per prompt family); output tokens = max(smoke mean, 230) per call; x 1.03 for parse retries."""
    import review_rewrite_nocontext as RNC
    rows = ledger_rows(("review-promptsens-smoke",))
    sm = [json.loads(l) for l in SMOKE_OUT.read_text().splitlines()]
    assert len(rows) == len(sm) == 5, "the estimate needs the 5 smoke calls"
    # ledger rows are in completion order, not item order: match by family using the per-family token totals
    prom = {x["id"]: x["prompt"] for x in
            [x | {"id": "disc/" + x["id"]} for x in build_items("disc") if x["id"] in ("V1/0/0", "V2/0/0")]
            + [x | {"id": "rep/" + x["id"]} for x in build_items("rep") if x["id"] == "V1/0/0"]
            + [x | {"id": "noctx/" + x["id"]} for x in RNC.build_items("disc")[:2]]}
    chars = {k: len(CFC.SYSTEM) + len(v) for k, v in prom.items()}
    tin, tout = sum(r["in"] for r in rows), sum(r["out"] for r in rows)
    tpc = tin / sum(chars.values())  # tokens per character, pooled over the 5 calls (same tokenizer, English text)
    out_tok = max(tout / len(rows), 230.0)
    parts = {"6a-disc": ("disc", build_items("disc")), "6a-rep": ("rep", build_items("rep")),
             "6b-disc": ("disc", RNC.build_items("disc")), "6b-rep": ("rep", RNC.build_items("rep"))}
    est = {"smoke_usd": sum(map(usd, rows)), "tokens_per_char": tpc, "out_tokens_per_call": out_tok, "parts": {}}
    for name, (_, items) in parts.items():
        n = len(items)
        ti = sum(len(CFC.SYSTEM) + len(x["prompt"]) for x in items) * tpc
        c = 1.03 * (ti * PRICE_IN + n * out_tok * PRICE_OUT) / 1e6
        est["parts"][name] = {"calls": n, "in_tokens": round(ti), "out_tokens": round(n * out_tok), "usd": round(c, 3)}
    est["total_all_usd"] = round(est["smoke_usd"] + sum(p["usd"] for p in est["parts"].values()), 3)
    est["total_disc_only_usd"] = round(est["smoke_usd"] + est["parts"]["6a-disc"]["usd"] + est["parts"]["6b-disc"]["usd"], 3)
    est["decision"] = "all four parts" if est["total_all_usd"] <= BUDGET_USD else "discovery sample only"
    (OUTDIR / "estimate.json").write_text(json.dumps(est, indent=1))
    print(json.dumps(est, indent=1))
    return est


def generate(study: str, workers: int) -> None:
    run_items(build_items(study), OUTDIR / study / "ideas.jsonl", TAGS[study], workers, ATTEMPTS)


# ------------------------------------------------------------------------------------------------ embedding
def rows_of(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return sorted((r for r in rows if r["abstract"]), key=lambda r: r["id"])  # order of the vectors


def texts(rows):
    from latent_gap.arxiv import item_text
    return [item_text(r["title"], r["abstract"]) for r in rows]


def embed_qwen_texts(T: list[str]) -> np.ndarray:
    from latent_gap.embedder import LocalQwen
    q = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6")))
    out = []
    for b in range(0, len(T), 100):
        out.append(q.embed(T[b:b + 100]))
        log(f"qwen {min(b + 100, len(T))}/{len(T)}")
    return np.concatenate(out).astype(np.float32)


def embed_bge_texts(T: list[str]) -> np.ndarray:
    """Settings of revision_neutral.embed_bge (sentence-transformers, max_seq_length 512, normalised, CPU)."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    m = SentenceTransformer("BAAI/bge-m3", device="cpu")
    m.max_seq_length = 512
    out = []
    for b in range(0, len(T), 64):
        out.append(m.encode(T[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        log(f"bge {min(b + 64, len(T))}/{len(T)}")
    return np.concatenate(out).astype(np.float32)


def embed(study: str, space: str) -> None:
    rows = rows_of(OUTDIR / study / "ideas.jsonl")
    V = embed_qwen_texts(texts(rows)) if space == "qwen3e06" else embed_bge_texts(texts(rows))
    np.save(OUTDIR / study / f"ideas_emb_{space}.npy", V)
    log(f"{study} {space} {V.shape}")


# ------------------------------------------------------------------------------------------------ analysis
def boot_ci(x: np.ndarray, rng, n=N_BOOT) -> list[float]:
    idx = rng.integers(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def ratio_ci(num: np.ndarray, den: np.ndarray, rng, n=N_BOOT) -> list[float]:
    idx = rng.integers(0, len(num), (n, len(num)))
    b = num[idx].mean(1) / den[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def wil(x, alt):
    from scipy import stats
    return float(stats.wilcoxon(x, alternative=alt).pvalue)


def holm(p: dict) -> dict:
    keys, out, run = sorted(p, key=lambda k: p[k]), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        out[k] = run
    return out


def original_idea_steps(cell, study: str) -> dict:
    """Per context: steps of the original arm's ideas. DISC: one-shot CTX ideas (cell.pb); REP: round-0 ideas of the
    three critique arms (the same CTX prompt, Section 5.2)."""
    if study == "disc":
        return {i: cell.pb[1][cell.pb_index[i]] @ cell.C[i] for i in cell.pb_index}
    out = {}
    for i in range(cell.n):
        idx = [n for a in ("GEO", "LLM", "ORACLE") for n in cell.index.get((a, i, 0), [])]
        if idx:
            out[i] = cell.V[idx] @ cell.C[i]
    return out


def analyse_cell(cell, study: str, rows: list[dict], V: np.ndarray, include_fallback: bool) -> dict:
    rng = np.random.default_rng(SEED)
    orig = original_idea_steps(cell, study)
    fstep = {i: float(np.median(cell.F[i] @ cell.C[i])) for i in range(cell.n)}
    steps = {v: {} for v in VARIANTS}
    for n, r in enumerate(rows):
        if r["via"] != "azure" and not include_fallback:
            continue
        steps[r["variant"]].setdefault(r["ctx"], []).append(float(V[n] @ cell.C[r["ctx"]]))
    ctxs = sorted(set(orig) & set.intersection(*(set(s) for s in steps.values())))
    g0 = np.array([np.median(orig[i]) - fstep[i] for i in ctxs])
    res = {"n_contexts": len(ctxs), "ideas_per_variant": {v: int(sum(len(steps[v][i]) for i in ctxs)) for v in VARIANTS},
           "original": {"mean_step_gap": float(g0.mean()), "ci": boot_ci(g0, rng), "p_wilcoxon_greater": wil(g0, "greater"),
                        "mean_idea_step": float(np.mean([np.median(orig[i]) for i in ctxs])),
                        "mean_followup_step": float(np.mean([fstep[i] for i in ctxs]))}}
    for v in VARIANTS:
        g = np.array([np.median(steps[v][i]) - fstep[i] for i in ctxs])
        d = g - g0
        res[v] = {"mean_step_gap": float(g.mean()), "ci": boot_ci(g, rng), "p_wilcoxon_greater": wil(g, "greater"),
                  "mean_idea_step": float(np.mean([np.median(steps[v][i]) for i in ctxs])),
                  "diff_vs_original": {"mean": float(d.mean()), "ci": boot_ci(d, rng), "p_wilcoxon_two_sided": wil(d, "two-sided")},
                  "share_of_original_gap": {"est": float(g.mean() / g0.mean()), "ci": ratio_ci(g, g0, rng)}}
    res["holm_gap"] = holm({v: res[v]["p_wilcoxon_greater"] for v in VARIANTS})
    res["holm_diff"] = holm({v: res[v]["diff_vs_original"]["p_wilcoxon_two_sided"] for v in VARIANTS})
    res["robust_to_wording"] = bool(all(res[v]["ci"][0] > 0 for v in VARIANTS))
    return res


def analyse(studies: list[str], spaces: list[str]) -> None:
    import review_tangent_confounds as RTC
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.update({"prereg": "docs/prereg/review_prompt_sensitivity.md", "seed": SEED, "n_boot": N_BOOT})
    for space in spaces:
        corp = RTC.Corpus(space)
        for study in studies:
            p = OUTDIR / study / "ideas.jsonl"
            if not p.exists():
                continue
            cell = RTC.load_cell({"disc": "DISC-4o", "rep": "REP-4o"}[study], space, corp)
            rows = rows_of(p)
            V = np.load(OUTDIR / study / f"ideas_emb_{space}.npy")
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            assert len(V) == len(rows)
            allr = [json.loads(l) for l in p.read_text().splitlines()]
            key = f"{study}/{space}"
            res[key] = {"counts": {"items": len(allr), "dropped_parse": sum(not r["abstract"] for r in allr),
                                   "needed_retry": sum(r["attempts"] > 1 for r in allr),
                                   "via": {w: sum(r["via"] == w for r in allr if r["abstract"]) for w in ("azure", "openrouter", "codex")}},
                        "primary_azure_only": analyse_cell(cell, study, rows, V, False)}
            if any(r["via"] != "azure" for r in rows):
                res[key]["sensitivity_with_fallback_filled"] = analyse_cell(cell, study, rows, V, True)
            OUT.write_text(json.dumps(res, indent=1) + "\n")
            log(f"{key} done")
        del corp
    OUT.write_text(json.dumps(res, indent=1) + "\n")


def spend_report() -> None:
    rows = ledger_rows()
    by = {}
    for r in rows:
        day = dt.datetime.fromtimestamp(r["t"], dt.timezone.utc).strftime("%Y-%m-%d")
        k = (r["tag"], r["model"], day)
        a = by.setdefault(k, [0, 0, 0, 0.0])
        a[0] += 1
        a[1] += r["in"]
        a[2] += r["out"]
        a[3] += usd(r)
    for k, a in sorted(by.items()):
        print(f"{k[2]}  {k[0]:45s} {k[1]:28s} calls {a[0]:5d}  in {a[1]:9d}  out {a[2]:8d}  USD {a[3]:.4f}")
    print(f"TOTAL USD {sum(map(usd, rows)):.4f} over {len(rows)} ledger rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry-run", "smoke", "estimate", "generate", "embed-qwen", "embed-bge", "analyse", "spend"])
    ap.add_argument("--study", choices=["disc", "rep"], default="disc")
    ap.add_argument("--studies", default="disc,rep")
    ap.add_argument("--spaces", default="qwen3e06,bgem3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--confirm-paid", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("smoke", "generate") and not a.confirm_paid:
        raise SystemExit("paid step: pass --confirm-paid (Javier approved a 30 USD cap for 6a + 6b)")
    {"dry-run": dry_run, "smoke": smoke, "estimate": estimate, "generate": lambda: generate(a.study, a.workers),
     "embed-qwen": lambda: embed(a.study, "qwen3e06"), "embed-bge": lambda: embed(a.study, "bgem3"),
     "analyse": lambda: analyse(a.studies.split(","), a.spaces.split(",")), "spend": spend_report}[a.cmd]()


if __name__ == "__main__":
    main()
