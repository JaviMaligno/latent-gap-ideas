"""Study 6b of the review round: the follow-up rewrite without the context in the prompt
(docs/prereg/review_rewrite_nocontext.md). PAID (gpt-4o through an API gateway; shared 30 USD budget guard of
review_prompt_sensitivity.py).

  uv run python scripts/review_rewrite_nocontext.py dry-run
  uv run python scripts/review_rewrite_nocontext.py generate --study disc|rep --confirm-paid   (resumable)
  OMP_NUM_THREADS=6 uv run python scripts/review_rewrite_nocontext.py embed-qwen --study disc|rep
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow \
      python scripts/review_rewrite_nocontext.py embed-bge --study disc|rep
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow \
      python scripts/review_rewrite_nocontext.py agreement      # local vs Kaggle vectors of 50 context-shown rewrites
  OMP_NUM_THREADS=4 uv run python scripts/review_rewrite_nocontext.py analyse   # -> docs/prereg/review_rewrite_nocontext.json

Items: every saved follow-up (RW-F of review_rewrite_generate.py), prompt R with the context and the words that refer
to it removed; system prompt, temperature, max_tokens, JSON mode, parser and one retry exactly as RW-F.
Raw: runs/review-rewrite-nocontext/{disc,rep}/rewrites.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import continue_from_context as CFC  # noqa: E402
import review_prompt_sensitivity as RPS  # noqa: E402

DATA = CFC.DATA
OUTDIR = DATA / "runs/review-rewrite-nocontext"
OUT = ROOT / "docs/prereg/review_rewrite_nocontext.json"
TAGS = {"disc": "review-rewrite-noctx-disc", "rep": "review-rewrite-noctx-rep"}
ATTEMPTS = 2  # one retry, as review_rewrite_generate.py
SEED = RPS.SEED
CONTENT_MIN = 0.80

# review_rewrite_generate.PROMPT_R with "Here is a paper:\n\n{ctx}\n\n" and ", building on the paper above" removed and
# "Below is a later paper" -> "Below is a paper" (no earlier paper is shown); nothing else changes.
PROMPT_NC = ("Below is a paper. Rewrite it as the proposal for the paper that should come next: give its title and an "
             "arXiv-style abstract of 120-180 words. Keep exactly its content: the same subject, method, claims and "
             "details. Add nothing, remove nothing, invent nothing; only the wording and the framing as a proposal "
             "change.\n\n{later}")


def build_items(study: str) -> list[dict]:
    """Same follow-ups, ids and order as RW-F in review_rewrite_generate.build_items (rwf/{ctx}/{j})."""
    import review_rewrite_generate as RRG
    ctx = RRG.study_contexts(study)
    items = []
    for i, c in enumerate(ctx):
        for j, f in enumerate(c["followups"]):
            items.append({"id": f"rwf/{i}/{j}", "set": "RW-F", "ctx": i, "j": j, "orig_id": f["arxiv_id"],
                          "continuation": j in set(c["consensus"]),
                          "prompt": PROMPT_NC.format(later=CFC.paper_block(f["title"], f["abstract"]))})
    return items


def dry_run() -> None:
    import review_rewrite_generate as RRG
    it = build_items("disc")
    print(f"=== DRY RUN: no call ===\nsystem: {CFC.SYSTEM}\nmodel {RPS.ALIAS} temperature {RPS.TEMPERATURE} max_tokens "
          f"{RPS.MAX_TOKENS} json_mode attempts {ATTEMPTS}\n\n--- {it[0]['id']} (no context) ---\n{it[0]['prompt']}")
    ctx_items = [x for x in RRG.build_items("disc", n_ctx=1) if x["id"] == it[0]["id"]]
    print(f"\n--- same item, context shown (review_rewrite_generate, already run) ---\n{ctx_items[0]['prompt']}")
    for s in ("disc", "rep"):
        print(f"{s}: {len(build_items(s))} items")


def generate(study: str, workers: int) -> None:
    RPS.run_items(build_items(study), OUTDIR / study / "rewrites.jsonl", TAGS[study], workers, ATTEMPTS)


def embed(study: str, space: str) -> None:
    rows = RPS.rows_of(OUTDIR / study / "rewrites.jsonl")
    T = RPS.texts(rows)
    V = RPS.embed_qwen_texts(T) if space == "qwen3e06" else RPS.embed_bge_texts(T)
    np.save(OUTDIR / study / f"rewrites_emb_{space}.npy", V)
    RPS.log(f"{study} {space} {V.shape}")


def agreement() -> None:
    """Local (ONNX Qwen3, CPU BGE-M3) vs Kaggle vectors of the first 50 DISC context-shown rewrites (sorted by id),
    to show that comparing locally embedded no-context rewrites with Kaggle-embedded items is like for like."""
    rows = RPS.rows_of(DATA / "runs/review-rewrite/disc/rewrites.jsonl")[:50]
    T = RPS.texts(rows)
    out = {}
    for space in ("qwen3e06", "bgem3"):
        K = np.load(DATA / f"kaggle-output/review-rewrite-disc-embed/emb_{space}.npy")[:50].astype(np.float32)
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        try:
            L = RPS.embed_qwen_texts(T) if space == "qwen3e06" else RPS.embed_bge_texts(T)
        except Exception as e:  # the other environment lacks this encoder
            out[space] = {"skipped": str(e)}
            continue
        c = np.einsum("ij,ij->i", K, L / np.linalg.norm(L, axis=1, keepdims=True))
        out[space] = {"n": 50, "mean_cos": float(c.mean()), "min_cos": float(c.min())}
    p = OUTDIR / "agreement.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update({k: v for k, v in out.items() if "skipped" not in v})
    p.write_text(json.dumps(prev, indent=1))
    print(json.dumps(prev, indent=1))


def load_nc(study: str, space: str, include_fallback: bool):
    rows = RPS.rows_of(OUTDIR / study / "rewrites.jsonl")
    V = np.load(OUTDIR / study / f"rewrites_emb_{space}.npy").astype(np.float32)
    assert len(rows) == len(V)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    keep = [n for n, r in enumerate(rows) if include_fallback or r["via"] == "azure"]
    return [rows[n] for n in keep], V[keep]


def per_ctx_steps(rows, V, C) -> dict:
    out = {}
    for n, r in enumerate(rows):
        out.setdefault(r["ctx"], {})[r["j"]] = float(V[n] @ C[r["ctx"]])
    return out


def analyse_cell(cell, study: str, space: str, rows, V, ctx_rows, ctx_V, exclude: set) -> dict:
    """gap_NC = per context, median step of the ideas - median step of the no-context rewrites (as review_rewrite_null.p1);
    the context-shown gap gap_RW recomputed on the same follow-ups; their paired difference; Delta_NC vs originals."""
    rng = np.random.default_rng(SEED)
    idea = RPS.original_idea_steps(cell, study)
    nc = per_ctx_steps([r for r in rows if r["id"] not in exclude], V[[n for n, r in enumerate(rows) if r["id"] not in exclude]], cell.C)
    cs = per_ctx_steps(ctx_rows, ctx_V, cell.C)
    lab = cell.labels or {}
    g_nc, g_rw, g_orig, d_pair, d_nc, g_nc_cont = [], [], [], [], [], []
    for i in sorted(set(idea) & set(nc)):
        mi = float(np.median(idea[i]))
        js = sorted(nc[i])
        s_nc = np.array([nc[i][j] for j in js])
        g_nc.append(mi - np.median(s_nc))
        orig = np.array([float(cell.F[i][j] @ cell.C[i]) for j in js])
        d_nc.append(float(np.median(s_nc) - np.median(orig)))
        g_orig.append(mi - float(np.median(cell.F[i] @ cell.C[i])))
        jj = [j for j in js if j in cs.get(i, {})]
        if jj:
            gr = mi - np.median([cs[i][j] for j in jj])
            g_rw.append(gr)
            d_pair.append((mi - np.median([nc[i][j] for j in jj])) - gr)
        cont = [j for j in lab.get(i, {}).get("cont", []) if j in nc[i]]
        if cont:
            g_nc_cont.append(mi - np.median([nc[i][j] for j in cont]))
    out = {}
    for nm, x, alt in (("gap_NC", g_nc, "greater"), ("gap_RW_same_followups", g_rw, "greater"),
                       ("gap_original", g_orig, "greater"), ("gap_NC_minus_gap_RW", d_pair, "two-sided"),
                       ("delta_NC_vs_originals", d_nc, "two-sided"), ("gap_NC_cont", g_nc_cont, "greater")):
        x = np.array(x, float)
        if len(x) >= 2:
            out[nm] = {"mean": float(x.mean()), "ci": RPS.boot_ci(x, rng), "n": int(len(x)),
                       "p_wilcoxon": RPS.wil(x, alt), "alternative": alt}
    g, go = np.array(g_nc), np.array(g_orig)
    out["share_of_original_gap"] = {"est": float(g.mean() / go.mean()), "ci": RPS.ratio_ci(g, go, rng)}
    if g_rw and len(g_rw) == len(g_nc):
        gr = np.array(g_rw)
        out["ratio_to_context_shown_gap"] = {"est": float(g.mean() / gr.mean()), "ci": RPS.ratio_ci(g, gr, rng)}
    sec = {k: out[k]["p_wilcoxon"] for k in ("gap_NC_minus_gap_RW", "gap_NC_cont") if k in out}
    if sec:
        out["holm_secondary"] = RPS.holm(sec)
    out["survives_cleaner_genre_control"] = bool(out["gap_NC"]["ci"][0] > 0)
    return out


def content_check(cell, rows, V) -> tuple[dict, dict]:
    """As review_rewrite_null.content_check for RW-F: cos(rewrite, original follow-up) in the same embedder."""
    cos = {r["id"]: float(V[n] @ cell.F[r["ctx"]][r["j"]]) for n, r in enumerate(rows)}
    x = np.array(list(cos.values()))
    return ({f"q{q}": float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95)} |
            {"mean": float(x.mean()), "share_below_0.80": float((x < CONTENT_MIN).mean()), "n": int(len(x))}), cos


def analyse(studies: list[str], spaces: list[str]) -> None:
    import review_rewrite_null as RRN
    import review_tangent_confounds as RTC
    res = {"prereg": "docs/prereg/review_rewrite_nocontext.md", "seed": SEED, "n_boot": RPS.N_BOOT}
    if (OUTDIR / "agreement.json").exists():
        res["local_vs_kaggle_agreement"] = json.loads((OUTDIR / "agreement.json").read_text())
    excl = {}
    for space in spaces:
        corp = RTC.Corpus(space)
        for study in studies:
            p = OUTDIR / study / "rewrites.jsonl"
            if not p.exists():
                continue
            cell = RTC.load_cell({"disc": "DISC-4o", "rep": "REP-4o"}[study], space, corp)
            crow, cV = RRN.load_rewrites(study, space)
            keep = [n for n, r in enumerate(crow) if r["set"] == "RW-F"]
            crow, cV = [crow[n] for n in keep], cV[keep]
            allr = [json.loads(l) for l in p.read_text().splitlines()]
            key = f"{study}/{space}"
            e = {"counts": {"items": len(allr), "dropped_parse": sum(not r["abstract"] for r in allr),
                            "needed_retry": sum(r["attempts"] > 1 for r in allr),
                            "via": {w: sum(r["via"] == w for r in allr if r["abstract"]) for w in ("azure", "openrouter", "codex")}}}
            rows, V = load_nc(study, space, False)
            e["content_check_no_context"], cos = content_check(cell, rows, V)
            e["content_check_context_shown"] = content_check(cell, crow, cV)[0]
            if space == "qwen3e06":
                excl[study] = {k for k, v in cos.items() if v < CONTENT_MIN}
            e["primary_azure_only"] = analyse_cell(cell, study, space, rows, V, crow, cV, set())
            if study in excl:
                e["sensitivity_excluding_below_0.80_qwen"] = analyse_cell(cell, study, space, rows, V, crow, cV, excl[study]) | {
                    "n_excluded": len(excl[study])}
            if e["counts"]["via"]["openrouter"] + e["counts"]["via"]["codex"]:
                r2, V2 = load_nc(study, space, True)
                e["sensitivity_with_fallback_filled"] = analyse_cell(cell, study, space, r2, V2, crow, cV, set())
            res[key] = e
            OUT.write_text(json.dumps(res, indent=1) + "\n")
            RPS.log(f"{key} done")
        del corp
    OUT.write_text(json.dumps(res, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry-run", "generate", "embed-qwen", "embed-bge", "agreement", "analyse"])
    ap.add_argument("--study", choices=["disc", "rep"], default="disc")
    ap.add_argument("--studies", default="disc,rep")
    ap.add_argument("--spaces", default="qwen3e06,bgem3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--confirm-paid", action="store_true")
    a = ap.parse_args()
    if a.cmd == "generate" and not a.confirm_paid:
        raise SystemExit("paid step: pass --confirm-paid (Javier approved a 30 USD cap for 6a + 6b)")
    {"dry-run": dry_run, "generate": lambda: generate(a.study, a.workers),
     "embed-qwen": lambda: embed(a.study, "qwen3e06"), "embed-bge": lambda: embed(a.study, "bgem3"),
     "agreement": agreement, "analyse": lambda: analyse(a.studies.split(","), a.spaces.split(","))}[a.cmd]()


if __name__ == "__main__":
    main()
