"""Revision of Section 5.3 / Table 3 (direction_tangent.py), raised in external review. Exploratory, not pre-registered.

(a) Point 8: Holm over the 36 tests declared in docs/prereg/direction_tangent.json (4 generator x embedder cells
    x 3 arms x {alignment change, change along p, change along s}); also Holm within the 12 follow-up-specific tests.
(b) Point 15: per context, f = a p + b s with a = f.p and b = sqrt(1 - a^2), so for every idea T.f = a T.p + b T.s
    exactly and the alignment change splits into a*d(T.p) + b*d(T.s). Reports the mean weighted contributions,
    their share of the alignment change, and Wilcoxon p of each weighted contribution.
(c) Point 14: recomputes the same quantities with the 'nearest later non-citing papers' pool cleaned of the complete
    citing sets (runs/followups/complete_citations, from revision_citing_fetch.py):
      - 'saved'     : excludes only the saved follow-up ids (original analysis, must reproduce direction_tangent.json);
      - 'complete24': excludes every citing arXiv paper dated >= 2024-01-01 (removes the 50-cap effect);
      - 'complete'  : excludes every citing arXiv paper of any date (true non-citing pool; the future split starts
                      2023-01-01, before the follow-up window).
Output: docs/prereg/revision_direction_tangent.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continuation_reanalysis as R  # noqa: E402
from direction_tangent import tang, unit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/prereg/direction_tangent.json"
OUT = ROOT / "docs/prereg/revision_direction_tangent.json"
CACHE = R.M.DATA / "runs/followups/complete_citations"
TESTS = (("alignment_change", "alignment"), ("towards_nearest_noncitingdir", "along_p"),
         ("towards_followup_specific", "along_s"))


def load_generator(gen: str, space: str):
    """R.load_generator, but iter_ideas.jsonl now also holds ORACLE-H rows appended after the embeddings were made
    (oracle_history.py); they come after the 4,050 GEO/LLM/ORACLE rows, so keeping the first rows of the three
    original arms keeps idea i aligned with embedding row i."""
    orig = R.json.loads
    it_path = R.M.RUN / "iter_ideas.jsonl"
    text = it_path.read_text()
    kept = "\n".join(l for l in text.splitlines() if orig(l)["arm"] in R.I_ARMS)
    real_read = Path.read_text
    Path.read_text = lambda self, *a, **k: kept if self == it_path else real_read(self, *a, **k)
    try:
        return R.load_generator(gen, space)
    finally:
        Path.read_text = real_read


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def part_a() -> dict:
    old = json.loads(OLD.read_text())
    rows = [{"cell": cell, "arm": arm, "test": name, "mean": old[cell][arm][k], "p_raw": old[cell][arm][k + "_p"]}
            for cell in old for arm in R.I_ARMS for k, name in TESTS]
    p = np.array([r["p_raw"] for r in rows])
    for r, a in zip(rows, holm(p)):
        r["p_holm36"] = float(a)
    spec = [r for r in rows if r["test"] == "along_s"]
    for r, a in zip(spec, holm(np.array([r["p_raw"] for r in spec]))):
        r["p_holm12_within_along_s"] = float(a)
    return {"n_tests": len(rows), "tests": rows,
            "survive_holm36_at_0.05": [f"{r['cell']} {r['arm']} {r['test']}" for r in rows if r["p_holm36"] < 0.05 and r["test"] == "along_s"]}


def cited(mode: str, ctx: list[dict]) -> list[set]:
    out = []
    for c in ctx:
        s = {f["arxiv_id"] for f in c["followups"]}
        if mode != "saved":
            d = json.loads((CACHE / f"{c['arxiv_id'].replace('/', '_')}.json").read_text())
            s |= {x["arxiv_id"] for x in d["citing"] if mode == "complete" or (x["date"] or "") >= "2024-01-01"}
        out.append(s)
    return out


def w(x: np.ndarray) -> float:
    return float(stats.wilcoxon(x).pvalue)


def run_mode(mode: str, meta_ids: np.ndarray, Fes: dict) -> dict:
    res = {}
    for space in R.SPACES:
        Fe = Fes[space]
        for gen in ("gpt-4o", "olmo-7b-0424"):
            ctx, C, F, pb, PB, it, IT = load_generator(gen, space)
            excl = cited(mode, ctx)
            rec = {a: {k: [] for k in ("dal", "dp", "ds", "wa", "wb")} for a in R.I_ARMS}
            pool_removed = []
            for i in range(len(ctx)):
                c = C[i]
                t = unit(tang(F[i], c).mean(0))
                m = ~np.isin(meta_ids, list(excl[i]))
                pool_removed.append(int((~m).sum()))
                pool = Fe[m]
                s = pool @ c
                tp = unit(tang(pool[np.argsort(-s)[:len(F[i])]], c).mean(0))
                ts = unit(t - (t @ tp) * tp)
                a_ = float(t @ tp)
                b_ = float(np.sqrt(max(0.0, 1 - a_ ** 2)))
                for arm in R.I_ARMS:
                    v = {}
                    for r in (0, 2):
                        idx = [k for k, x in enumerate(it) if x["ctx"] == i and x["arm"] == arm and x["round"] == r]
                        if idx:
                            T = tang(IT[idx], c)
                            v[r] = (float((T @ t).mean()), float((T @ tp).mean()), float((T @ ts).mean()))
                    if 0 in v and 2 in v:
                        q = rec[arm]
                        q["dal"].append(v[2][0] - v[0][0]); q["dp"].append(v[2][1] - v[0][1]); q["ds"].append(v[2][2] - v[0][2])
                        q["wa"].append(a_); q["wb"].append(b_)
            cell = {"n_ctx": len(ctx), "mean_citing_removed_from_pool": float(np.mean(pool_removed))}
            for arm in R.I_ARMS:
                q = {k: np.array(x) for k, x in rec[arm].items()}
                cp, cs = q["wa"] * q["dp"], q["wb"] * q["ds"]
                assert np.allclose(cp + cs, q["dal"], atol=1e-5), (mode, gen, space, arm)
                dal = float(q["dal"].mean())
                cell[arm] = {"n": int(len(q["dal"])), "alignment_change": dal, "alignment_change_p": w(q["dal"]),
                             "along_p_unweighted": float(q["dp"].mean()), "along_p_unweighted_p": w(q["dp"]),
                             "along_s_unweighted": float(q["ds"].mean()), "along_s_unweighted_p": w(q["ds"]),
                             "weight_a_fp_mean": float(q["wa"].mean()), "weight_b_mean": float(q["wb"].mean()),
                             "contrib_p_weighted": float(cp.mean()), "contrib_p_weighted_p": w(cp),
                             "contrib_s_weighted": float(cs.mean()), "contrib_s_weighted_p": w(cs),
                             "share_p": float(cp.mean() / dal), "share_s": float(cs.mean() / dal)}
            res[f"{gen}/{space}"] = cell
            print(mode, gen, space, {a: (round(cell[a]["contrib_p_weighted"], 4), round(cell[a]["contrib_s_weighted"], 4),
                                         round(cell[a]["along_s_unweighted_p"], 4)) for a in R.I_ARMS}, flush=True)
    return res


def cap_counts() -> dict:
    ctx = R.M.contexts()
    out = {"n_contexts": len(ctx), "followups_saved": int(sum(c["n_followups"] for c in ctx)),
           "contexts_at_cap_50": int(sum(c["n_citing_arxiv"] >= 50 for c in ctx))}
    if CACHE.exists() and all((CACHE / f"{c['arxiv_id'].replace('/', '_')}.json").exists() for c in ctx):
        full = [json.loads((CACHE / f"{c['arxiv_id'].replace('/', '_')}.json").read_text())["citing"] for c in ctx]
        n24 = np.array([sum((x["date"] or "") >= "2024-01-01" for x in f) for f in full])
        nall = np.array([len(f) for f in full])
        capped = np.array([c["n_citing_arxiv"] >= 50 for c in ctx])
        out.update({"complete_citing_arxiv_ge2024_total": int(n24.sum()), "complete_citing_arxiv_all_dates_total": int(nall.sum()),
                    "contexts_with_more_than_50_ge2024_now": int((n24 > 50).sum()),
                    "capped_contexts_complete_ge2024": sorted(int(x) for x in n24[capped]),
                    "saved_followups_missing_from_complete": int(sum(len({f2["arxiv_id"] for f2 in c["followups"]} - {x["arxiv_id"] for x in f})
                                                                     for c, f in zip(ctx, full)))})
    return out


def main() -> None:
    D = R.M.DATA
    meta = pd.read_parquet(D / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    ids = meta.loc[fut, "id"].to_numpy()
    Fes = {s: R.norm(np.load(D / f"prepared/arxiv/emb_{s}.npy").astype(np.float32)[keep][fut]) for s in R.SPACES}
    out = {"holm": part_a(), "cap": cap_counts()}
    modes = ["saved"] + (["complete24", "complete"] if "capped_contexts_complete_ge2024" in out["cap"] else [])
    out["decomposition"] = {m: run_mode(m, ids, Fes) for m in modes}
    # Holm over 36 for each pool definition, with the unweighted tests as declared
    for m in modes:
        rows = [(cell, arm, k) for cell in out["decomposition"][m] for arm in R.I_ARMS
                for k in ("alignment_change", "along_p_unweighted", "along_s_unweighted")]
        adj = holm(np.array([out["decomposition"][m][c][a][k + "_p"] for c, a, k in rows]))
        out["decomposition"][m]["holm36_survivors_along_s"] = [f"{c} {a}" for (c, a, k), q in zip(rows, adj)
                                                               if k == "along_s_unweighted" and q < 0.05]
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["holm"]["survive_holm36_at_0.05"]), json.dumps(out["cap"]))


if __name__ == "__main__":
    main()
