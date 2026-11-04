"""Revision (point 3): first-public dates and versions of the continuation follow-ups/contexts, and the continuation
step gaps rerun after excluding follow-ups whose arXiv v1 predates the generator's training cutoff.

Needs scripts/revision_arxiv_versions.py to have written runs/{followups,followups-2025}/arxiv_audit/versions.json.
Reuses saved embeddings only (no encoder is loaded):
  - pre-registered B1 (Section 5.1): runs/continue-from-context/{emb.npz, idea_emb.npy} (gpt-4o, Qwen3);
  - robustness grid (Table A6 step-gap column): kaggle-output/* via continuation_reanalysis.load_generator;
  - Section 5.4 M1: runs/continue-modern/{emb.npz, partB_emb.npy} (gpt-5.4-mini, Qwen3).
Cutoffs (Table 1), excluded if v1 'published' date <= the last day of the cutoff month (conservative):
  gpt-4o October 2023 -> v1 <= 2023-10-31; OLMo February-March 2023 -> v1 <= 2023-03-31;
  gpt-5.4-mini 31 August 2025 -> v1 <= 2025-08-31. Also "design": the stated rule (Section 5) that every follow-up
  appeared on or after 2024-01-01 (2025-09-01 for Section 5.4), applied to v1.
Output: docs/prereg/revision_first_public.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continuation_reanalysis as R  # noqa: E402
import continue_from_context as M  # noqa: E402

D = M.DATA
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/revision_first_public.json"
SEED, N_BOOT = 20261030, 2000
CUT = {"gpt-4o": "2023-10-31", "olmo-7b-0424": "2023-03-31", "gpt-5.4-mini": "2025-08-31"}
DESIGN = {"followups": "2024-01-01", "followups-2025": "2025-09-01"}


def versions(run: str) -> dict:
    v = json.loads((D / "runs" / run / "arxiv_audit/versions.json").read_text())
    assert not v["missing"], v["missing"]
    return v["ids"]


def ctx_rows(run: str) -> list[dict]:
    rows = [json.loads(l) for l in (D / "runs" / run / "followups.jsonl").read_text().splitlines()]
    return [r for r in rows if r["n_followups"] >= 5][:150]


def v1(V, aid):
    return V[aid]["published"][:10]


def audit(run: str, V: dict, cutoffs: dict) -> dict:
    ctx = ctx_rows(run)
    fups = [(i, j, f) for i, c in enumerate(ctx) for j, f in enumerate(c["followups"])]
    d1 = np.array([v1(V, f["arxiv_id"]) for _, _, f in fups])
    s2 = np.array([f["date"] for _, _, f in fups])
    nv = np.array([V[f["arxiv_id"]]["n_versions"] for _, _, f in fups])
    upd = np.array([V[f["arxiv_id"]]["updated"][:10] for _, _, f in fups])
    cv1 = np.array([v1(V, c["arxiv_id"]) for c in ctx])
    cdate = np.array([c["date"][:10] for c in ctx])
    cnv = np.array([V[c["arxiv_id"]]["n_versions"] for c in ctx])
    cupd = np.array([V[c["arxiv_id"]]["updated"][:10] for c in ctx])
    out = {
        "n_contexts": len(ctx), "n_followup_pairs": len(fups), "n_unique_followups": len({f["arxiv_id"] for _, _, f in fups}),
        "followups": {
            "v1_before_design_date": {"date": DESIGN[run], "n": int((d1 < DESIGN[run]).sum()),
                                      "share": float((d1 < DESIGN[run]).mean()),
                                      "contexts_affected": int(len({i for (i, _, _), x in zip(fups, d1) if x < DESIGN[run]}))},
            "v1_on_or_before_cutoff": {g: {"cutoff": c, "n": int((d1 <= c).sum()), "share": float((d1 <= c).mean()),
                                            "contexts_affected": int(len({i for (i, _, _), x in zip(fups, d1) if x <= c}))}
                                       for g, c in cutoffs.items()},
            "v1_year_counts": dict(sorted(Counter(x[:4] for x in d1).items())),
            "v1_differs_from_s2_date": int((d1 != s2).sum()),
            "v1_earlier_than_s2_date": int((d1 < s2).sum()),
            "median_days_s2_minus_v1_when_earlier": float(np.median([(np.datetime64(b) - np.datetime64(a)).astype(int)
                                                                     for a, b in zip(d1, s2) if a < b])) if (d1 < s2).any() else None,
            "n_versions_gt1": int((nv > 1).sum()), "share_versions_gt1": float((nv > 1).mean()),
            "n_versions_counts": {int(k): int(v) for k, v in sorted(Counter(nv.tolist()).items())},
        },
        "contexts": {
            "v1_equals_corpus_date": int((cv1 == cdate).sum()),
            "v1_differs": [{"id": c["arxiv_id"], "corpus": a, "v1": b} for c, a, b in zip(ctx, cdate, cv1) if a != b],
            "n_versions_gt1": int((cnv > 1).sum()), "share_versions_gt1": float((cnv > 1).mean()),
            "n_versions_counts": {int(k): int(v) for k, v in sorted(Counter(cnv.tolist()).items())},
            "latest_version_after_cutoff": {g: int(((cnv > 1) & (cupd > c)).sum()) for g, c in cutoffs.items()},
        },
    }
    if run == "followups":
        out["followups"]["v1_before_2023-10-01"] = int((d1 < "2023-10-01").sum())
        out["followups"]["latest_version_updated_after_snapshot_fetch_2026-10-30"] = int((upd > "2026-10-30").sum())
    return out


def boot_ci(x, rng):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarise(g: list[float], dropped: int, rng) -> dict:
    g = np.array(g)
    return {"n_contexts": int(len(g)), "contexts_dropped": dropped, "mean_step_gap": float(g.mean()), "ci": boot_ci(g, rng),
            "p_one_sided": float(stats.wilcoxon(g, alternative="greater").pvalue),
            "p_two_sided": float(stats.wilcoxon(g).pvalue)}


def gaps(C, F_list, ideas_by_ctx, keep_list, min_keep, rng) -> dict:
    g, dropped = [], 0
    for i in range(len(C)):
        k = keep_list[i]
        if k.sum() < min_keep or not len(ideas_by_ctx[i]):
            dropped += 1
            continue
        g.append(float(np.median(ideas_by_ctx[i] @ C[i]) - np.median(F_list[i][k] @ C[i])))
    return summarise(g, dropped, rng)


def load_partb(gen: str, space: str):
    """As continuation_reanalysis.load_generator, without the iterative ideas (gpt-4o's iter_ideas.jsonl has since
    grown with the ORACLE-H arm of Section 5.4, so its row-count assertion no longer holds; Part B is unaffected)."""
    import pandas as pd
    ctx = M.contexts()
    nf = [len(c["followups"]) for c in ctx]
    if gen == "gpt-4o":
        t = pd.read_parquet(D / "kaggle-upload/continuation-texts/texts.parquet")
        Vv = R.norm(np.load(D / f"kaggle-output/embed-continuation-v1/emb_{space}.npy").astype(np.float32))
        kinds = t["kind"].to_numpy()
        pb = [json.loads(l) for l in (M.RUN / "ideas.jsonl").read_text().splitlines()]
        C, Fall, PB = Vv[kinds == "ctx"], Vv[kinds == "fup"], Vv[kinds == "partB"]
    else:
        K = D / "kaggle-output/llama-continuation-v1"
        Vv = R.norm(np.load(K / f"emb_{space}.npy").astype(np.float32))
        pb = [json.loads(l) for l in (K / "ideas_llama.jsonl").read_text().splitlines()]
        n_pb = sum(1 for r in pb if r["abstract"])
        C = Vv[:len(ctx)]
        Fall = Vv[len(ctx):len(ctx) + sum(nf)]
        PB = Vv[len(ctx) + sum(nf):len(ctx) + sum(nf) + n_pb]
    pb = [r for r in pb if r["abstract"]]
    assert len(pb) == len(PB) and len(Fall) == sum(nf), (gen, len(pb), len(PB))
    starts = np.cumsum([0] + nf)
    return C, [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))], pb, PB


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"cutoff_rule": "excluded if arXiv v1 published date <= cutoff", "cutoffs": CUT, "design_dates": DESIGN}
    V = versions("followups")
    res["audit_section5"] = audit("followups", V, {g: CUT[g] for g in ("gpt-4o", "olmo-7b-0424")})
    ctx = ctx_rows("followups")
    intents = json.loads((M.RUN / "followup_intents.json").read_text())
    v1s = [np.array([v1(V, f["arxiv_id"]) for f in c["followups"]]) for c in ctx]
    cont = [np.array([bool((v := intents.get(f"{i}/{j}", {})) and (v.get("influential") or {"methodology", "result"} & set(v.get("intents", []))))
                      for j in range(len(c["followups"]))]) for i, c in enumerate(ctx)]

    def masks(scen: str, gen: str, fset: str):
        base = cont if fset == "continuing" else [np.ones(len(x), bool) for x in v1s]
        if scen == "original":
            return base
        bound = CUT[gen] if scen == "own_cutoff" else None
        return [b & ((x > bound) if bound else (x >= DESIGN["followups"])) for b, x in zip(base, v1s)]

    # pre-registered B1 (runtime embeddings of the pre-registered run)
    z = np.load(M.RUN / "emb.npz")
    C, Fall, owner = z["C"], z["F"], z["owner"]
    Fl = [Fall[owner == i] for i in range(len(C))]
    ideas = [r for r in map(json.loads, (M.RUN / "ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    I = np.load(M.RUN / "idea_emb.npy")
    assert len(I) == len(ideas)
    b1 = {}
    for arm in ("CTX", "CTX+NB"):
        ibc = [I[[n for n, r in enumerate(ideas) if r["ctx"] == i and r["arm"] == arm]] for i in range(len(C))]
        for scen in ("original", "own_cutoff", "design_2024"):
            for mk in (1, 5):
                b1[f"{arm}/{scen}/min{mk}"] = gaps(C, Fl, ibc, masks(scen, "gpt-4o", "all"), mk, rng)
    res["B1_prereg_gpt4o_qwen3"] = b1

    # robustness grid (Table A6 step-gap column; the "p < 1e-11 in all eight cells" claim, two-sided as in the grid)
    grid = {}
    for space in R.SPACES:
        for gen in ("gpt-4o", "olmo-7b-0424"):
            Cg, Fg, pb, PB = load_partb(gen, space)
            ibc = [PB[[n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == "CTX"]] for i in range(len(Cg))]
            for fset in ("all", "continuing"):
                for scen in ("original", "own_cutoff", "design_2024"):
                    mins = (1, 5) if fset == "all" else (1, 3)
                    for mk in mins:
                        grid[f"{gen}/{space}/{fset}/{scen}/min{mk}"] = gaps(Cg, Fg, ibc, masks(scen, gen, fset), mk, rng)
    res["grid_CTX"] = grid

    # Section 5.4 (gpt-5.4-mini, 2025 contexts)
    V2 = versions("followups-2025")
    res["audit_section5_4"] = audit("followups-2025", V2, {"gpt-5.4-mini": CUT["gpt-5.4-mini"]})
    ctx2 = ctx_rows("followups-2025")
    run2 = D / "runs/continue-modern"
    z = np.load(run2 / "emb.npz")
    C2, F2, own2 = z["C"], z["F"], z["owner"]
    Fl2 = [F2[own2 == i] for i in range(len(C2))]
    rows2 = [r for r in map(json.loads, (run2 / "ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    P2 = np.load(run2 / "partB_emb.npy")
    assert len(P2) == len(rows2)
    v1s2 = [np.array([v1(V2, f["arxiv_id"]) for f in c["followups"]]) for c in ctx2]
    m54 = {}
    for arm in ("CTX", "CTX+MEM"):
        ibc = [P2[[n for n, r in enumerate(rows2) if r["ctx"] == i and r["arm"] == arm]] for i in range(len(C2))]
        for scen, km in (("original", [np.ones(len(x), bool) for x in v1s2]),
                         ("own_cutoff", [x > CUT["gpt-5.4-mini"] for x in v1s2])):
            for mk in (1, 5):
                m54[f"{arm}/{scen}/min{mk}"] = gaps(C2, Fl2, ibc, km, mk, rng)
    res["M1_gpt54mini_qwen3"] = m54

    # Point 2 context: labelled sample
    s2 = json.loads((Path(__file__).resolve().parents[1] / "docs/paper/followup_check_s2.json").read_text())
    res["point2_labelled_sample"] = {k: s2[k] for k in ("n_pairs", "labels", "label_continuing", "s2_continuing",
                                                         "label_continuing_and_s2_continuing", "s2_continuing_by_label",
                                                         "s2_all_followups_continuing_share")}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("audit_section5", "audit_section5_4")}, indent=1))
    for blk in ("B1_prereg_gpt4o_qwen3", "grid_CTX", "M1_gpt54mini_qwen3"):
        for k, v in res[blk].items():
            print(blk, k, v["n_contexts"], "%.4f" % v["mean_step_gap"], "[%.3f, %.3f]" % tuple(v["ci"]),
                  "p1 %.1e p2 %.1e" % (v["p_one_sided"], v["p_two_sided"]))


if __name__ == "__main__":
    main()
