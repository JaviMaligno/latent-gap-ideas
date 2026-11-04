"""Exploratory baseline for Part A of continue_from_context (not pre-registered; added after the A2 result).

The pre-registered "hole" and "centre" directions both point from the context towards the bulk of its
neighbourhood, so any later paper near the context may project positively on them (regression to the mean). For
every real follow-up f of context c, take the later arXiv paper of our corpus (2023+, not a follow-up of c) whose
cosine to c is closest to cos(c, f), and project its displacement on the same three directions. Follow-ups and
matched non-follow-ups are compared per context (paired Wilcoxon, two-sided).
Output: key "A_baseline" in docs/prereg/continue_from_context_result.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402


def main() -> None:
    meta, emb = M.corpus()
    past = (meta["split"] == "past").to_numpy()
    P, pdate = emb[past], meta.loc[past, "date"].str[:4].to_numpy()
    fut_ids = meta.loc[~past, "id"].to_numpy()
    Fe = emb[~past]
    mu_hat = M.norm(P.mean(0))
    z = np.load(M.RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ctx = M.contexts()
    rng = np.random.default_rng(M.SEED)  # same RNG stream as part_a, so the directions are identical
    out = {k: {"fup": [], "ctl": []} for k in ("momentum", "hole", "centre")}
    gaps = []
    for i, c in enumerate(C):
        _, _, dirs = M.directions(c, P, pdate, mu_hat, rng)
        Fi = F[owner == i]
        fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
        ok = ~np.isin(fut_ids, list(fids))
        s_all = Fe[ok] @ c
        pool = Fe[ok]
        steps = Fi @ c
        # matched control per follow-up: the unused later paper with the closest cosine to the context
        order = np.argsort(s_all)
        used = set()
        ctl = []
        for s in steps:
            j = int(np.searchsorted(s_all[order], s))
            for cand in (order[k] for k in sorted(range(max(0, j - 50), min(len(order), j + 50)), key=lambda k: abs(s_all[order[k]] - s))):
                if cand not in used:
                    used.add(cand)
                    ctl.append(cand)
                    break
        Ci = pool[ctl]
        gaps.append(float(np.median(np.abs(Ci @ c - steps))))
        for k, v in dirs.items():
            out[k]["fup"].append(float(np.nanmean([M.cosv(f - c, v) for f in Fi])))
            out[k]["ctl"].append(float(np.nanmean([M.cosv(g - c, v) for g in Ci])))
    res = {"n_contexts": len(C), "median_abs_step_mismatch": float(np.median(gaps))}
    for k, d in out.items():
        a, b = np.array(d["fup"]), np.array(d["ctl"])
        okk = ~(np.isnan(a) | np.isnan(b))
        res[k] = {"followups": float(a[okk].mean()), "matched_nonfollowups": float(b[okk].mean()),
                  "diff": float((a - b)[okk].mean()), "wilcoxon_p": float(stats.wilcoxon(a[okk], b[okk]).pvalue)}
    M.save("A_baseline", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
