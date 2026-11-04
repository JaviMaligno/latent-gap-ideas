"""Exploratory (not pre-registered): is the drop in fit after critique mechanical or directional?

For every idea (iterative critique, all arms and rounds), take a real later arXiv paper of our corpus (2023+, not a
follow-up of the context; with replacement) whose cosine to the context is closest to the idea's
step, and compute its fit (max cos to the real follow-ups) and hit (nearest neighbour among follow-ups and the
same-category distractors, the paper itself excluded). Excess = idea - matched paper: positive means the idea points
towards the follow-ups better than an arbitrary paper at the same distance.
Output: key "iterative_critique_baseline" in docs/prereg/continue_from_context_result.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402
import iterative_critique as IC  # noqa: E402


def main() -> None:
    meta, emb = M.corpus()
    past = (meta["split"] == "past").to_numpy()
    fut = meta[~past].reset_index(drop=True)
    Fe = emb[~past]
    by_id = meta.set_index("id")
    z = np.load(M.RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ctx = M.contexts()
    rows = [json.loads(l) for l in IC.OUT.read_text().splitlines()]
    rows = [r for r in rows if r["abstract"] and np.isfinite(r["step"])]
    acc = {(a, r): {"idea_fit": [], "base_fit": [], "base_hit": []} for a in IC.ARMS for r in range(IC.N_ROUNDS)}
    for i, c in enumerate(ctx):
        Fi = F[owner == i]
        fids = {f["arxiv_id"] for f in c["followups"]}
        notf = ~fut["id"].isin(fids).to_numpy()
        same = (fut["primary"] == by_id.loc[c["arxiv_id"], "primary"]).to_numpy()
        pool_idx = np.where(notf)[0]
        s_all = Fe[pool_idx] @ C[i]
        order = np.argsort(s_all)
        sorted_s = s_all[order]
        used: set[int] = set()
        dist_idx = np.where(notf & same)[0]
        Dsim = Fe[dist_idx] @ Fi.T  # distractor-to-follow-up similarities, reused below
        for x in (r for r in rows if r["ctx"] == i):
            j = int(np.searchsorted(sorted_s, x["step"]))
            # nearest in step, with replacement: a shared no-reuse pool starved later arms of close matches
            k = min((k for k in (j - 1, j) if 0 <= k < len(order)), key=lambda k: abs(sorted_s[k] - x["step"]))
            pick = pool_idx[order[k]]
            v = Fe[pick]
            fit = float((Fi @ v).max())
            sd = Fe[dist_idx] @ v
            sd = sd[dist_idx != pick]
            hit = float(fit > sd.max())
            d = acc[(x["arm"], x["round"])]
            d["idea_fit"].append(x["fit"]); d["base_fit"].append(fit); d["base_hit"].append(hit)
        del Dsim
    prev = json.loads((Path(__file__).resolve().parents[1] / "docs/prereg/continue_from_context_result.json").read_text())["iterative_critique"]
    res = {}
    for a in IC.ARMS:
        res[a] = {}
        for r in range(IC.N_ROUNDS):
            d = {k: np.array(v) for k, v in acc[(a, r)].items()}
            ex = d["idea_fit"] - d["base_fit"]
            res[a][f"round{r}"] = {"idea_fit": float(d["idea_fit"].mean()), "matched_fit": float(d["base_fit"].mean()),
                                   "excess_fit": float(ex.mean()), "excess_fit_p": float(stats.wilcoxon(ex).pvalue),
                                   "idea_hit": prev[a][f"round{r}"]["hit"], "matched_hit": float(d["base_hit"].mean())}
        e0 = np.array(acc[(a, 0)]["idea_fit"]) - np.array(acc[(a, 0)]["base_fit"])
        e2 = np.array(acc[(a, 2)]["idea_fit"]) - np.array(acc[(a, 2)]["base_fit"])
        res[a]["excess_change_r0_r2"] = float(e2.mean() - e0.mean())
        res[a]["excess_change_mannwhitney_p"] = float(stats.mannwhitneyu(e0, e2).pvalue)
    M.save("iterative_critique_baseline", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
