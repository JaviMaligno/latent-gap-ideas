"""Exploratory (raised in review, not pre-registered): a step-invariant direction test for the critique arms.

Why: the distance-matched baseline of iterative_critique is out of support at round 0 (most gpt-4o ideas sit closer to
the context than any later paper of the corpus), so its "excess change" mixes direction with step mismatch.
This script reports, per generator x embedder cell:
1. support: share of ideas whose step exceeds the nearest later paper's cosine to the context, per round;
2. tangent alignment: the component of each idea orthogonal to its context, cosine with the follow-ups' mean
   tangent direction; change from round 0 to round 2 (step-invariant by construction);
3. specificity: the same change split into the direction of the nearest non-citing later papers (as many as the
   context has follow-ups) and the follow-up direction orthogonalised against it;
4. positive control: leave-one-out tangent alignment of the real follow-ups themselves.
Output: docs/prereg/direction_tangent.json
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

OUT = Path(__file__).resolve().parents[1] / "docs/prereg/direction_tangent.json"


def tang(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    T = X - np.outer(X @ c, c)
    return T / np.linalg.norm(T, axis=-1, keepdims=True)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main() -> None:
    D = R.M.DATA
    meta = pd.read_parquet(D / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    fut_ids = meta.loc[fut, "id"].to_numpy()
    res = {}
    for space in R.SPACES:
        Fe = R.norm(np.load(D / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep][fut])
        for gen in ("gpt-4o", "olmo-7b-0424"):
            ctx, C, F, pb, PB, it, IT = R.load_generator(gen, space)
            sup = {r: [] for r in range(3)}
            align = {a: {0: [], 2: []} for a in R.I_ARMS}
            pool_ch = {a: [] for a in R.I_ARMS}
            spec_ch = {a: [] for a in R.I_ARMS}
            loo = []
            for i in range(len(ctx)):
                c = C[i]
                TF = tang(F[i], c)
                t = unit(TF.mean(0))
                if len(TF) > 1:
                    S = TF.sum(0)
                    loo.append(float(np.mean([TF[j] @ unit(S - TF[j]) for j in range(len(TF))])))
                fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
                pool = Fe[~np.isin(fut_ids, list(fids))]
                s = pool @ c
                smax = float(s.max())
                tp = unit(tang(pool[np.argsort(-s)[:len(F[i])]], c).mean(0))
                ts = unit(t - (t @ tp) * tp)
                for a in R.I_ARMS:
                    v = {}
                    for r in range(3):
                        idx = [m for m, x in enumerate(it) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                        if not idx:
                            continue
                        X = IT[idx]
                        sup[r].extend((X @ c > smax).tolist())
                        T = tang(X, c)
                        v[r] = (float((T @ t).mean()), float((T @ tp).mean()), float((T @ ts).mean()))
                    if 0 in v and 2 in v:
                        align[a][0].append(v[0][0]); align[a][2].append(v[2][0])
                        pool_ch[a].append(v[2][1] - v[0][1]); spec_ch[a].append(v[2][2] - v[0][2])
            cell = {"followups_loo_alignment": float(np.mean(loo)),
                    "share_above_pool_max": {f"round{r}": float(np.mean(x)) for r, x in sup.items() if x}}
            for a in R.I_ARMS:
                a0, a2 = np.array(align[a][0]), np.array(align[a][2])
                pc, sc = np.array(pool_ch[a]), np.array(spec_ch[a])
                cell[a] = {"alignment_round0": float(a0.mean()), "alignment_round2": float(a2.mean()),
                           "alignment_change": float((a2 - a0).mean()), "alignment_change_p": float(stats.wilcoxon(a2 - a0).pvalue),
                           "towards_nearest_noncitingdir": float(pc.mean()), "towards_nearest_noncitingdir_p": float(stats.wilcoxon(pc).pvalue),
                           "towards_followup_specific": float(sc.mean()), "towards_followup_specific_p": float(stats.wilcoxon(sc).pvalue)}
            key = f"{gen}/{space}"
            res[key] = cell
            OUT.write_text(json.dumps(res, indent=1))
            print(key, json.dumps({a: {k: round(v, 4) for k, v in cell[a].items()} for a in R.I_ARMS}), cell["share_above_pool_max"],
                  "LOO %.3f" % cell["followups_loo_alignment"], flush=True)


if __name__ == "__main__":
    main()
