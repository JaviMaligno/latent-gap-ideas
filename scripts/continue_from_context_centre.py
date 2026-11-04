"""Exploratory (not pre-registered): do LLM continuations drift towards the centre of the cap more than real
follow-ups and matched non-follow-ups? Projects each idea's displacement from its context on the Part A centre
direction (same RNG stream, so the same directions). Output: key "B_centre" in continue_from_context_result.json
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
    mu_hat = M.norm(P.mean(0))
    z = np.load(M.RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ideas = [json.loads(l) for l in (M.RUN / "ideas.jsonl").read_text().splitlines()]
    ideas = [r for r in ideas if r["abstract"]]
    I = np.load(M.RUN / "idea_emb.npy")
    rng = np.random.default_rng(M.SEED)
    per = {a: [] for a in M.ARMS}
    fup = []
    for i, c in enumerate(C):
        _, _, dirs = M.directions(c, P, pdate, mu_hat, rng)
        v = dirs["centre"]
        fup.append(float(np.mean([M.cosv(f - c, v) for f in F[owner == i]])))
        for a in M.ARMS:
            idx = [n for n, r in enumerate(ideas) if r["ctx"] == i and r["arm"] == a]
            per[a].append(float(np.mean([M.cosv(I[n] - c, v) for n in idx])))
    fup = np.array(fup)
    res = {"followups": float(fup.mean())}
    for a, x in per.items():
        x = np.array(x)
        res[a] = {"mean": float(x.mean()), "diff_vs_followups": float((x - fup).mean()),
                  "wilcoxon_p": float(stats.wilcoxon(x, fup).pvalue)}
    M.save("B_centre", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
