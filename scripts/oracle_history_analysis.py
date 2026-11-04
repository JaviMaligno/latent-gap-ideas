"""Analysis of ORACLE-H vs ORACLE (docs/prereg/modern_and_oracle_history.md, H1 and H2; M4 when LG_RUN points to the
modern run). Uses the step/fit stored at generation (local Qwen3, same encoder as the corpus to cos >= 0.99999) and
embeds the ideas locally for the tangent-alignment measures of scripts/direction_tangent.py.
Output: key "<run>" in docs/prereg/oracle_history.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import continue_from_context as M  # noqa: E402
import iterative_critique as IC  # noqa: E402
from latent_gap.arxiv import item_text  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/prereg/oracle_history.json"


def tang(X, c):
    """Tangent part of X at c, normalised; NaN for an idea identical to its context (one gpt-5.4-mini idea copies the
    context title verbatim), which is then left out of that context's mean."""
    T = X - np.outer(X @ c, c)
    n = np.linalg.norm(T, axis=-1, keepdims=True)
    return np.where(n > 1e-6, T / np.maximum(n, 1e-12), np.nan)


def unit(v):
    return v / np.linalg.norm(v)


def boot(x, rng):
    return [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], q)) for q in (2.5, 97.5)]


def main() -> None:
    from latent_gap.embedder import LocalQwen
    arms = tuple(os.environ.get("LG_ITER_ARMS", "ORACLE,ORACLE-H").split(","))
    z = np.load(M.RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ctx = M.contexts()
    rows = [json.loads(l) for l in IC.OUT.read_text().splitlines()]
    rows = [r for r in rows if r["abstract"] and r["arm"] in arms]
    cache = M.RUN / f"oracle_history_emb_{'_'.join(arms)}.npy"
    if cache.exists() and len(np.load(cache)) == len(rows):
        V = np.load(cache)
    else:
        V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6"))).embed([item_text(r["title"], r["abstract"]) for r in rows])
        np.save(cache, V)
    meta = pd.read_parquet(M.DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult", "date"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    emb = M.norm(np.load(M.DATA / "prepared/arxiv/emb_qwen3e06.npy").astype(np.float32)[keep])
    later = (meta["date"] >= min(f["date"] for c in ctx for f in c["followups"])).to_numpy()
    pool_all, pool_ids = emb[later], meta.loc[later, "id"].to_numpy()
    rng = np.random.default_rng(M.SEED)
    per = {a: {"step": [], "fit": [], "align": [], "spec": []} for a in arms}
    for i, c in enumerate(C):
        Fi = F[owner == i]
        t = unit(tang(Fi, c).mean(0))
        fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
        pool = pool_all[~np.isin(pool_ids, list(fids))]
        tp = unit(tang(pool[np.argsort(-(pool @ c))[:len(Fi)]], c).mean(0))
        ts = unit(t - (t @ tp) * tp)
        for a in arms:
            v = {}
            for r in (0, 2):
                idx = [n for n, x in enumerate(rows) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                if idx:
                    T = tang(V[idx], c)
                    v[r] = (np.mean([rows[n]["step"] for n in idx]), np.mean([rows[n]["fit"] for n in idx]),
                            float(np.nanmean(T @ t)), float(np.nanmean(T @ ts)))
            if 0 in v and 2 in v:
                for j, k in enumerate(("step", "fit", "align", "spec")):
                    per[a][k].append(v[2][j] - v[0][j] if k != "step" else v[0][j] - v[2][j])
    res = {}
    for a in arms:
        d = {k: np.array(x) for k, x in per[a].items()}
        res[a] = {"n": int(len(d["fit"]))}
        for k, x in d.items():
            res[a][k] = {"mean": float(x.mean()), "ci": boot(x, rng), "p": float(stats.wilcoxon(x).pvalue)}
    if "ORACLE" in per and "ORACLE-H" in per:
        n = min(len(per["ORACLE"]["fit"]), len(per["ORACLE-H"]["fit"]))
        dh = np.array(per["ORACLE-H"]["fit"][:n]) - np.array(per["ORACLE"]["fit"][:n])
        res["H1_fit_gain_H_minus_plain"] = {"mean": float(dh.mean()), "ci": boot(dh, rng),
                                            "p_one_sided": float(stats.wilcoxon(dh, alternative="greater").pvalue)}
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres.setdefault(os.environ.get("LG_RUN", "continue-from-context"), {}).update(res)  # keep M1 if present
    OUT.write_text(json.dumps(allres, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
