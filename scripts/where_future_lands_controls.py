"""Exploratory controls for where_future_lands (raised in review, not pre-registered).

1. Cross-domain control for F1: future items of one corpus against another corpus's past map, both embedders,
   all six pairings, full future sets. If out-of-domain text never lands in the far region either, F1 follows
   from the cap and says nothing about novelty; the informative quantity is the isolation shift.
2. Cell-level AUCs without cross-validation (hole, momentum), with bootstrap CIs over cells. The pre-registered
   F3 used 5-fold CV on a single feature with ~26 positive cells, where AUC < 0.5 can come from the folds alone.
Same seeds and cell construction as scripts/where_future_lands.py. 3. F2 on the numerators only (future count vs Gaussian-draw count), since growth and hole share a denominator.
Output: docs/prereg/where_future_lands_controls.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_future_lands as W  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/prereg/where_future_lands_controls.json"


def load(domain: str, emb_name: str):
    root = W.DATA / "prepared" / domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date"])
    emb = np.load(root / f"emb_{emb_name}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return meta, emb


def max_cos(Q: np.ndarray, P: np.ndarray, chunk: int = 1000) -> np.ndarray:
    return np.concatenate([(Q[i:i + chunk] @ P.T).max(1) for i in range(0, len(Q), chunk)])


def cells(domain: str, emb_name: str):
    """Reproduces the cell features of where_future_lands.analyse, consuming the RNG in the same order."""
    rng = np.random.default_rng(W.SEED)
    meta, emb = load(domain, emb_name)
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    pm = meta[past].reset_index(drop=True)
    held = np.zeros(len(P), bool)
    held[rng.choice(len(P), len(P) // 10, replace=False)] = True
    R = P[~held]
    rm = pm[~held].reset_index(drop=True)
    for _ in range(W.N_BOOT):  # the F1 bootstrap in analyse() draws these before the cells
        rng.integers(0, len(F), len(F))
        rng.integers(0, int(held.sum()), int(held.sum()))
    mu = R.mean(0)
    km = KMeans(n_clusters=W.K_CELLS, random_state=W.SEED, n_init=1).fit(R)
    past_n = np.bincount(km.labels_, minlength=W.K_CELLS)
    fut_n = np.bincount(km.predict(F), minlength=W.K_CELLS)
    growth = (fut_n / len(F)) / (past_n / len(R))
    w, U = np.linalg.eigh(np.cov(R.T, bias=True))
    draw_n = np.zeros(W.K_CELLS, np.int64)
    for i in range(0, W.N_DRAWS, 20_000):
        Z = rng.standard_normal((20_000, R.shape[1])).astype(np.float32) * np.sqrt(np.clip(w, 0, None)).astype(np.float32)
        C = mu + Z @ U.T.astype(np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        draw_n += np.bincount(km.predict(C), minlength=W.K_CELLS)
    hole = (draw_n / W.N_DRAWS) / (past_n / len(R))
    lo, hi = W.MOMENTUM_YEARS[domain]
    ry = W.years(rm["date"], domain)
    mom = np.bincount(km.labels_, weights=((ry >= lo) & (ry <= hi)).astype(float), minlength=W.K_CELLS) / past_n
    return growth, hole, mom, past_n, fut_n, draw_n


def auc_ci(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> dict:
    vals = []
    for _ in range(W.N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return {"auc": float(roc_auc_score(y, s)), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def main() -> None:
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("cross_domain", {})
    res["auc_no_cv"], res["f2_numerators"] = {}, {}
    for e in ([] if res["cross_domain"] else W.EMBEDDERS):
        data = {d: load(d, e) for d in W.DOMAINS}
        for tgt in W.DOMAINS:
            meta, emb = data[tgt]
            P = emb[(meta["split"] == "past").to_numpy()]
            for src in W.DOMAINS:
                ms, es = data[src]
                Q = es[(ms["split"] == "future").to_numpy()]
                mx = max_cos(Q, P)
                res["cross_domain"][f"{e}/{src}->{tgt}"] = {
                    "n": int(len(Q)), "in_domain": src == tgt, "min_max_cos": float(mx.min()),
                    "median_max_cos": float(np.median(mx)), "median_iso": float(1 - np.median(mx)),
                    "share_far": float(np.mean(mx <= W.FAR_COS))}
                print(time.strftime("%H:%M:%S"), e, src, "->", tgt, round(float(np.median(mx)), 3), flush=True)
        del data
    for e in W.EMBEDDERS:
        for d in W.DOMAINS:
            rng = np.random.default_rng(W.SEED)
            g, h, m, pn, fn, dn = cells(d, e)
            top = (g >= np.quantile(g, 0.9)).astype(int)
            res["auc_no_cv"][f"{d}/{e}"] = {"n_top": int(top.sum()), "hole": auc_ci(top, h, rng), "momentum": auc_ci(top, m, rng)}
            # F2 without the shared denominator: future count vs Gaussian-draw count, given momentum and log past count
            ctrl = np.column_stack([m, np.log(pn)])
            rho, p = W.partial_spearman(fn, dn, ctrl)
            boots = [W.partial_spearman(fn[i], dn[i], ctrl[i])[0] for i in (rng.integers(0, len(fn), len(fn)) for _ in range(W.N_BOOT))]
            res["f2_numerators"][f"{d}/{e}"] = {"partial_rho": float(rho), "p": float(p),
                                               "ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}
            print(time.strftime("%H:%M:%S"), d, e, res["auc_no_cv"][f"{d}/{e}"], flush=True)
            OUT.write_text(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
