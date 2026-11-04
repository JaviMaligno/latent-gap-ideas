"""Revision reanalysis of the where-the-future-lands test (Section 4, Appendix A.1-A.2).

Answers three review points without touching the registered script or its result JSON:
  (a) F3 gain intervals with a cluster bootstrap: cells are resampled with replacement, folds are
      split by original cell id (StratifiedGroupKFold), so copies of one cell never sit on both
      sides of a fold, and the z-scoring is fitted inside each training fold only.
  (b) F6 matching diagnostics: how many GAP targets met the registered +-0.01 density match and
      exactly what the fallback (200 nearest-density held-out points) compared against.
  (c) Raw and momentum-conditional hole-growth associations per cell, with p-values and CIs.

The cell features are rebuilt with the exact RNG sequence of scripts/where_future_lands.py (same
seed, same order of draws) and checked against docs/prereg/where_future_lands_result.json.
Reuses saved embeddings only (no embedding model is loaded).

    OMP_NUM_THREADS=2 uv run python scripts/revision_future_cells.py [domain/embedder ...]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_future_lands as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/revision_future_cells.json"
OLD = json.loads((ROOT / "docs/prereg/where_future_lands_result.json").read_text())
CACHE = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/revision_future_cells"
SEED, K, N_BOOT = W.SEED, W.K_CELLS, W.N_BOOT
warnings.filterwarnings("ignore", message=".*least populated class.*")


def build_cells(domain: str, embedder: str) -> dict:
    """Replicates W.analyse up to the cell features, consuming the RNG in the same order."""
    cache = CACHE / f"{domain}_{embedder}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    rng = np.random.default_rng(SEED)
    root = W.DATA / "prepared" / domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date"])
    emb = np.load(root / f"emb_{embedder}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    pm = meta[past].reset_index(drop=True)
    held = np.zeros(len(P), bool)
    held[rng.choice(len(P), len(P) // 10, replace=False)] = True
    R, H = P[~held], P[held]
    rm = pm[~held].reset_index(drop=True)
    W.log(f"{domain}/{embedder}: reference {len(R)}, held-out {len(H)}, future {len(F)}")
    mu = R.mean(0)
    mu_hat = mu / np.linalg.norm(mu)
    nF = len(F)
    nH = len(H)
    for _ in range(N_BOOT):  # F1 bootstrap draws in the original, consumed here to keep the stream aligned
        rng.integers(0, nF, nF)
        rng.integers(0, nH, nH)
    km = KMeans(n_clusters=K, random_state=SEED, n_init=1).fit(R)
    r_cell = km.labels_
    f_cell = km.predict(F)
    past_n = np.bincount(r_cell, minlength=K)
    fut_n = np.bincount(f_cell, minlength=K)
    growth = (fut_n / len(F)) / (past_n / len(R))
    w, U = np.linalg.eigh(np.cov(R.T, bias=True))
    draw_n = np.zeros(K, np.int64)
    for i in range(0, W.N_DRAWS, 20_000):
        Z = rng.standard_normal((20_000, R.shape[1])).astype(np.float32) * np.sqrt(np.clip(w, 0, None)).astype(np.float32)
        C = mu + Z @ U.T.astype(np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        draw_n += np.bincount(km.predict(C), minlength=K)
    hole = (draw_n / W.N_DRAWS) / (past_n / len(R))
    lo, hi = W.MOMENTUM_YEARS[domain]
    ry = W.years(rm["date"], domain)
    recent = ((ry >= lo) & (ry <= hi)).astype(float)
    momentum = np.bincount(r_cell, weights=recent, minlength=K) / past_n
    cell_lat = np.bincount(r_cell, weights=R @ mu_hat, minlength=K) / past_n
    idx_ci = np.array([rng.integers(0, K, K) for _ in range(N_BOOT)])
    out = {"growth": growth, "hole": hole, "momentum": momentum, "cell_lat": cell_lat, "past_n": past_n,
           "fut_n": fut_n, "draw_n": draw_n, "idx_ci": idx_ci}
    if embedder == "qwen3e06":  # F6 inputs (no RNG is used by the fallback branch)
        _, h_dens = W.neighbours(H, R)
        with np.load(W.DATA / "runs" / domain / "pilot/targets.npz", allow_pickle=True) as zt:
            G = zt["GAP"].astype(np.float32)
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        _, g_dens = W.neighbours(G, R)
        out["g_dens"], out["h_dens"] = g_dens, h_dens
        out["g_cnt"] = W.count_above(G, F, W.kth_cos(G, R))
        out["h_cnt"] = W.count_above(H, F, W.kth_cos(H, R))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def zs(a, ref=None):
    ref = a if ref is None else ref
    return (a - ref.mean(0)) / ref.std(0)


def oof_auc(X, y, splits) -> float:
    """Out-of-fold AUC with standardisation fitted on the training rows of each fold."""
    pred = np.zeros(len(y))
    for tr, te in splits:
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return float(roc_auc_score(y, pred))


def f3_cluster(c: dict) -> dict:
    growth, hole, mom, lat, idx_ci = c["growth"], c["hole"], c["momentum"], c["cell_lat"], c["idx_ci"]
    top = (growth >= np.quantile(growth, 0.9)).astype(int)
    Xm, Xmhl = mom[:, None], np.column_stack([mom, hole, lat])
    # Registered point estimate (global z, StratifiedKFold) reproduced, then with in-fold z.
    old_m = W.cv_auc(zs(mom)[:, None], top)
    old_mhl = W.cv_auc(np.column_stack([zs(mom), zs(hole), zs(lat)]), top)
    sp = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xmhl, top))
    new_m, new_mhl = oof_auc(Xm, top, sp), oof_auc(Xmhl, top, sp)
    naive, clus, n_skip = [], [], 0
    Zm, Zmhl = zs(mom)[:, None], np.column_stack([zs(mom), zs(hole), zs(lat)])
    for i in idx_ci:
        if top[i].sum() < 5 or top[i].sum() > len(i) - 5:
            n_skip += 1
            continue
        naive.append(W.cv_auc(Zmhl[i], top[i]) - W.cv_auc(Zm[i], top[i]))  # the registered (flawed) loop
        spg = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xmhl[i], top[i], groups=i))
        clus.append(oof_auc(Xmhl[i], top[i], spg) - oof_auc(Xm[i], top[i], spg))
    naive, clus = np.array(naive), np.array(clus)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    p = lambda v: float(2 * min(np.mean(v <= 0), np.mean(v >= 0)))
    return {"n_positive": int(top.sum()), "n_boot_used": int(len(clus)), "n_boot_skipped": n_skip,
            "registered": {"auc_momentum": old_m, "auc_momentum_hole_lat": old_mhl, "gain": old_mhl - old_m,
                           "gain_ci_recomputed_naive": ci(naive), "p_gain_boot_naive": p(naive)},
            "infold_z": {"auc_momentum": new_m, "auc_momentum_hole_lat": new_mhl, "gain": new_mhl - new_m},
            "cluster_bootstrap": {"gain_ci": ci(clus), "p_gain_boot": p(clus),
                                  "gain_boot_median": float(np.median(clus)),
                                  "share_boot_gain_ge_003": float(np.mean(clus >= 0.03)),
                                  "upper_excludes_003": ci(clus)[1] < 0.03,
                                  "upper_excludes_005": ci(clus)[1] < 0.05,
                                  "upper_excludes_010": ci(clus)[1] < 0.10},
            "old_json": {k: OLD_KEY[k] for k in ("auc_momentum", "auc_momentum_hole_lat", "gain", "gain_ci", "p_gain_boot")}}


def f2_assoc(c: dict) -> dict:
    growth, hole, mom, past_n, idx_ci = c["growth"], c["hole"], c["momentum"], c["past_n"], c["idx_ci"]
    raw = stats.spearmanr(growth, hole)
    ctrl_m = mom[:, None]
    ctrl_ms = np.column_stack([mom, np.log(past_n)])
    pm_, pms = W.partial_spearman(growth, hole, ctrl_m), W.partial_spearman(growth, hole, ctrl_ms)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    b_raw = [stats.spearmanr(growth[i], hole[i])[0] for i in idx_ci]
    b_m = [W.partial_spearman(growth[i], hole[i], ctrl_m[i])[0] for i in idx_ci]
    b_ms = [W.partial_spearman(growth[i], hole[i], ctrl_ms[i])[0] for i in idx_ci]
    rho_hm = stats.spearmanr(hole, mom)
    return {"raw_rho_hole": float(raw[0]), "raw_p": float(raw[1]), "raw_ci": ci(b_raw),
            "partial_rho_given_momentum": float(pm_[0]), "partial_p_given_momentum": float(pm_[1]),
            "partial_ci_given_momentum": ci(b_m),
            "partial_rho_given_momentum_logsize": float(pms[0]), "partial_p_given_momentum_logsize": float(pms[1]),
            "partial_ci_given_momentum_logsize": ci(b_ms),
            "rho_hole_momentum": float(rho_hm[0]), "p_hole_momentum": float(rho_hm[1]),
            "raw_rho_momentum": float(stats.spearmanr(growth, mom)[0]),
            "old_json": {k: OLD_F2[k] for k in ("partial_rho", "p", "ci", "raw_rho_hole")}}


def f6_diag(c: dict) -> dict:
    g_dens, h_dens, g_cnt, h_cnt = c["g_dens"], c["h_dens"], c["g_cnt"], c["h_cnt"]
    pct, n_within, max_gap, mean_ctrl_dens, n_exact = [], [], [], [], 0
    for d, cnt in zip(g_dens, g_cnt):
        within = int((np.abs(h_dens - d) <= 0.01).sum())
        n_within.append(within)
        if within >= 200:
            n_exact += 1  # never reached; the registered branch would subsample with the RNG
        near = np.argsort(np.abs(h_dens - d))[:200]
        max_gap.append(float(np.abs(h_dens[near] - d).max()))
        mean_ctrl_dens.append(float(h_dens[near].mean()))
        pct.append(stats.percentileofscore(h_cnt[near], cnt, kind="mean"))
    pct = np.array(pct)
    above, below = int((pct > 50).sum()), int((pct < 50).sum())
    sign = stats.binomtest(above, above + below, 0.5)
    return {"n_targets": int(len(g_dens)), "matched_within_001": n_exact,
            "heldout_points_within_001_per_target": {"min": int(min(n_within)), "median": float(np.median(n_within)),
                                                     "max": int(max(n_within))},
            "fallback": "200 held-out past points with the smallest |density - target density| (np.argsort), no RNG",
            "fallback_max_abs_density_gap": {"median": float(np.median(max_gap)), "min": float(min(max_gap)),
                                             "max": float(max(max_gap))},
            "median_gap_density": float(np.median(g_dens)), "median_fallback_control_density": float(np.median(mean_ctrl_dens)),
            "median_heldout_density": float(np.median(h_dens)),
            "heldout_density_min": float(h_dens.min()),
            "share_gap_targets_below_heldout_min_density": float(np.mean(g_dens < h_dens.min())),
            "median_percentile": float(np.median(pct)), "percentile_iqr": [float(np.percentile(pct, 25)), float(np.percentile(pct, 75))],
            "n_pct_above_50": above, "n_pct_below_50": below, "sign_test_p": float(sign.pvalue),
            "median_gap_count": float(np.median(g_cnt)),
            "old_json": {k: OLD_F6[k] for k in ("matched_within_001", "median_percentile")},
            "percentiles_match_old": bool(np.allclose(pct, OLD_F6["percentiles"]))}


def main() -> None:
    global OLD_KEY, OLD_F2, OLD_F6
    which = sys.argv[1:] or [f"{d}/{e}" for e in W.EMBEDDERS for d in W.DOMAINS]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key in which:
        d, e = key.split("/")
        t0 = time.time()
        c = build_cells(d, e)
        OLD_KEY, OLD_F2, OLD_F6 = OLD[key]["F3"], OLD[key]["F2"], OLD[key]["F6"]
        r = {"F3": f3_cluster(c), "F2_assoc": f2_assoc(c)}
        r["reproduces_registered"] = bool(abs(r["F2_assoc"]["partial_rho_given_momentum_logsize"] - OLD_F2["partial_rho"]) < 1e-9
                                          and abs(r["F3"]["registered"]["gain"] - OLD_KEY["gain"]) < 1e-9)
        if e == "qwen3e06":
            r["F6"] = f6_diag(c)
        res[key] = r
        W.log(f"{key} done in {time.time() - t0:.0f}s; reproduces={r['reproduces_registered']}")
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
