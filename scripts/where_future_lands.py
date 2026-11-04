"""Where the future lands against the past map (docs/prereg/where_future_lands.md).

Local CPU only, no LLM calls. Every parameter is fixed by the pre-registration; run
`uv run python scripts/where_future_lands.py` and read docs/prereg/where_future_lands_result.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/where_future_lands_result.json"
SEED = 20261030
K_CELLS = 256
N_DRAWS = 200_000
N_BOOT = 2000
FAR_COS = 0.15
MOMENTUM_YEARS = {"steam": (2022, 2023), "arxiv": (2021, 2022), "patents": (2014, 2015)}
EMBEDDERS = ("qwen3e06", "bgem3")
DOMAINS = ("steam", "arxiv", "patents")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def neighbours(Q: np.ndarray, R: np.ndarray, k: int = 10, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Max cos and mean of the k largest cos of each row of Q against R (both L2-normalised)."""
    mx, mk = np.empty(len(Q), np.float32), np.empty(len(Q), np.float32)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ R.T
        top = np.partition(S, -k, axis=1)[:, -k:]
        mx[i:i + chunk], mk[i:i + chunk] = top.max(1), top.mean(1)
    return mx, mk


def kth_cos(Q: np.ndarray, R: np.ndarray, k: int = 10, chunk: int = 1000) -> np.ndarray:
    out = np.empty(len(Q), np.float32)
    for i in range(0, len(Q), chunk):
        out[i:i + chunk] = np.partition(Q[i:i + chunk] @ R.T, -k, axis=1)[:, -k]
    return out


def count_above(Q: np.ndarray, F: np.ndarray, tau: np.ndarray, chunk: int = 500) -> np.ndarray:
    out = np.empty(len(Q), np.int64)
    for i in range(0, len(Q), chunk):
        out[i:i + chunk] = ((Q[i:i + chunk] @ F.T) > tau[i:i + chunk, None]).sum(1)
    return out


def boot_ci(f, *arrays, rng: np.random.Generator) -> list[float]:
    n = len(arrays[0])
    vals = [f(*(a[idx] for a in arrays)) for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def residual_rank(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    ry = stats.rankdata(y)
    RX = np.column_stack([np.ones(len(y))] + [stats.rankdata(c) for c in X.T])
    return ry - RX @ np.linalg.lstsq(RX, ry, rcond=None)[0]


def partial_spearman(y, x, controls):
    return stats.pearsonr(residual_rank(y, controls), residual_rank(x, controls))


def cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def holm(ps: dict[str, float]) -> dict[str, float]:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, ps[k] * (len(keys) - i)))
        out[k] = run
    return out


def years(dates: pd.Series, domain: str) -> np.ndarray:
    if domain == "patents":
        return dates.astype(str).str[:4].astype(int).to_numpy()
    return pd.to_datetime(dates).dt.year.to_numpy()


def analyse(domain: str, embedder: str) -> dict:
    rng = np.random.default_rng(SEED)
    root = DATA / "prepared" / domain
    cols = ["split", "adult", "date"] + (["positive", "negative"] if domain == "steam" else []) + \
        (["decision"] if domain == "patents" else [])
    meta = pd.read_parquet(root / "corpus.parquet", columns=cols)
    emb = np.load(root / f"emb_{embedder}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    pm, fm = meta[past].reset_index(drop=True), meta[~past].reset_index(drop=True)

    held = np.zeros(len(P), bool)
    held[rng.choice(len(P), len(P) // 10, replace=False)] = True
    R, H = P[~held], P[held]
    rm = pm[~held].reset_index(drop=True)
    log(f"{domain}/{embedder}: reference {len(R)}, held-out {len(H)}, future {len(F)}")

    mu = R.mean(0)
    mu_hat = mu / np.linalg.norm(mu)
    f_max, f_dens = neighbours(F, R)
    h_max, h_dens = neighbours(H, R)
    f_iso, h_iso = 1 - f_max, 1 - h_max
    f_lat, h_lat = F @ mu_hat, H @ mu_hat
    log("  item measures done")

    # F1
    far_share = float((f_max <= FAR_COS).mean())
    med_diff = float(np.median(f_iso) - np.median(h_iso))
    f1 = {"far_share": far_share, "far_count": int((f_max <= FAR_COS).sum()),
          "median_iso_future": float(np.median(f_iso)), "median_iso_heldout": float(np.median(h_iso)),
          "median_iso_diff": med_diff,
          "median_iso_diff_ci": [float(np.percentile(v, q)) for v in [np.array([
              np.median(f_iso[rng.integers(0, len(f_iso), len(f_iso))]) -
              np.median(h_iso[rng.integers(0, len(h_iso), len(h_iso))]) for _ in range(N_BOOT)])] for q in (2.5, 97.5)],
          "max_cos_future_min": float(f_max.min()),
          "far_zero": far_share == 0.0, "far_near_zero": far_share <= 0.001, "diff_below_005": med_diff < 0.05}

    # Cells
    km = KMeans(n_clusters=K_CELLS, random_state=SEED, n_init=1).fit(R)
    r_cell = km.labels_
    f_cell = km.predict(F)
    past_n = np.bincount(r_cell, minlength=K_CELLS)
    fut_n = np.bincount(f_cell, minlength=K_CELLS)
    growth = (fut_n / len(F)) / (past_n / len(R))
    w, U = np.linalg.eigh(np.cov(R.T, bias=True))
    draw_n = np.zeros(K_CELLS, np.int64)
    for i in range(0, N_DRAWS, 20_000):
        Z = rng.standard_normal((20_000, R.shape[1])).astype(np.float32) * np.sqrt(np.clip(w, 0, None)).astype(np.float32)
        C = mu + Z @ U.T.astype(np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        draw_n += np.bincount(km.predict(C), minlength=K_CELLS)
    hole = (draw_n / N_DRAWS) / (past_n / len(R))
    lo, hi = MOMENTUM_YEARS[domain]
    ry = years(rm["date"], domain)
    recent = ((ry >= lo) & (ry <= hi)).astype(float)
    momentum = np.bincount(r_cell, weights=recent, minlength=K_CELLS) / past_n
    _, r_top11 = neighbours(R, R, k=11)  # the 11 largest include the item itself (cos 1)
    r_dens = (11 * r_top11 - 1) / 10
    cell_dens = np.bincount(r_cell, weights=r_dens, minlength=K_CELLS) / past_n
    cell_lat = np.bincount(r_cell, weights=R @ mu_hat, minlength=K_CELLS) / past_n
    log("  cells done")

    # F2
    ctrl = np.column_stack([momentum, np.log(past_n)])
    rho, p2 = partial_spearman(growth, hole, ctrl)
    idx_ci = [rng.integers(0, K_CELLS, K_CELLS) for _ in range(N_BOOT)]
    boots = [partial_spearman(growth[i], hole[i], ctrl[i])[0] for i in idx_ci]
    ci2 = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    f2 = {"partial_rho": float(rho), "p": float(p2), "ci": ci2,
          "raw_rho_hole": float(stats.spearmanr(growth, hole)[0]),
          "raw_rho_momentum": float(stats.spearmanr(growth, momentum)[0]),
          "raw_rho_density": float(stats.spearmanr(growth, cell_dens)[0]),
          "raw_rho_lat": float(stats.spearmanr(growth, cell_lat)[0]),
          "prediction_holds": abs(rho) < 0.1 or (ci2[0] <= 0 <= ci2[1])}

    # F3
    top = (growth >= np.quantile(growth, 0.9)).astype(int)
    z = lambda a: (a - a.mean()) / a.std()
    auc_m = cv_auc(z(momentum)[:, None], top)
    auc_h = cv_auc(z(hole)[:, None], top)
    auc_mhl = cv_auc(np.column_stack([z(momentum), z(hole), z(cell_lat)]), top)
    gains = []
    for i in idx_ci:
        if top[i].sum() < 5 or top[i].sum() > len(i) - 5:
            continue
        gains.append(cv_auc(np.column_stack([z(momentum), z(hole), z(cell_lat)])[i], top[i]) -
                     cv_auc(z(momentum)[i][:, None], top[i]))
    p3 = float(2 * min(np.mean(np.array(gains) <= 0), np.mean(np.array(gains) >= 0)))
    f3 = {"auc_momentum": auc_m, "auc_hole": auc_h, "auc_momentum_hole_lat": auc_mhl,
          "gain": auc_mhl - auc_m, "gain_ci": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
          "p_gain_boot": p3, "momentum_ge_hole": auc_m >= auc_h, "gain_below_003": (auc_mhl - auc_m) < 0.03}

    # F4
    u = stats.mannwhitneyu(f_lat, h_lat)
    f4 = {"median_lat_future": float(np.median(f_lat)), "median_lat_heldout": float(np.median(h_lat)),
          "rank_biserial": float(2 * u.statistic / (len(f_lat) * len(h_lat)) - 1), "p": float(u.pvalue),
          "direction": "edge" if np.median(f_lat) < np.median(h_lat) else "centre"}

    # F5
    f5: dict = {}
    if domain == "steam":
        n_rev = (fm["positive"] + fm["negative"]).to_numpy()
        m = n_rev >= 20
        share = fm["positive"].to_numpy()[m] / n_rev[m]
        q = f_iso[m]
    elif domain == "patents":
        m = fm["decision"].isin(["ACCEPTED", "REJECTED"]).to_numpy()
        share = (fm["decision"].to_numpy()[m] == "ACCEPTED").astype(float)
        q = f_iso[m]
    if domain in ("steam", "patents"):
        r5, p5 = stats.spearmanr(q, share)
        cut = np.quantile(q, 0.99)
        f5 = {"n": int(m.sum()), "rho": float(r5), "p": float(p5),
              "ci": boot_ci(lambda a, b: stats.spearmanr(a, b)[0], q, share, rng=rng),
              "top1_mean_quality": float(share[q >= cut].mean()), "rest_mean_quality": float(share[q < cut].mean()),
              "prediction_holds": r5 <= 0}

    # F6 (GAP targets were built in the Qwen space only)
    f6 = None
    if embedder == "qwen3e06":
        with np.load(DATA / "runs" / domain / "pilot/targets.npz", allow_pickle=True) as zt:
            G = zt["GAP"].astype(np.float32)
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        _, g_dens = neighbours(G, R)
        g_tau = kth_cos(G, R)
        g_cnt = count_above(G, F, g_tau)
        h_tau = kth_cos(H, R)
        h_cnt = count_above(H, F, h_tau)
        pct, n_exact = [], 0
        for d, c in zip(g_dens, g_cnt):
            near = np.where(np.abs(h_dens - d) <= 0.01)[0]
            if len(near) >= 200:
                n_exact += 1
                near = rng.choice(near, 200, replace=False)
            else:
                near = np.argsort(np.abs(h_dens - d))[:200]
            pct.append(stats.percentileofscore(h_cnt[near], c, kind="mean"))
        f6 = {"n_targets": len(G), "matched_within_001": n_exact, "median_percentile": float(np.median(pct)),
              "percentiles": [float(x) for x in pct], "median_gap_count": float(np.median(g_cnt)),
              "median_gap_density": float(np.median(g_dens)), "median_heldout_density": float(np.median(h_dens)),
              "prediction_holds": float(np.median(pct)) <= 60}

    ps = {"F2": f2["p"], "F3": f3["p_gain_boot"], "F4": f4["p"]}
    if f5:
        ps["F5"] = f5["p"]
    return {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5 or None, "F6": f6, "holm": holm(ps),
            "sizes": {"reference": len(R), "heldout": len(H), "future": len(F)}}


def main() -> None:
    which = sys.argv[1:] or [f"{d}/{e}" for e in EMBEDDERS for d in DOMAINS]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key in which:
        d, e = key.split("/")
        t0 = time.time()
        res[key] = analyse(d, e)
        log(f"{key} done in {time.time() - t0:.0f}s")
        OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == "__main__":
    main()
