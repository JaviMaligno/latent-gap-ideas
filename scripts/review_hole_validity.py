"""Is the hole score a gap score, and does Section 4 depend on it? (docs/prereg/review_hole_validity.md)

Local CPU only: saved embeddings, no embedding model, no LLM, no gateway, no GPU. Two stages, both resumable:

  features  per corpus x embedder: densities per space, k-means / growth / momentum / H0-H3 / covariates per
            (space, K), the planted-hole control (raw, K = 256). One .npz per checkpoint under
            D/runs/review_hole_validity/; an existing checkpoint is loaded, not recomputed.
  tests     per corpus x embedder: V1 per (space, K, definition), V2 AUCs, T1-T3 per configuration, T4, T5.
            One .json per configuration under D/runs/review_hole_validity/tests/<domain>_<embedder>/ (momentum
            arms cached once per outcome); then the result JSON is assembled with the decisions block.
  collect   only reassembles docs/prereg/review_hole_validity.json from the cached fragments.

    OMP_NUM_THREADS=4 uv run python scripts/review_hole_validity.py --stage features [domain/embedder ...]
    OMP_NUM_THREADS=1 uv run python scripts/review_hole_validity.py --stage tests --workers 3 [domain/embedder ...]

Smoke test (subsampled, tiny K, few draws/resamples, writes under --out-dir only; never the registered paths):

    uv run python scripts/review_hole_validity.py --smoke --out-dir /tmp/x --stage all steam/qwen3e06
    uv run python scripts/review_hole_validity.py --repro-only steam/qwen3e06   # full-data raw K=256 check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_future_lands as W  # noqa: E402
import revision_future_cells as RFC  # noqa: E402

warnings.filterwarnings("ignore", message=".*least populated class.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # degenerate rank correlations (H2 vs density) -> NaN

ROOT = Path(__file__).resolve().parents[1]
D = W.DATA
REG_CACHE = D / "runs" / "revision_future_cells"
OLD = json.loads((ROOT / "docs/prereg/where_future_lands_result.json").read_text())
SEED = W.SEED
SPACES = ("raw", "centred", "whitened")
SPACE_INDEX = {"raw": 0, "centred": 1, "whitened": 2}
DEFS = ("H0", "H1", "H2", "H3")
K_NN = 10
R_PCA = 10
WHITEN_DIM = 256
TAU_UP = (75, 90, 95)
TAU_DOWN = (25, 10, 5)
FUTURE_YEARS = {"steam": {"h12": (2024,), "h24": (2024, 2025)},
                "arxiv": {"h12": (2023,), "h24": (2023, 2024)},
                "patents": {"h12": (2016,), "h24": (2016, 2017)}}


class Cfg:
    """Registered constants; --smoke shrinks them (never used for the registered outputs)."""
    n_draws = W.N_DRAWS          # 200,000
    B = W.N_BOOT                 # 2000
    m_probe = 500
    ks = (128, 256, 512)
    kbase = 256
    n_planted = 52
    planted_min = 40
    min_size = 100
    sub = None                   # (n_ref, n_held, n_fut) in smoke
    out = D / "runs" / "review_hole_validity"
    result = ROOT / "docs/prereg/review_hole_validity.json"
    smoke = False


C = Cfg()


def log(msg: str) -> None:
    W.log(msg)


def l2(X: np.ndarray) -> np.ndarray:
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ----------------------------------------------------------------------------------------------- data
def load(domain: str, embedder: str):
    """Reference / held-out / future exactly as W.analyse and RFC.build_cells, plus the registered RNG
    stream positioned right before the Gaussian draws (held-out choice and F1 bootstrap draws consumed)."""
    rng = np.random.default_rng(SEED)
    root = D / "prepared" / domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date"])
    keep = np.flatnonzero(~meta["adult"].to_numpy())
    meta = meta.iloc[keep].reset_index(drop=True)
    past = (meta["split"] == "past").to_numpy()
    p_rows, f_rows = keep[past], keep[~past]
    pm, fm = meta[past].reset_index(drop=True), meta[~past].reset_index(drop=True)
    held = np.zeros(len(p_rows), bool)
    held[rng.choice(len(p_rows), len(p_rows) // 10, replace=False)] = True
    nF, nH = len(f_rows), int(held.sum())
    for _ in range(W.N_BOOT):  # F1 bootstrap draws of the registered script, consumed to keep the stream aligned
        rng.integers(0, nF, nF)
        rng.integers(0, nH, nH)
    r_rows, h_rows = p_rows[~held], p_rows[held]
    ry = W.years(pm[~held].reset_index(drop=True)["date"], domain)
    fy = W.years(fm["date"], domain)
    f_sel = np.arange(nF)
    if C.sub:  # smoke only: deterministic subsample, the registered stream is irrelevant then
        s = np.random.default_rng(SEED + 999)
        ri = np.sort(s.choice(len(r_rows), C.sub[0], replace=False))
        hi = np.sort(s.choice(len(h_rows), C.sub[1], replace=False))
        f_sel = np.sort(s.choice(nF, C.sub[2], replace=False))
        r_rows, h_rows, ry = r_rows[ri], h_rows[hi], ry[ri]
    e16 = np.load(root / f"emb_{embedder}.npy", mmap_mode="r")
    R = l2(np.asarray(e16[r_rows], dtype=np.float32))
    H = l2(np.asarray(e16[h_rows], dtype=np.float32))
    F = l2(np.asarray(e16[f_rows[f_sel]], dtype=np.float32))
    del e16
    log(f"{domain}/{embedder}: reference {len(R)}, held-out {len(H)}, future {len(F)}")
    return R, H, F, ry, fy[f_sel], rng


def make_space(space: str, R: np.ndarray):
    """Map fitted on the reference set only. Returns f(X) -> transformed, L2-normalised copy."""
    if space == "raw":
        return lambda X: X
    mu = R.mean(0)
    cen = lambda X: l2(X - mu)
    if space == "centred":
        return cen
    Rc = cen(R)
    m2 = Rc.mean(0)
    w, U = np.linalg.eigh(np.cov((Rc - m2).T, bias=True))
    del Rc
    Ut, st = U[:, -WHITEN_DIM:].astype(np.float32), np.sqrt(w[-WHITEN_DIM:]).astype(np.float32)
    return lambda X: l2(((cen(X) - m2) @ Ut) / st)


def maxcos(Q: np.ndarray, R: np.ndarray, drop: np.ndarray | None = None, chunk: int = 1000) -> np.ndarray:
    out = np.empty(len(Q), np.float32)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ R.T
        if drop is not None:
            S[:, drop] = -np.inf
        out[i:i + chunk] = S.max(1)
    return out


def topk_mean(Q: np.ndarray, R: np.ndarray, k: int, drop: np.ndarray | None = None, chunk: int = 1000) -> np.ndarray:
    out = np.empty(len(Q), np.float32)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ R.T
        if drop is not None:
            S[:, drop] = -np.inf
        out[i:i + chunk] = np.partition(S, -k, axis=1)[:, -k:].mean(1)
    return out


def nearest(X: np.ndarray, cent: np.ndarray, cn2: np.ndarray) -> np.ndarray:
    """Nearest centroid by Euclidean distance (= KMeans.predict), cn2 = squared centroid norms."""
    return np.argmin(cn2[None, :] - 2 * (X @ cent.T), axis=1)


def probes_for(X: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """H3 probe recipe for one cell: m_j + rho * u/|u|, u in the span of the top r principal directions."""
    n = len(X)
    mj = X.mean(0)
    off = np.linalg.norm(X - mj, axis=1)
    r = min(R_PCA, n - 1)
    if r < 1:
        return l2(np.repeat(mj[None, :], m, 0).astype(np.float32))
    _, s, Vt = np.linalg.svd(X - mj, full_matrices=False)
    lam = (s[:r] ** 2) / n
    z = rng.standard_normal((m, r))
    u = (z * np.sqrt(lam)) @ Vt[:r]
    rho = rng.choice(off, m, replace=True)
    p = mj + rho[:, None] * u / np.linalg.norm(u, axis=1, keepdims=True)
    return l2(p.astype(np.float32))


def h3_from(pmax: np.ndarray, kept: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    below = pmax < tau
    nk = kept.sum(1)
    h3 = np.where(nk > 0, (below & kept).sum(1) / np.maximum(nk, 1), below.mean(1))
    return h3, int((nk == 0).sum())


def choose_tau(pmax, kept, h_max):
    ladder = [50]
    h3, n_empty = h3_from(pmax, kept, np.percentile(h_max, 50))
    tried = [{"pct": 50, "share_zero": float(np.mean(h3 == 0)), "share_one": float(np.mean(h3 == 1))}]
    if np.mean(h3 == 0) > 0.8:
        ladder = list(TAU_UP)
    elif np.mean(h3 == 1) > 0.8:
        ladder = list(TAU_DOWN)
    else:
        ladder = []
    pct = 50
    for q in ladder:
        pct = q
        h3, n_empty = h3_from(pmax, kept, np.percentile(h_max, q))
        tried.append({"pct": q, "share_zero": float(np.mean(h3 == 0)), "share_one": float(np.mean(h3 == 1))})
        if np.mean(h3 == 0) <= 0.8 and np.mean(h3 == 1) <= 0.8:
            break
    return h3, float(np.percentile(h_max, pct)), pct, tried, n_empty


# ----------------------------------------------------------------------------------------------- features
def fpath(d, e, space, k):
    return C.out / f"{d}_{e}_{space}_K{k}.npz"


def save(path: Path, **arrs) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def reproduction(d, e, k, growth, momentum, hole, past_n) -> dict:
    with np.load(REG_CACHE / f"{d}_{e}.npz") as z:
        g0, m0, h0 = z["growth"], z["momentum"], z["hole"]
    ctrl = np.column_stack([momentum, np.log(past_n)])
    rho = float(W.partial_spearman(growth, hole, ctrl)[0])
    reg = OLD[f"{d}/{e}"]["F2"]["partial_rho"]
    out = {"max_abs_d_growth": float(np.abs(growth - g0).max()), "max_abs_d_momentum": float(np.abs(momentum - m0).max()),
           "max_abs_d_hole": float(np.abs(hole - h0).max()), "partial_rho": rho, "partial_rho_registered": reg}
    out["passes_cache"] = bool(out["max_abs_d_growth"] < 1e-6 and out["max_abs_d_momentum"] < 1e-6
                               and out["max_abs_d_hole"] < 5e-3 and abs(rho - reg) < 0.005)
    # Deviation (2026-11-14): KMeans is not bit-reproducible across runs on this machine. For arxiv/qwen3e06 two fresh
    # runs gave two solutions: one matches the cached registered features exactly, the other reproduces the published
    # partial rho exactly (cache and published number come from different registered runs). Either counts as a
    # faithful reproduction; both comparisons are kept. Each corpus x embedder also runs in its own process.
    out["passes_registered_rho"] = bool(abs(rho - reg) < 1e-6)
    out["passes"] = out["passes_cache"] or out["passes_registered_rho"]
    return out


def features(d: str, e: str, repro_only: bool = False) -> None:
    C.out.mkdir(parents=True, exist_ok=True)
    spaces_todo = [s for s in SPACES if any(not fpath(d, e, s, k).exists() for k in space_ks(s))]
    planted_done = (C.out / f"{d}_{e}_planted.npz").exists()
    if repro_only:
        spaces_todo, planted_done = ["raw"], True
    if not spaces_todo and planted_done:
        log(f"{d}/{e}: features already cached")
        return
    t0 = time.time()
    R, H, F, ry, fy, rng_reg = load(d, e)
    mu_hat_raw = R.mean(0)
    mu_hat_raw /= np.linalg.norm(mu_hat_raw)
    lat_item = R @ mu_hat_raw  # raw-space latitude of every reference item (F3 uses raw latitude in all spaces)
    lo, hi = W.MOMENTUM_YEARS[d]
    recent = ((ry >= lo) & (ry <= hi)).astype(float)
    if not planted_done and "raw" not in spaces_todo:
        spaces_todo = ["raw"] + spaces_todo  # planted needs the raw K-base objects; cached files are reloaded
    for space in spaces_todo:
        f = make_space(space, R)
        Rs, Hs, Fs = (R, H, F) if space == "raw" else (f(R), f(H), f(F))
        ks = [C.kbase] if repro_only else space_ks(space)
        todo = [k for k in ks if repro_only or not fpath(d, e, space, k).exists()]
        # Fit the registered K first: in one process an earlier KMeans fit changes a later one's result (checked
        # 2026-11-14: K=256 after K=128 differs from K=256 alone), and the registered K=256 was fitted fresh.
        todo.sort(key=lambda k: k != C.kbase)
        kms = {}
        for k in todo:
            t1 = time.time()
            kms[k] = KMeans(n_clusters=k, random_state=SEED, n_init=1).fit(Rs)
            log(f"  {space} K={k}: k-means {time.time() - t1:.0f}s")
        draw_n = {k: np.zeros(k, np.int64) for k in todo}
        if todo:
            rng = rng_reg if space == "raw" else np.random.default_rng(SEED + SPACE_INDEX[space])
            mu = Rs.mean(0)
            w, U = np.linalg.eigh(np.cov(Rs.T, bias=True))
            sw, Ut = np.sqrt(np.clip(w, 0, None)).astype(np.float32), U.T.astype(np.float32)
            t1 = time.time()
            for _ in range(0, C.n_draws, 20_000):  # common random numbers across K within a space
                Z = rng.standard_normal((20_000, Rs.shape[1])).astype(np.float32) * sw
                Cd = mu + Z @ Ut
                Cd /= np.linalg.norm(Cd, axis=1, keepdims=True)
                for k in todo:
                    draw_n[k] += np.bincount(kms[k].predict(Cd), minlength=k)
            del Z, Cd
            log(f"  {space}: {C.n_draws} draws {time.time() - t1:.0f}s")
        if repro_only or (space == "raw" and C.kbase in todo and not C.smoke):
            km = kms[C.kbase]
            pn = np.bincount(km.labels_, minlength=C.kbase)
            gr = (np.bincount(km.predict(F), minlength=C.kbase) / len(F)) / (pn / len(R))
            mo = np.bincount(km.labels_, weights=recent, minlength=C.kbase) / pn
            ho = (draw_n[C.kbase] / C.n_draws) / (pn / len(R))
            rep = reproduction(d, e, C.kbase, gr, mo, ho, pn)
            (C.out / f"{d}_{e}_reproduction.json").write_text(json.dumps(rep, indent=1))
            log(f"  reproduction check: {rep}")
            if not rep["passes"]:
                raise SystemExit(f"{d}/{e}: reproduction of the registered raw K=256 features failed; stopping")
            if repro_only:
                return
        # densities (once per space)
        dpath = C.out / f"{d}_{e}_{space}_dens.npz"
        if dpath.exists():
            with np.load(dpath) as z:
                r_dens, h_max = z["r_dens"], z["h_max"]
        else:
            t1 = time.time()
            _, top11 = W.neighbours(Rs, Rs, k=K_NN + 1)
            r_dens = (11 * top11 - 1) / 10
            h_max = maxcos(Hs, Rs)
            save(dpath, r_dens=r_dens, h_max=h_max)
            log(f"  {space}: densities {time.time() - t1:.0f}s")
        mu_s = Rs.mean(0)
        mu_s /= np.linalg.norm(mu_s)
        for k in todo:
            t1 = time.time()
            km = kms[k]
            lab = km.labels_.astype(np.int32)
            f_cell = km.predict(Fs).astype(np.int32)
            cent = km.cluster_centers_.astype(np.float32)
            cn = cent / np.linalg.norm(cent, axis=1, keepdims=True)
            past_n = np.bincount(lab, minlength=k)
            fut_n = np.bincount(f_cell, minlength=k)
            growth = (fut_n / len(Fs)) / (past_n / len(Rs))
            momentum = np.bincount(lab, weights=recent, minlength=k) / past_n
            cell_lat_raw = np.bincount(lab, weights=lat_item, minlength=k) / past_n
            H0 = (draw_n[k] / C.n_draws) / (past_n / len(Rs))
            H1 = -topk_mean(cn, Rs, K_NN)
            cov_a = np.bincount(lab, weights=r_dens, minlength=k) / past_n
            H2 = -cov_a
            cov_b = cn @ mu_s
            cov_c = np.log(past_n)
            cov_d = np.bincount(lab, weights=1 - np.einsum("ij,ij->i", Rs, cn[lab]), minlength=k) / past_n
            prng = np.random.default_rng(SEED + 100 + k + 1000 * SPACE_INDEX[space])
            cn2 = (cent ** 2).sum(1)
            pmax = np.empty((k, C.m_probe), np.float32)
            kept = np.zeros((k, C.m_probe), bool)
            for j in range(k):
                p = probes_for(Rs[lab == j], C.m_probe, prng)
                kept[j] = nearest(p, cent, cn2) == j
                pmax[j] = maxcos(p, Rs)
            H3, tau, pct, tried, n_empty = choose_tau(pmax, kept, h_max)
            save(fpath(d, e, space, k), r_cell=lab, f_cell=f_cell, f_year=fy, centroids=cent, past_n=past_n,
                 fut_n=fut_n, growth=growth, momentum=momentum, cell_lat_raw=cell_lat_raw, draw_n=draw_n[k],
                 H0=H0, H1=H1, H2=H2, H3=H3, cov_a=cov_a, cov_b=cov_b, cov_c=cov_c, cov_d=cov_d,
                 probe_maxcos=pmax, probe_kept=kept, tau=tau, tau_pct=pct, tau_tried=json.dumps(tried),
                 n_cells_all_probes_dropped=n_empty, probe_drop_total=int((~kept).sum()), n_ref=len(Rs),
                 n_fut=len(Fs))
            log(f"  {space} K={k}: features {time.time() - t1:.0f}s (tau p{pct}, dropped probes "
                f"{int((~kept).sum())}/{kept.size}, cells with all dropped {n_empty})")
        if space == "raw" and not planted_done:
            planted(d, e, Rs, r_dens)
        if space != "raw":
            del Rs, Hs, Fs
        del kms
    log(f"{d}/{e}: features done in {time.time() - t0:.0f}s")


def space_ks(space):
    return C.ks if space == "raw" else (C.kbase,)


def planted(d, e, R, r_dens) -> None:
    """V2: ball depletion vs random thinning of half a cell's items, one cell at a time, centroids/draws fixed."""
    t0 = time.time()
    with np.load(fpath(d, e, "raw", C.kbase)) as z:
        f = {k: z[k] for k in ("r_cell", "centroids", "past_n", "draw_n", "H0", "H1", "H2", "H3", "tau")}
    lab, cent, past_n = f["r_cell"], f["centroids"], f["past_n"]
    cn = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    cn2 = (cent ** 2).sum(1)
    nR, tau = len(R), float(f["tau"])
    rng = np.random.default_rng(SEED + 7)
    elig = np.flatnonzero(past_n >= C.planted_min)
    n_sel = min(C.n_planted, len(elig))
    sel = rng.choice(elig, n_sel, replace=False)
    arm = np.array([1] * (n_sel // 2) + [0] * (n_sel - n_sel // 2))  # 1 = ball, 0 = thinning
    before, after = np.zeros((n_sel, 4)), np.zeros((n_sel, 4))
    n_drop_probe = np.zeros(n_sel, int)
    for t, j in enumerate(sel):
        idx = np.flatnonzero(lab == j)
        n, dn = len(idx), len(idx) // 2
        if arm[t] == 1:
            pick = rng.choice(idx)
            sims = R[idx] @ R[pick]
            sims[idx == pick] = np.inf  # the picked item is always deleted
            delete = idx[np.argsort(-sims, kind="stable")[:dn]]
        else:
            delete = rng.choice(idx, dn, replace=False)
        rem = np.setdiff1d(idx, delete)
        before[t] = [f["H0"][j], f["H1"][j], f["H2"][j], f["H3"][j]]
        h0 = (f["draw_n"][j] / C.n_draws) / ((n - dn) / (nR - dn))
        h1 = -topk_mean(cn[j:j + 1], R, K_NN, drop=delete)[0]
        top11 = topk_mean(R[rem], R, K_NN + 1, drop=delete)
        h2 = -float(np.mean((11 * top11 - 1) / 10))
        p = probes_for(R[rem], C.m_probe, rng)
        kept = nearest(p, cent, cn2) == j
        pm = maxcos(p, R, drop=delete)
        n_drop_probe[t] = int((~kept).sum())
        h3 = float(np.mean(pm[kept] < tau)) if kept.any() else float(np.mean(pm < tau))
        after[t] = [h0, h1, h2, h3]
    save(C.out / f"{d}_{e}_planted.npz", cells=sel, arm=arm, before=before, after=after, n_drop_probe=n_drop_probe)
    log(f"  planted control: {n_sel} cells in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------- tests
def tdir(d, e) -> Path:
    p = C.out / "tests" / f"{d}_{e}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ci(v) -> list[float]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def pboot(v, null=0.0) -> float:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(2 * min(np.mean(v <= null), np.mean(v >= null))) if len(v) else float("nan")


def load_feat(d, e, space, k) -> dict:
    with np.load(fpath(d, e, space, k)) as z:
        return {key: z[key] for key in z.files}


def boot_idx(d, e, k, n=None, seed=None) -> np.ndarray:
    if seed is not None:
        return np.random.default_rng(seed).integers(0, n, (C.B, n))
    if k == 256 and not C.smoke:
        with np.load(REG_CACHE / f"{d}_{e}.npz") as z:
            return z["idx_ci"]
    return np.random.default_rng(SEED + k).integers(0, k, (C.B, k))


def outcomes(d, e) -> dict:
    """outcome key -> (space, K, cell mask, growth on kept cells, resample indices over kept cells)."""
    out = {}
    for space in SPACES:
        for k in space_ks(space):
            f = load_feat(d, e, space, k)
            out[f"{space}_K{k}"] = (space, k, np.ones(k, bool), f["growth"], boot_idx(d, e, k))
    f = load_feat(d, e, "raw", C.kbase)
    k = C.kbase
    keep = f["past_n"] >= C.min_size
    out["min_size"] = ("raw", k, keep, f["growth"][keep], boot_idx(d, e, k, int(keep.sum()), SEED + 1000))
    E = f["past_n"] * f["n_fut"] / f["n_ref"]
    v = max(np.var(f["fut_n"] / E) - np.mean(1 / E), 1e-6)
    a = 1 / v
    out["shrunk"] = ("raw", k, np.ones(k, bool), (f["fut_n"] + a) / (E + a), boot_idx(d, e, k))
    for h in ("h12", "h24"):
        m = np.isin(f["f_year"], FUTURE_YEARS[d][h])
        g = (np.bincount(f["f_cell"][m], minlength=k) / m.sum()) / (f["past_n"] / f["n_ref"])
        out[f"horizon_{h[1:]}m"] = ("raw", k, np.ones(k, bool), g, boot_idx(d, e, k))
    return out


def configs(d, e) -> list[tuple[str, str]]:
    cfg = [(f"{s}_K{k}", h) for s in SPACES for k in space_ks(s) for h in DEFS]
    cfg += [(o, h) for o in ("min_size", "shrunk", "horizon_12m", "horizon_24m") for h in DEFS]
    return cfg


def momentum_task(args) -> str:
    d, e, okey = args
    path = tdir(d, e) / f"mom_{okey}.npz"
    if path.exists():
        return okey
    space, k, keep, growth, idx = outcomes(d, e)[okey]
    f = load_feat(d, e, space, k)
    mom = f["momentum"][keep]
    top = (growth >= np.quantile(growth, 0.9)).astype(int)
    arm = np.full(len(idx), np.nan)
    direct = np.full(len(idx), np.nan)
    for b, i in enumerate(idx):
        if top[i].sum() < 5 or top[i].sum() > len(i) - 5:
            continue
        spg = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(mom[i][:, None], top[i], groups=i))
        arm[b] = RFC.oof_auc(mom[i][:, None], top[i], spg)
        direct[b] = roc_auc_score(top[i], mom[i])
    save(path, arm=arm, direct=direct, cv=W.cv_auc(RFC.zs(mom)[:, None], top), auc_direct=roc_auc_score(top, mom),
         n_pos=int(top.sum()))
    return okey


def test_task(args) -> str:
    d, e, okey, h = args
    path = tdir(d, e) / f"T_{okey}_{h}.json"
    if path.exists():
        return path.name
    space, k, keep, growth, idx = outcomes(d, e)[okey]
    f = load_feat(d, e, space, k)
    mom, lat, hole, past_n = f["momentum"][keep], f["cell_lat_raw"][keep], f[h][keep], f["past_n"][keep]
    top = (growth >= np.quantile(growth, 0.9)).astype(int)
    with np.load(tdir(d, e) / f"mom_{okey}.npz") as z:
        arm = z["arm"]
    z = RFC.zs
    auc_m = W.cv_auc(z(mom)[:, None], top)
    auc_h = W.cv_auc(z(hole)[:, None], top)
    auc_mhl = W.cv_auc(np.column_stack([z(mom), z(hole), z(lat)]), top)
    auc_mh = W.cv_auc(np.column_stack([z(mom), z(hole)]), top)
    Xmhl = np.column_stack([mom, hole, lat])
    ctrl = np.column_stack([mom, np.log(past_n)])
    t1b, gains, t3b, n_skip = [], [], [], 0
    for b, i in enumerate(idx):
        yi = top[i]
        t1b.append(roc_auc_score(yi, hole[i]) if 0 < yi.sum() < len(i) else np.nan)
        t3b.append(W.partial_spearman(growth[i], hole[i], ctrl[i])[0])
        if yi.sum() < 5 or yi.sum() > len(i) - 5:
            n_skip += 1
            continue
        spg = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xmhl[i], yi, groups=i))
        gains.append(RFC.oof_auc(Xmhl[i], yi, spg) - arm[b])
    rho3, p3 = W.partial_spearman(growth, hole, ctrl)
    auc1 = float(roc_auc_score(top, hole))
    res = {"outcome": okey, "space": space, "K": int(k), "definition": h, "n_cells": int(keep.sum()),
           "n_positive": int(top.sum()),
           "T1": {"auc_direct": auc1, "ci": ci(t1b), "p_boot": pboot(t1b, 0.5)},
           "T2": {"auc_momentum": auc_m, "auc_hole": auc_h, "auc_momentum_hole_lat": auc_mhl,
                  "gain": auc_mhl - auc_m, "momentum_ge_hole": bool(auc_m >= auc_h),
                  "gain_cluster_ci": ci(gains), "p_gain_boot": pboot(gains),
                  "n_boot_used": len(gains), "n_boot_skipped": n_skip,
                  "secondary_gain_momentum_hole_no_lat": auc_mh - auc_m},
           "T3": {"partial_rho": float(rho3), "p": float(p3), "ci": ci(t3b)}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, path)
    return path.name


def v1(d, e, space, k, h) -> dict:
    f = load_feat(d, e, space, k)
    hole = f[h]
    cov = {"a_density": f["cov_a"], "b_cos_mean": f["cov_b"], "c_log_size": f["cov_c"], "d_extent": f["cov_d"]}
    G = np.column_stack([f["cov_b"], f["cov_c"], f["cov_d"]])

    def stat(ix):
        out = [stats.spearmanr(hole[ix], cov[c][ix]) for c in cov]
        ry = stats.rankdata(hole[ix])
        X = np.column_stack([np.ones(len(ix))] + [stats.rankdata(c) for c in G[ix].T])
        resid = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
        r2 = 1 - resid.var() / ry.var()
        part = W.partial_spearman(hole[ix], f["cov_a"][ix], G[ix])
        return out, r2, part

    full = np.arange(k)
    rhos, r2, part = stat(full)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(C.B):
        ix = rng.integers(0, k, k)
        o, r, p = stat(ix)
        boots.append([x[0] for x in o] + [r, p[0]])
    boots = np.array(boots, float)
    res = {}
    for n, (c, (r, p)) in enumerate(zip(cov, rhos)):
        res[c] = {"rho": float(r), "p": float(p), "ci": ci(boots[:, n])}
    res["r2_geom"] = {"value": float(r2), "ci": ci(boots[:, 4])}
    res["partial_a_given_bcd"] = {"rho": float(part[0]), "p": float(part[1]), "ci": ci(boots[:, 5])}
    ps = {c: res[c]["p"] for c in cov} | {"partial_a_given_bcd": res["partial_a_given_bcd"]["p"]}
    res["holm"] = W.holm({kk: (1.0 if np.isnan(v) else v) for kk, v in ps.items()})
    if h == "H2":
        res["note"] = "H2 = -cell mean density, so rho with (a) is -1 by construction"
    if space == "centred":
        res["note_b"] = "centred space: mean direction near-degenerate, (b) flagged"
    return res


def planted_auc(d, e) -> dict:
    with np.load(C.out / f"{d}_{e}_planted.npz") as z:
        arm, before, after, ndp = z["arm"], z["before"], z["after"], z["n_drop_probe"]
    delta = after - before
    rng = np.random.default_rng(SEED)
    n = len(arm)
    idx = [rng.integers(0, n, n) for _ in range(C.B)]
    out = {"n_cells": int(n), "n_ball": int(arm.sum()), "probe_drops_after": int(ndp.sum())}
    for c, h in enumerate(DEFS):
        a = float(roc_auc_score(arm, delta[:, c]))
        b = [roc_auc_score(arm[i], delta[i, c]) if 0 < arm[i].sum() < n else np.nan for i in idx]
        out[h] = {"auc": a, "ci": ci(b), "mean_delta_ball": float(delta[arm == 1, c].mean()),
                  "mean_delta_thin": float(delta[arm == 0, c].mean())}
    return out


def tests(d: str, e: str, workers: int) -> None:
    t0 = time.time()
    td = tdir(d, e)
    okeys = list(outcomes(d, e))
    todo = [(d, e, o) for o in okeys if not (td / f"mom_{o}.npz").exists()]
    run_pool(momentum_task, todo, workers)
    log(f"  momentum arms ready ({len(todo)} computed)")
    todo = [(d, e, o, h) for o, h in configs(d, e) if not (td / f"T_{o}_{h}.json").exists()]
    run_pool(test_task, todo, workers)
    log(f"  {len(todo)} T configurations computed")
    for space in SPACES:
        for k in space_ks(space):
            for h in DEFS:
                p = td / f"V1_{space}_K{k}_{h}.json"
                if not p.exists():
                    p.write_text(json.dumps(v1(d, e, space, k, h), indent=1))
    p = td / "V2.json"
    if not p.exists():
        p.write_text(json.dumps(planted_auc(d, e), indent=1))
    log(f"{d}/{e}: tests done in {time.time() - t0:.0f}s")


def run_pool(fn, todo, workers):
    if not todo:
        return
    if workers <= 1:
        for t in todo:
            log(f"    {fn(t)}")
        return
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg_dict(),)) as pool:
        for name in pool.imap_unordered(fn, todo):
            log(f"    {name}")


def cfg_dict() -> dict:
    return {k: getattr(C, k) for k in ("n_draws", "B", "m_probe", "ks", "kbase", "n_planted", "planted_min",
                                       "min_size", "sub", "out", "result", "smoke")}


def _init(cfg: dict) -> None:
    for k, v in cfg.items():
        setattr(C, k, v)


# ----------------------------------------------------------------------------------------------- collect
def collect(which: list[str]) -> dict:
    res: dict = {"reproduction_check": {}, "planted": {}, "min_size": {}, "shrunk": {}, "horizon_12m": {},
                 "horizon_24m": {}, "momentum_by_outcome": {}, "probe_and_tau": {}}
    holm_fam: dict = {}
    for key in which:
        d, e = key.split("/")
        td = C.out / "tests" / f"{d}_{e}"
        rp = C.out / f"{d}_{e}_reproduction.json"
        if rp.exists():
            res["reproduction_check"][key] = json.loads(rp.read_text())
        if (td / "V2.json").exists():
            res["planted"][key] = json.loads((td / "V2.json").read_text())
        for space in SPACES:
            for k in space_ks(space):
                if fpath(d, e, space, k).exists():
                    with np.load(fpath(d, e, space, k)) as z:
                        res["probe_and_tau"][f"{key}/{space}/K{k}"] = {
                            "tau": float(z["tau"]), "tau_pct": int(z["tau_pct"]), "tried": json.loads(str(z["tau_tried"])),
                            "probe_drop_total": int(z["probe_drop_total"]),
                            "n_cells_all_probes_dropped": int(z["n_cells_all_probes_dropped"])}
        for o in (outcomes(d, e) if (td / "V2.json").exists() else {}):
            mp = td / f"mom_{o}.npz"
            if mp.exists():
                with np.load(mp) as z:
                    res["momentum_by_outcome"][f"{key}/{o}"] = {
                        "auc_direct": float(z["auc_direct"]), "auc_direct_ci": ci(z["direct"]),
                        "cv_auc": float(z["cv"]), "cv_auc_cluster_ci": ci(z["arm"]), "n_positive": int(z["n_pos"])}
        for o, h in configs(d, e):
            tp = td / f"T_{o}_{h}.json"
            if not tp.exists():
                continue
            t = json.loads(tp.read_text())
            if o in ("min_size", "shrunk", "horizon_12m", "horizon_24m"):
                res[o][f"{key}/{h}"] = t
            else:
                space, kk = o.split("_K")
                vp = td / f"V1_{space}_K{kk}_{h}.json"
                res.setdefault(key, {}).setdefault(space, {}).setdefault(f"K{kk}", {})[h] = {
                    **t, "V1": json.loads(vp.read_text()) if vp.exists() else None}
            fam = holm_fam.setdefault(key, {"T1": {}, "T2": {}, "T3": {}})
            fam["T1"][f"{o}/{h}"] = t["T1"]["p_boot"]
            fam["T2"][f"{o}/{h}"] = t["T2"]["p_gain_boot"]
            fam["T3"][f"{o}/{h}"] = t["T3"]["p"]
    res["holm"] = {k: {t: W.holm({kk: (1.0 if np.isnan(p) else p) for kk, p in v.items()}) for t, v in f.items()}
                   for k, f in holm_fam.items()}
    res["decisions"] = decisions(res, which)
    res["constants"] = {"seed": SEED, "B": C.B, "k_nn": K_NN, "n_draws": C.n_draws, "m_probe": C.m_probe,
                        "r_pca": R_PCA, "ks": list(C.ks), "kbase": C.kbase, "whiten_dim": WHITEN_DIM, "smoke": C.smoke}
    return res


def decisions(res: dict, which: list[str]) -> dict:
    cells = [k for k in which if k in res]
    out: dict = {"n_corpus_embedder_cells": len(cells), "complete": len(cells) == 6}
    if not cells:
        return out
    ncell = len(cells)
    need5, need4 = (5, 4) if ncell == 6 else (max(1, round(5 * ncell / 6)), max(1, round(4 * ncell / 6)))
    planted_ok = {}
    for h in DEFS:
        n = sum(1 for k in cells if k in res["planted"] and res["planted"][k][h]["auc"] >= 0.75
                and res["planted"][k][h]["ci"][0] > 0.5)
        planted_ok[h] = {"n_cells": n, "sees_planted_hole": n >= need5}
    validity = {}
    for space in SPACES:
        for k in space_ks(space):
            for h in DEFS:
                ne = nd = 0
                for c in cells:
                    v = res[c].get(space, {}).get(f"K{k}", {}).get(h, {}).get("V1")
                    if not v:
                        continue
                    a, p = v["a_density"], v["partial_a_given_bcd"]
                    emp = (a["rho"] <= -0.3 and a["ci"][1] < 0 and p["rho"] <= -0.2 and p["ci"][1] < 0)
                    ne += emp
                    nd += (v["r2_geom"]["value"] >= 0.5 and not emp)
                tracks, dom = ne >= need5, nd >= need4
                validity[f"{space}/K{k}/{h}"] = {"n_tracks_emptiness": ne, "tracks_emptiness": tracks,
                                                 "n_dominated": nd, "dominated_by_geometry": dom,
                                                 "sees_planted_hole": planted_ok[h]["sees_planted_hole"],
                                                 "valid": planted_ok[h]["sees_planted_hole"] or (tracks and not dom)}
    out["planted"], out["validity"] = planted_ok, validity
    h0 = validity.get(f"raw/K{C.kbase}/H0", {})
    out["against_paper_H0"] = bool(h0.get("dominated_by_geometry") or not planted_ok["H0"]["sees_planted_hole"])
    valid = [k for k, v in validity.items() if v["valid"]]
    out["valid_configurations"] = valid
    holm = res["holm"]

    def get(c, cfg, src):
        space, k, h = cfg.split("/")
        if src == "grid":
            return res.get(c, {}).get(space, {}).get(k, {}).get(h)
        return res[src].get(f"{c}/{h}")

    def judge(cfgs, src):
        j: dict = {}
        for dom in W.DOMAINS:
            pair = [f"{dom}/{e}" for e in W.EMBEDDERS]
            c1, c2, inc, strong_all, c3 = [], [], [], True, []
            for cfg in cfgs:
                ts = [get(c, cfg, src) for c in pair]
                if any(t is None for t in ts):
                    strong_all = False
                    continue
                okey = (lambda t: t["outcome"])
                if all(t["T1"]["ci"][0] > 0.5 for t in ts):
                    c1.append({"config": cfg, "survives_holm_both": all(
                        holm[c]["T1"][f"{okey(t)}/{t['definition']}"] < 0.05 for c, t in zip(pair, ts))})
                if all(t["T2"]["gain"] >= 0.03 and t["T2"]["gain_cluster_ci"][0] > 0 for t in ts):
                    c2.append({"config": cfg, "survives_holm_both": all(
                        holm[c]["T2"][f"{okey(t)}/{t['definition']}"] < 0.05 for c, t in zip(pair, ts))})
                elif all(t["T2"]["gain"] >= 0.03 for t in ts):
                    inc.append(cfg)
                if not all(t["T2"]["gain_cluster_ci"][1] < 0.05 for t in ts):
                    strong_all = False
                if all(t["T3"]["partial_rho"] > 0 and t["T3"]["ci"][0] > 0 for t in ts):
                    c3.append(cfg)
            j[dom] = {"C1_threatened_by": c1, "C1_threatened": bool(c1), "C2_threatened_by": c2,
                      "C2_threatened": bool(c2), "C2_inconclusive": inc,
                      "C2_strengthened": bool(cfgs) and strong_all, "C3_positive_configs": c3}
        c3_ok = [cfg for cfg in cfgs if all(cfg in j[dm]["C3_positive_configs"] for dm in ("steam", "arxiv"))]
        j["C3"] = ("robust" if cfgs and len(c3_ok) == len(cfgs) else "definition-dependent" if c3_ok else "not supported")
        j["C3_configs_positive_in_steam_and_arxiv"] = c3_ok
        j["independent_of_definition_resolution_space"] = not any(j[dm]["C1_threatened"] or j[dm]["C2_threatened"]
                                                                  for dm in W.DOMAINS)
        return j

    out["conclusions_grid"] = judge(valid, "grid")
    base_valid = [f"raw/K{C.kbase}/{h}" for h in DEFS if validity[f"raw/K{C.kbase}/{h}"]["valid"]]
    for var in ("min_size", "shrunk", "horizon_12m", "horizon_24m"):
        out[f"conclusions_{var}"] = judge(base_valid, var)
    mo = res["momentum_by_outcome"]
    spec = {}
    for e in W.EMBEDDERS:
        cis = {dm: mo.get(f"{dm}/{e}/horizon_12m", {}).get("auc_direct_ci") for dm in W.DOMAINS}
        doms = [dm for dm in W.DOMAINS if cis[dm]]
        spec[e] = any(cis[a][1] < cis[b][0] or cis[b][1] < cis[a][0] for a in doms for b in doms if a < b)
    out["momentum_corpus_specific_12m"] = spec
    return out


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*")
    ap.add_argument("--stage", choices=("features", "tests", "collect", "all"), default="all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir")
    ap.add_argument("--repro-only", action="store_true", help="full-data raw K=256 reproduction check only")
    a = ap.parse_args()
    which = a.which or [f"{d}/{e}" for e in W.EMBEDDERS for d in W.DOMAINS]
    if a.smoke:
        if not a.out_dir:
            raise SystemExit("--smoke needs --out-dir (never the registered cache)")
        C.smoke, C.sub, C.n_draws, C.B, C.m_probe = True, (8000, 800, 5000), 40_000, 30, 60
        C.ks, C.kbase, C.n_planted, C.min_size = (64, 128, 256), 128, 52, 40
    if a.out_dir:
        C.out = Path(a.out_dir)
        C.result = C.out / "review_hole_validity.json"
    if a.repro_only:
        if not a.out_dir:
            raise SystemExit("--repro-only needs --out-dir")
        for key in which:
            features(*key.split("/"), repro_only=True)
        return
    for key in which:
        d, e = key.split("/")
        if a.stage in ("features", "all"):
            features(d, e)
        if a.stage in ("tests", "all"):
            tests(d, e, a.workers)
    if a.stage in ("tests", "collect", "all"):
        res = collect([k for k in ([f"{d}/{e}" for e in W.EMBEDDERS for d in W.DOMAINS])
                       if (C.out / "tests" / f"{k.replace('/', '_')}" / "V2.json").exists()])
        C.result.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        log(f"wrote {C.result}")


if __name__ == "__main__":
    main()
