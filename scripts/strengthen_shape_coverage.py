"""Coverage simulation for the selection-aware inspiration-shape bootstrap (Section 6).

Pre-registered in docs/prereg/strengthen_shape_coverage.md (committed before this was run). For each embedder x
framing cell it builds data-generating processes from the real design (same stimuli and within-source distance ranks,
source fixed effects at their fitted values, residuals of the fitted 4-df spline model resampled within source),
imposes a known curve f(x), and re-runs the bootstrap of scripts/strengthen_inspiration_shape.py (stimuli resampled
within source) on every simulated dataset. It reports how often the one-sided 95 % upper bound U on H_peak (4-df
natural spline) covers the true H_peak, the power of the one-sided lower bound, and the coverage of the two-lines
fall CI. Reuses the saved gpt-4o scored tables; no generation, no embedding model, no API calls.

    uv run python scripts/strengthen_shape_coverage.py            # full run (4 worker processes, 1 BLAS thread each)
    uv run python scripts/strengthen_shape_coverage.py --check    # equivalence + timing check only, writes nothing
Output: docs/prereg/strengthen_shape_coverage.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402
import strengthen_inspiration_shape as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/strengthen_shape_coverage.json"
SHAPE_JSON = ROOT / "docs/prereg/strengthen_inspiration_shape.json"
REV_JSON = ROOT / "docs/prereg/revision_inspiration.json"
SEED = 20261106
N_SIM = 500          # simulated datasets per scenario x cell
B_IN = 1000          # inner bootstrap resamples per simulated dataset
CHUNK = 125          # inner resamples processed at once (memory)
WORKERS = 4
EMB = ("qwen3e06", "bgem3")
FR = ("INCIDENTAL", "INSTRUCTED")
GRID = S.GRID
FINE = np.linspace(0.10, 0.90, 1601)  # continuous sup over [0.1, 0.9] (step 0.0005)
NB = len(A.SOURCES)


# ---------------------------------------------------------------- curves (functions of the rank x in [0, 1])
def raised_cos(x, c, fwhm):
    """Height-1 raised-cosine bump centred at c with full width at half maximum fwhm (support 2 * fwhm)."""
    u = (x - c) / (2 * fwhm)
    return np.where(np.abs(u) < 0.5, 0.5 * (1 + np.cos(2 * np.pi * u)), 0.0)


HET_CENTRES = np.linspace(0.2, 0.8, NB)


def make_curve(kind, h, slope):
    """Return f(x, src_index) for a scenario; src_index is an int array (or None for a common curve)."""
    if kind == "flat":
        return lambda x, si=None: np.zeros_like(x, float)
    if kind == "linear":
        return lambda x, si=None: slope * x
    if kind == "broad":
        return lambda x, si=None: h * np.sin(np.pi * x) ** 2
    if kind == "narrow":
        return lambda x, si=None: h * raised_cos(x, 0.60, 0.15)
    if kind == "between":
        return lambda x, si=None: h * raised_cos(x, 0.4575, 0.15)
    if kind == "spike":
        return lambda x, si=None: h * raised_cos(x, 0.4575, 0.05)
    if kind == "rise_narrow":
        return lambda x, si=None: slope * x + h * raised_cos(x, 0.60, 0.15)
    if kind == "het":
        def f(x, si=None):
            if si is None:  # population common curve = mean over sources (equal n per source, same ranks)
                return h * np.mean([raised_cos(x, c, 0.15) for c in HET_CENTRES], axis=0)
            return h * raised_cos(x, HET_CENTRES[si], 0.15)
        return f
    raise ValueError(kind)


# ---------------------------------------------------------------- batched weighted-LS machinery
def wls_batch(W, X, y):
    """W: (b, n) resampling counts; X: (n, p) or (b, n, p); returns beta (b, p), sse (b,), inv(G) (b, p, p)."""
    if X.ndim == 2:
        G = np.einsum("bn,ni,nj->bij", W, X, X, optimize=True)
        r = (W * y) @ X
    else:
        G = np.einsum("bn,bni,bnj->bij", W, X, X, optimize=True)
        r = np.einsum("bn,bni->bi", W * y, X, optimize=True)
    Gi = np.linalg.inv(G)
    beta = np.einsum("bij,bj->bi", Gi, r)
    sse = (W * y * y).sum(1) - np.einsum("bi,bi->b", beta, r)
    return beta, sse, Gi


def wquantile(W, order, xs, q):
    """numpy 'linear' quantile of the resampled x (counts W over the fixed x) for each row; q scalar or (b,)."""
    n = int(round(W[0].sum()))
    cum = np.cumsum(W[:, order], axis=1)  # (b, n) counts in sorted-x order
    q = np.broadcast_to(np.asarray(q, float), (W.shape[0],))
    pos = q * (n - 1)
    lo = np.floor(pos)
    frac = pos - lo
    hi = np.minimum(lo + 1, n - 1)

    def at(k):
        idx = np.array([np.searchsorted(cum[i], k[i], side="right") for i in range(len(k))])
        return xs[idx]
    vlo, vhi = at(lo), at(hi)
    return vlo + frac * (vhi - vlo)


class Design:
    """Everything fixed for one embedder x framing cell."""

    def __init__(self, y, x, src):
        self.y, self.x, self.src = y, x, src
        self.n = len(y)
        self.si = np.array([A.SOURCES.index(s) for s in src])
        self.D = A.dummies(src, A.SOURCES)
        self.xmin, self.xmax = float(x.min()), float(x.max())
        self.kn = np.quantile(x, [0, .25, .5, .75, 1])
        self.Xs = np.column_stack([S.ns_basis(x, self.kn), self.D])
        self.at = np.concatenate([GRID, [self.xmin, self.xmax]])
        self.Bat = S.ns_basis(self.at, self.kn)
        self.Xq = np.column_stack([x, x ** 2, self.D])
        self.order = np.argsort(x, kind="stable")
        self.xs = x[self.order]
        self.idx = [np.flatnonzero(self.si == k) for k in range(NB)]

    # spline H_peak for each row of W
    def hpeak(self, W, y):
        beta, _, _ = wls_batch(W, self.Xs, y)
        f = beta[:, :4] @ self.Bat.T
        g = f[:, :len(GRID)]
        return g.max(1) - np.maximum(f[:, len(GRID)], f[:, len(GRID) + 1])

    def _tl_fit(self, W, y, xc):
        x = self.x[None, :]
        c = xc[:, None]
        lo = np.where(x < c, x - c, 0.0)
        hi = np.where(x >= c, x - c, 0.0)
        ind = (x >= c).astype(float)
        X = np.concatenate([lo[..., None], hi[..., None], ind[..., None],
                            np.broadcast_to(self.D, (W.shape[0],) + self.D.shape)], axis=2)
        beta, sse, Gi = wls_batch(W, X, y)
        dof = self.n - X.shape[2]
        s2 = sse / dof
        se0, se1 = np.sqrt(s2 * Gi[:, 0, 0]), np.sqrt(s2 * Gi[:, 1, 1])
        return beta[:, 0] / se0, beta[:, 1] / se1, beta[:, 0], beta[:, 1], dof

    def two_lines(self, W, y):
        """Batched exact replica of inspiration_analysis.two_lines on the resampled data."""
        bq, _, _ = wls_batch(W, self.Xq, y)
        med = wquantile(W, self.order, self.xs, 0.5)
        p10 = wquantile(W, self.order, self.xs, 0.10)
        p90 = wquantile(W, self.order, self.xs, 0.90)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = np.where(bq[:, 1] < 0, -bq[:, 0] / (2 * bq[:, 1]), med)
        xc = np.clip(xc, p10, p90)
        z1, z2, *_ = self._tl_fit(W, y, xc)
        a1, a2 = np.abs(z1), np.abs(z2)
        share = np.where(a1 < a2, a2 / (a1 + a2), 1 - a1 / (a1 + a2))
        xc2 = np.clip(wquantile(W, self.order, self.xs, share), p10, p90)
        z1, z2, b1, b2, dof = self._tl_fit(W, y, xc2)
        p1, p2 = stats.t.sf(z1, dof), stats.t.cdf(z2, dof)
        return {"bp": xc2, "slope_low": b1, "slope_high": b2,
                "fall": -b2 * (self.xmax - xc2), "hump": (p1 < 0.05) & (p2 < 0.05)}

    def boot_counts(self, rng, b):
        """Counts of a with-replacement resample of each source's stimuli (multinomial = same distribution)."""
        W = np.zeros((b, self.n))
        for ix in self.idx:
            W[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1.0 / len(ix)), size=b)
        return W

    # population quantities of a curve (no noise)
    def pop_spline_hpeak(self, f0):
        return float(self.hpeak(np.ones((1, self.n)), f0)[0])

    def pop_fall_at(self, f0, bps):
        W = np.ones((len(bps), self.n))
        _, _, _, b2, _ = self._tl_fit_noiseless(W, f0, np.asarray(bps, float))
        return -b2 * (self.xmax - np.asarray(bps, float))

    def _tl_fit_noiseless(self, W, y, xc):
        x = self.x[None, :]
        c = xc[:, None]
        lo = np.where(x < c, x - c, 0.0)
        hi = np.where(x >= c, x - c, 0.0)
        ind = (x >= c).astype(float)
        X = np.concatenate([lo[..., None], hi[..., None], ind[..., None],
                            np.broadcast_to(self.D, (W.shape[0],) + self.D.shape)], axis=2)
        beta, _, _ = wls_batch(W, X, y)
        return None, None, beta[:, 0], beta[:, 1], None


def load_cell(emb, fr):
    d = pd.read_parquet(A.RUN / f"scored_{emb}.parquet")
    g = d[d["arm"] == fr].reset_index(drop=True)
    y, src = g["margin"].to_numpy(float), g["source"].to_numpy()
    x = g.groupby("source")["dist"].rank(pct=True).to_numpy()
    return y, x, src


def dgp_parts(des: Design):
    """Source fixed effects and residuals of the fitted 4-df spline + source FE model on the real data."""
    b, *_ = np.linalg.lstsq(des.Xs, des.y, rcond=None)
    resid = des.y - des.Xs @ b
    fe = b[4:]
    return fe, resid


# ---------------------------------------------------------------- one scenario x cell
def simulate(task):
    emb, fr, name, kind, h, slope, seed, n_sim = task
    y, x, src = load_cell(emb, fr)
    des = Design(y, x, src)
    fe, resid = dgp_parts(des)
    f = make_curve(kind, h, slope)
    f_obs = f(x, des.si) if kind == "het" else f(x)
    # true estimands of the population (common) curve
    fc = lambda t: f(t)  # noqa: E731
    end = max(fc(np.array([des.xmin]))[0], fc(np.array([des.xmax]))[0])
    H_true = float(fc(FINE).max() - end)
    H_true_grid = float(fc(GRID).max() - end)
    H_proj = des.pop_spline_hpeak(f_obs)
    if kind == "het":
        H_src = [float(h * raised_cos(FINE, c, 0.15).max()
                       - max(h * raised_cos(np.array([des.xmin]), c, 0.15)[0],
                             h * raised_cos(np.array([des.xmax]), c, 0.15)[0])) for c in HET_CENTRES]
    else:
        H_src = None
    p10, p90 = np.percentile(x, [10, 90])
    bp_grid = np.linspace(p10, p90, 161)
    fall_max_true = float(des.pop_fall_at(f_obs, bp_grid).max())
    rng = np.random.default_rng(seed)
    base = fe[des.si] + f_obs
    rec = {k: [] for k in ("H_hat", "U", "L", "fall_hat", "fall_lo", "fall_hi", "fall_u95", "bp_hat",
                           "fall_true_at_bp", "hump_point")}
    for _ in range(n_sim):
        e = np.empty(des.n)
        for ix in des.idx:
            e[ix] = resid[ix][rng.integers(0, len(ix), len(ix))]
        ys = base + e
        one = np.ones((1, des.n))
        Hh = float(des.hpeak(one, ys)[0])
        tl0 = des.two_lines(one, ys)
        Hb, Fb = [], []
        for c0 in range(0, B_IN, CHUNK):
            W = des.boot_counts(rng, min(CHUNK, B_IN - c0))
            Hb.append(des.hpeak(W, ys))
            Fb.append(des.two_lines(W, ys)["fall"])
        Hb, Fb = np.concatenate(Hb), np.concatenate(Fb)
        bp = float(tl0["bp"][0])
        rec["H_hat"].append(Hh)
        rec["U"].append(float(np.percentile(Hb, 95)))
        rec["L"].append(float(np.percentile(Hb, 5)))
        rec["fall_hat"].append(float(tl0["fall"][0]))
        rec["fall_lo"].append(float(np.nanpercentile(Fb, 2.5)))
        rec["fall_hi"].append(float(np.nanpercentile(Fb, 97.5)))
        rec["fall_u95"].append(float(np.nanpercentile(Fb, 95)))
        rec["bp_hat"].append(bp)
        rec["fall_true_at_bp"].append(float(des.pop_fall_at(f_obs, [bp])[0]))
        rec["hump_point"].append(bool(tl0["hump"][0]))
    return {"emb": emb, "fr": fr, "name": name, "kind": kind, "h": h, "slope": slope, "seed": seed,
            "H_true": H_true, "H_true_grid": H_true_grid, "H_proj": H_proj, "H_src": H_src,
            "fall_max_true": fall_max_true, "rec": rec}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - hw), float(c + hw)]


def verdict(k, n, target=0.95):
    lo, hi = wilson(k, n)
    if lo > target:
        return "conservative"
    if hi < target:
        return "anti-conservative"
    return "calibrated"


def summarise(r):
    rec = {k: np.asarray(v) for k, v in r["rec"].items()}
    n = len(rec["U"])
    out = {k: r[k] for k in ("kind", "h", "slope", "H_true", "H_true_grid", "H_proj", "fall_max_true", "seed")}
    out["n_sim"] = n
    for est, val in (("true", r["H_true"]), ("true_grid", r["H_true_grid"]), ("proj", r["H_proj"])):
        k = int((val <= rec["U"]).sum())
        out[f"coverage_U_vs_H_{est}"] = {"share": k / n, "wilson95": wilson(k, n), "verdict": verdict(k, n)}
    if r["H_src"] is not None:
        hs = float(np.max(r["H_src"]))
        out["H_source_specific_max"] = hs
        k = int((hs <= rec["U"]).sum())
        out["coverage_U_vs_H_source_specific_max"] = {"share": k / n, "wilson95": wilson(k, n),
                                                      "verdict": verdict(k, n)}
    k = int((rec["L"] > 0).sum())
    out["power_lower_bound_gt0"] = {"share": k / n, "wilson95": wilson(k, n)}
    out["H_hat_mean"] = float(rec["H_hat"].mean())
    out["H_hat_sd"] = float(rec["H_hat"].std(ddof=1))
    out["U_mean"] = float(rec["U"].mean())
    out["U_quantiles_5_50_95"] = [float(q) for q in np.percentile(rec["U"], [5, 50, 95])]
    out["mean_U_minus_H_true"] = float((rec["U"] - r["H_true"]).mean())
    ft = rec["fall_true_at_bp"]
    k = int(((rec["fall_lo"] <= ft) & (ft <= rec["fall_hi"])).sum())
    out["fall_ci95_coverage_at_selected_bp"] = {"share": k / n, "wilson95": wilson(k, n),
                                                "verdict": verdict(k, n)}
    k = int((ft <= rec["fall_hi"]).sum())
    out["fall_upper97.5_covers_true_at_selected_bp"] = {"share": k / n, "wilson95": wilson(k, n),
                                                       "verdict": verdict(k, n, 0.975)}
    k = int((r["fall_max_true"] <= rec["fall_hi"]).sum())
    out["fall_upper97.5_covers_max_true_fall_over_bp"] = {"share": k / n, "wilson95": wilson(k, n),
                                                         "verdict": verdict(k, n, 0.975)}
    k = int((rec["fall_lo"] > 0).sum())
    out["fall_ci_excludes_zero_from_above_share"] = k / n
    out["two_lines_hump_declared_share"] = float(rec["hump_point"].mean())
    out["fall_true_at_bp_mean"] = float(ft.mean())
    return out


def scenarios(cell_U, sd_idea, slope_q):
    rows = [("flat", "flat", 0.0), ("linear_qwen3_slope", "linear", 0.0)]
    for tag, hv in (("U", cell_U), ("halfSD", 0.5 * sd_idea)):
        for kind in ("broad", "narrow", "between", "spike", "het", "rise_narrow"):
            rows.append((f"{kind}_h{tag}", kind, hv))
    return [(nm, kind, float(h), float(slope_q)) for nm, kind, h in rows]


def build_tasks(n_sim=N_SIM):
    shp = json.loads(SHAPE_JSON.read_text())
    rev = json.loads(REV_JSON.read_text())["reproduction_full"]
    ss = np.random.SeedSequence(SEED)
    tasks, meta = [], {}
    for emb in EMB:
        for fr in FR:
            s = shp[emb]["shape"][fr]
            U = s["reading_rule"]["U_spline4_H_peak"]
            sd = s["scales"]["sd_idea_within_source"]
            slope_q = rev["qwen3e06"][fr]["P1"]["slope"]
            meta[f"{emb}/{fr}"] = {"U_observed": U, "sd_idea": sd, "half_sd": 0.5 * sd, "qwen3_slope": slope_q}
            for nm, kind, h, sl in scenarios(U, sd, slope_q):
                seed = int(ss.spawn(1)[0].generate_state(1)[0])
                tasks.append((emb, fr, nm, kind, h, sl, seed, n_sim))
    return tasks, meta


# ---------------------------------------------------------------- checks
def check():
    """Equivalence of the batched procedure with the reference loop, and timing. Produces no results."""
    rng = np.random.default_rng(1)
    for emb in EMB:
        for fr in FR:
            y, x, src = load_cell(emb, fr)
            des = Design(y, x, src)
            W = des.boot_counts(rng, 20)
            hb = des.hpeak(W, y)
            tl = des.two_lines(W, y)
            for i in range(20):
                b = np.repeat(np.arange(des.n), W[i].astype(int))
                f = S.spline_curve(y[b], x[b], src[b], des.kn, des.at)
                href = f[:len(GRID)].max() - max(f[len(GRID)], f[len(GRID) + 1])
                ref = A.two_lines(y[b], x[b], src[b])
                assert abs(hb[i] - href) < 1e-9, (hb[i], href)
                assert abs(tl["bp"][i] - ref["breakpoint"]) < 1e-9, (tl["bp"][i], ref["breakpoint"])
                assert abs(tl["slope_high"][i] - ref["slope_high"]) < 1e-9
                assert bool(tl["hump"][i]) == ref["hump"]
            # also the full-sample point values reproduce the saved JSON
            shp = json.loads(SHAPE_JSON.read_text())[emb]["shape"][fr]["point"]
            one = np.ones((1, des.n))
            assert abs(des.hpeak(one, y)[0] - shp["spline4_H_peak"]) < 1e-9
            assert abs(des.two_lines(one, y)["fall"][0] - shp["fall_after"]) < 1e-9
            print("equivalent:", emb, fr, flush=True)
    tasks, _ = build_tasks(n_sim=2)
    t0 = time.time()
    simulate(tasks[0])
    dt = (time.time() - t0) / 2
    print(f"seconds per simulated dataset (B_IN={B_IN}, 1 thread): {dt:.2f}; "
          f"tasks={len(tasks)}; projected wall hours with {WORKERS} workers: "
          f"{dt * N_SIM * len(tasks) / WORKERS / 3600:.2f}", flush=True)


def main():
    tasks, meta = build_tasks()
    t0 = time.time()
    results = []
    with get_context("fork").Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(simulate, tasks)):
            results.append(r)
            sm = summarise(r)
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:.0f}s] {r['emb']} {r['fr']} {r['name']}: "
                  f"H_true={r['H_true']:.5f} cov={sm['coverage_U_vs_H_true']['share']:.3f} "
                  f"power={sm['power_lower_bound_gt0']['share']:.3f}", flush=True)
    res = {"prereg": "docs/prereg/strengthen_shape_coverage.md", "seed": SEED, "n_sim": N_SIM, "B_inner": B_IN,
           "workers": WORKERS, "threads_per_worker": 1, "grid": [0.1, 0.9], "fine_sup_step": 0.0005,
           "cells": meta, "wall_seconds": time.time() - t0, "results": {}}
    for r in sorted(results, key=lambda r: (r["emb"], r["fr"], r["name"])):
        res["results"].setdefault(f"{r['emb']}/{r['fr']}", {})[r["name"]] = summarise(r)
    OUT.write_text(json.dumps(res, indent=1))
    raw = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/shape_coverage"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "per_dataset.json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in results]))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
