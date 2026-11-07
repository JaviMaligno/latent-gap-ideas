"""Calibrated upper bound for the Section 6 inspiration shape: a simultaneous sup-t bootstrap band projected onto
the peak functional, validated by simulation against the projected and the true peak.

Pre-registered in docs/prereg/strengthen_shape_band.md (committed before this was run for results). It replaces the
uncalibrated percentile U of scripts/strengthen_inspiration_shape.py (95th percentile of the bootstrap H_peak), which
docs/prereg/strengthen_shape_coverage.json showed covers the fitted-spline peak in only 0.85-0.89 under broad humps.

Procedure (per dataset, per spline df in 4 / 6 / 8; same bootstrap as the shape analysis: stimuli resampled with
replacement within source, B = 4,000):
  contrasts D(t, e) = f(t) - f(e) of the source-common spline, t in the grid 0.10..0.90 (step 0.01), e in {x_min, x_max};
  H_peak = max_t min_e D(t, e)  (= max over the grid minus the higher end, the estimand of the existing scripts);
  sigma = bootstrap SD of each contrast; c = 95th percentile over resamples of max_(t,e) (D_hat - D*) / sigma;
  U_band = max_t min_e (D_hat + c * sigma).  If D <= D_hat + c * sigma for every contrast (probability ~0.95), then
  H_peak <= U_band, so U_band is a (conservative) 95 % upper bound on the peak of whatever curve D_hat estimates.
Secondary (the curve-band form): two-sided 95 % sup-t band on the curve values f (basis normalisation f(0) = 0),
  U_curveband = max_t upper(t) - max(lower(x_min), lower(x_max)).
Reference: the old percentile U_pct = 95th percentile of H_peak*.  Every bound is also computed from the first 1,000 of
the same 4,000 resamples (inner B = 1,000 vs 4,000 on shared simulated datasets).

    uv run python scripts/strengthen_shape_band.py            # observed bound (4 cells) + full simulation, 4 workers
    uv run python scripts/strengthen_shape_band.py --check    # equivalence + timing only; writes nothing
Outputs: docs/prereg/strengthen_shape_band_observed.json, docs/prereg/strengthen_shape_band.json,
         ~/Documents/experiment-data/latent-gap-ideas/runs/shape_band/{run.log, per-task records}
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
import strengthen_shape_coverage as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/strengthen_shape_band.json"
OUT_OBS = ROOT / "docs/prereg/strengthen_shape_band_observed.json"
RAW = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/shape_band"
COV_RAW = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/shape_coverage/per_dataset.json"
SEED = 20261108       # simulation (per-task seeds spawned in task order)
SEED_OBS = 20261109   # observed-data bootstrap
N_SIM = 1000
B_IN = 4000
B_SMALL = 1000        # the first B_SMALL of the same B_IN resamples
CHUNK = 500
WORKERS = 4
DFS = (4, 6, 8)
GRID = S.GRID
NG = len(GRID)
LEVEL = 0.95
Z_FAMILY = float(stats.norm.ppf(1 - 0.05 / (2 * 56)))  # Bonferroni over the 56 scenario-cells, two-sided


def knots(x, df):
    q = {4: [0, .25, .5, .75, 1], 6: [0, 1 / 6, 2 / 6, .5, 4 / 6, 5 / 6, 1]}.get(df, np.linspace(0, 1, df + 1))
    return np.quantile(x, q)


class Band:
    """Fixed design of one embedder x framing cell, with batched spline fits for every df."""

    def __init__(self, y, x, src):
        self.y, self.x, self.src = y, x, src
        self.n = len(y)
        self.si = np.array([A.SOURCES.index(s) for s in src])
        self.idx = [np.flatnonzero(self.si == k) for k in range(len(A.SOURCES))]
        D = A.dummies(src, A.SOURCES)
        self.xmin, self.xmax = float(x.min()), float(x.max())
        self.at = np.concatenate([GRID, [self.xmin, self.xmax]])
        self.fit = {}
        for df in DFS:
            kn = knots(x, df)
            X = np.column_stack([S.ns_basis(x, kn), D])
            p = X.shape[1]
            self.fit[df] = {"kn": kn, "X": X, "p": p, "q": X.shape[1] - D.shape[1],
                            "XX": (X[:, :, None] * X[:, None, :]).reshape(self.n, p * p),
                            "Bat": S.ns_basis(self.at, kn)}

    def curves(self, W, y, df):
        """Curve values at `at` (grid + the two ends; f(0) = 0 normalisation) for each row of the counts W."""
        F = self.fit[df]
        p, q = F["p"], F["q"]
        G = (W @ F["XX"]).reshape(-1, p, p)
        r = W @ (F["X"] * y[:, None])
        beta = np.linalg.solve(G, r[..., None])[..., 0]
        return beta[:, :q] @ F["Bat"].T

    def boot_counts(self, rng, b):
        """Same draws, in the same order, as strengthen_shape_coverage.Design.boot_counts."""
        W = np.zeros((b, self.n))
        for ix in self.idx:
            W[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1.0 / len(ix)), size=b)
        return W


def contrasts(f):
    """(b, NG + 2) curve values -> (b, NG, 2) contrasts f(t) - f(end)."""
    return f[:, :NG, None] - f[:, None, NG:NG + 2]


def hpeak(f):
    return f[:, :NG].max(1) - f[:, NG:NG + 2].max(1)


def bounds(fhat, fstar):
    """All bounds from the point curve fhat (NG + 2,) and the bootstrap curves fstar (B, NG + 2)."""
    Dh = contrasts(fhat[None])[0]
    Ds = contrasts(fstar)
    sd = Ds.std(0, ddof=1)
    T = ((Dh[None] - Ds) / sd[None]).reshape(len(fstar), -1).max(1)
    c = float(np.percentile(T, 100 * LEVEL))
    U_band = float((Dh + c * sd).min(1).max())
    sdf = fstar.std(0, ddof=1)
    sdf = np.where(sdf > 0, sdf, np.inf)  # never occurs on this design (x_min > 0); guards a 0/0
    T2 = (np.abs(fstar - fhat[None]) / sdf[None]).max(1)
    c2 = float(np.percentile(T2, 100 * LEVEL))
    up, lo = fhat + c2 * sdf, fhat - c2 * sdf
    U_cb = float(up[:NG].max() - max(lo[NG], lo[NG + 1]))
    Hs = hpeak(fstar)
    return {"U_band": U_band, "c_band": c, "U_curveband": U_cb, "c_curveband": c2,
            "U_pct": float(np.percentile(Hs, 95)), "L_pct": float(np.percentile(Hs, 5))}


def boot_bounds(des: Band, ys, rng, b_in=B_IN, chunk=CHUNK, b_small=B_SMALL):
    one = np.ones((1, des.n))
    fhat = {df: des.curves(one, ys, df)[0] for df in DFS}
    fs = {df: [] for df in DFS}
    for c0 in range(0, b_in, chunk):
        W = des.boot_counts(rng, min(chunk, b_in - c0))
        for df in DFS:
            fs[df].append(des.curves(W, ys, df))
    out = {}
    for df in DFS:
        F = np.concatenate(fs[df])
        out[df] = {"H_hat": float(hpeak(fhat[df][None])[0]), f"B{b_in}": bounds(fhat[df], F)}
        if b_small and b_small < b_in:
            out[df][f"B{b_small}"] = bounds(fhat[df], F[:b_small])
    return out


# ---------------------------------------------------------------- one scenario x cell
def simulate(task, b_in=B_IN, chunk=CHUNK, b_small=B_SMALL, write=True):
    emb, fr, name, kind, h, slope, seed, n_sim = task
    path = RAW / f"{emb}__{fr}__{name}.json"
    if write and path.exists():
        r = json.loads(path.read_text())
        if r.get("n_sim") == n_sim and r.get("seed") == seed:
            return r
    y, x, src = C.load_cell(emb, fr)
    des = Band(y, x, src)
    fe, resid = C.dgp_parts(C.Design(y, x, src))
    f = C.make_curve(kind, h, slope)
    f_obs = f(x, des.si) if kind == "het" else f(x)
    end = max(f(np.array([des.xmin]))[0], f(np.array([des.xmax]))[0])
    H_true = float(f(C.FINE).max() - end)
    H_true_grid = float(f(GRID).max() - end)
    one = np.ones((1, des.n))
    H_proj = {str(df): float(hpeak(des.curves(one, f_obs, df))[0]) for df in DFS}
    H_src = None
    if kind == "het":
        H_src = float(max(h * C.raised_cos(C.FINE, c, 0.15).max()
                          - max(h * C.raised_cos(np.array([des.xmin]), c, 0.15)[0],
                                h * C.raised_cos(np.array([des.xmax]), c, 0.15)[0]) for c in C.HET_CENTRES))
    rng = np.random.default_rng(seed)
    base = fe[des.si] + f_obs
    recs = []
    for _ in range(n_sim):
        e = np.empty(des.n)
        for ix in des.idx:
            e[ix] = resid[ix][rng.integers(0, len(ix), len(ix))]
        bb = boot_bounds(des, base + e, rng, b_in, chunk, b_small)
        recs.append({str(df): v for df, v in bb.items()})
    r = {"emb": emb, "fr": fr, "name": name, "kind": kind, "h": h, "slope": slope, "seed": seed, "n_sim": n_sim,
         "B_inner": b_in, "B_small": b_small, "H_true": H_true, "H_true_grid": H_true_grid, "H_proj": H_proj,
         "H_source_specific_max": H_src, "rec": recs}
    if write:
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r))
    return r


def verdicts(k, n):
    lo, hi = C.wilson(k, n)
    flo, fhi = C.wilson(k, n, Z_FAMILY)
    return {"share": k / n, "wilson95": [lo, hi], "verdict": C.verdict(k, n),
            "wilson_family": [flo, fhi], "anti_conservative_familywise": bool(fhi < LEVEL)}


def summarise(r):
    n = len(r["rec"])
    out = {k: r[k] for k in ("kind", "h", "slope", "seed", "H_true", "H_true_grid", "H_proj",
                             "H_source_specific_max", "n_sim")}
    for df in map(str, DFS):
        o = {}
        for bkey in (f"B{r['B_inner']}", f"B{r['B_small']}"):
            ob = {}
            for meth in ("U_band", "U_curveband", "U_pct"):
                u = np.array([d[df][bkey][meth] for d in r["rec"]])
                m = {"mean": float(u.mean()), "q05_50_95": [float(v) for v in np.percentile(u, [5, 50, 95])]}
                for est, val in (("H_true", r["H_true"]), ("H_true_grid", r["H_true_grid"]),
                                 ("H_proj", r["H_proj"][df])):
                    m[f"coverage_{est}"] = verdicts(int((val <= u).sum()), n)
                if r["H_source_specific_max"] is not None:
                    m["coverage_H_source_specific_max"] = verdicts(int((r["H_source_specific_max"] <= u).sum()), n)
                ob[meth] = m
            ob["c_band_mean"] = float(np.mean([d[df][bkey]["c_band"] for d in r["rec"]]))
            o[bkey] = ob
        hh = np.array([d[df]["H_hat"] for d in r["rec"]])
        o["H_hat_mean"], o["H_hat_sd"] = float(hh.mean()), float(hh.std(ddof=1))
        big, small = f"B{r['B_inner']}", f"B{r['B_small']}"
        for meth in ("U_band", "U_pct"):
            a = np.array([d[df][big][meth] for d in r["rec"]])
            b = np.array([d[df][small][meth] for d in r["rec"]])
            o[f"{meth}_B{r['B_small']}_minus_B{r['B_inner']}"] = {"mean": float((b - a).mean()),
                                                                  "mean_abs": float(np.abs(b - a).mean()),
                                                                  "sd": float((b - a).std(ddof=1))}
        out[f"df{df}"] = o
    return out


# ---------------------------------------------------------------- observed data
def observed():
    shp = json.loads(C.SHAPE_JSON.read_text())
    ss = np.random.SeedSequence(SEED_OBS)
    res = {"prereg": "docs/prereg/strengthen_shape_band.md", "seed": SEED_OBS, "B": B_IN, "level": LEVEL,
           "grid": [0.1, 0.9], "dfs": list(DFS), "cells": {}}
    for emb in C.EMB:
        for fr in C.FR:
            y, x, src = C.load_cell(emb, fr)
            des = Band(y, x, src)
            rng = np.random.default_rng(ss.spawn(1)[0])
            bb = boot_bounds(des, y, rng)
            s = shp[emb]["shape"][fr]
            sd = s["scales"]["sd_idea_within_source"]
            rngm = s["scales"]["range_source_means"]
            cell = {"sd_idea": sd, "range_source_means": rngm,
                    "old_percentile_U_spline4": s["reading_rule"]["U_spline4_H_peak"]}
            for df, v in bb.items():
                for bk in (f"B{B_IN}", f"B{B_SMALL}"):
                    v[bk]["U_band_over_sd_idea"] = v[bk]["U_band"] / sd
                    v[bk]["U_band_over_range_source_means"] = v[bk]["U_band"] / rngm
                    v[bk]["meaningful_optimum_excluded"] = bool(v[bk]["U_band"] < 0.2 * sd
                                                                and v[bk]["U_band"] < 0.25 * rngm)
                cell[f"df{df}"] = v
            res["cells"][f"{emb}/{fr}"] = cell
            log(f"observed {emb} {fr}: " + ", ".join(
                f"df{df} U_band={cell[f'df{df}'][f'B{B_IN}']['U_band']:.5f} "
                f"({cell[f'df{df}'][f'B{B_IN}']['U_band_over_sd_idea']:.2f} SD)" for df in DFS))
    OUT_OBS.write_text(json.dumps(res, indent=1))
    log(f"wrote {OUT_OBS}")


def build_tasks(n_sim=N_SIM):
    """The 56 scenario-cells of strengthen_shape_coverage (same curves and heights), with new seeds."""
    tasks, meta = C.build_tasks(n_sim=n_sim)
    ss = np.random.SeedSequence(SEED)
    return [t[:6] + (int(ss.spawn(1)[0].generate_state(1)[0]), n_sim) for t in tasks], meta


_LOG = None


def log(msg):
    print(msg, flush=True)
    if _LOG is not None:
        with open(_LOG, "a") as fh:
            fh.write(msg + "\n")


# ---------------------------------------------------------------- checks
def check():
    """Equivalence with the existing code where shared, and timing. Computes no coverage and no observed bound."""
    rng = np.random.default_rng(1)
    shp = json.loads(C.SHAPE_JSON.read_text())
    for emb in C.EMB:
        for fr in C.FR:
            y, x, src = C.load_cell(emb, fr)
            des = Band(y, x, src)
            W = des.boot_counts(rng, 20)
            for df in DFS:
                f = des.curves(W, y, df)
                for i in range(20):
                    b = np.repeat(np.arange(des.n), W[i].astype(int))
                    ref = S.spline_curve(y[b], x[b], src[b], des.fit[df]["kn"], des.at)
                    assert np.max(np.abs(f[i] - ref)) < 1e-9, (df, np.max(np.abs(f[i] - ref)))
                    # contrast form of H_peak == the existing max-minus-max
                    href = ref[:NG].max() - max(ref[NG], ref[NG + 1])
                    assert abs(contrasts(ref[None]).min(2).max() - href) < 1e-12
            one = np.ones((1, des.n))
            pt = shp[emb]["shape"][fr]["point"]
            for df in (4, 6):
                h0 = float(hpeak(des.curves(one, y, df))[0])
                assert abs(h0 - pt[f"spline{df}_H_peak"]) < 1e-9, (df, h0, pt[f"spline{df}_H_peak"])
            print("equivalent (curves df 4/6/8 vs spline_curve; point H_peak df 4/6 vs saved JSON):", emb, fr,
                  flush=True)
    # the old simulation's estimands and its percentile U on its own first dataset (same seed, B = 1,000, chunk 125)
    old = json.loads(COV_RAW.read_text())
    old = {(r["emb"], r["fr"], r["name"]): r for r in old}
    otasks, _ = C.build_tasks(n_sim=1)
    for t in otasks:
        o = old[(t[0], t[1], t[2])]
        r = simulate(t, b_in=1000, chunk=125, b_small=0, write=False) if t[2] in ("flat", "narrow_hU", "het_hU") \
            else None
        y, x, src = C.load_cell(t[0], t[1])
        des = Band(y, x, src)
        f = C.make_curve(t[3], t[4], t[5])
        f_obs = f(x, des.si) if t[3] == "het" else f(x)
        hp = float(hpeak(des.curves(np.ones((1, des.n)), f_obs, 4))[0])
        assert abs(hp - o["H_proj"]) < 1e-9, (t[:3], hp, o["H_proj"])
        if r is not None:
            assert abs(r["H_true"] - o["H_true"]) < 1e-12 and abs(r["H_true_grid"] - o["H_true_grid"]) < 1e-12
            assert abs(r["H_proj"]["4"] - o["H_proj"]) < 1e-9
            got = r["rec"][0]["4"]["B1000"]
            assert abs(r["rec"][0]["4"]["H_hat"] - o["rec"]["H_hat"][0]) < 1e-9
            assert abs(got["U_pct"] - o["rec"]["U"][0]) < 1e-9, (t[:3], got["U_pct"], o["rec"]["U"][0])
            assert abs(got["L_pct"] - o["rec"]["L"][0]) < 1e-9
            print(f"reproduces old simulation dataset 0 ({t[0]} {t[1]} {t[2]}): H_hat, U_pct, L_pct, H_true, H_proj",
                  flush=True)
    print("H_proj (df 4) reproduces the old simulation in all 56 scenario-cells", flush=True)
    # band sanity on a pure-noise toy: U_band >= the contrast-max of the point curve, c >= one-sided z
    tasks, _ = build_tasks(n_sim=N_SIM)
    assert len(tasks) == 56 and len({t[6] for t in tasks}) == 56
    nt = 10
    t0 = time.time()
    r = simulate(tasks[0][:7] + (nt,), write=False)
    dt = (time.time() - t0) / nt
    for d in r["rec"]:
        for df in map(str, DFS):
            v = d[df][f"B{B_IN}"]
            assert v["U_band"] >= d[df]["H_hat"] and v["c_band"] > 1.64, v
    print(f"timing {tasks[0][0]} {tasks[0][1]} {tasks[0][2]}: {dt:.2f} s per simulated dataset "
          f"(B_IN={B_IN}, dfs {DFS}, 1 thread); one scenario-cell of {N_SIM} = {dt * N_SIM / 60:.1f} min; "
          f"projected wall hours, {len(tasks)} scenario-cells on {WORKERS} workers: "
          f"{dt * N_SIM * len(tasks) / WORKERS / 3600:.2f}", flush=True)
    print(f"familywise Wilson z = {Z_FAMILY:.3f}", flush=True)


def main():
    global _LOG
    RAW.mkdir(parents=True, exist_ok=True)
    _LOG = RAW / "run.log"
    log(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} N_SIM={N_SIM} B_IN={B_IN} dfs={DFS} workers={WORKERS}")
    if not OUT_OBS.exists():
        observed()
    tasks, meta = build_tasks()
    t0 = time.time()
    results = []
    with get_context("fork").Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(simulate, tasks)):
            results.append(r)
            sm = summarise(r)["df4"][f"B{B_IN}"]
            log(f"[{i + 1}/{len(tasks)} {time.time() - t0:.0f}s] {r['emb']} {r['fr']} {r['name']}: "
                f"H_true={r['H_true']:.5f} H_proj4={r['H_proj']['4']:.5f} "
                f"band cov true={sm['U_band']['coverage_H_true']['share']:.3f} "
                f"proj={sm['U_band']['coverage_H_proj']['share']:.3f} | "
                f"pct cov true={sm['U_pct']['coverage_H_true']['share']:.3f} "
                f"proj={sm['U_pct']['coverage_H_proj']['share']:.3f}")
    res = {"prereg": "docs/prereg/strengthen_shape_band.md", "seed": SEED, "n_sim": N_SIM, "B_inner": B_IN,
           "B_small": B_SMALL, "dfs": list(DFS), "level": LEVEL, "z_familywise": Z_FAMILY, "workers": WORKERS,
           "cells": meta, "wall_seconds": time.time() - t0, "results": {}}
    for r in sorted(results, key=lambda r: (r["emb"], r["fr"], r["name"])):
        res["results"].setdefault(f"{r['emb']}/{r['fr']}", {})[r["name"]] = summarise(r)
    OUT.write_text(json.dumps(res, indent=1))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
