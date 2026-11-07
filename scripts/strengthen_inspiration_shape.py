"""Selection-aware inference on the shape of margin vs distance (Section 6) and power for the anchoring comparison.

Pre-registered in docs/prereg/strengthen_inspiration_shape.md (committed before this was run). Reuses the saved
gpt-4o scored tables; no generation and no embedding model.
    OMP_NUM_THREADS=2 uv run python scripts/strengthen_inspiration_shape.py
Output: docs/prereg/strengthen_inspiration_shape.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/prereg/strengthen_inspiration_shape.json"
SEED, B = 20261105, 4000
FR = ("INCIDENTAL", "INSTRUCTED")
GRID = np.round(np.arange(0.10, 0.9001, 0.01), 2)
SHAPE_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 1)
BW = 0.15


def ns_basis(x, knots):
    """Natural cubic spline basis (ESL eq. 5.4-5.5) without the constant: x and K-2 truncated terms."""
    k = np.asarray(knots, float)
    K = len(k)

    def d(j):
        return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    cols = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def spline_curve(y, x, src, knots, at):
    """Fit y ~ source FE + ns(x); return the curve f at `at` (up to a constant)."""
    X = np.column_stack([ns_basis(x, knots), A.dummies(src, A.SOURCES)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = len(knots) - 1
    return ns_basis(at, knots) @ b[:p]


def demean(y, src):
    out = y.copy()
    for s in A.SOURCES:
        m = src == s
        out[m] -= y[m].mean()
    return out


def loclin(y, x, at, h=BW):
    """Local-linear (Gaussian kernel) fit of y on x evaluated at `at`."""
    u = x[None, :] - at[:, None]
    w = np.exp(-0.5 * (u / h) ** 2)
    s0, s1, s2 = w.sum(1), (w * u).sum(1), (w * u * u).sum(1)
    t0, t1 = (w * y).sum(1), (w * u * y).sum(1)
    return (s2 * t0 - s1 * t1) / (s0 * s2 - s1 ** 2)


def heights(f_grid, f_lo, f_hi):
    top = f_grid.max()
    return {"H_peak": float(top - max(f_lo, f_hi)), "H_avg": float(top - (f_lo + f_hi) / 2),
            "argmax": float(GRID[int(f_grid.argmax())])}


def run_once(y, x, src, xmin, xmax, knots_by_df):
    """Every step of the procedure on one (re)sample."""
    tl = A.two_lines(y, x, src)
    out = {"breakpoint": tl["breakpoint"], "slope_low": tl["slope_low"], "slope_high": tl["slope_high"],
           "fall_after": -tl["slope_high"] * (xmax - tl["breakpoint"]),
           "rise_before": tl["slope_low"] * (tl["breakpoint"] - xmin), "hump": tl["hump"]}
    at = np.concatenate([GRID, [xmin, xmax], SHAPE_GRID])
    for df, kn in knots_by_df.items():
        f = spline_curve(y, x, src, kn, at)
        h = heights(f[:len(GRID)], f[len(GRID)], f[len(GRID) + 1])
        out.update({f"spline{df}_{k}": v for k, v in h.items()})
        if df == 4:
            fs = f[len(GRID) + 2:]
            fs = np.concatenate([[f[len(GRID)]], fs[1:-1], [f[len(GRID) + 1]]])  # ends at x_min / x_max
            out["shape"] = (fs - fs[0]).tolist()
            # true argmax over the full fitted range, to say whether the maximum is interior
            fine = np.linspace(xmin, xmax, 401)
            ff = spline_curve(y, x, src, kn, fine)
            out["spline4_argmax_full"] = float(fine[int(ff.argmax())])
    fl = loclin(demean(y, src), x, at[:len(GRID) + 2])
    h = heights(fl[:len(GRID)], fl[len(GRID)], fl[len(GRID) + 1])
    out.update({f"loclin_{k}": v for k, v in h.items()})
    return out


def pct(a, q):
    return float(np.percentile(np.asarray(a, float), q))


def summarise(obs, boots, sd_idea, src_means):
    sh = lambda k: [b[k] for b in boots]  # noqa: E731
    res = {"point": {k: v for k, v in obs.items() if k != "shape"},
           "scales": {"sd_idea_within_source": sd_idea, "sd_source_means": float(np.std(src_means, ddof=1)),
                      "range_source_means": float(np.ptp(src_means))}}
    res["two_lines_selection_aware"] = {
        "breakpoint_ci95": [pct(sh("breakpoint"), 2.5), pct(sh("breakpoint"), 97.5)],
        "fall_after_breakpoint": {"est": obs["fall_after"], "ci95": [pct(sh("fall_after"), 2.5),
                                                                    pct(sh("fall_after"), 97.5)],
                                  "upper95_one_sided": pct(sh("fall_after"), 95)},
        "rise_before_breakpoint": {"est": obs["rise_before"], "ci95": [pct(sh("rise_before"), 2.5),
                                                                       pct(sh("rise_before"), 97.5)]},
        "slope_high_ci95": [pct(sh("slope_high"), 2.5), pct(sh("slope_high"), 97.5)],
        "slope_low_ci95": [pct(sh("slope_low"), 2.5), pct(sh("slope_low"), 97.5)],
        "share_resamples_declaring_hump": float(np.mean(sh("hump")))}
    bounds = {}
    for m in ("spline4", "spline3", "spline6", "loclin"):
        for h in ("H_peak", "H_avg"):
            v = np.array(sh(f"{m}_{h}"))
            bounds[f"{m}_{h}"] = {"est": obs[f"{m}_{h}"], "upper95": pct(v, 95), "ci90": [pct(v, 5), pct(v, 95)],
                                  "share_positive": float((v > 0).mean())}
    res["peak_bounds"] = bounds
    U = bounds["spline4_H_peak"]["upper95"]
    res["reading_rule"] = {"U_spline4_H_peak": U, "U_over_sd_idea": U / sd_idea,
                           "U_over_range_source_means": U / res["scales"]["range_source_means"],
                           "U_over_sd_source_means": U / res["scales"]["sd_source_means"],
                           "meaningful_optimum_excluded": bool(U < 0.2 * sd_idea and
                                                               U < 0.25 * res["scales"]["range_source_means"])}
    S = np.array([b["shape"] for b in boots])
    res["shape_spline4_rel_to_xmin"] = {"x": SHAPE_GRID.tolist(), "est": obs["shape"],
                                        "lo95": np.percentile(S, 2.5, 0).tolist(),
                                        "hi95": np.percentile(S, 97.5, 0).tolist()}
    i2 = 2
    rise = S[:, i2]
    after = S[:, -1] - S[:, i2]
    res["saturation"] = {"rise_xmin_to_0.2": {"est": obs["shape"][i2], "ci95": [pct(rise, 2.5), pct(rise, 97.5)]},
                         "change_0.2_to_xmax": {"est": obs["shape"][-1] - obs["shape"][i2],
                                                "ci95": [pct(after, 2.5), pct(after, 97.5)]},
                         "argmax_full_ci95": [pct(sh("spline4_argmax_full"), 2.5),
                                              pct(sh("spline4_argmax_full"), 97.5)],
                         "share_argmax_interior_0.1_0.9": float(np.mean([(0.1 <= a <= 0.9)
                                                                         for a in sh("spline4_argmax_full")]))}
    return res


def shape_analysis(d: pd.DataFrame, rng) -> dict:
    out = {}
    for fr in FR:
        g = d[d["arm"] == fr].reset_index(drop=True)
        y, src = g["margin"].to_numpy(float), g["source"].to_numpy()
        x = g.groupby("source")["dist"].rank(pct=True).to_numpy()
        xmin, xmax = float(x.min()), float(x.max())
        knots = {3: np.quantile(x, [0, 1 / 3, 2 / 3, 1]), 4: np.quantile(x, [0, .25, .5, .75, 1]),
                 6: np.quantile(x, [0, 1 / 6, 2 / 6, .5, 4 / 6, 5 / 6, 1])}
        obs = run_once(y, x, src, xmin, xmax, knots)
        idx = {s: np.flatnonzero(src == s) for s in A.SOURCES}
        boots = []
        for _ in range(B):
            b = np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])
            boots.append(run_once(y[b], x[b], src[b], xmin, xmax, knots))
        resid = y - pd.Series(y).groupby(src).transform("mean").to_numpy()
        sd_idea = float(np.sqrt((resid ** 2).sum() / (len(y) - len(A.SOURCES))))
        src_means = pd.Series(y).groupby(src).mean().to_numpy()
        out[fr] = summarise(obs, boots, sd_idea, src_means)
        print(fr, json.dumps(out[fr]["reading_rule"]), json.dumps(out[fr]["two_lines_selection_aware"]
                                                                    ["fall_after_breakpoint"]), flush=True)
    return out


def power(d: pd.DataFrame, rng) -> dict:
    """Power (not a test) for steam_game INSTRUCTED vs NONE, two-sided."""
    s = d[(d["arm"] == "INSTRUCTED") & (d["source"] == "steam_game")]["margin"].to_numpy(float)
    n = d[d["arm"] == "NONE"]["margin"].to_numpy(float)
    delta, ss, sn = s.mean() - n.mean(), s.std(ddof=1), n.std(ddof=1)
    bd = [rng.choice(s, len(s)).mean() - rng.choice(n, len(n)).mean() for _ in range(4000)]
    res = {"diff": float(delta), "n_steam": len(s), "n_none": len(n), "sd_steam": float(ss), "sd_none": float(sn),
           "diff_boot_ci95": [pct(bd, 2.5), pct(bd, 97.5)],
           "p_welch_now": float(stats.ttest_ind(s, n, equal_var=False).pvalue),
           "p_mw_now": float(stats.mannwhitneyu(s, n).pvalue)}
    for name, alpha in (("alpha_0.05", 0.05), ("alpha_0.05_over_9_holm_worst", 0.05 / 9)):
        zz = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)) ** 2
        need = zz / delta ** 2  # required 1 / (ss^2/ns + sn^2/nn)
        n_eq = int(np.ceil(zz * (ss ** 2 + sn ** 2) / delta ** 2))
        r = {"equal_n_per_arm": n_eq, "extra_each_if_equal": max(0, n_eq - 200)}
        rem = 1 / need - ss ** 2 / 200
        r["extra_none_only_steam_fixed_200"] = (int(np.ceil(sn ** 2 / rem)) - 200) if rem > 0 else "infeasible"
        rem2 = 1 / need - sn ** 2 / 200
        r["extra_steam_only_none_fixed_200"] = (int(np.ceil(ss ** 2 / rem2)) - 200) if rem2 > 0 else "infeasible"
        # cheapest total: optimal allocation n_s/n_n = ss/sn
        tot = zz * (ss + sn) ** 2 / delta ** 2
        r["optimal_alloc"] = {"n_steam": int(np.ceil(tot * ss / (ss + sn))),
                              "n_none": int(np.ceil(tot * sn / (ss + sn)))}
        # Monte Carlo check with Mann-Whitney at equal n, resampling the observed distributions
        hits = [stats.mannwhitneyu(rng.choice(s, n_eq), rng.choice(n, n_eq)).pvalue < alpha for _ in range(2000)]
        hw = [stats.ttest_ind(rng.choice(s, n_eq), rng.choice(n, n_eq), equal_var=False).pvalue < alpha
              for _ in range(2000)]
        r["mc_power_at_equal_n"] = {"mann_whitney": float(np.mean(hits)), "welch": float(np.mean(hw))}
        res[name] = r
    # sensitivity to the true effect: equal n per arm for 80 % power at alpha 0.05
    zz = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2
    res["equal_n_by_true_diff"] = {f"{dd:+.4f}": int(np.ceil(zz * (ss ** 2 + sn ** 2) / dd ** 2))
                                   for dd in (-0.002, -0.003, -0.004, -0.005, -0.006, -0.008)}
    res["mde_now_80pct_alpha_0.05"] = float(np.sqrt(zz * (ss ** 2 / 200 + sn ** 2 / 200)))
    return res


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"prereg": "docs/prereg/strengthen_inspiration_shape.md", "seed": SEED, "B": B, "grid": [0.1, 0.9],
           "loclin_bandwidth": BW}
    rev = json.loads((Path(__file__).resolve().parents[1] / "docs/prereg/revision_inspiration.json").read_text())["reproduction_full"]
    for emb in ("qwen3e06", "bgem3"):
        d = pd.read_parquet(A.RUN / f"scored_{emb}.parquet")
        # check: the two-lines step reproduces the paper's slopes on the original data
        for fr in FR:
            g = d[d["arm"] == fr]
            x = g.groupby("source")["dist"].rank(pct=True).to_numpy()
            tl = A.two_lines(g["margin"].to_numpy(float), x, g["source"].to_numpy())
            ref = rev[emb][fr]["two_lines"]
            assert abs(tl["slope_high"] - ref["slope_high"]) < 1e-6 and abs(tl["breakpoint"] - ref["breakpoint"]) < 1e-9
        print(emb, flush=True)
        res[emb] = {"shape": shape_analysis(d, rng), "power_steam_instructed_vs_none": power(d, rng)}
        print(json.dumps(res[emb]["power_steam_instructed_vs_none"], indent=1), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__" and "--mw-power" not in sys.argv:
    main()


def mw_n_search() -> dict:
    """Added after the main run: equal n per arm giving 80 % Mann-Whitney power (the paper's test), by Monte Carlo
    resampling of the observed steam_game INSTRUCTED and NONE margins. Updates the JSON in place."""
    rng = np.random.default_rng(SEED + 7)
    res = json.loads(OUT.read_text())
    for emb in ("qwen3e06", "bgem3"):
        d = pd.read_parquet(A.RUN / f"scored_{emb}.parquet")
        s = d[(d["arm"] == "INSTRUCTED") & (d["source"] == "steam_game")]["margin"].to_numpy(float)
        n = d[d["arm"] == "NONE"]["margin"].to_numpy(float)
        tab = {}
        for alpha in (0.05, 0.05 / 9):
            row = {}
            for nn in (500, 600, 700, 800, 900, 1000, 1200, 1400):
                row[nn] = float(np.mean([stats.mannwhitneyu(rng.choice(s, nn), rng.choice(n, nn)).pvalue < alpha
                                         for _ in range(1000)]))
            tab[f"alpha_{alpha:.4f}"] = row
        res[emb]["power_steam_instructed_vs_none"]["mw_power_by_equal_n"] = tab
        print(emb, tab, flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__" and "--mw-power" in sys.argv:
    mw_n_search()
