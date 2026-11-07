# Pre-registration: a calibrated upper bound on the inspiration "middle optimum" (Section 6)

Written 2026-11-05, committed before `scripts/strengthen_shape_band.py` is run for results. Only its equivalence and
timing check (`--check`) has been run: it computes no coverage and no bound on the observed data. It confirms that the
batched spline fits reproduce `strengthen_inspiration_shape.spline_curve` for 4, 6 and 8 df (to 1e-9, 20 resamples per
cell), that the full-sample H_peak reproduces the saved 4- and 6-df point estimates, and that with the old seed and
inner B = 1,000 it reproduces the first simulated dataset of `strengthen_shape_coverage` (H_hat, U, L, H_true, H_proj)
in 12 scenario-cells, and H_proj in all 56. Simulation and resampling only: no generation, no embedding model, no API.

## Why

The paper (Section 6, "No sweet spot was detected") reports U, the 95th percentile of 4,000 bootstrap values of
H_peak (max over distance ranks 0.10-0.90 of a source-common 4-df natural spline, minus the higher endpoint). The
pre-registered simulation `strengthen_shape_coverage` showed that U is not a calibrated bound: it covers the peak of
the fitted spline itself (H_proj) in only 0.85-0.89 of datasets under broad humps, and a true hump of width 0.15 in
0.20-0.60. A percentile of a max-minus-max statistic has no coverage guarantee. Here I replace it by the projection of
a simultaneous confidence set onto the peak functional, which has one by construction for H_proj, and measure its
coverage of H_proj and of the true peak H_true in the same scenarios, including the failed ones.

## Procedure (one dataset, one spline df)

Same data, same estimand, same bootstrap as `strengthen_inspiration_shape.py`: y ~ source fixed effects + natural
spline in the within-source distance rank x (knots at quantiles of x), stimuli resampled with replacement within
source, B = 4,000.

1. Contrasts D(t, e) = f(t) − f(e) of the common curve, t on the grid 0.10, 0.11, …, 0.90 (81 points), e ∈ {x_min,
   x_max}: 162 contrasts. H_peak = max_t min_e D(t, e), identical to the existing max-minus-max (checked).
2. σ(t, e) = bootstrap SD of each contrast. c = 95th percentile over resamples of max_(t,e) (D̂ − D*)/σ (one-sided
   sup-t).
3. **U_band = max_t min_e (D̂(t, e) + c·σ(t, e)).** If every contrast lies below its band, which happens with
   probability ≈ 0.95, then H_peak ≤ U_band (the functional is monotone in each contrast). So U_band is a conservative
   95 % upper bound on the peak of whatever curve D̂ estimates (the projection of the truth on the spline space).

Secondary, the curve-band form: a two-sided 95 % sup-t band on the curve values (basis normalisation f(0) = 0),
U_curveband = max_t upper(t) − max(lower(x_min), lower(x_max)). Reference: the old U_pct (95th percentile of H_peak*).

Spline df: **4 (primary, the paper's curve)**; 6 and 8 as sensitivity (knots at 0, 1/6, …, 1 and 0, 1/8, …, 1), to see
whether a more flexible curve reduces the misspecification that loses narrow humps. All three use the same resamples.
Each bound is also computed from the first 1,000 of the same 4,000 resamples (inner B = 1,000 vs 4,000 on shared
simulated datasets, in every scenario-cell).

## Scenarios

Exactly the 14 scenarios × 4 cells (Qwen3, BGE-M3 × INCIDENTAL, INSTRUCTED) of
`docs/prereg/strengthen_shape_coverage.md`, same data-generating process (fitted source FE + f(x) + residuals of the
4-df fit resampled within source), same curves and heights (h = the cell's old observed U and h = half the cell's
per-idea SD): flat, linear_qwen3_slope, and broad, narrow, between, spike, het, rise_narrow at both heights.
Estimands as there: H_true (continuous sup over [0.10, 0.90] of the population common curve minus the higher end),
H_true_grid, H_proj (now per df: H_peak of that df's spline fitted to the noiseless curve on the real design), and for
het H_source_specific_max.

**1,000 simulated datasets per scenario-cell**, as Codex suggested. Timing check: 0.93 s per dataset (B = 4,000, the
three dfs, one thread), so 56 scenario-cells take about 3.6 h on 4 single-thread workers. Seed 20261108 (per-task seeds
spawned in task order); observed-data bootstrap seed 20261109.

## Outputs

Per scenario-cell, df and B: coverage (share of datasets with estimand ≤ bound) of H_true, H_true_grid, H_proj (and
H_source_specific_max for het), each with a Wilson 95 % interval and a familywise Wilson interval (z = 3.32,
Bonferroni over 56 scenario-cells); mean and 5/50/95 % quantiles of each bound; mean c; mean, mean absolute and SD of
the B = 1,000 minus B = 4,000 difference for U_band and U_pct. Observed data (same process, run first):
U_band, U_curveband and U_pct for each df and cell, in margin units, per-idea SDs and ranges of source means, with the
existing reading rule (meaningful optimum excluded if the bound is < 0.2 per-idea SD and < 0.25 of the range of source
means). Files: `docs/prereg/strengthen_shape_band_observed.json`, `docs/prereg/strengthen_shape_band.json`,
per-task records and `run.log` under `~/Documents/experiment-data/latent-gap-ideas/runs/shape_band/`.

## Success criteria (fixed now) and what the paper may say

A scenario-cell **fails** for an estimand if the familywise Wilson upper end of the coverage is below 0.95 (with
1,000 datasets: observed coverage below ≈ 0.927). The plain 95 % Wilson verdict (calibrated / conservative /
anti-conservative) is reported too.

- **P (projection):** U_band (4 df, B = 4,000) does not fail for H_proj in any of the 56 scenario-cells.
- **T (truth):** U_band (4 df, B = 4,000) does not fail for H_true in flat, linear_qwen3_slope, broad (both heights)
  and het (both heights) in all four cells (24 scenario-cells).

Readings:

1. **P and T met:** the paper replaces the percentile by the observed U_band as a 95 % upper bound, calibrated by
   simulation, on the peak of the source-common 4-df curve and on a true broad or source-averaged hump, stated with
   the reading rule above; it still reports the coverage of H_true under width-0.15, between-grid, spike and
   rise-narrow humps as the limit of the statement (narrower peaks are not bounded unless reading 4 applies).
2. **P met, T not:** U_band is stated only as a bound on the peak of the fitted 4-df curve; the paper says a true
   broad or heterogeneous hump is not bounded, with the failing coverages.
3. **P not met:** no calibrated bound is claimed; the paper keeps "not a calibrated bound" and reports U_band as a
   sensitivity with its coverage.
4. **Flexible curves (sensitivity):** if U_band with 6 or 8 df meets P (for its own H_proj) in all 56 scenario-cells
   and T extended to narrow, between and rise_narrow at both heights (48 scenario-cells), the paper may also quote the
   observed U_band of the smallest such df as a bound for humps at least 0.15 wide (full width at half maximum). The
   spike (width 0.05) is a stress test; no reading claims a bound for it.
5. **Inner B:** if the B = 1,000 and B = 4,000 coverages of U_band and U_pct differ by at most 0.01 in every
   scenario-cell, the paper may say the earlier B = 1,000 simulation represents the B = 4,000 percentile; otherwise it
   reports the difference. No other decision depends on B.

Power is not tested; the bound's width (mean U_band relative to H_true and to the per-idea SD) is reported.
