# Pre-registration: coverage of the selection-aware shape bound (Section 6)

Written 2026-11-05, committed before `scripts/strengthen_shape_coverage.py` is run for results. Only an
equivalence-and-timing check (`--check`, which writes nothing and computes no coverage) has been run: it confirms that
the batched bootstrap reproduces `strengthen_inspiration_shape.py` (spline H_peak and every two-lines step, to 1e-9,
on 20 resamples per cell and on the saved point estimates). Simulation only; no generation, no embedding model, no API.

## Why

`strengthen_inspiration_shape.json` gives U, the one-sided 95 % bootstrap upper bound (95th percentile of 4,000
within-source stimulus resamples) of H_peak = max over ranks 0.10-0.90 of a source-common 4-df natural spline minus the
higher of its two endpoint values. The paper reports U = 0.004-0.007 (0.18-0.31 per-idea SD) but says the bootstrap's
coverage for this max-minus-max statistic has not been checked. A percentile bootstrap of a maximum can be
anti-conservative (when the curve is flat or tied the bootstrap distribution of a max is shifted) or conservative; and
a 4-df spline cannot follow a narrow or source-specific peak. This simulation measures which, per scenario.

## Data-generating processes (built from the real design)

For each of the 4 cells (Qwen3, BGE-M3 × INCIDENTAL, INSTRUCTED), n = 1,800 (200 stimuli in each of 9 sources), the
x of every stimulus is its real within-source distance rank (as in the shape analysis). Fit y ~ source FE + 4-df
natural spline (same knots) on the real margins; keep the fitted source FE and the residuals. A simulated dataset is
y* = FE_source + f(x) + e*, with e* drawn with replacement from the residuals of the same source. Scenarios f
(x in [0, 1]; "bump(c, w)" = height-1 raised cosine centred at c with full width at half maximum w, support 2w):

| name | f(x) |
|---|---|
| flat | 0 (endpoints tied, true H_peak = 0) |
| linear_qwen3_slope | s·x, s = observed Qwen3 P1 slope of the same framing (0.00381 INCIDENTAL, 0.00527 INSTRUCTED), in all 4 cells |
| broad_h | h·sin²(πx) (peak 0.5, FWHM 0.5; endpoints tied at ≈ 0) |
| narrow_h | h·bump(0.60, 0.15) |
| between_h | h·bump(0.4575, 0.15): peak between grid points (grid step 0.01) and between data ranks (step 0.005) |
| spike_h | h·bump(0.4575, 0.05): very narrow, between grid points (stress test) |
| het_h | source k has h·bump(c_k, 0.15), c_k = 0.20, 0.275, …, 0.80 (one per source, in the order of `SOURCES`) |
| rise_narrow_h | s·x + h·bump(0.60, 0.15): hump on a rising trend (the comparator is the high end) |

Each hump scenario is run at two heights: h = the cell's observed U (0.00453, 0.00736, 0.00399, 0.00498) and
h = 0.5 × the cell's per-idea SD. 14 scenarios × 4 cells = 56 scenario-cells.

## Procedure on each simulated dataset

Exactly the shape bootstrap's primary procedure: B_inner within-source stimulus resamples; 4-df spline H_peak on the
grid 0.10, 0.11, …, 0.90 → U = 95th percentile, L = 5th percentile; and the full two-lines algorithm (breakpoint
re-selected in each resample) → the fall after the breakpoint F, its 95 % percentile CI and whether the point
estimate declares a hump. Only the 4-df spline is simulated (not 3/6 df or local-linear).

## Estimands

- **H_true** (primary): sup over continuous x in [0.10, 0.90] (step 0.0005) of the population common curve, minus
  max(f(x_min), f(x_max)); for het, the population common curve is the mean of the nine source curves (equal n and
  identical ranks per source, which is what a source-FE common spline targets).
- H_true_grid: the same on the 0.01 grid. H_proj: H_peak of the 4-df spline fitted to the noiseless curve on the real
  design (the "fitted curve" the paper's sentence scopes the bound to).
- het only: H_source_specific_max = the largest single-source H_peak (= h); is the common bound informative about it?
- Two-lines: F_true(bp̂) = population fall at the breakpoint the dataset selected (two-lines design fitted to the
  noiseless curve at that breakpoint); and max over breakpoints in the 10th-90th percentile of F_true.

## Outputs and reading rule

Per scenario-cell: coverage of U = share of datasets with H_true ≤ U (and the same for H_true_grid, H_proj),
target ≥ 0.95, with a Wilson 95 % interval; power = share with L > 0; mean Ĥ, SD, mean U, mean U − H_true; coverage
of the two-sided fall CI for F_true(bp̂), of its upper end for F_true(bp̂) and for max F_true; share declaring a hump.

Verdict per scenario-cell, fixed now: **calibrated** if the Wilson interval of coverage contains 0.95;
**conservative** if its lower end is above 0.95; **anti-conservative** if its upper end is below 0.95. The paper's
bound statement is supported as stated if U is calibrated or conservative for H_proj and H_true in flat,
linear, broad, between and rise_narrow at both heights in all four cells; narrow, spike and het are read as limits
of the statement (they test the scope sentence "not narrower or source-specific peaks"). Power is reported, not
tested. With 500 datasets the Monte Carlo SE of a coverage near 0.95 is ≈ 0.010.

## Sizes (deviation from the reviewer's suggestion)

Codex suggested 1,000 simulated datasets. I use **500 simulated datasets per scenario-cell and inner B = 1,000**
(the paper's bound used B = 4,000), so that the 56 scenario-cells fit in about 2.5 hours on 4 single-thread worker
processes (timing check: ≈ 0.6 s per dataset at B = 500). Inner B = 1,000 makes the 95th percentile noisier than with
4,000, which can only widen the spread of U, not shift its expected coverage materially. Seed 20261106
(per-task seeds spawned from it in task order).

Outputs: `docs/prereg/strengthen_shape_coverage.json` (summaries) and per-dataset records under
`~/Documents/experiment-data/latent-gap-ideas/runs/shape_coverage/`.
