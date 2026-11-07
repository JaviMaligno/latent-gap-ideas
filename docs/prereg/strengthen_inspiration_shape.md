# Pre-registration: selection-aware bound on a middle optimum (Section 6)

Written 2026-11-05, committed before `scripts/strengthen_inspiration_shape.py` is run. Reanalysis of existing
gpt-4o data (no new generation, no embedding model): `runs/inspiration/gpt-4o-0806/scored_{qwen3e06,bgem3}.parquet`.
The data have been seen (Table 5, `revision_inspiration.json`); what is fixed here is the procedure and the reading
rule, so that the bound is not tuned after seeing it.

## Why

The two-lines test (Simonsohn 2018) chooses its breakpoint from the data, so its segment CIs are not selection-aware
and the paper could not say how large a middle optimum the data exclude. This analysis bootstraps the whole procedure.

## Data and variables (same estimands as P1 in the paper)

Per embedder (Qwen3, BGE-M3) and framing (INCIDENTAL, INSTRUCTED): the 1,800 stimulus ideas; y = margin; x = the
stimulus's within-source percentile rank of the embedder's distance among the drawn stimuli (as P1 and the two-lines
test in `revision_inspiration.py`); source fixed effects. The x grid is identical in every source (200 ranks each).

## Bootstrap

B = 4,000 resamples per embedder × framing, seed 20261105. Each resample draws 200 stimuli with replacement within
each of the nine sources (stratified; a stimulus keeps its x), and re-runs every step:

1. **Two-lines**: the exact algorithm of `inspiration_analysis.two_lines` (quadratic peak, clip to the 10th-90th
   percentile, Robin Hood re-split). Record breakpoint, both segment slopes, and the **fall after the breakpoint**,
   F = −slope_high × (x_max − breakpoint) (positive = the curve falls). Also the rise below,
   R = slope_low × (breakpoint − x_min), and whether the resample would declare a hump.
2. **Smooth curve (primary)**: y ~ source FE + natural cubic spline in x with 4 df (boundary knots at min/max of x,
   interior knots at the 25th, 50th and 75th percentiles of x, fixed from the original x). Evaluated on the grid
   g = 0.10, 0.11, …, 0.90 and at the endpoints x_min, x_max.
   - **H_peak** (primary) = max_g f(g) − max(f(x_min), f(x_max)): how far an interior point rises above the *higher*
     end. This is the height of an inverted U; a monotone curve gives H_peak ≤ 0.
   - **H_avg** (secondary, as requested) = max_g f(g) − (f(x_min) + f(x_max)) / 2. A monotone trend also makes it
     positive, so it is not by itself evidence of a middle optimum.
3. **Sensitivity**: the same two heights from a local-linear smoother (Gaussian kernel, bandwidth 0.15) on
   source-demeaned y; and from the spline with 3 and 6 df.

Intervals: percentile, 95 % two-sided for F, R and curve values; the **upper bound U** on a height is the 95th
percentile of its bootstrap distribution (one-sided 95 %). The bootstrap of a maximum is biased upward, so U is
conservative for a bound.

## Scales and reading rule

- per-idea SD: SD of the margin within source (pooled residual SD of y ~ source FE), per embedder × framing;
- spread between sources: SD and range of the nine source means.

Reading rule, fixed now: in an embedder × framing cell, **the data exclude a meaningful middle optimum** if
U(H_peak, primary spline) < 0.2 × per-idea SD (a stimulus at the best interior distance would beat the better end by
less than a small standardised effect) **and** U < one quarter of the range of source means. The paper may then state
"a middle optimum higher than U (margin units) is excluded" for that cell. If U ≥ 0.2 SD in a cell, the paper keeps
"not detected" for it. The claim for an embedder needs both framings; "in both spaces" needs all four cells. The
fall after the breakpoint is read as excluded beyond the upper end of its 95 % CI.

## Exploratory, not tested

The shape of BGE-M3 INCIDENTAL (rise then flattening): spline values relative to f(x_min) on 0, 0.1, …, 1 with
pointwise 95 % bands, the rise f(0.2) − f(x_min) and the change f(x_max) − f(0.2), and the bootstrap share of
resamples whose spline maximum is interior (0.1-0.9). And a power calculation (not a test) for the anchoring
comparison steam_game INSTRUCTED vs NONE.
