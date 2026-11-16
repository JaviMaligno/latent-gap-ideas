# Review follow-up: does F5 (isolation vs reception of new items) survive controls, and are both tails reported?

Registered 2026-11-15, before the script `scripts/review_f5_controls.py` is written or run. Deviations will be
disclosed in the result section appended below. Local CPU only: no LLM, no gateway, no GPU, no paid call, no GCP.

## Review concerns addressed

| Concern | What this registration does |
|---|---|
| F5 "more isolated new Steam games are slightly better reviewed" is uncontrolled | C1: partial Spearman controlling for review volume, release year, owners bracket and primary genre (Steam); filing year and CPC section (patents). C2: Steam review-count strata. |
| Patents tails reported asymmetrically (only the top 1 % most isolated) | C3: top and bottom 1 % and 10 % by isolation against the rest, both domains, both embedders, with CIs |

## What is known before this registration (disclosure)

Registered F5 (`where_future_lands_result.json`): Steam ρ = +0.107 (Qwen3) / +0.132 (BGE-M3), n = 6,697; patents
ρ = −0.081 / −0.084, n = 9,477; top 1 % most isolated mean quality Steam 0.842 vs rest 0.840 (Qwen3), 0.851 vs
0.840 (BGE-M3); patents 0.800 vs 0.842, 0.768 vs 0.843. For feasibility I looked at the marginal distributions of the
covariates in the F5 samples (no association with isolation or quality was computed): Steam review count 20-99:
3,682, 100-999: 2,224, ≥ 1000: 791; release year 2024: 4,524, 2025: 2,173; owners bracket "0 - 20000" 4,664, then
885, 446, 319, 225, ... down to single digits in the top brackets; first genre Action 2,876, Adventure 1,508,
Casual 1,042, Indie 769, Simulation 180, RPG 125, Strategy 109, then < 30 each (7 empty). Patents: filing year 2016:
7,497, 2017: 1,980; CPC section H 3,304, G 2,349, A 1,153, B 1,119, C 786, F 492, E 235, D 39, no empty CPC in the
decided set; grant rates by section range from 0.73 (D, A) to 0.89 (H).

## Data and samples

Same loading, adult filter, normalisation, held-out draw and reference set R as `scripts/where_future_lands.py`
(SEED 20261030, held-out = first draw of `default_rng(SEED)`); isolation iso = 1 − max cos to R (`W.neighbours`),
for future items only. F5 samples exactly as registered:

- **Steam**: future games with positive + negative ≥ 20; quality = positive / (positive + negative).
- **Patents**: future applications with decision ACCEPTED (quality 1) or REJECTED (0); PENDING excluded.

**Reproduction check (run first)**: n, ρ, p, top-1 % and rest mean quality must equal the registered JSON values to
1e-9 for the four domain × embedder cells; otherwise the run stops. The registered bootstrap CI is not reproduced
bit-for-bit (it consumed the registered RNG stream after other analyses); the CIs here use their own seeds.

## Covariates

- Steam: log(review count); release year (2024/2025); owners bracket as an ordinal (lower bound of the `owners`
  string); primary genre = first comma-separated entry of `genres`, categories with < 30 games in the sample
  (and the empty string) merged into "Other", entered as dummies (reference = Action).
- Patents: filing year (first four characters of `date`); CPC section = first letter of the first `cpc` code,
  dummies (reference = H).

## Exact measures

- **C1 partial Spearman**: `W.partial_spearman(quality, iso, controls)` (rank residualisation of both on the ranks
  of every control column with intercept; Pearson of the residuals; p from `pearsonr`). 95 % CI: percentile
  bootstrap over items, B = 2000, `default_rng(SEED + 33)`. Also reported: the raw ρ with the same bootstrap and
  the attenuation ratio partial / raw. Secondary: partial ρ with each control block alone (to show which one moves
  ρ).
- **C2 Steam strata** by review count 20-99, 100-999, ≥ 1000: raw ρ(iso, quality) and partial ρ given the other
  controls (log count included, it still varies within a stratum), bootstrap CIs (B = 2000, `default_rng(SEED +
  44)`).
- **C3 tails**, Steam and patents, both embedders: for top 1 %, bottom 1 %, top 10 %, bottom 10 % by iso
  (top: iso ≥ the 99th / 90th percentile, as registered; bottom: iso ≤ the 1st / 10th percentile), mean quality of
  the tail, mean quality of the rest, difference tail − rest. CI: bootstrap over items with the cut points recomputed
  in each replicate, B = 2000, `default_rng(SEED + 55)`; two-sided bootstrap p for difference = 0. Secondary: the same
  differences after residualising quality on the controls (mean OLS residual of quality on the C1 controls, raw
  values not ranks, in tail minus rest) so tail contrasts are also shown net of composition.

## Decision rules (fixed now)

- **F5 survives controls** in a domain if, in both embedders, the partial ρ has the same sign as the registered raw
  ρ and its 95 % CI excludes 0 (and Holm-adjusted p < 0.05 in the family below). If it survives but |partial ρ| <
  0.05 in either embedder, it is reported as "survives controls but negligible in size".
- **F5 is descriptive** (the paper calls it an uncontrolled descriptive association and drops any reading of it as
  evidence that isolated work is better or worse received) if the sign flips or the CI includes 0 in either
  embedder.
- **Strata (Steam)**: "consistent across review volume" if all three strata have the overall sign in both embedders;
  if any stratum has the opposite sign with CI excluding 0 in both embedders, the paper says so explicitly. This does
  not override the C1 decision.
- **Tails**: the paper reports all four tails per domain symmetrically. A tail is described as different from the rest
  only if its CI excludes 0 in both embedders; otherwise it is reported as "not distinguishable from the rest". The
  current asymmetric sentence on the patents top 1 % is replaced by the symmetric report whatever the outcome.

## Multiplicity

Holm within: the C1 family (2 domains × 2 embedders = 4 partial-ρ tests); the C3 family (4 tails × 2 domains × 2
embedders = 16 tests). C2 strata and the secondary analyses are descriptive (CIs reported, no Holm-based claim).

## Runtime and memory

Per domain × embedder: one pass of future items against R (~40 s), bootstraps of rank regressions on ≤ 9,477 items
(2000 × a few ms). ~5-8 min total; peak ~1.5 GB. One process, `OMP_NUM_THREADS=4`.

## Outputs

`scripts/review_f5_controls.py`, `docs/prereg/review_f5_controls.json` (keyed `<domain>/<embedder>`, with the
reproduction check and a `decisions` block), and a result section appended to this file.

## Result (run 2026-11-15, `docs/prereg/review_f5_controls.json`)

Reproduction check passed in all four cells (n, ρ, p, top-1 % and rest means equal to the registered values within
1e-9). Runtime ~7 min (85-136 s per cell), one process.

**C1, partial Spearman (95 % bootstrap CI, B = 2000)**

| Cell | n | raw ρ [CI] | partial ρ [CI] | p (Holm) | partial / raw |
|---|---|---|---|---|---|
| Steam / Qwen3 | 6,697 | +0.107 [+0.083, +0.130] | +0.102 [+0.078, +0.125] | 1.4e-16 | 0.95 |
| Steam / BGE-M3 | 6,697 | +0.132 [+0.109, +0.155] | +0.126 [+0.103, +0.149] | 1.2e-24 | 0.96 |
| Patents / Qwen3 | 9,477 | −0.081 [−0.100, −0.062] | −0.059 [−0.079, −0.040] | 1.7e-08 | 0.73 |
| Patents / BGE-M3 | 9,477 | −0.084 [−0.104, −0.064] | −0.055 [−0.074, −0.035] | 1.0e-07 | 0.65 |

Partial ρ with one control block at a time: Steam log reviews +0.101 / +0.128, year +0.109 / +0.133, owners
+0.104 / +0.130, genre +0.107 / +0.128; patents year −0.082 / −0.085, CPC section −0.058 / −0.054 (Qwen3 / BGE-M3).
The CPC section absorbs about a third of the patents association; nothing moves the Steam one.

**Decision: F5 survives controls in Steam and in patents** (same sign, CI excludes 0, Holm p < 0.05, |partial ρ| ≥
0.05 in both embedders).

**C2, Steam review-count strata (raw ρ [CI]; partial ρ in the JSON is within 0.013 of raw)**

| Stratum | n | Qwen3 | BGE-M3 |
|---|---|---|---|
| 20-99 | 3,682 | +0.074 [+0.043, +0.106] | +0.119 [+0.086, +0.150] |
| 100-999 | 2,224 | +0.159 [+0.119, +0.199] | +0.145 [+0.104, +0.184] |
| ≥ 1000 | 791 | +0.110 [+0.038, +0.182] | +0.144 [+0.077, +0.209] |

Consistent across review volume (positive in every stratum, both embedders).

**C3, tails: mean quality tail − rest [95 % CI] (Qwen3 / BGE-M3)**

| Tail | Steam | Patents (grant rate) |
|---|---|---|
| top 1 % (most isolated; n 67 / 95) | +0.002 [−0.035, +0.039] / +0.011 [−0.024, +0.049] | −0.042 [−0.137, +0.028] / −0.074 [−0.158, +0.015] |
| bottom 1 % (least isolated) | +0.014 [−0.035, +0.041] / −0.015 [−0.049, +0.019] | +0.043 [−0.032, +0.102] / +0.032 [−0.037, +0.100] |
| top 10 % (n 670 / 948) | +0.014 [+0.001, +0.023] / +0.021 [+0.009, +0.032] | −0.033 [−0.059, −0.005] / −0.076 [−0.101, −0.047] |
| bottom 10 % | −0.026 [−0.038, −0.015] / −0.032 [−0.044, −0.020] | +0.063 [+0.042, +0.083] / +0.057 [+0.037, +0.079] |

Tail means: Steam top 1 % 0.842 / 0.851, bottom 1 % 0.854 / 0.825, top 10 % 0.852 / 0.859, bottom 10 % 0.816 / 0.811,
rest ≈ 0.84; patents top 1 % 0.800 / 0.768, bottom 1 % 0.884 / 0.874, top 10 % 0.812 / 0.773, bottom 10 % 0.899 /
0.893, rest ≈ 0.84. Net of controls (residualised quality), the 10 % contrasts keep their signs; patents top 10 %
Qwen3 becomes −0.017 [−0.042, +0.010] and Steam top 10 % Qwen3 +0.011 [−0.001, +0.021].

Registered tail reading (CI excludes 0 in both embedders): **1 % tails not distinguishable from the rest in either
domain** (in particular the patents top 1 % reported in the paper is not); **10 % tails different**: Steam top 10 %
higher, bottom 10 % lower; patents top 10 % lower, bottom 10 % higher. Under Holm across the 16 tail tests the bottom
10 % contrasts hold in all four cells (bootstrap p < 0.0005), the top 10 % hold only in BGE-M3 (Holm p: Steam Qwen3
0.35, patents Qwen3 0.14).

**Deviations / clarification.** None from the registered parameters. The registration lists Holm for the C3 family
but states the tail reading on CIs only; both are reported above, and where they differ (top 10 % in Qwen3) the
conservative reading is that only the least-isolated 10 % differ robustly. The partial-ρ p-values are from `pearsonr`
on residuals without a degrees-of-freedom adjustment for the controls (negligible at n ≥ 6,697).
