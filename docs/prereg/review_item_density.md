# Review follow-up: do new items land in empty regions? An item-level test without cells

Registered 2026-11-15, before the script `scripts/review_item_density.py` is written or run. Group "Part 2 /
Section 4, item level". Deviations will be disclosed in the result section appended below. Local CPU only: no LLM,
no gateway, no GPU, no paid call, no GCP.

## Review concern addressed

| Concern | What this registration does |
|---|---|
| Reviewer M2: Part 2 tests "sparse k-means cells grow", not "holes get filled"; asks for an item-level test without cells | A1: item-level AUC of past isolation / past kNN density for future vs held-out past items, with a drift-matched comparator. A2: coverage of empty probe points by future vs held-out items, as a function of the probe's past emptiness. No k-means anywhere. |

## What is known before this registration (disclosure)

From the registered F1 (`where_future_lands_result.json`), already in the paper: the median isolation of future items
exceeds that of held-out past items by 0.004 (patents) to 0.03 (arXiv/Qwen3), i.e. future items are slightly more
isolated in every cell. This implies item-level isolation AUCs above 0.5 but probably modest. No AUC, no density
comparison, no recent-year comparator, no probe statistic has been computed. Sizes checked for feasibility only
(no outcome): held-out items from the momentum years Steam 2,282, arXiv 2,304, patents 3,622; held-out from earlier
years 4,540 / 4,896 / 3,530; reference items from earlier years 40,823 / 43,980 / 32,352. The thresholds below were
set with the F1 medians known.

## Question

Does new work (the future split) appear preferentially where the past map is empty, measured on items and points
rather than on k-means cells?

- **A1**: Are future items more isolated / in sparser places of the past map than held-out past items, and does this
  survive comparison with held-out items from the last past years only (so that "later in time" is not the whole
  effect)?
- **A2**: Are emptier points of the past map covered relatively more by future items than by held-out past items?

## Data

Root `D = ~/Documents/experiment-data/latent-gap-ideas`. `D/prepared/{steam,arxiv,patents}/corpus.parquet`
(`split`, `adult`, `date`), `emb_qwen3e06.npy`, `emb_bgem3.npy` (row-aligned). Adult rows dropped, float32,
L2-normalised. Exactly as `scripts/where_future_lands.py::analyse`: `SEED = 20261030`, `rng = default_rng(SEED)`,
held-out H = `rng.choice(len(P), len(P)//10, replace=False)` of the past P (first use of the stream), reference
R = P \ H, future F = the future split. Sizes must reproduce (reference 61,400 / 64,800 / 64,369; held-out 6,822 /
7,200 / 7,152; future 40,989 / 41,000 / 40,342); otherwise the run stops. Six cells = 3 domains × 2 embedders,
processed one at a time.

## Exact measures

Item measures against R (helpers `W.neighbours`, `W.kth_cos`, `W.count_above` imported from
`where_future_lands.py`, not copied): isolation iso = 1 − max cos to R; density dens = mean of the 10 largest cos to
R.

### A1. Item AUCs

Groups: F (positive, label 1) against

- **H_all**: all held-out past items (primary comparator);
- **H_recent**: held-out items whose year is in `MOMENTUM_YEARS` (Steam 2022-2023, arXiv 2021-2022, patents
  2014-2015) — the drift-matched comparator.

Scores: (i) iso (higher = predicted future); (ii) −dens (lower density = predicted future). AUC = Mann–Whitney
U / (n1 n0) with mid-ranks. 95 % CI: percentile bootstrap, B = 2000, resampling F and the comparator separately
with replacement, `default_rng(SEED + 11)`. One-sided Mann–Whitney p (alternative: future scores higher).
Also reported: ΔAUC = AUC(H_all) − AUC(H_recent) with a bootstrap CI (same resamples; H_recent resampled as the
recent subset of the H_all resample).

Secondary, descriptive only (**A1c, in-past backtest**): R_early = reference items with year < first momentum year;
pseudo-future = H_recent, pseudo-control = held-out items with year < first momentum year; iso and dens of both
against R_early; AUC with the same bootstrap. This is what the same design gives inside the past (time passage plus
whatever novelty the last two past years had); it is reported next to A1 and has no decision rule.

### A2. Coverage of empty probe points

- Probes: 20,000 per cell from the Gaussian of R exactly as in `where_future_lands.py` (μ = mean of R, covariance of
  R with `bias=True`, eigendecomposition, z ~ N(0, I) scaled by √λ, C = μ + Z Uᵀ, L2-normalised), RNG
  `default_rng(SEED + 200)`, one 20,000-row block.
- Per probe j: τ_j = 10th-largest cos to R (`W.kth_cos`); emptiness e_j = 1 − τ_j; c_F(j) = #{f ∈ F: cos > τ_j},
  c_H(j) = #{h ∈ H_all: cos > τ_j}.
- Per-probe rate ratio rr_j = ((c_F + 0.5)/|F|) / ((c_H + 0.5)/|H|).
- Emptiness deciles cut on the 20,000 probes (fixed cut points). Pooled rate ratio of a probe set S:
  RR(S) = (Σ_S c_F / |F|) / (Σ_S c_H / |H|) (pooled counts; no continuity correction needed unless Σ c_H = 0, then
  +0.5).
- **A2 primary**: ratio of ratios RoR = RR(emptiest decile, e ≥ 90th pct) / RR(middle deciles, e in the 30th-70th
  pct).
- **A2 secondary**: Spearman ρ(e_j, log rr_j) over the 20,000 probes.
- **Uncertainty: two-way bootstrap**, B = 2000, `default_rng(SEED + 22)`: in replicate b the probes are resampled
  with replacement **and** every future and held-out item gets an independent Poisson(1) weight; counts are the
  weighted counts (computed in the same chunked pass as a mask × weight-matrix product). This accounts for both
  probe sampling and the sampling of the corpora (balls of nearby probes share items). Percentile CIs; two-sided
  bootstrap p = 2 · min(share ≤ null, share ≥ null), null log RoR = 0 and ρ = 0.
- **A2c, secondary, density-matched real centres (F6-style)**: for each probe in the emptiest decile, the
  percentile of its c_F among the c_F of 200 held-out items (as ball centres, their own τ = 10th cos to R) matched on
  dens within ±0.01 (if fewer than 200 qualify, the 200 nearest in dens; the number matched within ±0.01 is
  reported). Reported: median percentile. Descriptive: real held-out points may sit in denser places than any
  Gaussian probe; then the fallback rule dominates and A2c is uninformative, which the result will say.

## Decision rules (fixed now)

Per cell (domain × embedder):

**A1 (primary score iso vs H_all; the same rule is applied, and reported, for −dens and for H_recent):**

- **supports** the premise: AUC ≥ 0.60 and lower CI bound ≥ 0.55, and Holm-adjusted p < 0.05;
- **contradicts**: upper CI bound < 0.55 (any shift is at most small);
- **weak**: otherwise with lower CI bound > 0.50 (a reliable but small shift; reported as "future slightly more
  isolated, not concentrated in empty regions");
- **inconclusive**: anything else.

The drift-matched reading: if the H_all verdict is "supports" but the H_recent verdict is not, the paper attributes
the separation at least partly to time passage and says so.

**A2 (primary RoR):**

- **supports**: RoR ≥ 1.25 with lower CI bound > 1 and Holm-adjusted bootstrap p < 0.05, and the secondary ρ > 0
  with CI excluding 0;
- **contradicts**: upper CI bound of RoR < 1.10 (emptiest probes get at most marginally more future relative to
  held-out than middle probes);
- **inconclusive**: otherwise.

**Replication rule (project-wide)**: a domain-level statement is made only if both embedders give the same verdict.

**What the paper will say.** If A1 and A2 support in a domain (replicated): "at item level, new work does land in
empty regions" for that domain, reported as a correction of Part 2's cell-level reading. If A1 is weak or
contradicts and A2 contradicts or is inconclusive: "at item level future items are at most slightly more isolated
than held-out past items (AUC X-Y) and empty points of the past map are not covered preferentially by the future
(RoR X-Y)", which answers M2 without cells. Mixed verdicts are reported cell by cell with no domain claim.

## Multiplicity

Holm across the 6 cells within each sub-test family: A1 (iso, H_all), A1 (−dens, H_all), A1 (iso, H_recent),
A1 (−dens, H_recent), A2 RoR, A2 ρ — six families of 6 tests each. A1c and A2c are descriptive, no tests.

## Runtime and memory

Per cell: neighbours of F, H against R (~1 min), probes against R (20,000 × ~64,000), probe counts with the
2000-column Poisson weight matrices (~1.6 TFLOP for F, float32, chunked 500 probes), A1 bootstraps (2000 × 4 × 2
AUCs). Estimated 3-5 min per cell, ~25 min total; peak memory ~2.5 GB (embeddings 0.46 GB, weight matrix
41,000 × 2000 float32 = 0.33 GB, 500 × 41,000 masks). `OMP_NUM_THREADS=4`, one process.

## Outputs

`scripts/review_item_density.py`, `docs/prereg/review_item_density.json` (keyed `<domain>/<embedder>`, plus a
`decisions` block applying the rules above mechanically), and a result section appended to this file.

## Result (run 2026-11-15, `docs/prereg/review_item_density.json`)

Sizes reproduced in all six cells; median future isolation reproduces registered F1 (e.g. Steam/Qwen3 0.2768).
Runtime 21 min for the six cells (165-316 s each), one process. Holm p printed as 0 in the JSON means the asymptotic
Mann–Whitney p underflowed (< 1e-300).

**A1, isolation AUC (future vs held-out; 95 % bootstrap CI; verdict by the registered rule)**

| Cell | iso vs H_all | iso vs H_recent | −dens vs H_all | −dens vs H_recent | A1c backtest iso (in past) |
|---|---|---|---|---|---|
| Steam / Qwen3 | 0.559 [0.551, 0.566] weak | 0.551 [0.540, 0.563] weak | 0.536 [0.529, 0.544] contradicts | 0.538 [0.526, 0.550] contradicts | 0.551 [0.536, 0.566] |
| Steam / BGE-M3 | 0.591 [0.584, 0.598] weak | 0.604 [0.592, 0.615] supports | 0.574 [0.567, 0.581] weak | 0.592 [0.581, 0.605] weak | 0.521 [0.506, 0.536] |
| arXiv / Qwen3 | 0.621 [0.614, 0.629] supports | 0.641 [0.630, 0.652] supports | 0.605 [0.598, 0.613] supports | 0.633 [0.621, 0.645] supports | 0.518 [0.505, 0.533] |
| arXiv / BGE-M3 | 0.613 [0.606, 0.620] supports | 0.637 [0.625, 0.649] supports | 0.586 [0.578, 0.593] weak | 0.625 [0.613, 0.637] supports | 0.510 [0.496, 0.523] |
| Patents / Qwen3 | 0.517 [0.509, 0.524] contradicts | 0.511 [0.501, 0.521] contradicts | 0.514 [0.507, 0.521] contradicts | 0.510 [0.500, 0.520] contradicts | 0.518 [0.505, 0.531] |
| Patents / BGE-M3 | 0.513 [0.505, 0.520] contradicts | 0.513 [0.504, 0.523] contradicts | 0.504 [0.497, 0.511] contradicts | 0.508 [0.498, 0.518] contradicts | 0.502 [0.490, 0.515] |

ΔAUC (iso, H_all − H_recent): Steam +0.007 [−0.002, +0.017] / −0.013 [−0.022, −0.003]; arXiv −0.020 [−0.029,
−0.010] / −0.024 [−0.034, −0.014]; patents +0.006 [−0.001, +0.012] / −0.000 [−0.007, +0.006] (Qwen3 / BGE-M3).
The drift-matched comparator does not shrink the separation anywhere; in arXiv it grows. "Contradicts" in the table
follows the registered rule (upper bound < 0.55): in Steam/Qwen3 −dens and in patents the AUC is still reliably above
0.5 in most cells but the shift is small.

Replicated domain readings (both embedders same verdict): **arXiv supports** (iso vs H_all and H_recent; −dens vs
H_recent); **patents contradicts** (all four A1 families); **Steam weak** for iso vs H_all, mixed elsewhere
(weak/supports for H_recent, contradicts/weak for −dens).

**A2, coverage of empty Gaussian probes (two-way bootstrap)**

| Cell | RR emptiest decile | RR middle (30-70 %) | RoR [CI] | Holm p | ρ(e, log rr) [CI] | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| Steam / Qwen3 | 1.191 | 1.126 | 1.057 [0.944, 1.174] | 0.69 | +0.032 [+0.004, +0.058] | 0.084 | inconclusive |
| Steam / BGE-M3 | 0.893 | 0.817 | 1.093 [0.945, 1.255] | 0.67 | +0.026 [−0.019, +0.064] | 0.68 | inconclusive |
| arXiv / Qwen3 | 0.719 | 0.626 | 1.150 [1.024, 1.290] | 0.12 | +0.079 [+0.052, +0.096] | < 0.006 | inconclusive |
| arXiv / BGE-M3 | 0.766 | 0.698 | 1.097 [0.986, 1.221] | 0.44 | +0.100 [+0.071, +0.114] | < 0.006 | inconclusive |
| Patents / Qwen3 | 0.960 | 0.987 | 0.972 [0.889, 1.069] | 0.69 | −0.002 [−0.022, +0.023] | 0.90 | contradicts |
| Patents / BGE-M3 | 1.042 | 1.132 | 0.921 [0.832, 1.023] | 0.48 | −0.049 [−0.066, −0.008] | 0.044 | contradicts |

Replicated: Steam inconclusive, arXiv inconclusive (emptier probes do get somewhat more future relative to held-out,
ρ ≈ +0.08 to +0.10, but RoR stays below the registered 1.25 and is not Holm-significant), patents contradicts.
Probe balls are small (median 1 held-out item, 3-7 future items; 33-36 % of probes contain no held-out item), which is
why the pooled decile ratio, not the per-probe ratio, is primary.

A2c (descriptive): density matching failed as anticipated — emptiest-decile probes are sparser than any held-out
item in five cells (0 of 2,000 matched within ±0.01; Steam/BGE-M3 1,108); median percentiles (Qwen3 / BGE-M3) Steam 56.2 / 46.4, arXiv
40.8 / 59.0, patents 45.0 / 46.2 are dominated by the nearest-200 fallback and are not interpreted.

**Reading under the registered rules.** At item level, without cells: in arXiv new papers are more isolated from the
past map than held-out past papers (AUC 0.61-0.64), and not because they are later in time (the separation is larger
against the last two past years, and the same design inside the past gives 0.51-0.52); in Steam the shift is reliable
but small (0.55-0.60); in patents it is negligible (0.50-0.52). No domain shows that emptier points of the past map are
covered preferentially by the future (A2: inconclusive in Steam and arXiv, contradicts in patents). So "new work is
somewhat more isolated" holds in arXiv (and weakly in Steam), while "holes get filled" is not supported in any domain.

**Deviations.** None from the registered parameters or rules. A2c fell into the uninformative case foreseen in the
registration.
