# Where the future lands: real novelty against the past map

Pre-registered 2026-10-30, before any analysis of where future items sit. Nothing in this file was
chosen after looking at future-item geometry: only split sizes, date ranges and the availability of
quality signals were inspected (counts below).

## Why

Every test so far looks from the generated side: we pick a point and ask whether a text reaches it.
This test looks from the other side. Each corpus already holds ground truth for "a new idea": the
items released after the split. If the tweet's premise were right, real novelty should appear where
the past map is empty (holes, or far from everything). If the thesis of the article is right, it
appears next to what exists or at the edge of the cap, and past-only geometry tells us where.

Relation to earlier work in this repo: H4 (`design.md`) measured growth around *our* targets, and
`scripts/anticipation.py` measured whether *generated* ideas land near the future. Neither asks where
the future itself lands, nor whether past geometry forecasts it.

Note on the Einstein test (former "part 2"): it is not the novel contribution here. Hassabis's proposal is
widely known and has been attempted (e.g. a 3.3B model trained on pre-1900 text, and "Talkie", a 1930-cutoff
model). Part 2 stays as an optional follow-up, not as the paper's claim to novelty.

## Data (frozen)

- Corpora `~/Documents/experiment-data/latent-gap-ideas/prepared/<domain>/corpus.parquet`, adult rows
  removed as in every earlier analysis. Splits as stored (counts before the adult filter):
  - Steam: past 71,939 (release < 2024-01-01), future 41,497; 4,225 adult rows across both are removed.
  - arXiv: past 72,000 (1990-2022), future 41,000 (2023-01-01 to 2026-10-15).
  - Patents: past 71,521 (filed 2012-2015), future 40,342 (2016-2017).
- Embeddings: `emb_qwen3e06.npy` (primary) and `emb_bgem3.npy` (replication), L2-normalised.
- Quality signals (secondary, as stored, no re-collection):
  - Steam: review share `positive / (positive + negative)` for items with ≥ 20 reviews (7,081 future items).
  - Patents: decision ACCEPTED vs REJECTED; PENDING excluded (30,865 future items are pending, so the
    future quality analysis is censored to 9,477 decided applications and is reported as such).
  - arXiv: none available. No citation data will be fetched for this test.
- Saved targets: GAP and FAR targets from the existing grid runs (`runs/<domain>/…/targets.npz`), unchanged.

## Measures (past-only geometry, computed on past items only)

For every future item f, and for a **held-out past control** (a seeded random 10 % of past items, measured
against the remaining 90 %; seed 20261030), compute against the reference past set:

- `iso(f)` = 1 − max cos to the reference past set (geometric novelty).
- `lat(f)` = cos(f, μ̂) with μ̂ the normalised mean of the reference past set (position on the cap:
  high = centre, low = edge).
- `dens(f)` = mean cos to the 10 nearest reference past items.
- `far_reg(f)` = 1 if max cos to the reference past set ≤ 0.15 (the upper end of what any coherent text
  reached from a FAR point), else 0.

Cells: k-means with K = 256 on the reference past set (seed 20261030, primary embedder). Per cell:
past count, future count, growth = future share / past share, and past-only features (mean `dens`,
mean `lat`, **`hole`** and **momentum**). `hole` uses the same model that builds GAP targets
(`targets.gap`: draws from the past corpus's own Gaussian N(μ, Σ), normalised): draw 200,000 points
(seed 20261030), assign each to its nearest centroid, and set `hole` = the cell's share of draws divided by its
share of past items. High `hole` = space the corpus distribution fills but real items do not. **Momentum** is the share of the cell's past items in the last two past years,
Steam 2022-2023, arXiv 2021-2022, patents 2014-2015.

## Hypotheses and predictions

**F1, the future does not land far away.** The share of future items with `far_reg = 1` is 0 in every
domain and embedder (prediction: exactly 0; a non-zero share up to 0.1 % is reported as "near zero").
`iso` of the future exceeds the held-out control by a median of less than 0.05 in every domain.

**F2, the future does not fill holes preferentially.** Among cells, growth is not predicted by mean `hole`
once momentum is controlled. Test: Spearman partial correlation of growth with `hole` given momentum and
log past count; prediction: |ρ| < 0.1 or a CI that includes 0.

**F3, momentum beats geometry as a forecaster.** For "top-decile growth cell" as the outcome, AUC of
momentum alone ≥ AUC of `hole` alone, and adding `hole` and `lat` to momentum raises the AUC by less than
0.03 (logistic regression, 5-fold cross-validation over cells, seed 20261030).

**F4, edge vs centre (two-sided, no directional prediction).** Is the future shifted towards the edge of
the cap? Mann–Whitney on `lat`, future vs held-out control. Reported with its direction and effect size.

**F5, the most isolated future items are not the most successful.** Within the future, Spearman of `iso`
with Steam review share and with patent acceptance; prediction: ρ ≤ 0 (more isolated is not better
received). The top 1 % by `iso` are compared against the rest on the same signals.

**F6, our GAP targets are not filled faster than matched points.** For each saved GAP target, count future
items with cos > τ, with τ the target's own 10th-nearest past cosine; compare with 200 density-matched past
interior points per target (matched on `dens` within ±0.01). Prediction: the median percentile of GAP
targets is not above 60.

## Analysis rules

- Primary: Qwen3-Embedding-0.6B. Replication: BGE-M3 with the same rules. A finding counts as replicated
  only if the direction and the decision match in both embedders.
- Holm correction within the family {F2, F3, F4, F5} per domain. F1 and F6 are descriptive with fixed
  thresholds.
- 95 % bootstrap CIs (2,000 resamples, seed 20261030) on every median and correlation, resampling cells for
  cell-level tests and items for item-level tests.
- No parameter in this file changes after the first result is seen. Any additional analysis goes into a
  separate "exploratory" section of the result file, labelled as such.

## Cost and resources

Local CPU only, no LLM calls, no gateway, no paid services. Exact nearest-neighbour search over about 40k ×
72k vectors in blocks; peak memory under about 3 GB. Estimated time: 20-40 minutes per domain and embedder.

## Outputs

- `scripts/where_future_lands.py` (single script, all seeds fixed)
- `docs/prereg/where_future_lands_result.json` and a short result section appended to this file.

## Result (run 2026-10-30, `where_future_lands_result.json`)

Decision per prediction; "replicated" = same direction and decision with Qwen3 and BGE-M3 (pre-registered rule).

| | Steam | arXiv | Patents |
|---|---|---|---|
| **F1** future items in the far region (cos ≤ 0.15) | 0 / 0 | 0 / 0 | 0 / 0 |
| min max-cos of a future item to the past (Qwen / BGE) | 0.48 / 0.57 | 0.41 / 0.55 | 0.47 / 0.57 |
| median isolation, future − held-out | +0.012 / +0.015 | +0.029 / +0.018 | +0.004 / +0.002 |
| **F2** partial ρ(growth, hole \| momentum) | +0.20 / +0.33 **fails, replicated** | +0.18 / +0.34 **fails, replicated** | +0.07 / +0.07 holds |
| **F3** AUC momentum (Qwen / BGE) | 0.91 / 0.84 | 0.92 / 0.84 | 0.71 / 0.76 |
| AUC hole alone | 0.39 / 0.45 | 0.42 / 0.48 | 0.47 / 0.43 |
| gain from adding hole + lat | −0.006 / −0.007 holds | −0.016 / +0.006 holds | −0.021 / +0.031 **not replicated** (BGE just over 0.03) |
| **F4** rank-biserial of lat, future vs held-out | −0.01 (ns) / −0.12 | −0.17 / +0.02 | −0.01 (ns) / +0.01 (ns) |
| **F5** ρ(isolation, quality) | +0.11 / +0.13 **fails, replicated** | no signal available | −0.08 / −0.08 holds |
| **F6** median percentile of GAP targets (Qwen only) | 55 holds | 54 holds | 39 holds |

Reading, as pre-registered:

- **F1 holds everywhere, in both embedders.** Not one of ~122k future items lands in the far region; the most
  isolated future item still has cos 0.41–0.57 with some past item, and the future is only 0.002–0.029 more
  isolated than a random held-out slice of the past.
- **F3 holds in 5 of 6.** Momentum forecasts where the future grows (AUC 0.71–0.92); the hole score alone is at or
  below chance (0.39–0.48), and past geometry adds nothing to momentum except +0.031 in patents/BGE, at the threshold.
- **F2 fails in Steam and arXiv, replicated.** Holding momentum and size fixed, cells the corpus Gaussian fills but
  items do not grow somewhat faster (ρ 0.18–0.34). Holes do get filled, modestly; they still do not forecast growth (F3).
- **F4 is embedder-dependent.** The edge shift seen in arXiv/Qwen (−0.17) is absent in BGE, and Steam shows the
  reverse pattern. No edge-vs-centre claim survives.
- **F5 splits by domain, replicated.** In Steam, more isolated future games are slightly better reviewed (ρ ≈ +0.12);
  in patents, more isolated applications are slightly less often granted (ρ = −0.08). Both effects are small.
- **F6 holds, with a matching limit.** GAP targets are not filled faster than the controls (percentiles 39–55), but
  no control met the ±0.01 density match (GAP density 0.41–0.52 vs 0.64–0.70 for real items), so the controls
  are the nearest available in density, not matched pairs.
