# Review follow-up: is the hole score a gap score, and does Section 4 depend on it?

Draft pre-registration, written 2026-11-14 before any of the analyses below is run. Group "hole score validity"
(Section 4). To be committed to `docs/prereg/review_hole_validity.md` before the script is run; deviations will be
disclosed in a result section appended later. Nothing here calls an LLM, the gateway or a GPU.

## Review concerns addressed

| Concern (fable_peer_review.md) | What this registration does |
|---|---|
| M7.3, hole score validity | V1: correlations across cells of the hole score with density, centroid geometry and cell size; V2: a planted-hole positive control; three further hole definitions (H1-H3) |
| Question 3 | V1 answers it exactly: ρ across cells of the hole score with (a) mean kNN density, (b) centroid cosine to the corpus mean, (c) cell size, plus (d) angular extent |
| M7 "two further resolutions" | K = 128 and 512 next to the registered 256 |
| M7 "show the conclusion does not depend on them" | T1-T3: F2 and F3 (the two registered tests that use cells and the hole) rerun for every definition × resolution × space |
| Question 2 (hole part only) | Spaces: raw (registered), mean-centred, whitened. The FAR-target part of question 2 belongs to Section 3 and is not covered here |
| Minor 6, patents F3 wording | Text change registered unconditionally (below); T1-T3 report patents separately |
| Minor 7, F3 with minimum cell size or shrunken growth | T4: both variants |
| Minor 8, different horizons | T5: growth recomputed at common 12- and 24-month horizons |

M7.1 (claim is about volume growth), M7.2 (F5 in the main text) and M7.4 (abstract quotes cross-validated AUCs) need no
new analysis; they are text changes and are listed at the end so this registration does not lose them.

### Disclosure: what I saw before writing this

While checking feasibility I loaded the cached registered cell features
(`runs/revision_future_cells/<domain>_<embedder>.npz`, already used in the paper) and computed, for the registered
hole score H0 only: Spearman ρ(H0, past cell size) = +0.34 to +0.53 and ρ(H0, mean item latitude) = +0.11 to +0.46
across the six corpus × embedder cells; cells with fewer than 100 past items: 3 to 25 per cell, of which at most 1
is in the top decile of growth. Nothing about the new definitions, resolutions, spaces, horizons or the planted
control was computed. Those two numbers already lean towards the reviewer's concern for H0; the thresholds below
were set with that knowledge, and the result section will say so.

## Question

1. Does the registered hole score (H0: share of normalised corpus-Gaussian draws assigned to a cell over its share of
   past items) measure local emptiness, or mainly where a cell sits (near the mean direction, large, wide)?
2. Do the Section 4 conclusions survive other reasonable gap scores, other resolutions and centred or whitened
   embeddings? The conclusions at stake are:
   - **C1** a gap score alone ranks the top decile of growth at chance (directly scored AUC, CI including 0.5);
   - **C2** adding the gap score (and latitude) to momentum gains less than the registered margin 0.03 (F3);
   - **C3** holding momentum and log size fixed, gap and growth are positively associated in Steam and arXiv, not in
     patents (F2 fails in Steam and arXiv).
3. Do C1-C3 hold with a minimum cell size or shrunken growth (minor 7) and at common horizons (minor 8)?

## Data (confirmed to exist 2026-11-14)

Root `D = ~/Documents/experiment-data/latent-gap-ideas`.

- `D/prepared/{steam,arxiv,patents}/corpus.parquet` (columns `split`, `adult`, `date`); adult rows removed as in
  every earlier analysis.
- `D/prepared/{steam,arxiv,patents}/emb_qwen3e06.npy` and `emb_bgem3.npy`: float16, shapes (113436, 1024),
  (113000, 1024), (111863, 1024); cast to float32 and L2-normalised.
- Reference past set, held-out slice and the registered RNG stream exactly as `scripts/where_future_lands.py` and
  `scripts/revision_future_cells.py::build_cells` (reference 61,400 / 64,800 / 64,369; held-out 6,822 / 7,200 /
  7,152; future 40,989 / 41,000 / 40,342).
- Registered cell features for the reproduction check: `D/runs/revision_future_cells/<domain>_<embedder>.npz`
  (keys `growth, hole, momentum, cell_lat, past_n, fut_n, draw_n, idx_ci[2000,256]`). They hold no cell labels, so
  k-means is refitted (same seed, `n_init=1`) and checked against them.
- Future years after the adult filter (for T5): Steam 2024: 17,032, 2025: 20,771, 2026: 3,186; arXiv 2023: 7,561,
  2024: 9,720, 2025: 11,389, 2026: 12,330; patents 2016: 20,225, 2017: 20,117.

Not used: GAP/FAR targets, judges, quality signals (F5 is item-level and does not involve the hole), any `.pkl`.

## Exact computation

Constants: `SEED = 20261030`, `B = 2000` bootstrap resamples, `k_nn = 10`, `N_DRAWS = 200,000`, `M_PROBE = 500`
per cell, `r_PCA = 10`. Processing is one corpus × embedder at a time; arrays are freed before the next.

### Spaces (Q2)

Fitted on the reference set R only and applied to R, held-out H and future F:

- **raw**: the registered space.
- **centred**: x ↦ (x − μ_R) / ‖x − μ_R‖, μ_R the mean of the normalised reference vectors.
- **whitened**: centre as above, project on the top 256 principal components of the centred reference set,
  divide each by √λ_k, L2-normalise. (256, not 1024, because the trailing eigenvalues are tiny and full whitening
  would let noise directions dominate the cosine; fixed now, not tuned.)

In every space all steps below are redone: k-means, cell assignment of R and F, growth, momentum, densities, draws,
probes. Latitude in F3 is always the **raw-space** cell latitude (mean raw cosine of the cell's reference items to the
raw normalised mean), because in a centred space the mean direction is undefined; this keeps F3's model identical in
meaning across spaces.

### Resolutions

K ∈ {128, 256, 512}; `KMeans(n_clusters=K, random_state=SEED, n_init=1)` on R. Raw K = 256 must reproduce the
cached registered `growth`, `hole`, `momentum` (tolerance: max |Δ| < 1e-6 for growth and momentum; for hole the
known arXiv/Qwen3 drift is to the third decimal, so tolerance 5e-3 and the partial ρ within 0.005 of the registered
value). If the reproduction fails beyond that, the run stops and the result section reports it.

### Cell features (per space × K)

- `past_n`, `fut_n`, growth = (fut_n / |F|) / (past_n / |R|), momentum (share of the cell's reference items from the
  last two past years: Steam 2022-2023, arXiv 2021-2022, patents 2014-2015), as registered.
- Item density r_dens(i) = mean cosine of reference item i to its 10 nearest other reference items (as registered:
  11 largest including itself, `(11·top11_mean − 1)/10`), computed once per space (independent of K).

### Hole definitions (high = more gap, orientation fixed now)

- **H0 Gaussian hole (registered).** Fit mean and covariance of the reference set in the space, draw 200,000 points,
  L2-normalise, assign each to its nearest centroid; H0 = draw share / past share. Raw space: the registered RNG
  stream (held-out choice, the F1 bootstrap draws consumed, then the draws), so raw K = 256 reproduces the paper.
  The same standard-normal matrix Z is reused for K = 128 and 512 in the raw space (common random numbers). Centred
  and whitened spaces: `default_rng(SEED + 1)` and `default_rng(SEED + 2)`, 20,000-row blocks as registered.
- **H1 density at the centroid.** c_j = normalised k-means centroid; H1(j) = − mean of the 10 largest cosines
  between c_j and the reference items.
- **H2 mean item sparseness.** H2(j) = − mean of r_dens over the cell's reference items (the `cell_dens` already
  computed in the registered script as `raw_rho_density`, now used as a competitor score).
- **H3 uncovered share inside the cell's extent (proxy for "share of the cell's angular volume not within a fixed
  cosine of any item").** The literal quantity is not computable: uniform sampling of a Voronoi cell on S^1023 is not
  feasible, and by concentration of measure nearly all of any cell's volume lies farther from every item than any
  cosine real items reach, so the share would be ≈ 1 in every cell. The proxy restricts the volume to where the
  cell's items actually spread. For cell j with reference items X_j (n_j rows) and mean m_j (not normalised): take
  the top r = min(10, n_j − 1) principal directions v_k of X_j − m_j with variances λ_k; draw 500 probes
  p = normalise(m_j + ρ · u / ‖u‖), u = Σ_k z_k √λ_k v_k, z ~ N(0, I_r), and ρ resampled from the empirical
  offsets ‖x_i − m_j‖ of the cell's items (so probes sit at the same distance from the centre as real items, in the
  directions the cell spans). Probes whose nearest centroid is not j are dropped (count reported). H3(j) = share of
  kept probes whose maximum cosine to the whole reference set is below τ. τ is the 50th percentile of the held-out
  items' maximum cosine to R in that space (a typical "nearest past neighbour" of a genuinely new past item; past
  only). Degeneracy rule, applied before any look at growth: if more than 80 % of cells have H3 = 0, move τ up the
  ladder p75, p90, p95 of the same held-out distribution; if more than 80 % have H3 = 1, move it down p25, p10, p05;
  take the first non-degenerate rung, record it. Probe RNG `default_rng(SEED + 100 + K + 1000·space_index)`,
  cells processed in index order.

### V1. Validity correlations (M7.3, question 3)

For every space × K × definition, across cells: Spearman ρ of the hole score with
(a) mean item density (cell mean of r_dens; for H2 this is −1 by construction and is reported as such),
(b) cos(c_j, μ̂) with μ̂ the normalised mean of the reference set in that space (in centred space μ̂ is the
normalised mean of the re-normalised centred vectors; reported but flagged as near-degenerate),
(c) log past_n,
(d) angular extent, mean over the cell's items of 1 − cos(x_i, c_j).
Also: R²_geom = R² of OLS of rank(hole) on ranks of (b), (c), (d) with intercept; the partial Spearman of hole
with (a) given (b), (c), (d) (residualised ranks, as `W.partial_spearman`). CIs: percentile, B = 2000 cell resamples
with replacement, `default_rng(SEED)` per configuration. Holm across the 4 correlations + partial within each
configuration (reported; the reading rule uses counts across the six corpus × embedder cells).

### V2. Planted-hole positive control (raw space, K = 256, all four definitions)

Does each score see an empty region created inside a cell, as opposed to fewer items? `default_rng(SEED + 7)`
selects 52 cells with past_n ≥ 40; the first 26 get a **ball depletion** (pick one of the cell's items at random,
delete the ⌊n_j/2⌋ reference items of the cell nearest to it by cosine, the picked item included), the other 26 a
**random thinning** (delete a random ⌊n_j/2⌋ of the cell's items). Each depletion is applied alone, the rest of the
map intact, centroids and Gaussian draws fixed. Recompute the cell's score on the depleted map: H0 (draw count over
the new past share), H1 (centroid kNN against the depleted reference set), H2 (r_dens of the remaining items of the
cell against the depleted set), H3 (same probe recipe from the remaining items, same τ). Outcome per definition:
AUC of Δscore = score_after − score_before for ball (1) vs thinning (0), with a B = 2000 bootstrap over the 52
cells. Both arms halve the cell's count, so a score that only reads the count (H0 by construction, since draws and
centroids are fixed) should sit at 0.5.

### T1-T3. F2 and F3 rerun (the conclusion test)

For every configuration g = (space, K, definition) in the grid below and every corpus × embedder:

- **T1 (C1)**: directly scored AUC of the hole for top-decile growth (`roc_auc_score`, top = growth ≥ 90th
  percentile, as registered), percentile CI over B = 2000 cell resamples.
- **T2 (C2, F3 as registered)**: momentum cv-AUC (`W.cv_auc`, global z, StratifiedKFold(5, shuffle, SEED)),
  momentum ≥ hole check on cv-AUCs, gain = cv-AUC(momentum, hole, lat) − cv-AUC(momentum). Interval: the revision
  cluster bootstrap (`revision_future_cells.f3_cluster`: cells resampled, `StratifiedGroupKFold` by original cell id,
  z fitted in training folds), B = 2000; resample indices = the cached `idx_ci` for K = 256, `default_rng(SEED + K)`
  draws of shape (2000, K) otherwise; resamples with fewer than 5 positives or negatives skipped and counted. The
  momentum-only arm is computed once per (space, K, outcome) and shared across definitions. Secondary, reported
  only: the gain of momentum + hole without latitude.
- **T3 (C3, F2 as registered)**: partial Spearman of growth with the hole given momentum and log past_n, CI over
  B = 2000 cell resamples (same indices as T2).

Grid: raw × {128, 256, 512} × {H0, H1, H2, H3} (12) and {centred, whitened} × 256 × {H0..H3} (8): 20 configurations
per corpus × embedder, 120 in all.

### T4. Minor 7 (raw, K = 256, H0-H3)

- **Minimum size**: drop cells with past_n < 100, recompute the top-decile threshold on the kept cells, rerun T1-T3
  on them (indices `default_rng(SEED + 1000)`).
- **Shrunken growth**: E_j = past_n_j · |F| / |R|; v̂ = max(var_j(fut_n_j / E_j) − mean_j(1 / E_j), 1e-6) (between-
  cell variance of the rate ratio net of Poisson noise); α = 1 / v̂; growth_shr = (fut_n_j + α) / (E_j + α)
  (Poisson-gamma posterior mean with prior mean 1). Top decile and T1-T3 on growth_shr (partial Spearman in T3 also
  on growth_shr).

### T5. Minor 8, common horizons (raw, K = 256, H0-H3)

Growth recomputed with the future restricted to (a) the first 12 months after the split (Steam 2024, arXiv 2023,
patents 2016) and (b) the first 24 months (Steam 2024-2025, arXiv 2023-2024, patents 2016-2017 = full). Past
features unchanged. Report momentum's direct and cv AUC with CIs, and T1-T3, at each horizon.

### Multiplicity

The paper's Section 4 claims are nulls (C1, C2), so correcting for many configurations makes them easier to keep.
Therefore: **the reading rules that can count against the paper use uncorrected 95 % intervals plus the project's
replication rule** (same direction and decision in Qwen3 and BGE-M3). Holm-adjusted p-values are reported for
transparency within the family "all T1-T5 tests of one outcome in one corpus × embedder" (T1: 20 + 8 + 8 = 36 tests;
T2 bootstrap two-sided p as in `f3_cluster`; T3 36 tests). Claims in favour of a gap score's forecasting value are
made in the paper only if they also survive Holm in that family in both embedders.

## Reading rules (fixed now)

**Validity of each definition (V1, V2), per space × K:**

- *Tracks emptiness* if ρ(hole, a) ≤ −0.3 with CI below 0 **and** the partial ρ(hole, a | b, c, d) ≤ −0.2 with CI
  below 0, in at least 5 of 6 corpus × embedder cells. (Trivially true for H2; H1 nearly so; it matters for H0, H3.)
- *Dominated by geometry* if R²_geom ≥ 0.5 and the emptiness criterion fails, in at least 4 of 6.
- *Sees a planted hole* if the V2 AUC ≥ 0.75 with CI excluding 0.5 in at least 5 of 6.
- A definition is **valid** if it sees a planted hole, or tracks emptiness without being dominated by geometry.
- **Against the paper's current use of H0**: H0 dominated by geometry, or H0 failing V2 (expected by construction),
  means the registered score does not measure gaps; the paper must then say so and rest C1-C3 on the valid
  definitions.

**Conclusion tests (T1-T5), per corpus:**

- **C1 threatened** if some valid definition at some resolution or space has a directly scored AUC whose CI lies
  above 0.5 in both embedders.
- **C2 threatened** if some valid definition has gain ≥ 0.03 with the cluster-bootstrap lower bound > 0 in both
  embedders. Gain ≥ 0.03 with an interval including 0 is "inconclusive", as patents/BGE-M3 already is.
- **C2 strengthened** if every valid configuration has a cluster-bootstrap upper bound below 0.05 in both embedders.
- **C3** is "robust" if the partial ρ is positive with CI excluding 0 for every valid definition in Steam and arXiv in
  both embedders, "definition-dependent" if this holds for some valid definitions only, and "not supported" if it
  holds for none.
- **Dependence on choices**: the conclusion is "independent of definition, resolution and space" if no
  configuration of a valid definition threatens C1 or C2 in any corpus; otherwise the paper names the
  configurations that do.
- **T4/T5** use the same C1-C3 rules within their variant. Momentum's AUC range is reported per horizon; if the
  12-month AUCs differ across corpora with non-overlapping CIs, the text says momentum's strength is
  corpus-specific.

## What the paper will say

- **H0 valid, C1/C2 not threatened anywhere** (the outcome the current text assumes): Section 4 adds one sentence:
  "The conclusion does not depend on the hole score: three other gap scores, 128 to 512 cells and centred or
  whitened embeddings give directly scored AUCs of X to Y and gains below Z (Appendix A.2b)." Abstract unchanged
  except for the text changes below.
- **H0 not valid (dominated by geometry or blind to planted holes), valid alternatives do not threaten C1/C2**:
  Section 4 says the registered hole score mostly reflects cell size and position (give ρ and R²_geom), that the
  pre-registered tests therefore test a weak operationalisation, and that gap scores which do see empty regions
  (name them) also forecast at chance and add nothing to momentum. The F2 paragraph is rewritten on the valid
  definitions; if C3 is not supported by them, "Holes go with growth once momentum is held fixed" is withdrawn and
  the Luo and Evans agreement sentence goes with it.
- **No definition is valid**: the paper says that none of four gap scores could be shown to measure emptiness at
  cell level, that Section 4 therefore cannot separate "gaps do not forecast growth" from "these scores do not
  measure gaps", and the abstract's item (2) becomes "a hole score and three alternatives ... none validated".
- **C1 or C2 threatened by a valid definition**: the abstract and Section 4 heading change: "gaps do not forecast
  growth" becomes "the registered hole score does not forecast growth; score X at resolution Y does (AUC, gain,
  interval)", reported as exploratory and requiring a fresh confirmatory split. This would count against the paper's
  current claim.
- **T4 changes a decision**: Section 4 reports the F3 decision on the variant and says the registered decision rested
  on noise in small cells. If not, one sentence: "With cells of at least 100 items or with shrunken growth the
  decisions are unchanged."
- **T5**: the momentum range is always reported per corpus with its horizon (minor 8 is a text change whatever the
  outcome); if hole AUCs differ by horizon in a way that threatens C1 at one horizon, that is stated with the
  horizon.

Text changes registered unconditionally (no analysis needed):

- Minor 6: "add nothing as a forecast" covers Steam and arXiv; patents is stated as inconclusive (gain +0.031 in
  BGE-M3, cluster intervals up to +0.051 and +0.114).
- M7.4: the abstract quotes the directly scored hole AUCs (0.45-0.54), not the cross-validated ones.
- M7.1: the claim is restated as about where volume grows at cell level, not about single items.
- M7.2: F5 (more isolated Steam games slightly better reviewed, replicated; patents the reverse) enters Section 4
  and the abstract.

## Paid generation, GPU

None. Local CPU, saved embeddings only; no embedding model is loaded, no gateway call, no Kaggle.

## Runtime and memory (16 GB Mac)

By analogy to `runs/where_future_lands.log` (5.5-7 min per corpus × embedder for the whole registered script, of
which ~3 min for k-means at K = 256, 200,000 draws and item densities):

- Features, per corpus × embedder: raw (densities once; k-means, draws and 500-probe H3 for three K) ~12-15 min;
  centred and whitened at K = 256 ~5 min each; V2 ~3 min. ~25-30 min each, **~2.5-3 h** for all six.
- Tests: 36 T configurations per corpus × embedder, cluster bootstrap B = 2000 with two 5-fold logistic fits per
  resample (~40 ms per resample estimated), momentum arm shared: ~30-50 min per corpus × embedder single-threaded,
  ~1.5 h with three worker processes. A 20-resample timing pilot on the registered configuration (output discarded)
  sets the worker count.
- **Total ~4-5 h wall**, run in two stages so features are cached and tests can be rerun.
- Peak memory ~3 GB: full embeddings as float32 (~0.46 GB), reference set and its transformed copy (~0.27 GB each),
  1,000 × 64,800 similarity blocks plus `np.partition` copy (~0.5 GB), 20,000-row draw blocks, 500-probe blocks.
  One corpus × embedder at a time, `OMP_NUM_THREADS=4`.

## Outputs

- Script: `scripts/review_hole_validity.py` (`--stage features|tests`, `[domain/embedder ...]`), importing
  `where_future_lands` and `revision_future_cells` helpers so the registered functions are reused, not copied.
- Feature cache: `D/runs/review_hole_validity/<domain>_<embedder>_<space>_K<k>.npz` (labels, all features, H0-H3,
  covariates, τ rung, probe drop counts) and `<domain>_<embedder>_planted.npz`.
- Result: `docs/prereg/review_hole_validity.json`, keyed `<domain>/<embedder>/<space>/K<k>/<definition>` for V1 and
  T1-T3, plus `planted`, `min_size`, `shrunk`, `horizon_12m`, `horizon_24m`, `reproduction_check`, and a
  `decisions` block applying the reading rules above mechanically.
