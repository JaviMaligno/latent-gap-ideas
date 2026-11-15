# Review follow-up: reliability of the OLMo continuation direction, and multiplicity in the gpt-4o replication (pre-registered)

Draft pre-registration, 2026-11-14, written after the peer review and before any of the computations below were run.
To be committed to `docs/prereg/review_olmo_reliability.md` before either script is run; deviations disclosed in the
JSON and in Appendix B. Two scripts, one per concern:

| Concern | Script | Output |
|---|---|---|
| M8 + question 8 (OLMo, Section 5.3, last paragraph) | `scripts/review_olmo_reliability.py` | `docs/prereg/review_olmo_reliability.json` |
| M9 (gpt-4o replication, Section 5.3, abstract, conclusion) | `scripts/review_critique_multiplicity.py` | `docs/prereg/review_critique_multiplicity.json` |

**What I know before this registration.** Every number in `strengthen_olmo_direction.json` and
`strengthen_critique_replication.json` (including the pooled secondary), `revision_neutral.json`, the paper text and
the review. In particular: the OLMo primary contrasts (GEO −0.002, reviewer −0.003, ORACLE +0.010, Qwen3, 2,000-resample
percentile CIs), the four per-arm changes along s (ORACLE +0.016, CI +0.005 to +0.028, uncorrected Wilcoxon p = 0.019;
NEUTRAL +0.006), the replication E-tests and their uncorrected p-values, and the pooled 300-context estimates
(GEO − NEUTRAL +0.004 in Qwen3, +0.011 in BGE-M3). From `eligible.json` (read while checking feasibility) the number
of consensus continuations k of the 150 eligible OLMo contexts: k = 2 in 83, 3-4 in 44 (21 + 23), 5 or more in 23,
k ≥ 4 in 46. I have **not** computed any t interval, 20,000-resample interval, leave-one-out projection on s,
split-half reliability, stratified contrast or corrected p-value registered below. These are therefore registered
follow-ups to known results, not original registrations, and the paper will label them so (review minor point 2).

No paid API, no GPU, no generation, no new labels. Everything is recomputed from saved vectors.

---

## Part A. M8 and question 8: how much does the OLMo bound bound?

### Review concerns addressed

M8: (1) s is built from a median of two LLM-labelled continuations, so projections on it are attenuated and an
interval inside ±0.015 on s bounds less than it appears; (2) report the bound as a fraction of the continuations' own
leave-one-out projection on s, stratified by k = 2, 3-4, 5+; (3) a split-half reliability for s where k ≥ 4 and a
disattenuated interval; (4) report all four per-arm changes, including ORACLE's +0.016; (5) 20,000 bootstrap
resamples and the t interval, since the reviewer's interval clears −0.015 by 0.0006. Question 8: how many contexts
have exactly two continuations, and do the contrasts differ between those and the rest.

### Question

(i) On the registered estimand (change along the estimated s, critique arm minus NEUTRAL), do the GEO and reviewer
intervals still lie inside ±0.015 with 20,000 resamples and with the t interval? (ii) How large is ±0.015 compared
with what a real continuation obtains on a direction built without it? (iii) How reliable is s, and what interval do
the contrasts have on the latent continuation direction once the attenuation from estimating s is corrected?
(iv) Do the contrasts differ by k?

### Data (all confirmed to exist, 2026-11-14)

- `~/Documents/experiment-data/latent-gap-ideas/runs/olmo-direction/eligible.json` (1,800 context records; 150 with
  `eligible = true`; `consensus` = indices of consensus continuations into the context's saved follow-ups),
  `runs/olmo-direction/citing_all.jsonl`.
- `~/Documents/experiment-data/latent-gap-ideas/kaggle-output/olmo-direction-w1p1/` and `.../olmo-direction-w2p1/`:
  `run_info.json`, `iter_olmo.jsonl`, `{ctx,fup,iter}_{qwen3e06,bgem3}.npy` (float16; e.g. `fup_qwen3e06.npy`
  2,655 × 1,024 and 1,479 × 1,024), `fup_owner.npy`.
- `~/Documents/experiment-data/latent-gap-ideas/prepared/arxiv/emb_{qwen3e06,bgem3}.npy` (113,000 × 1,024 float16)
  and `prepared/arxiv/corpus.parquet` (direction pool: future split, not adult).
- Loading, the pool, f, p, s and the per-arm/round values are taken unchanged from `scripts/olmo_direction_analysis.py`
  (`load_outputs`, `future_pool`, `per_context`, `delta`), imported, not copied. A first check asserts that the means
  of the three primary contrasts and of the four per-arm changes reproduce `strengthen_olmo_direction.json` to 1e-9
  in both embedders; if not, the script stops.

### Exact computation

Notation per context c (unit vectors, Kaggle encoders): t_j = unit tangent of consensus continuation j (j = 1..k),
P = I − pp^T with p the registered primary p (k nearest non-citing future-split papers; it depends on the continuations
only through their number, so it is held fixed throughout), u_j = P t_j. s = unit(Σ_j u_j) (identical to the
registered s = unit(f − (f·p)p)). For a subset A of continuations, s_A = unit(Σ_{j∈A} u_j). d_A,c = registered
contrast of arm A in context c (Δ_A − Δ_NEUTRAL along s); Δ_A,c = registered per-arm change; v_A,0,c = round-0 level
of arm A along s.

**A1. Interval robustness (registered estimand).** For each of the 3 contrasts and the 4 per-arm changes, both
embedders: mean; 95 % t interval (df = n − 1); 95 % percentile bootstrap with **20,000** resamples of contexts,
`numpy.random.default_rng(20261106)`, one index matrix of shape (20,000, n) drawn once per embedder and reused for
every statistic of that embedder (so all intervals are on the same resamples); Monte Carlo SE of each percentile
endpoint = SD of the endpoint over the 20 consecutive blocks of 1,000 resamples / √20; two-sided Wilcoxon signed-rank.
Holm: the 3 contrasts per embedder (as registered); the 4 per-arm changes per embedder (Wilcoxon, new family of 4).
Distance of each contrast interval from the bound: margin = 0.015 − max(|lower|, |upper|).

**A2. Continuations' own leave-one-out projection on s.** For each j: s_{−j} = unit(Σ_{i≠j} u_i) and
L_j = t_j · s_{−j}; L_c = mean_j L_j. L̄ = mean of L_c over contexts (overall and per stratum k = 2, 3-4, 5+).
Ratios, per embedder, overall and per stratum: ρ_bound = 0.015 / L̄; ρ_CI,A = max(|lower|, |upper|) of the t interval
of contrast A / L̄; room_A = L̄ − mean(v_A,0) (how far round-0 ideas sit below a held-out continuation) and
0.015 / room_A. Bootstrap 95 % CIs of L̄ and ρ_bound with the A1 resamples (strata: the resampled contexts falling in
the stratum). Note registered now: L_c uses a direction from k − 1 continuations while the ideas are projected on one
from k, so ρ is conservative (it makes the bound look larger), most of all at k = 2, where s_{−j} is one continuation.

**A3. Reliability of s and disattenuation.** Measurement model, stated now: u_j = μ_c + e_j with e_j independent,
mean zero, E|e_j|² = σ², |μ_c|² = m²; in 1,024 dimensions the cosine between the means of two disjoint sets of sizes a
and b concentrates on √(R(a)R(b)), with R(n) = nλ/(nλ + 1) and signal-to-noise λ = m²/σ²; the cosine between the mean
of n continuations and μ is a(n) = √R(n). A projection of an idea on s estimates (projection on unit(μ)) × a(k),
provided the idea is independent of the continuation noise (true for GEO, reviewer and NEUTRAL, which never see a
follow-up; not strictly for ORACLE, whose feedback is the max cosine to the saved follow-ups: its corrected values are
reported with that caveat).

- *Split-half (primary, k ≥ 4, 46 contexts).* Halves of sizes ⌊k/2⌋ and ⌈k/2⌉. All distinct partitions if there
  are at most 200, otherwise 200 drawn without replacement with `default_rng(20261106 + 7000 + pos)`. r_c = mean over
  the partitions of cos(s_A, s_B).
- *Pooled λ.* λ̂ solves mean_c √(R(a_c, λ) R(b_c, λ)) = mean_c r_c over the k ≥ 4 contexts (the left side is
  increasing in λ): bisection on log λ in [1e−6, 1e6], 200 iterations; λ̂ = 0 if mean r_c ≤ 0. Reported with the
  implied a(2), a(3), a(4), a(5), a(10) and the Spearman-Brown full-length reliability R(k) per stratum.
- *Disattenuated estimates.* d*_c = d_c / a(k_c, λ̂); mean over contexts; for each contrast and per-arm change, both
  embedders. Corrected continuation projection L*_c = L_c / a(k_c − 1, λ̂) and corrected ratio
  ρ* = max(|lower*|, |upper*|) / L̄*.
- *Interval.* Percentile bootstrap on the A1 resamples, recomputing λ̂ from the resampled k ≥ 4 contexts (their
  r_c are fixed per context) and then the corrected mean. If λ̂ = 0 in a resample its corrected mean is ±∞ (the sign
  of the raw mean); if more than 2.5 % of resamples give λ̂ = 0 the corresponding CI end is reported as unbounded.
- *Model checks (descriptive).* (a) Pairwise route: r1_c = mean over pairs of cos(u_i, u_j); λ̂_pair = r̄1/(1 − r̄1),
  per stratum (k = 2, 3-4, 5+) and on the k ≥ 4 set, compared with the split-half λ̂ on the same set (the model
  predicts agreement). (b) Model-predicted vs observed L̄ per stratum: predicted L̄_k ∝ a(k − 1), so
  L̄(2) : L̄(3-4) : L̄(5+) is compared with the ratio of mean a(k − 1). (c) Sensitivity: d* with stratum-specific
  λ̂_pair instead of the pooled split-half λ̂.

**A4. Question 8 and stratification.** Counts per stratum (83 / 44 / 23 expected from `eligible.json`). Per stratum:
the 3 contrasts and 4 per-arm changes, mean, t interval, 20,000-resample bootstrap (same resamples, restricted).
Heterogeneity test: k = 2 versus k ≥ 3 for each of the 3 contrasts, Welch two-sample t, 95 % CI of the difference,
Holm over 3 (Qwen3 primary; BGE-M3 own Holm). A trend test across the three strata (OLS slope of d_c on stratum index
0/1/2, HC3 SE) is descriptive. Power, stated now: with SD ≈ 0.07 and n = 83 vs 67 the SE of the difference is about
0.0115, so differences below about 0.032 have under 80 % power; a non-significant heterogeneity test does not show
homogeneity and the paper will not say it does.

### Reading rules (fixed now)

Primary cell Qwen3 throughout; BGE-M3 reported alongside and read with the same rules as a secondary cell.

- **R-A1 (registered estimand).** For GEO and reviewer: "no feedback-specific turn of 0.015 or more" stands only if
  *both* the 20,000-resample percentile interval and the t interval lie strictly inside (−0.015, +0.015). If either
  reaches or crosses the bound, that arm is re-read as **insufficient** (this stricter rule replaces the registered
  2,000-resample-only reading and is disclosed as a deviation in its favour of caution). If both are inside but the
  smaller margin is below 0.001 or below 2 Monte Carlo SEs, the text gives the margin explicitly ("inside the bound
  by 0.000x"). ORACLE: unchanged unless both intervals now exclude zero with Holm p < 0.05 (then "detected").
- **R-A2 (size of the bound).** ρ_bound (Qwen3, overall): below 0.25 → "the bound is under a quarter of the
  projection a held-out real continuation obtains on such a direction"; 0.25 to 0.5 → "a quarter to a half";
  0.5 or more → the bound is not small relative to the target, and "barely" / "bounded" language for OLMo is replaced
  by "the test excludes only turns of at least half what a real continuation obtains". The k = 2 stratum's ratio is
  reported next to it whatever it is.
- **R-A3 (disattenuated).** (a) Corrected 95 % CI inside ±0.015 for GEO and reviewer → "the bound also holds on the
  latent continuation direction, correcting for the noise of s". (b) Corrected CI crosses ±0.015 but its larger
  absolute end is X < 0.03 → "on the estimated direction the data exclude 0.015; corrected for the reliability of a
  direction built from a median of two continuations (split-half λ = ..., a(2) = ...), they exclude turns of about X
  or more". (c) X ≥ 0.03, or an unbounded end (λ̂ = 0 in > 2.5 % of resamples) → "the reliability of s cannot support
  a bound on the latent direction; the 0.015 bound applies to the estimated direction only", and the abstract drops
  "0.015" for OLMo in favour of that sentence. (d) If the model checks fail badly (λ̂_pair on k ≥ 4 outside the
  bootstrap 95 % CI of the split-half λ̂, or the stratum ordering of L̄ opposite to the model's), the corrected
  interval is reported as model-dependent and only (c)'s weaker wording may be used.
- **R-A4 (per-arm changes).** All four per-arm changes go in the text with t intervals and Holm-4 p-values, whatever
  they are. ORACLE's own change is described as "the largest of the four" if it is, with its uncorrected and Holm p.
  If its Holm p < 0.05, the text says "the oracle arm, the only one with information about the future, moves the
  ideas along s by +x (Holm p = ...), but no more than plain revision does at the resolution of this test" (its
  contrast with NEUTRAL stays as read under R-A1).
- **R-A5 (question 8).** Holm-significant k = 2 vs k ≥ 3 difference for GEO or reviewer → the pooled "no turn of
  0.015" is qualified as heterogeneous and the stratum estimates go in the text. Otherwise the text reports the
  counts (83 with exactly two) and the stratum estimates in an appendix table, with the power caveat. A k ≥ 5 stratum
  (n = 23) t interval excluding zero for GEO or reviewer is reported as a flag, not read as a finding.

**What counts against the paper's current OLMo claim** ("distance and reviewer feedback add no turn of 0.015 or more
beyond plain revision"): any of R-A1 re-reading an arm as insufficient; R-A2 ρ_bound ≥ 0.5; R-A3 outcome (b) (claim
holds only on the estimated-direction scale, must be scoped), or (c) (claim cannot be stated as a bound on the
continuation direction); R-A5 a Holm-significant heterogeneity.

### What the paper will say

- Section 5.3, OLMo paragraph: the registered reading, then one sentence per R-A1 to R-A3 outcome as above, the four
  per-arm changes (R-A4) and the count of k = 2 contexts. The population sentence of the abstract is expanded from
  "selected for having an S2-flagged continuation" to "contexts with an S2-flagged consensus continuation and at least
  two continuations (8.3 % of those labelled)" in every outcome.
- Abstract and conclusion: "bounded at 0.015" for OLMo is kept only under R-A1 inside + R-A3 (a); with R-A3 (b) it
  becomes "bounded at 0.015 on the estimated direction (about X on the latent one)"; with any other outcome it is
  replaced by "not shown, and not bounded below X".
- Appendix: a table per embedder with strata × (n, L̄, ρ_bound, contrasts, per-arm changes), the reliability block
  (r̄, λ̂, a(k), model checks) and the Monte Carlo SEs.

### Cost, runtime, memory

No paid calls, no GPU. Local: two embedders × (pool load, 150 contexts, ≤ 46 × 200 split partitions, 20,000 × 150
resample matrix ≈ 24 MB, ~20 statistics plus a vectorised bisection per resample). Estimated 3-6 min with
`OMP_NUM_THREADS=4`; peak memory about 2 GB (the 113,000 × 1,024 corpus matrix as float32, 463 MB, plus its
future-split slice and the pool copy per context in `per_context`).

---

## Part B. M9: the single "detected" feedback effect, pooled and corrected

### Review concerns addressed

M9: in the gpt-4o replication GEO − NEUTRAL along s is +0.011 in both embedders, "insufficient" in Qwen3 (t CI
−0.0005 to +0.022) and "detected" in BGE-M3 (CI +0.001 to +0.021, uncorrected p = 0.03); the registered rule applied
Holm only to the equivalence leg, while "turn detected" was read from an uncorrected 95 % interval across six E-tests
in each of two embedders; in the discovery sample the same contrast was −0.003 in Qwen3. Requested: pooled
discovery + replication estimate per embedder, Holm across all E-tests of the replication family, and wording.

### Question

(i) Which "turn detected" readings of E1-E6 survive Holm over all twelve detection tests (6 E-tests × 2 embedders)?
(ii) What is the pooled (300-context) feedback-specific turn per embedder, and do discovery and replication differ?
(iii) Is the BGE-M3 / Qwen3 difference itself different from zero?

### Data (all confirmed to exist)

- Replication: `~/Documents/experiment-data/latent-gap-ideas/runs/critique-replication/` (`followups.jsonl`,
  `iter_ideas.jsonl`, `iter_ideas_neutral.jsonl`, `emb.npz`, `ideas_emb_qwen3e06.npy`, `emb_bgem3.npz`).
- Discovery 150: `kaggle-upload/continuation-texts/texts.parquet`, `kaggle-output/embed-continuation-v1/emb_{space}.npy`,
  `runs/continue-from-context/iter_ideas_neutral.jsonl`, `runs/continue-from-context/neutral_emb_{space}.npy`
  (loaded by `revision_neutral.load_gpt4o` / `neutral_rows`).
- Pool: `prepared/arxiv/emb_{space}.npy`, `prepared/arxiv/corpus.parquet`.
- Registered numbers: `docs/prereg/strengthen_critique_replication.json` (confirmatory E1-E6 with `t_ci95`, `se`;
  `secondary.pooled_with_original_150`), `docs/prereg/revision_neutral.json` (discovery N6).

### Exact computation

Per-context arrays are reproduced with exactly the calls of `strengthen_critique_replication.py` `analyse()` (new 150)
and `pooled()` (original 150), via `neutral_contrasts.per_context` / `change`. Check first: means and `t_ci95` of
E1-E6 and the pooled means reproduce the JSON to 1e-9 in both embedders; otherwise stop. (Bootstrap CIs of the old
JSON are not compared: they depend on the order the shared rng was consumed.)

- **B1. Holm over the detection family.** For E1-E3 (Δs_A) and E4-E6 (Δs_A − Δs_NEUTRAL), both embedders: two-sided
  one-sample t-test of zero (the test whose 95 % interval the registered rule reads), p_t. Primary family: the 12
  p_t, Holm, α = 0.05. Reported alongside: Holm within embedder (6), Holm over the 6 feedback-specific tests only
  (E4-E6 × 2), the same 12 with Wilcoxon p-values, and a sensitivity family of 28 (the 16 registered tests, R1, R2 and
  E1-E6 TOST as 2·p_TOST, × 2 embedders, plus the 12 detection tests).
- **B2. Pooled estimate.** For E4-E6 (and, descriptively, each arm's Δs), per embedder, over the 300 concatenated
  contexts: mean, t interval, 20,000-resample percentile bootstrap (`default_rng(20261106)`, one (20,000, 300) index
  matrix per embedder), and a sample-stratified bootstrap (150 drawn within each sample, same seed + 1). Discovery
  and replication means, each with its t interval; heterogeneity = replication − discovery, Welch t interval and p.
  Fixed-effect inverse-variance pool of the two sample means as a check. Holm over the 6 pooled contrasts (3 × 2
  embedders) on two-sided t p-values; each pooled interval also read against ±0.015.
- **B3. Embedder difference.** Per context, d_BGE − d_Qwen3 for E4 (GEO − NEUTRAL) in the replication and pooled
  samples: mean, paired t interval. Descriptive, for the "difference between detected and insufficient" point.

### Reading rules (fixed now)

- **R-B1.** An E-test is "turn detected" only if its Holm-12 p_t < 0.05; one with an uncorrected 95 % interval
  excluding zero but Holm-12 p_t ≥ 0.05 is "estimated +x (95 % CI ...), not significant after correction"; the
  equivalence readings (E3 BGE-M3 confirmed within ±0.015) are unchanged because they were already Holm-corrected.
  The paper's withdrawal of the discovery sample's "at most about 0.015" stays in every outcome, since it rests on
  the failed equivalence tests, not on detection.
- **R-B2.** Pooled GEO − NEUTRAL: Holm-6 p < 0.05 in both embedders → "pooled over 300 contexts, distance feedback adds
  a turn of about x beyond revision in both embedders"; in one embedder only → both estimates given, with no
  embedder-dependent claim unless B3's interval excludes zero; in neither → "not shown". Heterogeneity p < 0.05 for a
  contrast → the pooled estimate is reported but not used as the headline for it, and the two samples are given
  separately.
- **R-B3.** If B3's interval includes zero (expected, identical estimates), no sentence may contrast the two
  embedders for E4.

**What counts against the paper's current claim** ("detected only for distance feedback in one of two embedders
(+0.011)"): E4 in BGE-M3 not surviving Holm-12; or B3 including zero (the "one of two embedders" framing is then
unsupported whatever B1 gives). Either removes that clause.

### What the paper will say

- Under R-B1 failure for E4-BGE and B3 including zero (the review's expectation): abstract and conclusion replace "was
  detected only for distance feedback in one of two embedders (+0.011)" by "is about +0.01 for distance feedback in
  the replication in both embedders (not in the discovery sample in Qwen3) and not significant after correction";
  Section 5.3 gives the Holm-12 p-values for all twelve, the pooled estimates with their heterogeneity tests, and
  relabels E1-E3 "turns" by the same rule; the Appendix B table row is rewritten accordingly. The pooled BGE-M3 value
  is quoted as pooled and exploratory (two samples with different follow-up dating and vectors).
- If E4-BGE survives Holm-12: "detected in BGE-M3 (Holm p = ...), with the same estimate in Qwen3 (+0.011, not
  significant)", and B3 is reported so the reader sees the embedders do not differ.
- If the pooled GEO − NEUTRAL passes R-B2 in both embedders, the abstract may say "pooling both samples, about +x in
  both embedders"; otherwise the pooled numbers stay in Section 5.3 only.

### Cost, runtime, memory

No paid calls, no GPU. Local: `per_context` over 300 contexts × 2 embedders with hit distractors (same code path as
`pooled()`, which ran locally), 20,000-resample bootstraps on vectors of length ≤ 300. Estimated 5-15 min with
`OMP_NUM_THREADS=4`; peak memory about 2.5-3 GB (corpus matrix float32 plus not-past and future slices).

---

## Feasibility notes and limits (honest)

- Split-half reliability is only possible where k ≥ 4 (46 of 150 contexts); applying its λ̂ to the k = 2 and 3
  contexts assumes a common signal-to-noise ratio. The pairwise-cosine route (A3 check a, c) is the only way to
  estimate it at k = 2, and it rests on the same model; neither can separate "noisy labels" from "continuations that
  genuinely go in different directions", which the model treats the same way.
- The disattenuation corrects for sampling noise among the labelled continuations. It cannot correct for label
  errors that are systematic (e.g. a labeller consistently calling a kind of follow-up a continuation); no human
  relabelling is in scope.
- Question 8's heterogeneity test has low power (about 0.032 detectable at 80 %); only large differences can show.
- M9's Holm-12 family follows the reviewer's request; it is not the registered family, so it is a post hoc
  correction applied to a registered test, disclosed as such.
