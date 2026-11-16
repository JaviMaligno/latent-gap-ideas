# Review follow-up: Section 4 in embedding spaces that cannot have seen the future (pre-registered)

Draft pre-registration, written 2026-11-15 before any of the spaces below is fitted or embedded. To be committed alone
to `docs/prereg/review_leakfree_encoder.md` before the scripts run; deviations will be disclosed in a result section
appended later. No paid API, no LLM call, no GCP; Kaggle for compute.

## Review concern addressed

Reviewer minor 7: "Encoder leakage is acknowledged but untested; an encoder released before the splits… for Sec. 4
would close it cheaply." Qwen3-Embedding-0.6B and BGE-M3 were trained on text that may include the future splits
(Steam 2024+, arXiv 2023+, patents 2016+), so the map of the past might already be shaped by where the future went.
Section 4 is rerun in two spaces that cannot have seen the future.

## What I know before this registration

Every number of `where_future_lands_result.json`, `review_hole_validity.json` and Section 4 of the paper: in Qwen3 and
BGE-M3, F1 far share 0 everywhere (most isolated future item at max cosine 0.41-0.57); momentum direct AUC 0.73-0.92;
registered hole H0 direct AUC 0.45-0.54 and not valid (blind to planted holes, dominated by geometry); H1-H3 valid,
direct AUCs 0.30-0.66, none with CI above 0.5 in both embedders; F3 gains −0.016 to +0.006 in Steam/arXiv, patents
inconclusive; F2 partial ρ positive in Steam and arXiv, not in patents. Nothing has been computed in LSA or
all-mpnet-base-v2 space.

## Spaces

- **LSA-256** (all three corpora). Per corpus: `TfidfVectorizer(lowercase=True, stop_words="english", min_df=5,
  max_df=0.5, sublinear_tf=True, dtype=float32)` fitted on the `text` column of the non-adult **past** rows only
  (reference and held-out together; future rows never enter the fit); `TruncatedSVD(n_components=256,
  algorithm="randomized", n_iter=7, random_state=20261030)` fitted on the same past matrix; every row transformed
  and L2-normalised. Zero leakage by construction: no future text and no pretrained model.
- **mpnet** = `sentence-transformers/all-mpnet-base-v2` (released 2021; training data predates the arXiv 2023 and
  Steam 2024 splits). Steam and arXiv only. **Not valid for patents** (split 2016, the model's training data postdates
  it), so patents are not embedded and the paper says so. Embedded on Kaggle (both T4s, multi-process pool, the
  `kaggle/embed-continuation/embed.py` recipe) from the same `text` column, `normalize_embeddings=True`, fp16 output,
  `max_seq_length` left at the model's 384 tokens (its trained length; corpus texts are capped at 1,200 characters of
  abstract, so few are truncated).

Row alignment with `corpus.parquet` as for the existing embeddings; adult rows removed exactly as in every analysis.

## Analyses (reused functions, not reimplemented)

Script `scripts/review_leakfree_encoder.py`, kernels `kaggle/review-leakfree-lsa` (CPU) and
`kaggle/review-leakfree-mpnet` (GPU: embed, then analyse). Both import the repo scripts unchanged.

1. **Section-4 tests**: `where_future_lands.analyse(domain, space)` with the module's data root pointed at the new
   embedding files: F1, F2, F3, F4 (and F5 as the function computes it; F6 is Qwen-only and not run). Same SEED
   20261030, K = 256 cells, held-out 10 % of the past, momentum years Steam 2022-2023, arXiv 2021-2022, patents
   2014-2015, 200,000 Gaussian draws, B = 2,000.
2. **Gap-score validity**: `review_hole_validity.features` and `.tests` with spaces restricted to `raw`, K = 256,
   definitions H0-H3: V1 (correlations with density, centroid cosine to mean, log size, extent; R²_geom; partial ρ),
   V2 (52 planted cells, ball depletion vs random thinning), T1 (directly scored AUC for top-decile growth, CI over
   2,000 cell resamples), T2 (F3 gain with the cluster bootstrap of `revision_future_cells`), T3 (partial Spearman).
   Necessary adaptations (all mechanical): the reproduction check against the registered Qwen/BGE cache is skipped
   (there is nothing to reproduce in a new space); the cell-resample indices for K = 256 are
   `default_rng(SEED + 256).integers(0, 256, (2000, 256))` instead of the cached registered ones (which belong to the
   Qwen/BGE runs); the 128/512 resolutions, centred/whitened spaces and the T4/T5 variants are not run.

## Replication criteria (fixed now), per corpus × space

- **R1 momentum beats the gap scores**: (a) momentum's directly scored AUC exceeds that of H0 and of every definition
  valid in that space, and (b) no valid definition threatens C2 (T2 gain ≥ 0.03 with cluster-bootstrap lower bound
  > 0). As in the paper, a gain ≥ 0.03 with an interval including 0 is "inconclusive", not a failure.
- **R2 valid gap scores at or below chance**: no definition valid in that space has a T1 interval lying above 0.5.
- **R3 nothing in the far region (F1)**: future far share (max cosine to the reference past ≤ 0.15) ≤ 0.001.
  Because 0.15 was calibrated on the far targets of the Qwen space and cosines in LSA space run much lower, a
  calibrated secondary is also reported and decides R3 if the literal one fails only through calibration: future far
  share ≤ held-out past far share + 0.001 at the same threshold (new work does not land beyond the past's own
  isolation), together with the median isolation difference (future − held-out) and its CI.

**Validity in a new space** (the paper's rules, counts scaled to the number of corpora): a definition *sees planted
holes* if V2 AUC ≥ 0.75 with CI above 0.5 in ≥ 2 of 3 corpora (LSA) / 2 of 2 (mpnet); *tracks emptiness* if
ρ(hole, density) ≤ −0.3 with CI below 0 and partial ρ ≤ −0.2 with CI below 0 in the same counts; *dominated by
geometry* if R²_geom ≥ 0.5 without tracking emptiness in ≥ 2 of 3 / 2 of 2. Valid = sees planted holes, or tracks
emptiness without being dominated. If no definition is valid in a space, R1(b) and R2 are evaluated on H1-H3 (the
definitions valid in the paper) and the result says so.

**Verdict per space**: "replicates Section 4" if R1, R2 and R3 hold in every corpus of that space; otherwise
"replicates except …" naming the corpus and criterion. No multiplicity correction: the criteria are nulls and
corrections would make them easier to meet (as in `review_hole_validity.md`); uncorrected 95 % intervals throughout.
F2 (partial ρ), F4 (latitude direction) and the validity table are reported descriptively next to the paper's.

## What the paper says under each outcome

- **Both spaces replicate**: Section 4 / the limitations paragraph on encoder leakage say that in a TF-IDF/LSA space
  fitted on past text only (all three corpora) and with an encoder released before the splits (all-mpnet-base-v2,
  Steam and arXiv), momentum still beats the gap scores, valid gap scores stay at or below chance and nothing lands in
  the far region; the numbers go in an appendix table. Encoder leakage is then tested, not only acknowledged.
- **One space replicates**: both are reported; the text says which conclusion depends on the space.
- **A criterion fails in a leakage-free space**: the failing conclusion is qualified in Section 4 and the abstract as
  possibly dependent on encoders trained on text that includes the future, with the numbers.
- Patents: mpnet is not a leakage-free encoder for patents; only LSA covers them, and the text says so.

## Result, part 1: LSA-256 (2026-11-15)

Kaggle CPU kernel `kaggle/review-leakfree-lsa`: LSA fit 2.0 min, analysis 87.5 min (about 3 min F-tests, 2 min
features and 22-25 min hole tests per corpus). Output `docs/prereg/review_leakfree_encoder.json` (LSA cells only so far).
LSA fit: vocabulary 29,286 / 25,035 / 18,191 (Steam / arXiv / patents), SVD explained variance 0.195 / 0.184 / 0.240,
no zero rows.

| Corpus | F1 far share (min max-cos) | momentum direct AUC | H0 T1 | H1 / H2 / H3 T1 (CI) | F3 gain (H0) | F2 partial ρ | F4 |
|---|---|---|---|---|---|---|---|
| Steam | 0 (0.363) | 0.772 [0.675, 0.863] | 0.522 | 0.537 [0.408, 0.663] / 0.538 [0.404, 0.675] / 0.552 [0.419, 0.686] | +0.038 [−0.048, 0.149] | 0.156 [0.038, 0.274] | centre |
| arXiv | 0 (0.393) | 0.885 [0.823, 0.937] | 0.504 | 0.509 [0.396, 0.613] / 0.478 [0.377, 0.569] / 0.489 [0.377, 0.587] | 0.000 [−0.016, 0.017] | 0.217 [0.098, 0.321] | centre |
| patents | 0 (0.413) | 0.687 [0.573, 0.783] | 0.591 | 0.486 [0.348, 0.613] / 0.440 [0.305, 0.561] / 0.434 [0.299, 0.558] | −0.014 [−0.067, 0.054] | 0.213 [0.104, 0.328] | centre |

Validity in LSA: H0 not valid (V2 AUC 0.43-0.55, dominated by geometry in 3/3); H1, H2 and H3 valid (planted-hole AUC
0.87-0.97 / 0.86-0.88 / 0.70-0.85). T2 cluster gains of the valid scores: Steam +0.043 / +0.054 / +0.070 (all ≥ 0.03
with intervals including 0, so inconclusive), arXiv −0.000 to +0.011, patents −0.019 to +0.029.

**Decisions (LSA):** R1, R2 and R3 (literal) hold in all three corpora. **Verdict: replicates Section 4.** Differences
from the paper, descriptive only: F2 is positive in patents (0.21, CI excluding 0), whereas the paper finds it in
Steam and arXiv only; F4 points to the centre in all three; momentum is weaker in Steam (0.77 against 0.84-0.91).

Deviations: none for LSA. For mpnet, the registered GPU kernel failed at start-up (`run.py` had no `__main__` guard,
which the sentence-transformers spawn pool needs; now fixed but not re-run). Before that, it had queued for about 2.5 h
because both batch-GPU slots were taken. An identical CPU copy
(`kaggle/review-leakfree-mpnet-cpu`: fp32, single process) was started as a fallback; see part 2.

## Result, part 2: all-mpnet-base-v2 (2026-11-16)

The registered GPU kernel `kaggle/review-leakfree-mpnet`, with the `__main__` guard added and nothing else changed, was
re-run as version 3 (2026-11-15 22:17 UTC, finished 23:33 UTC, 60 min). It is the run reported. The CPU fallback hit
Kaggle's 12-hour session limit at 05:17 UTC on 2026-11-16 and produced no output; it is not used. Output merged into
`docs/prereg/review_leakfree_encoder.json` with `review_leakfree_encoder.py collect`.

| Corpus | F1 far share (min max-cos) | momentum direct AUC | H0 T1 | H1 / H2 / H3 T1 (CI) | H1 / H2 / H3 T2 gain (cluster CI) | F2 partial ρ | F4 |
|---|---|---|---|---|---|---|---|
| Steam | 0 (0.472) | 0.850 [0.744, 0.937] | 0.469 | 0.567 [0.428, 0.689] / 0.545 [0.433, 0.659] / 0.534 [0.425, 0.645] | −0.012 [−0.069, 0.047] / −0.011 [−0.060, 0.052] / −0.014 [−0.073, 0.041] | 0.125 [0.014, 0.240] | centre |
| arXiv | 0 (0.350) | 0.891 [0.829, 0.942] | 0.493 | 0.242 [0.162, 0.330] / 0.239 [0.150, 0.335] / 0.238 [0.156, 0.327] | −0.013 [−0.044, 0.009] / −0.014 [−0.052, 0.010] / −0.012 [−0.045, 0.011] | 0.185 [0.063, 0.294] | centre |

Validity in mpnet: H0 not valid (V2 AUC 0.49 / 0.46, dominated by geometry in 2/2); H1, H2 and H3 valid (planted-hole
AUC 0.91 / 0.99, 0.98 / 0.77, 0.94 / 0.83 in Steam / arXiv).

**Decisions (mpnet):** R1, R2 and R3 (literal) hold in Steam and arXiv; no gain reaches 0.03, so neither corpus is
inconclusive. **Verdict: replicates Section 4.** Descriptive differences: in arXiv the valid scores forecast top-decile
growth below chance (AUC about 0.24, intervals below 0.34), within the paper's range of "at or below chance"; F4 points
to the centre in both corpora (Holm p < 1e-9), as in LSA for arXiv and patents (LSA Steam: rank-biserial 0.006, Holm
p 0.80, no shift).

**Overall verdict: in both leak-free spaces the Section-4 conclusions replicate.** Deviations: none in the reported runs.
