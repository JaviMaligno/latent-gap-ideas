# Does novel combination beat momentum? (arXiv)

Pre-registered 2026-10-30, before any analysis that looks at future outcomes. Before writing this file I
inspected only the arXiv corpus schema, split sizes, per-year counts and text lengths (past 72,000 items
1990-2022, of which 11,163 in 2021 and 11,961 in 2022; future 41,000 items 2023-01-01 to 2026-10-15; no
adult rows). No atom, topic or combination has been computed yet.

## Why

`where_future_lands.md` showed that past geometry (holes, distance) does not forecast where the future
grows, while momentum does. "The Alien Space of Science" (arXiv 2603.01092) makes a different and stronger
claim: decompose papers into idea atoms, score three-atom combinations with a coherence model and an
availability model, and a top 0.875 % of the candidate space recovers held-out 2025 combinations at 35.9x
chance; weighting unavailability raises recovery of *strictly novel* combinations (atoms that never
co-occurred before) from 2.35x to 3.24x chance. Their chance level is uniform random. They report no
popularity or momentum baseline in the temporal holdout. The open question: does a combination signal
forecast what appears better than momentum, or add to it? And does our embedding hole score add anything
once combination signals are in?

## What differs from Alien Space (stated up front)

- **No LLM atom extraction.** Atoms are NMF topics on TF-IDF of title + abstract, fitted on the past only.
  A paper's atom set is its top 3 topics by NMF weight. This is a cheap proxy for "conceptual units".
- **Coherence is a co-occurrence model** (mean pairwise PMI over the past), not a trained Transformer.
- **No availability model**: the corpus has no authors. "Rarity" of the triple in the past (−log(1 + past
  count)) stands in for unavailability. It is a weaker, different signal, and I say so in the result.
- Corpus is arXiv cs.* (all subfields) sampled as stored, not 16k LLM-venue papers; holdout is four years
  (2023-2026), not one.

## Atoms (frozen)

- `TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=20, max_df=0.5, max_features=20000,
  sublinear_tf=True)` fitted on past `text`.
- `NMF(n_components=V, init="nndsvda", max_iter=200, random_state=SEED)` fitted on the past TF-IDF.
  Primary V = 200 (C(200,3) = 1,313,400 candidate triples, close to Alien Space's 1,143,135). Robustness
  V = 100.
- Atom set of a paper = indices of its 3 largest NMF weights (`transform` with the past-fitted model for
  future papers). Papers with fewer than 3 non-zero weights have no triple and are excluded (count
  reported).
- Candidates: all C(V,3) triples (primary) and all C(V,2) pairs (secondary).

## Two windows (to fit combined models without touching the future)

- **Pseudo window (training):** features from papers dated ≤ 2020, "recent" = 2019-2020; outcome = the
  triple is the atom set of at least one paper dated 2021-2022. The NMF/vocabulary is the one fitted on all
  past (a mild leak inside the training window only; it never sees the future).
- **Test window:** features from all past papers (≤ 2022), "recent" = 2021-2022 (same as
  `where_future_lands` momentum years for arXiv); outcome on the future split.

## Scores per candidate triple S = {a, b, c} (from the window's feature papers only)

- `n_i` past papers whose atom set contains atom i; `c_ab`, `c_S` the same for pairs and the triple;
  `rec` superscript = same counts restricted to the recent years. N = number of feature papers.
- **POP** (static popularity, descriptive): Σ log(n_i + 1).
- **Momentum features**: `m_atom` = Σ_i log(n_i^rec + 1), `m_pair` = Σ_pairs log(c_pair^rec + 1),
  `m_trip` = log(c_S^rec + 1). Unfitted momentum score **M** = z(m_atom) + z(m_pair) + z(m_trip) (z over
  candidates).
- **COH** (coherence): mean over the 3 pairs of PMI = log((c_ab + 1) · N / ((n_a + 1)(n_b + 1))).
- **RAR** (rarity, unavailability stand-in): −log(1 + c_S).
- **NC_β** (novel-combination score) = z(COH) + β · z(RAR), with β = 0 (coherence only) and β = 0.4
  (Alien Space's reported best weight, transferred as is, not tuned).
- **HOLE** (embedding): atom centroid = normalised mean embedding of the window's feature papers whose atom
  set contains the atom; triple point = normalised mean of its 3 atom centroids; HOLE = the
  `where_future_lands` hole score (k-means K = 256 on the window's feature papers, 200,000 draws from their
  Gaussian N(μ, Σ), share of draws / share of items) of the cell the triple point falls in. Computed with
  Qwen3-Embedding-0.6B and with BGE-M3.
- **Fitted models** (logistic regression on standardised features, `max_iter=2000`, trained on the pseudo
  window over all candidates, applied unchanged to the test window):
  `FIT_M` = {m_atom, m_pair, m_trip}; `FIT_MC` = FIT_M + {COH, RAR}; `FIT_MCH` = FIT_MC + {HOLE}.
  For pairs, the same with the pair analogues (m_trip and HOLE use the pair; COH = the pair's PMI).

## Outcomes and metrics (test window)

- **All targets**: candidate triples that are the atom set of ≥ 1 future paper.
- **Strictly novel targets**: targets with c_S = 0 in the test-window feature papers (all past).
- **Enrichment@K** (Alien Space style): rank all N candidates by a score, K = round(0.00875 · N); hits =
  targets in the top K; chance = K · n_targets / N; enrichment = hits / chance. For strictly novel targets
  this is computed two ways: (a) on the full ranking, as Alien Space does; (b) ranking only the strictly
  novel candidates (K = round(0.00875 · N_novel)), which asks which novel combinations appear rather than
  whether a score prefers novel ones.
- **AUC** over candidates (threshold-free), for all targets and within strictly novel candidates.
- Ties in a score are broken by a fixed random permutation (seed SEED).

## Hypotheses and predictions (triples, V = 200, primary)

**H1, the combination proxy works (replicates the Alien Space pattern).** COH alone gives enrichment@K on
all targets > 5x chance, bootstrap CI lower bound > 1. Prediction: holds.

**H2, momentum beats novel combination on all targets.** M ≥ NC_0 and M ≥ NC_0.4 in both enrichment@K and
AUC on all targets. Prediction: holds.

**H3a, rarity buys novel hits on the full ranking (Alien Space's effect).** On strictly novel targets with
the full ranking (a), enrichment of NC_0.4 > NC_0. Prediction: holds (rarity favours unseen triples
partly by construction; reported as such). Also reported: NC_0.4 vs M on (a), no prediction.

**H3b, among strictly novel candidates, momentum still beats coherence.** Within strictly novel candidates
(b), AUC of M ≥ AUC of COH. Prediction: holds.

**H4, combination signals add little to momentum.** AUC(FIT_MC) − AUC(FIT_M) < 0.02 on all targets, and
< 0.02 within strictly novel candidates. Prediction: holds.

**H5, the embedding hole adds nothing once combination signals are in.** AUC(FIT_MCH) − AUC(FIT_MC)
< 0.01 on all targets, in both embedders (replication rule: decision must match in Qwen and BGE).
Prediction: holds.

Secondary (same rules, reported with direction only, no new hypotheses): pairs with V = 200; triples and
pairs with V = 100. POP is reported next to M everywhere.

## Analysis rules

- Bootstrap: 2,000 resamples of future papers (seed SEED); each resample recomputes the target sets and
  every metric with the scores fixed; 95 % percentile CIs on every enrichment, AUC and difference.
- Paired bootstrap p-values for H2 (AUC M − NC_0.4), H3b, H4 (all targets), H5 (Qwen):
  p = 2 · min(P(diff ≤ 0), P(diff ≥ 0)); Holm within the family {H2, H3b, H4, H5}. H1 and H3a are
  descriptive with fixed thresholds. A threshold decision uses the point estimate; the CI is reported next
  to it.
- SEED = 20261030 everywhere. No parameter in this file changes after the first result is seen. Anything
  extra goes into a section labelled "exploratory".

## Cost, runtime and pre-decided limits

Local CPU only (OMP_NUM_THREADS=6), no LLM calls, no gateway, no paid services, no downloads. Estimate:
NMF 5-20 min per V, k-means/hole 4 fits × ~2 min, scoring and logistic fits on ~1.3M candidates a few
minutes, bootstrap a few minutes. Target < 1.5 h total. **Pre-decided reductions**, applied in this order
only if the smoke test projects more than 1.5 h: (1) drop the V = 100 robustness arm; (2) NMF
`max_iter=100`; (3) bootstrap 1,000 resamples. Whatever is applied is logged in the result.

## Outputs

- `scripts/combination_vs_momentum.py`
- `docs/prereg/combination_vs_momentum_result.json` and a Result section appended to this file.
- Log: `~/Documents/experiment-data/latent-gap-ideas/runs/combination_vs_momentum.log`.

## Result (run 2026-10-30, `combination_vs_momentum_result.json`)

Run facts: 40 min total (`runtime_s` 2413). Reduction (1) was applied before the run (the V = 100 arm was
dropped) because the full-scale NMF timing test projected about 1.5 h; the real run took 40 min, so the
projection was pessimistic. I keep the pre-registered rule and do not add the V = 100 arm here. NMF stopped
at the pre-registered `max_iter=200` with a convergence warning (log). Every future paper had a triple
(0 excluded). Triples: N = 1,313,400 candidates, K = 11,492; 24,312 distinct future targets, 12,908 of them
strictly novel (never the atom set of a past paper); 15,580 positives in the pseudo window.

Primary arm (triples, V = 200). Enrichment = hits in the top 0.875 % over uniform chance.

| score | enrichment, all targets | AUC, all | novel targets, full ranking (a) | novel, within novel (b) | AUC within novel |
|---|---|---|---|---|---|
| POP (static popularity) | 28.2 | 0.961 | 9.6 | 23.5 | 0.953 |
| M (momentum, unfitted) | 29.1 | 0.969 | 0.0 | 30.9 | 0.962 |
| COH (PMI coherence) = NC_0 | 0.0 | 0.164 | 0.0 | 0.0 | 0.140 |
| NC_0.4 (coherence + rarity) | 0.0 | 0.086 | 0.0 | 0.0 | 0.140 |
| RAR (rarity alone) | 0.50 | 0.272 | 0.94 | 0.90 | 0.50 (constant) |
| HOLE, Qwen / BGE | 1.43 / 2.30 | 0.622 / 0.693 | 1.87 / 2.36 | 1.87 / 2.39 | 0.618 / 0.696 |
| FIT_M (fitted momentum) | 34.2 | 0.970 | 8.3 | 31.0 | 0.961 |
| FIT_MC (+ COH, RAR) | 34.5 | 0.970 | 6.2 | 30.9 | 0.961 |
| FIT_MCH, Qwen / BGE (+ HOLE) | 34.5 / 34.6 | 0.970 / 0.970 | 6.4 / 5.9 | 30.6 / 30.9 | 0.961 / 0.961 |

Decisions:

| | prediction | measured | decision |
|---|---|---|---|
| **H1** COH enrichment > 5x | holds | 0.0x (AUC 0.164, below chance) | **fails** |
| **H2** M ≥ NC_0 and NC_0.4 (enrichment and AUC) | holds | AUC 0.969 vs 0.164 / 0.086; enrichment 29.1 vs 0 / 0 | holds (but see reading) |
| **H3a** NC_0.4 > NC_0 on novel targets, full ranking | holds | 0.0 vs 0.0 | **fails** |
| **H3b** within novel, AUC M ≥ COH | holds | 0.962 vs 0.140 | holds |
| **H4** FIT_MC − FIT_M < 0.02 (all; within novel) | holds | +0.0002; −0.0001 | holds |
| **H5** FIT_MCH − FIT_MC < 0.01 (Qwen; BGE) | holds | −0.0003; −0.0000 | holds, replicated |

Holm-adjusted bootstrap p for H2, H3b, H4, H5: all < 0.001 (0 of 2,000 resamples on the other side); the
H4 and H5 differences are significant only in the sense of being reliably tiny, far below the thresholds.

Secondary arm (pairs, V = 200, 19,900 candidates) has the same decisions, but it is saturated: 6,781 of
19,900 pairs (34 %) appear in the future, so the maximum possible enrichment is 2.93x, which M, POP and every
fitted model reach. Only the AUCs separate scores there (M 0.982, COH 0.075, FIT_MC − FIT_M +0.0015).

Reading:

- **Momentum is the forecaster, again.** Recent activity of the atoms and of their pairs ranks the combinations
  that appear in 2023-2026 with AUC 0.97 and 29-34x chance enrichment in the top 0.875 %. Static popularity
  alone already gets 28x. For comparison only (different atoms, corpus and holdout), Alien Space reports 35.9x
  with a trained coherence model and no popularity baseline; a chance-only reference makes such numbers look
  stronger than they are.
- **Strictly novel combinations are also forecast by momentum.** Within never-seen triples, M reaches 30.9x
  and AUC 0.962, and fitted momentum puts enough novel triples in the overall top K to get 8.3x on the full
  ranking. Alien Space's reported strictly-novel figures are 2.35-3.24x; I cannot claim that their models would
  not beat momentum on their data, only that on ours a momentum baseline alone exceeds those magnitudes.
- **My coherence proxy failed (H1), so H2 is a weak win.** Smoothed PMI rewards pairs of rare atoms, which is
  anti-predictive (AUC 0.164). H2 holds against a broken proxy, which says nothing about Alien Space's trained
  coherence model. H4 is the stronger test: a logistic model can use COH and RAR with any sign, and they add
  +0.0002 AUC to momentum (+0.0015 for pairs).
- **Rarity buys nothing here (H3a fails).** Adding rarity to a coherence score that never reaches the top K
  leaves 0 novel hits; rarity alone is at or below chance (enrichment 0.50 all, 0.94 novel). In the fitted
  model, adding COH and RAR lowers novel enrichment on the full ranking (8.3 to 6.2). The unavailability effect
  Alien Space reports needs author data I do not have; my stand-in is not the same signal.
- **The embedding hole, alone, is weakly positive here but adds nothing (H5).** HOLE of a triple's point has AUC
  0.62 (Qwen) / 0.69 (BGE), unlike the cell-level result in `where_future_lands` (0.39-0.48 for growth). Once
  momentum and combination signals are in, it adds −0.0003 / −0.0000 AUC, in both embedders.

Limitations:

- **Bootstrap CIs are biased for set-based metrics.** Resampling papers with replacement keeps about 63 % of
  distinct papers, which drops rare targets and shifts enrichment; several 95 % intervals in the JSON exclude
  their own point estimate (e.g. POP enrichment 28.2, interval 31.1-31.8). Point estimates are the
  pre-registered decision basis and none of the decisions is near a threshold, but the intervals and
  bootstrap p-values should not be quoted as calibrated. A paper version needs a different resampling scheme
  (e.g. by target, or a split-half future).
- Atoms are NMF topics, not LLM-extracted conceptual units; several are generic ("state art", "propose novel").
  The top-3 rule gives one triple per paper. Coherence is PMI, not a trained sequence model; no availability
  model (no authors). arXiv cs.* sample, four-year holdout. The V = 100 robustness arm was not run.
- The pseudo-window fit uses an NMF vocabulary fitted on all past papers, including the 2021-2022 outcome years
  of that window (a leak inside training only; the test window never sees the future).
