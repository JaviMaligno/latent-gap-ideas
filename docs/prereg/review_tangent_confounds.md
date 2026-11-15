# Pre-registration (review round): tangent confounds, the gap yardstick, prompt echo and the context sample (Section 5)

Status: DRAFT for commit. To be committed to `docs/prereg/review_tangent_confounds.md` (Part A) and
`docs/prereg/review_rewrite_null.md` (Part B, paid) **before** any of the computations below is run and before any
generation call of Part B. Deviations found while implementing are disclosed in a dated "Clarification" section
appended after the commit, as in earlier registrations.

What I know before this analysis: every published number of Section 5 and Appendix A.8 (Tables 2, 2b, 3, 4, A6 and
the JSONs `revision_direction_tangent.json`, `revision_neutral.json`, `strengthen_critique_replication.json`,
`strengthen_olmo_direction.json`, `strengthen_olmo_neutral.json`, `strengthen_modern_neutral.json`,
`strengthen_continuations_full.json`), and the independent peer review. While designing this I only inspected file
formats, row counts and array shapes (listed under Data); I did not compute any projection, overlap, shared-ID count
or citation quantile. In particular the number of follow-ups shared between contexts (minor 11), the n-gram overlaps
(question 9) and the citation quantiles (minor 10) are unknown to me.

Two parts:

- **Part A (no generation, no GPU, no paid call)**: analyses T1-T6 on existing vectors and texts.
  Script `scripts/review_tangent_confounds.py`, output `docs/prereg/review_tangent_confounds.json`.
- **Part B (paid generation + free Kaggle GPU)**: the rewrite/paraphrase study for M3(b) and M4.
  Scripts `scripts/review_rewrite_generate.py`, `kaggle/review-rewrite/run.py` (OLMo), `scripts/review_rewrite_null.py`
  (analysis), output `docs/prereg/review_rewrite_null.json`.

---

## 1. Review concerns addressed

| Concern (peer review) | What is asked | Here |
|---|---|---|
| M1(b) + question 4 | Every turn as a fraction of the idea-to-follow-up gap along s, per generator and embedder; the follow-ups' (and continuations') own leave-one-out projection on s | T1 |
| M3(a) + question 5 | Per-context regression of the change along s and p on the change in step, within arm; arm contrasts adjusted for step change; correlation of step change and s change | T2 |
| M3(c) | Remove a genre direction (mean idea minus mean real abstract), recompute tangents, Table 3 and the replication contrasts | T3 |
| Question 9 | n-gram overlap of ideas vs real continuations with the context abstract | T4 |
| Minor 11 | Follow-ups shared between contexts (non-independence for the bootstrap) | T5 |
| Minor 10 | Citation counts of sampled contexts vs the frame | T6 |
| M3(b) | Paraphrase null for the tangent at matched step | Part B, P2 |
| M4 | Generator rewrites follow-ups/continuations as proposals in the idea schema; gap between ideas and rewritten continuations | Part B, P1 |

Scope: the discovery sample (gpt-4o and OLMo, Section 5.2-5.3), the gpt-4o confirmatory replication, the OLMo
confirmatory test, and, as secondary cells where the existing data allow, gpt-5.4-mini (Section 5.4). Both embedders
(Qwen3-Embedding-0.6B primary, BGE-M3 secondary) throughout.

---

## 2. Cells and data (paths confirmed to exist on 2026-11-14)

Root `D = ~/Documents/experiment-data/latent-gap-ideas`. Corpus: `D/prepared/arxiv/corpus.parquet` and
`D/prepared/arxiv/emb_{qwen3e06,bgem3}.npy` (113,000 × 1024, float16; the direction pool is the non-adult
`split == "future"` rows, as in `direction_tangent.py`).

| Cell id | Study | Generator | Contexts / follow-ups | Ideas and vectors | Loader to reuse unchanged |
|---|---|---|---|---|---|
| DISC-4o | Discovery (5.2-5.3) | gpt-4o | `D/runs/followups/followups.jsonl` (first 150 with ≥ 5 follow-ups, `continue_from_context.contexts()`), 2,883 follow-ups | critique: `D/runs/continue-from-context/iter_ideas.jsonl` (GEO/LLM/ORACLE rows only); one-shot: `.../ideas.jsonl` (CTX arm, 5 per context); vectors `D/kaggle-output/embed-continuation-v1/emb_{space}.npy` (10,083 rows) with kinds from `D/kaggle-upload/continuation-texts/texts.parquet`; NEUTRAL `.../iter_ideas_neutral.jsonl` + `.../neutral_emb_{space}.npy` (1,350 × 1024) | `continuation_reanalysis.load_generator("gpt-4o", space)` via `revision_direction_tangent.load_generator`; NEUTRAL via `revision_neutral.load_gpt4o` |
| DISC-OLMo | Discovery | OLMo-7B-0424 | same 150 | `D/kaggle-output/llama-continuation-v1/{iter_llama.jsonl, ideas_llama.jsonl, emb_{space}.npy}` (1 idea per context and arm); NEUTRAL `D/kaggle-output/olmo-neutral-v1/{iter_olmo_neutral.jsonl, neutral_emb_{space}.npy}` | `load_generator("olmo-7b-0424", space)`; NEUTRAL as `strengthen_olmo_neutral.py` |
| REP-4o | gpt-4o replication | gpt-4o | `D/runs/critique-replication/followups.jsonl` (150, 2,419 follow-ups) | `.../iter_ideas.jsonl` (4,050 rows), `.../iter_ideas_neutral.jsonl` (1,350); Qwen3: `.../emb.npz` (C, F, owner) + `.../ideas_emb_qwen3e06.npy` (5,400); BGE: `.../emb_bgem3.npz` (ctx, fup, ideas 5,400) | `strengthen_critique_replication.ctx_rows/idea_rows` + the vector assembly of its `analyse()` |
| OLMO-C | OLMo confirmatory | OLMo-7B-0424 | `D/runs/olmo-direction/{eligible.json, followups.jsonl, citing_all.jsonl}`: 150 eligible contexts, 4,134 follow-ups, 485 consensus continuations (83 contexts with exactly 2) | `D/kaggle-output/olmo-direction-w{1,2}p1/{iter_olmo.jsonl, run_info.json, ctx_/fup_/iter_{space}.npy, fup_owner.npy}` (3 ideas × 4 arms × 3 rounds) | `olmo_direction_analysis.load_outputs`, `per_context` |
| MOD (secondary) | 5.4 | gpt-5.4-mini | `D/runs/followups-2025/` via `LG_FUPS=followups-2025` (150, 2,331) | `D/runs/continue-modern/{iter_ideas.jsonl, iter_ideas_neutral.jsonl, emb.npz, oracle_history_emb_GEO_LLM_ORACLE_ORACLE-H.npy, neutral_emb_{space}.npy, ideas.jsonl, partB_emb.npy}` | `strengthen_modern_neutral.py` loaders |

Continuation labels for DISC: `docs/prereg/strengthen_continuations_full_labels.json` (2,883 rows; consensus =
both `gpt61sol` and `opus55` equal `CONTINUES`, 112 pairs in 47 contexts, as `strengthen_continuations_full_analysis.py`).
REP has no continuation labels (T1 and P1 use follow-ups only there).

**Reproduction gate (before any new quantity is computed).** The script first recomputes, with the loaders above and
no change, the published quantities it builds on, and stops if any differs by more than 5 × 10⁻⁴: Table 3 change along
s per arm and cell (`revision_direction_tangent.json`, mode `saved`); N5/N6 for DISC-4o (`revision_neutral.json`); the
E1-E6 means of REP-4o (`strengthen_critique_replication.json`); the three primary OLMO-C contrasts
(`strengthen_olmo_direction.json`); the DISC-4o B1 step gap (0.149). A failed gate is fixed and disclosed before
proceeding; the new estimands are not looked at until the gate passes.

**Common definitions (unchanged from the paper).** Vectors L2-normalised. Tangent of x at context c:
T(x) = unit(x − (x·c)c). f = unit mean tangent of the follow-ups (OLMO-C: of the consensus continuations);
p = unit mean tangent of the k nearest future-split papers to c outside the excluded set (k and the excluded set exactly
as in each cell's original script); s = unit(f − (f·p)p). Per context and arm, the round-r value of a measure is the
mean over that context's ideas with an abstract and a finite tangent. step = cos(x, c); Δstep = step_r0 − step_r2
(positive = longer); Δy = y_r2 − y_r0 for y ∈ {align = T·f, along p = T·p, along s = T·s}. Arms: GEO, LLM (Reviewer),
ORACLE, NEUTRAL (NEUTRAL shares GEO's round-0 ideas). ORACLE-H (Table 4) is included only in T1.

**Common inference.** Unit = context. Percentile bootstrap over contexts, **10,000 resamples, seed 20261114**
(numpy `default_rng`; one generator per cell, drawn in the fixed order cells → analyses as listed in this file),
chosen larger than the paper's 2,000 because the review (M8) notes Monte Carlo error near thresholds. Ratios are
bootstrapped by resampling numerator and denominator together. Wilcoxon signed-rank (scipy default, two-sided unless
stated). Holm families are stated per analysis.

---

## 3. T1 — Turns as a fraction of the gap along s (M1(b), question 4)

**Question.** How large is the idea-to-follow-up distance along s, measured with the real follow-ups' own
leave-one-out (LOO) projection, and what fraction of it does each arm (and each arm beyond NEUTRAL) close in two
revisions?

**Computation, per cell × embedder.**

1. Follow-ups' LOO projection on s. For context i and follow-up j among those that build f: f₋ⱼ = unit mean tangent of
   the other follow-ups; s₋ⱼ = unit(f₋ⱼ − (f₋ⱼ·p)p) with p unchanged; FUᵢ = meanⱼ T(xⱼ)·s₋ⱼ. Contexts with a single
   follow-up in the f-set are dropped (none in DISC/REP/MOD; none in OLMO-C, minimum 2 continuations).
   OLMO-C: the f-set is the consensus continuations; also reported for its non-continuation follow-ups projected on the
   full s (they are not in f).
2. Continuations' projection (DISC-4o, DISC-OLMo, 47 contexts): consensus continuations j projected on s₋ⱼ (s built
   from all follow-ups excluding j); CONTᵢ = mean over the context's continuations. Background follow-ups (both labels
   BACKGROUND) likewise, BGᵢ.
3. Ideas' round-0 projection I0ᵢ = mean of T·s over the round-0 ideas of GEO, LLM and ORACLE (independent draws).
4. Gap Gᵢ = FUᵢ − I0ᵢ; G = meanᵢ Gᵢ. Also reported: G_cont = mean(CONTᵢ − I0ᵢ) on the 47 contexts.
5. For each arm A: own turn fraction φ_A = mean Δs_A / G; feedback-specific fraction ψ_A = mean(Δs_A − Δs_NEUTRAL) / G
   (A ∈ GEO, LLM, ORACLE; ORACLE-H − ORACLE for DISC-4o and MOD). Paired by context; contexts missing an arm dropped
   from that ratio (counted).
6. The same for alignment: G_align = mean(FU_align,LOO,ᵢ − I0_align,ᵢ), with FU_align,LOO the paper's positive control
   (0.468/0.525 in DISC); fractions of Δalign.
7. The paper's bound expressed on this scale: 0.015 / G per cell.

Outputs: G, G_cont, FU, CONT, BG, I0 (means and bootstrap CIs), φ_A and ψ_A with 95 % bootstrap CIs, 0.015/G.

**Reading rule (fixed).** T1 is descriptive; it gives "barely" a denominator. The words the paper may use are tied to
the confirmatory cells (REP-4o and OLMO-C, Qwen3) and fixed now:

- ψ_A (feedback beyond revision): "barely / little" is kept for arm A only if the 95 % upper bound of ψ_A is below
  **0.25** in both confirmatory cells where A exists. If the point estimate is ≥ 0.10 with a CI excluding 0 in a
  confirmatory cell, the paper states "feedback closed about X % (CI) of the gap beyond revising" for that arm and drops
  "barely" for it. Otherwise (upper bound ≥ 0.25, CI including 0) the paper says the feedback-specific turn is not
  resolved and may be as large as the upper bound.
- φ_A (turn including revision): reported in the main tables. **Counts against the current claim**: if φ_A ≥ 0.33
  (point) for any critique arm in a confirmatory cell, the headline "distances barely steer" must be reworded to
  separate the two things: "two revisions close about X % of the gap along s, most of which plain revision also does;
  feedback adds Y %".
- The sentence "0.015 is about a sixth of the distance" is replaced by the measured 0.015/G per generator.
- If CONT is below I0 (continuations project less on s than the ideas) in DISC, the paper says the all-follow-up
  direction is not where the continuations go and qualifies Table 3 accordingly.

**Paper text by outcome.** The new column "fraction of the gap" enters Tables 3 and 4 and the replication paragraph;
abstract and conclusion scope "barely" to "two rounds of feedback" (M1(c)) and quote the ψ range. If the
against-the-claim condition holds, Section 5's title changes from "Distances barely steer a continuation" to a
descriptive one ("Revision moves the ideas part of the way; distance feedback adds little or an unresolved amount").

---

## 4. T2 — Is the turn a by-product of lengthening the step? (M3(a), question 5)

**Question.** Within each arm, do contexts whose ideas lengthen the step more also turn more along s, p and f, and do
the arm contrasts survive adjustment for different step changes? Does tangent alignment depend on step even without
revision?

**Computation, per cell × embedder.**

(i) Within-arm dose-response (question 5). For each arm A (incl. NEUTRAL), across contexts: Pearson r and Spearman ρ
between Δstep and Δy (y ∈ s, p, align), and OLS Δy = a_A + b_A·Δstep with HC3 standard errors. Report b_A (HC3 t-CI and
bootstrap CI), a_A (the turn predicted at zero lengthening), and the share of the observed turn the step predicts,
π_A = b_A·mean(Δstep_A) / mean(Δy_A) (bootstrap CI; reported as "not defined" when |mean Δy_A| < 0.002).
Holm over the 12 slope tests (4 arms × 3 measures) within each cell × embedder (DISC-OLMo: 12; MOD: 12).

(ii) Step-adjusted arm contrasts (primary of T2). Stack context × arm rows (GEO, LLM, ORACLE, NEUTRAL) and fit
Δs_{i,A} = μ_A + u_i + b·Δstep_{i,A} + e (arm effects μ_A, context fixed effects u_i, one common slope b; OLS).
The adjusted feedback-specific turns are μ_A − μ_NEUTRAL. CIs: 95 % cluster bootstrap over contexts (10,000) and the
cluster-robust (by context) t-interval. Same for along p and alignment. Secondary: the arm-specific-slope version
(b_A per arm, contrasts evaluated at the NEUTRAL mean Δstep), and the paired-difference regression
(Δs_A − Δs_N) = α + β·(Δstep_A − Δstep_N), α reported.
Holm: the adjusted contrasts are read with the same family and rule as the original contrasts of each cell
(DISC-4o: N6 family of 14 → here the 3 along-s contrasts plus 3 alignment contrasts, Holm over 6; REP-4o: E4-E6 with the
registered ±0.015 interval rule; OLMO-C: the 3 primary contrasts, Holm over 3, ±0.015).

(iii) Cross-section without revision. Among round-0 ideas only (GEO, LLM, ORACLE draws; up to 9 per context in DISC-4o,
REP-4o, OLMO-C; 3 in DISC-OLMo), idea-level OLS of T·s (and T·p, T·f) on (1 − step) with context fixed effects; slope
with cluster-bootstrap CI. The same regression for the real follow-ups (LOO projection on s₋ⱼ vs 1 − step) gives the
reference slope among real papers.

**Reading rule (fixed).**

- **Counts against the current claim** ("tangent alignment rises after critique in all 12 cells" read as a turn, and
  "step-invariant by construction" read as independence from step): in the primary cells (DISC-4o, REP-4o, OLMO-C,
  Qwen3), if the common slope b for alignment is positive with a CI excluding 0 **and** the step predicts at least half
  of the alignment rise (π ≥ 0.5) for a majority of critique arms. Then the paper states that the rise in alignment
  is mostly what a longer step predicts and cannot be separated, with these data, from leaving the paraphrase regime;
  "step-invariant by construction" is rewritten as "invariant to radial scaling by construction; empirically it
  co-varies with step (b = …)".
- Adjusted contrasts: each adjusted feedback-specific turn is classified with the cell's original rule (REP-4o and
  OLMO-C: equivalent within ±0.015 / detected / insufficient; DISC-4o: Holm-significant or not). If any classification
  changes relative to the unadjusted contrast, the paper reports both, gives the adjusted one as the step-controlled
  reading and says which statement changes. If none changes, one sentence and an appendix table.
- If (iii) shows a positive within-context slope for round-0 ideas (CI excluding 0) larger than the real follow-ups'
  slope, the paper says that ideas further from their context are more aligned even before any revision, which supports
  the review's mechanism; the turn is then described as "partly a step effect" even if the adjusted contrasts are
  unchanged.
- If b's CI includes 0 and |b·mean Δstep| < 0.005 in every primary cell, the paper states that the turns are not
  explained by step change (one sentence, numbers in the appendix).
- Caveat stated in all outcomes: a positive b is also what genuine turning produces when longer moves carry more
  content, so b > 0 limits the claim rather than refuting a turn; measurement error in Δstep attenuates b, so the
  adjustment is conservative (it under-adjusts).

---

## 5. T3 — Remove a genre direction and recompute (M3(c))

**Question.** Do Table 3, the DISC-4o contrasts with NEUTRAL, the replication contrasts and the OLMo confirmatory
contrasts survive removing the mean difference between generated proposals and real abstracts?

**Genre direction (fixed).** Per cell × embedder (a cell = one study and one generator; the direction is never shared
between cells or embedders):

- Primary: g = unit(mean(I) − mean(R)), where I = all critique-arm idea vectors of the cell (GEO, LLM, ORACLE; rounds
  0, 1, 2; NEUTRAL excluded because in DISC-4o its Qwen3 vectors come from a different encoder build, and to keep g
  identical whether or not NEUTRAL is analysed) and R = the cell's context vectors ∪ all its saved follow-up vectors
  (real abstracts). Means of unit vectors, unweighted over items.
- Secondary: g_ctx = unit(mean(I) − mean(contexts)); g_fu = unit(mean(I) − mean(follow-ups)); leave-one-context-out
  primary g₋ᵢ (context i's ideas, context and follow-ups removed from both means, applied to context i only); and, for
  DISC-4o only, g from the one-shot CTX ideas instead of the critique arms.
- Removal: every vector v (contexts, follow-ups, all ideas incl. NEUTRAL, pool papers) → v' = unit(v − (v·g)g). For the
  pool, the nearest-k ranking uses v'·c' = (v·c − (v·g)(g·c)) / (√(1 − (v·g)²)·√(1 − (c·g)²)), computed without
  materialising the projected pool.
- Then every quantity is recomputed exactly as in the original cell scripts (f, p, s, tangents, Δ's, tests).
- Diagnostics: ‖mean(I) − mean(R)‖; mean cos(g, idea tangent) at rounds 0 and 2; mean cos(g, follow-up tangent); the
  B1 step gap of DISC-4o and the round-0 step gap of the other cells after removal.

**Recomputed tables.**

1. Table 3 (DISC-4o and DISC-OLMo, both embedders; 36 tests, Holm over 36 as `revision_direction_tangent.py`):
   alignment r0 → r2, weighted contributions along p and s, change along s; plus the NEUTRAL rows.
2. DISC-4o N5 and N6 (Holm over the 14 of `neutral_revision.md`, recomputed with the other 12 also after removal);
   DISC-OLMo NEUTRAL contrasts as `strengthen_olmo_neutral.md`.
3. REP-4o: R1, R2, E1-E6 with the registered rule (Holm over 8, 95 % t-interval vs ±0.015, `2·p_E` clarification).
4. OLMO-C: the three primary contrasts (Holm over 3, ±0.015, registered bootstrap and Wilcoxon) and the secondary
   BGE-M3 cell.
5. MOD (secondary): N5, N6.
6. Positive controls after removal: follow-ups' LOO alignment; T1's G.

**Reading rule (fixed).** "Survives" is defined per item before running:

- Table 3: alignment rises with Holm p < 0.05 in at least 10 of the 12 cells (now 11 of 12); the share carried by
  the move along p stays above 50 % in at least 10 of 12; each of the two Holm-significant along-s changes keeps its
  sign and its Holm significance, and no non-significant one becomes significant with the opposite sign.
- Contrasts (DISC-4o N5/N6, REP-4o E1-E6, OLMO-C primary three): the reading category (equivalent / detected /
  insufficient, or Holm-significant / not) is unchanged.
- **Counts against the current claim**: alignment loses Holm significance in 3 or more of the 12 Table 3 cells, or
  the follow-ups' LOO alignment falls by more than half (the metric no longer registers a shared orientation), or a
  REP-4o/OLMO-C category changes from "detected" to "insufficient/equivalent" or from "equivalent" to "detected".
- Primary g decides; secondary g's are reported. If the primary survives but a secondary g flips a category, the paper
  says the reading depends on how genre is defined, naming which.
- Step gap: if removing g shrinks the DISC-4o B1 step gap by more than half, the paper says the short step is at least
  half genre in the embedding and points to Part B for the content-controlled test; if by less than a fifth, one
  sentence.

**Paper text by outcome.** Survives: one paragraph in 5.3 ("removing the mean difference between generated proposals
and real abstracts changes no reading; Table A.x") and the same statement for the replication and OLMo tests.
Partial: the table is shown in the main text with both versions and the affected sentences are rewritten from the
genre-removed version. Against: Section 5.3's reading that critique "turns the ideas towards the follow-ups' direction"
is withdrawn as a direction result and restated as a step/genre by-product, pending Part B.

---

## 6. T4 — Prompt echo: n-gram overlap with the context abstract (question 9)

**Question.** How often do ideas reuse phrases of the context abstract compared with real follow-ups and
continuations, and how much of the step gap does that overlap account for?

**Computation.**

- Text of an item = title + " " + abstract (ideas and follow-ups in full). Context text = what the generator saw:
  title + " " + abstract[:1200] (`continue_from_context.paper_block`). Normalisation: Unicode NFKC, lowercase,
  tokens = regex `[a-z0-9]+`; no stop-word removal, no stemming.
- Primary measure: 3-gram containment C3(x) = |3grams(x) ∩ 3grams(ctx)| / |3grams(x)| (set-based). Secondary: the same
  for n = 1, 2, 4; longest common token subsequence / |x| (ROUGE-L recall against the context).
- Items: DISC-4o one-shot CTX ideas (5 per context); DISC-4o round-0 and round-2 critique ideas per arm; DISC-OLMo CTX
  ideas (`ideas_llama.jsonl`); REP-4o round-0 ideas; OLMO-C round-0 ideas; MOD CTX ideas (secondary). Real items:
  all follow-ups; consensus continuations (DISC, 47 contexts; OLMO-C, 150); background follow-ups (DISC).
- Per context: median C3 over ideas minus median over follow-ups (and over continuations). Mean over contexts with
  bootstrap CI and two-sided Wilcoxon. Holm over the primary family {DISC-4o vs follow-ups, DISC-4o vs continuations,
  OLMO-C vs continuations}.
- Echo-adjusted step gap: item-level OLS with context fixed effects over ideas ∪ real items of a context,
  step = u_i + γ·[idea] + β·C3 (+ β₂·C1 in a secondary model); γ is the step gap at equal overlap; γ₀ the same model
  without C3. Echo share = 1 − γ/γ₀ with cluster-bootstrap CI. Run against all follow-ups and against continuations.
- Critique arms: mean change in C3 from round 0 to round 2 per arm and its Spearman correlation with Δstep (descriptive).

**Reading rule (fixed).**

- Answer to question 9: the overlaps are reported as numbers in Section 5.1.
- Echo share ≥ 0.5 (point, primary cell DISC-4o Qwen3, against continuations or against all follow-ups) → **counts
  against the current claim**: the paper says at least half of the short step is accounted for by the ideas reusing
  the context's wording, and treats the step gap as unresolved until Part B. Echo share in [0.2, 0.5) → the paper
  states the share and keeps the claim with that qualification. < 0.2 → one sentence.
- Caveat stated: lexical overlap is one channel of echo; genre (proposal style) without shared wording is not captured,
  which is why Part B exists.

---

## 7. T5 — Follow-ups shared between contexts (minor 11)

**Computation.** Per study (DISC, REP, OLMO-C, MOD): number of distinct follow-up arXiv IDs; number appearing under ≥ 2
contexts; number of contexts that share at least one follow-up; connected components of the context graph (edge = a
shared follow-up), their count and size distribution. Also shared IDs across studies (DISC ∩ REP, DISC ∩ OLMO-C,
REP ∩ OLMO-C; REP and OLMO-C come from the same frame but different context sets).

**Sensitivity (run whatever the counts).** Cluster bootstrap with components as resampling units (10,000, seed as
above) for: DISC-4o B1 step gap; Table 3 change along s per arm (DISC-4o); N6 (DISC-4o); REP-4o E1-E6 bootstrap
intervals; OLMO-C primary three contrasts. Secondary: drop every shared follow-up from f (and from the step gap) and
recompute the same quantities.

**Reading rule.** If no reading category or Holm decision changes, the paper adds one sentence with the counts. If one
changes, the paper reports the component-bootstrap result as the primary interval for that quantity and says so.

---

## 8. T6 — Citation counts of sampled contexts vs the frame (minor 10)

**Computation.**

- DISC frame: `D/runs/followups/followups.jsonl` (1,000 seeded 2021-2022 arXiv cs papers; fields `n_citing_arxiv` =
  uncapped number of arXiv citing papers dated ≥ 2024-01-01 per Semantic Scholar, `n_followups`). Compare the 150
  sampled contexts with (a) all 1,000 and (b) the rows before the 150th eligible in file order (the frame actually
  traversed). Quantiles 10/25/50/75/90, share with ≥ 5.
- REP and OLMO-C frame: `D/runs/olmo-direction/followups.jsonl` (5,381 frame positions; `n_citing_all`,
  `n_citing_arxiv_all`, `n_followups_uncapped`): same quantiles for the frame, the 150 REP contexts and the 150 OLMO-C
  contexts. `n_citing_all` gives the total (all-date, all-venue) citation count, which DISC lacks.
- Within-sample gradient (descriptive): Spearman ρ, across the 150 contexts, between log(1 + n_citing) and (a) the
  per-context step gap (DISC-4o B1; REP-4o round 0), (b) GEO's Δs and (c) the GEO − NEUTRAL along-s contrast; bootstrap
  CIs. Also the step gap in the bottom vs top tercile of citing counts.

**Reading rule.** Descriptive. The paper adds the sentence the review asks for (median and IQR of citing counts in the
frame vs the sample; the share of the frame that meets the ≥ 5 rule). If |ρ| for the step gap has a CI excluding 0,
the paper says the step gap varies with how cited the context is (with direction) and that the estimates describe
well-cited papers; otherwise it says no gradient was detected within the sample, which does not show the absence of
one below the threshold.

**Infeasible as asked, and the closest version.** Measuring the step gap or the turn for typical (less cited) papers
would need new generation on contexts with fewer than five follow-ups, where f and s are poorly defined (s from one or
two citing papers). Not proposed. The within-sample gradient and the frame-vs-sample distribution are the feasible
substitutes. DISC's frame has no total citation count; fetching it (Semantic Scholar batch API, free, no key,
~1,000 IDs) is optional and would be committed as data before use; not needed for the sentence.

---

## 9. Multiplicity summary (Part A)

Families are per cell × embedder and per analysis, as stated in each section; the original families are reused
unchanged for recomputed tests (Table 3: 36; DISC N-family: 14; REP: 8; OLMO-C: 3). T2(i): 12 per cell × embedder.
T2(ii): reuses the cell's contrast family. T4 primary: 3. Everything else is descriptive with 95 % intervals and no
test. Following the review's minor 3, the paper will state the number of families and that no correction is applied
across them.

## 10. Compute (Part A)

No paid call, no GPU, no encoder loaded (all vectors exist). Single process, `OMP_NUM_THREADS=4`.
Per cell × embedder × genre variant, one pass over 150 contexts with a pool ranking (≈ 60k × 1024 matvec) takes about
15 s; ≈ 10 cell-embedder combinations × 6 variants (original + 5 genre) ≈ 15 min. Bootstraps (10,000 resamples,
vectorised; cluster bootstrap of the fixed-effects regressions ≈ 60 models × ~10 s) ≈ 10-15 min. n-gram counts on
≈ 20k texts: < 1 min. **Total ≈ 30-45 min; peak memory ≈ 2.5 GB** (one corpus embedding as float32 ≈ 460 MB plus
temporaries, loaded one embedder at a time; corpus parquet read with the needed columns only). Fits the 16 GB Mac.

Script `scripts/review_tangent_confounds.py` with subcommands `gate`, `t1` … `t6`, `all`; output
`docs/prereg/review_tangent_confounds.json` (one top-level key per analysis, plus `gate`, `seed`, `n_boot`, and the
counts of dropped contexts per test).

---

## 11. Part B — Rewrite and paraphrase nulls (M4, M3(b)) — paid sub-study

Separate registration `docs/prereg/review_rewrite_null.md`, committed before the smoke calls. Generation is paid
spend on Azure through the gateway; it needs Javier's go-ahead with the cost below.

### 11.1 Questions

- **P1 (M4).** When the same generator rewrites each real follow-up (and each consensus continuation) as a proposal in
  the idea schema, with the context shown, how far from the context do the rewrites sit? The step gap between the
  ideas and the *rewritten* follow-ups/continuations is the quantity the short-step claim needs.
- **P2 (M3(b)).** What tangent alignment with f and s, and what change along s, do texts with no follow-up information
  have at matched step: a paraphrase of the context, and a rewrite of a random nearby later paper?

### 11.2 Items and prompts (fixed)

Generator per study: gpt-4o-2024-08-06 (`gpt-4o-0806`) for DISC and REP; OLMo-7B-0424-Instruct on Kaggle for OLMO-C.
System prompt = the ideas' system prompt (`continue_from_context.SYSTEM`); temperature 1.0 and max_tokens 600 as the
ideas; one sample per item; JSON parsed with the ideas' parser; one retry on parse failure, then dropped (counted).

- **Prompt R (rewrite as proposal, context shown):**
  "Here is a paper:\n\n{context block}\n\nBelow is a later paper. Rewrite it as the proposal for the paper that should
  come next, building on the paper above: give its title and an arXiv-style abstract of 120-180 words. Keep exactly
  its content: the same subject, method, claims and details. Add nothing, remove nothing, invent nothing; only the
  wording and the framing as a proposal change.\n\n{later block}" (blocks via `paper_block`, abstracts ≤ 1,200 chars).
- **Prompt PA (paraphrase of the context):** the `style_probe.PROMPT` wording ("Rewrite this paper in your own words as
  the JSON object. Keep exactly the same content … only the wording changes.") applied to the context block.

Items:

| Set | DISC (gpt-4o) | REP (gpt-4o) | OLMO-C (OLMo, Kaggle) |
|---|---|---|---|
| RW-F: every saved follow-up, prompt R | 2,883 | 2,419 | consensus continuations 485 + a seeded sample (seed 20261114) of up to 5 non-continuation follow-ups per context (≤ 750) |
| NA: context paraphrases, prompt PA | 3 per context = 450 | 450 | 450 |
| NB: nearby later papers, prompt R | 3 per context = 450 | 450 | 450 |

NB selection (fixed, computed and written to `runs/review-rewrite/{study}/nb_ids.json` before any call): from the
cell's future-split pool with all citing papers and the context removed (REP/OLMO-C: `citing_all.jsonl`; DISC: the
complete citing sets in `runs/followups/complete_citations/`), ranked by Qwen3 cosine to the context, exclude the
first k (those that build p) and draw 3 uniformly from ranks k+1 … k+200, seed 20261114 + context index.
DISC continuations are a subset of RW-F (no extra calls).

Embedding: each rewrite with the same encoders and settings as the study's ideas: DISC on Kaggle with the
`kaggle/embed-continuation` settings (sentence-transformers, both encoders); REP locally (`LocalQwen` ONNX and BGE-M3
CPU, `revision_neutral.py embed-bge` settings); OLMO-C inside the Kaggle kernel as `olmo-direction`.
Raw data under `D/runs/review-rewrite/{disc,rep}/` and `D/kaggle-output/review-rewrite-olmo/`.

### 11.3 Estimands and tests

Content check (all sets): cos(rewrite, original) in both embedders; distribution reported; RW-F rewrites below 0.80
in Qwen3 are excluded in a sensitivity analysis (never in the primary).

**P1 (primary: DISC-4o, Qwen3).** Per context, gap_RW = median step of the one-shot CTX ideas − median step of the
rewritten follow-ups; and gap_RW,cont against the rewritten consensus continuations (47 contexts). Also the effect of
rewriting on a real paper: Δ_RW = median step(rewrites) − median step(originals). Means over contexts, bootstrap CIs
(10,000, seed 20261114), one-sided Wilcoxon (H1: gap_RW > 0). Holm over the two primary tests. Secondary: BGE-M3;
REP-4o (round-0 ideas vs rewritten follow-ups); OLMO-C (round-0 ideas vs rewritten continuations, and the original
gap against unrewritten continuations for comparison); the T4 echo-adjusted model refitted with rewrites.

**P2.** With f, p, s from the original follow-ups (Table 3 construction; for rewritten follow-ups, s₋ⱼ LOO):
(a) projections on f, p and s of NA, NB and RW-F items, round-0 and round-2 ideas; (b) matched-step comparison,
item-level OLS with context fixed effects on round-2 critique ideas ∪ NB: T·s = u_i + β·(1 − step) + δ·[idea]; δ =
the ideas' s-projection beyond a non-follow-up paper rewritten as a proposal at the same step (cluster-bootstrap CI);
same for alignment and along p; (c) the null dose-response β₀, fitted on NA ∪ NB only, and the turn it predicts for
each arm, β₀·mean Δstep_A, compared with the observed Δs_A and Δalign_A (ratio with bootstrap CI). Holm over the
three δ tests (s, p, align) per cell × embedder.

### 11.4 Reading rule (fixed)

P1:
- gap_RW (follow-ups) with CI lower bound > 0 and point ≥ 0.5 × the original gap (0.149 in DISC-4o Qwen3) → the short
  step survives a genre-and-echo control; the paper says so.
- CI lower bound > 0 but point < 0.5 × original → "about X % of the short step is genre and echo; the rest remains".
- **Counts against the current claim**: CI includes 0, or point < 0.03 → the claim "ideas stay closer to the context
  than the citing papers" is withdrawn as stated and replaced by "not distinguishable from genre and prompt echo".
- Continuations: if gap_RW,cont's CI includes 0, the sentence "a third as large against the papers that continue the
  work" is replaced by "against continuations rewritten as proposals, no gap is detected"; if it stays positive with
  the CI excluding 0, the claim stands with the rewritten number.

P2:
- **Counts against the current claim**: the null predicts at least the observed turn (β₀·mean Δstep_A ≥ observed
  Δalign_A, point, for a majority of critique arms in the primary cells) and δ for s has a CI including 0 → the paper
  states that the rise in alignment is what leaving the paraphrase regime produces in texts with no follow-up
  information, and withdraws the reading of Table 3 as a turn towards the follow-ups.
- δ > 0 with a CI excluding 0 and the predicted turn below half the observed → the turn exceeds the null; the claim
  stands and the null is reported next to Table 3.
- Otherwise: the paper reports both numbers and describes the turn as "not separable from a step effect with these
  controls".

### 11.5 Cost (by analogy to measured runs)

gpt-4o measured cost: NEUTRAL arms, 900 calls each, `iter-neutral` 493k in / 204k out tokens and
`critique-rep-neutral` 491k / 204k, $3.27 per 900 calls (`strengthen_critique_replication.md`, "Cost"; ledger
`D/ledger.jsonl`), i.e. ≈ 546 in / 226 out tokens and $0.0036 per call, consistent with $2.50 / $10 per M tokens.
Prompt R carries two paper blocks, so I assume ≈ 720 input and ≈ 230 output tokens per call (≈ $0.0041).

| Part | Calls | Tokens (in / out) | Cost USD | Cost AUD (×1.52, as the earlier "$21 ≈ 32 AUD") |
|---|---|---|---|---|
| DISC (RW-F 2,883 + NA 450 + NB 450) | 3,783 | 2.7M / 0.87M | ≈ 15.5 | ≈ 24 |
| REP (2,419 + 450 + 450) | 3,319 | 2.4M / 0.76M | ≈ 13.6 | ≈ 21 |
| Smoke (prompt and parser check) | 20 | — | < 0.1 | < 0.2 |
| **Total gpt-4o** | **≈ 7,120** | **≈ 5.1M / 1.6M** | **≈ 29-31** | **≈ 45-47** |

Run in one day: ≈ $30 USD (≈ 46 AUD) that day, ledger tags `review-rewrite-disc`, `review-rewrite-rep`; wall time
≈ 2-3 h at the gateway concurrency used for the replication. A reduced version (DISC only, which carries the primary
tests) costs ≈ $15.5 USD (≈ 24 AUD).

OLMo (OLMO-C): ≤ 2,135 generations on Kaggle T4 × 2 (free quota, no paid spend). Measured rate: wave 1 of
`olmo-direction` produced 3,906 generations plus embeddings in 22,330 s (≈ 5.7 s per generation), so ≈ 3.4 h plus
≈ 20 min embedding, inside one 12-hour session. DISC embeddings on Kaggle: ≈ 3,800 texts, < 1 GPU hour.
Local work: REP embeddings (≈ 3,300 texts, Qwen3 ONNX + BGE-M3 on CPU) ≈ 30-45 min, peak ≈ 4 GB; analysis
≈ 10 min, peak ≈ 2.5 GB.

Not included (would extend the cost): rewrites for DISC-OLMo (≈ 3,800 OLMo generations, ≈ 6 h Kaggle) and for
gpt-5.4-mini (≈ 3,250 calls, ≈ $1.5 at the rates of `strengthen_modern_neutral.md`). Either can be added by amendment
before the generation starts.

### 11.6 Order of work (fixed)

1. Commit Part A and Part B registrations. 2. Run Part A (gate first). 3. Write `nb_ids.json` and the OLMo sample of
follow-ups, commit them. 4. Smoke (20 calls), check parse rate ≥ 95 %; if lower, fix the parser only (prompt
unchanged) and disclose. 5. Generate, embed, analyse once.

---

## 12. Feasibility notes and honest limits

- All Part A inputs exist with the expected shapes (checked: DISC Qwen3/BGE vectors 10,083 rows incl. contexts,
  follow-ups, Part B and critique ideas; DISC NEUTRAL 1,350; REP C 150, F 2,419, ideas 5,400 in both embedders; OLMO-C
  two Kaggle parts with ctx/fup/iter vectors; MOD emb.npz 150/2,331).
- DISC NEUTRAL's Qwen3 vectors come from the local ONNX build, the other DISC arms from the Kaggle sentence-transformers
  build (an existing, disclosed difference). T3's genre direction excludes NEUTRAL for that reason; T2(ii) mixes them
  as the published N-tests already do.
- REP has no continuation labels: continuation-specific parts of T1, T4 and P1 run on DISC and OLMO-C only.
- DISC-OLMo has one idea per context and arm; its T2 and T3 cells are secondary and low-powered.
- T2 cannot distinguish "turning because the step is longer" from "a longer step that also turns"; only Part B's
  null (P2) gives an external dose-response. Without Part B, T2 and T3 can limit the claim but not refute a turn.
- M3(b)'s "paraphrase of a random nearby later paper at matched step": exact step matching per item is not
  controllable in generation; it is replaced by regression adjustment on step (P2(b)) over NB items drawn from a
  fixed rank band.
- Minor 10's "typical papers" cannot be measured without new generation on poorly-defined contexts (Section 8).
