# Pre-registration (draft): a positive control for steering, and a fixed-target oracle

Status: DRAFT design, written 2026-11-14 for review. Nothing below has been run, generated or committed. To become
binding it must be copied to `docs/prereg/review_posctrl.md` and committed **before** the hold-out file is written,
before any generation, and before any rescoring of existing ideas on the new directions. Proposed script:
`scripts/review_posctrl.py`; proposed output: `docs/prereg/review_posctrl.json`.

## Review concerns addressed

- **M1(a)** (`fable_peer_review.md`): "distance feedback barely steers" has no positive control. Nothing shows that
  *any* intervention moves the ideas along the follow-up-specific direction s by more than ~0.02 in two rounds, so
  "barely" has no denominator. The review asks for an arm that reveals the title or a one-line topic hint of one
  real follow-up, held out from the construction of s, with the change along s reported.
- **M1(b)** (secondary here): report turns as a fraction of the idea-to-follow-up gap along s (also question 4: the
  follow-ups' own leave-one-out projection on s).
- **Question 7**: why the ORACLE score is the maximum cosine to any follow-up, not the cosine to their centroid or to
  the nearest continuation, given that a maximum can change its target between rounds.
- M1(c) and M1(d) (scope the claim to "two rounds of scalar feedback"; stop calling ±0.015 a "bound" without a
  justification) are wording changes; the outcomes below say what the text becomes under each result.

## What is known before this registration

All results of the discovery sample (Tables 2, 2b, 3, 4), of the gpt-4o replication
(`strengthen_critique_replication.json`), of the OLMo direction study (`strengthen_olmo_direction.json`), and the
peer review. In particular, on the replication contexts (Qwen3, original s): round-0 projection on s ≈ 0.118; Δs
GEO +0.019, Reviewer +0.012, ORACLE +0.012, NEUTRAL +0.0085; follow-ups' leave-one-out alignment 0.467. No hint
text, hold-out choice, or projection on any held-out direction has been computed.

Found while checking feasibility (read-only, reported so the paper can be corrected independently of this study):
`strengthen_critique_replication.json` records `overlap_with_olmo_direction_eligible: 0`, but against the current
`runs/olmo-direction/eligible.json` 17 of the 150 replication contexts are among the 150 OLMo-eligible (generated)
contexts (the analysis code does `set(json.loads(...))` on what is now a dict, which yields its keys). The paper
(draft.md line 437) says the OLMo contexts are "disjoint from the original 150 and from the gpt-4o replication". That
sentence is wrong for 17 contexts and needs correcting whatever this study finds.

## Question

1. **Positive control (primary).** If gpt-4o is told the title of one real follow-up of the source paper, does its
   proposal move along the follow-up-specific direction built from the *other* follow-ups, beyond what content-free
   revision does? And how large is that move compared with the turns that distance (GEO), reviewer and single-score
   oracle feedback produce on the same contexts? This gives "barely" a denominator, or shows that the measure cannot
   register even informed steering within two revisions.
2. **Fixed-target oracle (secondary, question 7).** Does a scalar oracle with a fixed target (the cosine to the
   follow-ups' centroid) raise its own score and turn the ideas along s, where the max-cosine oracle did neither?

## Choice of sample

| | gpt-4o replication contexts | OLMo direction contexts (Kaggle) |
|---|---|---|
| s after holding out one follow-up | from ≥ 4 saved follow-ups (all citing papers; ≥ 5 per context, mean 16.1) | continuation s: median k = 2 consensus continuations, so holding one out leaves **one** for 83 of 150 contexts; only 67 contexts have k ≥ 3 |
| pairing | GEO round 0 and NEUTRAL already exist (450 ideas); a new arm can start from GEO's round 0, so HINT, GEO, NEUTRAL share round 0 | the same is possible (saved GEO round 0 in `iter_olmo.jsonl`), but the kernel must be changed to load round 0 instead of generating it |
| role in the paper | the confirmatory basis of every current gpt-4o statement in Section 5.3 | the confirmatory basis of the OLMo "no turn of 0.015 or more" statement |
| money | paid (Azure, paid spend): about USD 3.3 per arm | free (Kaggle T4 x2), about 1.5 h GPU per arm |
| pipeline change | one new arm in `iterative_critique.py` (round-0 reuse already exists, `LG_ITER_R0_FROM`) | new kernel input (round 0 from file) and arm |

**Decision.** Primary: **gpt-4o on the 150 replication contexts** (Block A). It is the cleanest comparison: the held-out
follow-up is removed from s and s is still estimated from at least four follow-ups; the new arm is paired with the
existing GEO and NEUTRAL ideas from the same round-0 proposals; and it is the sample on which the paper's gpt-4o
claims now rest. It is also cheap in absolute terms (about USD 3.3 for the primary arm). The OLMo sample is free but
its continuation direction cannot lose a continuation and stay estimable, so a held-out *continuation* as primary is
infeasible there; Block C below is an optional free secondary on OLMo using the all-follow-up direction.

## Data (all confirmed to exist, 2026-11-14)

Root `D = ~/Documents/experiment-data/latent-gap-ideas`.

- Contexts and follow-ups: `D/runs/critique-replication/followups.jsonl` (150 rows; each follow-up has `arxiv_id`,
  `title`, `abstract`, `v1`; 2,419 follow-ups).
- Qwen3 vectors: `D/runs/critique-replication/emb.npz` (`C` 150×1024, `F` 2419×1024, `owner`), ideas
  `D/runs/critique-replication/ideas_emb_qwen3e06.npy` (5400×1024 = 4,050 critique rows of `iter_ideas.jsonl` +
  1,350 NEUTRAL rows of `iter_ideas_neutral.jsonl`, in the order of `strengthen_critique_replication.idea_rows`).
- BGE-M3 vectors: `D/runs/critique-replication/emb_bgem3.npz` (`ctx`, `fup`, `ideas` 5400×1024).
- Existing ideas: `iter_ideas.jsonl` (GEO, LLM, ORACLE × 150 × 3 × rounds 0-2, no empty abstract),
  `iter_ideas_neutral.jsonl` (NEUTRAL, round 0 copied from GEO).
- Direction pool: `D/prepared/arxiv/corpus.parquet` and `D/prepared/arxiv/emb_{qwen3e06,bgem3}.npy` (113,000×1024
  float16); citing lists `D/runs/olmo-direction/citing_all.jsonl`.
- Discovery sample (Block B0 only): `D/runs/continue-from-context/iter_ideas.jsonl`, `iter_ideas_neutral.jsonl`,
  `neutral_emb_{space}.npy`, and the Kaggle vectors `D/kaggle-output/embed-continuation-v1/emb_{space}.npy` with
  `D/kaggle-upload/continuation-texts/texts.parquet`, loaded exactly as `revision_neutral.load_gpt4o`.
- OLMo (Block C only): `D/kaggle-output/olmo-direction-w1p1/` and `-w2p1/` (`iter_olmo.jsonl` with GEO round 0 for
  all 150 eligible contexts, `ctx_*`, `fup_*`, `fup_owner.npy`, `iter_*`), `D/runs/olmo-direction/eligible.json`,
  `followups.jsonl`, labels under `labels/`.

## Block A: positive control with gpt-4o (primary; paid, about USD 3.3)

### Hold-out (written once, before generation)

`review_posctrl.py holdout`: for context i = 0..149 in file order, with `rng = numpy.random.default_rng(20261114)`,
h_i = saved follow-up number `rng.integers(n_i)` (one draw per context, in order), where n_i is the number of saved
follow-ups. Written to `D/runs/critique-replication/posctrl_holdout.json` (`ctx`, `h_index`, `arxiv_id`, `title`).
A random follow-up, not a selected one: choosing the follow-up most aligned with the others would inflate the
control. The same h_i serves all ideas and arms of context i.

### Generation (new arm HINT; `iterative_critique.py` gains it, nothing else changes)

- gpt-4o-2024-08-06 through `gpt-4o-0806`, temperature 1, 3 ideas per context, rounds 0-2, CTX and revise
  prompts as Section 5.2. **Round 0 copied from GEO** (`LG_ITER_R0_FROM=GEO`), as NEUTRAL.
- Feedback in both revisions, built to mirror the ORACLE message with the scalar replaced by content:
  `The title of a paper actually published as a follow-up is "{title of h_i}". Try to get closer.`
  (ORACLE's is `Your proposal's similarity to the closest paper actually published as a follow-up is 0.67 (embedding
  cosine). Try to get closer.`). Only the title, never the abstract.
- Command: `LG_RUN=critique-replication LG_FUPS=critique-replication LG_ITER_ARMS=HINT LG_ITER_R0_FROM=GEO
  LG_ITER_OUT=iter_ideas_posctrl.jsonl LG_POSCTRL_HOLDOUT=posctrl_holdout.json LG_TAG=review-posctrl-hint
  ALIAS=gpt-4o-0806 uv run python scripts/iterative_critique.py generate`.
- Why a title and not a one-line topic hint: a topic hint would need another model to write it from h's abstract
  (extra paid calls and a free choice of how much to leak). The title is fixed text written by the follow-up's
  authors. The topic-hint version is not run.
- Embedding: Qwen3-Embedding-0.6B ONNX (`review_posctrl.py embed-qwen`, settings of
  `strengthen_critique_replication.embed_qwen`) and BGE-M3 on CPU (`embed-bge`, settings of `embed_bge`), of the new
  rows only, to `D/runs/critique-replication/posctrl_emb_{qwen3e06,bgem3}.npy`.

### Construction on the held-out direction

For context i (unit vectors in the embedder), F_{-h} = saved follow-ups without h_i, n' = n_i − 1 ≥ 4.
Tangent of x: unit(x − (x·c)c). f_{-h} = unit mean tangent of F_{-h}. p_{-h} = unit mean tangent of the n' nearest (cosine
to c) future-split corpus papers not among the saved follow-ups (h_i included in that exclusion), as
`neutral_contrasts.per_context`. s_{-h} = unit(f_{-h} − (f_{-h}·p_{-h}) p_{-h}). t_h = tangent of h_i.

For each arm A ∈ {GEO, LLM (Reviewer), ORACLE, NEUTRAL, HINT} and round r, per context, mean over its ideas with an
abstract and a finite tangent: step = x·c; s = tangent·s_{-h}; p = tangent·p_{-h}; align = tangent·f_{-h};
fit_{-h} = max cosine to F_{-h}; hinted = tangent·t_h; cos_h = x·h_i. Changes round 0 → 2 (step: r0 − r2, so positive
= longer step). Contrast with B: d_A,B(m) = Δm_A − Δm_B, per context; contexts lacking a valid round 0 and round 2 in
either arm are dropped from that contrast (counted). This is `neutral_contrasts.per_context` with F replaced by
F_{-h} and two extra measures; the existing arms are rescored, not regenerated, so their changes on s_{-h} will differ
slightly from the published ones on s (both reported).

### Estimators and tests

- Per contrast: mean over contexts; 95 % paired t interval; 95 % percentile bootstrap interval, **20,000 resamples
  over contexts, `default_rng(20261114)`**, one generator consumed in the fixed order in which the contrasts are listed
  below; two-sided Wilcoxon signed-rank p.
- **Manipulation check MC (not in the Holm family, read first):** d_HINT,NEUTRAL(hinted) > 0 by its t interval.
- **Confirmatory family PC, Qwen3, Holm over 3, alpha 0.05 (Wilcoxon p):**
  - PC1: d_HINT,NEUTRAL(s). Same round-0 ideas.
  - PC2: d_HINT,GEO(s). Same round-0 ideas.
  - PC3: d_HINT,ORACLE(s). Independent round 0, paired by context.
- **Yardstick ratios (estimated only if PC1 is Holm-significant and positive):** for A ∈ {GEO, LLM, ORACLE},
  R_A = mean d_A,NEUTRAL(s) / mean d_HINT,NEUTRAL(s), and Q_A = mean Δs_A / mean Δs_HINT; 95 % percentile bootstrap
  intervals of the ratio of means, resampling contexts jointly (same 20,000 index draws for numerator and
  denominator).
- BGE-M3: the same MC, PC1-PC3 (own Holm over 3) and ratios, as replication.
- **Secondary, descriptive:** each arm's Δs, Δp, Δalign, Δstep, Δfit_{-h} on the held-out construction; the HINT
  contribution split of Δalign into along p and along s (Table 3 weights); d_HINT,NEUTRAL for p and step; the
  hinted paper's own projection t_h·s_{-h} and the share of its gap closed, mean Δs_HINT / mean (t_h·s_{-h} − s at
  round 0); the variant with the pool excluding all citing papers (`citing_all.jsonl`); HINT along the original s
  (which contains h, labelled as leaky); title echo, the share of round-2 HINT titles whose lower-cased word set has
  Jaccard ≥ 0.5 with h's title; the 17 contexts that are also OLMo-eligible, reported apart (descriptive only).

### Reading rule (fixed)

Read in order. "Detected" = Holm-adjusted p < 0.05 with the t interval excluding 0 on the same side.

1. **MC fails** (t interval of d_HINT,NEUTRAL(hinted) includes 0 or is below it): gpt-4o did not move towards the
   hinted paper itself. The positive control is **uninformative** (a failed manipulation, not a test of the
   measure). Paper: says so in one sentence in 5.3 and makes the M1(c)/(d) wording changes only.
2. **MC passes, PC1 not detected, t interval of PC1 inside (−0.015, +0.015):** the ideas move towards the hinted
   follow-up but not along the direction the other follow-ups share. **This counts against the paper's current
   claim**: the follow-up-specific direction does not register a move that carries real follow-up content, so its
   nulls (Sections 5.3 and 5.4, the ±0.015 statements, the OLMo bound by extension) say little about steering.
   Paper: "barely steer" is withdrawn from the Section 5 title, abstract and conclusion; the direction results are
   reported as "not distinguishable from revision on a measure that failed its positive control"; the step results
   stand.
3. **MC passes, PC1 not detected, interval crossing ±0.015:** positive control **insufficient**. Paper: reports it,
   does not use it as a yardstick, and makes the M1(c)/(d) wording changes ("two rounds of scalar or verbal
   feedback"; "smaller than X of the gap" in place of "bounded").
4. **PC1 detected and positive** (the measure registers informed steering). Then, per critique arm A, read R_A:
   - upper 95 % bound of R_A < 0.5 → **"barely" supported for A, with a denominator**: "beyond revision, A turns the
     ideas towards the citing papers by at most [upper R_A] of what revealing one real follow-up's title does";
   - lower bound ≥ 0.5 → **counts against the claim for A**: that feedback adds at least half the turn that a real
     follow-up's title adds; "barely" is withdrawn for A and Section 5's headline is rewritten around it;
   - otherwise (interval straddles 0.5) → "A adds [R_A] of the hint's turn (CI …)"; "barely" is replaced by that
     number for A.
   PC2 and PC3 are reported alongside: if HINT does not turn more than GEO (PC2 not detected), the text may not
   claim that content steers more than distance feedback on these contexts.
5. If PC1 is detected **negative** (the hint turns the ideas away from the others' direction): reported as such;
   the paper says the follow-up-specific direction is not a measure of "towards the follow-ups" for informed
   moves, with the same consequences as outcome 2.

The BGE-M3 cell replicates or not; where Qwen3 and BGE-M3 readings differ the paper states both and does not pick one.

### Power

Planning SD of a contrast along s: 0.05-0.07 (gpt-4o, `revision_neutral.json`, replication); a hint contrast may be
more variable, so 0.06-0.08. At n = 150 the 95 % half-width is 0.010-0.013; PC1 detects a true d of about 0.016-0.021
with 80 % power at alpha 0.05/3. The plausible size under a working measure is larger: if the ideas moved halfway
from their round-0 projection (≈ 0.118) to a follow-up's typical projection on s, Δs would be several hundredths.
R_A intervals are wide when PC1 is small; a small but detected PC1 leaves most R_A in the "straddles" case, by design.

## Block B: fixed-target oracle (secondary, question 7; paid, about USD 3.3; separable from Block A)

### Why the published ORACLE uses the maximum

ORACLE's score is the same quantity as the pre-registered fit measure (maximum cosine to any saved follow-up,
`iterative_critique.md`, C5), so that its feedback names the number C5 predicted it would raise, as score feedback
did in the earlier grid. The review is right that a maximum can change its reference paper between rounds, so the
signal is not a fixed target.

### B0 (free re-analysis of existing ideas; no generation)

On both gpt-4o samples (discovery and replication), Qwen3 (the space the feedback used) and BGE-M3: for every idea
chain, the index of the argmax follow-up at rounds 0, 1, 2; the **switch rate** (share of revisions 0→1 and 1→2 in
which the argmax changes) for ORACLE, GEO and NEUTRAL (the latter two as the rate under revision without that
signal); and the change round 0 → 2 in (a) cosine to the unit centroid of the saved follow-ups and (b) cosine to the
round-0 argmax follow-up (a fixed target), ORACLE − NEUTRAL paired by context, with 20,000-resample bootstrap
intervals (`default_rng(20261114)`) and Wilcoxon p. Descriptive, no correction. Paper: answers question 7 with the
switch rate and states whether ORACLE ideas moved towards the centroid or towards their own first target.

### B1 (new arm ORACLE-C; run only if approved together with Block A or after it)

- As HINT (gpt-4o, 3 ideas, round 0 from GEO, same contexts) with feedback
  `Your proposal's similarity to the average of the papers actually published as follow-ups is {x:.2f} (embedding
  cosine). Try to get closer.`, x = cos(e, unit(mean of F_{-h} in Qwen3)), computed locally with the same Qwen3
  embedder that scores ORACLE. The centroid excludes h_i so that ORACLE-C can be scored on s_{-h} with HINT.
- `LG_ITER_ARMS=ORACLE-C ... LG_TAG=review-posctrl-oraclec`, appended to `iter_ideas_posctrl.jsonl`.
- **Family OC, Qwen3, Holm over 2:** OC1 d_ORACLE-C,NEUTRAL(cos to centroid of F_{-h}) (does it raise its own score
  beyond revision?); OC2 d_ORACLE-C,NEUTRAL(s on s_{-h}). Same estimators; BGE-M3 as replication.
- Reading: OC1 detected positive and OC2 detected positive → the fixed target steers where the max did not; the
  paper says ORACLE's failure was partly the moving target, and ORACLE-C's R (as R_A in Block A, if PC1 holds) is
  reported. OC1 detected, OC2 not → the score is optimisable but the move is not follow-up-specific (it goes towards
  nearby later work, read from Δp). OC1 not detected → a fixed-target scalar fails like the max; the paper says the
  failure is of two-round scalar feedback, not of the maximum. ORACLE-C against HINT (d_HINT,ORACLE-C(s)) is
  reported descriptively: it is the content-versus-scalar contrast with the target held fixed.

The nearest-continuation variant is **infeasible on these contexts**: only 24 of the 150 replication contexts have a
consensus continuation (17 have ≥ 2). It is feasible only on the OLMo contexts (Block C).

## Block C: OLMo on Kaggle (optional secondary; free)

Same HINT arm with OLMo-7B-0424-Instruct on the 150 OLMo-direction eligible contexts, round 0 copied from the saved
GEO round 0 of `olmo-direction-w1p1`/`-w2p1` `iter_olmo.jsonl` (kernel `kaggle/olmo-posctrl/run.py`, copied from
`kaggle/olmo-direction/run.py` with round 0 loaded from file and the HINT feedback above; seed 20261114; embeddings in
the kernel as before). h_i drawn as in Block A among each context's saved follow-ups. **Primary direction for this
block: the all-follow-up s_{-h}** (≥ 4 follow-ups remain), PC1-PC3 and MC as Block A with NEUTRAL, GEO and ORACLE
from the saved OLMo-direction run, own Holm family. Secondary: the continuation direction rebuilt without h_i where
h_i is a consensus continuation, for the 67 contexts with k ≥ 3 continuations only (the denominator for the OLMo
"0.015" statement and M8), descriptive. Optional ORACLE-CC (cosine to the centroid of consensus continuations without
h_i) only if Block B1 is run, so the two generators answer question 7 together. Reading rules as Blocks A and B, applied
to OLMo statements only.

## Secondary for M1(b): turns as a fraction of the gap (free, existing data)

For the discovery and replication gpt-4o samples, both embedders, on the published (all-follow-up) s: per context
G_c = mean over follow-ups j of t_j·s_{c,−j} (s rebuilt without j, p with n − 1 neighbours), the follow-ups' own
leave-one-out projection (question 4). Report G − s_A,0 and, for each arm, mean Δs_A / mean (G − s_A,0) and mean
d_A,NEUTRAL / mean (G − s_A,0) with 20,000-resample bootstrap intervals (ratio of means, joint resampling). Descriptive.
Paper: the 5.3 tables gain a "fraction of the gap" column, replacing the "about a sixth" sentence.

## What the paper text will say (summary)

- Outcome 4 with all R_A upper bounds < 0.5: Section 5 keeps "barely steer", now with "Revealing the title of one
  real follow-up turns gpt-4o's ideas along s by H beyond revision (CI …); distance, reviewer and oracle feedback add
  at most … of that" in 5.3, abstract and conclusion, scoped to two rounds.
- Outcome 4 with some R_A ≥ 0.5: "barely" is withdrawn for that arm; the abstract states the ratio.
- Outcomes 2 or 5: "barely steer" is withdrawn; Section 5's direction results are demoted to "no difference from
  revision on a measure that failed its positive control"; the headline of Section 5 rests on the step results.
- Outcomes 1 or 3: the control is reported as uninformative/insufficient; the text applies M1(c)/(d) wording only.
- Block B0 answers question 7 in a sentence regardless; B1 adds one row to Table 4 style reporting and one sentence.
- In every case the "disjoint from the gpt-4o replication" sentence (line 437) is corrected (17 shared contexts).

## Cost (paid, paid spend, Azure via the gateway)

Measured analogue: the replication's NEUTRAL arm (`critique-rep-neutral`, 900 gpt-4o calls with round 0 copied from
GEO) used 491,366 input and 203,525 output tokens: USD 3.26 at USD 2.50 / 10.00 per million (546 in, 226 out per
call). HINT has the same call structure (2 revisions × 450 ideas = 900 calls) plus about 25 input tokens per call for
the title: about 514,000 input and 204,000 output tokens, **USD 3.3 (≈ AUD 5)**. ORACLE-C: 900 calls, about the same,
**USD 3.3 (≈ AUD 5)**; its feedback embeddings are local. A smoke test of 2 contexts × 1 idea per arm (8 calls) adds
about USD 0.03. **Total Block A + B1: about USD 6.6 (≈ AUD 10), in one day, project latent-gap-ideas.** Ledger tags
`review-posctrl-hint`, `review-posctrl-oraclec`. Blocks B0 and the M1(b) secondary use no API. Block C: no paid API,
Kaggle T4 x2 only (OLMo-direction run: 11,630 s for 50 contexts × 39 calls, about 6 s per call; HINT is 6 calls per
context, about 1.5 h for 150 contexts plus about 20 min model load and embedding; ORACLE-CC the same again).

## Local runtime and memory (16 GB Mac)

- Generation (A, B1): about 25-40 min each (NEUTRAL took 25 min for 900 calls; ORACLE-C embeds each idea locally for
  its score, as ORACLE). Peak about 2 GB (local Qwen3 ONNX in-process).
- Embedding the new rows: Qwen3 ONNX about 10-15 min per 1,350 rows (the replication embedded 5,400 in about 60 min);
  BGE-M3 on CPU about 15-25 min, peak about 4 GB. Run alone, not next to a generation (MPS ran out of memory once).
- Analysis (`review_posctrl.py analyse`, B0, M1(b)): one corpus embedding matrix at a time as float32 (113,000×1024,
  about 0.46 GB) plus the parquet (about 1 GB in pandas): peak about 3 GB; about 5-10 min with vectorised 20,000-sample
  bootstraps.

## Script and outputs

- `scripts/review_posctrl.py` with subcommands `holdout`, `embed-qwen`, `embed-bge`, `analyse` (Block A, B1 if
  present, M1(b)), `oracle-switch` (B0), `olmo-inputs` / `olmo-analyse` (Block C).
- `scripts/iterative_critique.py`: arms HINT and ORACLE-C added (feedback strings above, hold-out read from
  `LG_POSCTRL_HOLDOUT`); existing arms unchanged.
- Output: `docs/prereg/review_posctrl.json` (sections `holdout`, `qwen3e06`, `bgem3`, `oracle_switch`, `gap_fraction`,
  and `olmo` if Block C runs). Raw: `D/runs/critique-replication/posctrl_holdout.json`, `iter_ideas_posctrl.jsonl`,
  `posctrl_emb_{space}.npy`.

## Deviations

Any change after commit (seed, prompt, family, reading rule) is recorded in a dated section below and in the paper's
Appendix B, as in earlier registrations.
