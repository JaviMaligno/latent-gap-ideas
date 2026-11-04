# Steering by iterative critique: can feedback lengthen the LLM's step?

Pre-registered 2026-10-30, after the results of `continue_from_context.md` (Part B) and before any call of this
experiment.

## Why

In `continue_from_context`, gpt-4o continuations sit 0.14–0.15 (cosine) closer to their context than the real
follow-ups (B1), whether or not neighbours are shown, and no one-shot guidance fixes it: memory of earlier titles
diversifies the ideas (B3) but does not lengthen the step, and a verbal hint of where the area is moving lowers the
hit rate (B4 reversed). This test asks whether *iterative* guidance — feedback after each attempt, then a revision —
moves the model to a realistic step, and whether that improves the match with what was really written next.

Exploratory finding that shapes the design: in the matched baseline, real follow-ups drift towards the centre of
the cap *less* than distance-matched non-citing papers (0.427 vs 0.471). LLM ideas drift even less (0.13–0.15), but
at a shorter step, so the comparison is confounded by step length. The rule-based critique therefore targets step
length only, not "be more specific".

## Design

- **Contexts:** the same 150 contexts as `continue_from_context` (first 150 with ≥ 5 follow-ups, file order),
  same embeddings, same follow-ups and distractors.
- **Generator:** gpt-4o-2024-08-06 through the gateway (`gpt-4o-0806`), temperature 1. Training cutoff before
  every follow-up (all ≥ 2024-01-01).
- **Per context and arm:** 3 independent ideas, each refined over **3 rounds** (initial attempt + 2 revisions).
  The initial attempt uses the CTX prompt of `continue_from_context` verbatim. Each revision sees the context, its
  own previous proposal and the feedback of that round, and returns a full new title and abstract.
- **Arms (feedback after each round):**
  - **GEO (realistic, rule-based):** "Your proposal's similarity to the original paper is s (embedding cosine).
    Real follow-up papers typically sit between 0.57 and 0.66. [Too close: build on the paper but take a bigger
    step — a new question, method or setting, not a variation of the same one.] / [Too far: keep a clearer link to
    the original paper.] / [Within the range: keep the step, sharpen the contribution.]" The band is the
    inter-quartile range of real follow-up steps from Part A (0.574–0.661), an aggregate statistic of the same 150
    contexts; per-context targets are never shown. This is a stated, mild leakage.
  - **LLM (realistic, reviewer):** a separate gpt-4o call acts as a reviewer: "Is this a genuine follow-up that
    moves the research forward, or mostly a rephrasing or minor variation of the original paper? In at most 80
    words, say what would make it a stronger next paper." Its answer is the feedback. The reviewer sees the context
    and the proposal, never any follow-up.
  - **ORACLE (ceiling, leaks the future):** "Your proposal's similarity to the closest paper actually published as a
    follow-up is f (embedding cosine). Try to get closer." f = max cosine to the context's real follow-ups. This is
    the score-only hot-and-cold of the grid, applied to continuation; it is an upper bound on score feedback, not a
    realistic method.
- **Embedding:** local Qwen3-Embedding-0.6B, same item text as before.

## Measures

For every idea and round: `step = cos(idea, context)`, `fit` = max cos to the context's follow-ups, `hit` = the
idea's nearest neighbour in (follow-ups ∪ distractors) is a follow-up (as in Part B). Unit = context (average over
its 3 ideas); round 0 vs round 2 paired within arm, and round 2 compared with the Part B CTX arm (k = 0..2).

## Predictions

- **C1:** GEO lengthens the step: mean step at round 2 lower than at round 0, and its gap to the real follow-ups'
  median step shrinks by at least half of the Part B gap (≥ 0.07). Prediction: holds.
- **C2:** GEO does not lower the hit rate (round 2 ≥ round 0 − 0.03). Directional, uncertain.
- **C3:** LLM lengthens the step (round 2 lower than round 0). Prediction: holds, smaller than GEO.
- **C4 (two-sided):** hit rate change from round 0 to round 2 under LLM.
- **C5:** ORACLE raises fit from round 0 to round 2, but by less than 0.07 (as score feedback did in the grid).

Tests: paired Wilcoxon over contexts; bootstrap 95 % CIs (2,000 resamples over contexts, seed 20261030); Holm over
{C1, C2, C3, C5}. C4 reported with its CI. Anything else is exploratory and labelled as such.

## Cost

Calls: GEO and ORACLE 150 × 3 × 3 = 1,350 each; LLM 1,350 generations + 900 reviewer calls. Total ≈ 4,950 calls.
Tokens (from the Part B ledger, ~1,170 input and ~215 output per CTX call; revisions add the previous proposal and
the feedback): ≈ 1,000 input and 250 output per call on average → ≈ 5.0M input + 1.2M output ≈ $12.4 + $12.4
≈ **$25 USD in one day** (≈ 38 AUD at ~1.5 AUD/USD), paid spend on Azure through an API gateway, logged in
`ledger.jsonl` with tag `iter-critique`. Wall time ≈ 1–1.5 h with 8 workers; embedding of ~13,500 texts locally
≈ 1 h.

## Result (run 2026-10-30; `continue_from_context_result.json`, key `iterative_critique`; spend $16.06, 4,950 calls)

Real follow-ups sit at a median step of 0.620 from their context; one-shot CTX ideas at 0.765.

| Arm | step r0 → r2 | fit r0 → r2 | hit r0 → r2 |
|---|---|---|---|
| GEO | 0.766 → 0.679 | 0.677 → 0.636 | 0.627 → 0.518 |
| LLM reviewer | 0.766 → 0.703 | 0.679 → 0.647 | 0.644 → 0.591 |
| ORACLE | 0.761 → 0.729 | 0.677 → 0.666 | 0.664 → 0.611 |

- **C1 holds:** GEO lengthens the step by 0.087 (CI 0.080–0.094) and closes 0.087 of the 0.148 gap (≥ 0.07).
- **C2 fails:** the hit rate drops by 0.11 (CI −0.16 to −0.06).
- **C3 holds:** the LLM reviewer lengthens the step by 0.063 (CI 0.056–0.069).
- **C4:** its hit rate drops by 0.05 (CI −0.09 to −0.02).
- **C5 holds as stated (gain < 0.07), and in the strongest form:** with the similarity to the real follow-ups as
  feedback, fit does not rise at all; it falls by 0.012 (CI −0.016 to −0.007).
- **Correction (2026-11-01, after review): C5 fails.** The prediction was that ORACLE *raises* fit (by less than
  0.07); fit fell by 0.0115, so the prediction failed. The line above, written on 2026-10-30, is kept as history.
  Also corrected: ORACLE is a single-score oracle with no history or best-so-far, not "an upper bound on score
  feedback" as the Design section says; a version with the score history and best proposal so far is planned.

Reading: feedback can make gpt-4o take a longer step, but not in the direction research actually went; every arm
matches the real follow-ups worse after revision, including the one that is told how close it is to them.
Caveat (not pre-registered): moving away from the context in any direction lowers the cosine to follow-ups that
share the context's direction, so part of the fit/hit drop is mechanical; separating it needs a length-matched
random-direction baseline.

### Exploratory: distance-matched baseline (`scripts/iterative_critique_baseline.py`)

Each idea is compared with the later arXiv paper of the corpus (2023+, not a follow-up of the context) whose cosine
to the context is closest to the idea's step (nearest match, with replacement; a first version that shared a no-reuse
pool across arms starved later arms of close matches and was discarded). Excess fit = idea fit − matched paper fit.

| Arm | excess r0 | excess r2 | change r0 → r2 |
|---|---|---|---|
| GEO | −0.038 | −0.044 | −0.006 (p 0.20) |
| LLM | −0.037 | −0.044 | −0.007 (p 0.20) |
| ORACLE | −0.036 | −0.039 | −0.003 (p 0.34) |

Reading: (1) the fit drop after critique is mechanical — at the distance the revised ideas reach, an arbitrary later
paper loses about as much fit; critique neither improves nor measurably worsens the direction. (2) Before any
critique, gpt-4o continuations already point towards the real follow-ups about 0.037 less than an arbitrary later
paper at the same distance. Two confounds, in opposite directions: the matched papers are from 2023–2026, the same era
as the follow-ups, so they share period trends the generator cannot know (inflates the gap); ideas and follow-ups are
embedded with the local ONNX encoder and corpus papers with the Kaggle one, a ~0.002 mismatch that favours the ideas
(shrinks it). The hit comparison is not reported because the same encoder mismatch decides near ties.

**Correction (2026-11-01, after review): this baseline is withdrawn as not identified.** At round 0, 69–97 % of the
ideas are closer to their context than any later paper of the pool (40–84 % at round 2;
`direction_tangent.json`, key `share_above_pool_max`), so the matched paper is systematically further out than the
idea and "excess fit" and its change mix direction with step mismatch. Readings (1) and (2) above, and the later
robustness-grid excess numbers in `continuation_reanalysis.json`, are kept for the record but no longer supported.
The encoder-mismatch caveat is moot (`encoder_agreement.json`, min cosine ≥ 0.99999). The exploratory replacement is
the step-invariant tangent-alignment test in `scripts/direction_tangent.py`.
