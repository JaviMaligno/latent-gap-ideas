# Neutral revision: is the critique effect feedback-specific?

Pre-registered 2026-11-04, before any call of this experiment (a `--limit 2` smoke test of the plumbing is run first;
its outputs are kept in the same file and are part of the data, since the protocol is identical).

## Why

External review (point 16): the iterative-critique study (`iterative_critique.md`, paper Sections 5.2-5.3) has no arm
that revises without information, so its round-0 → round-2 changes (longer step, lower fit and hit, higher tangent
alignment) could come from revising as such rather than from the feedback. A NEUTRAL arm separates the two.

## Design

- **Same as `iterative_critique`:** the same 150 contexts, follow-ups, distractors and embeddings; generator
  gpt-4o-2024-08-06 through the gateway alias `gpt-4o-0806`, temperature 1; 3 independent ideas per context, each
  over 3 rounds (initial attempt + 2 revisions); the same revise prompt (context, previous proposal, "Feedback: ...",
  "Write a revised proposal: a full new title and abstract.").
- **NEUTRAL feedback (content-free), every round:** "Revise your proposal: write an improved version." No
  information about the paper, the distance, the follow-ups or any critique.
- **Round 0 is reused, not regenerated (budget).** Round 0 is the one-shot CTX prompt and carries no feedback, so it
  is arm-independent by design. To keep the spend near the approved ~3 USD, NEUTRAL's round-0 idea k of context i is
  GEO's round-0 idea k of context i (copied into the NEUTRAL file with `r0_from: GEO`); only the two revisions
  (900 calls) are new. Consequence, stated in advance: NEUTRAL and GEO share round 0, so NEUTRAL vs GEO compares round
  2 directly from the same starting ideas; NEUTRAL vs Reviewer and vs ORACLE compare changes from independent round-0
  draws.
- **Output:** `runs/continue-from-context/iter_ideas_neutral.jsonl`, ledger tag `iter-neutral`. Existing files untouched.
- **Embedding:** local Qwen3-Embedding-0.6B (ONNX), the encoder of the original step/fit (agreement with the Kaggle
  encoder cos ≥ 0.99999, `encoder_agreement.json`). The existing arms are read from their Kaggle Qwen3 vectors.
  BGE-M3 only if it can be run locally with the same settings as the Kaggle kernel; otherwise it is not reported.

## Measures (as in Sections 5.2-5.3 and Table 3)

Per context, averaged over its 3 ideas: step = cos(idea, context); fit = max cos to the follow-ups; hit (nearest
neighbour among follow-ups ∪ same-category later non-citing papers is a follow-up); tangent alignment with the
follow-ups' mean tangent; its change along the nearest non-citing later papers' direction p and along the
follow-up-specific direction s (`scripts/direction_tangent.py`). For each arm A, Δ_A = round 2 − round 0 (step as
reduction, round 0 − round 2).

## Predictions and tests

If the critique effects are feedback-specific, each critique arm differs from NEUTRAL:

- **N1:** GEO lengthens the step more than NEUTRAL (Δstep_GEO > Δstep_NEUTRAL; one-sided).
- **N2:** the Reviewer lengthens the step more than NEUTRAL (one-sided).
- **N3 (two-sided):** the fit change differs between each critique arm and NEUTRAL (three tests: GEO, Reviewer, ORACLE).
- **N4 (two-sided):** the hit change differs between each critique arm and NEUTRAL (three tests).
- **N5 (two-sided):** the tangent-alignment change differs between each critique arm and NEUTRAL (three tests).
- **N6 (two-sided):** the follow-up-specific change (along s) differs between each critique arm and NEUTRAL (three
  tests). In particular the paper's one positive follow-up-specific result for gpt-4o, GEO +0.015 (Qwen3), survives
  only if GEO's change along s exceeds NEUTRAL's.

Also reported: NEUTRAL's own round-0 → round-2 changes (Wilcoxon, as in Tables 2-3).

Tests: paired Wilcoxon over contexts of Δ_A − Δ_NEUTRAL; bootstrap 95 % CIs (2,000 resamples over contexts, seed
20261030); Holm over the 14 tests of N1-N6. Prediction direction: N1 and N2 hold (critique is more than revising);
N3-N6 are open. Anything else is exploratory.

## Cost

900 gpt-4o calls (2 revisions × 3 ideas × 150 contexts) at the measured ~600 input / ~220 output tokens of a
revision ≈ 0.54M input + 0.20M output ≈ $1.4 + $2.0 ≈ **$3.4 USD** (≈ 5 AUD) in one day, paid spend on Azure
through the gateway, logged in `ledger.jsonl` with tag `iter-neutral`. Wall time ≈ 15-25 min with 8 workers.

## Result (run 2026-11-04; `revision_neutral.json`; `scripts/revision_neutral.py`)

Spend: 900 gpt-4o calls (tag `iter-neutral`, incl. the 12-call smoke test), 0.493M input / 0.204M output tokens,
≈ $3.27 at gpt-4o-2024-08-06 list prices. All 900 revisions parsed. The existing arms reproduce Tables 2-3 of the paper
exactly; local vs Kaggle encoder agreement on the 450 shared round-0 texts: min cos 0.999996 (Qwen3), 0.99997 (BGE-M3,
local CPU float32 vs Kaggle T4 float16).

NEUTRAL alone (Qwen3, round 0 → 2): step lengthened 0.045 (p 5e-24); fit −0.015 (p 1e-6); hit −0.013 (p 0.67);
alignment 0.232 → 0.262 (+0.030, p 2e-8); along p +0.024 (p 2e-6); along s (follow-up-specific) +0.019 (p 5e-4).
BGE-M3: step 0.034, fit −0.008, hit −0.053 (p 0.03), alignment 0.393 → 0.423 (+0.029), along p +0.026, along s +0.014
(p 0.007).

Critique change minus NEUTRAL change (Qwen3, Holm over the 14 tests; BGE-M3 in brackets, its own Holm):

| Test | GEO | Reviewer | ORACLE |
|---|---|---|---|
| N1/N2 step lengthened (one-sided) | +0.042, Holm 1e-16 [+0.013, 2e-5] | +0.018, Holm 2e-4 [+0.014, 6e-6] | — (expl.: −0.013, p 0.003 [−0.004, 0.09]) |
| N3 fit | −0.026, Holm 6e-11 [−0.002, 1] | −0.016, Holm 0.001 [−0.009, 0.001] | +0.004, Holm 1 [−0.003, 0.75] |
| N4 hit | −0.096, Holm 0.001 [−0.033, 0.61] | −0.038, Holm 0.74 [−0.033, 1] | −0.040, Holm 0.86 [+0.002, 1] |
| N5 alignment | +0.003, Holm 1 [+0.012, 0.21] | +0.000, Holm 1 [−0.003, 1] | −0.016, Holm 0.12 [−0.014, 0.22] |
| N6 along s | −0.003, Holm 1 [+0.010, 0.48] | −0.013, Holm 0.14 [−0.008, 0.94] | −0.022, Holm 0.018 [−0.014, 0.27] |

- **N1, N2 hold** in both embedders: GEO and the reviewer lengthen the step beyond ordinary revision; about half of
  GEO's lengthening (0.045 of 0.087) and 71 % of the reviewer's (0.045 of 0.063) is reproduced by revising with no
  information. ORACLE lengthens the step *less* than NEUTRAL (exploratory).
- **N3:** GEO and the reviewer lose more fit than NEUTRAL in Qwen3 (the reviewer also in BGE-M3); ORACLE does not differ.
- **N4:** only GEO's hit drop exceeds NEUTRAL's, in Qwen3 only.
- **N5:** no critique arm raises alignment more than NEUTRAL (Holm ≥ 0.12 in both embedders).
- **N6:** no critique arm has a larger follow-up-specific change than NEUTRAL; ORACLE's is smaller (Qwen3, Holm 0.018;
  BGE-M3 p 0.03 uncorrected, Holm 0.27).

Reading: what critique adds over plain revision is a longer step and (GEO, reviewer) a further loss of fit; the turn
towards the follow-ups' direction, including its follow-up-specific part, comes with revision as such. The single
similarity score towards the real follow-ups does worse than no information on the step and on the follow-up-specific
direction. Limitation: gpt-4o only (no OLMo or gpt-5.4-mini NEUTRAL arm); round 0 shared with GEO by design.
