# Pre-registration: a NEUTRAL revision arm for gpt-5.4-mini

Written and committed 2026-11-06, before any call of this arm. Analysis script: `scripts/strengthen_modern_neutral.py`
(committed before any NEUTRAL idea is scored). Output: `docs/prereg/strengthen_modern_neutral.json`.

## Why

Section 5.4 runs GEO, Reviewer (LLM), ORACLE and ORACLE-H with gpt-5.4-mini on 150 contexts from 2025, but none of
them has a revise-without-information control, so the paper says (Section 8) that for gpt-5.4-mini "the critique arms
have no neutral counterpart, so their changes are not separated from revising as such". gpt-4o
(`neutral_revision.md`) and OLMo (`strengthen_olmo_neutral.md`) have that arm; this gives it to the 2026 generator.

What I know before this run: all gpt-5.4-mini results of Section 5.4 and Table 4 (Qwen3 and BGE-M3), and the gpt-4o
and OLMo NEUTRAL results. Nothing about NEUTRAL ideas of gpt-5.4-mini.

## Design (fixed)

- Exactly `neutral_revision.md` with the generator, contexts and follow-ups of `modern_and_oracle_history.md`:
  gpt-5.4-mini through `gpt-5.4-mini`, temperature 1, `reasoning_effort: none`; the 150 contexts of
  `runs/continue-modern` and their follow-ups (`runs/followups-2025`); 3 ideas per context, 3 rounds; NEUTRAL feedback
  "Revise your proposal: write an improved version." every round; the revise prompt of `iterative_critique.py`.
- Round 0 is GEO's round-0 idea k of context i (`r0_from: GEO`), so only the two revisions are new (900 calls).
  NEUTRAL vs GEO compares round 2 from the same starting ideas; vs Reviewer and ORACLE, changes from independent
  round-0 draws.
- Command: `LG_RUN=continue-modern LG_FUPS=followups-2025 LG_ITER_ARMS=NEUTRAL LG_ITER_R0_FROM=GEO
  LG_ITER_OUT=iter_ideas_neutral.jsonl LG_TAG=modern-neutral LG_EXTRA='{"reasoning_effort": "none"}'
  ALIAS=gpt-5.4-mini uv run python scripts/iterative_critique.py generate`.
- Embedding: local Qwen3-Embedding-0.6B (ONNX), the encoder of the existing gpt-5.4-mini vectors
  (`oracle_history_emb_GEO_LLM_ORACLE_ORACLE-H.npy`, `emb.npz`); BGE-M3 locally on CPU with the settings of
  `revision_neutral.py embed-bge`, against the BGE-M3 vectors of `strengthen_bge_replication`.

## Measures

As `neutral_revision.md`: per context (mean over its ideas) step, fit, tangent alignment with the follow-ups' mean
tangent, change along the nearest non-citing later papers' direction p and along the follow-up-specific direction s
(construction of `direction_tangent.py` as applied to the 2025 contexts in `revision_null_effects.py`). Δ = round 2 −
round 0 (step as lengthening).

## Tests (Qwen3 primary; BGE-M3 its own Holm family)

The 14 tests of `neutral_revision.md`, critique arm minus NEUTRAL, paired by context: N1 (GEO step, one-sided), N2
(Reviewer step, one-sided), N3 fit, N4 hit, N5 alignment, N6 along s (two-sided, GEO / Reviewer / ORACLE each).
Paired Wilcoxon over contexts, bootstrap 95 % CIs (2,000 resamples, seed 20261030), Holm over the 14. Predictions:
N1 and N2 hold (critique lengthens the step beyond revision, as for gpt-4o); N3-N6 open. Each N6 CI is also read
against ±0.015: inside → "no feedback-specific turn of 0.015 or more"; crossing and not significant → "insufficient".
Also reported: NEUTRAL's own round-0 → round-2 changes (Wilcoxon). ORACLE-H vs NEUTRAL is exploratory.

## What the paper may say

The Section 8 sentence about gpt-5.4-mini is replaced by the result, per arm. No transfer to gpt-4o or OLMo.

## Cost

900 gpt-5.4-mini calls. At Section 5.4's measured 3.81M / 1.85M tokens per 7,801 calls and the conservative
$0.75 / $4.50 per M tokens of `modern_and_oracle_history.md`, ≈ $1.3 USD (≈ 2 AUD), 2026-11-06, paid spend on
Azure through the gateway, ledger tag `modern-neutral`. Wall time ≈ 20-30 min.

## Clarification on chronology (recorded 2026-11-06, after the analysis, on Codex review round 14)

This registration was committed in `67db500` before any call of the arm. The sentence "committed before any NEUTRAL
idea is scored" is not accurate for the analysis script: `scripts/strengthen_modern_neutral.py` was committed in
`f3011ed` at 17:07:50 UTC, while generation was running (about 265 of the 900 calls done). `iterative_critique.py`
computes each idea's Qwen3 step and fit during generation and stores them in the run file; apart from one
row's title and step, printed to check the output format, those values were not inspected before the script was
committed, and the analysis re-embeds the texts. The tests and reading rule are those
registered here.
