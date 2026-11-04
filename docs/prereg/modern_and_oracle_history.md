# A modern generator, and score feedback with history

Pre-registered 2026-11-01, before any call of either experiment (one 9-call smoke test of the ORACLE-H prompt on
context 0 with gpt-4o, and three single-call smoke tests of gpt-5.4-mini on an unrelated prompt, were run to check
the plumbing; their outputs are not analysed separately).

## Why

Two questions left open by `iterative_critique.md`, its exploratory direction test (`direction_tangent.json`) and the
OLMo replication (`continuation_reanalysis.json`):

1. **Is the pattern specific to 2023-era models?** gpt-4o-2024-08-06 and OLMo-7B-0424 both predate 2024. A 2026 model
   continuing 2025 papers tests whether the short step and the weak steering hold for a current model.
2. **Is "a similarity score does not steer" an artefact of a score without memory?** The ORACLE arm showed the model
   one scalar per round, without its earlier scores or proposals. ORACLE-H shows the full history of proposals with
   their scores and revises from the best one so far, like the grid decoder's best-of-rounds.

## Experiment M: gpt-5.4-mini on 2025 contexts

- **Generator:** gpt-5.4-mini (2026-03-17) through the gateway alias `gpt-5.4-mini`, temperature 1,
  `reasoning_effort: none`. Knowledge cutoff 2025-08-31 per OpenRouter and PromptLayer (no OpenAI primary source
  found; stated as such).
- **Contexts:** arXiv papers of our corpus dated 2025-01-01 to 2025-06-30, adult rows dropped, sampled with seed
  20261030; the first 150 with ≥ 5 follow-ups, in file order.
- **Follow-ups:** papers citing the context according to Semantic Scholar, with an arXiv id, published on or after
  2025-09-01 (after the cutoff); abstracts from the arXiv API; at most 50 per context
  (`runs/followups-2025`, `scripts/followups_fetch.py --from-date 2025-01-01 --to-date 2025-06-30 --since 2025-09-01`).
- **Arms:** CTX and CTX+MEM (5 ideas each, prompts of `continue_from_context`), and GEO, LLM, ORACLE, ORACLE-H
  (3 ideas × 3 rounds each, prompts of `iterative_critique`; the GEO band is unchanged, 0.574–0.661, so no statistic of
  the new contexts enters a prompt). CTX+NB and CTX+MOM are not run: their neighbours would have to come from a 2025
  map that the corpus covers only sparsely.
- **Embedding:** Qwen3-Embedding-0.6B (local ONNX; agrees with the Kaggle encoder to cos ≥ 0.99999,
  `encoder_agreement.json`); BGE-M3 via Kaggle as a robustness check.

## Experiment H: ORACLE-H with gpt-4o on the original 150 contexts

Same contexts, follow-ups, generator and protocol as `iterative_critique`, with one new arm, ORACLE-H, 3 ideas × 3
rounds, appended to `iter_ideas.jsonl` (arm `ORACLE-H`).

## Measures

As in `iterative_critique` and `direction_tangent.py`, per context (averaged over its ideas): step = cos(idea,
context); fit = max cos to the follow-ups; tangent alignment with the follow-ups' mean tangent direction, and its
follow-up-specific component (orthogonalised against the nearest non-citing later papers' direction). The
distance-matched "excess" is not used (out of support, see `direction_tangent.json`).

## Predictions

- **M1:** gpt-5.4-mini's CTX ideas sit closer to the context than the real follow-ups (median step gap > 0, paired
  Wilcoxon). Prediction: holds.
- **M2:** GEO lengthens the step from round 0 to round 2. Prediction: holds.
- **M3:** ORACLE (single score) does not raise the follow-up-specific alignment (two-sided test; prediction: no
  significant change at 0.05).
- **H1:** with gpt-4o, ORACLE-H raises fit from round 0 to round 2 more than ORACLE does (paired by context;
  one-sided). Directional, uncertain.
- **H2 (two-sided):** ORACLE-H's change in follow-up-specific alignment, round 0 to round 2. Reported with its CI.
- **M4 (two-sided):** the same H1/H2 quantities for gpt-5.4-mini.

Tests: Wilcoxon over contexts, bootstrap 95 % CIs (2,000 resamples, seed 20261030); Holm over {M1, M2, M3, H1}.
Everything else is exploratory.

## Cost

- M: CTX 750 + MEM 750 + four critique arms × 1,350 + reviewer 900 ≈ 7,800 calls; ~600 input and ~220 output tokens
  per call (smoke tests: 97 → 214–236). At a conservative $0.75 / $4.50 per M tokens (no Azure price on record),
  ≈ $3.5 + $7.7 ≈ $11.
- H: 1,350 gpt-4o calls at the measured ~520 / ~195 tokens ≈ $1.8 + $2.6 ≈ $4.5.
- Total ≈ $16 USD (≈ 24 AUD) in one day, paid spend on Azure through an API gateway, logged in `ledger.jsonl`
  with tags `modern-*` and `iter-critique`.

## Result (2026-11-02; `oracle_history.json`, keys `continue-from-context` and `continue-modern`)

Spend on 2026-11-01: gpt-5.4-mini 7,801 calls (3.81M input / 1.85M output tokens), gpt-4o 1,350 calls
(0.69M / 0.30M); ≈ $16 at the conservative prices of the Cost section. All 6,900 gpt-5.4-mini generations parsed. One
gpt-5.4-mini ORACLE idea (round 2) copies its context's title verbatim (step ≈ 1, tangent undefined); it is left out
of that context's tangent means.

Changes from round 0 to round 2, mean over 150 contexts (Wilcoxon p). "align" = tangent alignment with the follow-ups'
mean direction; "spec" = its follow-up-specific component (orthogonal to the nearest non-citing papers' direction).

| Run / arm | step lengthened | fit | align | spec |
|---|---|---|---|---|
| gpt-4o ORACLE | 0.032 (2e-18) | −0.012 (1e-6) | +0.014 (0.005) | −0.003 (0.52) |
| gpt-4o ORACLE-H | 0.023 (5e-14) | −0.008 (0.003) | +0.018 (5e-6) | +0.005 (0.13) |
| gpt-5.4-mini GEO | 0.061 (2e-26) | −0.021 (2e-11) | +0.025 (6e-5) | −0.003 (0.37) |
| gpt-5.4-mini LLM | 0.050 (7e-26) | −0.022 (3e-17) | +0.004 (0.40) | −0.004 (0.59) |
| gpt-5.4-mini ORACLE | 0.008 (2e-4) | −0.008 (8e-6) | −0.022 (2e-5) | −0.006 (0.19) |
| gpt-5.4-mini ORACLE-H | 0.012 (1e-9) | −0.007 (4e-4) | −0.012 (0.03) | +0.001 (0.96) |

- **M1 holds:** gpt-5.4-mini's CTX ideas sit 0.211 closer to the context than the real follow-ups (CTX+MEM 0.198;
  p ≈ 1e-26); the follow-ups' median step is 0.644. The modern model takes an even shorter step than gpt-4o (0.149).
- **M2 holds:** GEO lengthens the step by 0.061.
- **M3 holds:** ORACLE does not change the follow-up-specific alignment significantly (−0.006, p 0.19); its overall
  alignment falls (−0.022, p 2e-5).
- **H1 fails:** with gpt-4o, ORACLE-H's fit gain over ORACLE is +0.004 (CI −0.003 to +0.011, one-sided p 0.20).
  **M4:** the same for gpt-5.4-mini, +0.001 (CI −0.004 to +0.005, p 0.48). H2: ORACLE-H's follow-up-specific change
  is +0.005 (gpt-4o, p 0.13) and +0.001 (gpt-5.4-mini, p 0.96).
- Holm over {M1, M2, M3, H1}: M1 and M2 significant; H1 not; M3 is a no-change prediction and holds.

Reading: a 2026 model continuing 2025 papers takes the short step too, and a similarity score towards what was really
published does not orient it, with or without the history of its scores and the best proposal so far. With
gpt-5.4-mini even the verbal critique moves the direction only towards nearby papers in general (align +0.025), not
towards the follow-ups specifically (−0.003), unlike gpt-4o and OLMo, where GEO's specific component rose.
