# Pre-registration: is building on a past game worse than no material? (gpt-4o, powered)

Written and committed 2026-11-06, before any call of this run. Scripts: `scripts/strengthen_anchoring_none.py`
(`draw`, `generate`, `score`, `analyse`). Raw data: `runs/inspiration/gpt-4o-anchoring/`. Output:
`docs/prereg/strengthen_anchoring_none.json`.

## Why

In the inspiration study (`inspiration_from_anything.md`, Section 6) gpt-4o told to build on a past Steam game gives
the worst anticipation of any source in Qwen3 (margin −0.0053 against −0.0011 for no material) and the second worst in
BGE-M3, but it is not significantly worse than no material (Mann-Whitney p 0.12, Holm over the nine sources p ≥ 0.47;
`revision_anchoring_test.json`), so the paper calls it exploratory. With 200 ideas per arm the test had little power
for a difference of 0.004. This run tests that one comparison, fixed in advance, with new ideas.

What I know before this run: all results of the inspiration study, its stress tests and `revision_anchoring_test.json`.
No idea of this run exists.

## Design (fixed)

- Generator, prompts, output format, parser and scoring exactly as the gpt-4o arms of `inspiration_from_anything.md`
  (`inspiration_generate.py`: gpt-4o-2024-08-06 via `gpt-4o-0806`, temperature 1, one idea per call).
- **STEAM-INSTRUCTED:** 1,000 new `steam_game` stimuli, 200 from each within-source distance quintile, drawn at random
  (`numpy.random.default_rng(20261107)`) from the pool's `steam_game` items not used in the original 200 and not
  sharing a name with a future game (the original exclusion), each shown once under the INSTRUCTED framing.
- **NONE:** 1,000 new calls with no material.
- Calls interleaved in a random order (`default_rng(20261108)`), 8 workers, ledger tag `anchoring-none`.
- Failed generations (empty after 3 attempts) are reported and dropped; no replacement.

## Measure

`margin` exactly as `inspiration_analysis.py` / `inspiration_cli_analysis.py`: max cos to the clean future set
(appid > 2,680,000) minus max cos to a random past subsample of the same size, mean over B = 5 subsamples (seed
20261103). Qwen3 locally (ONNX, the encoder of the stress-test scoring); BGE-M3 locally on CPU, against the
corpus's BGE-M3 vectors.

## Confirmatory test

**A1:** in Qwen3, the mean margin of STEAM-INSTRUCTED is lower than that of NONE, on the new ideas only. One-sided
Welch t-test, alpha 0.05 (a single pre-specified comparison; no multiplicity family). Mann-Whitney (one-sided) and a
bootstrap 95 % CI of the difference (2,000 resamples, seed 20261107) reported alongside.

## Secondary (reported, not confirmatory)

- A1 in BGE-M3 (replication; same test).
- The pooled estimate with the original 200 + 200 ideas, and the difference by quintile.
- Uptake of the new stimuli and isolation, as in the original.

## Reading rule

- A1 significant: the paper may say that, told to build on a past game, gpt-4o anticipates later releases worse than
  with no material, by the estimated amount; "exploratory" is dropped for that sentence (the rank among sources stays
  exploratory).
- A1 not significant: the paper says the comparison was tested with 1,000 ideas per arm and not detected, and gives
  the CI as the bound.

## Power

Idea-level SD of margin ≈ 0.026 (original gpt-4o arms). With 1,000 per arm the SE of the difference is ≈ 0.0012, so a
true difference of 0.004 is detected with ≈ 0.96 power at one-sided 0.05 and 0.003 with ≈ 0.80; BGE-M3's original
difference (0.003) is at that limit.

## Cost

2,000 gpt-4o calls. At the inspiration study's measured cost (3,800 calls, $5.96) ≈ $3.1 USD (≈ 5 AUD), 2026-11-06,
paid spend on Azure through an API gateway, ledger tag `anchoring-none`. Wall time ≈ 15-25 min.
