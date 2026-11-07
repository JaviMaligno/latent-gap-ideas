# Pre-registration: powered re-run of the Section 6 CLI stress test

Written and committed 2026-11-05, before any idea of this run was generated.

## Why

The Section 6 stress test (scripts/inspiration_cli_generate.py) put 25 stimuli in each call, so each source filled
only two calls and no call-level test could reach significance (smallest exact permutation p = 1/15). The paper
therefore reports the steam_game-below-NONE direction (−0.008 codex, −0.010 claude) "as a direction, not a test".
This run keeps the stimuli and prompt and only shrinks the call, so the call-level test has enough calls.

## Design (fixed)

- Generators: gpt-6.1-sol (`codex exec --skip-git-repo-check --ephemeral -s read-only -m gpt-6.1-sol`) and Claude
  Opus 5.5 (`claude -p --model claude-opus-5-5 --output-format text`, tools Bash, Edit, Write, Read, WebSearch,
  WebFetch, Glob, Grep, Agent disallowed), each from an empty temporary directory, stdin closed.
- Stimuli: identical to the original stress test, the first 10 stimuli of each source x quintile cell of
  stimuli.parquet (450). INSTRUCTED framing only, same prompt text and JSON format as the original script.
- Calls: 5 stimuli per call, consecutive in the original pick order, so every call holds 5 stimuli of one
  source x quintile cell (2 calls per cell, 90 calls). NONE: 20 calls of 5 ideas (ids 0-4) with no material, same
  prompt as the original with 5 instead of 25.
- Call order: the 110 calls of each generator run in one random order (numpy default_rng(20261105) permutation), so
  NONE calls are interleaved with stimulus calls instead of coming last. Calls are sequential, one generator after
  the other or the two in parallel processes (they hit different providers); never two calls to the same provider at
  once.
- Failures: a call whose answer does not parse or lacks an id is retried up to 3 times; calls still missing are
  retried once more in a final pass. Calls still missing after that are reported and excluded. Ideas with an empty
  title are dropped (as in the original).
- Output: runs/inspiration/<gen>-cli5/ideas.jsonl with a `call` field.
- Scoring (after all generation finishes): local Qwen3 ONNX embedder (src/latent_gap/embedder.py LocalQwen) on
  item_text(genres, tags, short_description); anticipation margin and isolation exactly as
  scripts/inspiration_cli_analysis.py (future set = non-past, appid > 2,680,000; margin = max cos to future minus the
  mean over B = 5 size-matched random past subsets of max cos to past; seed 20261103, a fresh generator per
  generator). Distance = the stimuli table's pool `rank`, as in the paper's stress-test slope.

## Confirmatory tests (per generator; Holm over the two generators within each test, alpha = 0.05)

H1. steam_game vs NONE at the call level. Unit = call mean margin; 10 steam_game calls vs 20 NONE calls; two-sided
exact Mann-Whitney U. Effect reported as difference of idea-weighted means with a cluster bootstrap 95 % CI
(calls resampled within group, 4,000 draws).

H2. Distance slope. OLS of margin on rank with source fixed effects over the 450 stimulus ideas; CR1 cluster-robust
SE by call (90 clusters), two-sided t test with G − 1 df.

## Secondary (reported, not confirmatory)

- Permutation of call labels (20,000 random permutations) for H1; wild cluster bootstrap-t (Rademacher, null
  imposed, 9,999 draws) for H2.
- Each of the nine sources vs NONE at the call level, Holm over nine.
- Isolation slope (the P3 manipulation check) with the same clustered model.
- Pooled over the two generators: steam_game vs NONE with generator-stratified call means (difference within each
  generator, then averaged; permutation within generator).
- Replication check against the original 25-per-call run (same direction and magnitude of the steam_game gap).

## Interpretation fixed in advance

- If H1 is significant after Holm for a generator, the paper may say that for that generator a past game as material
  anticipates the future worse than no material at the call level. If not, the paper keeps the direction-only
  wording and adds the call-level CI of this run as a bound on the gap.
- If H2 is not significant, the null slope is reported with its clustered CI as the bound; a null slope is read with
  the P3 manipulation check (if distance does not raise isolation here either, the null says little about distance).

## Power (stated in advance)

From the original run: steam_game − NONE ≈ −0.009, idea-level SD of margin ≈ 0.025. With 50 vs 100 ideas the
standard error of the gap is ≈ 0.0043 whatever the call size, so if the true gap is −0.009 the power of H1 is about
50-55 % per generator, lower if the original estimate is inflated. This run removes the structural impossibility of
a call-level test; it does not guarantee a significant result.
