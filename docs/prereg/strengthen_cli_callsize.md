# Pre-registration: does call size move the stress-test baseline? (5 vs 25 ideas per call, one run)

Written and committed 2026-11-05, before any idea of this run was generated.

## Why

The Section 6 stress test was run twice: at 25 ideas per call (scripts/inspiration_cli_generate.py) and at 5 ideas
per call (docs/prereg/strengthen_cli_powered.md). Between the runs the no-material (NONE) baseline moved (Claude
−0.0003 → −0.0062) and the past-game gap to NONE shrank (≈ −0.009 → −0.0035). The runs differ in call size, call
order and model pinning, so the paper calls this "run-to-run sensitivity, not an identified batching effect". This
run randomises call size within one run, with one call order and the model pinned, so that call size is the only
difference between arms and its effect is identified.

## Design (fixed)

- Generators, CLI flags, empty temporary working directory, closed stdin, prompt wording and JSON format: identical
  to docs/prereg/strengthen_cli_powered.md (gpt-6.1-sol via `codex exec ... -s read-only -m gpt-6.1-sol`; Claude Opus
  5.5 via `claude -p --model claude-opus-5-5` with the same disallowed tools). INSTRUCTED framing only.
- Arms per generator (4):
  - NONE-5: 80 calls of 5 ideas with no material (400 ideas).
  - NONE-25: 16 calls of 25 ideas with no material (400 ideas).
  - SG-5: all 200 steam_game stimuli of stimuli.parquet (40 per quintile), 40 calls of 5, each call 5 stimuli of one
    quintile in table order.
  - SG-25: the same 200 stimuli, 8 calls of 25, each call 5 stimuli from each quintile in table order.
  Each steam_game stimulus therefore appears once at each call size.
- Call order: the 144 calls of each generator run in one random order (numpy default_rng(20261106) permutation), so
  both sizes and both arms are interleaved over the same period. The two generators run as two parallel processes
  (different providers); never two calls to the same provider at once.
- Failures: a call whose answer does not parse or lacks an id is retried up to 3 times; calls still missing get one
  final pass. Every attempt (call, attempt number, status, seconds) is logged to attempts.jsonl, including parse
  failures. Missing calls are reported and excluded. Ideas with an empty title are dropped.
- Output: runs/inspiration/<gen>-clisize/ideas.jsonl with `call`, `size` and `arm` fields.
- Scoring (after generation): exactly as strengthen_cli_powered.py `analyze` (local Qwen3 ONNX on
  item_text(genres, tags, short_description); margin = max cos to future minus the mean over B = 5 size-matched past
  subsets of max cos to past; seed 20261103, a fresh generator per generator).

## Confirmatory tests (per generator; Holm over the two generators within each test, alpha = 0.05)

C1. Call size and the NONE baseline. NONE ideas only: OLS of margin on an indicator of 25-per-call, CR1
cluster-robust SE by call (96 clusters), two-sided t test with G − 1 df. Effect = NONE-25 minus NONE-5 idea mean,
with a cluster bootstrap 95 % CI (calls resampled within size, 4,000 draws).

C2. Call size and the past-game gap. NONE and SG ideas: OLS of margin on SG, size25 and SG × size25, CR1 by call (144
clusters); two-sided t test of the interaction with G − 1 df. Effect = (SG − NONE at 25) minus (SG − NONE at 5).

## Secondary (reported, not confirmatory)

- Wild cluster bootstrap-t (Rademacher, null imposed, 9,999 draws) for C1 and C2.
- SG − NONE gap at each size, with cluster bootstrap 95 % CIs.
- C1 and C2 pooled over the two generators (generator fixed effects, clusters = calls).
- Descriptive comparison of each arm's mean with the corresponding arm of the two earlier runs (NONE and steam_game
  at 25 and at 5 per call). The earlier runs used 10 steam_game stimuli per quintile; this run uses all 40.
- Attempt log summary: parse failures and retries per arm and size.

## Interpretation fixed in advance

- C1 significant after Holm for a generator: the paper may say that, for that generator, the no-material baseline
  depends on call size (direction and size as estimated), and that this accounts for the between-run shift if the
  estimate has the same sign and a CI covering the observed shift. Not significant: the paper reports the C1 CI as a
  bound on the call-size effect and keeps the run-to-run wording, saying that call size was tested and not detected.
- C2 likewise for the past-game gap. A null C2 with a CI that excludes the between-run change of the gap (+0.0055)
  would mean call size does not explain that change.

## Power (stated in advance)

From the 5-per-call run: idea-level SD of margin ≈ 0.025-0.027, and the SD of 5-idea call means (0.010-0.011) matches
independent ideas, so the within-call correlation is near zero. C1: SE ≈ 0.026 × sqrt(2/400) ≈ 0.0018, so a 0.005
shift is detected with ≈ 80 % power before Holm and Claude's observed 0.006 with ≈ 90 %. C2: SE ≈ 0.0032, so the
observed gap change of ≈ 0.0055-0.0065 has ≈ 40-50 % power; a null C2 is read through its CI. If the within-call
correlation is higher at 25 ideas per call, power is lower than stated.
