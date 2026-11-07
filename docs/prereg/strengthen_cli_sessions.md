# Pre-registration: call size and the no-material baseline across sessions

Written and committed 2026-11-05, before any idea of this run was generated.

## Why

The stress-test no-material (NONE) baseline moved between runs (Claude −0.0003 → −0.0062; gpt-6.1-sol by 0.006 between
two runs at the same call size). A within-run test (docs/prereg/strengthen_cli_callsize.md) did not detect a call-size
effect, but one run cannot say whether the effect is stable across sessions or how much the baseline varies from
session to session (Codex review round 6). This run repeats interleaved size arms over eight sessions.

## Design (fixed)

- Generators, CLI flags, empty temporary directory, closed stdin, NONE prompt wording and JSON format: identical to
  docs/prereg/strengthen_cli_callsize.md (gpt-6.1-sol via codex exec, read-only sandbox, model pinned; Claude Opus 5.5
  via claude -p, model pinned, same disallowed tools). NONE arm only (no stimuli).
- Sessions: 8 per generator. Each session = 16 calls of 5 ideas (80 ideas) and 4 calls of 25 ideas (100 ideas), in a
  random order drawn per session (numpy default_rng(20261110 + session)). Session k + 1 starts 2 hours after session
  k ends. The two generators run as two parallel processes, never two calls to the same provider at once.
- Failures: up to 3 attempts per call, every attempt logged with session, call, size, status and seconds; calls still
  missing get one retry at the end of their session. Missing calls are reported and excluded.
- Output: runs/inspiration/<gen>-clisessions/{ideas,attempts}.jsonl with session, call, size fields.
- Scoring: exactly as strengthen_cli_callsize.py (local Qwen3 ONNX, margin with B = 5 size-matched past subsets,
  seed 20261103, a fresh generator per generator).

## Confirmatory tests (per generator; Holm over the two generators within each test, alpha = 0.05)

S1. Average call-size effect. OLS of margin on size25 with session fixed effects, CR1 cluster-robust SE by call
(160 clusters), two-sided t with G − 1 df. Effect = coefficient on size25, with a cluster bootstrap 95 % CI
(calls resampled within session × size, 4,000 draws).

S2. Session-to-session variation of the baseline. Joint Wald test (CR1, by call) that the 7 session fixed effects
are zero in the S1 model; F reference with (7, G − 1) df. Effect size: SD across sessions of the session means of
the 5-per-call ideas, with a cluster bootstrap 95 % CI.

## Secondary (reported, not confirmatory)

- Size × session interaction: joint Wald test of the 7 interaction terms (CR1, F reference).
- Wild cluster bootstrap-t (Rademacher, null imposed, 9,999 draws) for S1.
- S1 pooled over generators (generator × session fixed effects).
- The size effect per session, and the previous runs (25-per-call original, 5-per-call powered, within-run call size)
  shown next to the eight sessions as further sessions, descriptively.
- Attempt-log summary.

## Interpretation fixed in advance

- S1 significant: the paper may say that, for that generator, the baseline depends on call size by the estimated
  amount. Not significant: report the CI as the bound on an average call-size effect.
- S2 significant: the paper may say that the baseline varies between sessions beyond call-level noise, by the
  estimated SD, which is the measured form of "run-to-run sensitivity". Not significant: report the SD and CI; the
  run-to-run wording is then not supported beyond the between-run differences already reported.
- Whether call size explains the historical shift is read from S1's CI against the historical shift and its own
  uncertainty (docs/prereg/strengthen_cli_callsize_joint.json), not from a point estimate.

## Power (stated in advance)

Idea-level SD ≈ 0.026, near-zero within-call correlation at 5 per call. S1: 640 ideas at 5 and 800 at 25,
SE ≈ 0.026 × sqrt(1/640 + 1/800) ≈ 0.0014, so a 0.004 effect has ≈ 80 % power before Holm. S2: each session mean of
80 ideas has SE ≈ 0.0029, so a between-session SD of about 0.003 or more is detectable with moderate power; smaller
session variation will not be.

## Deviation (recorded 2026-11-06, committed 09:38 UTC in b444d49, before any analysis of this run)

Between 01:08 and 07:13 on 2026-11-06 every call of both CLIs failed (non-zero exit after ~20 s for codex and ~170 s
for claude, at the same times for both, consistent with a connectivity outage rather than a model-side limit). The
generation loop moved on, so codex sessions 4 and 5 and claude session 4 (1 of 20 calls written) are missing; codex
session 6 completed in its end-of-session retry pass. Rather than analyse 6 and 7 sessions, the missing sessions are
re-run after each generator's eighth session ends, with the same per-session call order, prompts and 2-hour gap
(codex session 5 starts 2 h after the re-run session 4 ends). Their calls therefore fall later in the sequence than
planned; session labels are kept. The analysis is unchanged, and the attempt log records the times.

## Second deviation (recorded 2026-11-06, after the analysis, on Codex review round 11)

The re-run did not keep the two-hour gap stated above for the first re-run session of each generator: the generation
loop measured the gap from the end of session k − 1 (session 3, the previous night), not from the end of the eighth
session. Codex re-run session 4 started at 09:51, one minute after session 7 ended (09:50); Claude re-run session 4
started at 13:45, one minute after session 7 ended (13:44). Codex re-run session 5 did start two hours after re-run
session 4. Claude session 4 also keeps one call completed at 07:14:18, when connectivity returned, alongside the 19
re-run calls. The registration and the analysis are unchanged; the paper reports the actual timing.

