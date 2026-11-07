# The short step against true continuations: full census of the follow-ups (pre-registered)

Pre-registered 2026-11-05, before any stage-3 pair is drawn or labelled. Extends
`docs/prereg/strengthen_continuations.md` (commit `ae4f914`), whose outputs (`strengthen_continuations.json`,
`strengthen_continuations_labels.json`) stay unchanged. Script: `scripts/strengthen_continuations_full.py` (it imports
the sampling, prompt, CLI and parsing code of `scripts/strengthen_continuations.py` unchanged). Output:
`docs/prereg/strengthen_continuations_full.json` (estimates and tests) and
`docs/prereg/strengthen_continuations_full_labels.json` (all 2,883 pairs with both labels).

## Why

The first study labelled 731 of the 2,883 context/follow-up pairs: 400 stratified (stage 1) and every follow-up of the
25 contexts where a labeller had found a continuation (stage 2). Only 18 contexts had a consensus continuation and they
were adaptively ascertained (found because the first sample hit one). The pre-registered test that gpt-4o's one-shot
step gap stays positive against continuations failed in its primary cell (gpt-4o / Qwen3: 0.023, CI −0.025 to
+0.069, n = 18). A reviewer judged n = 18 insufficient and the ascertainment a problem for generalisation. Labelling
every remaining pair makes the set of contexts with a continuation a census (no sampling, no adaptive step) and gives
the largest n this design allows.

What is already known to me before this pre-registration: all stage-1/2 labels and results above. Nothing about the
2,152 unlabelled pairs has been looked at by any labeller.

## Stage 3 (the rest of the census)

- **Pairs:** every (context, follow-up) of the 150 contexts / 2,883 follow-ups (`continue_from_context.contexts()`)
  that is not in stage 1 or stage 2: expected 2,152 pairs. Ids `s3-{ctx}-{followup}`.
- **Batches:** pairs shuffled with numpy `default_rng(20261105 + 3)` and cut into batches of 20 (expected 108); both
  labellers see the same batches.
- **Labellers, prompt, rubric:** identical to the first study, through the same code path (`prompt_for`, `run_cli`,
  `parse` of `strengthen_continuations.py`): gpt-6.1-sol via `codex exec -m gpt-6.1-sol` (stdin closed, read-only
  sandbox, ephemeral) and Claude Opus 5.5 via `claude -p --model claude-opus-5-5` with all tools disallowed. Each
  labeller is blind to the other and to every earlier label. Labels CONTINUES / USES / BACKGROUND / UNCLEAR.
- **Execution:** the two labellers run as two parallel processes, each labelling its batches sequentially.
- **Retries (all logged):** a batch whose output does not parse to one valid label per pair (no JSON, invalid JSON,
  missing id, invalid label, empty output, timeout) is re-run with the same prompt, up to 3 attempts per pass. Failed
  batches are re-run in a resume pass (up to 3 more attempts). If after that a batch still lacks valid labels for some
  pairs, the valid rows of its most complete attempt are kept and each missing pair is labelled in a single-pair call
  with the same prompt (up to 3 attempts), as was done for one pair in the first study. Every attempt is appended to
  `runs/strengthen-continuations/stage3/attempts_{labeller}.jsonl` (batch, attempt, seconds, status, missing ids) and
  the counts per labeller and status go into the output JSON.

## Constructs

As before: **consensus continuation** = both labellers say CONTINUES (primary); **lenient** = either says CONTINUES;
**both-background** = both say BACKGROUND. All estimands below use the labels of stages 1 + 2 + 3, i.e. all 2,883
pairs, each pair labelled once by each labeller.

## Estimands and tests

- **F1 (share, census).** Share of the 2,883 follow-ups that are continuations: per labeller, consensus, lenient and
  CONTINUES-or-USES (both). This is a census of this population, so the share has no sampling error; a cluster
  bootstrap CI over contexts (2,000 resamples, seed 20261105) is reported as the uncertainty for a superpopulation of
  similar contexts. Also: number of contexts with >= 1, >= 2, >= 3 consensus continuations; the stage-1 weighted
  estimate (4.4 %) set against the census value.
- **F2 (agreement, census).** Cohen's κ over all 2,883 pairs for CONTINUES vs not, CONTINUES-or-USES vs not, and the
  four labels, with cluster bootstrap CIs; raw agreement; confusion matrix. Also κ per stage.
- **F3 (confirmatory, primary).** Contexts with >= 1 consensus continuation in the census. Per context, gap = median
  step of its gpt-4o one-shot CTX ideas (Section 5.1, B1) − median step of its consensus continuations; step =
  cos(context, x) in Qwen3. Estimate: mean gap over contexts, 95 % CI by bootstrap over contexts (each context one
  cluster). **Prediction: mean gap > 0**, one-sided Wilcoxon signed-rank over contexts, alpha 0.05. Single test, no
  multiplicity correction.
  - **Reading rule (fixed now).** Reference effect = half of the gap against all follow-ups of the same contexts
    (computed on the same contexts in the same cell; on the 18 earlier contexts it was 0.120, so about 0.06). If the
    test passes: the short step holds against continuations. If it fails and the upper 95 % CI bound is below the
    reference effect: the evidence *contradicts* a gap of that size against continuations. If it fails and the upper
    bound is at or above the reference effect: the evidence is *insufficient*.
  - **Minimum detectable effect.** For the realised number n of contexts and the realised SD of the per-context gaps,
    the mean gap detectable with 80 % power at one-sided alpha 0.05: (t_{0.95,n−1} + t_{0.80,n−1}) · SD/√n, plus the
    normal-approximation value and an approximate Wilcoxon version (the t value divided by √(3/π), the Wilcoxon
    test's asymptotic relative efficiency under normality). Also the same MDE computed with n = 18 and the realised SD, to show what the census added.
- **F3 secondary cells (reported, not confirmatory):** the same gap and one-sided test for gpt-4o / BGE-M3,
  OLMo / Qwen3, OLMo / BGE-M3, and for the lenient set in all four cells; in each cell, the gap against all follow-ups
  of the same contexts and the paired difference (continuations − all, two-sided Wilcoxon). Subsets of the primary
  cell: (a) the contexts that were not among the 18 of the first study (a replication on contexts never analysed);
  (b) the 18 earlier contexts with census labels.
- **T2 (two-sided).** Within contexts that have both a consensus continuation and a both-background follow-up: mean
  continuation step minus mean background step, per cell, with bootstrap CI and two-sided Wilcoxon.
- **F4 (direction from continuations, Section 5.3).** Same construction as E4 of the first study (f from the
  continuations' tangents; p from the nearest later papers outside the saved follow-ups, as many as there are
  continuations, robustness as many as follow-ups; s = unit(f − (f·p)p); change along s from round 0 to round 2 for
  GEO, Reviewer, ORACLE, and NEUTRAL for gpt-4o; each critique arm minus NEUTRAL; positive control = leave-one-out
  alignment of the continuations; also the original all-follow-up s change on the same contexts).
  - **Confirmatory only if >= 10 contexts have >= 2 consensus continuations.** Then: gpt-4o / Qwen3, k =
    continuations; the three contrasts critique arm − NEUTRAL along s, two-sided Wilcoxon, Holm over the 3.
    Prediction from the paper's reading ("no feedback-specific turn detected"): no contrast significant after Holm.
    Because a null is not support, each contrast's 95 % CI is also read against ±0.015 (the post-hoc bound already
    used in Section 5.3): inside = consistent with no turn of that size; crossing it = insufficient. All other cells
    descriptive.
  - **Otherwise descriptive only**, with no test read as confirmatory.
- **F5 (S2-flagged subset as analysed).** Over the census: of the follow-ups in the S2-flagged subset as analysed
  (flagged follow-ups of the 68 contexts with >= 3 flagged ones), how many are consensus continuations; share of
  continuations among flagged vs not flagged follow-ups (all contexts); share of consensus continuations that are in
  the subset; and the 30-pair original check against the census labels.

## What would change the paper

- Sections 5 and 8 get a census share of continuations and κ over all pairs instead of a 400-pair sample.
- Section 5.1 and Table 6: per the reading rule, the short step against continuations either holds, is contradicted
  at half its citing-paper size, or stays undetermined, now on every context with a continuation rather than 18
  adaptively found ones.
- Section 8: whether the continuation direction can be tested (>= 10 contexts) and what it says.

## Cost

No paid API: both labellers are subscription CLIs (about 108 batches each, about 40 s per batch). No embedding model
is loaded; all vectors are the saved ones.
