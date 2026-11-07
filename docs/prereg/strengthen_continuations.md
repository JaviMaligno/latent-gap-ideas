# Do the Section 5 results hold for true continuations? (pre-registered)

Pre-registered 2026-11-05, before any pair of this sample is drawn or labelled. Script:
`scripts/strengthen_continuations.py`; output `docs/prereg/strengthen_continuations.json` (estimates and tests) and
`docs/prereg/strengthen_continuations_labels.json` (every pair with both labels).

## Why

Section 5 calls "follow-ups" every later arXiv paper that cites a context (Semantic Scholar). Only 30 pairs were
labelled (`docs/paper/followup_check.md`): 3 continue their source (Claude), 2 (gpt-6.1-sol). So the paper can only say
its ideas are compared with later *citing* papers. This study labels a much larger blind sample to (i) measure the
share of follow-ups that continue their context, with a CI and inter-labeller agreement, and (ii) recompute the
Section 5.1 step gap and the Section 5.3 follow-up-specific direction on true continuations only.

## Sample

- Population: the 150 contexts and 2,883 saved follow-ups of Section 5 (`continue_from_context.contexts()`).
- **Stage 1 (400 pairs, stratified by context):** numpy `default_rng(20261105)`; 100 contexts drawn by
  `rng.permutation(150)[:100]` get 3 pairs, the other 50 get 2; within a context the follow-ups are drawn uniformly
  without replacement (`rng.choice`). Every context has >= 5 follow-ups.
- **Stage 2 (enrichment, adaptive):** every context where at least one labeller labels at least one stage-1 pair
  CONTINUES gets all its remaining follow-ups labelled, so that more contexts have >= 2 continuations for the
  direction test. If this exceeds 1,000 pairs, contexts are taken in a seeded random order until 1,000 is reached.
  Stage 2 is used only for E3 and E4, never for the share (E1) or agreement (E2), which use stage 1 only.
- Batches: within each stage the pairs are shuffled with the same generator and cut into batches of 20; both
  labellers see the same batches.

## Labellers and rubric

Two independent LLM labellers, each blind to the other and to every earlier label, reading only the two titles and
abstracts: **gpt-6.1-sol** (`codex exec -m gpt-6.1-sol`, subscription CLI) and **Claude Opus 5.5**
(`claude -p --model claude-opus-5-5`, subscription CLI). Same prompt for both, with the rubric of the original check:

- **CONTINUES**: builds directly on the source: extends, applies, improves or tests its method or result.
- **USES**: uses the source as a tool, dataset or baseline without building on its idea.
- **BACKGROUND**: cites it in passing / related work only.
- **UNCLEAR**: the abstracts do not allow a call.

Where the abstracts do not say how the source is used, the label leans to BACKGROUND or UNCLEAR. A batch whose output
does not parse to one valid label per pair is re-run (same prompt) up to 3 times.

**Primary construct: a continuation is a pair both labellers label CONTINUES** ("consensus"). Secondary: either
labeller CONTINUES ("lenient"); and CONTINUES + USES (the original check's "continuing").

## Estimands and tests

- **E1 (share, stage 1).** Share of follow-ups that are continuations, per labeller and consensus: (a) the
  follow-up-weighted share (each pair weighted by n_followups / pairs drawn in its context), which estimates the share
  among the 2,883 follow-ups; (b) the unweighted share of the 400 pairs. 95 % CIs by cluster bootstrap over contexts
  (2,000 resamples, seed 20261105); Wilson interval also reported for (b).
- **E2 (agreement, stage 1).** Cohen's κ for CONTINUES vs not (and for the four labels), with bootstrap CI;
  raw agreement.
- **E3 (confirmatory: the short step against continuations).** Contexts with >= 1 consensus continuation (stages
  1+2). Per context, gap = median step of its gpt-4o one-shot CTX ideas (Section 5.1, B1) − median step of its
  consensus continuations; step = cos(context, x). **T1 prediction: mean gap > 0** (one-sided Wilcoxon over contexts;
  primary cell gpt-4o / Qwen3, alpha 0.05). Reported alongside, not tested for confirmation: the same contexts' gap
  against all their follow-ups; the BGE-M3 and OLMo cells; the lenient set. **T2 (two-sided, exploratory):** within
  contexts that have both a consensus continuation and a pair labelled BACKGROUND by both, continuation step minus
  background step (are continuations closer to the context?).
- **E4 (exploratory: follow-up-specific direction from continuations, Section 5.3).** Contexts with >= 2 consensus
  continuations (stages 1+2). f = unit mean tangent of the continuations; p = unit mean tangent of the nearest later
  papers outside the saved follow-up set, as many as the context has continuations (robustness: as many as it has
  follow-ups, the original p); s = unit(f − (f·p)p). Change along s from round 0 to round 2 for GEO, Reviewer, ORACLE
  (gpt-4o and OLMo, Qwen3 and BGE-M3) and NEUTRAL (gpt-4o), and each critique arm minus NEUTRAL; Wilcoxon and
  bootstrap CIs, no multiplicity correction (exploratory). Positive control: leave-one-out alignment of the
  continuations. Also reported: the original (all-follow-up) s change on the same contexts, to separate a change of
  construct from a change of context subset. If fewer than 10 contexts qualify, E4 is reported descriptively only.

## What would change the paper

- E1/E2 give the paper a measured share of continuations with a CI instead of 3 of 30.
- If T1 holds, Section 5.1 can say the short step holds against true continuations (in N contexts). If it fails, the
  short step is a property of citing papers in general, not of continuations.
- E4 says whether the "no feedback-specific turn" reading holds for a direction built from continuations, and in how
  many contexts; with few contexts it bounds nothing and the paper says so.

## Cost

No paid API: both labellers are subscription CLIs. Stage 1 is 20 batches per labeller; stage 2 at most 50 more. No
embedding model is loaded: all vectors are the saved ones (Kaggle Qwen3 / BGE-M3 vectors of contexts, follow-ups and
ideas; local NEUTRAL vectors).
