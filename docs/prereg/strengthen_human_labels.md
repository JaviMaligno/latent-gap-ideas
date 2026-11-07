# Pre-registration: blind human check of the LLM continuation labels (40 pairs)

Written and committed 2026-11-05, after the LLM census labels of docs/prereg/strengthen_continuations_full.md existed
and before the sample was drawn or any pair was labelled by a person.

## Why

Every continuation label in Sections 5 and 5.1 comes from two LLM labellers (gpt-6.1-sol and Claude Opus 5.5). No
person has checked them. This is a small check of how often their consensus "CONTINUES" is right, and of which way
they err when they disagree. It is not a substitute for independent expert labels.

## Labeller

The author (one person), blind to the LLM labels and reasons, to the stratum of each pair and to its embedding
distances. The author knows the paper's hypotheses and is not independent of them; this is stated wherever the
result is used.

## Sample (fixed)

From the census file docs/prereg/strengthen_continuations_full_labels.json (all 2,883 context × follow-up pairs),
three strata, drawn without replacement with numpy default_rng(20261107), each stratum in census order then
`rng.choice`:

- S-cons: both labellers CONTINUES — 25 pairs.
- S-dis: exactly one labeller CONTINUES — 8 pairs.
- S-bg: both labellers BACKGROUND — 7 pairs.

The 40 pairs are shown in one random order (same generator, after the draws), with the source and follow-up titles
and abstracts exactly as the LLMs saw them, and the same four labels and definitions as the LLM prompt (CONTINUES,
USES, BACKGROUND, UNCLEAR), plus an optional free-text note. Labels are stored per pair as they are given; a pair
may be changed until the author marks the set as finished; nothing is analysed before that.

## Analysis (fixed)

Primary.
- P1. Precision of consensus CONTINUES: share of S-cons pairs the author labels CONTINUES, with a Wilson 95 % CI;
  also the share labelled CONTINUES or USES.

Secondary (descriptive).
- In S-dis: how many times the author agrees with each labeller on CONTINUES vs not.
- In S-bg: how many pairs the author labels CONTINUES (a check for gross misses only; with 7 pairs and a ≈ 5 %
  base rate this cannot estimate the miss rate).
- Agreement of each labeller with the author on the 40 pairs (raw, and Cohen's κ on CONTINUES vs not), stated as
  stratified-sample figures, not census rates.
- The author's notes, summarised.

## Interpretation fixed in advance

- If the Wilson lower bound of P1 is at least 0.6, the paper may say that a blind check by the author confirmed most
  consensus continuations (with the share and CI) and keep the consensus definition.
- If the point estimate of P1 is below 0.6, the paper says that the consensus label overstates continuation by the
  measured amount and that the 5.1 continuation results rest on a label of that precision; the 5.1 estimates are
  not recomputed on 25 pairs.
- Between the two, the paper reports the share and CI without either conclusion.
- In every case the paper states that the check is by the author, on 40 pairs, and that no independent expert
  labels exist.
