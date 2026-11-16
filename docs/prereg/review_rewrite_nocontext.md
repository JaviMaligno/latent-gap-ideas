# Pre-registration (review round): the follow-up rewrite without the context in the prompt (Study 6b)

Status: binding from its commit, written 2026-11-15. Committed alone, after `review_prompt_sensitivity.md` (Study 6a)
and **before** the smoke calls and any generation. Script `scripts/review_rewrite_nocontext.py`; output
`docs/prereg/review_rewrite_nocontext.json`; raw rewrites `D/runs/review-rewrite-nocontext/{disc,rep}/rewrites.jsonl`
(`D = ~/Documents/experiment-data/latent-gap-ideas`). Paid (gpt-4o), under the **30 USD budget shared with Study 6a**;
the smoke, the cost estimate, the discovery-only rule and the hard stop are those registered in
`review_prompt_sensitivity.md` ("Budget, smoke and order of work"); the smoke includes two no-context rewrites of this
study (discovery follow-ups rwf/0/0 and rwf/0/1), kept out of the analysis.

## Review concern addressed

"gpt-4o rewrites each follow-up 'with the context shown'; 22 % drift from the paper's content… a rewrite without the
context in the prompt would be the cleaner genre control."

## What is known before this registration

Part B of `review_tangent_confounds.md` (`review_rewrite_null.json`), RW-F set, prompt R with the context shown:

| | discovery Qwen3 | discovery BGE-M3 | replication Qwen3 | replication BGE-M3 |
|---|---|---|---|---|
| gap against original follow-ups | 0.149 | 0.168 | 0.161 | 0.181 |
| gap against context-shown rewrites (gap_RW) | 0.112 (0.102 to 0.122) | 0.096 | 0.108 | 0.098 |
| content: median cos(rewrite, original) | 0.864 | 0.910 | 0.858 | 0.902 |
| content: share of rewrites below 0.80 | **22.1 %** | 11.0 % | 24.8 % | 13.6 % |

The "22 %" of the paper is the discovery Qwen3 share of RW-F rewrites with cos(rewrite, original follow-up) < 0.80
(`review_rewrite_null.content_check`). Discovery continuations: gap 0.081 against context-shown rewritten continuations.
No rewrite without the context has been generated; no number of this study has been computed.

## Question

When gpt-4o rewrites each real follow-up as a proposal **without seeing the context paper**, does the step gap between
the one-shot ideas and these rewrites stay positive? Is the context-shown rewrite's smaller gap (0.112 vs 0.149) a
genre effect (the proposal register itself) or an echo of the context it was shown?

## Items and prompt (fixed)

Items: every saved follow-up, the RW-F set of `review_rewrite_generate.py` with the same ids (`rwf/{ctx}/{j}`) and order:
discovery 2,883; replication 2,419 (**replication only if the budget rule allows it**). No NA or NB items (they require
the context or are not part of this question).

- **Prompt R (already run, context shown):** "Here is a paper:\n\n{context block}\n\nBelow is a later paper. Rewrite it
  as the proposal for the paper that should come next, building on the paper above: give its title and an arXiv-style
  abstract of 120-180 words. Keep exactly its content: the same subject, method, claims and details. Add nothing, remove
  nothing, invent nothing; only the wording and the framing as a proposal change.\n\n{later block}"
- **Prompt NC (this study):** "Below is a paper. Rewrite it as the proposal for the paper that should come next: give its
  title and an arXiv-style abstract of 120-180 words. Keep exactly its content: the same subject, method, claims and
  details. Add nothing, remove nothing, invent nothing; only the wording and the framing as a proposal change.\n\n{later
  block}"

NC = R with three deletions only: the context block and its lead-in, ", building on the paper above", and "later" (no
earlier paper is shown). Everything else identical to RW-F: system prompt `continue_from_context.SYSTEM`, gpt-4o-2024-08-06
through `gpt-4o-0806`, temperature 1.0, max_tokens 600, JSON mode, the ideas' parser, one retry on a parse failure
then dropped (counted), one sample per item, later block `paper_block` (abstract ≤ 1,200 characters). Ledger tags
`review-rewrite-noctx-disc`, `review-rewrite-noctx-rep`. Concurrency 6.

Content-filter fallback: as Study 6a. Each rewrite records its route (azure / openrouter / codex); **primary analysis on
Azure-generated rewrites only** (a filtered call = a failed generation, dropped); sensitivity with the fallback-filled
rewrites added.

Embedding: Qwen3-Embedding-0.6B (local ONNX `LocalQwen`, as the replication rewrites of Part B) and BGE-M3
(sentence-transformers on CPU, Part B settings). The discovery context-shown rewrites, ideas, contexts and follow-ups keep
their stored vectors (Kaggle, same encoders and settings). **Agreement check** (free, reported): the first 50 discovery
context-shown rewrites (sorted by id) embedded locally in both encoders and compared with their Kaggle vectors (mean and
minimum cosine).

## Measures

Ideas = the original arm of each sample (discovery: the 5 one-shot CTX ideas per context; replication: the 9 round-0
critique ideas, CTX prompt), loaded by `review_tangent_confounds.load_cell` exactly as `review_rewrite_null.py`.

- **gap_NC** per context = median step of the ideas − median step of the context's no-context rewrites (the estimator of
  `review_rewrite_null.p1`, `gap_RW`, with NC rewrites in place of R rewrites); mean over contexts.
- gap_RW recomputed on the same follow-ups (those with an Azure NC rewrite), and the paired difference gap_NC − gap_RW
  per context (> 0: showing the context pulled the rewrites towards it).
- Δ_NC = median step of NC rewrites − median step of the original follow-ups (effect of the genre alone).
- Ratios: mean gap_NC / mean original gap (0.149 in the primary cell) and mean gap_NC / mean gap_RW (0.112), bootstrap
  intervals of the ratio of means (joint resampling).
- Discovery continuations (47 contexts with consensus continuations): gap_NC,cont.
- **Content-drift check, as before:** cos(NC rewrite, original follow-up) per rewrite in each encoder; quartiles, mean,
  and the share below 0.80; compared with the context-shown figures above (22.1 % in the primary cell). Sensitivity:
  gap_NC excluding NC rewrites below 0.80 in Qwen3 (never in the primary).

## Estimators and tests

95 % percentile bootstrap over contexts, 10,000 resamples, `numpy.random.default_rng(20261115)`, one generator per sample ×
embedder, consumed in the order gap_NC, gap_RW (same follow-ups), original gap, gap_NC − gap_RW, Δ_NC, gap_NC,cont, then
the two ratios. Wilcoxon signed-rank: gaps one-sided (> 0), differences two-sided. **Primary test (single, no Holm):
discovery, Qwen3, gap_NC > 0.** Secondary family (discovery Qwen3, Holm over 2): gap_NC − gap_RW ≠ 0 and gap_NC,cont > 0.
BGE-M3 and the replication sample replicate.

## Decision rule (fixed)

Primary cell: discovery, Qwen3, Azure-only rewrites.

1. **The short step "survives the cleaner genre control"** if gap_NC is positive with its bootstrap CI excluding 0. The
   paper reports its size relative to 0.149 and 0.112 (ratios with CIs) and, with the thresholds of Part B: point ≥
   0.5 × 0.149 → "survives"; CI excluding 0 but point < 0.5 × 0.149 → "about X % of the short step is genre; the rest
   remains".
2. **Counts against the claim**: CI includes 0, or point < 0.03 → the short step is "not distinguishable from the
   proposal genre"; the claim is withdrawn as stated.
3. Echo reading (secondary): gap_NC − gap_RW > 0 with CI excluding 0 → part of the context-shown shrinkage was echo of
   the context, and the paper replaces 0.112 by gap_NC as the genre-controlled number; CI including 0 → the context in
   the prompt did not matter, and both numbers are reported; < 0 with CI excluding 0 → the context-shown rewrite was the
   more conservative control, 0.112 remains the headline genre-controlled number.
4. Content drift: the share below 0.80 is reported next to 22 %; if NC rewrites drift more than R rewrites, the
   sensitivity without them is reported in the same sentence.

Replication cells (BGE-M3; replication sample if run): same table; disagreements are stated, not resolved.

## What the paper will say

Section 5.1's genre-control sentence ("the gap survives in every cell: 0.112 … against the rewritten follow-ups")
gains the no-context number and its share of 0.149 under outcome 1, or is withdrawn under outcome 2; the appendix
reports the drift share of the NC rewrites next to the 22 %.

## Result (run 2026-11-15, `review_rewrite_nocontext.json`)

**Budget.** The shared estimate (`review_prompt_sensitivity.md`, Result) projected 30.55 USD for all four parts, so
only the discovery sample was run; no replication rewrites were generated.

**Generation.** 2,883 items, 2,885 calls (2 parse failures, both recovered on the retry), 0 dropped, 0 content-filter
fallbacks (all Azure; primary and fallback sensitivity coincide). Ledger tag `review-rewrite-noctx-disc`: 942,882 in /
675,874 out tokens = **9.1159 USD**, all on 2026-11-15 (UTC). Hard stop not reached. Agreement check (local vs Kaggle,
50 context-shown rewrites): mean cosine 0.999998 in both encoders (minimum 0.999997 Qwen3, 0.999987 BGE-M3).

Discovery sample, 150 contexts, all 2,883 follow-ups; means over contexts, 95 % bootstrap CI (10,000):

| | Qwen3 | BGE-M3 |
|---|---|---|
| gap vs original follow-ups | 0.149 (0.138 to 0.160) | 0.168 (0.161 to 0.176) |
| gap vs context-shown rewrites (gap_RW) | 0.112 (0.103 to 0.123) | 0.096 (0.089 to 0.104) |
| **gap vs no-context rewrites (gap_NC)** | **0.232** (0.221 to 0.242) | **0.167** (0.159 to 0.175) |
| gap_NC / 0.149-type original gap | 1.55 (1.50 to 1.61) | 0.99 (0.98 to 1.01) |
| gap_NC / gap_RW | 2.06 (1.92 to 2.22) | 1.73 (1.63 to 1.85) |
| gap_NC − gap_RW (paired) | +0.119 (0.110 to 0.129), Holm p = 4.6 × 10⁻²⁶ | +0.071 (0.064 to 0.078) |
| Δ_NC: NC rewrites − originals (step) | −0.082 (−0.087 to −0.078) | +0.001 (−0.001 to +0.004), p = 0.31 |
| continuations, gap_NC,cont (47 contexts) | 0.134 (0.105 to 0.164), Holm p = 2.3 × 10⁻¹¹ | 0.093 (0.074 to 0.111) |
| content: median cos(rewrite, original), NC / context-shown | 0.884 / 0.864 | 0.933 / 0.910 |
| **content: share below 0.80, NC / context-shown** | **6.9 % / 22.1 %** | 0.07 % / 11.0 % |

Primary test (gap_NC > 0, Qwen3): Wilcoxon p = 1.1 × 10⁻²⁶. Sensitivity excluding the 199 NC rewrites below 0.80
(Qwen3): gap_NC 0.228 (0.217 to 0.239) in Qwen3, 0.166 (0.158 to 0.174) in BGE-M3.

**Registered reading.**

1. **The short step survives the cleaner genre control** (outcome 1, "survives": point 0.232 ≥ 0.5 × 0.149, CI excluding
   0). Relative sizes: 1.55 × the gap against the originals (0.149) and 2.06 × the gap against context-shown rewrites
   (0.112) in Qwen3; 0.99 × and 1.73 × in BGE-M3.
2. **Echo reading: gap_NC − gap_RW > 0 with the CI excluding 0** in both encoders → part of the context-shown shrinkage
   (0.149 → 0.112) was echo of the context shown in the prompt, and by the registered rule the paper replaces 0.112 with
   gap_NC as the genre-controlled number. Without the context, rewriting into the proposal genre does not move the
   follow-ups towards the context: in BGE-M3 it leaves their step unchanged (Δ_NC +0.001, n.s.), and in Qwen3 it moves
   them *away* from it (Δ_NC −0.082), so the Qwen3 gap_NC (0.232) exceeds the original 0.149. The encoders disagree on
   that last point; as registered, both are stated: the genre-controlled gap is "at least the original gap" (0.167 vs
   0.168 in BGE-M3; 0.232 vs 0.149 in Qwen3).
3. Content drift: the no-context rewrites keep the follow-ups' content better than the context-shown ones (6.9 % below
   0.80 in Qwen3 against 22.1 %), and excluding them changes gap_NC by 0.004.

Deviations: none in prompt or analysis. Environment only: BGE-M3 with the pinned `sentence-transformers==3.0.1` /
`transformers==4.44.2` / `numpy<2` (latest `transformers` fails to import). The two smoke calls of this study are in
the tag `review-promptsens-smoke` (5 calls in all, 0.0147 USD, not split by study in the ledger).
