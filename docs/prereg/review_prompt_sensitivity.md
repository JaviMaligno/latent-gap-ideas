# Pre-registration (review round): prompt sensitivity of the one-shot step (Study 6a)

Status: binding from its commit, written 2026-11-15. Committed alone, **before** the smoke calls and before any
generation of this study. Script `scripts/review_prompt_sensitivity.py`; output `docs/prereg/review_prompt_sensitivity.json`;
raw ideas `D/runs/review-prompt-sensitivity/{disc,rep}/ideas.jsonl` (`D = ~/Documents/experiment-data/latent-gap-ideas`).
Paid (gpt-4o, paid spend) under a hard budget of **30 USD shared with Study 6b** (`review_rewrite_nocontext.md`),
approved by Javier.

## Review concern addressed

"Single prompt, temperature 1. No prompt-sensitivity arm for the one-shot step ('a paper that will cite this one in
three years' vs 'the paper that should come next'). The gap may be partly an instruction effect."

## What is known before this registration

The original one-shot CTX arm (`continue_from_context.md`, Part B): mean step gap (ideas − real follow-ups, median per
context) **0.149** (CI 0.138 to 0.160) on the 150 discovery contexts, Qwen3; recomputed through the Part A engine of
`review_tangent_confounds.py` (the loader used here) 0.1494 in Qwen3 and 0.168 in BGE-M3. On the 150 replication contexts
(`strengthen_critique_replication`), the round-0 ideas of the three critique arms, which use the same CTX prompt
(3 arms × 3 ideas = 9 per context), give 0.161 (Qwen3) and 0.181 (BGE-M3) (`review_rewrite_null.json`, `original_gap`).
No idea has been generated with any other wording; no number of this study has been computed.

## Question

Does the short step of gpt-4o's one-shot continuation survive changes of the instruction wording that drop "the paper
that should come next"? How much of the original gap does each wording keep?

## Sample

- **Primary: the 150 discovery contexts** (`continue_from_context.contexts()`; follow-ups `D/runs/followups/followups.jsonl`,
  2,883 saved follow-ups). Original arm: the 750 CTX ideas (5 per context), vectors as loaded by
  `review_tangent_confounds.load_cell("DISC-4o", space)` (`cell.pb`).
- **Secondary: the 150 replication contexts** (`strengthen_critique_replication.ctx_rows()`, 2,419 follow-ups). Original
  arm: the 1,350 round-0 ideas of GEO, LLM and ORACLE (9 per context, CTX prompt), loaded by `load_cell("REP-4o", space)`.
  **Run only if the budget rule below allows it.**

## Prompts (verbatim; everything else identical to the original arm)

System prompt, unchanged (`continue_from_context.SYSTEM`): `You are a researcher proposing the next paper. Answer with JSON
{"title": ..., "abstract": ...}.` Context block `continue_from_context.paper_block` (title + abstract ≤ 1,200 characters),
the same blocks as the original arm. `{ctx}` below is that block.

- **Original (not regenerated):** "Here is a paper:\n\n{ctx}\n\nPropose the paper that should come next, building on
  this one: an original follow-up a research group could write. Give its title and an arXiv-style abstract of 120-180
  words."
- **V1 (citation horizon, the reviewer's wording):** "Here is a paper:\n\n{ctx}\n\nPropose a paper that will cite this
  one, written about three years from now: an original paper a research group could write. Give its title and an
  arXiv-style abstract of 120-180 words."
- **V2 (neutral):** "Here is a paper:\n\n{ctx}\n\nPropose a research idea that builds on this paper: an original idea a
  research group could pursue. Give its title and an arXiv-style abstract of 120-180 words."

Generation: gpt-4o-2024-08-06 through `gpt-4o-0806`, temperature 1.0, max_tokens 600, JSON mode, parsed with the
ideas' parser (JSON with non-empty title and abstract), up to 3 attempts per idea as `continue_from_context.generate`,
then dropped (counted). Ideas per context = the original arm's: 5 (discovery), 9 (replication), independent draws, no
memory of earlier proposals. Ledger tags `review-promptsens-disc`, `review-promptsens-rep`.

Limit fixed in advance: the system prompt still says "proposing the next paper". Changing it would change two things at
once relative to the published arm; the variants test the user instruction, which is what the reviewer quotes.

Content-filter fallback: the gateway retries a call that Azure's content filter rejects through OpenRouter (same model,
`openai/gpt-4o-2024-08-06`) and, if that fails too, through the Codex CLI (a different model). Each idea records its
route (`via`: azure / openrouter / codex). **Primary analysis: Azure-generated ideas only**; a filtered call counts as a
failed generation and is dropped. Sensitivity: the same analysis with the fallback-filled ideas added.

Embedding: Qwen3-Embedding-0.6B (local ONNX, `latent_gap.embedder.LocalQwen`, the encoder of the original arm's
`idea_emb.npy`) and BGE-M3 (sentence-transformers on CPU, max_seq_length 512, normalised: the settings of
`revision_neutral.embed_bge`), text `latent_gap.arxiv.item_text(title, abstract)`. The original arm's and the contexts'
and follow-ups' vectors are the stored ones (for the discovery BGE-M3 and Qwen3 vectors, the Kaggle sentence-transformers
run with the same settings; the local-vs-Kaggle agreement is measured on 50 texts in Study 6b and reported).

## Measures

For context i (unit vectors, one embedder): step of an item x = cos(x, c_i); follow-up step f_i = median over the saved
follow-ups; gap_A,i = median step of arm A's ideas − f_i (exactly `continue_from_context.part_b` / `review_rewrite_null.p1`).
For each variant V: mean gap_V over contexts; paired difference d_V,i = gap_V,i − gap_orig,i (= the difference of the
median idea steps, since f_i cancels); share kept = mean gap_V / mean gap_orig.

Contexts enter when both the original arm and every variant have at least one idea (expected: all 150).

## Estimators and tests

- 95 % percentile bootstrap intervals over contexts, **10,000 resamples, `numpy.random.default_rng(20261115)`**, one
  generator per sample × embedder, consumed in the order: original gap, then for V1 and V2 in turn: gap, difference,
  share (ratio of means, numerator and denominator resampled jointly).
- Wilcoxon signed-rank: gap_V > 0 one-sided; d_V two-sided.
- **Confirmatory family G (discovery, Qwen3): Holm over 2** (gap_V1 > 0, gap_V2 > 0), alpha 0.05.
- **Family D (discovery, Qwen3): Holm over 2** (d_V1 ≠ 0, d_V2 ≠ 0), alpha 0.05.
- BGE-M3 and the replication sample: the same families, each Holm over 2 within sample × embedder, as replication.

## Decision rule (fixed)

Primary cell: discovery, Qwen3, Azure-only ideas.

1. **Robust to wording**: every variant's step gap is positive with its bootstrap CI excluding 0. Then:
   - each variant keeps ≥ 0.5 of the original gap (point) → the paper says "the short step is robust to wording: with
     two other instructions the gap is X and Y (original 0.149), i.e. A % and B % of it";
   - some variant keeps < 0.5 → "the short step survives every wording, but its size depends on the instruction:
     [variant] keeps only A % of it". The headline number stays tied to the original instruction and the dependence is
     stated next to it.
2. **Counts against the claim**: some variant's CI includes 0, or the gap is negative → the paper states that the
   short step is partly an instruction effect: under [wording] no gap is detected; the claim "the ideas stay closer to
   the context than the citing papers" is scoped to the instruction "the paper that should come next".
3. Family D is reported in either case: whether a wording changes the gap at all (Holm-adjusted p and CI of d_V),
   independently of whether the gap survives.

Replication (BGE-M3 always; replication sample if run): reported in the same table; where a replication cell
disagrees with the primary reading, the paper states both readings and does not pick one.

## Budget, smoke and order of work (fixed)

1. Commit this file alone; then commit `review_rewrite_nocontext.md` alone.
2. **Smoke, 5 calls in all, tag `review-promptsens-smoke`, kept out of every analysis:** V1 and V2 on discovery
   context 0, V1 on replication context 0, and the no-context rewrite (Study 6b) of discovery follow-ups rwf/0/0 and
   rwf/0/1. One attempt each. Output `D/runs/review-prompt-sensitivity/smoke.jsonl`.
3. **Estimate** (`estimate`): input tokens of every planned call = characters (system + user prompt) × the smoke's pooled
   tokens per character; output tokens = max(smoke mean, 230) per call; × 1.03 for parse retries; gpt-4o at 2.50 USD/M
   input and 10 USD/M output. Parts: 6a-disc (1,500 calls), 6a-rep (2,700), 6b-disc (2,883), 6b-rep (2,419).
   **If smoke + all four parts ≤ 30 USD, run all four; otherwise run the discovery sample only (6a-disc and 6b-disc)
   and say so.** (A rough pre-smoke estimate at 4.3 characters per token and 225 output tokens gives 4.7 + 8.4 + 9.3 +
   7.8 ≈ 30.1 USD, so the discovery-only outcome is the likely one.)
4. **Hard stop** in the script: before every call the spend of the five tags (`review-promptsens-{smoke,disc,rep}`,
   `review-rewrite-noctx-{disc,rep}`), computed from the ledger rows (tokens × list price), is checked; at ≥ 29.5 USD no
   new call is made (in-flight calls stay below 0.5 USD). A part cut by the stop is analysed on the contexts it
   completed and reported as such.
5. Generate 6a (concurrency 6), embed, analyse; then 6b. Spend is reported per study and per day (UTC) from the ledger
   (`review_prompt_sensitivity.py spend`).

Compute: embedding ≈ 1,500-4,200 texts per encoder on CPU; analysis minutes.

## What the paper will say

Under outcome 1 the instruction-effect concern is answered with the two numbers and shares; under outcome 2 the
short-step claim is scoped to the original instruction in the abstract, Section 5.1 and the summary table. Either way
the variants' gaps go into the appendix table of the CTX arms.

## Result (run 2026-11-15, `review_prompt_sensitivity.json`)

**Budget decision (step 3).** Smoke: 5 calls, all parsed, all Azure, 1,432 in / 1,117 out tokens, 0.0147 USD;
0.2088 tokens per character, mean output 223 tokens (floor 230 applied). Projection (`runs/review-prompt-sensitivity/estimate.json`):
6a-disc 4.75, 6a-rep 8.53, 6b-disc 9.39, 6b-rep 7.86 USD; total 30.55 USD > 30 → **discovery sample only**, as registered.
The replication sample was not generated for either study.

**Generation.** 1,500 calls (V1 750, V2 750), 0 parse failures, 0 retries, 0 dropped, 0 content-filter fallbacks (all
Azure), so the primary analysis and the fallback sensitivity coincide. Ledger tag `review-promptsens-disc`: 448,190 in /
336,534 out tokens = **4.4858 USD**, all on 2026-11-15 (UTC). Hard stop not reached.

Discovery sample, 150 contexts, 5 ideas per context and arm; step gap = median idea step − median follow-up step, mean
over contexts, 95 % bootstrap CI (10,000):

| | Qwen3 | BGE-M3 |
|---|---|---|
| original (the paper's arm) | 0.149 (0.139 to 0.160) | 0.168 (0.160 to 0.176) |
| **V1** "a paper that will cite this one, written about three years from now" | **0.135** (0.123 to 0.146) | 0.164 (0.156 to 0.171) |
| **V2** "a research idea that builds on this paper" | **0.130** (0.118 to 0.142) | 0.159 (0.151 to 0.167) |
| share of original gap kept, V1 | **90 %** (87 to 94) | 97 % (95 to 100) |
| share kept, V2 | **87 %** (83 to 91) | 95 % (92 to 97) |
| d_V1 = V1 − original (paired) | −0.015 (−0.020 to −0.010), Holm p = 9.5 × 10⁻⁷ | −0.005 (−0.009 to −0.0002), Holm p = 0.020 |
| d_V2 = V2 − original (paired) | −0.020 (−0.026 to −0.014), Holm p = 1.5 × 10⁻⁷ | −0.009 (−0.013 to −0.005), Holm p = 1.9 × 10⁻⁵ |

Family G (gap > 0), Holm-adjusted Wilcoxon p: Qwen3 V1 3.6 × 10⁻²⁶, V2 5.9 × 10⁻²⁶; BGE-M3 both 2.3 × 10⁻²⁶. Mean idea step
(Qwen3): original 0.768, V1 0.753, V2 0.748; mean follow-up step 0.618.

**Registered reading: outcome 1, robust to wording, with every variant keeping ≥ 0.5 of the gap.** In the primary cell
both variants keep a positive gap with the CI far from 0 (0.135 and 0.130, i.e. 90 % and 87 % of 0.149). Family D: both
wordings shorten the gap slightly but detectably (by 0.015 and 0.020 in Qwen3; 0.005 and 0.009 in BGE-M3), so the
instruction accounts for about a tenth of the gap (V2 in Qwen3: 13 % at the point, 17 % at the lower
share bound). BGE-M3 replicates the reading. Paper sentence licensed: "the short step is robust to wording: with two other
instructions the gap is 0.135 and 0.130 (original 0.149), i.e. 90 % and 87 % of it."

Deviations: none in prompts or analysis. Environment only: BGE-M3 embedding ran with `sentence-transformers==3.0.1`,
`transformers==4.44.2`, `numpy<2` (the pins of `strengthen_modern_neutral.py`), because the latest `transformers` fails
to import; local-vs-Kaggle agreement on 50 texts: mean cosine 0.999998 (Qwen3) and 0.999998 (BGE-M3), minimum 0.99999.
The registered system prompt limit applies ("proposing the next paper" unchanged).
