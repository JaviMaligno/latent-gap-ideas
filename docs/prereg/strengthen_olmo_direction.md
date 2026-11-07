# Does critique turn OLMo towards the continuations beyond plain revision? (pre-registered)

Pre-registered 2026-11-05, before any fetch, label or generation of this study. Scripts:
`scripts/olmo_direction_fetch.py` (stage 1), `scripts/olmo_direction_label.py` (stage 2),
`scripts/olmo_direction_kaggle.py` + `kaggle/olmo-direction/run.py` (stage 3), `scripts/olmo_direction_analysis.py`
(analysis). Raw data: `~/Documents/experiment-data/latent-gap-ideas/runs/olmo-direction/`. Output:
`docs/prereg/strengthen_olmo_direction.json`.

## Why

Section 5.3 measures direction with the step-invariant tangent construction. For OLMo-7B-0424-Instruct the
follow-up-specific changes after critique are +0.012 to +0.032 and the reviewer's survives Holm, but OLMo has no
NEUTRAL arm, so none of this is tested against plain revision; for gpt-4o the NEUTRAL arm showed that revision with no
information produces the turn and no critique arm adds to it detectably (N5, N6). The continuation census
(`strengthen_continuations_full.md`, F4) rebuilt the direction from consensus continuations instead of all citing
papers, but only 17 contexts had >= 2 continuations: the gpt-4o contrasts (critique arm − NEUTRAL along s) are not
significant and every CI crosses ±0.015 (insufficient), and OLMo had no NEUTRAL arm at all. This study gives OLMo the
missing arm, on new contexts, with enough contexts to read each contrast against ±0.015. It uses no paid API:
OLMo runs on Kaggle (free GPU), labels come from two subscription CLIs, follow-ups from Semantic Scholar and arXiv.

What is known to me before this pre-registration: all results of the 150-context studies and of the census; for the
196 contexts of `runs/followups/followups.jsonl` (rows 416-999 with >= 5 follow-ups) their follow-up lists as fetched
on 2026-10-30, never labelled, never given to a generator, never analysed. Nothing else about the new contexts.

## Stage 1: contexts and follow-ups (`olmo_direction_fetch.py`)

- **Frame (fixed order, written once to `frame.json`).** First the 196 unused contexts above, in file order. Then
  the same pool as `followups_fetch.py` (arXiv cs.* corpus, not adult, past split, dated 2021-01-01 to 2022-12-31;
  23,124 items) minus every context id in any `runs/*/followups.jsonl` (the 1,000 + 680 + 15 already drawn, so the
  388 contexts of rows 416-999 already screened out are not redrawn), in the order of
  `numpy.random.default_rng(20261106).permutation`.
- **Per context, one Semantic Scholar call** (paginated): `/paper/arXiv:{id}/citations` with fields
  `externalIds,publicationDate,intents,isInfluential`. Every citing paper with an arXiv id, of any date, is saved
  uncapped (`citing_all.jsonl`, used to exclude citing papers from the distractor pool). Candidates: citing arXiv
  papers with S2 publication date >= 2024-01-01. Their arXiv metadata (title, abstract, first-version date
  `published`) come from the arXiv API. A candidate is a **follow-up** if its arXiv v1 date is >= 2024-01-01 and it
  has an abstract (the stated design rule, now applied to v1; it also clears OLMo's cutoff, Feb-Mar 2023). If more
  than 50, a seeded sample of 50 is saved (`default_rng(20261106 + frame position)`), as before.
- **Unit of the study:** a context with >= 5 saved follow-ups; these are numbered in frame order (`pos`).
- Backoff on HTTP 429 / 5xx (Retry-After or exponential to 300 s, without giving up); resumable; progress log.
- **Size.** Fetching continues in frame order until the contexts needed by the current wave are in hand
  (planning yield: 34.6 % of drawn contexts have >= 5 follow-ups, 19.2 saved follow-ups each, as in the first 1,000).
  Wave 1 target: 1,200 contexts with >= 5 follow-ups.

## Stage 2: continuation labels (`olmo_direction_label.py`)

- **Labellers, prompt, labels, parsing:** identical to `strengthen_continuations_full.md`, through the same code
  (`prompt_for`, `run_cli`, `parse` of `strengthen_continuations.py`; `diagnose`, `label_batch`,
  `single_pair_fallback` of `strengthen_continuations_full.py`): gpt-6.1-sol via `codex exec` and Claude Opus 5.5 via
  `claude -p`, blind to each other; CONTINUES / USES / BACKGROUND / UNCLEAR; batches of 20; up to 3 attempts per
  pass, a resume pass of up to 3 more, then the most complete attempt's valid rows plus single-pair calls (up to 3);
  every attempt logged (`attempts_{labeller}.jsonl` per stage directory) and counted in the output.
- **Labelling rule: the S2-flag hybrid.** A follow-up is *flagged* if S2 marks the citation influential or its
  intents include methodology or result. Contexts are labelled in frame-order blocks of 200 (the last block of a
  wave may be shorter).
  - *Stage A of a block:* every flagged pair of the block, plus every pair of the block's **audit contexts** (a seeded
    10 %, 20 per full block, `default_rng(20261106 + 100 + block)`).
  - *Stage B of a block* (after both labellers finish stage A): every remaining pair of each **triggered** context,
    i.e. one with at least one flagged follow-up that both labellers call CONTINUES.
  - Batches: the stage's pairs shuffled with `default_rng(20261106 + 10·block + {A: 1, B: 2})`, both labellers on the
    same batches, the two labellers in parallel, each sequential.
- **Why the hybrid and not the census.** On the 150 census contexts the hybrid would have labelled 35 % of the pairs
  (1,010 of 2,883) and found 14 of the 17 contexts with >= 2 consensus continuations; the 3 lost (18 %) all have
  exactly 2. For 250 eligible contexts that is about 21,600 pairs (incl. audit) instead of about 42,400, i.e. about
  9 h instead of about 18 h of each labeller's CLI quota, at the census's 30 s per batch. The selection is on an S2
  covariate fixed before any OLMo output and the estimand is a within-context contrast between arms randomised
  within the same context, so it shifts the population (towards contexts with a flagged continuation) but cannot
  create an arm difference. The population is therefore stated as such, and the audit measures the loss on the new
  contexts.
- **Fallback, named now: the full census** (every pair of every context, stage B dropped, all earlier blocks
  completed to a census, eligibility = >= 2 consensus continuations), used if (a) intents/influence are missing
  (citing paper absent or both fields null) for > 20 % of the saved follow-ups of block 1, checked before any label,
  or (b) after wave 1 the pooled audit contexts contain >= 4 missed eligible contexts and the estimated loss
  (missed / (missed + found) among audit contexts) exceeds 40 %.
- **Constructs:** consensus continuation = both say CONTINUES. **Eligible context** = triggered and >= 2 consensus
  continuations (all its pairs are then labelled). Audit contexts that are not triggered but have >= 2 consensus
  continuations are **missed**: they count for the loss estimate and a sensitivity analysis, not for the primary set.

## Stage 3: OLMo generation (Kaggle, `kaggle/olmo-direction/run.py`)

- **Contexts:** the eligible contexts of the wave, in `pos` order (plus the wave's missed audit contexts, flagged,
  for the sensitivity analysis only).
- **Generator and protocol as Section 5.2 / the OLMo kernel of Section 5.3:** `allenai/OLMo-7B-0424-Instruct-hf`,
  float16 on T4 x2, plain transformers sampling, temperature 1, top-p 1, max 450 new tokens, 2,048-token window,
  system text prepended to the user turn, the same prose/JSON parser with up to 3 attempts per item; CTX prompt,
  revise prompt, GEO feedback (band 0.574-0.661), reviewer prompt (the reviewer is OLMo itself, never sees a
  follow-up) and ORACLE feedback (max cosine to the context's saved follow-ups, Qwen3) copied verbatim from
  `continue_from_context.py` / `iterative_critique.py`; NEUTRAL feedback "Revise your proposal: write an improved
  version." (`neutral_revision.md`).
- **Arms:** GEO, Reviewer (LLM), ORACLE, NEUTRAL; **3 independent ideas per context and arm** (OLMo had 1 before);
  **3 rounds** (initial attempt + 2 revisions), as Section 5.2. As in `neutral_revision.md`, NEUTRAL's round-0 idea k
  is GEO's round-0 idea k (`r0_from: GEO`); NEUTRAL vs GEO therefore compares round 2 from the same starting ideas,
  NEUTRAL vs Reviewer / ORACLE compares changes from independent round-0 draws.
- Embeddings in the kernel with the corpus settings: Qwen3-Embedding-0.6B and BGE-M3 (float16, max 512 tokens) of
  contexts, saved follow-ups and every idea. Seed `20261106 + wave`. A wave may be split over several kernel runs of
  at most 140 contexts (operational only).

## Construction (F4 of the census, unchanged except the pool)

For context c (Kaggle vectors, unit norm): tangent of x = unit(x − (x·c)c). **f** = unit mean tangent of its
consensus continuations. **p** = unit mean tangent of the k nearest (cosine to c) future-split corpus papers that
are not citing papers of c (uncapped list, any date) and not c, with k = number of consensus continuations
(primary). **s** = unit(f − (f·p)p). For arm A and round r, v_A,r = mean over the context's ideas with an abstract of
tangent·s. Δ_A = v_A,2 − v_A,0. Contrast d_A = Δ_A − Δ_NEUTRAL. A context without a valid round-0 and round-2 idea in
an arm is dropped from the contrasts with that arm (counts reported).

## Estimands and tests

- **Confirmatory (primary cell OLMo / Qwen3, k = continuations):** the three contrasts d_GEO, d_Reviewer, d_ORACLE.
  Mean over eligible contexts, 95 % percentile bootstrap CI (2,000 resamples over contexts, seed 20261106),
  two-sided Wilcoxon signed-rank, Holm over the 3, alpha 0.05. No direction is predicted: Section 5.3's OLMo turns
  (+0.012 to +0.032) could be feedback or revision; the gpt-4o result (revision explains the turn) suggests the
  latter. Each CI is also read against **±0.015**, fixed now.
- **Reading rule per contrast:** Holm-significant → a feedback-specific turn is detected (its sign and size
  reported; if the whole CI is inside ±0.015 it is called detected but below 0.015). Not significant and CI inside
  (−0.015, +0.015) → no feedback-specific turn of 0.015 or more. Not significant and CI crossing ±0.015 →
  insufficient.
- **Secondary (descriptive, reported, not confirmatory):** the same contrasts in OLMo / BGE-M3 (own Holm); each arm's
  Δ along s, NEUTRAL's included (one-sample Wilcoxon); alignment with f and the change along p; p with k = number of
  saved follow-ups; the all-follow-up construction of Table 3 (f from all saved follow-ups, k = their number, pool
  excluding all citing papers) on the same contexts; positive control (continuations' leave-one-out alignment) and
  mean f·p; step lengthening per arm and vs NEUTRAL; sensitivity with the missed audit contexts added; the per-wave
  estimates; label counts, κ (CONTINUES vs not) on labelled pairs, attempts and fallbacks; the audit loss estimate;
  the share of eligible contexts among contexts with >= 5 follow-ups.

## Stopping rule and waves

- **Wave 1 = the first 100 eligible contexts in `pos` order.** Labelling proceeds block by block until the
  completed blocks contain >= 100 eligible contexts; eligible contexts beyond the 100th carry into wave 2.
- After wave 1 the SD of each contrast is re-estimated; planned total n* = ceil((1.96 · max SD / 0.015)²), and the
  wave 2 adds min(250, max(n*, 150)) − 100 eligible contexts (at least 50); if the stop condition is not met after
  it, further waves of 50 follow, up to 250. The stopping check is repeated after every wave.
- **Stop** as soon as, after a completed wave, all three primary CIs have half-width ((upper − lower)/2) <= 0.015, or
  when 250 eligible contexts have been generated. The check uses only CI widths, never the estimates or p-values;
  interim estimates are computed for the width but not reported or read until the stop. The tests are read once,
  on the final set.
- If the frame, the CLI quota or the Kaggle quota runs out before either condition, the study stops at the last
  complete wave and says so; the reading rule applies unchanged (most likely "insufficient").

## Power

Planning SD of a contrast: OLMo's along-s changes with 1 idea had SD 0.09-0.13 per arm (census F4, 17 contexts);
gpt-4o's contrasts with NEUTRAL had SD 0.05-0.07, about 0.9-1.3 times its per-arm SD. With 3 ideas per arm, the
planning range for an OLMo contrast is SD 0.08-0.12. 95 % CI half-width 1.96·SD/√n reaches 0.015 at n ≈ 110 (SD
0.08), 171 (0.10), 246 (0.12), hence the 100-250 range. At n = 250 and SD 0.10, the smallest contrast detected with
80 % power is about 0.018 (two-sided alpha 0.05) or 0.020 (alpha 0.05/3, Holm's first step).

## Interpretation (what the paper may say)

- About **OLMo** only, on the eligible population (2021-2022 arXiv cs.* papers with >= 5 arXiv follow-ups since
  2024, a flagged consensus continuation and >= 2 consensus continuations), and on the continuation direction (not
  the all-follow-up direction of Table 3, reported alongside): per arm, one of "a feedback-specific turn of X beyond
  content-free revision", "no feedback-specific turn of 0.015 or more", or "insufficient". The paper's "OLMo has no
  NEUTRAL arm, so its follow-up-specific changes are untested against plain revision" is replaced by that result,
  and the "(OLMo's moves more)" clause of the abstract is kept, qualified or withdrawn accordingly.
- **gpt-4o stays unresolved.** No gpt-4o generation is made (no paid API). Its F4 contrasts (17 contexts, every CI
  crossing ±0.015) stay as reported, and nothing here may be read as evidence about gpt-4o or transferred to it.
- Limitations stated with the result: one 7B generator; contexts differ from the 150 (a new sample, not a
  replication on the same papers); NEUTRAL shares round 0 with GEO; hybrid population; LLM labels.

## Cost

No paid API of any kind. Labelling: about 9,000 pairs (450 batches per labeller, about 4 h) for wave 1, about
21,600 (about 9 h) for 250 eligible contexts. Kaggle: 33 generations + 6 reviews per context, about 220 s per
context on T4 x2 (from the earlier OLMo run): about 6 h for wave 1, about 15 h for 250 contexts, within the weekly
free GPU quota. Fetch: one S2 call and about one arXiv call per drawn context.
