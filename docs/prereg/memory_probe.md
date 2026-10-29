# Pre-registration: memory probe (MEM) and anticipation by time window

Frozen 2026-10-25, before any MEM item is generated.

**Question.** Does a generator's anticipation of later items (scripts/anticipation.py) come from memory of those items
(training data after the corpus split) or from the generator itself? Two within-model contrasts, no second model needed.

**MEM arm.** 20 real items per half-year window, 2022H1-2026H2 (arXiv 10 windows, Steam 9; seed 20261024), past or
future split alike. The decoder sees the 8 nearest *past* items (near-duplicates of the target, cos > 0.95, hidden, as
in REAL) and writes one round of best-of-3 with the REAL prompt. Outcome: landing = cos(best decoded, target).

**Prediction (gpt-4o 2024-08-06, published training cutoff October 2023).** Landing on windows it saw (2022H1-2023H2)
exceeds landing on windows it did not see (2024H1 onwards) by more than the drift between adjacent unseen windows:
step(2023H2 -> 2024H1) > max |difference between consecutive unseen windows|. No step = memory does not explain it.

**Control for topic drift.** The same windows scored with anticipation-by-window on real held-out past items: a smooth
decay with distance in time is drift; only a step at the generator's own cutoff counts as memory.

**Other generators** (gpt-6-luna, claude-sonnet-5-5 via subscription): their cutoffs are to be taken from published
model documentation before reading their curves; the location of their step is reported, not predicted.

## Result (2026-10-25, scripts/memory_curve.py, scripts/anticipation.py --future-from 2024-01-01)

**Memory probe.** excess = landing minus the per-generator anchoring fit (s.e.m. per window ~0.010-0.015).
- arXiv, gpt-4o: step 2023H2 -> 2024H1 = 0.022 - (-0.008) = 0.030; largest change between consecutive unseen windows
  0.057 (2025H2 -> 2026H1). Prediction **fails**: no step beyond window-to-window noise.
- Steam, gpt-4o: step 0.045 - (-0.013) = 0.058 > largest unseen change 0.027: formally **passes**, but 2023H2 is high
  for all three generators (luna 0.033, sonnet 0.019), which have also seen 2024: a property of that window's items,
  not of gpt-4o's cutoff. Difference-in-differences gpt-4o minus luna, mean of pre-2024 windows vs 2024H1-2025H2:
  arXiv 0.013 -> 0.004, Steam 0.002 -> 0.003, both within one s.e.
- Absolute landing ranks sonnet > luna > gpt-4o in every window and is flat in time (it drops in 2026 with the
  nearest visible neighbour, nn1). **No generator shows item-level memory of what it was trained on.**

**Anticipation with the pre-cutoff generator.** gpt-4o, which cannot have seen 2024+, anticipates *more* than the
current generators: arXiv PLAIN 0.86, PLAIND 0.90, FAR 0.59, GAP 0.50 (luna 0.32/0.39/0.36/0.28; past baseline 0.24);
Steam PLAIND 0.74, GAP 0.69. So anticipation is not memory. But gpt-4o's REAL reconstructions of *past* items also
lean to the future (arXiv 0.34 vs 0.10-0.12 for the new models), which points to a style confound: post-2023 abstracts
and game pages are increasingly written with LLM help, so LLM-written text of any content lands nearer to them. Open
check, not run: paraphrase real past items with each generator (content fixed, style changed) and measure how far
style alone moves the hit rate.

## Style check (2026-10-25, scripts/style_probe.py; exploratory)

200 real past items per domain, paraphrased by each generator (same content, its wording; schema of the generated
ideas). Anticipation hit rate (future from 2024-01-01) of the paraphrases vs the originals; content check =
cos(paraphrase, original).

| | arXiv hit | content cos | Steam hit | content cos |
|---|---|---|---|---|
| originals | 0.224 | 1 | 0.359 | 1 |
| gpt-4o 2024-08-06 | **0.363** | 0.915 | **0.469** | 0.906 |
| gpt-6-luna (gateway) | 0.237 | 0.980 | 0.389 | 0.961 |
| claude-sonnet-5-5 | 0.217 | 0.984 | 0.368 | 0.961 |

Binomial s.e. ~0.03. Only gpt-4o's rewording moves real past items towards the future (+0.14 arXiv, +0.11 Steam),
and almost all of it comes from its loosest third of paraphrases (arXiv 0.22 -> 0.46, Steam 0.36 -> 0.57; the most
faithful third +0.06 and 0.00). Reading those by hand, the content is the same and the change is gpt-4o's 2023-24
register ("comprehensive", "utilizing", "significantly impacts", added padding sentences): the register of many
post-2023 abstracts and store pages. So there is a style confound, specific to gpt-4o, not to LLM text in general.

Against each generator's own paraphrase baseline, PLAIND still anticipates: arXiv gpt-4o 0.90 vs 0.36, luna 0.39 vs
0.24, sonnet 0.49 vs 0.22. Caveat: a generated idea is written entirely in the generator's register, a paraphrase
keeps the original's structure, so the paraphrase baseline is a lower bound on the style effect. A style-free
version: rewrite each idea and its nearest past and future candidates with one common normaliser, re-embed, decide.

## Style-free anticipation (2026-10-25, scripts/anticipation_normalised.py; exploratory)

Every text in the comparison (idea, its 10 nearest past and 10 nearest future candidates, 50 held-out real items per
side) rewritten by one normaliser (gpt-6-luna, the judges' neutral-pitch prompt; arXiv without results), re-embedded.
Hit rate, n = 50 per cell (s.e. ~0.07); p = one-sided Fisher test against REF_PAST. Texts lost to the content filter:
arXiv 2, Steam 27.

| arXiv | normalised | raw | p vs past |
|---|---|---|---|
| REF_PAST (real) | 0.28 | 0.12 | |
| REF_FUTURE (real) | 0.62 | 0.68 | |
| PLAIND sonnet | **0.58** | 0.50 | 0.002 |
| PLAIND luna | **0.50** | 0.40 | 0.020 |
| PLAIND gpt-4o | 0.38 | 0.94 | 0.198 |
| FAR sonnet / luna / gpt-4o | 0.42 / 0.38 / 0.36 | 0.18 / 0.44 / 0.54 | 0.10 / 0.20 / 0.26 |

- arXiv keeps a clear past/future separation after normalising (0.28 vs 0.62): the measure works on content.
- gpt-4o's 0.94 was almost all register: style-free it is 0.38, not distinguishable from the past baseline.
- The generators trained after the split (sonnet, luna) still anticipate with PLAIND, near the real-future ceiling.
  Together with MEM (no item-level memory) the reading is topic-level knowledge of what came later, not foresight:
  the one generator that cannot know the future does not anticipate it once style is removed. n = 50: a direction.
- FAR (the tweet's recipe) is not distinguishable from the baseline for any generator.
- Steam loses its separation after normalising (REF_PAST 0.44 vs REF_FUTURE 0.53), so Steam cannot answer this; no
  Steam cell is clearly above the baseline (best p = 0.04, twelve tests).

Cost: the normalisation took about 7.8 USD of gpt-6-luna (estimated 4).
