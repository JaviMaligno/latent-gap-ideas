# Inspiration from anything: does the distance or the content of incidental material shape a new game idea?

Drafted 2026-11-03; **frozen at commit 7ea8ae8**, after the pool was embedded and the stimuli drawn, and before any
generation call except a 6-call smoke test of the format (2026-11-04, kept in the run). The analysis script
(`scripts/inspiration_analysis.py`) was committed while generation was running, before any idea was scored; its
implementation choices (Robin Hood step of the two-lines test, Holm family per framing including P2 through its
one-sided bootstrap p) are part of the plan. Nothing here was chosen after looking at any generated idea or at how pool
items relate to future games; only pool counts and text lengths were inspected.

## Why

The paper's thesis is that distances in an embedding map do not guide ideation: what forecasts where the future
lands is momentum, and what moves a model is content (anchoring 12/12, `anchoring_causal.md`). Every test so far
showed the model material of the same kind as the target (games for games, papers for papers). A person with ideas
also has around them material they never think of as relevant: a recipe, the rules of a sport, the news, a random
encyclopaedia page. This test shows an LLM one such item, graded from obvious (another game) to unobvious (a random
Wikipedia article) and, within each kind, from near to far in embedding space, and asks which of the two, kind or
distance, decides whether the game it proposes anticipates games published after its training cutoff.

Positioning (`docs/paper/related_work_inspiration.md`): the inverted-U hypothesis is not new (Fu et al. 2013;
Gonçalves et al. 2013; contradicted by Chan, Dow and Schunn 2015), and an LLM design with sources graded by embedding
distance exists (Liu, Dubova et al. 2026, arXiv:2603.19087, judged originality only). New here: later releases as the
ground truth, kind crossed with within-kind distance, incidental vs instructed framing, and everyday stimuli.

## Data (frozen)

- **Pool:** `prepared/inspiration/pool.parquet`, built by `scripts/inspiration_pool.py` (seed 20261102), nine
  sources, text cut at 700 characters, every item dated before October 2023:
  `steam_game` (past split of our Steam corpus, release < 2024-01-01), `board_game` (BoardGameGeek, published
  ≤ 2022), `movie` (Wikipedia plots, 2018 dump), `sports_rules` (Wikipedia 2022-03-01 articles defining a sport or
  game whose body uses rule vocabulary; all 41 shards, 1,576 items), `paper` (past arXiv split, ≤ 2022), `patent` (past patent
  split, filed 2012-2015), `news` (AG News, 2004-05), `recipe` (Food.com, ≤ 2019), `wikipedia` (other Wikipedia
  2022-03-01 articles at random).
- **A-priori obviousness order** (fixed before any embedding, used only as a descriptive axis): steam_game >
  board_game > movie > sports_rules > paper > patent > news > recipe > wikipedia.
- **Past map:** past Steam games, adult rows removed. **Future set:** Steam future split with appid > 2,680,000
  (page created after gpt-4o's October 2023 cutoff; 32,322 of 40,989 non-adult future games). Future games with
  appid ≤ 2,680,000 (pages visible before the cutoff) are kept apart as a contamination check.
- **Embedders:** Qwen3-Embedding-0.6B (primary) and BGE-M3 (replication), same item text as the corpus.

## Stimulus distance and sampling

- `dist(s)` = 1 − max cos(s, past Steam games) in the primary embedder: how far the item is from any existing game.
- Within each source, items are cut into **quintiles of `dist`** (Q1 near ... Q5 far). From each source × quintile
  cell, **40 stimuli** are drawn at random (seed 20261103): 9 × 5 × 40 = 1,800 stimuli. Each `steam_game` stimulus
  is checked not to be a duplicate name of a future game.
- Distance is analysed both as quintile and as the continuous within-source rank (0-1), so kind and distance are not
  collinear. The BGE-M3 `dist` is reported for the same stimuli (rank agreement) and the shape tests are repeated with
  it.
- A `steam_game` stimulus is not compared with itself. Drawn by `scripts/inspiration_stimuli.py` (outputs
  `stimuli.parquet`, `momentum.parquet`, `pool_dist.parquet`).
- **Measured before generation (2026-11-03), descriptive only.** Median `dist` per source: steam_game 0.26,
  board_game 0.38, sports_rules 0.39, movie 0.52, paper 0.57, patent 0.59, wikipedia 0.59, news 0.59, recipe 0.60:
  close to the a-priori order, with sports_rules before movie. Within-source ranges are narrow (Q1 min to Q5 max, e.g.
  recipe 0.47-0.70), so source and distance are largely collinear across sources, which is why P1 is tested within
  source. Within-source rank agreement between embedders (Spearman over the 200 stimuli): 0.33 (recipe) to 0.75
  (steam_game); the BGE-M3 replication is therefore informative, not a formality.
- **Overlap contrast (exploratory):** sources whose quintiles share a distance band (e.g. board_game Q5 0.43-0.52 and
  movie Q1 0.37-0.48) are compared at matched `dist`, to separate kind from distance across sources.

## Arms (generator gpt-4o-2024-08-06, gateway `gpt-4o-0806`, temperature 1, one idea per call)

Each stimulus is used under both framings (paired):

- **INCIDENTAL:** the stimulus appears as "something you happened to read today", the task asks for an original
  video game for Steam, and the stimulus is not mentioned again.
- **INSTRUCTED:** the same text, followed by "use it as inspiration for the game".

Baselines without a stimulus:

- **NONE:** the same task with no material, 200 calls.
- **MOMENTUM (no calls):** 200 real past Steam games released in 2023, drawn at random from the cells of highest
  momentum (`where_future_lands.md`, share of a cell's items in the last two past years, top quartile of cells).
  Scored exactly as ideas, except that each game is removed from the past set it is compared with. Cells rebuilt
  identically (same seed, held-out draw and k-means): 64 top cells, 4,102 candidate 2023 games.

Output format: JSON with `title`, `short_description` (≤ 60 words), `genres`, `tags`; embedded in the same text
template as Steam items (`scripts/inspiration_generate.py`; prompts in that file). Total: 3,600 + 200 = 3,800 calls.

**Stress test, exploratory, contamination-tolerant (no API spend):** Claude Code and Codex through their
subscriptions (`claude -p` and `codex exec ... < /dev/null`), 25 stimuli per call returned as a JSON list, on a
subsample of 10 stimuli per cell, INSTRUCTED only (450 ideas each). Both models were trained after the future games
appeared, so their anticipation is read only relative to their own NONE arm (100 ideas each), never against gpt-4o.

## Measures (per idea)

- **Anticipation (primary):** as in `scripts/anticipation.py`: `hit` = max cos to the clean future set > max cos to a
  random past subsample of the same size (averaged over 5 subsamples); `margin` = their difference.
- **Isolation:** 1 − max cos to past Steam games.
- **Uptake (mediator, donor swap):** cos(idea, own stimulus) − mean cos(idea, 5 other stimuli of the same source and
  quintile). > 0 means the idea moved towards the material it was shown.
- **Coherence (secondary):** a blind judge (Codex through the subscription) rates "is this a coherent, playable game
  concept?" 1-5 on a stratified sample of 10 ideas per source × quintile × framing plus 20 NONE (920 ideas). Codex is
  never shown the arm, source or stimulus. Two-judge agreement on 100 of them with Claude Code (κ reported).

## Predictions (confirmatory, gpt-4o, primary embedder)

- **P1, distance does not raise anticipation.** Within source (fixed effects), the slope of `margin` on within-source
  distance rank is not positive (one-sided test of slope > 0 fails at 0.05). Pre-registered competitors, all
  reported: H-down (negative slope; Chan et al. 2015), H-up (positive; Liu et al. 2026), H-hump (two-lines test,
  Simonsohn 2018: a significant positive slope below and a significant negative slope above the breakpoint chosen by
  the Robin Hood algorithm). Prediction: H-up fails; between H-down and flat, no prediction.
- **P2, kind matters more than distance.** In a regression of `margin` on source (9 levels) and distance quintile (5
  levels), the partial η² of source exceeds that of quintile (bootstrap over stimuli, 2,000 resamples, CI of the
  difference excludes 0).
- **P3, isolation rises with distance.** Positive within-source slope of isolation on distance rank. Prediction:
  holds (it is the manipulation check; partly mechanical).
- **P4, instruction raises uptake.** Paired by stimulus, uptake INSTRUCTED > INCIDENTAL (Wilcoxon). Prediction: holds.
- **P5, no stimulus level beats the baselines.** No source × quintile cell has a mean `margin` above both NONE and
  MOMENTUM (each cell vs each baseline, one-sided Mann-Whitney, Holm over the 45 cells). Prediction: holds; MOMENTUM
  scores highest.

Holm over {P1, P2, P3, P4} at 0.05; P5 has its own Holm family. Tests use stimuli as the unit (INCIDENTAL and
INSTRUCTED analysed separately for P1-P3, and pooled as a robustness check).

## Exploratory (labelled as such)

- Shape tests for isolation, coherence and hit rate; the composite isolation × coherence is reported but cannot
  support a hump (built from opposite monotone trends).
- Uptake as a mediator: does `margin` rise with uptake within source and quintile?
- Contamination check: the same scores against future games with appid ≤ 2,680,000.
- The a-priori obviousness order vs the measured mean `dist` per source.
- Claude Code and Codex: the same curves relative to their NONE arms.

## Cost

gpt-4o: 3,800 calls × ~350 input and ~200 output tokens ≈ 1.3M input + 0.76M output tokens; at $2.50 / $10 per M
tokens ≈ $3.3 + $7.6 ≈ **$11 USD (≈ 17 AUD) in one day**, paid spend on Azure through an API gateway, logged in
`ledger.jsonl` with tag `inspiration`. Claude Code and Codex: subscription only, no API spend. Embedding: Kaggle
kernel (T4 × 2), both embedders, no spend.

## Result (2026-11-04, primary embedder; `inspiration_from_anything_result.json`, key `qwen3e06`)

Spend on 2026-11-04: 3,800 gpt-4o calls (0.91M input / 0.37M output tokens), $5.96 (≈ 9 AUD); 0 failed generations.
Clean future set: 32,322 games; contaminated (page visible before the cutoff): 8,667.

| | INCIDENTAL | INSTRUCTED |
|---|---|---|
| P1 within-source slope of margin on distance rank | +0.0038 (SE 0.0021, one-sided p 0.034; Holm 0.034) | +0.0053 (SE 0.0022, p 0.008; Holm 0.008) |
| P1 two-lines (Robin Hood breakpoint) | no hump (low +0.083, p 0.11; high +0.003, p 0.84) | no hump (low +0.045, p 0.17; high +0.003, p 0.87) |
| P2 partial η²: source vs quintile | 0.041 vs 0.004 (diff CI 0.021-0.059) | 0.060 vs 0.007 (diff CI 0.033-0.078) |
| P3 within-source slope of isolation | +0.022 (p < 1e-9) | +0.024 (p < 1e-10) |

- **P1 fails, weakly:** within a source, a farther stimulus gives slightly better anticipation (H-up, not H-down and
  not a hump). Over the whole within-source range the margin rises by 0.004-0.005, against between-source differences
  of up to 0.02.
- **P2 holds:** what kind of material is shown explains about ten times more variance in anticipation than its
  distance quintile.
- **P3 holds** (manipulation check).
- **P4 holds:** uptake INSTRUCTED 0.218 vs INCIDENTAL 0.193 (paired diff +0.025, p 6e-27, n = 1,800).
- **P5 fails:** six cells beat both baselines after Holm: recipe Q2-Q5, paper Q2, movie Q3. Baselines: NONE margin
  −0.001 (hit 0.525), MOMENTUM −0.002 (hit 0.511). The MOMENTUM baseline (real 2023 games from fast-growing cells)
  does not anticipate the clean future better than the stimulus-free ideas.

Mean margin by source (INCIDENTAL / INSTRUCTED): recipe 0.015 / 0.018, paper 0.013 / 0.013, movie 0.012 / 0.011,
patent 0.006 / 0.006, wikipedia 0.006 / 0.005, steam_game 0.004 / −0.005, news 0.002 / 0.005, sports_rules
0.001 / 0.002, board_game −0.001 / 0.001.

### Exploratory (not pre-registered)

- **The distance effect of P1 goes through isolation.** Adding the idea's isolation to the P1 regression leaves a
  slope of +0.0015 (SE 0.0021) and +0.0027 (SE 0.0022); margin and isolation correlate at r ≈ 0.2. A farther stimulus
  pushes the idea away from the past catalogue, and that alone raises the margin; distance adds nothing beyond it.
- **What the winning content does.** Recipes lead to cooking, cozy and crafting games (top tags: cooking, casual,
  relaxing, crafting); papers to strategy and simulation; movies to story-rich narrative games. With no stimulus, 60 %
  of the ideas involve time and the tags converge on exploration / atmospheric / time travel (178 distinct titles of
  200). The anticipation gain is content, not distance: recipes are among the farthest sources and also the lowest in
  uptake (0.11-0.13), yet the genre they push towards is one the post-cutoff catalogue grew into.
- **Showing a game and asking to use it hurts:** steam_game INSTRUCTED has the lowest margin (−0.005), with uptake
  doubling from 0.11 to 0.23: the anchoring of `anchoring_causal.md`, now towards a past item.
- Contamination check: against future games whose page predates the cutoff the margins are similar (INCIDENTAL
  0.006, INSTRUCTED 0.005, NONE 0.009), so the clean/dirty split does not change the reading.
- Pending: BGE-M3 replication (ideas to embed on Kaggle), coherence judge, Claude Code / Codex stress test.

### BGE-M3 replication (2026-11-04; key `bgem3`; ideas embedded on Kaggle, stimuli re-ranked by BGE-M3 `dist`)

| | INCIDENTAL | INSTRUCTED |
|---|---|---|
| P1 slope of margin on distance rank | +0.0015 (SE 0.0013, p 0.13) | −0.0008 (SE 0.0013, p 0.72) |
| P1 two-lines | no hump (low +0.049, p 0.0002; high −0.002, p 0.13) | no hump (low +0.009, p 0.06; high −0.003, p 0.17) |
| P2 η² source vs quintile | 0.017 vs 0.007 (diff CI −0.003 to 0.028; Holm 0.11) | 0.016 vs 0.008 (diff CI −0.004 to 0.026; Holm 0.17) |
| P3 isolation slope | +0.023 (p < 1e-15) | +0.022 (p < 1e-14) |

P4 replicates (uptake 0.163 vs 0.149, p 4e-16). P5 holds: no cell beats both baselines; every source has a negative
mean margin (−0.001 to −0.007), NONE −0.003, MOMENTUM −0.002.

**Reading across embedders.** Robust: distance does not produce a hump in anticipation (both embedders), farther
stimuli give more isolated ideas (P3), and instruction raises uptake (P4). The small positive distance slope of the
primary embedder does not replicate (and was already explained by isolation). Not robust: the source effect on
anticipation (P2) and the recipe / paper / movie advantage over the baselines (P5) appear in Qwen3 and vanish in
BGE-M3, so they cannot be reported as findings; the qualitative content result that survives is the shift of genre
(recipes → cooking games, NONE → time manipulation), which is visible in the ideas themselves, not in a distance.

### Coherence (secondary; key `coherence`)

Codex rated 920 ideas (stratified, blind), Claude Code 100 of them: quadratic-weighted κ 0.62 (unweighted 0.34,
Spearman 0.65); both judges ran from the repository directory (not the empty directory later used for generation),
which should not affect a blind coherence rating. Mean coherence: NONE 3.70, INCIDENTAL 4.02, INSTRUCTED 3.97. By
source (INCIDENTAL / INSTRUCTED): paper 3.28 / 3.14 lowest, patent 3.68 / 3.80, recipe 4.46 / 4.30 and news
4.38 / 4.42 highest. Within source, coherence does not change with distance when the material is incidental (slope
−0.04, SE 0.13) and falls when the model is told to use it (−0.27, SE 0.12, one-sided p 0.014): the farther the
material it is asked to build on, the less coherent the game. The stimulus-free ideas are the least coherent of all.

### Stress test: Claude Code and Codex (exploratory; keys `cli_codex`, `cli_claude`)

Subscription only, no API spend; 450 INSTRUCTED ideas (10 stimuli per cell, 25 per call) and 100 NONE per generator,
0 failed; calls from an empty directory with tools disabled. Read only against each generator's own NONE arm.

- **No distance effect:** within-source slope of margin +0.0045 (SE 0.0039, Codex) and +0.0016 (SE 0.0044,
  Claude); the isolation slope is near zero too (+0.006, +0.001), unlike gpt-4o's +0.022.
- **No recipe advantage:** recipes are indistinguishable from NONE in both (p 0.55, 0.59). Codex's only source above
  NONE is paper (+0.007, p 0.04 uncorrected).
- **Showing a game is worse than showing nothing, in all three generators:** steam_game margin −0.009 (Codex,
  p 0.03) and −0.010 (Claude, p 0.03) against NONE, as with gpt-4o INSTRUCTED.
- **The stimulus-free collapse is gpt-4o's:** NONE ideas about time 13 % (Codex) and 0 % (Claude) against 60 %;
  89 and 95 distinct titles of 100. Batching 25 ideas per call likely adds diversity, so this comparison is not clean.
