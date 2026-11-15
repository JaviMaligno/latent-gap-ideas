# Review: validating the anticipation margin of Section 6 (pre-registration draft)

Status: design only. Nothing below has been run. Written 2026-11-14 against `docs/paper/draft.md` at `5258bbf`. To be
committed in `docs/prereg/review_anticipation_control.md` before any computation. The paid arm (A4) is committed
together with the rest but runs only after Javier approves the spend.

## Review concerns addressed

- **M5** (major): the margin of Section 6 has no positive control. Its only near-control, 200 real 2023 games from
  high-momentum cells (MOMENTUM), scores −0.002, no better than ideas generated with no material (NONE, −0.001). The
  reviewer asks for (a) random held-out subsets of the future games themselves, scored against the rest of the future
  set; (b) the same with LLM rewrites of those games in the idea style; (c) signal-to-noise = (held-out future margin −
  NONE margin) / per-idea SD; (d) one sentence on the contaminated ("dirty") future set, where NONE scores +0.0086
  against −0.0011 with the strict set.
- **Question 6**: "What margin do held-out real post-cutoff games obtain against the rest of the future set?" Answered
  by A1.
- **Minor 15**: range restriction. State the within-source range of isolation for each source. Answered by A6.

## A fact found during design that changes the design

A read-only count made for this design (no margin was computed) shows that **tag presence differs strongly between the
sets that the margin compares**. Share of items with at least one Steam user tag in their embedded text, adult rows
removed (`prepared/steam/corpus.parquet`):

| set | n | share with tags | median tag count | median text length (chars) |
|---|---|---|---|---|
| clean future (appid > 2,680,000, released from 2024) | 32,322 | 0.36 | 0 | 1,206 |
| dirty future (released from 2024, appid ≤ 2,680,000) | 8,667 | 0.68 | 19 | 1,271 |
| past, released 2023 | 12,377 | 0.73 | 16 | 1,240 |
| past, before 2023 | 55,845 | 0.90 | 11 | 1,193 |
| MOMENTUM games | 200 | 0.73 | | 1,248 |
| gpt-4o ideas (all arms) | 3,800 | ≈1.00 | 9 | ≈ 30 words |

Recent games have had less time to collect user tags. Every idea has 5-10 tags (the prompt asks for them), so on format
alone ideas look like past games, and a tagless held-out future game looks like the rest of the clean future set.
Consequences, fixed now:

1. The raw held-out-future control (A1) is expected to be inflated by format, so it cannot validate the margin on its
   own. It answers question 6 but is not the arm that decides the reading.
2. A free format-matched control (A2) rescores everything inside the tagged-only universe (tagged future vs tagged
   past), where the tag difference is removed.
3. The style-matched paid control (A4) rewrites held-out games into the idea JSON, which gives them tags, the idea
   length and the generator's register. It is the arm that decides the reading.
4. The dirty-set sentence (A5) is tested against this format explanation before it is written.

## Question

Does the margin of Section 6 give real near-future game content a clearly higher score than ideas written with no
material, when the comparison is not confounded by format or style? If it does, how large are the Section 6 effects and
bounds as a fraction of that gap? If it does not, the Section 6 results are results on an unvalidated proxy.

## Data (all confirmed to exist, shapes checked 2026-11-14)

Root `D = ~/Documents/experiment-data/latent-gap-ideas/`.

- `D/prepared/steam/corpus.parquet`: 113,436 rows (`id, name, date, genres, tags, …, split, text, adult`); 4,225
  adult rows removed as in `scripts/inspiration_analysis.py` → past 68,222, clean future 32,322, dirty future 8,667.
- `D/prepared/steam/emb_qwen3e06.npy`, `D/prepared/steam/emb_bgem3.npy`: (113,436, 1024) float16, row-aligned with the
  corpus.
- `D/runs/inspiration/gpt-4o-0806/ideas.jsonl` (3,800 ideas: 1,800 INCIDENTAL, 1,800 INSTRUCTED, 200 NONE),
  `ideas_emb_qwen3e06.npy`, `ideas_emb_bgem3.npy`, and `scored_{qwen3e06,bgem3}.parquet` (3,800 rows with `margin`,
  `iso`, `source`, `quintile`, `rank`, `dist`).
- `D/prepared/inspiration/momentum.parquet` (200 MOMENTUM game ids), `stimuli.parquet` (1,800 stimuli,
  `dist_qwen3e06`, `dist_bgem3`, `quintile`), `pool_dist.parquet` (49,531 pool items with `source`, `dist_qwen3e06`,
  `dist_bgem3`, `rank`, `quintile`). `dist_e` is the isolation 1 − max cos to past Steam games (a Steam stimulus is not
  compared with itself; `scripts/inspiration_stimuli.py`).
- `docs/prereg/inspiration_from_anything_result.json` (existing Table 5 values, including
  `exploratory_dirty_future_margin`).
- Embedders for new texts (A4 only): Qwen3 through `latent_gap.embedder.LocalQwen` (ONNX, as used for the gpt-4o
  ideas); BGE-M3 locally on Apple MPS with the settings of `scripts/strengthen_bge_embed.py` (sentence-transformers,
  max_seq_length 512, normalised), which reproduced the Kaggle embedder in that study.

No `.pkl` file is read. No data outside these paths is needed.

## Common definitions (identical to `scripts/inspiration_analysis.py` unless stated)

- Embeddings L2-normalised in float32. Text of a generated item: `item_text(genres, tags, short_description)`; text of
  a real game: the corpus `text` field (the same template). Each embedder is processed separately (`e` ∈ {qwen3e06,
  bgem3}).
- **Margin** of a query q against a future reference set F and the past set P: `maxcos(q, F) − mean_{b=1..5}
  maxcos(q, P[S_b])`, where S_b is a random sample of P of size |F| without replacement. When q is itself a past game,
  q's own row is excluded from every S_b (as for MOMENTUM). **Common random numbers**: within one scoring of one
  reference set, the same five samples S_b are used for every arm.
- **Per-idea SD** (`sd_idea`): pooled within-source residual SD of the existing Table 5 margins (full clean future
  set), from y ~ source FE, computed separately per framing exactly as in `scripts/strengthen_inspiration_shape.py`
  (line 160-161), then averaged over the two framings. The NONE SD is reported as a second scale.
- Seeds: `SEED = 20261114`. All held-out sets (H0, the 100 splits H_r, H_T, the past-2023 draws, A5's clean subsets)
  are drawn once from the corpus index, before the embedder loop, with `numpy.random.default_rng(SEED)`, in the order
  the analyses are listed below, so they are the same in both embedders. The past samples S_b use a fresh
  `default_rng(SEED + 1)` at the start of each embedder, drawn per reference set in the same order.
- Bootstrap: 4,000 resamples, items resampled with replacement independently within each arm (paired arms resampled by
  pair), percentile 95 % intervals, `default_rng(SEED + 2)`.

## Analyses

### A1. Held-out real future games, as they are (answers question 6; free)

- H0: 200 games drawn uniformly without replacement from the clean future set. Reference F0 = clean future \ H0
  (32,122 games). Past samples of size 32,122.
- Arms scored against F0: H0 as-is (`fut_asis`); NONE (200); MOMENTUM (200); `past23_asis`: 200 past games released in
  2023 drawn uniformly (a real-game negative control from the era of MOMENTUM), each excluded from its own past samples.
- Statistics: mean margin per arm; Δ_F = mean(fut_asis) − mean(NONE); SNR_F = Δ_F / sd_idea; Δ_P23 = mean(past23_asis) −
  mean(NONE); and fut_asis − past23_asis. Bootstrap CIs.
- Stratified by tag presence of the held-out game (tagged / untagged), descriptive.
- Split variability (descriptive): 100 further random splits (H_r, F_r = clean future \ H_r, r = 1..100, drawn
  after H0 from the same generator); per split, mean(fut_asis) − mean(NONE scored against F_r). Report mean and 2.5 /
  97.5 % quantiles over splits.
- Near-duplicate sensitivity (descriptive): Δ_F recomputed with every reference item (future in F0, and past in each
  S_b) whose cosine to the query exceeds 0.97 excluded, applied to all arms alike.

### A2. Tag-matched universe (free; format control)

- Universe restricted to items with at least one tag: F_T = tagged clean future (≈ 11,600), P_T = tagged past
  (≈ 59,100). H_T: 200 games drawn uniformly from F_T; reference F_T \ H_T; past samples of that size from P_T.
- Arms: `fut_tagged` (H_T as-is), NONE, MOMENTUM (tagged ones only, ≈ 145, plus all 200 descriptively),
  `past23_tagged` (200 tagged 2023 past games, self-excluded).
- Statistics: Δ_T = mean(fut_tagged) − mean(NONE); SNR_T = Δ_T / sd_idea; Δ_P23,T; fut_tagged − past23_tagged.
- Exploratory rescoring of Table 5 in this universe (one line in the paper at most): every stimulus idea's margin
  recomputed against F_T (full, no hold-out) and P_T samples of size |F_T|; P1 slope (within-source OLS of margin on
  the within-source distance rank, as in `slope_within`) per embedder × framing, with 95 % CI from the classical SE;
  per-source means; NONE and MOMENTUM baselines. No new decision on P1-P5 is made from it.

### A3. Signal-to-noise and the yardstick for Table 5 (free, computed from A1, A2, A4)

For each validating gap Δ ∈ {Δ_F, Δ_T, Δ_R} and embedder: SNR = Δ / sd_idea (and / NONE SD) with bootstrap CI (sd_idea
held fixed). Then express, as fractions of Δ: the four P1 slopes of Table 5, the anchoring effect (past game vs no
material, 0.0052 and 0.0040), the source spread (SD of source means), and the observed 4-df and 8-df sup-t bounds on a
middle optimum (`strengthen_shape_band_observed.json`). These fractions are descriptive.

### A4. Style-matched held-out games: LLM rewrites (PAID; decides the reading)

- Items: the same 200 games of H0 (so that A1 and A4 are paired), plus the same 200 past-2023 games of A1 (style-matched
  negative control). 400 calls in one random order (`default_rng(SEED + 3)`).
- Generator: gpt-4o, alias `gpt-4o-0806` (the Section 6 generator), through `latent_gap.gateway.Gateway`,
  `json_mode=True`, `temperature=1.0`, `max_tokens=400`, the same retry policy as `scripts/inspiration_generate.py`
  (up to 3 attempts until the JSON has `title` and `short_description`), ledger tag `review-rewrite`. One rewrite per
  item.
- Prompt: system = the Section 6 system prompt ("You are a creative video game designer. Answer with a single JSON
  object and nothing else."). User:

  > Here is the Steam store entry of a video game:
  >
  > {corpus text of the game}
  >
  > Rewrite it as a proposal for a video game that could be released on Steam, keeping its concept, setting and
  > mechanics. Do not reuse its title. Return a JSON object with the keys "title", "short_description" (at most 60
  > words, like a Steam store blurb), "genres" (comma-separated Steam genres) and "tags" (5-10 comma-separated Steam
  > user tags).

  The output specification is the Section 6 `TASK` text verbatim, so rewrites have the ideas' fields, length and tag
  count. The game's name is not given (the corpus `text` field does not contain it).
- Embedding: `item_text(genres, tags, short_description)` in Qwen3 (LocalQwen) and BGE-M3 (local MPS). Failed
  generations (empty after 3 attempts) are dropped and counted; if more than 10 of either group fail, this is reported
  and the analysis runs on the rest.
- Scoring against F0 with the same past samples as A1 (common random numbers); the original past game of each
  `past23_rw` item is excluded from its past samples.
- Statistics: Δ_R = mean(fut_rw) − mean(NONE); SNR_R = Δ_R / sd_idea; D_R = mean(fut_rw) − mean(past23_rw) (the
  future-vs-past contrast with style held fixed); paired style cost = mean(fut_rw − fut_asis) over the 200 pairs.
- **Confirmatory tests** (one-sided, alpha 0.05, Holm over the family of 8 = {Δ_F > 0, Δ_T > 0, Δ_R > 0, D_R > 0} × 2
  embedders): Welch t tests (D_R and Δ tests on independent arms), reported with the bootstrap CIs. The reading rule
  below uses the SNR intervals, not only the tests.

### A5. The dirty future set (free; the one sentence)

The observed NONE margin is −0.0011 against the clean set (32,322) and +0.0086 against the dirty set (8,667),
a gap G_obs = 0.0097 in Qwen3; in BGE-M3 −0.0027 and +0.0053, a gap of 0.0080 (`inspiration_from_anything_result.json`). Decomposition,
per embedder, NONE ideas only:

1. Size: NONE margin against 20 random subsets of the clean set of size 8,667 (past samples of the same size);
   G_size = dirty − mean of these.
2. Format: NONE margin against the tagged dirty items and against 20 random subsets of the tagged clean items, both at
   the size of the tagged dirty set (≈ 5,860), past samples drawn from tagged past; G_tag = dirty_tagged −
   clean_tagged. The same for untagged items at the size of the smaller untagged set (descriptive).
3. Release timing (descriptive): median release date of each set.

Reading rule: if |G_tag| < 0.5 · G_size, the sentence attributes the dirty set's higher baseline mainly to its higher
tag share (format), not to contamination; if |G_tag| ≥ 0.5 · G_size, it says the gap is not explained by tag format
and is consistent with the dirty games being nearer the cutoff and partly visible before it. Either way the sentence
also states that the gap is about twice the largest effect in Table 5, so the choice of future set matters for levels
but not for within-set contrasts, which use one set throughout.

### A6. Within-source range of isolation (minor 15; free, descriptive)

Per embedder and source, for (a) the 49,531-item pool (`pool_dist.parquet`), (b) the 1,800 drawn stimuli
(`stimuli.parquet`), and (c) the ideas' own isolation `iso` (`scored_e.parquet`, stimulus arms): min, 5th, 25th,
50th, 75th, 95th percentile, max, IQR, the 5-95 range, the four quintile cut points of the pool, and the ratio of the
source's 5-95 range to the 5-95 range of all 1,800 drawn stimuli pooled. A source is called **range-restricted** if its
ratio in (b) is below 0.5 (fixed now). Also reported: the share of the pooled 5-95 range covered by the union of the
nine within-source 5-95 ranges, and, exploratory, the P1 slope re-expressed per 0.1 of raw isolation (within-source OLS
of margin on `dist_e` with source FE, per framing), so that the slope can be read against the range each source
spans.

Reading: descriptive. The paper adds a table (appendix) and one sentence stating the per-source ranges, how many sources
are range-restricted, and that between-source distance is confounded with content, so the design tests distance only
within the ranges listed; no claim is made about an optimum outside or across them.

## Reading rule (fixed in advance)

The governing quantity is **SNR_R** (style-matched), per embedder; the weaker embedder governs the paper's wording.
Let L be the lower end of its 95 % bootstrap interval.

| outcome | condition | reading |
|---|---|---|
| **V: validated** | L ≥ 0.5 in both embedders, and D_R > 0 after Holm in both | The margin gives real post-cutoff content in idea form a clearly higher score than no material, and more than a recent past game in the same form. |
| **W: weakly validated** | Δ_R > 0 after Holm in both, but L < 0.5 in at least one | The margin is sensitive to future content but with low signal-to-noise; Section 6 nulls are weak. |
| **F: format-only** | Δ_F (or Δ_T) > 0 after Holm, but Δ_R not > 0 after Holm in at least one embedder | The margin detects real future games through their surface (format, register), not their content once written as an idea. |
| **N: not validated** | none of Δ_F, Δ_T, Δ_R > 0 after Holm in at least one embedder | The margin does not separate real future games from no-material ideas. |

Additional fixed thresholds, applied under V and W:

- **Null informativeness**: let U4 and U8 be the observed 4-df and 8-df sup-t bounds (largest over cells). If U4 <
  0.25 · Δ_R in both embedders, the null on a broad optimum is informative relative to the measure; if U4 ≥ 0.5 · Δ_R,
  it is uninformative (an optimum up to half the distance between "no material" and "a real future game" cannot be
  excluded); in between, reported as is.
- **MOMENTUM reading**: if Δ_P23 (and D_R) shows real 2023 games scoring at or below NONE while fut_asis and fut_rw
  score above, the MOMENTUM ≈ NONE result is a statement about past games being past, consistent with the measure, not
  evidence of insensitivity.

**What counts against the paper's current claim.** The current heading and conclusion of Section 6 ("No sweet spot
detected for inspiration distance"; "distance in the map does not tell you how far to look") count against the paper
under outcome **N** or **F** (the margin is not shown to measure anticipation of content), and under **V/W with U4 ≥
0.5 · Δ_R** (the null is too wide relative to the measure to say distance does not matter). The P1 failure in Qwen3 (a
small positive slope) is weakened under N and F, since it is then a slope on an unvalidated score.

## What the paper will say under each outcome

- **V**, U4 < 0.25 · Δ_R: keep the heading. Add to Section 6 a paragraph "Validating the margin": a real post-cutoff
  game rewritten by the same model in the idea format scores Δ_R above no material (SNR_R, interval) and D_R above a
  2023 game rewritten the same way; the largest Table 5 slope is x % of that gap and the bound on a broad middle
  optimum y %. Answer question 6 with Δ_F, noting that as-is games are helped by format (tag share 0.36 vs 0.90). Lines
  639-641 reread per the MOMENTUM rule.
- **V or W**, U4 ≥ 0.25 · Δ_R: as above, but the conclusion of Section 6 says that optima up to y % of the gap between
  no material and a real future game are not excluded; if ≥ 0.5, the sentence "distance in the map does not tell you how
  far to look" becomes "no effect of distance was detected, and effects of practical size relative to the measure are
  not excluded". Abstract part (4) changes accordingly.
- **W** additionally: state the low signal-to-noise in the abstract's description of part 4.
- **F**: heading becomes "No effect of inspiration distance detected on a margin that tracks format more than
  content"; the abstract's part (4) and the conclusion say that the margin separates real future games from ideas mainly
  by format, so the part-4 nulls and the P1 slope are on a proxy not shown to measure anticipation of content; the
  anchoring result (past game below no material) is kept as a within-measure contrast, worded "on this margin".
- **N**: heading becomes the reviewer's "No effect of inspiration distance on an unvalidated anticipation proxy";
  part 4 is demoted in the abstract to one clause; the Discussion states that the design cannot speak to inspiration
  distance until a validated measure exists.
- **Embedders disagree**: the weaker outcome sets the heading; the stronger is reported alongside.
- In all outcomes: the dirty-set sentence from A5; the range table and sentence from A6; A2's rescored P1 slopes in
  one line (exploratory), and any reversal of sign relative to Table 5 stated plainly.

## Cost and compute

- **Paid generation: A4 only.** 400 gpt-4o calls. Input is longer than in Section 6 (a real store entry, median 1,206
  characters ≈ 300 tokens, plus ≈ 120 tokens of prompt) ≈ 420 input and ≈ 110 output tokens per call ≈ 0.17M input +
  0.044M output tokens; at $2.50 / $10 per M ≈ $0.42 + $0.44 ≈ **$0.9 USD**. By analogy with the measured Section 6 run
  (3,800 calls, $5.96, i.e. $0.0016 per call with shorter inputs), allow up to **$1.5 USD (≈ 2.3 AUD), all on one day**,
  paid spend on Azure through an API gateway, ledger tag `review-rewrite`. Wall time ≈ 5-10 min at the Section 6
  concurrency.
- **GPU: none.** New embeddings: 400 texts, Qwen3 ONNX on CPU (≈ 2-4 min) and BGE-M3 on Apple MPS (≈ 2-3 min, the
  route of `strengthen_bge_embed.py`). No Kaggle kernel needed.
- **Local runtime (free parts A1-A3, A5, A6)**: about 10-15 min in total. A1's 101 splits score ≈ 800 queries against
  ≈ 32k future items and 5 past samples each (≈ 1-2 s per split per embedder); A2's exploratory rescoring scores 3,600
  ideas against ≈ 11.6k items × 6 (≈ 1 min); A5 needs 41 scorings of 200 ideas (seconds); bootstraps are on arm means.
- **Peak memory**: one embedder at a time, corpus embeddings as normalised float32 (≈ 465 MB), similarity chunks of
  1,000 queries (≈ 130 MB) → ≈ 1.5-2 GB; the BGE-M3 embedding step, run separately, ≈ 3 GB. Within the 16 GB Mac.

## Scripts and outputs

- `scripts/review_anticipation_control.py` (no LLM calls): subcommands `free` (A1, A2, A5, A6, and the A3 parts that
  do not need A4) and `rewrite_score` (embeds A4's outputs, scores them, runs the Holm family and A3, writes the
  outcome label). Output: `docs/prereg/review_anticipation_control.json`, keys per embedder: `A1_heldout`,
  `A1_splits`, `A1_dedup`, `A2_tagged`, `A2_rescored_table5`, `A4_rewrite`, `A3_snr_yardstick`, `holm`, `outcome`, and
  top-level `A5_dirty`, `A6_isolation_range`, `sd_idea`, `n_failed_rewrites`, `seed`.
- `scripts/review_anticipation_rewrite.py` (paid; Gateway): writes `D/runs/review-anticipation/rewrites.jsonl`
  (one record per item: corpus id, group `fut`/`past23`, title, short_description, genres, tags), resumable, with
  `LG_LIMIT` for a format check of ≤ 6 calls that is kept in the data and disclosed.
- Held-out ids (H0, H_T, past-2023 draws, the 100 split seeds) are written to
  `D/runs/review-anticipation/heldout_ids.json` by the `free` step before any margin is computed, and A4 reads them
  from there, so the rewritten games are the A1 games.

## Order

1. Commit this file. 2. Run `free`. 3. With Javier's approval of ≈ $1.5 USD: format check (≤ 6 calls), then the 400
calls. 4. Run `rewrite_score`. 5. Disclose any deviation in a dated section of this file.

## Infeasible or approximate versions (honest limits)

- **True style match is approximate.** A rewrite by gpt-4o is in the generator's style, which is the point, but it is
  also a compression of the original that may drop or regularise content; Δ_R therefore estimates the margin of "a real
  future game's concept, written as an idea by this model", a lower-bound-type positive control, not the margin of the
  game itself. The paired style cost (fut_rw − fut_asis) is reported so the reader sees how much the rewriting costs.
- **Rewrites of past games as the negative control** carry the model's prior knowledge of those games (released before
  its October 2023 cutoff, some possibly seen); this can only make past rewrites more fluent, not nearer the future set.
- **No positive control exists for the stimulus-to-idea path itself** (an idea inspired by material known to anticipate
  the future), because no such material is known; A4 validates the scoring of ideas, not the generator's ability to use
  inspiration.
- **Range restriction cannot be fixed with existing data**: an inverted U over the full range of distance would need
  stimuli matched in content but differing in distance, which the pool does not provide; A6 states the ranges instead
  of testing across them.
