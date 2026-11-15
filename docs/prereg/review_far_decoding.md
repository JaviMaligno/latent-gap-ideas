# Review follow-ups for Section 3: far targets and decoding (pre-registration draft)

Status: DRAFT design, written 2026-11-14 before any of the analyses below was run. Nothing here has been computed.
To be committed to `docs/prereg/review_far_decoding.md` (one file, four studies) BEFORE the scripts run. Deviations
will be disclosed in a dated "Deviations" section appended after the runs.

All four studies are exploratory follow-ups registered after the main results were known (review minor 2). The
paper will label them "registered follow-up (review)", not "pre-registered" in the sense of the original studies.

## Review concerns addressed

| Concern (fable_peer_review.md) | Study here |
|---|---|
| M6.2 / Question 2: "After mean-centring (or whitening), are FAR targets still unreachable? The cap is largely a property of the uncentred space." | R1 (W1, W2, W3) |
| Minor 21 (last sentence): hubness and anisotropy corrections would change the cap geometry on which Section 3 depends; say whether results were checked after centring or whitening | R1 (W1 gives the sentence) |
| Minor 4: the 12 cells share targets and donor pairings across decoders, so they are not 12 independent confirmations; 20 targets per cell is small for FAR | R2 |
| Minor 5: the 0.03-0.07 own-vs-other margin is evidence that the score channel carries target information; with no no-score control the paper cannot say "the score alone does not move text towards a far point" | R3 |
| M6 (end): run one trained inverter (vec2text or a successor for a supported encoder) on REAL, GAP and FAR targets, or say plainly that the claim does not cover inversion | R4 (closest feasible version, gtr-t5-base) |

Out of scope here: the hole-score half of Question 2 and Question 3 (Section 4, forecasting group). M6.1 and M6.3
are text changes (retitle the claim; move the GAP seed-and-rewrite row of Table A3 into the main text); R4's GAP arm
gives a second, independent decoder for M6.3.

## Data (all confirmed to exist on 2026-11-14 by read-only inspection)

Root `DATA = ~/Documents/experiment-data/latent-gap-ideas`.

- Corpora: `DATA/prepared/{steam,arxiv,patents}/corpus.parquet` (columns include `split`, `text`, `adult`; Steam
  71,939 past + 41,497 future rows before the adult filter, 4,225 adult rows; arXiv 72,000 + 41,000; patents
  71,521 + 40,342). Analysis rows: `~adult`, as in every prior script.
- Embeddings: `DATA/prepared/{dom}/emb_qwen3e06.npy` and `emb_bgem3.npy`, float16, shapes (113436, 1024),
  (113000, 1024), (111863, 1024), row-aligned with `corpus.parquet`. Cast to float32 and L2-normalise on load.
- Anchoring targets: `DATA/runs/{steam,arxiv}/pilot/targets.npz` (keys `REAL`, `REAL_hidden`, `GAP`, `FAR`, ...;
  targets k = 1..20 are the anchoring set, `KS = range(1, 21)` in `scripts/anchoring_causal.py`).
- Anchoring decodes: `DATA/runs/{steam,arxiv}/anchoring-causal/{luna,sonnet}/{true,swap,none}/{REAL,GAP,FAR}/items.jsonl`
  (20 rows each; per row `neighbours` = indices into the non-adult past matrix in corpus order, `donor_k`, `history`
  with 4 rounds, each round holding the round's best candidate `emb` (Qwen3, 1024) and the three `candidate_sims`
  to the own target, and `best`). Row-level analysis table: `DATA/runs/anchoring-causal-analysis/items.csv`
  (720 rows = 2 domains x 2 decoders x 3 arms x 3 conditions x 20).
- Saved FAR targets of the grid (for the future check): `DATA/runs/{dom}/grid-gpt-6-luna/targets.npz` (50 per arm).
- Target construction code: `src/latent_gap/targets.py` (`real`, `gap`, `far`, `max_cos`; `N_CANDIDATES = 20_000`,
  `DUP_COS = 0.95`). Coherence gate code: `src/latent_gap/coherent.py` (`gibberish_stats`, `gibberish_rule`,
  `judge`) and `scripts/seed_rewrite.py` (`JUDGE1 = claude:claude-haiku-5-5`, `JUDGE2 = codex:gpt-6-luna`,
  `COH_PASS = 7`), run through subscription CLIs (`latent_gap.cli_judges`), never the gateway.

Common constants: bootstrap B = 10,000 percentile resamples; master seed 20261114; per-study seeds stated below.

---

## R1. Centring and whitening: does the far-point result depend on the uncentred cap?

**Question.** Section 3 explains that far points are far from every text because texts occupy a narrow cap
(median pairwise cosine 0.41 on Steam; corpus-mean norm 0.51-0.64). After the standard anisotropy corrections,
(a) how much of that cap remains, (b) are FAR targets, rebuilt by the same rule in the corrected space, still farther
from every past and future text than real items are from their nearest neighbour, and (c) does the causal anchoring
result of Table A2 survive when targets, texts and centroids are mapped into the corrected space?

**Transformations.** Parameters estimated on the non-adult PAST split of each corpus and embedder only (as every map
in the paper). Let X be the L2-normalised past embeddings (N x d, float32), mu = X.mean(0), Sigma = cov(X, bias=True),
eigen-decomposition Sigma = U diag(lam) U^T with lam descending.

- T0 raw: f(x) = x (reproduction check: must reproduce Table A2 to 3 decimals and the A.3 FAR ceilings).
- T1 centring: f(x) = norm(x - mu).
- T2 all-but-the-top (Mu and Viswanath, 2018), D = 10 (= round(d/100)): f(x) = norm((I - U_D U_D^T)(x - mu)).
- T3 BERT-whitening (Su et al., 2021), k = 256: f(x) = norm(diag(lam_1..k)^(-1/2) U_k^T (x - mu)), d' = 256.
- T4 full ZCA whitening with shrinkage: f(x) = norm(U diag((lam + eps)^(-1/2)) U^T (x - mu)), eps = 0.01 * mean(lam).

f is applied to every text embedding (past, future, decoded). Embedders: Qwen3 (primary) and BGE-M3 (replication).
Corpora: Steam, arXiv, patents for W1-W2; Steam and arXiv for W3 (the anchoring corpora; Qwen3 only, because the
anchoring targets are Qwen3 points).

### W1. Cap geometry (descriptive, no test)

Per corpus x embedder x transformation (3 x 2 x 5 = 30 rows), on the transformed past Y = f(X):
- median and IQR of the cosine of 100,000 random item pairs (seed 20261114 + 1, pairs with i != j);
- norm of mean(Y) (the "latitude" scale);
- k90 = number of principal directions of Y holding 90 % of its energy (uncentred second-moment matrix, as in A.3),
  and the in-slab fraction m(v) = ||U_k90^T v|| of (i) 2,000 random items, (ii) the 50 rebuilt FAR' targets of W2,
  (iii) the value sqrt(k90/d') of a random direction;
- participation ratio (sum lam)^2 / sum lam^2 of cov(Y);
- hubness: skewness of the 10-occurrence distribution over a 10,000-item random subsample (seed 20261114 + 2),
  cosine kNN within the subsample.

### W2. Rebuilt targets in each space (primary for M6.2)

Per corpus x embedder x transformation (T0-T4):
1. REAL', GAP', FAR': 50 targets each, built on Y with `targets.real`, `targets.gap`, `targets.far` exactly as in
   the code (20,000 candidates; FAR' = uniform random directions in R^{d'}; GAP' = normalised draws from N(mean(Y),
   cov(Y)); REAL' hides its cos > 0.95 duplicates). rng = `np.random.default_rng(20261114 + 1000*c + 100*e + 10*t)`
   with c in {steam 0, arxiv 1, patents 2}, e in {qwen3 0, bge 1}, t = transformation index; arms drawn in the order
   REAL, GAP, FAR from the same generator.
2. Corpus ceiling of each target: max cosine to Y (REAL' with its hidden set excluded).
3. Nearest-neighbour reference: 5,000 random past items (seed 20261114 + 3), each its max cosine to the rest of Y
   after excluding itself and its cos > 0.95 duplicates. q01 = 1st percentile, q50 = median of these.
4. Future reach: future items mapped with the same f; per FAR' target, the max cosine of any future item; share of
   FAR' targets where it exceeds q01.
5. Null scale: median ceiling over all 20,000 FAR candidates (before selecting the 50 most isolated).

Estimators: median FAR' ceiling with bootstrap 95 % CI over the 50 targets (B = 10,000, seed 20261114 + 4); the
ratio rho = median FAR' ceiling / q50 (bootstrap resamples targets and reference items independently); max FAR'
ceiling; same for GAP'.

**Reading rule (fixed in advance).** A cell is "FAR' unreachable" if the MAXIMUM FAR' ceiling is below q01 and no
future item exceeds q01 for any FAR' target (every far target is farther from every past and future text than 99 % of
real items are from their nearest non-duplicate neighbour). "Mostly unreachable" if the median ceiling is below q01
but the maximum is not. "Reachable" if the median FAR' ceiling is >= q01. Replication rule as in the paper: a
transformation's verdict for a corpus counts only if Qwen3 and BGE-M3 agree.
- Supports the current claim: every corpus is "FAR' unreachable" under T1-T4 in both embedders.
- Counts against the current claim: any corpus "Reachable" under any of T1-T4 in both embedders. Then far
  unreachability is partly a property of the uncentred metric, and the paper must say so for that case.
- Separately, the EXPLANATION in Section 3 ("texts occupy a narrow cap") counts as wrong for a corrected space if
  W1's median pairwise cosine there is <= 0.05 while W2 still says unreachable: the paper then attributes far
  unreachability to concentration in a low-dimensional slab (k90, in-slab fraction) and to the max of N cosines in d'
  dimensions, not to the cap.

### W3. Transported anchoring (Table A2 in corrected spaces)

For the 12 anchoring cells (Steam, arXiv x luna, Sonnet x REAL, GAP, FAR; Qwen3), recompute exactly the Table A2
quantities with every vector mapped by f before the cosine: decoded embedding e = `best.emb` of each condition,
shown centroid C = norm(mean(f(P[neighbours]))) (centroid taken AFTER the map), target t.
- Targets are not texts, so their map is ambiguous. Primary: point map f(t) (the target treated as a point on the
  sphere, like a text). Sensitivity: direction map g(t) = norm(A t), with A the linear part of f (identity for T1,
  I - U_D U_D^T for T2, the whitening matrix for T3/T4), i.e. the target as a direction from the corpus centre.
- D = f(e_true).f(t) - f(e_swap).f(t); A = (f(e_swap) - f(e_true)) . (C_donor - C_true); true/swap/none landings.
- Intervals: the original one-sided Bonferroni intervals (t-intervals over 20 targets, each bound at 0.05/24) and,
  for the four corrected spaces together, 0.05/96 (24 endpoints x 4 transformations).

**Reading rule.** Per transformation, the direction-of-anchoring claim "survives" if A's lower 0.05/96 bound is > 0
in all 12 cells, "weakens" if in 9-11, and "fails" if in fewer than 9 or if any A bound is entirely below 0. The FAR
claim "the small landing with true neighbours disappears with swapped or none" survives if in all four FAR cells
the true-condition landing exceeds the swap and none landings (D lower bound > 0 at 0.05/96). T0 must reproduce
Table A2 to 3 decimals, or the script stops.

### What the paper will say (R1)

- Survives (W2 all unreachable, W3 survives): Section 3 adds: "After mean-centring, all-but-the-top, BERT-whitening
  and shrunk whitening of both embedders, the cap shrinks [W1 numbers] but far targets rebuilt in each space remain
  farther from every past and future text than 99 % of real items are from their nearest neighbour, and swapping the
  neighbours still moves the text towards the donor in 12 of 12 cells." Minor 21 sentence: as this one. If the W1
  condition fires, the explanation is rewritten from "texts occupy a narrow cap" to "texts concentrate in a
  low-dimensional slab"; the cap remains the description of the raw space.
- Counts against (any corpus Reachable in both embedders): the abstract's far-point claim is restricted to the raw
  (uncentred) cosine the recipe uses, and Section 3 states for which corrected spaces and corpora far targets come
  within the nearest-neighbour range, with the numbers.
- W3 weakens or fails: the 12-of-12 statement is qualified to the raw cosine, and the corrected-space counts are
  reported next to it.

**Cost and runtime (R1).** No generation, no GPU. Local, numpy float32. Dominant cost: max_cos of 2 x 20,000
candidates against ~72k items per cell (about 3e12 multiply-adds per cell at d = 1024, about 10-20 s with Accelerate
BLAS), 30 cells, plus 5,000 NN queries and future queries: about 25-40 min total. Peak memory about 1.5-2 GB (one
corpus' float32 embeddings ~450 MB, one transformed copy, 1000 x 72,000 similarity chunks of 290 MB). Run corpora
sequentially and free arrays between them.

Script: `scripts/review_far_whitening.py` -> `docs/prereg/review_far_whitening.json`.

---

## R2. Independent units in the anchoring test (minor 4)

**Question.** Counting targets, not decoder cells, as the independent units, does the anchoring result hold in each
of the 6 independent target sets (domain x arm, 20 targets each), and how precise is it?

**Data.** `DATA/runs/anchoring-causal-analysis/items.csv` (T0) and, for the corrected spaces, the per-item table W3
writes.

**Computation.** For each domain x arm (6 sets) and each target k, D_k^g and A_k^g per decoder g in {luna, Sonnet}
(definitions as Table A2), and the decoder-pooled D_k = (D_k^luna + D_k^Sonnet)/2, A_k likewise. Per set: mean of
D_k and of A_k with a percentile bootstrap that resamples TARGETS (both decoders' values of a target move together;
B = 10,000, seed 20261114 + 20); number of targets with A_k^g > 0 for both decoders; Spearman correlation of
D_k^luna and D_k^Sonnet across the 20 targets (decoder agreement); for FAR, the CI half-width of pooled D. Holm over
the 6 sets on the one-sided bootstrap p-value of mean A_k <= 0 (p = share of resamples <= 0, floored at 1/B).

**Reading rule.** The direction-of-anchoring claim is restated in independent units if all 6 sets reject at Holm
0.05. It counts against the current wording if any set fails to reject: the paper then reports "k of 6 independent
target sets" and drops "12 of 12" from the abstract. A decoder agreement rho >= 0.5 in a set means the two decoder
cells of that set are largely the same confirmation (reported as such).

**Paper text.** "The 12 cells are 6 independent sets of 20 targets each, shared by the two decoders; counting targets,
the donor attraction is positive in 6 of 6 sets (Holm), with pooled FAR target loss D = x [CI]." Under failure, the
count and the failing set are stated. The 12-of-12 figure stays only as a per-decoder description.

**Cost and runtime.** No generation, no GPU; seconds, < 200 MB.

Script: `scripts/review_far_units.py` -> `docs/prereg/review_far_units.json`.

---

## R3. Does the score alone steer? A within-run no-feedback control (minor 5)

**Question.** In the no-neighbour condition the only target-specific input the decoder ever receives is the score
feedback. Does the decoded text move towards its OWN target, more than towards other targets, across rounds?

**Key observation that makes this feasible without new generation.** In `decode()` the round-0 prompt of the `none`
condition contains no target information and no feedback; rounds 1-3 add only the score messages. Each round keeps its
best-of-3 candidate (selected by own-target cosine), so selection is the same best-of-3 in every round. Round 0 is
therefore a same-prompt, same-selection, no-feedback control inside each run.

**Data.** `DATA/runs/{steam,arxiv}/anchoring-causal/{luna,sonnet}/none/{REAL,GAP,FAR}/items.jsonl`: per target,
`history[r].emb` (the round-r best) and `history[r].candidate_sims` (three own-target cosines), r = 0..3; targets
from `pilot/targets.npz`, k = 1..20. Rounds with no valid candidate (`sim` None) are dropped; a target missing round
0 or round 3 is excluded and counted.

**Computation (raw Qwen3; repeated under T1 and T3 of R1 as sensitivity).**
- S1 (own-target gain): per target, max(candidate_sims round 3) - max(candidate_sims round 0).
- S2 (target-specific steering, primary): m_r = e_r . t_k - mean_{j != k} e_r . t_j over the 19 other targets of the
  same domain and arm; S2_k = m_3 - m_0. Generic drift across rounds (style, length) cancels in m; selection is equal
  in rounds 0 and 3.
- S2' (all feedback rounds): mean(m_1, m_2, m_3) - m_0.
- Selection-only margin: m_0 itself (what the paper's 0.03-0.07 margin would be with no feedback at all, for 3
  candidates instead of 12), reported next to the final `best` margin.
Pooled over decoders per domain x arm (target-level average of the two decoders), percentile bootstrap over targets,
B = 10,000, seed 20261114 + 30. Family for Holm: S2 in {REAL, GAP, FAR} x {Steam, arXiv} = 6 one-sided tests of
S2 > 0. The same quantities are also run in the `true` condition (descriptive: there the neighbours are target-specific
but constant across rounds).

**Reading rule.**
- "No detectable steering by the score alone" for FAR if S2 (FAR) does not reject at Holm 0.05 in either domain AND
  the upper 95 % bound of S2 is below 0.02 (the FAR minimum effect registered in `anchoring_causal.md`).
- Counts against the sentence "the score alone does not move text towards a far point" if S2 (FAR) rejects in either
  domain: the score channel then carries target-specific steering of size S2.
- Inconclusive if it does not reject but the upper bound is >= 0.02.

**Paper text.** Supported: "Within each no-neighbour run, round 0 sees no feedback; from round 0 to round 3 the text
moves towards its own far target by S2 = x [CI] more than towards the other targets, so the score alone does not
detectably steer (bound 0.02); the 0.03-0.07 margin is mostly selection (round-0 margin m_0 = y)." Against: the
sentence is replaced by "the score alone moves the text towards a far point by x, from a landing of z", and the
abstract's "disappears" is softened. Inconclusive: the sentence is removed and the limitation kept.

A true no-score arm (12 one-shot candidates, `feedback_mode="none"`) is not needed for this question: round 0 is that
arm with 3 candidates. If wanted anyway it runs on the subscription CLIs (the anchoring generators were
`codex:gpt-6-luna` and `claude:claude-sonnet-5-5`), not the gateway: 2 domains x 2 decoders x 3 arms x 20 targets x 12
candidates = 2,880 CLI calls, no API spend, about 3-5 h wall time with rate limits. Not part of this registration.

**Cost and runtime.** No generation, no GPU; under a minute, < 500 MB.

Script: `scripts/review_far_score_channel.py` -> `docs/prereg/review_far_score_channel.json`.

---

## R4. A trained inverter on REAL, GAP and FAR targets (closest feasible version)

**Feasibility.** No public inverter exists for Qwen3-Embedding-0.6B or BGE-M3. vec2text ships pretrained correctors
only for `gtr-base` (inverter `jxm/gtr__nq__32`, corrector `jxm/gtr__nq__32__correct`, 32-token inputs) and OpenAI
`text-embedding-ada-002`. Training an inverter for Qwen3 is unreasonable here: vec2text's own notes put hypothesis
generation on MS MARCO at days on one GPU before corrector training, beyond a free Kaggle quota (30 GPU-h per week,
12 h sessions) and outside a review revision. ada-002 would be paid external spend through an Azure deployment that
may not exist, so it is not the primary. The closest feasible version is therefore to rebuild the Section 3 targets
in the gtr-t5-base space and invert them there. This tests "inversion as the decoder" in a different, supported
space with 32-token texts; it does not test inversion of Qwen3 targets, and the paper will say so.

**Space.** gtr-t5-base as vec2text uses it: `AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder`,
mean pooling of the last hidden state over the attention mask (`vec2text.models.model_utils.mean_pool`), inputs
truncated to 32 tokens (`max_length=32, truncation=True, padding="max_length"`). Geometry (targets, ceilings, scores)
uses the L2-normalised vectors; vectors passed to the corrector are the unit target times the median raw norm of the
corpus embeddings (the corrector expects unnormalised mean-pooled vectors).

**Texts.** Non-adult PAST rows of Steam and arXiv (the anchoring corpora): `corpus.parquet` `text`, which the
gtr tokenizer truncates to its first 32 tokens (title and opening). Patents left out (not an anchoring corpus).

**Stages.**
1. Kaggle kernel `kaggle/review-gtr-embed` (T4 x2): embed the ~67.7k Steam and 72k arXiv past texts with both GPUs
   (one process per GPU, half the rows each), saving float16 raw mean-pooled vectors per block and printing one plain
   progress line per block of 4,096 texts (no tqdm). Output ~2 x 72k x 768 x 2 B = ~220 MB.
2. Local (`scripts/review_far_inversion.py build`): build 50 REAL, 50 GAP, 50 FAR targets per domain on the
   normalised gtr past matrix with `targets.real/gap/far` (rng `default_rng(20261114 + 500 + 10*c)`, c in {steam 0,
   arxiv 1}, arms in order REAL, GAP, FAR); store targets, REAL hidden sets, corpus ceilings (REAL with hidden
   excluded) and W1 geometry of the gtr space (median pairwise cosine, mean norm, k90). Write the targets npz into a
   private Kaggle dataset.
3. Kaggle kernel `kaggle/review-gtr-invert` (T4 x2, one domain per GPU): `vec2text.load_pretrained_corrector("gtr-base")`;
   `vec2text.invert_embeddings` with three budgets: C0 num_steps=0 (inverter hypothesis only), C1 num_steps=20
   sequence_beam_width=1, C2 num_steps=50 sequence_beam_width=4. Plus a round-trip check: 50 REAL items' own
   embeddings (C1), BLEU and exact-token F1 against their 32-token text. Progress printed per batch of 25 targets per
   budget. Outputs: one jsonl row per target x budget with the text.
4. Local scoring (`scripts/review_far_inversion.py score`): re-embed each inverted text with the same gtr pipeline
   (CPU is enough for ~1,000 texts; or appended to stage 3 on GPU), h = cos(E(x), t); ceiling c; margin h - c;
   target discrimination h - mean_{j != k} cos(E(x), t_j) within domain and arm (as Section 3). Coherence gate exactly
   as A.4: `gibberish_rule` on `gibberish_stats`, then JUDGE1 (Haiku 5.5) >= 7, then JUDGE2 (gpt-6-luna via Codex)
   >= 7, subscription CLIs, prompt from `coherent.judge` with the domain noun. Because 32-token fragments are cut
   mid-sentence, the gate is calibrated first on the 32-token truncations of 50 random past items per domain (seed
   20261114 + 40) and on the 50 REAL targets' own texts.

**Outcomes and estimators.** Per domain, arm and budget: median h with bootstrap 95 % CI over targets (B = 10,000,
seed 20261114 + 41); median ceiling; share of targets whose best coherent inversion (best h over the three budgets
among gate-passing texts) reaches h >= c + 0.05 ("escape", the A.4 definition), with Clopper-Pearson 95 % interval;
the maximum h over all texts regardless of coherence (token-soup reach, as the fluent token search reported); median
target discrimination.

**Validity gates (fixed in advance; if a gate fails, the study is reported as uninformative for that domain and no
GAP/FAR verdict is drawn).**
- V-gate (coherence gate): >= 80 % of real 32-token truncations pass the gate.
- V0 (inverter works on this corpus): on REAL, median h >= 0.90 at C2, and >= 70 % of REAL targets get a
  gate-passing inversion with h > c (closer than the nearest visible non-duplicate item).

**Reading rule.**
- V1 GAP: "interior holes decode into new coherent text" if >= 50 % of GAP targets escape; this would replicate the
  seed-and-rewrite GAP row (19/20) with a second, neighbour-free decoder and supports moving it to the main text.
- V2 FAR (the claim at stake): supports "no text I found or constructed reaches a far point" if < 20 % of FAR targets
  escape (the A.4 strong-form threshold) in both domains. Counts against it if >= 20 % escape in either domain; counts
  strongly against if, in addition, the median h of coherent FAR inversions is >= 0.30 (a far point decoded into
  coherent text at a cosine an order of magnitude above the corpus ceiling).
- Token soup: if the maximum h over incoherent texts is far above the coherent maximum (difference >= 0.1), it is
  reported as the inversion analogue of the fluent-search result (reach only through incoherent text).

**Paper text.**
- Supports: Section 3 and Section 8 add: "A trained inverter (vec2text, gtr-t5-base, the closest supported encoder;
  targets rebuilt in that space by the same rules) reconstructs hidden real items (median cosine x) and decodes y of
  50 interior holes into coherent text beyond the corpus ceiling, but z of 50 far points (cosine w)." The claim stays,
  with the explicit scope "in gtr-t5-base, 32-token texts; no inverter exists for the paper's encoders".
- Counts against: the far-point claim in the abstract is restricted to the neighbour-prompted and search decoders, and
  the text reports that inversion reached coherent text near far points in the gtr space.
- Uninformative (a gate fails): the paper says plainly that the claim does not cover inversion, as M6 asks, and
  reports why the attempt was uninformative.

**Compute and cost.** Free Kaggle T4 x2, no paid API, no gateway. Estimates (by analogy to the repo's Kaggle embed
kernels and vec2text's per-step cost of one T5-base generate plus one gtr encode): embedding 140k 32-token texts
< 10 min; inversion of 300 targets: C0 and C1 a few minutes, C2 (50 steps, 16 hypotheses per target per step) about
30-45 min per domain on one T4, the two domains in parallel on the two GPUs; round-trip check and re-embedding minutes.
Total about 1.5-2 Kaggle GPU-hours including model downloads (internet on) and a 25-target smoke run first; budget 3 h,
inside one 12 h session. Coherence judging: up to ~1,100 texts (900 inversions + 100 calibration + REAL texts),
JUDGE2 only on JUDGE1 passes, through subscription CLIs (no API spend), about 1-2 h wall time. Local stage 2: one
20,000 x 72,000 x 768 product per arm and domain, about 1-2 min, peak < 1.5 GB.

Scripts: `kaggle/review-gtr-embed/embed.py`, `kaggle/review-gtr-invert/invert.py`, `scripts/review_far_inversion.py`
(`build`, `score`) -> `docs/prereg/review_far_inversion.json`. Private Kaggle dataset for the texts and targets
(corpus texts are public datasets; the decoded texts stay private as all run data).

---

## Summary of resource needs

| Study | Generation | GPU | Local runtime | Peak memory |
|---|---|---|---|---|
| R1 centring/whitening | none | none | 25-40 min | ~2 GB |
| R2 independent units | none | none | seconds | < 200 MB |
| R3 score channel | none | none | < 1 min | < 500 MB |
| R4 inversion (gtr-t5-base) | none paid; ~1,100 coherence judgements on subscription CLIs | Kaggle T4 x2, ~1.5-2 h (budget 3 h) | ~5 min local + 1-2 h judging | < 1.5 GB |

Infeasible as asked: an inverter for Qwen3 or BGE-M3 (none public; training one is days of GPU). R4 is the closest
feasible version and changes both the encoder and the text length (32 tokens).
