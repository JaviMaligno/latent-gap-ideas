# Continuing from a given paper: the geometry of the next step

Pre-registered 2026-10-30. Only a 40-context fetch pilot has been run (coverage counts below); no follow-up
was embedded and no idea was generated before this file was committed.

## Why

Research rarely starts from an empty point in a map; it starts from a concrete paper or result and asks what
comes next. Prior work generates continuations from a context and scores them against later papers
(SciMON, ACL 2024; PreScience, arXiv 2602.20459; HindSight, arXiv 2603.15164), but none measures the *step*:
how far and in which direction real follow-ups move from their context, and whether an LLM's continuation
takes the same step. Our earlier results predict that it does not: an LLM decoder stays anchored to what it is
shown (`anchoring_causal.md`), and real novelty grows where there is momentum (`where_future_lands.md`).

## Data

- **Contexts:** past arXiv items of our corpus dated 2021–2022 (adult rows dropped), sampled with seed
  20261030 in batches until **150 contexts with ≥ 5 follow-ups** are collected (pilot: 17 of 40 had ≥ 3,
  10 of 40 had ≥ 10; expected sample ~1,000 contexts). The selection conditions on having follow-ups, which is
  the definition of the task; it is stated as such.
- **Follow-ups:** papers that cite the context according to Semantic Scholar, that have an arXiv id and were
  published on or after 2024-01-01 (after the generator's training cutoff); titles and abstracts from the arXiv
  API; at most 50 per context (seeded sample) (`scripts/followups_fetch.py`). Citing is the operational
  definition of "continuing"; it includes citations that are only background, a stated limitation.
  *Amendment before any measurement (2026-10-30):* the first version used OpenAlex; its keyless quota
  (~1,000 credits/day) stopped the full fetch, so the source moved to Semantic Scholar + arXiv. This restricts
  follow-ups to arXiv papers, the same domain as the corpus. Second pilot: 4 of 15 contexts with >= 5 follow-ups.
- **Distractors:** for each context, the future arXiv items of our corpus (2023-01-01 onwards) in the same
  primary category, excluding its follow-ups.
- **Embedding:** Qwen3-Embedding-0.6B locally (ONNX, same normalisation as the corpus). BGE-M3 replication of the
  geometry part is optional (it needs the Kaggle job) and is labelled as such if done.

## Part A: where real follow-ups land (free, local)

For context c and each follow-up f: `step = cos(c, f)`. Displacement `d = f − c`, projected (cosine) onto three
past-only directions computed from the 50 nearest past items of c:
- **momentum**: mean of those dated in the last two past years minus mean of the older ones;
- **hole**: from c towards the most isolated of 2,000 draws from the local Gaussian of those 50 items;
- **centre**: towards μ̂, the normalised corpus mean (negative = towards the edge of the cap).

- **A1**: the median step of real follow-ups is reported with its distribution (no directional prediction).
- **A2**: follow-up displacements align more with momentum than with hole: per context, mean projection on
  momentum minus mean projection on hole > 0, Wilcoxon over contexts, prediction positive.
- **A3** (two-sided): the mean projection on centre, reported with its sign.

## Part B: does an LLM take the same step (paid; runs only after the owner approves the estimate)

Generator: **gpt-4o-2024-08-06** (training cutoff Oct 2023) through the gateway alias already used for the
anticipation test, temperature 1, one idea per call, written as a title and abstract.

Arms, 5 ideas per context per arm:
- **CTX**: the context's title and abstract only. "Propose the paper that should come next."
- **CTX+NB**: plus the 8 nearest past items, as in the grid decoder.
- **CTX+MEM**: as CTX, sequential, seeing the titles it already proposed (the PLAIND fix).
- **CTX+MOM**: plus the 8 most recent items among the context's 50 nearest past neighbours ("where this area is moving").

Measures per idea: `step_idea = cos(c, idea)`; `fit = max cos(idea, follow-ups of c)`; **hit** = the idea's
nearest neighbour in (follow-ups of c ∪ distractors) is a follow-up.

- **B1 (anchoring)**: in CTX and CTX+NB, ideas stay closer to the context than real follow-ups: median
  `step_idea` > median `step` of the follow-ups, paired by context. Prediction: holds in both arms.
- **B2**: CTX+NB ideas are closer to the shown neighbours' centroid than to the context's follow-ups'
  centroid (prediction: holds), the anchoring law transported to continuation.
- **B3**: CTX+MEM ideas are more diverse (mean pairwise cos within context lower) than CTX. Prediction: holds.
- **B4 (the useful one)**: CTX+MOM has the highest hit rate and fit. Prediction: CTX+MOM > CTX on hit rate
  (paired over contexts, McNemar on context-level majority, plus a bootstrap CI on the difference).
  Directional, but stated as uncertain: a failure is reported as such.

## Analysis rules

Unit = context; ideas within a context are averaged before testing. Holm within {A2, B1, B2, B3, B4}.
Bootstrap 2,000 resamples over contexts, seed 20261030. No parameter changes after the first result; anything
else goes to an "exploratory" section.

## Cost and resources

- Fetch: OpenAlex free API, ~1,000 contexts, ~30–60 min, no money.
- Embedding: local CPU, ~6,000 follow-ups + 3,000 ideas, ~20–40 min.
- Generation (Part B, paid spend on Azure through an API gateway): 150 contexts × 4 arms × 5 ideas
  = 3,000 calls, ~700 input + ~300 output tokens each ≈ 2.1M input + 0.9M output tokens. At gpt-4o-2024-08-06
  list prices ($2.50 / $10 per M) ≈ **$5 + $9 = ~$14 USD in one day** (≈ 21 AUD at ~1.5 AUD/USD), ~1 h wall
  time with 8 workers. Logged in `ledger.jsonl` as the earlier runs.
