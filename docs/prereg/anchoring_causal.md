# Round 5, track B: causal test of neighbour anchoring

Pre-registered 2026-10-28, before any intervention decoding. No intervention outputs were inspected.
The existing grid is observational calibration, not a fresh confirmation sample. This experiment reuses its
saved target vectors; it tests a new intervention on those targets, not generalisation to new targets.
References: `docs/bounds.md` §3–§4; `docs/geometry.md` G5, §11–§12.

## Hypotheses and estimand

H_A: with target and score feedback fixed, changing the shown examples moves the decoded embedding toward
their centroid. The approximate law `E[e·v] ≈ alpha * (C_shown·v)` should transport to swapped examples,
for both `v=t` (real target) and `v=u` (donor target), with `E[e·C_shown] ≈ alpha`.
This is a prospective mechanistic hypothesis; the decomposition
`e·t = (e·C_shown)(C_shown·t) + r·t` is an identity and cannot itself be falsified.

H_K: the target-only predictor `kappa * ceiling(t)` is invariant to neighbour condition. A swap effect
rejects its condition-invariant extension; it does not invalidate the old descriptive fit or establish
an upper bound on all possible text. Kappa alone specifies no output direction: it cannot identify
similarity to a shown centroid or donor target. Those entries are deliberately not invented.

The unit of analysis is a target, paired across conditions within domain × generator × arm. Never pool
outputs, candidates, iterations, generators, or arms as independent replicates. The fixed target/donor
pool and fresh stateless CLI sessions make interference unlikely, but t intervals are a working
between-target sampling approximation, not an exact randomisation test or population-wide guarantee.

## Frozen targets and interventions

- Domains: Steam and arXiv. Qwen3-Embedding-0.6B, 1024 dimensions; existing adult-filtered past corpora.
- Targets: `runs/<domain>/pilot/targets.npz`, REAL/GAP/FAR, **k=1…20 inclusive**, 20 per arm/domain.
  k=0 smoke is excluded. No target construction or selection using new outcomes. No replacement of failures.
- Calibration: successful historical `grid-gpt-6-luna` and `grid-claude-sonnet-5-5` items with **k=21…49**.
  Alpha is the mean of `e·C_true` in the same domain/generator/arm; kappa is the median
  `landing/visible_ceiling` on REAL only in the same domain/generator. Neither is re-estimated after the run.
- TRUE: current eight nearest visible neighbours, current prompt, current real-target similarity labels.
- SWAP: a seeded single-cycle permutation maps each k to another k in the same arm/domain. Every donor is
  used exactly once; the same mapping is used by both generators. Seeds are 20261028 + 100*domain_index
  + arm_index (Steam/arXiv; REAL/GAP/FAR). Use the donor's original eight visible neighbours. Verify that
  none is hidden for the recipient; **all six frozen mappings pass this check**. The decoder additionally
  unions recipient/donor masks as a guard. Keep neighbour order from donor retrieval; score all displayed
  neighbours against **the real recipient target**, not the donor. All candidate selection and feedback
  also use the real target. Centroid scores are not matched: moving their target score is the intervention.
- NONE: no neighbour block, and no references to listed examples in score feedback. The initial request
  contains only the location/task instruction; subsequent rounds supply the real-target scalar score.
  This is not `--feedback none` (which disables iteration).
- Random-corpus condition is omitted to keep budget and the confirmatory family fixed.

SWAP retains the legacy claim that the listed examples are closest. This intentional misdirection holds
framing constant: only the eight displayed entries (including their correct recipient scores) change.
The causal estimand is the whole example block, not text semantics isolated from numeric score labels.
NONE necessarily deletes example-dependent wording. System instruction, schema, rendering, best-of search,
feedback precision and task wording otherwise remain fixed. Candidate responses will naturally diverge
in real runs; prompt equivalence is checked under identical mocked responses.

Use only `claude:claude-sonnet-5-5` and `codex:gpt-6-luna` through the existing isolated subscription CLI
harness. **No gateway.** Four score-feedback rounds × three candidates = 12 calls per target/condition;
retain the best of all four round winners against the real target. Fresh calls/sessions as in the grid,
no controllable generation seed. Serial candidates preserve the algorithm and budget while limiting RAM.
Both Python workers and ONNX threads are bounded: one target worker per process, one candidate worker,
`OMP_NUM_THREADS=4` (also passed as LocalQwen's thread count), two decoder processes maximum.

Condition order is pre-fixed and balanced across the six domain/arm blocks, with one queue per CLI:
luna REAL/GAP/FAR uses true-swap-none / swap-none-true / none-true-swap in each domain; Sonnet shifts
these orders by one. k runs in ascending order. Each queue completes Steam before arXiv. This balances
position over blocks but does not remove within-cell temporal drift; there is no adaptive scheduling
based on scores. Do not change model versions or account settings between conditions.

## Numerical predictions (frozen before any intervention run)

All entries are **means over the 20 selected targets**, cosine units, rounded here to three decimals.
The code computes per-target predictions at full precision from the hashed inputs and frozen coefficients.
`C` is the normalised mean of eight **unit** corpus embeddings. Retrieval/prompt scores retain the original
float32 arrays for byte compatibility; analysis renormalises corpus, target and output vectors.
`u` is the assigned donor target even for TRUE/NONE. The same donor permutation is used in all comparisons.

For TRUE/SWAP, the prediction sets the mean residual projection to zero. The old law is approximate;
its transfer to incompatible examples is exactly what is at risk. NONE has no shown centroid, so the
anchoring law is **undefined**, not zero. Its numerical entries below are an auxiliary no-information
prior: mean unit embedding of the same generator's historical PLAIN k=21…49, **without renormalising
that mean**, projected onto t and u. This does not model the gain from best-of-12 score optimisation;
NONE is an ablation, not a confirmatory test of the alpha law.

| Domain | CLI | Arm | Condition | Predicted e·t | Predicted e·C_shown | Predicted e·u | κ·ceiling(t), all conditions |
|---|---|---|---|---:|---:|---:|---:|
| steam | luna | REAL | true | 0.645 | 0.776 | 0.376 | 0.676 |
| steam | luna | REAL | swap | 0.381 | 0.776 | 0.645 | 0.676 |
| steam | luna | REAL | none | 0.287 | — | 0.287 | 0.676 |
| steam | luna | GAP | true | 0.479 | 0.722 | 0.332 | 0.481 |
| steam | luna | GAP | swap | 0.327 | 0.722 | 0.479 | 0.481 |
| steam | luna | GAP | none | 0.243 | — | 0.243 | 0.481 |
| steam | luna | FAR | true | 0.032 | 0.629 | -0.048 | 0.036 |
| steam | luna | FAR | swap | -0.049 | 0.629 | 0.032 | 0.036 |
| steam | luna | FAR | none | -0.045 | — | -0.045 | 0.036 |
| steam | sonnet | REAL | true | 0.720 | 0.866 | 0.420 | 0.733 |
| steam | sonnet | REAL | swap | 0.425 | 0.866 | 0.720 | 0.733 |
| steam | sonnet | REAL | none | 0.339 | — | 0.339 | 0.733 |
| steam | sonnet | GAP | true | 0.540 | 0.814 | 0.374 | 0.521 |
| steam | sonnet | GAP | swap | 0.368 | 0.814 | 0.540 | 0.521 |
| steam | sonnet | GAP | none | 0.261 | — | 0.261 | 0.521 |
| steam | sonnet | FAR | true | 0.037 | 0.719 | -0.055 | 0.039 |
| steam | sonnet | FAR | swap | -0.056 | 0.719 | 0.037 | 0.039 |
| steam | sonnet | FAR | none | -0.045 | — | -0.045 | 0.039 |
| arxiv | luna | REAL | true | 0.693 | 0.829 | 0.304 | 0.728 |
| arxiv | luna | REAL | swap | 0.302 | 0.829 | 0.693 | 0.728 |
| arxiv | luna | REAL | none | 0.280 | — | 0.280 | 0.728 |
| arxiv | luna | GAP | true | 0.429 | 0.739 | 0.238 | 0.426 |
| arxiv | luna | GAP | swap | 0.253 | 0.739 | 0.429 | 0.426 |
| arxiv | luna | GAP | none | 0.252 | — | 0.252 | 0.426 |
| arxiv | luna | FAR | true | 0.052 | 0.665 | -0.045 | 0.053 |
| arxiv | luna | FAR | swap | -0.050 | 0.665 | 0.052 | 0.053 |
| arxiv | luna | FAR | none | -0.052 | — | -0.052 | 0.053 |
| arxiv | sonnet | REAL | true | 0.736 | 0.881 | 0.323 | 0.775 |
| arxiv | sonnet | REAL | swap | 0.321 | 0.881 | 0.736 | 0.775 |
| arxiv | sonnet | REAL | none | 0.274 | — | 0.274 | 0.775 |
| arxiv | sonnet | GAP | true | 0.471 | 0.812 | 0.262 | 0.454 |
| arxiv | sonnet | GAP | swap | 0.278 | 0.812 | 0.471 | 0.454 |
| arxiv | sonnet | GAP | none | 0.227 | — | 0.227 | 0.454 |
| arxiv | sonnet | FAR | true | 0.060 | 0.776 | -0.052 | 0.057 |
| arxiv | sonnet | FAR | swap | -0.058 | 0.776 | 0.060 | 0.057 |
| arxiv | sonnet | FAR | none | -0.045 | — | -0.045 | 0.057 |

The κ column is invariant within a cell; its predictions for e·C_shown and e·u are **not identified**.
In particular, H_A predicts positive donor-target landing under SWAP, even though feedback rewards t.
For FAR this is a sign reversal in real-target landing, not simply staying inside the domain's common cap.

## Frozen decision rules

For each of 12 domain × generator × arm cells, with C_i=true centroid and C_j=donor centroid:

- Target loss: `D_i = e_true·t_i − e_swap·t_i`.
- Donor attraction: `A_i = (e_swap − e_true)·(C_j − C_i)` (unnormalised centroid contrast).
  Measuring both centroids controls for generic shared domain direction better than e_swap·C_j alone.
- Compute paired means and t intervals, `mean ± t_(1−0.05/24,19) * sd/sqrt(20)`.
  Each directional primary test is one-sided at **0.05/24 = 0.0020833333**. Bonferroni covers the 24
  positive primary tests without assuming their independence. The reported two-sided intervals have
  99.5833% marginal coverage; no simultaneous two-sided/opposite-direction family claim is made.
- **Causal anchoring supported in a cell** iff mean D ≥ **0.04** for REAL/GAP or **0.02** for FAR,
  mean A ≥ **0.05**, and both lower interval endpoints are > 0.
- **Equivalent within margins; anchoring weakened** iff the entire D interval lies strictly within
  ±0.04 (REAL/GAP) or ±0.02 (FAR), and the entire A interval lies strictly within ±0.05.
  Failure to reject zero alone is never equivalence.
- If neither rule above holds and either upper endpoint < 0: **opposite-direction effect** (descriptive).
  Otherwise **inconclusive/mixed**. Thresholds apply to means, not counts of individual successes.
- **Global confirmation** requires all 4 FAR cells, at least 3/4 GAP cells and at least 3/4 REAL cells
  to support causal anchoring. Otherwise report cell-level results without a global confirmation claim.
- Require **20/20 valid, paired targets in all three conditions** in a cell for any confirmatory verdict.
  Invalid JSON/refusals consume candidate budget; final best=null is a terminal failure, retained and
  never rerun to seek success. Exceptions leave the item incomplete and can be retried on resume;
  report them and any extra calls. Missing/failed cells produce no confirmation, not a null effect.
  Duplicate rows, wrong donors, input hashes, targets, config, budgets, scores or neighbour IDs are hard errors.

Quantitative transport is separately labelled (descriptive, not another significance family): for each
TRUE/SWAP cell, MAE of predicted vs actual **both e·t and e·u** ≤ **0.08 REAL / 0.05 GAP / 0.025 FAR**,
and absolute mean error of e·C_shown ≤ **0.10**. Report alpha-predictor MAE and κ-predictor MAE,
signed exact residual `e·t − (e·C_shown)(C_shown·t)`, and all per-item projections even on a failed fit.
Causal evidence can hold while the simple zero-residual transport fails. NONE's real-target loss vs TRUE,
its e·u and similarities to both unshown centroids are descriptive. No novelty or coherence inference is
made: this run has no judges. No early stopping on significance, no post-run refitting or target exclusions.

## Power from existing item spread

The following uses actual historical k=1…20 outputs (not the calibration subset) to estimate SD of
`Y=e_true·t` and `V=e_true·(C_j−C_i)` per cell. The paired-SD planning assumption is `sqrt(2)*SD_old`
(equal marginal variance, zero cross-condition correlation); the conservative sensitivity uses `2*SD_old`.
This is a planning assumption: new SWAP variability and cross-condition covariance are unobserved.
An intervention may change variance, and shared donor geometry can induce dependence.

The mean effect comes from the **frozen anchoring predictions**, not simulated observed effects.
MDE80 ≈ `(t_critical + z_.8)*SD_pair/sqrt(20)`; significance power is also computed with a noncentral t.
The joint practical-threshold/significance power uses a normal plug-in approximation,
`P(mean > max(practical_threshold, t_critical*SE))`. It is not a guarantee. Columns below show the
historical SD and MDE80 for D / A; every full-law effect has estimated joint power >99.9% even under
the 2*SD_old sensitivity. Small real effects near the 0.04/0.02/0.05 margins can be underpowered.

| Domain | CLI | Arm | Predicted D / A | Historical SD(Y) / SD(V) | MDE80 D / A |
|---|---|---|---:|---:|---:|
| steam | luna | REAL | 0.264 / 0.597 | 0.084 / 0.105 | 0.109 / 0.135 |
| steam | luna | GAP | 0.152 / 0.357 | 0.036 / 0.056 | 0.047 / 0.073 |
| steam | luna | FAR | 0.081 / 0.407 | 0.020 / 0.085 | 0.026 / 0.110 |
| steam | sonnet | REAL | 0.295 / 0.666 | 0.063 / 0.111 | 0.081 / 0.143 |
| steam | sonnet | GAP | 0.171 / 0.403 | 0.037 / 0.058 | 0.048 / 0.075 |
| steam | sonnet | FAR | 0.093 / 0.466 | 0.022 / 0.078 | 0.028 / 0.101 |
| arxiv | luna | REAL | 0.391 / 0.907 | 0.056 / 0.070 | 0.072 / 0.090 |
| arxiv | luna | GAP | 0.176 / 0.537 | 0.040 / 0.086 | 0.052 / 0.112 |
| arxiv | luna | FAR | 0.102 / 0.563 | 0.024 / 0.101 | 0.031 / 0.130 |
| arxiv | sonnet | REAL | 0.416 / 0.964 | 0.050 / 0.086 | 0.065 / 0.112 |
| arxiv | sonnet | GAP | 0.193 / 0.590 | 0.033 / 0.074 | 0.043 / 0.096 |
| arxiv | sonnet | FAR | 0.119 / 0.656 | 0.020 / 0.090 | 0.025 / 0.117 |

## Budget, operations and audit

120 distinct targets × 2 CLIs × 3 conditions = **720 target-runs**, each 4 × 3 = 12 candidate calls:
**8,640 planned CLI calls total; 4,320 Claude and 4,320 Codex**. No gateway calls, judges, pitches,
new embeddings of the corpus, or GPU work. At most 2 simultaneous decoder processes and 2 active CLI
calls globally within this runner. A mkdir lock prevents a second copy of this runner from multiplying
concurrency; do not run another decode job alongside it. Model and corpus are loaded once per 20-target block.

Wall-time planning: **about 24–48 hours** if each sequential CLI-call-plus-local-embedding averages
20–40 seconds (4,320 × 20–40 s on the slower queue), plus model loads; **72 hours at 60 s/call**.
This is a planning envelope, not a measured throughput benchmark. Earlier seed-rewrite tuning took
7–10 min / 100 calls with four workers (`fluent_search.md` rev 3), suggesting 17–24 s/call there,
but prompts, machine load and quota behaviour differ. Quota waits can extend runtime substantially.
The existing CLI harness retries rate limits/network failures (up to seven underlying invocations per
candidate), so 8,640 counts logical candidate calls, not a hard cap on process attempts. Rerunning an
interrupted target also spends extra calls; all completed rows are skipped on resume. There is no
additional result-driven retry. Logs append; final best=null rows are not retried.

Run from the repository root:

```sh
bash scripts/anchoring_causal.sh
```

The runner exports OMP_NUM_THREADS=4, checks frozen inputs, uses the two exact CLI models, and automatically
analyses on completion. Resume with the identical command. Outputs stay in the external experiment-data
root, `runs/<domain>/anchoring-causal/<luna|sonnet>/<condition>/<arm>/`; each has `items.jsonl`, a copied
`targets.npz`, `intervention.json`, and append-only `run.log`. The config check rejects mixed conditions,
generators, target files or selection. Analysis writes `runs/anchoring-causal-analysis/{items.csv,results.json}`
and exits 2 while incomplete. Do not edit or add rows to the frozen historical source files during the run.

Offline-only commands (no CLI, gateway or ONNX model invocation):

```sh
export OMP_NUM_THREADS=4
uv run python scripts/anchoring_causal.py design
uv run python scripts/anchoring_causal.py dry-check --out /tmp/anchoring-causal-prompts
uv run python scripts/anchoring_causal.py analyze --out /tmp/anchoring-causal-analysis
```

Dry audit: REAL k=1 in **both** domains, 4 rounds with deterministic fake responses/embeddings.
Compare omitted-default and explicit TRUE to the literal legacy templates at every round; compare SWAP
with TRUE after replacing only the neighbour listing; compare NONE after removing only example-dependent
text. Assert all 12 mocked candidate calls score the real target. Full flattened prompts and unified diffs
are saved outside the repo. These are prompt construction checks, not generated experiment outcomes.

Executed 2026-10-28: **PASS** Steam REAL-1 → donor 18 and arXiv REAL-1 → donor 9, all four mocked
rounds. Default/TRUE prompts are byte-identical; SWAP changes only the listing; NONE removes only the
example block/references. A separate comparison with the actual HEAD decoder passed for all three
legacy feedback modes (score/residual/none). Full prompts and diffs: `/tmp/anchoring-causal-prompts/`.
Resume/configuration checks, the 720-row mock analysis, missing/wrong-donor rejection, and the macOS
Bash 3.2 two-process scheduler check passed. **Zero LLM calls; no intervention outcomes generated.**

## Machine-readable frozen inputs and coefficients

This block is read by the analysis and runner preflight. SHA-256 checks cover corpus metadata, embeddings,
saved pilot targets and historical CLI outputs. Donor IDs and full-precision alpha/kappa values are frozen.
No coefficient is estimated from intervention outcomes; NONE uses only the hashed historical PLAIN vectors.

<!-- frozen-manifest -->
```json
{
  "version": 1,
  "ks": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20
  ],
  "calibration_ks": [
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49
  ],
  "generators": {
    "luna": "codex:gpt-6-luna",
    "sonnet": "claude:claude-sonnet-5-5"
  },
  "conditions": [
    "true",
    "swap",
    "none"
  ],
  "alpha_per_endpoint": 0.0020833333333333333,
  "anchor_min": 0.05,
  "drop_min": {
    "REAL": 0.04,
    "GAP": 0.04,
    "FAR": 0.02
  },
  "mae_max": {
    "REAL": 0.08,
    "GAP": 0.05,
    "FAR": 0.025
  },
  "sources": {
    "prepared/steam/corpus.parquet": "fa146deb033b671e3f0a44f81df90490ddb09514c4996352e34b6113aa14a43f",
    "prepared/steam/emb_qwen3e06.npy": "57f525cec52efae349160d0d6800bda3ec2b7053172ca956068555949b5a8652",
    "runs/steam/pilot/targets.npz": "023a601baba93494b1a51d5262f1a02e27a90fb3b941a56d46c439a8819dbdd5",
    "runs/steam/grid-gpt-6-luna/items.jsonl": "43ab5990f98c48c14e02ce2defeb41f9d94ca756336a5996c87fb1e7b5073a84",
    "runs/steam/grid-claude-sonnet-5-5/items.jsonl": "f5f24d759fce12d13f2c578b377a8dc8fcf28d4d02136e66feef394a88308c04",
    "prepared/arxiv/corpus.parquet": "3f82b9d9af1ff6936fc2c7c104c2d3ce98b4e7771005a001b5fb886dc48e8b65",
    "prepared/arxiv/emb_qwen3e06.npy": "923ae3b6f6aec0044535eecc73a1a36af2870c8b8caa36815463857f5244023d",
    "runs/arxiv/pilot/targets.npz": "05a617fd7caed3de7c0cb562c591e73bdacbaae057af421c894f090a396dfa72",
    "runs/arxiv/grid-gpt-6-luna/items.jsonl": "63431b7cc73942f22e30e36e431aa3111e4b7214a72418bf80ff8506c1972712",
    "runs/arxiv/grid-claude-sonnet-5-5/items.jsonl": "e884ee21dd759eb8bdac6bb154ffbd788e7743aeeb9dd77f32bc076883b77146"
  },
  "donors": {
    "steam/REAL": {
      "2": 16,
      "16": 9,
      "9": 14,
      "14": 15,
      "15": 20,
      "20": 6,
      "6": 4,
      "4": 19,
      "19": 7,
      "7": 12,
      "12": 1,
      "1": 18,
      "18": 11,
      "11": 10,
      "10": 13,
      "13": 5,
      "5": 17,
      "17": 3,
      "3": 8,
      "8": 2
    },
    "steam/GAP": {
      "9": 7,
      "7": 20,
      "20": 4,
      "4": 11,
      "11": 16,
      "16": 12,
      "12": 14,
      "14": 17,
      "17": 3,
      "3": 18,
      "18": 15,
      "15": 10,
      "10": 2,
      "2": 13,
      "13": 8,
      "8": 19,
      "19": 1,
      "1": 6,
      "6": 5,
      "5": 9
    },
    "steam/FAR": {
      "18": 2,
      "2": 10,
      "10": 6,
      "6": 1,
      "1": 9,
      "9": 15,
      "15": 3,
      "3": 7,
      "7": 11,
      "11": 17,
      "17": 8,
      "8": 20,
      "20": 12,
      "12": 16,
      "16": 5,
      "5": 19,
      "19": 14,
      "14": 13,
      "13": 4,
      "4": 18
    },
    "arxiv/REAL": {
      "13": 18,
      "18": 11,
      "11": 14,
      "14": 16,
      "16": 1,
      "1": 9,
      "9": 4,
      "4": 17,
      "17": 2,
      "2": 12,
      "12": 20,
      "20": 6,
      "6": 19,
      "19": 8,
      "8": 10,
      "10": 15,
      "15": 7,
      "7": 3,
      "3": 5,
      "5": 13
    },
    "arxiv/GAP": {
      "6": 9,
      "9": 2,
      "2": 3,
      "3": 13,
      "13": 17,
      "17": 11,
      "11": 7,
      "7": 15,
      "15": 16,
      "16": 1,
      "1": 20,
      "20": 10,
      "10": 5,
      "5": 18,
      "18": 12,
      "12": 4,
      "4": 8,
      "8": 19,
      "19": 14,
      "14": 6
    },
    "arxiv/FAR": {
      "12": 8,
      "8": 13,
      "13": 16,
      "16": 14,
      "14": 2,
      "2": 5,
      "5": 15,
      "15": 6,
      "6": 11,
      "11": 20,
      "20": 4,
      "4": 17,
      "17": 1,
      "1": 3,
      "3": 18,
      "18": 19,
      "19": 7,
      "7": 9,
      "9": 10,
      "10": 12
    }
  },
  "calibration": {
    "steam/luna/REAL": {
      "alpha": 0.7762638169854896,
      "kappa": 0.9070209233264637
    },
    "steam/luna/GAP": {
      "alpha": 0.7223144365001687,
      "kappa": 0.9070209233264637
    },
    "steam/luna/FAR": {
      "alpha": 0.62864592614239,
      "kappa": 0.9070209233264637
    },
    "steam/sonnet/REAL": {
      "alpha": 0.8655547111108295,
      "kappa": 0.9831461169727369
    },
    "steam/sonnet/GAP": {
      "alpha": 0.8139492941377722,
      "kappa": 0.9831461169727369
    },
    "steam/sonnet/FAR": {
      "alpha": 0.7185986653298803,
      "kappa": 0.9831461169727369
    },
    "arxiv/luna/REAL": {
      "alpha": 0.8291462043939586,
      "kappa": 0.9998115305473817
    },
    "arxiv/luna/GAP": {
      "alpha": 0.7389763191883968,
      "kappa": 0.9998115305473817
    },
    "arxiv/luna/FAR": {
      "alpha": 0.6651958765187828,
      "kappa": 0.9998115305473817
    },
    "arxiv/sonnet/REAL": {
      "alpha": 0.8813518234292665,
      "kappa": 1.0649035470437545
    },
    "arxiv/sonnet/GAP": {
      "alpha": 0.8117061922544422,
      "kappa": 1.0649035470437545
    },
    "arxiv/sonnet/FAR": {
      "alpha": 0.7758632377114777,
      "kappa": 1.0649035470437545
    }
  }
}
```

## Result (run completed 2026-10-29 06:5x UTC; analysed with `scripts/anchoring_causal.py analyze`, frozen rules above)

All 36 cells complete, 20/20 valid paired targets in every condition (720 target-runs, CLIs only, no gateway).
Full machine output: [`anchoring_causal_result.json`](anchoring_causal_result.json).

**Global confirmation: cross-domain/model anchoring supported** (REAL 4/4, GAP 4/4, FAR 4/4 cells).

| Domain | Generator | Arm | Target loss D (true − swap) | Donor attraction A | e·t true / swap / none | Transport | Verdict |
|---|---|---|---|---|---|---|---|
| steam | luna | REAL | +0.265 [+0.161, +0.368] | 0.55 [0.42, 0.68] | 0.691 / 0.426 / 0.383 | pass | causal anchoring supported |
| steam | sonnet | REAL | +0.311 [+0.222, +0.399] | 0.65 [0.50, 0.79] | 0.765 / 0.454 / 0.465 | pass | causal anchoring supported |
| steam | luna | GAP | +0.142 [+0.085, +0.198] | 0.35 [0.28, 0.42] | 0.470 / 0.328 / 0.329 | fail | causal anchoring supported |
| steam | sonnet | GAP | +0.133 [+0.084, +0.182] | 0.41 [0.32, 0.49] | 0.526 / 0.393 / 0.373 | fail | causal anchoring supported |
| steam | luna | FAR | +0.038 [+0.019, +0.057] | 0.30 [0.21, 0.39] | 0.019 / -0.019 / -0.007 | fail | causal anchoring supported |
| steam | sonnet | FAR | +0.057 [+0.041, +0.072] | 0.41 [0.30, 0.51] | 0.035 / -0.022 / -0.019 | fail | causal anchoring supported |
| arxiv | luna | REAL | +0.400 [+0.333, +0.467] | 0.81 [0.68, 0.93] | 0.758 / 0.359 / 0.362 | pass | causal anchoring supported |
| arxiv | sonnet | REAL | +0.479 [+0.411, +0.547] | 0.99 [0.86, 1.12] | 0.809 / 0.330 / 0.377 | pass | causal anchoring supported |
| arxiv | luna | GAP | +0.128 [+0.080, +0.175] | 0.46 [0.37, 0.55] | 0.442 / 0.314 / 0.306 | fail | causal anchoring supported |
| arxiv | sonnet | GAP | +0.184 [+0.140, +0.227] | 0.64 [0.52, 0.76] | 0.486 / 0.302 / 0.283 | pass | causal anchoring supported |
| arxiv | luna | FAR | +0.054 [+0.035, +0.073] | 0.47 [0.35, 0.60] | 0.038 / -0.016 / -0.015 | fail | causal anchoring supported |
| arxiv | sonnet | FAR | +0.083 [+0.066, +0.100] | 0.64 [0.52, 0.77] | 0.066 / -0.017 / -0.012 | fail | causal anchoring supported |

Reading. Changing only the shown neighbours, with target and cosine feedback fixed, moves the decoded text
toward the shown neighbours' centroid in every cell. In FAR the whole small landing on the target
(0.02–0.07 with the true neighbours) disappears with swapped or no neighbours (−0.02 to −0.01): the
feedback alone does not move the text toward a far point; what landing there is comes with the
neighbours. The direction of the anchoring law is therefore causal, not only a fitted description. Its
simple quantitative form (zero-residual transport within the frozen MAE margins) holds in all REAL cells
and arXiv-Sonnet GAP but fails in 7/12 cells (all FAR, 3 of 4 GAP): the effect is causal, the
one-coefficient law is only approximate there. No novelty or
coherence inference is made (no judges in this run).
