# Reachable text, decoder budgets, and the FAR conjecture

2026-10-24. Follow-up to [bounds.md](bounds.md), in the requested order. No LLM calls,
no arXiv decodes, and **no empirical N × selection-fraction × embedding-space battery**.
New scripts live in `scripts/bounds/`; generated data stay in the external data directory.

## 1. Reachable support: a model is possible, but search gives lower bounds

**DEFINITION.** Fix the encoder, tokenizer, document formatting and EOS policy in
`src/latent_gap/embedder.py`. Let S be the finite set of accepted token strings of total
length ≤512, including the terminal EOS (thus ≤511 content tokens under this contract).
Exclude inputs with undefined/zero embeddings. With unit target t, define

\[
h_{any}(t)=\max_{s\in S}\langle enc(s),t\rangle,\qquad
h_{coh}(t)=\max_{s\in S:J_D(s)=1}\langle enc(s),t\rangle.
\]

Here J_D is a **specified judgment rule on the exact text**, not a property inferred
from cosine. Use two independent, blinded human raters, both accepting that the string
is an intelligible description of a game / research study in the domain. They need not
accept its novelty, factual claims or commercial/scientific merit. Resolve disagreement
by excluding it from the accepted set. Fix this operational meaning before rating.
For an empty accepted search set, report “unmeasured”, not zero. Canonical round-trippable text
is a subset of any larger raw-token input space, so its witnesses lower-bound that too.

**THEOREM.** For any finite search set W, its maximum score lower-bounds h_any, and the
maximum among accepted witnesses lower-bounds h_coh. Also h_coh ≤ h_any ≤ 1. Adaptive
search, multiple restarts and target selection do not invalidate an exhibited witness.
They do invalidate interpreting the best found score as an upper bound. Numerical
scores below are empirical floating-point witnesses, not interval-arithmetic certificates.

**Measurement implemented.** `support_probe.py` packages all 50 FAR and 50 GAP targets
in each of Steam and arXiv, with three nearest corpus texts per target plus the available recorded Steam decoder
winners as warm starts. ArXiv decoder files are never read. The private
Kaggle input file is `runs/support-search/support_inputs.json`. The kernel in
`kaggle/support-search/` performs gradient-guided discrete token replacement, scoring
actual strings after decode/re-tokenize, with the best available text start and a random-token
restart. A separate sentence-deletion branch retains three high-scoring candidate texts
for coherence review even if unrestricted token search beats them. It uses two separate model processes on **both T4s**, prints every block,
checkpoints each target, allocates about 45 seconds per target, and stops workers at
110 minutes overall. Search uses ≤128 tokens; full seed evaluation uses ≤512. Therefore
it explores a strict subset of S. Time limits may produce partial coverage, which is
reported rather than silently imputed. No language generator is invoked.

Gradient steps use a frozen Qwen3-Embedding-0.6B in fp16, last-token pooling and explicit
EOS. At each step, gradients propose one-token replacements at eight positions; actual
forward scores decide acceptance. This measures encoder reachability, not whether a
language decoder would produce the resulting string. The kernel has been syntax-checked,
**not GPU-executed here**; its runtime ceiling is a watchdog, not a throughput benchmark.

Run `verify_support.py` on downloaded witness JSONL files. Its default five-minute
local budget is resumable through an encoder-fingerprinted embedding cache; rerun until
all witnesses are verified, or set `--minutes`. Partial results are still lower bounds. It re-embeds saved text and
seeds with the canonical ONNX model, creates a shuffled `coherence_blind.csv` (no arm or
score), and joins completed ratings by domain and SHA-256. It writes per-target `h_any_lower` and
`h_coh_lower`; the latter stays missing without two accepting votes. Coherent seed
texts can certify h_coh even if the optimized string is nonsense. Rate the exact
truncated string, not an untruncated source or the pilot's separately normalized pitch.

**EMPIRICAL probe, fixed target k=0 in each stratum.** Nine short/general/repetitive
strings plus each nearest text and two 24-word truncations were scored locally. All
four winners were the original nearest corpus text; the cheap search found no gain.
This is deliberately a tiny probe, not an estimate of support.

| Domain / arm | Stored seed score | ONNX h_any witness | Accepted h_coh witness |
|---|---:|---:|---|
| Steam FAR | .03042 | .03046 | Pending exact-text ratings |
| Steam GAP | .51568 | .51558 | Pending exact-text ratings |
| arXiv FAR | .04328 | .04333 | Pending exact-text ratings |
| arXiv GAP | .40838 | .40842 | Pending exact-text ratings |

The <.00011 differences are encoder/storage numerical differences, not improvements.
Witnesses, token counts and all probe scores are in `runs/support-search/local_probe.json`.

**Upper-bound audit.** The only unconditional numerical upper bound established here is
**1.000**. Final L2 normalization supplies it. Last-token pooling merely selects a hidden
vector; attention's convex weights do not constrain the full output to a convex hull of
input embeddings once residuals, signed projections and MLPs are included. An invertible
diagonal final RMSNorm followed by L2 normalization can preserve every direction if its
input is unconstrained. Thus these architectural facts do not establish the corpus cap.
A useful Lipschitz certificate would need certified layer constants, a lower bound on
pre-normalization norm, and a sufficiently fine cover of **all** token sequences; none
has been computed. Enumerating the finite input space is a legitimate exact model, but
not feasible here. Failure of a two-hour search is not evidence for a hard .05 ceiling.

### Result of the GPU search (Kaggle 2×T4, run 2026-10-24; all 200 witnesses re-embedded with the local ONNX model)

| Domain / arm | Best corpus text | Best sentence-trimmed real text (coherent candidate) | Token search, ONNX-verified: median [min, max] |
|---|---:|---:|---:|
| arxiv FAR | 0.058 | 0.059 | 0.185 [0.106, 0.233] |
| arxiv GAP | 0.433 | 0.443 | 0.557 [0.509, 0.618] |
| steam FAR | 0.046 | 0.046 | 0.164 [0.108, 0.223] |
| steam GAP | 0.541 | 0.554 | 0.669 [0.617, 0.713] |

**h_any is far above the corpus ceiling on every target** (FAR 3–4×: no FAR target stays below 0.10). The witnesses
are multilingual token soup (e.g. Steam FAR k=1: "Mosulruz禄 beg Ladadi électrique…", 0.223), so they are
obviously not coherent; the coherent candidates (real texts trimmed to sentences) stay at the corpus ceiling. GAP
gains come from appending such tokens to a real description. So the encoder reaches far directions, but within
this search budget only through nonsense: the measured gap between h_any and the best coherent witness is the
original objection, now with numbers. No human coherence ratings were collected: the token-soup witnesses are
incoherent by inspection and the coherent candidates are trimmed corpus texts; h_coh beyond the corpus remains
unmeasured (it needs a search constrained to coherent text, e.g. an LLM rewriting loop scored by the encoder).

## 2. A decoder-specific, budget-aware prediction

**DEFINITION, not MAP.** Let Q_g(· | H_j) be generator g's output distribution given the
actual prompt history H_j. `decode.py` draws three candidates per round, embeds valid
JSON descriptions, feeds back the round winner and best-so-far score, and repeats for
four rounds. Its output is the best across all valid candidates. There are **12 attempts,
not 12 iid draws**. Define

\[
L_{g,4\times3}(t)=E_{Q_g}\!\left[\max_{j\le4,\ell\le3}
 \langle enc(S_{j\ell}),t\rangle\mid H_1(t)\right].
\]

Set an invalid candidate's score to −1 for a total definition. The full law includes
feedback transitions H_{j+1}. This is an expected attainable score under a stated
budget, not a support maximum or a particular reachable vector. A literal “most
probable point” would be a mode of the selected embedding's discrete pushforward law;
it depends on probability masses, ties and the protocol, and need not attain L.
For iid scores with CDF F, E max = −1 + ∫[-1,1](1−F(z)^K) dz. The iid expression does
not apply directly to the feedback loop.

**EMPIRICAL LAW, candidate-moment model.** `predict_budget.py` fits, separately by
arm and generator, the candidate score location

`score[j,l] = b0 + b1*(shown centroid · t) + round_effect[j] + error[j,l]`.

Training uses **all individual candidate scores**, equal target weights, a fixed .001
slope ridge, and no regression on final landing. A 12-dimensional residual covariance
is averaged over permutations of candidate labels within rounds and shrunk 20% toward
its diagonal. A joint Gaussian with these means/covariances, clipped to [−1,1], supplies the extreme-value
step (16,384 antithetic simulations). Dependence across rounds is included through the
covariance; non-Gaussian feedback and history-dependent tails are not fully modelled.
No held-out output, score or history enters a prediction. These are learned decoder
moments, not corpus moments. They are not a parameter-free prediction for a new generator.

Five folds hold out `k mod 5` **across all generators and arms**. Training has about 40
targets per generator/arm. κ is refitted on training REAL only; anchoring is the original
training mean of `output · centroid`, per generator/arm. All 1,746 successful guided
outputs are evaluated: five arms × six generators, plus GAP × five generators. Gateway
luna has no GAP; four other outputs failed. PLAIN and PAIR have no directional target
and hence no landing to predict. The fit conditions on successful logged candidates;
it does not estimate the probability of a completely failed decode. One success has
11 candidates; the others have 12. Its forecast still uses the planned 12 attempts.

| Arm | n | Observed median | Process prediction | Process MAE | κ MAE | Anchoring MAE |
|---|---:|---:|---:|---:|---:|---:|
| REAL | 300 | .7278 | .7394 | .0424 | .0413 | .0549 |
| INTERP | 300 | .7543 | .7663 | .0430 | .0492 | .0695 |
| NOISE | 298 | .4288 | .4419 | .0290 | .0251 | .0356 |
| CROWD | 299 | .4304 | .4432 | .0455 | .0475 | .0557 |
| FAR | 299 | .0335 | .0368 | .0157 | .0175 | .0158 |
| GAP | 250 | .5298 | .5379 | .0279 | .0265 | .0312 |
| All available | 1746 | .5360 | .5450 | .0341 | .0348 | .0442 |

| Generator, all available arms | Process MAE | κ MAE | Anchoring MAE |
|---|---:|---:|---:|
| luna gateway (no GAP) | .0366 | .0393 | .0458 |
| luna | .0383 | .0382 | .0447 |
| sol61 | .0317 | .0324 | .0405 |
| opus | .0321 | .0316 | .0542 |
| sonnet | .0326 | .0305 | .0420 |
| haiku | .0336 | .0374 | .0381 |

FAR's .0368 prediction is substantially closer than the old corpus-MAP .061. Against κ,
the paired FAR MAE difference is −.0019 (target-cluster bootstrap 95% interval
[−.0031, −.0007]); against anchoring it is −.0002 [−.0008, .0004]. Overall difference
versus κ is −.0007 [−.0028, .0014]: **no clear overall improvement**. Nominal 80% process
intervals cover 77.1% overall and 78.6% FAR. Bootstrap intervals keep the fitted folds
fixed, cluster by k, and are descriptive, without multiplicity correction or training
uncertainty. This is retrospective held-out-target evaluation, not external validation;
shared corpus content and neighbours can still occur across folds.

Two recorded diagnostics clarify budget effects. Resampling whole centred training
trajectories gives FAR .0332 / MAE .0154 and overall MAE .0330, but **implicitly reuses
historical final maxima**; it is not the answer to the stricter “no landing fit” request.
An iid empirical first-round extrapolation gives FAR .0403 / MAE .0166 and overall MAE
.0359; with three draws its overall MAE is .0372, with one .0617. These are model
ablations, not randomized measurements of a changed decoder budget. The full table and
paired uncertainty are in `runs/steam/bounds2/`.

## 3. The FAR-ceiling conjecture: refuted generally, bounded under explicit hypotheses

Write K_sel for the number of retained directions (distinct from decoder budget 12).
**CONJECTURE (a restricted approximation, not a universal law):** the old formula predicts
the selected median by

\[
 c_*=[z_N\sqrt{1-r^2}+r\Phi^{-1}(K_{sel}/(2M))]/\sqrt d.
\]

### THEOREM: mean norm and catalogue size do not determine this ceiling

Take an even N, half the unit corpus vectors equal to u and half to −u. Then μ=0,
r=0, and **C(t)=|u·t|, independent of N**. The identity is elementary and added to Lean.
For a population lower fraction q=K_sel/M of uniform directions, its selected median
is the q/2 quantile of |u·T|. At d=1024, N=68,222, q=50/20,000:

| Mathematical distribution (no sampling) | Conjecture | Exact population selected median |
|---|---:|---:|
| Antipodal corpus, r=0 | .13490 | .0000490 |

For a finite pool the rank fluctuates; it does not acquire the missing √log N term.
This refutes a universal finite-accuracy or leading-order claim from r,N,d alone.
Requiring distinct texts/vectors does not rescue it: perturb the atoms within distance
ε in antipodal pairs to make arbitrarily many distinct unit vectors while keeping r=0. Every ceiling, and hence every
ordered ceiling, changes by at most ε (Cauchy–Schwarz and order-statistic monotonicity).
It is not just a duplicate-counting objection. Even isotropic covariance is insufficient:
a dictionary containing repeated ±coordinate basis vectors has μ=0, Σ=I/d, yet its
ceiling is max_j |T_j| and stops growing once those 2d directions are present.

### THEOREM: finite-sample, selected-ceiling interval for an iid latitude corpus

Assume d≥3, N independent corpus vectors
`X_i = r u + s V_i`, where `s=√(1−r²)`, 0≤r<1 is a **population** parameter and V_i are
uniform on the unit sphere of the (d−1)-dimensional space perpendicular to u. Draw M
independent uniform targets, independent of this corpus; select the K_sel lowest
ceilings **against that same corpus**. Let F_m be the CDF of one coordinate of a uniform
unit vector in R^m, and Q_m its quantile function. For each target, exactly

\[
 C_j=rA_j+s\sqrt{1-A_j^2}\,W_j,\quad
 A_j=u\cdot T_j,\quad P(W_j\le w\mid T_j)=F_{d-1}(w)^N.
\]

The W_j are **not independent across targets** after sharing the corpus. We do not
assume they are. Choose positive failure budgets δ_W+δ_A+δ_Q=δ and numbers

\[
F_{d-1}(w_-)^N=\delta_W/(2M),\quad
1-F_{d-1}(w_+)=\delta_W/(2MN),\quad
1-F_d(a_*)=\delta_A/(2M).
\]

Require w_-≥0. With probability ≥1−δ_W−δ_A, simultaneously for every j,

\[
 rA_j+s\sqrt{1-a_*^2}w_-\le C_j\le rA_j+s w_+.
\]

**Proof.** For any fixed target, the lower-tail probability of W_j is the displayed
N-th power; its upper-tail probability is at most N[1−F(w_+)]. Union-bound over M
without assuming independence. Another union bound gives |A_j|≤a_* for every target.
Substitute, using w_-≥0 and s≥0. Coordinatewise inequalities imply the same inequalities
between ordered values, even if the ordering of C differs from that of A.

For selected rank h, `F_d(A_(h)) ~ Beta(h,M+1−h)` exactly. Take its lower and upper
δ_Q/(2R) tail quantiles and map through Q_d, calling the results a_h^−, a_h^+; R=1 for
odd K_sel and R=2 for even K_sel, using the one or two middle selected ranks. The
selected sample median is therefore, with probability ≥1−δ, in

\[
\left[r\,\operatorname{mean}_h a_h^-+s\sqrt{1-a_*^2}w_-,\quad
      r\,\operatorname{mean}_h a_h^++s w_+\right].
\]

This is a genuine finite-sample theorem, including shared-corpus selection. Probability
is over **both** corpus and targets. Plugging a measured corpus mean into r does not
certify that corpus: its distribution must satisfy the hypotheses independently.
`far_theory.py` evaluates these analytic distributions only, without data sampling.

| Model inputs | Old point formula | Proven 95% interval |
|---|---:|---:|
| N=68,222, d=1024, population r=.64, M=20k, K_sel=50 | .04319 | [.01967, .10376] |

Here w_−=.10982, w_+=.20970, a_*=.15318; each failure budget is .05/3. The interval is
loose because it controls all 20k maxima. **It does not prove .043 accurate to .005**.
The former .044 used the measured r, whereas this illustrative theorem uses exactly .64.
Replacing W by z_N/√d, √(1−A²) by 1, and the selected latitude by its normal quantile
recovers the conjecture. Those replacements are additional approximations, not hidden
steps in this theorem. At fixed d and sufficiently large N, W→1 and the √log N formula
must eventually fail by bounded support.

### CONJECTURES for the experimenter's falsification battery (not run here)

- **Heavy upper tails / rare clusters:** Gaussian moderate-tail extrapolation can
  underpredict maxima, with jumps when N first includes rare components. Bounded cosines
  cannot be literally asymptotically heavy-tailed; this concerns standardized finite-range
  tails. Duplicates or highly dependent clusters instead reduce the effective number of
  extremes and can make the formula overpredict. Record tail quantiles and duplicate rates.
- **Anisotropy:** selected directions can exploit small directional variance as well as
  negative μ·t. Large directional variance heterogeneity invalidates the constant-variance
  term; error sign is not universal. Pre-record separate mean, SD and standardized-tail
  contributions, and compare with the fixed-latitude theorem only on that synthetic model.
- **Small d / exponential N:** spherical coordinate tails differ from normal tails;
  saturation produces overprediction from √log N. As d→∞ with N=exp(αd), the iid isotropic corpus
  maximum has leading scale √(1−exp(−2α)), not √(2α). This follows by equating the spherical
  cap exponential rate −½ log(1−c²) to α.
- **Very small K_sel/M:** fixed-r normal-latitude approximations can become arbitrarily
  negative although scores are ≥−1; at fixed K_sel the rank has material sampling variation.
  Selection also preferentially picks unusually small residual maxima. Near full selection,
  using an expected maximum z_N for a median can still give a positive bias, especially
  for r≈0 where mean-direction variation no longer dominates. Use actual rank quantiles,
  not a deterministic K_sel/(2M) substitute when K_sel is small.
- **Additional discriminating test:** compare selection and evaluation on the same corpus
  versus independent corpus halves, at matched N. A change isolates selection of unusually
  low residual maxima; the mean latitude alone cannot explain that difference. Freeze all
  such predictions before running; no new measurements of these regimes were made here.

## Reproduction and formal status

```sh
uv run python scripts/bounds/support_probe.py --probe
uv run python scripts/bounds/predict_budget.py
uv run python scripts/bounds/far_theory.py
python3 scripts/bounds/pack_lean_check.py  # refresh existing Kaggle CPU checker
# Upload runs/support-search/ as the private dataset in kernel-metadata.json,
# select T4 x2, then push kaggle/support-search/. No upload/push was performed here.
uv run python scripts/bounds/verify_support.py /path/to/witness_gpu0.jsonl /path/to/witness_gpu1.jsonl
# Fill blind ratings, then rerun with --judgments /path/to/completed-ratings.csv.
```

The new Lean additions in `lean/LatentGap/Bounds.lean` prove only the elementary unit
upper bound, antipodal ceiling, two-point latitude ceiling and threshold-count sandwich
that underlies order-statistic stability. The beta quantiles, spherical probability
argument and decoder model are prose/math/Python, **not claimed Lean-verified**.
The complete bounds file (all eleven lemmas) **compiled** on Lean/Mathlib v4.36.0-rc1
with focused imports; `#print axioms` lists only `propext`, `Classical.choice`, `Quot.sound`.
The original blanket Mathlib import exceeded the local time limit; narrowing imports
resolved the check without changing the original seven proofs. The successful log is
`runs/steam/bounds2/lean-check.log`. The existing `kaggle/lean-check/` CPU kernel has also
been refreshed with current sources and all eleven axiom checks; it now fails if either
compilation or the axiom-check command fails. No kernel was pushed.
