# Hard limits and an explanation of FAR decoding

2026-10-24. Constructive follow-up to `geometry.md`, not a replacement for its measurements.
**The strongest established obstruction is coverage by a finite dictionary, not a 0.05
support ceiling for all possible text.** Exact cap optimisation allows FAR scores much
higher than 0.05. A neighbour-based decomposition explains the observed decoder scores;
a density model explains a direction of motion but does not identify a reachable text.

Reproduce with:

```sh
uv run python scripts/bounds/analyse.py
uv run python scripts/bounds/summarise.py
```

Inputs: the adult-filtered past corpus and the same six runs as `geometry.py`. Corpus
and output vectors are renormalised; target vectors are also renormalised. There are
68,222 corpus vectors, 250 targets, and 1,496 successful guided outputs. Hidden rows
are excluded from every reported corpus ceiling. Tables give target medians or pooled
output medians, not bounds on a median substituted for bounds on individual vectors.
CSV/JSON results are written to `~/Documents/experiment-data/latent-gap-ideas/runs/steam/bounds/`.
No new LLM calls. The bge comparison below uses the already recorded independent-space
ceiling experiment, not Qwen targets evaluated in bge coordinates.

## 1. THEOREM: exact support bounds, and what they actually exclude

Let `x,t,u` be unit vectors, `u` a fixed cap axis, and suppose **every output under
consideration** satisfies `u·x >= a`, with `0 <= a < 1`. Write `b=u·t`. Then the sharp bound is

\[
 x\cdot t\le H_a(b)=
 \begin{cases}1&b\ge a,\\
 ab+\sqrt{(1-a^2)(1-b^2)}&b<a.
 \end{cases}
\]

**Proof and equality case.** If `b>=a`, choose `x=t`. Otherwise the optimum is
`x*=a u + sqrt(1-a²) (t-bu)/sqrt(1-b²)` for `-1<b<a`. At `b=-1`, any
unit vector of latitude `a` attains `-a` (dimension at least two). For every `lambda>=0`,
Cauchy–Schwarz gives `t·x <= ||t+lambda u||-lambda a`. Substituting
`lambda=a sqrt((1-b²)/(1-a²))-b >= 0` gives the displayed bound. The endpoint follows
directly. Thus this is an exact maximum for the geometric cap, not a fitted inequality.
For unit vectors `||x-t||²=2-2x·t`: it is also the nearest point of that cap.

An anisotropic refinement uses a positive-definite matrix `Sigma`, a centre `mu`, and
the hypothesis `(x-mu)' Sigma^{-1}(x-mu) <= Q`. In its eigenbasis, with eigenvalues `s_i>0`,
**every** `lambda>0, eta>=0` supplies the certificate

\[
 t\cdot x\le t\cdot\mu+\lambda Q+\eta(1-\|\mu\|^2)
 +\frac14\sum_i\frac{(t_i-2\eta\mu_i)^2}{\lambda/s_i+\eta}.
\]

**Proof.** Put `y=x-mu`, add the nonnegative slacks
`lambda(Q-y' Sigma^{-1}y)+eta(1-||mu+y||²)` to the objective, and maximise the resulting
concave quadratic in `y` by completing squares. Take the smaller of this certificate
and 1. Optimising two multipliers tightens it; validity does not depend on convergence.
This is an upper bound for the ellipsoid intersected with the unit ball, hence also
for unit embeddings. It does not require Gaussian data.

**Data audit.** Using the empirical `mu,Sigma`, the minimum corpus latitude is **0.2411**,
not 0.64; 0.64 is approximately the mean latitude. `Sigma` has minimum eigenvalue
`1.07e-7`. The maximum squared Mahalanobis radius is **3278.4**, its 99th percentile
**1801.2**, and its mean **1024**. All 1,496 decoded outputs satisfy the full-corpus
latitude and radius envelopes. This checks the hypotheses on these outputs; it does
not establish them for every string the embedding model could accept.

| Arm | Visible ceiling | Landing | Cap, a=min | Ellipsoid+ball, Q=max | Ellipsoid+ball, Q=99th percentile |
|---|---:|---:|---:|---:|---:|
| REAL | .736 | .728 | 1.000 | 1.000 | 1.000 |
| INTERP | .800 | .754 | 1.000 | 1.000 | 1.000 |
| NOISE | .450 | .429 | 1.000 | .942 | .895 |
| CROWD | .450 | .430 | 1.000 | 1.000 | .997 |
| FAR | .046 | .034 | .947 | .821 | .709 |

The 99th-percentile radius excludes one INTERP output. The 1st-percentile latitude
`a=.4653` gives a FAR bound **.844**, but excludes **7.0% of FAR outputs**. Neither
percentile envelope is a universal constraint or a calibrated prediction interval.
Even imposing the hypothetical lower latitude `.64` would allow FAR landing about
**.713**. Cap geometry therefore rules out landing 1 under these hypotheses, but does
**not** rule out landing .3 or explain landing .05. The cap bound is mathematically
sharp and empirically very loose; the covariance bound remains loose as well.

There is an important distinction between a rare **mean shift** and an individual
embedding: a Gaussian extreme score at `z≈4.3` corresponds to a conditional mean
with Mahalanobis radius about 4.3. Individual corpus vectors have RMS radius **32**.
Using `Q=4.3²` as a support constraint would produce an attractive, false hard limit.

**Identifiability obstruction (THEOREM).** Finite samples and concentration statistics
alone cannot exclude an unseen text at `t`. For any distribution `P` on the sphere,
`P_epsilon=(1-epsilon)P+epsilon delta_t` has `t` in its support. Its mean changes by at
most `2 epsilon`, covariance in operator norm by at most `5 epsilon`, and `n` iid
draws miss the added component with probability `(1-epsilon)^n`. The covariance claim
follows from `Sigma_epsilon=(1-epsilon)Sigma+epsilon(1-epsilon)(t-mu)(t-mu)'`.
For `epsilon=1e-7`, 68,222 samples miss it with probability **.9932**. This is an
obstruction to inference from samples, not a claim that the actual encoder has such
a text. A universal reachability certificate needs an independently justified
constraint on the encoder's entire output range.

## 2. THEOREM: a dictionary cannot cover arbitrary directions cheaply

For **any fixed** set of `N` unit embeddings in ambient dimension `d>=2`, and a uniform
target `T` independent of the dictionary, let `C(T)=max_i x_i·T`. For `0<c<1`,

\[
 \Pr\{C(T)\ge c\}\le\min(1,Np_d(c)),\qquad
 p_d(c)=\tfrac12 I_{1-c^2}((d-1)/2,1/2).
\]

Consequently a dictionary covering **every** direction of a region `R` of uniform
spherical measure `alpha` must have `N>=alpha/p_d(c)`. Whole-sphere coverage requires
`N>=1/p_d(c)`. These are necessary conditions, not sufficient covering constructions.

**Proof.** Each dictionary item covers a spherical cap of measure `p_d(c)`; the measure
of their union is at most the sum of their measures. No independence between caps,
Gaussian scores, isotropic corpus, or effective-dimension assumption is used.
The same bound applies to **any decoder whose complete output range has at most N
embeddings**, even if it chooses its output adaptively after observing `T`.

| Cosine c | Necessary N for whole-sphere coverage, d=1024 | Maximum fraction covered by 68,222 items |
|---|---:|---:|
| .10 | 1.48e3 | 1 (uninformative) |
| .20 | 1.92e10 | 3.56e-6 |
| .30 | 2.17e22 | 3.15e-18 |
| .50 | 3.24e65 | 2.11e-61 |

This turns the cap-volume calculation into an arrangement-independent lower bound.
For fraction coverage its scale is meaningful: an iid uniform dictionary has expected
covered fraction `1-(1-p_d(c))^N`, reaching about .632 at `N=1/p_d(c)`. It is not tight
for this anisotropic corpus and does not predict the FAR ceiling .046. REAL, INTERP,
NOISE and CROWD are corpus-dependent targets, so the uniform-target probability bound
does not apply to them. For FAR's **20,000 initially independent uniform candidates**,
another union bound gives probability at most `20000*N*p_1024(.3)=6.30e-14` that even
one candidate has a corpus neighbour at cosine .3. Selecting the lowest ceilings
cannot defeat this statement. It says nothing comparable about an unlimited decoder.

In particular, `2.17e22` is only about **74.2 bits** of distinguishable outputs;
`3.24e65` is about **217.6 bits**. The space of possible descriptions can be much larger
than the corpus. Astronomical corpus size is not itself a proof of textual impossibility.
If all outputs additionally lie in the cap of section 1 with `a>0`, covering the whole
sphere at positive cosine is impossible for any N, since direction `-u` has support `-a`.

## 3. THEOREM + EMPIRICAL LAW: neighbour averaging explains much of the landing

Let `m8` be the unnormalised mean of the eight shown embeddings, `s=||m8||`, `C=m8/s`,
`a_e=e·C`, and `r=e-a_e C`. There is an exact identity

\[
 e\cdot t=a_e(C\cdot t)+r\cdot t
 =\frac{a_e}{s}\,\overline{x_i\cdot t}+r\cdot t.
\]

This explains how matching the corpus ceiling can arise without maximising over all
text: the shown neighbours already have scores close to that ceiling, normalising
their mean amplifies the score, and imperfect anchoring attenuates it again.
For FAR, median `a_e/s=1.017`, top-eight mean score / ceiling is **.862**, and the
anchored term / ceiling is **.862**. The remaining target-directed component is small
in absolute score, but not negligible relative to FAR landing. **No theorem sets it to
zero or forces kappa=1.** The approximate cancellation explains an order-one ratio;
it does not derive the REAL-calibrated kappa values exactly.

The empirical predictor is `landing_hat=mean_training(a_e | generator,arm)*(C·t)`.
It fits anchoring from vectors, not landing scores. Five folds hold out `k mod 5`
across all generators and arms; coefficients never see the held-out target's outputs.
Kappa is refitted on the training REAL outputs in each fold. This is retrospective
cross-validation within this experiment, not prospective validation on new targets or
another domain. The anchoring predictor uses arm-specific calibration, unlike kappa.

| Arm | Landing | Exact anchored term | Median remainder | Anchoring OOF prediction / MAE | Kappa OOF MAE |
|---|---:|---:|---:|---:|---:|
| REAL | .728 | .695 | +.026 | .691 / .055 | .041 |
| INTERP | .754 | .707 | +.053 | .704 / .070 | .049 |
| NOISE | .429 | .413 | +.014 | .406 / .036 | .025 |
| CROWD | .430 | .402 | +.030 | .403 / .056 | .047 |
| FAR | .034 | .038 | -.004 | .039 / .016 | .018 |

Medians of terms need not sum to the median total. The exact FAR remainder has MAE
**.0148**. FAR anchoring to the centroid is **.676**, versus **.544** for a neighbour
against the centroid of the other seven. This is consistent with generating a more
central synthesis of the examples. It does not establish a causal effect of showing
them: that needs a randomised neighbour intervention.

Two attempted alternatives were worse: full-vector regressions `e≈alpha C+beta mu`
and `e≈alpha unit(G)+beta mu` (G below), trained without held-out targets, have FAR
MAE **.021 / .030** and REAL MAE **.083 / .176**. They are retained in the scripts/results,
not presented as successful general laws. The kappa benchmark remains better on the
other arms. No claim of statistically established superiority for FAR is made.

### Why Sigma t appears, and what “most probable” could mean

**THEOREM (not restricted to Gaussians).** For any distribution on the unit sphere,
define an exponentially tilted law `dP_beta(x) = exp(beta t·x)dP(x)/Z(beta)`. Differentiating
the bounded integrals gives

\[
 \frac{d}{d\beta}\mathbb E_\beta[X]=\operatorname{Cov}_\beta(X)t.
\]

Hence the initial response to score preference is exactly `Sigma t`, for an arbitrary
embedding distribution. Finite differences of the empirical tilted mean verify the
identity with relative error **8.3e-10**. This is a local response theorem; finite preference integrates
changing covariances and need not stay on the original Sigma t ray. It does not assert
that an LLM implements exponential tilting.

For a Gaussian surrogate, put `sigma²=t' Sigma t>0`. The conditional mean, also the
conditional mode on the affine hyperplane `t·x=v`, is
`mu+Sigma t*(v-t·mu)/sigma²`. This follows by completing the Gaussian quadratic; it is
the minimum Mahalanobis-cost way to attain score v. For the upper tail with fraction
`p=8/N_visible`, define `z=Phi^{-1}(1-p)`, `L=phi(z)/p`. Its conditional **mean** is
`G=mu+Sigma t*L/sigma`. This is a prediction of a neighbour mean, not a unit embedding.
For FAR, `||G||=.654`, versus measured `||m8||=.664`, but the vector error is **.325**;
the residual direction cosine is only **.578**. The Gaussian has captured some
structure, not the individual selected neighbourhood.

A distinct, fully specified **spherical density surrogate** is
`f_beta(x) ∝ exp[-(x-mu)'Sigma^{-1}(x-mu)/2 + beta t·x]` on `||x||=1`.
We plug in corpus moments (not a spherical maximum-likelihood fit) and choose
`beta=L/sigma` without decoder outcomes. Its global MAP solves

\[
 x_i=\frac{\mu_i+\beta s_i t_i}{1+\lambda s_i},\qquad
 \sum_i x_i^2=1,\quad \lambda>-1/\max_i s_i.
\]

The script solves the one-dimensional norm equation. Positive-definiteness of the
Lagrangian quadratic certifies the optimum; completing squares proves this. A mode
is density-relative, not an atom of positive probability, and this surrogate's mode
need not be an actual text. Normalising G is yet another operation, with another answer.

| Arm | G·t (nonunit mean) | unit(G)·t | Spherical MAP score | Observed landing | MAP MAE |
|---|---:|---:|---:|---:|---:|
| REAL | .670 | .770 | .748 | .728 | .074 |
| INTERP | .693 | .808 | .770 | .754 | .057 |
| NOISE | .405 | .485 | .473 | .429 | .047 |
| CROWD | .372 | .461 | .444 | .430 | .060 |
| FAR | .038 | .058 | .061 | .034 | .030 |

These are precise candidate formulations, not interchangeable definitions of a
“most probable reachable point”. In particular the spherical MAP overpredicts FAR;
neither density construction establishes maximum achievable landing. Indeed this
spherical surrogate has positive density everywhere on the sphere: its support-maximal
landing is **1**, even though its FAR MAP score is .061. A density preference is not
a support exclusion.

After orthogonally removing `span(mu,C)` from both e and Sigma t, their FAR residual
cosine falls from the original `cos(e-mu,Sigma t)=.350` to **.105**. This partial
alignment uses output vectors only for evaluation. Much of the original association
is shared with the neighbour centroid; positive residual alignment remains. The
data support a neighbour-conditioned response, not the claim `e=mu+lambda Sigma t`.

## 4. Cross-domain predictions and ways to falsify them

**CONJECTURE: selected extremes from mean norm alone.** A useful parameter-free
extension of the two-moment approximation is obtained by assuming uniform-direction
variance is approximately constant, `sigma≈sqrt((1-r²)/d)`, where `r=||mu||`, and
`mu·T≈N(0,r²/d)`. Approximate the corpus maximum by `mu·T+sigma*z_N`, with the same
Gaussian expected-maximum approximation z_N as `geometry.py`. If the K most isolated
of M directions are selected, their median is approximately the K/(2M) quantile:

\[
 C_{\rm FAR,median}\approx
 \frac{z_N\sqrt{1-r^2}+r\,\Phi^{-1}(K/(2M))}{\sqrt d}.
\]

This follows within those approximations, not from a support theorem. It has no
parameters fitted to FAR scores. For K=50, M=20,000, it predicts a strong penalty
from selecting directions opposed to the mean. The numerical table is retrospective;
the half-corpus observations are the existing growth experiment's last points.

| Space / evaluation corpus | Predicted median FAR ceiling | Observed |
|---|---:|---:|
| Qwen, full 68,222 | .044 | .046 |
| Qwen, half 34,111 | .040 | .044 |
| bge-m3, half 34,111 | .011 | .014 |

Agreement to a few thousandths is promising, but cancellation can conceal incorrect
components: full-Qwen predicted target latitude is about -.094, measured -.083; FAR
score SD is .0233 versus the constant approximation .0241. Selection also responds
to directional variance and tail shape. Do not treat this as another established law.

For arXiv/patents, freeze the embedding dimension d, N, M and K and compute r and the
spectrum **before decoding**. The following distinguish mechanisms:

- **Finite-dictionary obstruction:** at the same d,N the covering bound is unchanged
  by domain, mean norm or participation ratio. A corpus violating it in an independent
  uniform-target experiment would contradict the sampling/normalisation assumptions.
- **Ceiling conjecture:** predict the preceding formula before seeing new ceilings;
  report absolute error, plus errors in its mean and variance components. A proposed
  success criterion is median error <=.02 in a new domain, fixed before its analysis.
  Under this approximation, median FAR ceiling >=c requires
  `z_N sqrt(1-r²)+r Phi^{-1}(K/(2M)) >= c sqrt(d)`. For Qwen's statistics, reaching
  .3 needs roughly **d<=22**, rather than 1024, at this N and selection budget. This
  is a model threshold, not a sufficient condition for a text to exist.
- **Spectrum matters through directional variation, not a substituted sphere dimension:**
  exactly `E_T[T' Sigma T]=tr(Sigma)/d`, and
  `Var_T(T' Sigma T)=2[tr(Sigma²)-tr(Sigma)²/d]/[d(d+2)]` for independent uniform T.
  With `r_eff=tr(Sigma)²/tr(Sigma²)`, relative variance is
  `2d/(d+2)*(1/r_eff-1/d)`. Smaller r_eff increases heterogeneity; it does not increase
  average directional variance at fixed r,d. Large heterogeneity should break the
  constant-variance selection approximation. These identities follow from the uniform
  sphere's fourth moments, `E T_i^4=3/[d(d+2)]` and `E T_i²T_j²=1/[d(d+2)]`.
- **Actual low-dimensional support:** if texts are exactly
  `x=r u+sqrt(1-r²)v`, `v` unit in a k-dimensional space L orthogonal to u, their sharp
  support is `r(u·t)+sqrt(1-r²)||Proj_L t||`. Ambient-uniform targets have mean projected
  squared norm k/d; decreasing k makes them harder to reach. Only if targets are drawn
  in L may a k-dimensional cap covering formula be used. Neither k_90 nor participation
  ratio establishes this support hypothesis.
- **Neighbour mechanism:** on new FAR targets, predict from calibrated anchoring and
  the shown centroid, retaining the signed residual and MAE. Randomise examples among
  targets with matched top-eight score/norm statistics, or remove them, keeping the
  search budget fixed. A large change in landing/novelty would identify a prompt effect;
  unchanged outputs would weaken the proposed explanation. Compare Sigma alignment
  before and after removing the actual shown centroid, not just the global mean.

The design document still contains an effective-dimension covering prediction in
Study 2. It is a historical pre-registration, not a consequence of the theorems here;
the ambient-target prediction above has the opposite low-dimensional-support effect.

## 4b. Falsification battery (run 2026-10-24, predictions committed first)

`scripts/bounds/conjecture_battery.py`: 32 configurations, 2 domains (Steam, arXiv) × 2 spaces (Qwen3, bge-m3) ×
corpus size N ∈ {2k, 5k, 10k, 20k, full} at (K, M) = (50, 20,000), plus (10, 20,000), (500, 20,000) and (50, 2,000)
at full N. Predictions from corpus statistics only were committed before measuring
(`docs/prereg/conjecture_battery_predictions.csv`, commit 7c90bfa); results in `docs/prereg/conjecture_battery_results.csv`.
Pre-fixed criteria: |error| ≤ 0.005 strict, ≤ 0.02 lenient.

| | Result |
|---|---|
| within lenient (0.02) | 32 / 32 |
| within strict (0.005) | 30 / 32 (failures: bge-m3 Steam, 0.0054 and 0.0060) |
| median / max absolute error | 0.0024 / 0.0060 |
| only K/M matters: (500, 20,000) vs (50, 2,000), observed | 0.0584 / 0.0602, 0.0365 / 0.0356, 0.0713 / 0.0711, 0.0465 / 0.0480 |
| systematic bias by cell | Qwen-arXiv −0.004 (all 8), bge-Steam +0.003 (7 of 8), others within ±0.003 |

**Not refuted.** The conjecture tracks the N-dependence, the selection fraction, both domains and both spaces with
no fitted parameter. The residual is a per-(domain, space) offset of a few thousandths, the size expected from the
constant-directional-variance step it assumes; it stays a CONJECTURE until that step is derived or bounded.

**Third domain, patents (HUPD; predictions committed in c9c278f before any patents target):** 16 / 16 within
0.02, 9 / 16 within 0.005, max error 0.0072. Qwen-patents overpredicts in all 8 cells (≈ −0.006; full-N FAR
predicted 0.068, observed 0.063); bge-m3-patents is within 0.004. Across domains the Qwen offset grows as ‖μ‖ falls
(Steam 0.64: ≈ 0; arXiv 0.54: −0.004; patents 0.51: −0.006), consistent with selection picking directions of
below-average variance, which the constant-variance step (σ² = (1 − r²)/d, exact only on average) ignores. Still
not refuted at the lenient criterion; the strict one fails for one cell. Counter-example in bounds2.md §3 shows it
cannot hold for arbitrary distributions.

## Formalisation and limits of this deliverable

`lean/LatentGap/Bounds.lean`, imported by the library root, contains proof attempts for
the cap dual certificate, unit-distance identity, anchoring decomposition, normalised
mean score, quadratic square completion, ellipsoid/ball certificate, and general
measure covering bound. **All seven compile** (Mathlib v4.36.0-rc1, Kaggle CPU kernel
`kaggle/lean-check`, 2026-10-24) and `#print axioms` lists only `propext`,
`Classical.choice`, `Quot.sound`; the only edits after this document was drafted were
renaming the reserved identifier `λ` and opening `ENNReal` notation. Sharp cap
attainment, the beta-function cap integral, contamination, tilt differentiation,
Gaussian/MAP statements and spherical fourth moments are proved/derived in prose,
not claimed as Lean-verified. Python checks cap attainment, realised-output
certificates, MAP feasibility and the tilt derivative. It cannot certify global
support of all text.

What remains open is substantive: an independently justified tight support model for
the encoder, a distribution over *reachable* texts rather than surrogate vectors,
prospective cross-domain validation, and a causal test of neighbour anchoring. The
present results establish why a finite corpus misses almost every ambient direction
and explain much of what these prompted decoders do there. They do not establish
that all decoders must stop at the corpus ceiling.
