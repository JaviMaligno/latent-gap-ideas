# Round 3: directional extremes from corpus moments

2026-10-24. No LLM calls. The 48 old measurements were known when choosing this
correction: **retrospective, in-sample for model form**, not a fresh validation.
No numeric coefficient is fitted to a ceiling. Future predictions are separate;
no future ceiling is measured and no target is selected against a future corpus.

## 1. Corrected prediction and its inputs

Let μ be the empirical mean, Σ the empirical covariance (denominator N), r=‖μ‖,
u=μ/r, L=uᵀ(X−μ), τ²=uᵀΣu, v=Σu/τ², and R=Σ−τ²vvᵀ. Thus
X=μ+vL+E, E is uncorrelated with L, and E⊥u. Require r>0, τ²>0,
and tr R>0. These are prerequisites, not conventions for degenerate corpora (r and τ² are explicitly guarded).

Sort corpus rows by L into **32 equal-count bins**, fixed across all domains/spaces.
For each bin b retain its weight w_b, latitude mean ℓ_b, latitude variance ν_b,
and residual-energy ratio s_b=E_b‖E‖²/E‖E‖². The surrogate is

\[
 X^*\mid B=b\sim\mathcal N(\mu+v\ell_b,\;s_bR+\nu_bvv^T),\quad P(B=b)=w_b.
\]

This is a **corpus-moment Gaussian mixture**, not a Gaussian fit to maxima. It
preserves μ and Σ exactly: ∑wℓ=0, ∑w(ν+ℓ²)=τ², ∑ws=1. It additionally retains
latitude-dependent residual scale and the non-Gaussian latitude distribution.
It does not preserve every conditional covariance, skewness or cluster dependence.
Like a Gaussian corpus, its samples need not be unit vectors; application to the
normalised text corpus is an approximation, not an exact distributional identity.

For a uniform unit direction T define

\[
 F_t(c)=\sum_b w_b\Phi\!\left(
 \frac{c-\mu^Tt-\ell_bv^Tt}{\sqrt{s_bt^TRt+\nu_b(v^Tt)^2}}\right),\qquad
 H_N(c)=E_T[F_T(c)^N],\qquad
 \boxed{C_{new}=H_N^{-1}(K/(2M)).}
\]

**EMPIRICAL LAW (candidate):** use this plug-in quantile for the selected median
of a fixed text corpus. **CONJECTURE:** it transfers to the untouched future splits
with the success criterion below. The old formula remains a conjecture; neither
formula becomes a universal law from the retrospective results.

The new maximum step uses the **entire** iid maximum CDF F_t^N, not its expected
value z_N. Selection jointly reweights latitude, directional variance and residual
maximum fluctuations. For example, the model-selected direction density below c
is proportional to F_t(c)^N. No “selected variance” is substituted by hand.

**Exact inputs consumed:** N,d, μ, the complete eigensystem of Σ including μ/v
alignment with its eigenvectors, τ², and the 32 tuples (w,ℓ,ν,s). No selected target,
nearest neighbour, maximum, ceiling, decoder output or future outcome is an input.
The eigenspectrum alone would lose the alignment information. Bin count is a fixed
modelling resolution, not a ceiling-calibrated parameter. Sample moments are fitted
to corpus rows; “no ceiling fitting” does not mean “no estimated parameters”.

Numerics: two scrambled Sobol replicates of 2^15 Gaussian integration points in
covariance eigen-coordinates. Draw G~N(−2.5u,I), use T=G/‖G‖, and weight by
exp(2.5uᵀG+2.5²/2). This is a change of integration measure, not target selection.
Root-find H_N(c)=K/(2M); `quadrature_half_range` is half the replicate difference,
**not** a confidence interval or a bound on model error. Nothing is scored against
corpus rows during integration. Finite-pool rank noise remains; K/(2M) approximates
the midpoint of the two selected middle ranks when K is even.

## 2. What is proved

**THEOREM (oracle iid mixture law).** Suppose the N corpus rows are independent
samples from the stated mixture with fixed, known parameters, and T is independent
and uniform on the sphere. Then P(max_i X_iᵀT≤c)=H_N(c) exactly. More generally the
same identity holds for any row distribution with directional CDF F_t. Proof:
condition on T=t, multiply the N independent probabilities, and integrate. The
population lower-q conditional distribution has median H_N^{-1}(q/2) when H_N is
continuous and strictly increasing there. This is selection in the **ensemble**
(corpus and direction), not yet selection of many directions against one corpus.

**THEOREM (shared-corpus correction and finite pools).** For a realised corpus X,
put H_X(c)=P_T(max_i X_iᵀT≤c | X). Let F_{t,s}(c,c) be the joint one-row projection
CDF in two independent uniform directions. Then

\[
 E_XH_X(c)=H_N(c),\quad
 V(c):=\operatorname{Var}_XH_X(c)
 =E_{T,S}[F_{T,S}(c,c)^N]-H_N(c)^2.
\]

Proof: square the direction integral and use independence of rows, **not**
independence of maxima sharing rows. For the mixture, F_{t,s} is a weighted sum of
bivariate normal CDFs with means (tᵀm_b,sᵀm_b), variances tᵀΣ_bt,sᵀΣ_bs and covariance
tᵀΣ_bs. Thus V is specified by the same parameters and is checkable without a
corpus-ceiling experiment. We have not evaluated/certified that integral here.

Here is an explicit sufficient accuracy condition. Fix endpoints c_-<c_+, η>0,
ε∈(0,1), and each central selected rank h (K/2 and K/2+1 for even K). Choose l_h,u_h
as lower-tail/upper-tail ε/(2R_h) quantiles of Beta(h,M+1−h), where R_h is the number of ranks.
If

\[
 H_N(c_-)+\eta<\min_h l_h,\qquad
 H_N(c_+)-\eta>\max_h u_h,
\]

then the selected sample median lies in [c_-,c_+] with probability at least

\[
 1-\varepsilon-[V(c_-)+V(c_+)]/\eta^2.
\]

Assume H_X continuous at the ranks (or use generalised-quantile brackets).
Chebyshev at the two endpoints bounds |H_X−H_N|; conditional on X, the M directions
are iid and H_X(C_(h)) has the beta order-statistic law. Union-bound endpoints and
ranks. This handles the shared corpus explicitly and allows finite K and M.
No such small V or tight .005 interval has been established for the text corpora.

**THEOREM (robustness).** If a different iid row law has directional CDFs within
ε_t uniformly in c of the surrogate, its annealed maximum CDF differs by at most
N E_T ε_T (capped at one). Factor a^N−b^N or induct on N, then integrate.
If, additionally, a fixed corpus has directional CDF H_X within δ of that annealed
law, its quantile is bracketed by surrogate quantiles at p±(δ+N E ε_T), whenever
those probabilities are in (0,1). A .005 statement therefore needs tail accuracy
and corpus concentration, not just a measured covariance. Ordinary .99 quantile
agreement alone cannot certify tails at probabilities of order 1/N.

**Checkable hypotheses, not asserted facts:** iid rows with no unmodelled
cluster dependence; sufficiently accurate directional mixture tails around 1/N; sufficiently
small V (or separately justified fixed-corpus CDF discrepancy); positive local
CDF slope and manageable finite-rank uncertainty. Population parameters in these
theorems are fixed. Estimating them on the same corpus requires additional
parameter-estimation control; the plug-in calculation does not supply it. Even the
full covariance is insufficient in the antipodal/coordinate counterexamples of
[bounds2 §3](bounds2.md#3-the-far-ceiling-conjecture-refuted-generally-bounded-under-explicit-hypotheses).

**Lean scope.** Three elementary additions in `lean/LatentGap/Bounds.lean` prove the
centred directional second-moment identity, |a^N−b^N|≤N|a−b| on |a|,|b|≤1,
and the deterministic CDF endpoint bracket. The probability identities, beta law,
mixture derivation and numerical integration above are prose proofs, not claimed
Lean formalisation. The full Bounds source compiled successfully on Lean/Mathlib v4.36.0-rc1.
All 14 `#print axioms` checks report only `propext`, `Classical.choice`, `Quot.sound`;
no `sorry` or extra axioms. The exact source hash and output are in
`docs/prereg/conjecture_round3_lean.txt`; `lean/BoundsAxioms.lean` reproduces the
checks after building the library. The complete `lake build` also passed (18,116 jobs;
about 38 minutes), with style/module warnings and no errors.

## 3. Retrospective diagnostics (all 48 rows)

The covariance-only model uses F_t=Φ((c−μᵀt)/√(tᵀΣt)); the constant-variance
ablation uses tr(Σ)/d but retains the random maximum. The latter separates the
z_N approximation from anisotropy. The latitude mixture was fixed before future
statistics were computed; neither its bin count nor any offset was tuned against
ceilings. An initial six-cell Gaussian pilot motivated retaining latitude shape;
this model-form decision is part of the retrospective use of the known data.

| Prediction | Median absolute error | Maximum absolute error | Within .005 | Mean signed error (observed − predicted) |
|---|---:|---:|---:|---:|
| Old formula | 0.002594 | 0.007183 | 39/48 | -0.001815 |
| Constant variance, random maximum | 0.001447 | 0.007988 | 44/48 | +0.001432 |
| Full covariance, Gaussian | 0.001975 | 0.007801 | 43/48 | +0.002498 |
| Latitude mixture | 0.001580 | 0.003562 | 48/48 | -0.001225 |

| Domain / space | Old cell bias | Corrected cell bias | Corrected max absolute error |
|---|---:|---:|---:|
| steam / qwen3e06 | -0.000799 | -0.001778 | 0.002457 |
| steam / bgem3 | +0.003134 | -0.000964 | 0.003364 |
| arxiv / qwen3e06 | -0.004413 | -0.001731 | 0.002817 |
| arxiv / bgem3 | -0.001007 | -0.000764 | 0.002204 |
| patents / qwen3e06 | -0.005957 | -0.001183 | 0.003562 |
| patents / bgem3 | -0.001852 | -0.000930 | 0.002404 |

Full past N, K=50, M=20,000, to separate the approximations:

| Domain / space | Old | Random max, constant variance | Full covariance | Mixture | Observed |
|---|---:|---:|---:|---:|---:|
| steam / qwen3e06 | 0.04364 | 0.04070 | 0.03971 | 0.04598 | 0.04416 |
| steam / bgem3 | 0.01430 | 0.01239 | 0.01237 | 0.01982 | 0.01884 |
| arxiv / qwen3e06 | 0.06233 | 0.05849 | 0.05659 | 0.06037 | 0.05815 |
| arxiv / bgem3 | 0.03085 | 0.02851 | 0.02765 | 0.03146 | 0.02941 |
| patents / qwen3e06 | 0.06794 | 0.06376 | 0.06139 | 0.06357 | 0.06253 |
| patents / bgem3 | 0.02908 | 0.02677 | 0.02530 | 0.02876 | 0.02904 |

**Result:** all 48 corrected errors meet .005; the median absolute error improves.
The per-cell bias range contracts substantially, but **the offset is reduced, not
eliminated**: every corrected cell retains a small negative mean error. Steam/Qwen
gets worse. There is no ceiling-fitted recentering. The Gaussian ablation shows that
random maximum fluctuations explain much of the old error; anisotropy is not the
sole mechanism. The mixture repairs the opposite-sign Steam/bge error but remains
an approximate tail model. The constant-variance/random-maximum ablation has the
best median error, but four errors exceed .005; it is retained as a secondary forecast,
not substituted for the registered primary correction. This is evidence for a
candidate empirical law, not proof of a .005 law for text embeddings.


**Direct mechanism check on saved past Qwen FAR targets** (50 per domain; the pilot
pools, not the separately seeded 48-cell battery). No new target selection or maximum
was computed. Variance ratios use tr(Σ)/d as the uniform-direction average.

| Saved targets | Median variance ratio | Corr(μᵀt,tᵀΣt) | Median standardized .999 quantile |
|---|---:|---:|---:|
| Steam/Qwen | 0.93753 | −0.51255 | 3.11937 |
| arXiv/Qwen | 0.91536 | −0.39209 | 3.09737 |

This supports below-average variance and joint selection. It does **not** support
lighter tails at .999: the normal quantile is 3.09023. These diagnostics do not
identify the 1/N tail, establish causality, or validate patents/bge selection.
Unconditionally Cov(μᵀT,TᵀΣT)=0 by T↦−T symmetry; this does not imply independence
and does not survive conditioning on low ceilings.


## 4. Prospective test: frozen predictions, unmeasured outcomes

`docs/prereg/conjecture_future_predictions.csv`: all three domains × both spaces ×
{full future,20,000} × {(50,20,000),(500,20,000)} = **24 rows**. Eligible rows have
`split == "future"` and `adult == False`, matching the battery exclusion. Full N:
Steam 40,989; arXiv 41,000; patents 40,342. The same deterministic 20k row sample is
used in both spaces of a domain. New independent measurement seeds are fixed in
the script; the K=50 and K=500 selections will share one 20k-direction pool.

Frozen full-future predictions at K=50, M=20,000 (all 24 configurations in CSV):

| Domain / space | Old | Corrected |
|---|---:|---:|
| Steam / Qwen | .040257 | .042223 |
| Steam / bge | .013735 | .017303 |
| arXiv / Qwen | .059224 | .060004 |
| arXiv / bge | .026426 | .026329 |
| patents / Qwen | .064491 | .060042 |
| patents / bge | .025885 | .025393 |

The arXiv/Qwen correction changes sign between past and future. This is a real
prediction from changed corpus moments, not a transferred per-domain offset.

**Primary success (both conditions required, no row exclusions):** across all 24
rows median |corrected−observed| < median |old−observed|, **and every corrected
absolute error ≤0.005**. Secondary: all six cell biases, maximum absolute error and
fraction within .005. Failure on either primary condition falsifies this candidate
at the registered criterion; do not repair predictions after measurement.

The CSV includes old/corrected predictions, both ablations, moment-only selection
diagnostics, numerical replicate disagreement and SHA-256 of each normalised
corpus sample. The manifest hashes the prediction CSV and script. `--predict`
refuses to overwrite a frozen CSV. `--measure` requires the CSV, manifest and script
to match HEAD, checks fingerprints, and writes a distinct results CSV and verdict.
No `--measure` invocation was made in this round. Measurement is deliberately left
to the experimenter **after committing** predictions and code.

```sh
# Already run (corpus statistics only):
uv run python scripts/bounds/conjecture_future.py --retrospective
uv run python scripts/bounds/conjecture_future.py --predict
uv run python scripts/bounds/check_round3.py
uv run python scripts/bounds/conjecture_diagnostics.py
uv run python scripts/bounds/conjecture_numerics.py
# Formal checks, from lean/: lake build; lake env lean BoundsAxioms.lean
# Experimenter: commit the predictions, manifest and scripts, then:
uv run python scripts/bounds/conjecture_future.py --measure
```

**Verification and cost.** The 48-row retrospective took about 20 minutes and the
24-row future prediction about 10 minutes in concurrent local runs. Cached moments
avoid repeated eigendecompositions. Synthetic checks pass for mixture moment
preservation, the stable maximum CDF and reduction to Gaussian/isotropic cases.
Forecast hashes and all 24 unique finite rows were checked; no future-results file
exists. Maximum replicate half-range: .000085 retrospective / .000097 future.
Doubling the grid to 2^16 points per replicate on the six full-future K=50 cases
changes a prediction by at most **.000036**; frozen values were left unchanged.
The refinement clips inverse-normal inputs at half a 30-bit Sobol grid cell to
handle an exact zero in the larger grid; the frozen 2^15 runs had no such endpoint.
`conjecture_round3_numerics.csv` also records model-only conditional correlations.
Intermediate moments and logs live outside the repository in `runs/bounds3/`
under the configured data directory.

## Prospective result (measured 2026-10-24 after commit 76471fb froze the predictions)

24 future-split configurations (3 domains × 2 spaces × N ∈ {full, 20,000} × K ∈ {50, 500}, M = 20,000), never
measured before. Results: `docs/prereg/conjecture_future_results.csv`, verdict `conjecture_future_verdict.json`.

| | Old formula | Corrected |
|---|---:|---:|
| median absolute error | 0.0022 | 0.0018 |
| within 0.005 | 20 / 24 (the four misses: patents-Qwen, −0.006 to −0.008) | 24 / 24 (max 0.0032) |
| sign of error | mixed (bge-Steam positive) | negative in all 24 (≈ −0.002) |

**Pre-registered success criterion met.** The correction removes the patents-Qwen miss and the bge-Steam sign, but
is slightly worse where the old formula was already exact (Qwen-Steam −0.0002 → −0.0022), and it overpredicts
everywhere by about 0.002: a remaining systematic term it does not capture.
