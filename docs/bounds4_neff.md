# Round 4A: the corpus-equivalent size of coherent search

2026-10-28. Offline analysis of seed-and-rewrite revision 3, with no new LLM,
embedding or judge calls. Primary model: the corrected latitude mixture in
[bounds3](bounds3.md). Earlier geometric limits remain as stated in
[bounds](bounds.md); none of the effective sizes below is a support bound.

**Result:** a single size fails in every domain/writer pair. The REAL-calibrated
mixture overpredicts GAP and FAR; the GPD alternative improves average error but
fails support and misses the Steam escapes. The results do not establish an
arm-independent “equivalent catalogue” for coherent search.

**Disclosure.** The experimenter had already measured the test results. The
requested [fluent-search document](prereg/fluent_search.md) and the task itself
revealed aggregate test results, including the Steam FAR escapes. This analysis
read those summaries before choosing its analysis protocol. It fitted sizes only
on `tune-v2/REAL-11..20`, froze per-target predictions, and only then opened the
individual `seed-rewrite/test` JSON outcomes. This is an **out-of-set check, not a
prospective validation**. The freeze is a local hash manifest, not a git commit;
the experimenter controls commits. It does not erase knowledge of the summaries.

## 1. THEOREM: conditional inversion, not inversion of the selected median

For a fixed unit target t and a continuous one-row projection CDF F_t, N
independent rows have maximum M_t with

\[
 P(M_t\le h\mid t)=F_t(h)^N,\qquad
 h_{p,t}(N)=F_t^{-1}(p^{1/N}).
\]

This is exact for integer N under the stated independence and distribution
hypotheses. For any positive real N, F_t^N is also a CDF, defining a continuous
effective-size interpolation. Choose **p=1/2**, the median maximum, not the
expected maximum. With 0<F_t(h)<1 the unique inversion is

\[
 \boxed{N_{\rm eff,t}(h)=\frac{\log(1/2)}{\log F_t(h)}},\qquad
 \boxed{\log_{10}N_{\rm eff,t}
 =\frac{\log\log 2-\log[-\log F_t(h)]}{\log 10}}.
\]

Proof: take logarithms of F_t(h)^N=1/2. For F_t(h)=1 there is no finite solution;
we report infinity. A value below one is a formal fractional equivalent, not a
literal catalogue. The script uses log-survival functions and `log1p`, avoiding
the catastrophic rounding F_t(h)=1 when its survival is merely very small.
For small survival S_t=1-F_t, N_eff ≈ log(2)/S_t.

In bounds3, H_N(c)=E_T F_T(c)^N was integrated over *unobserved random
directions* and inverted at K/(2M) to predict the median of selected directions.
Here t is **already known**. Reusing that selected-median equation, a mean
latitude, or tr(Σ)/d would double-count selection and discard target information.
No new pool of directions is selected in this analysis.

### EMPIRICAL LAW under test: retain the entire directional latitude mixture

Each domain uses the normalised, adult-filtered past Qwen corpus. For REAL,
remove its exact `refs.jsonl.hidden_pos` rows **before** computing its moments
and 32 latitude bins. There is one excluded row for each REAL target. GAP/FAR
use the full visible corpus. Let

\[
 r=\|\mu\|,\quad u=\mu/r,\quad \tau^2=u^T\Sigma u,\quad
 v=\Sigma u/\tau^2,\quad R=\Sigma-\tau^2vv^T.
\]

The equal-count bins of L=u·(X−μ) retain w_b, ℓ_b, ν_b and residual-energy ratio
s_b exactly as in bounds3. For each target separately compute

\[
 a_t=\mu^Tt=r(u^Tt),\quad g_t=v^Tt,\quad
 \sigma_t^2=t^T\Sigma t,\quad \rho_t^2=\sigma_t^2-\tau^2g_t^2,
\]
\[
 m_{tb}=a_t+g_t\ell_b,\quad
 s_{tb}^2=s_b\rho_t^2+\nu_b g_t^2,\qquad
 F_t(h)=\sum_b w_b\Phi((h-m_{tb})/s_{tb}).
\]

The weighted component means/variances reproduce μ·t and tᵀΣt. These are
checked numerically, including masking. `neff_models.csv` preserves the target
latitude, variance, component parameters and fingerprints. No scalar effective
dimension replaces directional variance. Gaussian components have unbounded
support; predictions above one are model failures and are not silently clipped.
The previous 24-configuration prospective success concerns selected FAR corpus
medians, not arbitrary REAL/GAP projection tails or coherent-search maxima.

### Uncertainty and what an interval means

**THEOREM, oracle maximum uncertainty.** Under the continuous iid maximum
model, Q_t=−log F_t(M_t) is exponential with rate N: NQ_t∼Exp(1). Consequently
the single-observation median inverse has N_eff/N=log(2)/E, E∼Exp(1), even if
every target shares exactly the same true N. A 95% confidence interval for N from
one oracle maximum h is

\[
 [ -\log(.975)/Q_t(h),\; -\log(.025)/Q_t(h)]
 = [0.03653,\;5.32193]N_{\rm eff,t}(h).
\]

This interval is exact only for the oracle iid experiment; it is not a confidence
statement about a deterministic guided-search budget. The frozen CSV separately
gives the **95% maximum prediction interval**
[F_t^{-1}(.025^(1/N)), F_t^{-1}(.975^(1/N))] at the fitted size, and a confidence
band on its median induced by the fitted-size interval. They are different objects.

**EMPIRICAL uncertainty.** Fit the median of target-level log10 N_eff on measured
tuning targets per domain/writer. There are ten except arXiv/Sonnet, with nine:
REAL-17 has no judge-passing outcome. This is a robust typical-size estimand, not the
maximum-likelihood exponential rate (which would be n/ΣQ). Resample targets
10,000 times with a fixed seed; report percentile 95% intervals for this median.
For the domain-only fit, first take the two writers' median log-size per target,
then resample the ten target clusters; do not count the two writers as independent
targets. Test-arm inverse sizes receive the same target bootstrap. Missing
judge-passing outcomes have no inverse; report coverage and condition their
medians on observed outcomes. Do not replace a missing observation by zero.

These intervals condition on the realised corpus, targets' construction and gate.
They do **not** include uncertainty in corpus moments, tail extrapolation,
correlated targets, judge error, or repeated-search randomness. With only ten
tuning targets, bootstrap endpoints are coarse; failure to reject is not evidence
of equivalence. The independent Opus check is a separate gate sensitivity.

**THEOREM, local sensitivity.** With density f_t and Q=−log F_t,

\[
 d\log N_{\rm eff} = \frac{dF_t}{F_t Q},\qquad
 \frac{d\log N_{\rm eff}}{dh}=\frac{f_t(h)}{F_t(h)Q_t(h)}.
\]

For uncertain parameters θ, insert dF=f dh+(∇_θ F)ᵀdθ and propagate their joint
covariance. Divide by log(10) for log10 sizes. In a rare tail this is approximately
−d log S: multiplicative tail errors become multiplicative size errors. Thus
moment precision or q99 agreement alone cannot certify an extrapolated N_eff.

## 2. Frozen hypotheses, alternatives and decision rules

**CONJECTURE A:** one size per domain reproduces REAL, GAP and FAR for both
writers. **CONJECTURE B (less restrictive):** one size per domain × writer does
so. No outcome from GAP/FAR, or REAL k=1..10, estimates either size. Both scopes
are frozen; failure of A cannot be hidden by reporting only B.

For each domain × writer, a common size survives this operational test only if
**all three arms** satisfy all of:

1. Measured coverage ≥90%.
2. Absolute difference between observed median h_coh and predicted median
   on those same observed targets ≤0.03. The full-set predicted median is also
   reported, so missingness is visible.
3. No tuning-versus-arm difference in median log10 inverse size at familywise
   α=.05: 19,999 two-sided target-label permutations, Holm over the 12
   domain × writer × arm comparisons, separately for each model/scope.
4. No observed value outside a fitted finite tail endpoint.

The .03 tolerance is a practical cosine tolerance matching the prior FAR
closeness scale; it is not the .005 corpus-ceiling criterion from bounds3.
The domain-only model survives only if both writers pass. Bootstrap intervals
are descriptive, not an extra overlapping-interval decision rule. Permutation
tests require exchangeability of the standardised inverse sizes under the
hypothesis; label dependence from target selection is not magically removed.
For either scope, compare each writer's arm with that writer's tuning inverses;
pooling writers for this test would change the null dispersion. The domain-only
predictions must additionally meet the same cosine tolerance for both writers.

**Alternative 1, arm-dependent effective size:** the diagnostic test-arm
inverses systematically differ from the REAL-only fit. These are post-freeze
diagnostics, never refitted forecasts. Rejecting a universal size does not prove
the search itself samples a different number of independent texts.

**Alternative 2, a different projection tail:** independently fit a generalised
Pareto distribution (GPD) by maximum likelihood to each target's corpus
exceedances above its 95th percentile, with that threshold fixed in advance:

\[
 S_t(h)=p_u[1+\xi_t(h-u_t)/\beta_t]^{-1/\xi_t},\quad h\ge u_t,
\]

with exponential limit at ξ=0 and finite endpoint u−β/ξ for ξ<0. Here p_u is
the empirical exceedance fraction. Below u, a Gaussian body normalised to join
F(u)=1−p_u makes the CDF complete. No search outcome fits u, ξ or β. In the tail,

\[
 h_p(N)=u+\frac{\beta}{\xi}
 \left[\left(\frac{1-p^{1/N}}{p_u}\right)^{-\xi}-1\right]
\]

(exponential limit when ξ=0). Refit only its scalar size on the same tuning REAL
targets, then freeze the same test predictions. A GPD **rescue** requires all
12 cells to survive and at least 20% lower pooled target MAE than the mixture.
A target above its fitted endpoint is an explicit tail failure, not an enormous
finite size. Report corpus q99/q999 tail ratios as **in-sample diagnostics**;
they do not validate extrapolation or independence.

**Alternative 3, failure of the iid maximum step:** on test, calculate
U_t=F_t(h_coh)^Nhat. Under the oracle model with known N, these are uniform.
The secondary KS diagnostic simulates 10,000 null experiments, including fitting
the median log-size on nine or ten fresh tuning maxima as applicable, then evaluates the test sample.
Holm-correct its 12 domain/writer/arm tests per tail model. This accounts for
finite tuning noise within the iid surrogate, but not moment or selection error.
Report oracle 95% maximum-interval coverage too. A good median with incorrect
dispersion does not establish that search behaves like iid corpus sampling.

**Identifiability limit.** Changed N, changed F_t and dependent/adaptive samples
can all change F_t(h)^N. One maximum per target cannot identify which is responsible
without further assumptions. Alternative-tail success would support that model,
not prove its asymptotic law. The geometric cap bound and the existence of unseen
texts do not follow from these fitted distributions.

## 3. Results

**EMPIRICAL RESULT: the universal-size conjecture fails for both domains, even
with a separate size for each writer.** All four REAL cells meet the primary
median/arm-consistency criterion; all eight GAP/FAR cells fail. The simpler
one-size-per-domain model has the same verdict. This is a failure of the combined
projection-tail/search-equivalence model, not a theorem about unreachable text.

The freeze was **2026-10-28 17:35:32 UTC**; measurement/audit completed at
**17:36:07 UTC**. SHA-256 of `neff_predictions.csv`:
`321a55fb8509fb9dedbd92cb70982fb66a4e15e65e0d94a0ad5c94ef36e73f9d`.
There are 800 forecast rows: 100 test targets × 2 writers × 2 tail models ×
2 pooling scopes. Of 200 writer/target outcomes, 187 have a gate-passing score.
No forecast or fit changed after the individual test outcomes were opened.

### REAL-only fitted sizes, frozen before comparison

Write N_eff=10^x. Every interval below is a 95% target-bootstrap interval for x;
exponentiate **both endpoints** for an interval on N. The per-domain figures pool
writers by target, while per-writer figures use 10 targets except arXiv/Sonnet (9).


| Domain | Writer | x = log10 N_eff | 95% interval for x |
|---|---|---:|---:|
| steam | Sonnet | 13.14 | [10.37, 16.24] |
| steam | Luna | 13.36 | [9.64, 15.88] |
| steam | pooled | 12.97 | [10.39, 15.89] |
| arxiv | Sonnet | 18.39 | [14.33, 24.71] |
| arxiv | Luna | 17.70 | [13.47, 21.09] |
| arxiv | pooled | 18.22 | [13.47, 21.50] |

### Frozen mixture predictions against test

Predicted column: full-set median; the median restricted to measured targets is
added in parentheses when it differs at three decimals. The latter determines
the .03 criterion. Arm x values are **diagnostic inversions of test**, not new
forecasts. “Pass” is a cell-level result; all three cells must pass for a common size.

| Domain | Writer | Arm | Measured | Predicted h | Observed h | Arm x [95% interval] | Holm p | Cell |
|---|---|---|---:|---:|---:|---:|---:|---|
| steam | Sonnet | REAL | 10/10 | 0.835 | 0.840 | 13.25 [10.99, 17.97] | 1.0000 | Pass |
| steam | Sonnet | GAP | 20/20 | 0.739 | 0.550 | 4.15 [3.13, 5.38] | 0.0006 | FAIL |
| steam | Sonnet | FAR | 20/20 | 0.133 | 0.053 | 4.80 [3.92, 6.53] | 0.0010 | FAIL |
| steam | Luna | REAL | 10/10 | 0.838 | 0.812 | 11.51 [8.81, 16.50] | 1.0000 | Pass |
| steam | Luna | GAP | 20/20 | 0.743 | 0.536 | 3.75 [3.14, 4.49] | 0.0006 | FAIL |
| steam | Luna | FAR | 20/20 | 0.135 | 0.068 | 6.09 [4.23, 7.33] | 0.0010 | FAIL |
| arxiv | Sonnet | REAL | 9/10 | 0.881 (0.873) | 0.866 | 16.99 [9.74, 22.40] | 1.0000 | Pass |
| arxiv | Sonnet | GAP | 12/20 | 0.802 (0.746) | 0.474 | 4.47 [3.93, 6.32] | 0.0025 | FAIL |
| arxiv | Sonnet | FAR | 18/20 | 0.188 | 0.031 | 2.78 [2.19, 4.66] | 0.0015 | FAIL |
| arxiv | Luna | REAL | 10/10 | 0.869 | 0.850 | 16.47 [10.84, 19.31] | 1.0000 | Pass |
| arxiv | Luna | GAP | 19/20 | 0.789 (0.793) | 0.434 | 3.24 [3.01, 4.21] | 0.0010 | FAIL |
| arxiv | Luna | FAR | 19/20 | 0.183 | 0.051 | 4.48 [2.94, 5.19] | 0.0010 | FAIL |

All eight non-REAL comparisons reject arm consistency (Holm p≤.0025).
arXiv/Sonnet GAP additionally fails coverage (12/20). It is particularly
important not to present its complete-case median as 100% coverage.

### Alternative GPD tail: improved errors, failed support

Its fitted x is 7.79 [5.18, ∞] / 7.97 [5.13, ∞] for Steam Sonnet/Luna,
and 7.80 [5.47, 14.65] / 7.95 [6.07, 11.81] for arXiv. Already in tuning,
8/39 measured outcomes exceed their target-specific fitted endpoints
(Steam 3+3; arXiv 1+1). The finite median fit does not cure those contradictions.

| Domain | Writer | Arm | Predicted h (all) | Observed h | Arm x [95% interval] | Above endpoint / measured |
|---|---|---|---:|---:|---:|---:|
| steam | Sonnet | REAL | 0.774 | 0.840 | ∞ [5.26, ∞] | 5/10 |
| steam | Sonnet | GAP | 0.554 | 0.550 | 7.88 [4.36, ∞] | 9/20 |
| steam | Sonnet | FAR | 0.058 | 0.053 | 9.49 [4.42, 13.12] | 3/20 |
| steam | Luna | REAL | 0.776 | 0.812 | 8.94 [4.74, 14.98] | 1/10 |
| steam | Luna | GAP | 0.555 | 0.536 | 5.40 [4.15, ∞] | 6/20 |
| steam | Luna | FAR | 0.059 | 0.068 | 10.42 [4.96, ∞] | 8/20 |
| arxiv | Sonnet | REAL | 0.818 | 0.866 | ∞ [6.22, ∞] | 5/9 |
| arxiv | Sonnet | GAP | 0.449 | 0.474 | ∞ [5.18, ∞] | 7/12 |
| arxiv | Sonnet | FAR | 0.072 | 0.031 | 2.92 [2.21, 5.50] | 0/18 |
| arxiv | Luna | REAL | 0.821 | 0.850 | 7.82 [5.20, ∞] | 3/10 |
| arxiv | Luna | GAP | 0.450 | 0.434 | 6.59 [3.61, 12.80] | 5/19 |
| arxiv | Luna | FAR | 0.072 | 0.051 | 4.61 [3.05, 8.04] | 2/19 |

Pooled target MAE falls from **0.1598 to 0.0744 (53.4% improvement)** for the
domain/writer fits, exceeding the frozen 20% improvement threshold. But **54/187
test observations exceed fitted GPD endpoints**; 8/39 already did in tuning.
Therefore the GPD does **not** rescue the common-size conjecture. Infinite
inverses are not dropped from medians or bootstraps. Permutation p-values are
left undefined when an input inverse is infinite; these are support failures,
not claimed zero p-values. No threshold is retuned after seeing test.

The unbounded extrapolations also produce impossible cosine forecasts: six
mixture and ten GPD domain/writer predictions exceed one (mixture max 1.078,
GPD max 2.847). They remain in the CSV and MAE. Clipping them would change the
frozen model, conceal its support error and not fix the non-REAL failures.

### Which part failed? Corpus tails and the maximum step

The following uses the *actual visible corpus size*, not fitted N_eff, and the
same test directions. Entries are medians over targets, not a median target
substituted in the formula. These corpus diagnostics were computed in `--fit`
without coherent-search test outcomes.

| Domain | Arm | Visible corpus N | Recorded ceiling | Mixture median maximum | GPD median maximum |
|---|---|---:|---:|---:|---:|
| Steam | REAL | 68,221 | .729 | .675 | .719 |
| Steam | GAP | 68,222 | .533 | .569 | .531 |
| Steam | FAR | 68,222 | .042 | .048 | .039 |
| arXiv | REAL | 71,999 | .739 | .586 | .725 |
| arXiv | GAP | 72,000 | .428 | .480 | .426 |
| arXiv | FAR | 72,000 | .055 | .062 | .055 |

At the empirical REAL q999, the mixture's survival probability is only **0.103×
the observed tail fraction in Steam, 0.00262× in arXiv** (target medians).
At GAP q999 it is **2.49× / 2.51×**, and at FAR q999 **1.32× / 1.24×**.
Thus the same moment mixture gives too little mass to REAL's high-score tail
and too much to GAP's, before extrapolating to search-level scores. This is
direct evidence of directional tail misspecification, not just a change in
search efficiency. GPD q999 ratios are about 1.02–1.09 across these cells, but
those are in-sample tail fits, not independent validation of their endpoints.

The recorded FAR targets were selected for low maxima on this very corpus:
their median realised ceiling need not equal the median of a fresh iid corpus
at each selected direction. Hence the FAR maximum table is a descriptive
diagnostic, not an independent falsification of the conditional maximum CDF.
The earlier 24 future configurations tested the *integrated selected-median*
formula H_N; they did not establish tail calibration for REAL/GAP or validity
at N=10^13–10^18. This distinction preserves the scope of the prior success.

The fitted maximum-distribution check also fails beyond the medians. For the
mixture, **all eight GAP/FAR PIT tests reject**, Holm p=.0012 (Monte Carlo
resolution), as does Steam/Luna REAL (p=.0244). The other REAL p-values are
.2113 (Steam/Sonnet) and .1500 for each arXiv writer. The nominal 95% maximum
prediction intervals at the point fit cover only **20–40% of measured REAL
outcomes**, **0% of GAP**, and **0% of FAR except Steam/Luna's 10%**. Those
coverage figures omit size-estimation uncertainty; the PIT simulations include
its iid-model counterpart. Non-rejection in three REAL cells is not evidence
that such low point-fit coverage is satisfactory.

**EMPIRICAL conclusion:** tail misspecification is demonstrable; the combined
tail-plus-iid-max search model fails. These data do not isolate a failure of the
mathematical identity F_t^N, which remains exact for iid draws, from failures of
its applicability to an adaptive search. A new tail law with valid bounded
support and new out-of-set calibration would be needed to separate those effects.
The GPD tested here is useful evidence about sensitivity, not that new law.

## 4. Meaning of the sizes and the FAR escape

Under the mixture's assumptions, Steam REAL search resembles the **median best
match in a catalogue of roughly 10^13.1–10^13.4 games**, and arXiv REAL resembles
10^17.7–10^18.4 papers. This is a score-equivalent catalogue drawn independently
from the specified *surrogate*, not a count of calls, candidates, reachable
descriptions or distinct ideas. The procedure made ten rewrite calls per target;
the seed search, target dependence and gate are included in its observed score.
Resampling the literal empirical catalogue could never beat its existing maximum:
there F_t(h)=1 above the ceiling and no finite inverse exists. A finite size for
an above-ceiling witness necessarily extrapolates an underlying population tail.

The arm inversions make the limitation concrete. Steam FAR gives about
**10^4.80 ≈ 63,000** equivalents for Sonnet and **10^6.09 ≈ 1.2 million** for
Luna. arXiv FAR gives **10^2.78 ≈ 600** and **10^4.48 ≈ 30,000**. In comparison,
the visible catalogues contain 68,222 games and 72,000 papers. This does not mean
Sonnet literally sampled 600 papers or that an above-ceiling GAP witness has
become worse than the corpus. The surrogate has different tail errors in these
directions. It does mean that the REAL-fitted 10^13–10^18 interpretation fails
to transfer by many orders of magnitude; allowing a writer-specific constant
does not repair it. The GPD's very different x values demonstrate that model
dependence directly.

For the established FAR escape threshold a_t=ceiling*_t+.05, the iid surrogate
predicts

\[
 P(M_t\ge a_t)=1-F_t(a_t)^{N_{\rm eff}}.
\]

Average those probabilities over all 20 fixed FAR targets. For the max over two
writers, independence is **not** assumed: per target the union probability lies
between max(p_1,p_2) and min(1,p_1+p_2). The CSV gives both bounds and the
independence calculation as a secondary illustration. Missing outcomes count
as no exhibited escape. At these fitted sizes both bounds round to the same
values below; every point-median forecast gives the same escape/no-escape decision.

| Domain | Writer | Mixture predicted escape | GPD predicted escape | Original gate observed | Also requiring Opus ≥7 |
|---|---|---:|---:|---:|---:|
| Steam | Sonnet | ≈100% | 0% | 5% (1/20) | 5% (1/20) |
| Steam | Luna | ≈100% | 0% | 25% (5/20) | 10% (2/20) |
| Steam | max of writers | ≈100% | 0% | **25% (5/20)** | **15% (3/20)** |
| arXiv | Sonnet | ≈100% | 0% | 0% | 0% |
| arXiv | Luna | ≈100% | 0% | 0% | 0% |
| arXiv | max of writers | ≈100% | 0% | **0%** | **0%** |

The mixture therefore predicts an escape on essentially every target in both
domains and fails both the frequency and domain contrast. Its frozen FAR score
medians (.133–.135 Steam, .183–.188 arXiv) are much larger than the observed
per-writer medians (.053/.068 and .031/.051). Even the lower 95% fitted-size
endpoints give FAR median forecasts around .102–.109 and .150–.157 respectively.

The GPD captures the zero arXiv escape but forbids every Steam escape: **all
40 FAR target-specific fitted endpoints lie below ceiling*+.05**, independently
of N. The smallest margin is .00680 in Steam and .00109 in arXiv. Three Steam
witnesses still clear the threshold under the independent Opus check, so that
contradiction is not removed by the original gate's self-preference sensitivity.
The CSV stores a numerical floor of about 5e-324 for zero probabilities; their
mathematical value outside a finite GPD endpoint is zero.

Opus was only applied to the original gate's selected witness. Its rejection is
not a measured replacement maximum over all lower-scoring rewrites; these are
filtered lower bounds and coverage changes. The max-over-writers Opus FAR
coverage is 19/20 in both domains. No gate was rerun in this analysis.

**Status:** common N_eff rejected under the registered practical criterion;
arm dependence established for this inversion, with strong tail-model
dependence; the alternative GPD improves fit but is refuted as a complete
description by support violations. The findings support neither a universal
corpus-equivalence law nor a universal impossibility of FAR escape.

## 5. Reproduction and audit

```sh
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
uv run --no-sync --no-cache python scripts/bounds/neff.py --check
# The following were executed in this order, once, before their output files existed:
uv run --no-sync --no-cache python scripts/bounds/neff.py --fit
uv run --no-sync --no-cache python scripts/bounds/neff.py --predict
uv run --no-sync --no-cache python scripts/bounds/neff.py --measure
```

The script sets the BLAS/OpenMP/Accelerate thread caps to four before importing
NumPy. `--no-sync --no-cache` uses the existing uv environment without updating
dependencies or the shared uv cache. It processes one domain at a time, uses
batched second moments and target projections, and does not allocate a full
float64 centred corpus plus residual copy. No external data file is written.

- [Script](../scripts/bounds/neff.py): separate fit/predict/measure entry points;
  prediction, fit and results files refuse overwrite. `--fit` cannot run after
  predictions exist. `--measure` validates code, fitted artifacts and source
  hashes before opening test outcomes. It does not require or create a commit.
- [Models](prereg/neff_models.csv), [tuning inverses](prereg/neff_tuning.csv),
  [fits](prereg/neff_fits.csv), [input hashes](prereg/neff_sources.csv).
- [Frozen per-target predictions](prereg/neff_predictions.csv) and
  [freeze manifest](prereg/neff_manifest.csv), including the criterion and disclosure.
- [Measured per-target results](prereg/neff_results.csv),
  [48-cell summary](prereg/neff_summary.csv), [escape comparison](prereg/neff_escape.csv),
  [test-input hashes and measurement timestamp](prereg/neff_audit.csv).

Checks cover inversion/quantile round trips through N=10^100, Gaussian reduction,
finite GPD endpoints, exact masked covariance and latitude-mixture moment
preservation. The 100 supplied `steam4b/full/targets.csv` references agree with
`refs.*` to <5e-11 (serialization rounding). Corpus hashes, target hashes, all
39 measured tuning outcomes and all 200 test records are accounted for. The
source `fluent_search.md` and `seed_rewrite.py` are unchanged by this analysis.
