import Mathlib

/-!
# Geometry of decoding far points (docs/geometry.md)

Formal versions of the exact statements the analysis rests on. Embeddings are unit vectors of a real inner product
space `E`; the empirical laws (efficiency, the fitted ceilings) are not here, only the identities and bounds.

* `inner_split`: landing splits exactly into an in-subspace and an out-of-subspace part.
* `landing_le`: Cauchy-Schwarz bound from the in-subspace fractions `m(e)`, `m(t)`.
* `ladder_norm_sq`, `ladder_inner`: the NOISE ladder's norm and overlap with its source, exact for any
  `ε = ⟪x, r⟫`; `ladder_cos`: the cosine when `ε = 0` (for a random `r`, `ε = O(1/√d)`, so this is an approximation).
* `sum_norm_sub_mean_sq`: unit vectors have total variance `1 - ‖μ‖²`.
* `sum_directional_variance`: summed over an orthonormal basis, directional variances give `1 - ‖μ‖²`, so the mean
  direction sees `(1 - ‖μ‖²) / d ≤ 1 / d` whatever the corpus.
* `expected_max_le`: `E[max_i X_i] ≤ m + σ √(2 log n)` under the explicit MGF bound
  `E[exp(s X_i)] ≤ exp(s m + σ² s² / 2)` (σ² is a variance proxy, not the variance); no independence needed.
* `expected_max_le_of_bounds`: the same with per-variable `m_i ≤ m` and proxies `σ_i ≤ σ`.
-/

open RealInnerProductSpace Finset

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Landing splits exactly: `e = a + a'`, `t = b + b'` with `a, b ∈ K` and `a', b' ∈ Kᗮ`. -/
theorem inner_split (K : Submodule ℝ E) {a b a' b' : E} (ha : a ∈ K) (hb : b ∈ K) (ha' : a' ∈ Kᗮ)
    (hb' : b' ∈ Kᗮ) : ⟪a + a', b + b'⟫ = ⟪a, b⟫ + ⟪a', b'⟫ := by
  have h1 : ⟪a, b'⟫ = 0 := Submodule.inner_right_of_mem_orthogonal ha hb'
  have h2 : ⟪a', b⟫ = 0 := Submodule.inner_left_of_mem_orthogonal hb ha'
  simp only [inner_add_left, inner_add_right, h1, h2]
  ring

/-- The Cauchy-Schwarz bound used as `bound` in geometry.py: for unit `e`, `t` with in-subspace parts of norms
`me = ‖a‖`, `mt = ‖b‖`, the landing is at most `me·mt + √(1-me²)·√(1-mt²)`. -/
theorem landing_le (K : Submodule ℝ E) {a b a' b' : E} (ha : a ∈ K) (hb : b ∈ K) (ha' : a' ∈ Kᗮ)
    (hb' : b' ∈ Kᗮ) (he : ‖a + a'‖ = 1) (ht : ‖b + b'‖ = 1) :
    ⟪a + a', b + b'⟫ ≤ ‖a‖ * ‖b‖ + √(1 - ‖a‖ ^ 2) * √(1 - ‖b‖ ^ 2) := by
  have pa : ‖a'‖ = √(1 - ‖a‖ ^ 2) := by
    have h := norm_add_sq_eq_norm_sq_add_norm_sq_real (Submodule.inner_right_of_mem_orthogonal ha ha')
    rw [he] at h
    rw [show 1 - ‖a‖ ^ 2 = ‖a'‖ ^ 2 by nlinarith, Real.sqrt_sq (norm_nonneg _)]
  have pb : ‖b'‖ = √(1 - ‖b‖ ^ 2) := by
    have h := norm_add_sq_eq_norm_sq_add_norm_sq_real (Submodule.inner_right_of_mem_orthogonal hb hb')
    rw [ht] at h
    rw [show 1 - ‖b‖ ^ 2 = ‖b'‖ ^ 2 by nlinarith, Real.sqrt_sq (norm_nonneg _)]
  rw [inner_split K ha hb ha' hb', ← pa, ← pb]
  exact add_le_add (real_inner_le_norm a b) (real_inner_le_norm a' b')

/-- NOISE ladder: `t ∝ (1-l)x + l r` with unit `x, r` and overlap `ε = ⟪x, r⟫`. The squared norm and the inner
product with `x` are exact; with `ε = 0` the cosine is `(1-l)/√((1-l)²+l²)`. -/
theorem ladder_norm_sq {x r : E} (hx : ‖x‖ = 1) (hr : ‖r‖ = 1) (l : ℝ) :
    ‖(1 - l) • x + l • r‖ ^ 2 = (1 - l) ^ 2 + l ^ 2 + 2 * (1 - l) * l * ⟪x, r⟫ := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right,
    real_inner_self_eq_norm_sq, real_inner_comm x r]
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, mul_pow, mul_pow, sq_abs, sq_abs, hx, hr]
  ring

theorem ladder_inner {x r : E} (hx : ‖x‖ = 1) (l : ℝ) :
    ⟪(1 - l) • x + l • r, x⟫ = (1 - l) + l * ⟪x, r⟫ := by
  simp only [inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq, hx, real_inner_comm x r]
  ring

theorem ladder_cos {x r : E} (hx : ‖x‖ = 1) (hr : ‖r‖ = 1) (hxr : ⟪x, r⟫ = 0) (l : ℝ) :
    ⟪(1 - l) • x + l • r, x⟫ / ‖(1 - l) • x + l • r‖ = (1 - l) / √((1 - l) ^ 2 + l ^ 2) := by
  have h := ladder_norm_sq hx hr l
  rw [hxr, mul_zero, add_zero] at h
  rw [ladder_inner hx l, hxr, mul_zero, add_zero, ← h, Real.sqrt_sq (norm_nonneg _)]

/-- Unit vectors `x i` with mean `μ` have total variance `1 - ‖μ‖²`. -/
theorem sum_norm_sub_mean_sq {n : ℕ} (hn : 0 < n) (x : Fin n → E) (hx : ∀ i, ‖x i‖ = 1) :
    (∑ i, ‖x i - (n : ℝ)⁻¹ • ∑ j, x j‖ ^ 2) / n = 1 - ‖(n : ℝ)⁻¹ • ∑ j, x j‖ ^ 2 := by
  set μ := (n : ℝ)⁻¹ • ∑ j, x j
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hsum : ∑ i, x i = (n : ℝ) • μ := by simp [μ, smul_smul, hnR]
  have key : ∀ i, ‖x i - μ‖ ^ 2 = 1 - 2 * ⟪x i, μ⟫ + ‖μ‖ ^ 2 := by
    intro i
    rw [@norm_sub_sq_real, hx i]
    ring
  simp only [key, sum_add_distrib, sum_sub_distrib, ← sum_mul, ← mul_sum, sum_const, card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one]
  rw [← sum_inner, hsum, real_inner_smul_left, real_inner_self_eq_norm_sq]
  field_simp
  ring

/-- Parseval for the corpus spread: for any orthonormal basis `b` of a finite-dimensional `E`, the variances of the
corpus along the basis directions sum to the total variance `∑ ‖x i - μ‖² / n`, which is `1 - ‖μ‖²` for unit vectors.
So a uniformly random basis direction sees on average `(1 - ‖μ‖²)/d ≤ 1/d`. -/
theorem sum_directional_variance [FiniteDimensional ℝ E] {ι : Type*} [Fintype ι] (b : OrthonormalBasis ι ℝ E)
    {n : ℕ} (hn : 0 < n) (x : Fin n → E) (hx : ∀ i, ‖x i‖ = 1) :
    ∑ k, (∑ i, ⟪x i - (n : ℝ)⁻¹ • ∑ j, x j, b k⟫ ^ 2) / n = 1 - ‖(n : ℝ)⁻¹ • ∑ j, x j‖ ^ 2 := by
  rw [← sum_norm_sub_mean_sq hn x hx, ← sum_div, sum_comm]
  congr 1
  refine sum_congr rfl fun i _ => ?_
  rw [← b.sum_sq_inner_left]

open MeasureTheory Real in
/-- Expected maximum of `n ≥ 2` sub-Gaussian scores: if every `X i` has `E[exp(s X i)] ≤ exp(s m + σ² s² / 2)` for
`s > 0`, then any integrable `M` lying pointwise below some `X i` (in particular `max_i X i`) has
`E[M] ≤ m + σ √(2 log n)`. This is the upper-bound form of the two-moment ceiling `μ·t + √(tᵀΣt) z_n`. -/
theorem expected_max_le {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P] {n : ℕ}
    (hn : 2 ≤ n) (X : Fin n → Ω → ℝ) (M : Ω → ℝ) (m σ : ℝ) (hσ : 0 < σ) (hM : Integrable M P)
    (hle : ∀ ω, ∃ i, M ω ≤ X i ω)
    (hint : ∀ i, ∀ s : ℝ, 0 < s → Integrable (fun ω => exp (s * X i ω)) P)
    (hmgf : ∀ i, ∀ s : ℝ, 0 < s → ∫ ω, exp (s * X i ω) ∂P ≤ exp (s * m + σ ^ 2 * s ^ 2 / 2)) :
    ∫ ω, M ω ∂P ≤ m + σ * √(2 * log n) := by
  have hlog : 0 < log n := log_pos (by exact_mod_cast (show 1 < n by omega))
  set s := √(2 * log n) / σ with hs_def
  have hs : 0 < s := div_pos (sqrt_pos.2 (by positivity)) hσ
  -- pointwise: exp (s M) ≤ ∑ exp (s X i)
  have hpt : ∀ ω, exp (s * M ω) ≤ ∑ i, exp (s * X i ω) := by
    intro ω
    obtain ⟨i, hi⟩ := hle ω
    calc exp (s * M ω) ≤ exp (s * X i ω) := exp_le_exp.2 (mul_le_mul_of_nonneg_left hi hs.le)
      _ ≤ ∑ j, exp (s * X j ω) :=
        Finset.single_le_sum (f := fun j => exp (s * X j ω)) (fun j _ => (exp_pos _).le) (Finset.mem_univ i)
  -- the integral of a finite sum of integrable functions is the sum of the integrals
  have hpart : ∀ t : Finset (Fin n), Integrable (fun ω => ∑ i ∈ t, exp (s * X i ω)) P := by
    intro t
    refine Finset.induction_on t (by simp) ?_
    intro j t hj ih
    simp only [Finset.sum_insert hj]
    exact (hint j s hs).add ih
  have hsplit : ∀ t : Finset (Fin n),
      ∫ ω, ∑ i ∈ t, exp (s * X i ω) ∂P = ∑ i ∈ t, ∫ ω, exp (s * X i ω) ∂P := by
    intro t
    refine Finset.induction_on t (by simp) ?_
    intro j t hj ih
    simp only [Finset.sum_insert hj]
    rw [integral_add (hint j s hs) (hpart t), ih]
  have hsum_int : Integrable (fun ω => ∑ i, exp (s * X i ω)) P := hpart Finset.univ
  have hexpM : Integrable (fun ω => exp (s * M ω)) P := by
    refine hsum_int.mono' ?_ (Filter.Eventually.of_forall fun ω => ?_)
    · exact continuous_exp.comp_aestronglyMeasurable (hM.aestronglyMeasurable.const_mul s)
    · rw [Real.norm_eq_abs, abs_of_pos (exp_pos _)]
      exact hpt ω
  -- Jensen: exp (s E M) ≤ E exp (s M)
  have hjensen : exp (∫ ω, s * M ω ∂P) ≤ ∫ ω, exp (s * M ω) ∂P :=
    convexOn_exp.map_integral_le continuous_exp.continuousOn isClosed_univ
      (Filter.Eventually.of_forall fun _ => Set.mem_univ _) (hM.const_mul s) hexpM
  have hupper : ∫ ω, exp (s * M ω) ∂P ≤ n * exp (s * m + σ ^ 2 * s ^ 2 / 2) := by
    calc ∫ ω, exp (s * M ω) ∂P ≤ ∫ ω, ∑ i, exp (s * X i ω) ∂P := integral_mono hexpM hsum_int hpt
      _ = ∑ i, ∫ ω, exp (s * X i ω) ∂P := hsplit Finset.univ
      _ ≤ ∑ _i : Fin n, exp (s * m + σ ^ 2 * s ^ 2 / 2) := Finset.sum_le_sum fun i _ => hmgf i s hs
      _ = n * exp (s * m + σ ^ 2 * s ^ 2 / 2) := by simp
  rw [integral_const_mul] at hjensen
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  -- take logs: s E M ≤ log n + s m + σ² s² / 2
  have hlogb : s * ∫ ω, M ω ∂P ≤ log n + (s * m + σ ^ 2 * s ^ 2 / 2) := by
    have := (hjensen.trans hupper)
    have h2 := log_le_log (exp_pos _) this
    rwa [log_exp, log_mul hn0.ne' (exp_pos _).ne', log_exp] at h2
  -- with s = √(2 log n)/σ, σ² s² / 2 = log n
  have hss : σ ^ 2 * s ^ 2 / 2 = log n := by
    rw [hs_def, div_pow, sq_sqrt (by positivity)]
    field_simp
  rw [hss] at hlogb
  have key : ∫ ω, M ω ∂P ≤ m + 2 * log n / s := by
    rw [← sub_le_iff_le_add', le_div_iff₀ hs]
    nlinarith [hlogb]
  calc ∫ ω, M ω ∂P ≤ m + 2 * log n / s := key
    _ = m + σ * √(2 * log n) := by
      rw [hs_def, div_div_eq_mul_div]
      have h2 : (2 * log n) = √(2 * log n) * √(2 * log n) := (mul_self_sqrt (by positivity)).symm
      have hsq : 0 < √(2 * log n) := sqrt_pos.2 (by positivity)
      field_simp
      nlinarith [h2]

open MeasureTheory Real in
/-- Heterogeneous version: each `X i` has its own mean bound `mi i ≤ m` and variance proxy `0 ≤ σi i ≤ σ`. -/
theorem expected_max_le_of_bounds {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    {n : ℕ} (hn : 2 ≤ n) (X : Fin n → Ω → ℝ) (M : Ω → ℝ) (mi σi : Fin n → ℝ) (m σ : ℝ) (hσ : 0 < σ)
    (hm : ∀ i, mi i ≤ m) (hσi : ∀ i, 0 ≤ σi i ∧ σi i ≤ σ) (hM : Integrable M P)
    (hle : ∀ ω, ∃ i, M ω ≤ X i ω)
    (hint : ∀ i, ∀ s : ℝ, 0 < s → Integrable (fun ω => exp (s * X i ω)) P)
    (hmgf : ∀ i, ∀ s : ℝ, 0 < s → ∫ ω, exp (s * X i ω) ∂P ≤ exp (s * mi i + σi i ^ 2 * s ^ 2 / 2)) :
    ∫ ω, M ω ∂P ≤ m + σ * √(2 * log n) := by
  refine expected_max_le hn X M m σ hσ hM hle hint fun i s hs => (hmgf i s hs).trans (exp_le_exp.2 ?_)
  have h1 : s * mi i ≤ s * m := mul_le_mul_of_nonneg_left (hm i) hs.le
  have h2 : σi i ^ 2 ≤ σ ^ 2 := pow_le_pow_left₀ (hσi i).1 (hσi i).2 2
  have h3 : σi i ^ 2 * s ^ 2 ≤ σ ^ 2 * s ^ 2 := mul_le_mul_of_nonneg_right h2 (sq_nonneg s)
  linarith

