import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.MeasureTheory.Measure.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-!
# Constructive bounds (docs/bounds.md)

The original eleven results compiled with these focused imports on Mathlib v4.36.0-rc1,
2026-10-24; only propext, Classical.choice and Quot.sound in their axiom checks.
No `sorry` or extra axioms. The geometric/statistical interpretation and the spherical
cap integral are in the document, not silently included in these statements.

The quadratic certificate is stated in covariance eigen-coordinates. Positive
eigenvalues, a radius containing the output, and a unit-ball constraint are explicit.
-/

namespace LatentGap.Bounds

open RealInnerProductSpace Finset
open scoped ENNReal

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Every nonnegative multiplier supplies an upper bound on a latitude-constrained
unit vector. Optimising this certificate gives the exact spherical-cap support. -/
theorem cap_dual {x t u : E} {a l : ℝ} (hx : ‖x‖ = 1)
    (ha : a ≤ ⟪u, x⟫) (hl : 0 ≤ l) :
    ⟪t, x⟫ ≤ ‖t + l • u‖ - l * a := by
  have h := real_inner_le_norm (t + l • u) x
  rw [hx, mul_one, inner_add_left, real_inner_smul_left] at h
  have hmul := mul_le_mul_of_nonneg_left ha hl
  linarith

/-- For unit vectors, maximising landing is exactly minimising Euclidean distance.
This does not assert that a point of a geometric relaxation is a text embedding. -/
theorem unit_distance {x t : E} (hx : ‖x‖ = 1) (ht : ‖t‖ = 1) :
    ‖x - t‖ ^ 2 = 2 - 2 * ⟪x, t⟫ := by
  rw [norm_sub_sq_real, hx, ht]
  ring

/-- Exact anchoring decomposition. The final term is measured, not assumed zero. -/
theorem anchor_decomposition (e t c : E) :
    ⟪e, t⟫ = ⟪e, c⟫ * ⟪c, t⟫ + ⟪e - ⟪e, c⟫ • c, t⟫ := by
  rw [inner_sub_left, real_inner_smul_left]
  ring

/-- Normalising a neighbour mean amplifies its target score by the reciprocal norm. -/
theorem normalized_mean_score (m t : E) :
    ⟪(‖m‖ : ℝ)⁻¹ • m, t⟫ = ⟪m, t⟫ / ‖m‖ := by
  rw [real_inner_smul_left]
  ring

/-- Scalar square completion, used in the anisotropic support certificate. -/
theorem quadratic_young (a y q : ℝ) (hq : 0 < q) :
    a * y ≤ q * y ^ 2 + a ^ 2 / (4 * q) := by
  rw [← sub_le_iff_le_add', le_div_iff₀ (by positivity : 0 < 4 * q)]
  nlinarith [sq_nonneg (2 * q * y - a)]

/-- Rigorous dual bound for the intersection of an ellipsoid and a unit ball.
Here `y = x - m`, and `w` contains the positive covariance eigenvalues.
No numerical optimiser convergence assumption is needed. -/
theorem ellipsoid_ball_dual {ι : Type*} [Fintype ι]
    (t m y w : ι → ℝ) (R l η : ℝ)
    (hw : ∀ i, 0 < w i) (hl : 0 < l) (hη : 0 ≤ η)
    (hell : ∑ i, y i ^ 2 / w i ≤ R)
    (hball : ∑ i, (m i + y i) ^ 2 ≤ 1) :
    ∑ i, t i * (m i + y i) ≤
      (∑ i, t i * m i) + l * R + η * (1 - ∑ i, m i ^ 2) +
      ∑ i, (t i - 2 * η * m i) ^ 2 / (4 * (l / w i + η)) := by
  have hpt : ∀ i, t i * (m i + y i) ≤
      t i * m i + l * (y i ^ 2 / w i) +
      η * ((m i + y i) ^ 2 - m i ^ 2) +
      (t i - 2 * η * m i) ^ 2 / (4 * (l / w i + η)) := by
    intro i
    have hq : 0 < l / w i + η := add_pos_of_pos_of_nonneg (div_pos hl (hw i)) hη
    have h := quadratic_young (t i - 2 * η * m i) (y i) (l / w i + η) hq
    have heq : (l / w i) * y i ^ 2 = l * (y i ^ 2 / w i) := by ring
    rw [add_mul, heq] at h
    nlinarith
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ => hpt i)
  simp only [sum_add_distrib, sum_sub_distrib, ← mul_sum] at hs
  have h1 := mul_le_mul_of_nonneg_left hell hl.le
  have h2 := mul_le_mul_of_nonneg_left hball hη
  linarith

open MeasureTheory in
/-- Covering any region requires at least its measure divided by one cap's measure.
The exact spherical cap integral p_d(c) is supplied analytically in docs/bounds.md.
This union bound needs neither independence nor uniform placement of dictionary points. -/
theorem covering_measure_le {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) {n : ℕ} (A : Fin n → Set Ω) (R : Set Ω)
    (p : ℝ≥0∞) (hcover : R ⊆ ⋃ i, A i) (hcap : ∀ i, μ (A i) ≤ p) :
    μ R ≤ (n : ℝ≥0∞) * p := by
  calc
    μ R ≤ μ (⋃ i, A i) := measure_mono hcover
    _ ≤ ∑' i, μ (A i) := measure_iUnion_le A
    _ = ∑ i, μ (A i) := tsum_fintype _
    _ ≤ ∑ _i : Fin n, p := Finset.sum_le_sum (fun i _ => hcap i)
    _ = (n : ℝ≥0∞) * p := by simp

/-- Final normalisation alone gives only the trivial numerical support bound. -/
theorem unit_support_le_one {x t : E} (hx : ‖x‖ = 1) (ht : ‖t‖ = 1) :
    ⟪x, t⟫ ≤ 1 := by
  have h := real_inner_le_norm x t
  simpa [hx, ht] using h

/-- Counterexample building block: duplicating either antipodal atom never increases
its ceiling. The corpus can have mean zero for any even catalogue size. -/
theorem antipodal_ceiling (x t : E) :
    max ⟪x, t⟫ ⟪-x, t⟫ = |⟪x, t⟫| := by
  rw [inner_neg_left]
  by_cases h : 0 ≤ ⟪x, t⟫
  · rw [abs_of_nonneg h, max_eq_left (by linarith)]
  · have hn : ⟪x, t⟫ ≤ 0 := le_of_not_ge h
    rw [abs_of_nonpos hn, max_eq_right (by linarith)]

/-- Exact score ceiling for a two-point latitude corpus. -/
theorem two_point_latitude (r a s b : ℝ) :
    max (r * a + s * b) (r * a - s * b) = r * a + |s * b| := by
  by_cases h : 0 ≤ s * b
  · rw [abs_of_nonneg h, max_eq_left (by linarith)]
  · have hn : s * b ≤ 0 := le_of_not_ge h
    rw [abs_of_nonpos hn, max_eq_right (by linarith)]
    ring

/-- Uniform pointwise error controls threshold counts and hence every order statistic.
The probabilistic FAR proof supplies the pointwise event; this is its deterministic step. -/
theorem threshold_count_sandwich {ι : Type*} [Fintype ι]
    (x y : ι → ℝ) (eps v : ℝ) (h : ∀ i, |x i - y i| ≤ eps) :
    ((Finset.univ.filter (fun i => y i ≤ v - eps)).card ≤
      (Finset.univ.filter (fun i => x i ≤ v)).card) ∧
    ((Finset.univ.filter (fun i => x i ≤ v)).card ≤
      (Finset.univ.filter (fun i => y i ≤ v + eps)).card) := by
  classical
  constructor
  · apply Finset.card_le_card
    intro i hi
    have hv := (Finset.mem_filter.mp hi).2
    have he := (abs_le.mp (h i)).2
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ i, by linarith⟩
  · apply Finset.card_le_card
    intro i hi
    have hv := (Finset.mem_filter.mp hi).2
    have he := (abs_le.mp (h i)).1
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ i, by linarith⟩


/-- Score second moments retain the mean/variance decomposition in every direction.
The zero-sum residual hypothesis is explicit, as for an empirical centred corpus. -/
theorem directional_second_moment {ι : Type*} [Fintype ι]
    (a : ℝ) (y : ι → ℝ) (hy : ∑ i, y i = 0) :
    ∑ i, (a + y i) ^ 2 = (Fintype.card ι : ℝ) * a ^ 2 + ∑ i, y i ^ 2 := by
  simp_rw [show ∀ i, (a + y i) ^ 2 = a ^ 2 + 2 * a * y i + y i ^ 2 by
    intro i; ring]
  simp [sum_add_distrib, ← mul_sum, hy]

/-- A uniform one-score CDF error is amplified by at most N for an iid maximum.
This algebra does not assume or prove that corpus observations are independent. -/
theorem cdf_power_error (a b : ℝ) (ha : |a| ≤ 1) (hb : |b| ≤ 1) (n : ℕ) :
    |a ^ n - b ^ n| ≤ (n : ℝ) * |a - b| := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hp : |b ^ n| ≤ 1 := by
      rw [abs_pow]
      exact pow_le_one₀ (abs_nonneg b) hb
    have he : a ^ (n + 1) - b ^ (n + 1) =
        (a ^ n - b ^ n) * a + b ^ n * (a - b) := by ring
    rw [he]
    calc
      _ ≤ |(a ^ n - b ^ n) * a| + |b ^ n * (a - b)| := abs_add_le _ _
      _ = |a ^ n - b ^ n| * |a| + |b ^ n| * |a - b| := by rw [abs_mul, abs_mul]
      _ ≤ |a ^ n - b ^ n| + |a - b| := by
        nlinarith [abs_nonneg (a ^ n - b ^ n), abs_nonneg (a - b)]
      _ ≤ ((n + 1 : ℕ) : ℝ) * |a - b| := by
        push_cast
        nlinarith

/-- Model CDF bracketing transfers to the actual CDF at fixed endpoints.
With monotone continuous CDFs this brackets their p-quantiles. -/
theorem cdf_quantile_bracket (f g : ℝ → ℝ) (lo hi p eps : ℝ)
    (hlo : |f lo - g lo| ≤ eps) (hhi : |f hi - g hi| ≤ eps)
    (gl : g lo < p - eps) (gh : p + eps < g hi) :
    f lo < p ∧ p < f hi := by
  have hl := (abs_le.mp hlo).2
  have hh := (abs_le.mp hhi).1
  constructor <;> linarith

end LatentGap.Bounds
