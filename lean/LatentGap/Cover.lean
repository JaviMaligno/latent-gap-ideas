import LatentGap.Support

/-!
# Covered languages with cellwise encoder residuals

Companion to docs/bounds6_cover.md. The grammar supplies the set `L` and a
covering relation. Neither finite probes nor calibration supply the universal
residual hypothesis. The sampling/order-statistic arguments are in prose;
the elementary geometric and finite-mixture steps are proved here.
-/

namespace LatentGap.Cover

open RealInnerProductSpace Finset

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A normalized pair of corpus vectors stays in their positive cone. -/
theorem pair_polar (a b z : E) (w : ℝ) (hw : 0 ≤ w) (hw1 : w ≤ 1)
    (ha : ⟪a, z⟫ ≤ 0) (hb : ⟪b, z⟫ ≤ 0) :
    ⟪‖(1 - w) • a + w • b‖⁻¹ • ((1 - w) • a + w • b), z⟫ ≤ 0 := by
  have hm : ⟪(1 - w) • a + w • b, z⟫ ≤ 0 := by
    rw [inner_add_left, real_inner_smul_left, real_inner_smul_left]
    have h1 := mul_nonpos_of_nonneg_of_nonpos (sub_nonneg.mpr hw1) ha
    have h2 := mul_nonpos_of_nonneg_of_nonpos hw hb
    linarith
  rw [real_inner_smul_left]
  exact mul_nonpos_of_nonneg_of_nonpos (inv_nonneg.mpr (norm_nonneg _)) hm

/-- A positive corpus latitude rules out a zero mixture before normalization. -/
theorem pair_norm_lower (a b u : E) (w c : ℝ) (hu : ‖u‖ = 1)
    (hw : 0 ≤ w) (hw1 : w ≤ 1) (ha : c ≤ ⟪a, u⟫) (hb : c ≤ ⟪b, u⟫) :
    c ≤ ‖(1 - w) • a + w • b‖ := by
  have h1 := mul_le_mul_of_nonneg_left ha (sub_nonneg.mpr hw1)
  have h2 := mul_le_mul_of_nonneg_left hb hw
  have hn := real_inner_le_norm ((1 - w) • a + w • b) u
  rw [hu, mul_one, inner_add_left, real_inner_smul_left, real_inner_smul_left] at hn
  nlinarith

/-- One cell's conditional tube bound, including unit-sphere clipping. -/
theorem cell_certificate (e q t z : E) (eps : ℝ)
    (he : ‖e‖ = 1) (hq : ‖q‖ = 1) (ht : ‖t‖ = 1)
    (hz : ⟪q, z⟫ ≤ 0) (hr : ‖e - q‖ ≤ eps) :
    ⟪e, t⟫ ≤ min 1 (‖t - z‖ + eps) := by
  exact le_min (LatentGap.Bounds.unit_support_le_one he ht)
    (LatentGap.Support.cone_tube_certificate e q t z eps hq ht hz hr)

/-- A direction-specific residual hypothesis supplies a smaller certificate.
It is a separate hypothesis, not a consequence of fitting a small radius. -/
theorem directional_cell_certificate (e q t z : E) (delta : ℝ)
    (he : ‖e‖ = 1) (hq : ‖q‖ = 1) (ht : ‖t‖ = 1)
    (hz : ⟪q, z⟫ ≤ 0) (hr : ⟪e - q, t⟫ ≤ delta) :
    ⟪e, t⟫ ≤ min 1 (‖t - z‖ + delta) := by
  apply le_min (LatentGap.Bounds.unit_support_le_one he ht)
  have hqscore := LatentGap.Support.cone_dual_certificate q t z hq hz
  rw [inner_sub_left] at hr
  linarith

/-- A cover may use a different residual and polar certificate in every cell.
`B` bounds EVERY cell, not a favourable cell selected for the target. -/
theorem covered_language_certificate {α ι : Type*}
    (f : α → E) (L : Set α) (cell : ι → Set α)
    (q : ι → α → E) (z : ι → E) (eps : ι → ℝ) (t : E) (B : ℝ)
    (hL : L.Nonempty) (ht : ‖t‖ = 1) (hf : ∀ x ∈ L, ‖f x‖ = 1)
    (hcover : ∀ x ∈ L, ∃ i, x ∈ cell i)
    (hq : ∀ i x, x ∈ L → x ∈ cell i → ‖q i x‖ = 1)
    (hz : ∀ i x, x ∈ L → x ∈ cell i → ⟪q i x, z i⟫ ≤ 0)
    (hr : ∀ i x, x ∈ L → x ∈ cell i → ‖f x - q i x‖ ≤ eps i)
    (hB : ∀ i, ‖t - z i‖ + eps i ≤ B) :
    LatentGap.Support.support f L t ≤ min 1 B := by
  apply LatentGap.Support.support_le f L t hL
  intro x hx
  obtain ⟨i, hi⟩ := hcover x hx
  apply le_min (LatentGap.Bounds.unit_support_le_one (hf x hx) ht)
  exact (LatentGap.Support.cone_tube_certificate (f x) (q i x) t (z i) (eps i)
    (hq i x hx hi) ht (hz i x hx hi) (hr i x hx hi)).trans (hB i)

/-- Any fixed coherence filter inherits a pointwise language certificate. -/
theorem coherent_restriction {α : Type*} (f : α → E) (C L : Set α)
    (t : E) (B : ℝ) (hne : (C ∩ L).Nonempty)
    (h : ∀ x ∈ L, ⟪f x, t⟫ ≤ B) :
    LatentGap.Support.support f (C ∩ L) t ≤ B := by
  exact LatentGap.Support.support_le f (C ∩ L) t hne B fun x hx => h x hx.2

/-- A score violation necessarily violates the claimed residual for its anchor.
The converse is false: a large residual may be orthogonal to this target. -/
theorem score_failure_requires_residual (e q t z : E) (eps : ℝ)
    (hq : ‖q‖ = 1) (ht : ‖t‖ = 1) (hz : ⟪q, z⟫ ≤ 0)
    (hfail : ‖t - z‖ + eps < ⟪e, t⟫) : eps < ‖e - q‖ := by
  have h := LatentGap.Support.cone_tube_certificate e q t z ‖e - q‖ hq ht hz le_rfl
  linarith

omit [InnerProductSpace ℝ E] in
/-- Enlarging a calibrated radius preserves its residual event. -/
theorem residual_inflation (e q : E) (r s : ℝ) (hrs : r ≤ s)
    (hr : ‖e - q‖ ≤ r) : ‖e - q‖ ≤ s := hr.trans hrs

/-- A fixed mixture of cell laws preserves a common upper failure probability.
The probability interpretation needs the separately stated sampling hypotheses. -/
theorem mixture_risk_bound {ι : Type*} [Fintype ι]
    (w risk : ι → ℝ) (alpha : ℝ) (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i, w i = 1) (hr : ∀ i, risk i ≤ alpha) :
    ∑ i, w i * risk i ≤ alpha := by
  calc
    ∑ i, w i * risk i ≤ ∑ i, w i * alpha :=
      Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hr i) (hw i)
    _ = alpha := by rw [← sum_mul, hsum, one_mul]

/-- The elementary finite union budget; obtaining `hunion` requires a union
bound, and obtaining each risk bound requires the correct text sampling law. -/
theorem search_risk_budget {ι : Type*} [Fintype ι]
    (risk : ι → ℝ) (total alpha : ℝ)
    (hunion : total ≤ ∑ i, risk i) (hr : ∀ i, risk i ≤ alpha) :
    total ≤ (Fintype.card ι : ℝ) * alpha := by
  calc
    total ≤ ∑ i, risk i := hunion
    _ ≤ ∑ _i : ι, alpha := Finset.sum_le_sum fun i _ => hr i
    _ = (Fintype.card ι : ℝ) * alpha := by simp

end LatentGap.Cover

#print axioms LatentGap.Cover.pair_polar
#print axioms LatentGap.Cover.pair_norm_lower
#print axioms LatentGap.Cover.cell_certificate
#print axioms LatentGap.Cover.directional_cell_certificate
#print axioms LatentGap.Cover.covered_language_certificate
#print axioms LatentGap.Cover.coherent_restriction
#print axioms LatentGap.Cover.score_failure_requires_residual
#print axioms LatentGap.Cover.residual_inflation
#print axioms LatentGap.Cover.mixture_risk_bound
#print axioms LatentGap.Cover.search_risk_budget
