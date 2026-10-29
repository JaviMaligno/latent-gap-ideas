import LatentGap.Bounds

/-!
# Coherent support: conditional certificates and an observation obstruction

Companion to docs/bounds5_support.md. `C` is a fixed operational coherence set;
none of these theorems assumes that measured perturbations cover it. All global
cover / mixture / residual hypotheses are explicit. The oracle extension theorem
is about information in observations, not a modification of the deployed encoder.
The probability and smooth-manifold arguments are proved in prose only.
-/

namespace LatentGap.Support

open RealInnerProductSpace Finset

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Supremum over a specified, nonempty-in-use set of texts. -/
noncomputable def support {α : Type*} (f : α → E) (C : Set α) (t : E) : ℝ :=
  sSup ((fun x => ⟪f x, t⟫) '' C)

/-- Pointwise upper certificates transfer to the supremum. -/
theorem support_le {α : Type*} (f : α → E) (C : Set α) (t : E)
    (hC : C.Nonempty) (b : ℝ) (h : ∀ x ∈ C, ⟪f x, t⟫ ≤ b) :
    support f C t ≤ b := by
  apply csSup_le (hC.image _)
  rintro y ⟨x, hx, rfl⟩
  exact h x hx

/-- An observed coherent witness is a lower bound, never an upper bound. -/
theorem witness_le_support {α : Type*} (f : α → E) (C : Set α) (t : E)
    (ht : ‖t‖ = 1) (hunit : ∀ x ∈ C, ‖f x‖ = 1) {x : α} (hx : x ∈ C) :
    ⟪f x, t⟫ ≤ support f C t := by
  apply le_csSup
  · refine ⟨1, ?_⟩
    rintro y ⟨z, hz, rfl⟩
    exact LatentGap.Bounds.unit_support_le_one (hunit z hz) ht
  · exact ⟨x, hx, rfl⟩

theorem support_le_one {α : Type*} (f : α → E) (C : Set α) (t : E)
    (hC : C.Nonempty) (ht : ‖t‖ = 1) (hunit : ∀ x ∈ C, ‖f x‖ = 1) :
    support f C t ≤ 1 := by
  exact support_le f C t hC 1 fun x hx =>
    LatentGap.Bounds.unit_support_le_one (hunit x hx) ht

/-- Target scores are 1-Lipschitz in embedding distance for a unit target. -/
theorem score_le_score_add_distance (e a t : E) (ht : ‖t‖ = 1) :
    ⟪e, t⟫ ≤ ⟪a, t⟫ + ‖e - a‖ := by
  have h := real_inner_le_norm (e - a) t
  rw [inner_sub_left, ht, mul_one] at h
  linarith

/-- A GLOBAL embedding cover, not a sampled edit radius, bounds support. -/
theorem cover_certificate {α ι : Type*} (f : α → E) (C : Set α)
    (a : ι → E) (t : E) (c radius : ℝ) (hC : C.Nonempty) (ht : ‖t‖ = 1)
    (ha : ∀ i, ⟪a i, t⟫ ≤ c)
    (hcover : ∀ x ∈ C, ∃ i, ‖f x - a i‖ ≤ radius) :
    support f C t ≤ c + radius := by
  apply support_le f C t hC
  intro x hx
  obtain ⟨i, hi⟩ := hcover x hx
  have hs := score_le_score_add_distance (f x) (a i) t ht
  linarith [ha i]

/-- One local chart certificate. `p` can be the orthogonal projection of t onto
the chart's tangent space; orthogonality and radii are explicit hypotheses. -/
theorem tangent_certificate (e a v r t p : E) (radius eps : ℝ)
    (he : e = a + v + r) (ht : ‖t‖ = 1) (hp : ⟪v, t - p⟫ = 0)
    (hv : ‖v‖ ≤ radius) (hr : ‖r‖ ≤ eps) :
    ⟪e, t⟫ ≤ ⟪a, t⟫ + radius * ‖p‖ + eps := by
  have hvt : ⟪v, t⟫ = ⟪v, p⟫ := by
    rw [inner_sub_right] at hp
    linarith
  have h1 := (real_inner_le_norm v p).trans
    (mul_le_mul_of_nonneg_right hv (norm_nonneg p))
  have h2 := real_inner_le_norm r t
  rw [ht, mul_one] at h2
  rw [he, inner_add_left, inner_add_left, hvt]
  linarith

/-- Corpus mixtures inherit a linear score ceiling before normalization. -/
theorem convex_mixture_score {ι : Type*} [Fintype ι] (a : ι → E)
    (w : ι → ℝ) (t : E) (c : ℝ) (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i, w i = 1) (ha : ∀ i, ⟪a i, t⟫ ≤ c) :
    ⟪∑ i, w i • a i, t⟫ ≤ c := by
  rw [sum_inner]
  simp only [real_inner_smul_left]
  calc
    ∑ i, w i * ⟪a i, t⟫ ≤ ∑ i, w i * c :=
      Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (ha i) (hw i)
    _ = c := by rw [← sum_mul, hsum, one_mul]

/-- Normalized-mixture certificate for the nonnegative-ceiling case.
`hm` may be supplied by convex_mixture_score; the residual must hold for the
actual output, not merely its observed neighbours. -/
theorem normalized_mixture_certificate (e m t : E) (c s0 eps : ℝ)
    (ht : ‖t‖ = 1) (hc : 0 ≤ c) (hs0 : 0 < s0) (hs : s0 ≤ ‖m‖)
    (hm : ⟪m, t⟫ ≤ c) (hr : ‖e - ‖m‖⁻¹ • m‖ ≤ eps) :
    ⟪e, t⟫ ≤ c / s0 + eps := by
  have hn : 0 < ‖m‖ := hs0.trans_le hs
  have hscore := score_le_score_add_distance e (‖m‖⁻¹ • m) t ht
  rw [LatentGap.Bounds.normalized_mean_score] at hscore
  have h1 : ⟪m, t⟫ / ‖m‖ ≤ c / ‖m‖ :=
    div_le_div_of_nonneg_right hm hn.le
  have h2 : c / ‖m‖ ≤ c / s0 := by
    apply (div_le_div_iff₀ hn hs0).2
    exact mul_le_mul_of_nonneg_left hs hc
  linarith

/-- An excess score imposes a necessary residual, even after normalization. -/
theorem residual_required (score c s0 eps : ℝ) (h : score ≤ c / s0 + eps) :
    score - c / s0 ≤ eps := by linarith

/-- Any polar vector supplies a certificate for unit vectors in a cone. -/
theorem cone_dual_certificate (q t z : E) (hq : ‖q‖ = 1)
    (hz : ⟪q, z⟫ ≤ 0) : ⟪q, t⟫ ≤ ‖t - z‖ := by
  have h := real_inner_le_norm q (t - z)
  rw [hq, one_mul, inner_sub_right] at h
  linarith

/-- The same dual certificate tolerates an explicit embedding residual. -/
theorem cone_tube_certificate (e q t z : E) (eps : ℝ) (hq : ‖q‖ = 1)
    (ht : ‖t‖ = 1) (hz : ⟪q, z⟫ ≤ 0) (hr : ‖e - q‖ ≤ eps) :
    ⟪e, t⟫ ≤ ‖t - z‖ + eps := by
  have h1 := cone_dual_certificate q t z hq hz
  have h2 := score_le_score_add_distance e q t ht
  linarith

/-- Nonnegative corpus mixtures and their normalizations preserve polar inequalities.
No assertion about the encoder of a concatenated text is hidden here. -/
theorem normalized_mixture_polar {ι : Type*} [Fintype ι] (a : ι → E)
    (w : ι → ℝ) (z : E) (hw : ∀ i, 0 ≤ w i) (ha : ∀ i, ⟪a i, z⟫ ≤ 0) :
    ⟪‖∑ i, w i • a i‖⁻¹ • (∑ i, w i • a i), z⟫ ≤ 0 := by
  have hm : ⟪∑ i, w i • a i, z⟫ ≤ 0 := by
    rw [sum_inner]
    simp only [real_inner_smul_left]
    exact Finset.sum_nonpos fun i _ => mul_nonpos_of_nonneg_of_nonpos (hw i) (ha i)
  rw [real_inner_smul_left]
  exact mul_nonpos_of_nonneg_of_nonpos (inv_nonneg.mpr (norm_nonneg _)) hm

/-- Ratio certificate: if t - beta*u is polar and beta is nonnegative,
unit corpus-cone vectors score at most beta. -/
theorem cone_ratio_certificate (q t u : E) (beta : ℝ) (hq : ‖q‖ = 1)
    (hu : ‖u‖ = 1) (hb : 0 ≤ beta) (hp : ⟪q, t - beta • u⟫ ≤ 0) :
    ⟪q, t⟫ ≤ beta := by
  have h := real_inner_le_norm q u
  rw [hq, hu, mul_one] at h
  rw [inner_sub_right, real_inner_smul_right] at hp
  have hm := mul_le_mul_of_nonneg_left h hb
  linarith

/-- Exact coherent-set obstruction: any unobserved coherent input can be sent
to t while preserving EVERY observation (even an infinite observed subset).
The function class here is unit-valued encoders; architecture restrictions are
not asserted to be preserved. -/
theorem unseen_coherent_extension {α : Type*} (f : α → E) (observed C : Set α)
    (z : α) (t : E) (hz : z ∈ C) (hunseen : z ∉ observed)
    (ht : ‖t‖ = 1) (hunit : ∀ x, ‖f x‖ = 1) :
    ∃ g : α → E, (∀ x ∈ observed, g x = f x) ∧
      (∀ x, ‖g x‖ = 1) ∧ support g C t = 1 := by
  classical
  let g : α → E := fun x => if x = z then t else f x
  have hgunit : ∀ x, ‖g x‖ = 1 := by
    intro x
    by_cases hx : x = z
    · simp [g, hx, ht]
    · simp [g, hx, hunit]
  refine ⟨g, ?_, hgunit, ?_⟩
  · intro x hx
    have hne : x ≠ z := by
      intro heq
      exact hunseen (heq ▸ hx)
    simp [g, hne]
  · apply le_antisymm (support_le_one g C t ⟨z, hz⟩ ht (fun x _ => hgunit x))
    have hl := witness_le_support g C t ht (fun x _ => hgunit x) hz
    simpa [g, real_inner_self_eq_norm_sq, ht] using hl

/-- A genuine probability-mass floor gives a finite cardinality bound. An
average perplexity or a sampled mass floor does not supply this hypothesis. -/
theorem mass_floor_cardinality {ι : Type*} [Fintype ι] (mass : ι → ℝ)
    (p : ℝ) (hfloor : ∀ i, p ≤ mass i) (htotal : ∑ i, mass i ≤ 1) :
    (Fintype.card ι : ℝ) * p ≤ 1 := by
  calc
    (Fintype.card ι : ℝ) * p = ∑ _i : ι, p := by simp
    _ ≤ ∑ i, mass i := Finset.sum_le_sum fun i _ => hfloor i
    _ ≤ 1 := htotal

end LatentGap.Support

#print axioms LatentGap.Support.support_le
#print axioms LatentGap.Support.witness_le_support
#print axioms LatentGap.Support.support_le_one
#print axioms LatentGap.Support.score_le_score_add_distance
#print axioms LatentGap.Support.cover_certificate
#print axioms LatentGap.Support.tangent_certificate
#print axioms LatentGap.Support.convex_mixture_score
#print axioms LatentGap.Support.normalized_mixture_certificate
#print axioms LatentGap.Support.residual_required
#print axioms LatentGap.Support.cone_dual_certificate
#print axioms LatentGap.Support.cone_tube_certificate
#print axioms LatentGap.Support.normalized_mixture_polar
#print axioms LatentGap.Support.cone_ratio_certificate
#print axioms LatentGap.Support.unseen_coherent_extension
#print axioms LatentGap.Support.mass_floor_cardinality
