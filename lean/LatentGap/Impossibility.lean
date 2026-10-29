import LatentGap.Support
import Mathlib.Data.Finset.Max
import Mathlib.Data.Finset.Card

/-!
# Finite coherent support and coherence-query certificates

Companion to docs/bounds7_impossibility.md, Round 7. The score function is fixed
throughout: these worlds change coherence, never the encoder. A finite set of
texts represents a predicate restricted to the finite admissible universe `V`.
`A` contains known corpus labels and all coherence queries; `P` contains its
positive labels. Encoder queries do not add coherence labels.

The real maximum requires nonemptiness. `UpperBound` is pointwise, so it also
handles the empty set without assigning an artificial real maximum to it.
The adaptive decision tree abstracts away computations with the fixed encoder.
No probability, hardness reduction, or experimental judgment is formalized here.

Written against the repository's Mathlib sources; NOT compiled locally.
-/

namespace LatentGap.Impossibility

open Finset

variable {α : Type*} [DecidableEq α]

/-- A maximum over a nonempty finite coherent set. -/
noncomputable def hStar (score : α → ℝ) (C : Finset α) (hC : C.Nonempty) : ℝ :=
  (C.image score).max' (hC.image score)

def UpperBound (score : α → ℝ) (C : Finset α) (b : ℝ) : Prop :=
  ∀ x ∈ C, score x ≤ b

theorem maximum_exists (score : α → ℝ) (C : Finset α) (hC : C.Nonempty) :
    ∃ x ∈ C, ∀ y ∈ C, score y ≤ score x :=
  C.exists_max_image score hC

theorem maximum_attained (score : α → ℝ) (C : Finset α) (hC : C.Nonempty) :
    ∃ x ∈ C, score x = hStar score C hC := by
  exact Finset.mem_image.mp ((C.image score).max'_mem (hC.image score))

theorem witness_le_hStar (score : α → ℝ) (C : Finset α) (hC : C.Nonempty)
    {x : α} (hx : x ∈ C) : score x ≤ hStar score C hC := by
  exact (C.image score).le_max' (score x) (Finset.mem_image_of_mem score hx)

theorem hStar_le_iff (score : α → ℝ) (C : Finset α) (hC : C.Nonempty) (b : ℝ) :
    hStar score C hC ≤ b ↔ UpperBound score C b := by
  constructor
  · intro h x hx
    exact (witness_le_hStar score C hC hx).trans h
  · intro h
    apply (C.image score).max'_le (hC.image score) b
    intro y hy
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.mp hy
    exact h x hx

/-- Agreement with every positive AND negative label on the observed set. -/
def Compatible (V A P C : Finset α) : Prop :=
  C ⊆ V ∧ ∀ x ∈ A, (x ∈ C ↔ x ∈ P)

/-- Zero-error certification: the bound holds in every compatible world. -/
def Certifies (V : Finset α) (score : α → ℝ) (A P : Finset α) (b : ℝ) : Prop :=
  ∀ C : Finset α, Compatible V A P C → UpperBound score C b

noncomputable def dangerous (V : Finset α) (score : α → ℝ) (b : ℝ) : Finset α :=
  V.filter (fun x => b < score x)

theorem positive_compatible {V A P : Finset α} (hPV : P ⊆ V) :
    Compatible V A P P :=
  ⟨hPV, fun _ _ => Iff.rfl⟩

/-- Adding one unobserved text preserves all observed labels. -/
theorem insert_compatible {V A P : Finset α} (hPV : P ⊆ V)
    {z : α} (hzV : z ∈ V) (hzA : z ∉ A) :
    Compatible V A P (insert z P) := by
  constructor
  · intro x hx
    rcases Finset.mem_insert.mp hx with hx | hx
    · simpa only [hx] using hzV
    · exact hPV hx
  · intro x hx
    have hne : x ≠ z := by
      intro heq
      exact hzA (heq ▸ hx)
    simp only [Finset.mem_insert, hne, false_or]

/-- Two nonempty worlds, the same fixed score and transcript, opposite answers
to the upper-bound question. The positive-label safety assumption is essential. -/
theorem two_worlds (score : α → ℝ) {V A P : Finset α} (hPV : P ⊆ V)
    (hP : P.Nonempty) {b : ℝ} (hsafe : UpperBound score P b)
    {z : α} (hzV : z ∈ V) (hzA : z ∉ A) (hzscore : b < score z) :
    ∃ C₀ C₁ : Finset α, Compatible V A P C₀ ∧ Compatible V A P C₁ ∧
      ∃ (h₀ : C₀.Nonempty) (h₁ : C₁.Nonempty),
        hStar score C₀ h₀ ≤ b ∧ b < hStar score C₁ h₁ := by
  have h₁ : (insert z P).Nonempty := ⟨z, Finset.mem_insert_self z P⟩
  refine ⟨P, insert z P, positive_compatible hPV,
    insert_compatible hPV hzV hzA, hP, h₁, ?_, ?_⟩
  · exact (hStar_le_iff score P hP b).2 hsafe
  · exact hzscore.trans_le
      (witness_le_hStar score (insert z P) h₁ (Finset.mem_insert_self z P))

/-- Exact characterization, including the empty-positive case via pointwise
bounds: all positives must be safe, and every dangerous text must be labelled. -/
theorem certifies_iff (V : Finset α) (score : α → ℝ) (A P : Finset α) (b : ℝ)
    (hPV : P ⊆ V) (hPA : P ⊆ A) :
    Certifies V score A P b ↔
      UpperBound score P b ∧ dangerous V score b ⊆ A := by
  constructor
  · intro h
    refine ⟨h P (positive_compatible hPV), ?_⟩
    intro z hz
    have hz' := Finset.mem_filter.mp hz
    by_contra hzA
    have hb := h (insert z P) (insert_compatible hPV hz'.1 hzA)
      z (Finset.mem_insert_self z P)
    exact (not_le_of_gt hz'.2) hb
  · rintro ⟨hsafe, hcover⟩ C hC x hx
    by_contra hnot
    have hdanger : x ∈ dangerous V score b :=
      Finset.mem_filter.mpr ⟨hC.1 hx, lt_of_not_ge hnot⟩
    have hp : x ∈ P := (hC.2 x (hcover hdanger)).1 hx
    exact hnot (hsafe x hp)

/-- Corpus labels are free. Every other dangerous text costs a distinct query. -/
theorem query_lower_bound (V : Finset α) (score : α → ℝ)
    (K Q P : Finset α) (b : ℝ) (hPV : P ⊆ V) (hPA : P ⊆ K ∪ Q)
    (hcert : Certifies V score (K ∪ Q) P b) :
    (dangerous V score b \ K).card ≤ Q.card := by
  have hcover := ((certifies_iff V score (K ∪ Q) P b hPV hPA).1 hcert).2
  apply Finset.card_le_card
  intro x hx
  have hx' := Finset.mem_sdiff.mp hx
  rcases Finset.mem_union.mp (hcover hx'.1) with hk | hq
  · exact False.elim (hx'.2 hk)
  · exact hq

theorem budget_impossibility (V : Finset α) (score : α → ℝ)
    (K Q P : Finset α) (b : ℝ) (q : ℕ) (hPV : P ⊆ V) (hPA : P ⊆ K ∪ Q)
    (hbudget : Q.card ≤ q) (hsmall : q < (dangerous V score b \ K).card) :
    ¬ Certifies V score (K ∪ Q) P b := by
  intro hcert
  exact (not_le_of_gt hsmall)
    ((query_lower_bound V score K Q P b hPV hPA hcert).trans hbudget)

/-- A terminating deterministic coherence-query certifier, after fixing the
encoder, corpus and random tape. Query branches are yes, then no. -/
inductive Certifier (α : Type*) where
  | halt (bound : ℝ) : Certifier α
  | query (text : α) (yes no : Certifier α) : Certifier α

namespace Certifier

def eval (T : Certifier α) (C : Finset α) : ℝ :=
  match T with
  | .halt b => b
  | .query x yes no => if x ∈ C then yes.eval C else no.eval C

def queries (T : Certifier α) (C : Finset α) : Finset α :=
  match T with
  | .halt _ => ∅
  | .query x yes no => insert x (if x ∈ C then yes.queries C else no.queries C)

def calls (T : Certifier α) (C : Finset α) : ℕ :=
  match T with
  | .halt _ => 0
  | .query x yes no => (if x ∈ C then yes.calls C else no.calls C) + 1

/-- Adaptivity cannot separate worlds agreeing on the first world's queries:
the second world follows exactly the same path and returns the same bound. -/
theorem run_eq_of_agreement (T : Certifier α) (C D : Finset α)
    (h : ∀ x ∈ T.queries C, (x ∈ C ↔ x ∈ D)) :
    T.eval C = T.eval D ∧ T.queries C = T.queries D := by
  revert h
  induction T with
  | halt b =>
    intro _
    exact ⟨rfl, rfl⟩
  | query x yes no ihYes ihNo =>
    intro h
    have hxiff := h x (by simp [queries])
    by_cases hx : x ∈ C
    · have hd : x ∈ D := hxiff.1 hx
      have hy : ∀ y ∈ yes.queries C, (y ∈ C ↔ y ∈ D) := by
        intro y hy
        apply h y
        simp only [queries, if_pos hx]
        exact Finset.mem_insert_of_mem hy
      obtain ⟨he, hq⟩ := ihYes hy
      simp only [eval, queries, if_pos hx, if_pos hd]
      exact ⟨he, congrArg (insert x) hq⟩
    · have hd : x ∉ D := fun hd => hx (hxiff.2 hd)
      have hn : ∀ y ∈ no.queries C, (y ∈ C ↔ y ∈ D) := by
        intro y hy
        apply h y
        simp only [queries, if_neg hx]
        exact Finset.mem_insert_of_mem hy
      obtain ⟨he, hq⟩ := ihNo hn
      simp only [eval, queries, if_neg hx, if_neg hd]
      exact ⟨he, congrArg (insert x) hq⟩

theorem queries_card_le_calls (T : Certifier α) (C : Finset α) :
    (T.queries C).card ≤ T.calls C := by
  induction T with
  | halt b => simp [queries, calls]
  | query x yes no ihYes ihNo =>
    by_cases hx : x ∈ C
    · simp only [queries, calls, if_pos hx]
      exact (Finset.card_insert_le x (yes.queries C)).trans
        (Nat.add_le_add_right ihYes 1)
    · simp only [queries, calls, if_neg hx]
      exact (Finset.card_insert_le x (no.queries C)).trans
        (Nat.add_le_add_right ihNo 1)

/-- Universal soundness on worlds sharing the initial corpus labels implies
transcript certification on the actual adaptive path. -/
theorem sound_run_certifies (T : Certifier α) (V K C : Finset α) (score : α → ℝ)
    (hsound : ∀ D : Finset α, D ⊆ V →
      (∀ x ∈ K, (x ∈ D ↔ x ∈ C)) → UpperBound score D (T.eval D)) :
    Certifies V score (K ∪ T.queries C) (C ∩ (K ∪ T.queries C)) (T.eval C) := by
  intro D hD
  have hagree : ∀ x ∈ K ∪ T.queries C, (x ∈ D ↔ x ∈ C) := by
    intro x hx
    have h := hD.2 x hx
    simpa only [Finset.mem_inter, hx, and_true] using h
  have heval := (run_eq_of_agreement T C D
    (fun x hx => (hagree x (Finset.mem_union_right K hx)).symm)).1
  have hb := hsound D hD.1 (fun x hx => hagree x (Finset.mem_union_left _ hx))
  rw [← heval] at hb
  exact hb

/-- The lower bound counts actual calls too, including repeated queries. -/
theorem sound_run_query_lower_bound (T : Certifier α) (V K C : Finset α)
    (score : α → ℝ) (hCV : C ⊆ V)
    (hsound : ∀ D : Finset α, D ⊆ V →
      (∀ x ∈ K, (x ∈ D ↔ x ∈ C)) → UpperBound score D (T.eval D)) :
    (dangerous V score (T.eval C) \ K).card ≤ T.calls C := by
  have hPV : C ∩ (K ∪ T.queries C) ⊆ V := by
    intro x hx
    exact hCV (Finset.mem_inter.mp hx).1
  have hPA : C ∩ (K ∪ T.queries C) ⊆ K ∪ T.queries C := by
    intro x hx
    exact (Finset.mem_inter.mp hx).2
  exact (query_lower_bound V score K (T.queries C) (C ∩ (K ∪ T.queries C))
    (T.eval C) hPV hPA (sound_run_certifies T V K C score hsound)).trans
      (queries_card_le_calls T C)

end Certifier

end LatentGap.Impossibility

#print axioms LatentGap.Impossibility.maximum_exists
#print axioms LatentGap.Impossibility.maximum_attained
#print axioms LatentGap.Impossibility.witness_le_hStar
#print axioms LatentGap.Impossibility.hStar_le_iff
#print axioms LatentGap.Impossibility.positive_compatible
#print axioms LatentGap.Impossibility.insert_compatible
#print axioms LatentGap.Impossibility.two_worlds
#print axioms LatentGap.Impossibility.certifies_iff
#print axioms LatentGap.Impossibility.query_lower_bound
#print axioms LatentGap.Impossibility.budget_impossibility
#print axioms LatentGap.Impossibility.Certifier.run_eq_of_agreement
#print axioms LatentGap.Impossibility.Certifier.queries_card_le_calls
#print axioms LatentGap.Impossibility.Certifier.sound_run_certifies
#print axioms LatentGap.Impossibility.Certifier.sound_run_query_lower_bound
