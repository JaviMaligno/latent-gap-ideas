# Pre-registration: confirmatory replication of the critique study with gpt-4o, with the ±0.015 bound fixed

Written and committed 2026-11-06, before any context of this study was embedded or given to a generator. Scripts:
`scripts/strengthen_critique_replication.py` (context set, analysis) and `scripts/iterative_critique.py`
(generation, unchanged). Raw data: `runs/critique-replication/`. Output: `docs/prereg/strengthen_critique_replication.json`.

## Why

For gpt-4o the paper reads the follow-up-specific changes of the critique arms against an equivalence bound of ±0.015
that was chosen after the results (Sections 5.3, 5.4, abstract: "moves the ideas' follow-up-specific projection by at
most about 0.015"), and reads "no turn beyond revising with no information" from not-significant contrasts with
NEUTRAL (N5, N6). Both are post hoc on one sample of 150 contexts. This study repeats the four-arm design on 150 new
contexts with the bound, the tests and the reading rule fixed now, so that both statements become confirmatory or
are withdrawn.

What I know before this run: every result of the 150-context studies (Tables 2, 2b, 3, 4), the OLMo NEUTRAL arm, and,
for the new contexts, their follow-up counts (the fetch log of `strengthen_olmo_direction.md`) and the first row of
the file (seen while checking its format), never their labels or any generation.

## Contexts and follow-ups (fixed)

- The frame of `strengthen_olmo_direction.md` (`runs/olmo-direction/followups.jsonl`, in `frame_pos` order): arXiv
  cs.* papers dated 2021-2022, not adult, past split; follow-ups = citing arXiv papers with v1 date ≥ 2024-01-01
  (after gpt-4o's October 2023 cutoff) and an abstract, at most 50 per context (that study's seeded sample).
- **The first 150 contexts with ≥ 5 saved follow-ups**, in `frame_pos` order, copied once to
  `runs/critique-replication/followups.jsonl` (same row format as `runs/followups/followups.jsonl`). No label of that
  study is used here and no context is selected on labels; overlap with its eligible contexts is reported.
- Difference from the original 150, stated now: the original follow-ups were dated by Semantic Scholar publication
  date; these by arXiv v1 date, and the uncapped citing list (`citing_all.jsonl`) is available.

## Generation (fixed; `iterative_critique.py`, unchanged)

- gpt-4o-2024-08-06 through `gpt-4o-0806`, temperature 1; 3 ideas per context and arm, 3 rounds (initial + 2
  revisions); prompts, GEO band (0.574-0.661), reviewer prompt and ORACLE feedback exactly as Section 5.2.
- Arms GEO, LLM (Reviewer), ORACLE: `LG_RUN=critique-replication LG_FUPS=critique-replication
  LG_ITER_ARMS=GEO,LLM,ORACLE LG_TAG=critique-rep ALIAS=gpt-4o-0806 ... iterative_critique.py generate`.
- Then NEUTRAL with round 0 copied from GEO, as `neutral_revision.md`: the same with `LG_ITER_ARMS=NEUTRAL
  LG_ITER_R0_FROM=GEO LG_ITER_OUT=iter_ideas_neutral.jsonl LG_TAG=critique-rep-neutral`.
- Contexts and follow-ups embedded first, locally, with Qwen3-Embedding-0.6B (ONNX, `continue_from_context.py
  embed`), which also produces the ORACLE feedback and GEO's step. BGE-M3 vectors of contexts, follow-ups and ideas
  locally on CPU (settings of `revision_neutral.py embed-bge`).

## Measures

Per context, mean over its ideas with an abstract: step = cos(idea, context); fit = max cos to its follow-ups; hit as
Section 5.2 (nearest neighbour among follow-ups ∪ same-category not-past non-follow-up papers is a follow-up).
Direction exactly as `direction_tangent.py` (Table 3): tangent t of x = unit(x − (x·c)c); f = unit mean tangent of
the saved follow-ups; p = unit mean tangent of the k nearest (cosine to c) future-split corpus papers not among the
saved follow-ups, k = number of saved follow-ups; s = unit(f − (f·p)p). Per arm: alignment with f, change along p,
change along s, round 0 → round 2. Contexts without a valid round-0 and round-2 idea in an arm are dropped from that
arm's tests (counted).

## Confirmatory tests (Qwen3 primary; Holm over the 8, alpha 0.05)

- **R1 (N1):** GEO lengthens the step more than NEUTRAL (paired by context, one-sided Wilcoxon).
- **R2 (N2):** the Reviewer lengthens the step more than NEUTRAL (one-sided Wilcoxon).
- **E1-E3, bound on each critique arm's own turn:** for GEO, Reviewer and ORACLE, the follow-up-specific change
  Δs_A (round 2 − round 0) lies within ±0.015. Test: TOST with paired t over contexts, each one-sided test at 0.025
  (equivalent to the 95 % CI lying inside ±0.015), p_E = the larger one-sided p.
- **E4-E6, bound on what feedback adds:** for GEO, Reviewer and ORACLE, d_A = Δs_A − Δs_NEUTRAL lies within ±0.015,
  same TOST.

BGE-M3: the same 8 tests, own Holm family, reported as replication. CIs: 95 % bootstrap over contexts (2,000
resamples, seed 20261106) reported next to the t-based ones.

## Reading rule (fixed)

- E_k significant after Holm → "within ±0.015" is confirmed for that arm (or contrast).
- E_k not significant and the 95 % CI excludes 0 → a turn (or a feedback-specific turn) is detected; reported with its
  size; the paper's "at most about 0.015" / "no turn beyond revision" is withdrawn for that arm.
- E_k not significant and the CI includes 0 → insufficient; the paper says the bound is not confirmed.
- R1, R2: replicate or not the "critique lengthens the step beyond revision" result.
- The paper's gpt-4o statements are rewritten from this table; the original 150-context results stay as reported,
  labelled as the discovery sample.

## Secondary (reported, not confirmatory)

C1-C5 of `iterative_critique.md` on the new contexts; N3 (fit), N4 (hit), N5 (alignment) contrasts with NEUTRAL;
NEUTRAL's own changes; changes along p; positive control (follow-ups' leave-one-out alignment); a pooled estimate
over old and new contexts; the variant with p built excluding all citing papers (`citing_all.jsonl`); overlap with the
eligible contexts of `strengthen_olmo_direction.md`.

## Power

gpt-4o's per-arm Δs SD on the original contexts ≈ 0.05-0.06 and contrast SD 0.05-0.07 (`revision_neutral.json`). At
n = 150 the 95 % half-width is 1.96 · 0.06 / √150 ≈ 0.010, so equivalence at ±0.015 has about 80 % power when the
true value is within ±0.005 and the SD is 0.06 (less after Holm); a true change of 0.015, like GEO's original, will
not pass, by design.

## Cost

GEO 3 + Reviewer 5 (3 ideas + 2 reviews) + ORACLE 3 + NEUTRAL 2 calls per idea × 450 ideas = 5,850 gpt-4o calls. At
the NEUTRAL run's measured cost (900 calls, $3.27) ≈ $21 USD (≈ 32 AUD), 2026-11-06, paid spend on Azure
through the gateway, ledger tags `critique-rep` and `critique-rep-neutral`. Wall time ≈ 2-3 h after embedding.

## Clarification (recorded 2026-11-06, committed 19:17 UTC in 3eb74e5, during generation, before any idea of this study was embedded for analysis or any test computed; Codex review round 14)

The TOST wording above is inconsistent: "each one-sided test at 0.025 (equivalent to the 95 % CI lying inside
±0.015)" and "Holm over the 8, alpha 0.05" with p_E = the larger one-sided p would accept equivalence at a Holm
threshold of 0.05 when the 95 % t-interval crosses a bound. The registered intent is the 95 % reading, so the value
entering the Holm family for E1-E6 is 2 · p_E (capped at 1): with no other test in the family this is exactly "95 %
t-interval inside ±0.015". The reading rule is unchanged otherwise. Implemented in
`scripts/strengthen_critique_replication.py` in the commit that adds this section.
