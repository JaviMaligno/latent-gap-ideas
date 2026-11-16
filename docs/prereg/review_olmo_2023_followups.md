# Review follow-up: the OLMo step gap against citing papers first posted in 2023 (pre-registered)

Draft pre-registration, written 2026-11-15 before any 2023 citing paper is fetched or embedded. To be committed alone
to `docs/prereg/review_olmo_2023_followups.md` before the scripts run; deviations will be disclosed in a result
section appended later. No paid API, no LLM call, no GCP. Free APIs only (Semantic Scholar graph API, arXiv API).

## Review concern addressed

Reviewer M4: "papers that cited the source in 2022-2023 are excluded and are plausibly closer… A sensitivity with 2023
citing papers for OLMo (cutoff early 2023) would help." Section 5.1 restricts follow-ups to citing papers whose first
arXiv version is from 2024-01-01 on (leakage protection for gpt-4o-0806, cutoff October 2023). OLMo-7B-0424's data
(Dolma) stops in February–March 2023, so for OLMo citing papers first posted from April 2023 on are also unseen. If
the short step were a lag artefact (the field drifts away from a 2021-2022 context during the years before the
follow-ups appear), the gap should shrink markedly against the earliest unseen citing papers.

## What I know before this registration

The paper's numbers: gpt-4o step gap 0.149 (Qwen3) / 0.168 (BGE-M3) against all 2024+ follow-ups; OLMo 0.218 /
0.188; the exploratory by-year analysis (`docs/paper/step_gap_by_year.json`, gpt-4o only, year from Semantic Scholar's
`publicationDate`): Qwen3 0.153 / 0.154 / 0.155 for 2024 / 2025 / 2026, within-context slope of follow-up step on year
−0.0004 per year (CI −0.006 to +0.005). I have not computed any OLMo by-year gap, and I have not fetched or looked at
any citing paper dated before 2024.

## Data

- Contexts: the 150 discovery contexts of Section 5.1 (`continue_from_context.contexts()`, i.e. the first 150 rows
  with ≥ 5 follow-ups of `runs/followups/followups.jsonl`). Context vectors, follow-up vectors and idea vectors exactly
  as `scripts/continuation_reanalysis.load_generator` returns them (gpt-4o: `kaggle-output/embed-continuation-v1`;
  OLMo: `kaggle-output/llama-continuation-v1`, which holds the OLMo-7B-0424-Instruct run). Idea step = median cosine of
  the context's one-shot CTX ideas to the context (5 per context), as in `scripts/step_gap_by_year.py`.
- New 2023 follow-ups (script `scripts/review_olmo_2023_followups_fetch.py`): for each context, all citing papers
  from Semantic Scholar (`/paper/arXiv:<id>/citations`, the same endpoint and fields as `scripts/followups_fetch.py`,
  ≥ 1.1 s between requests, exponential back-off; no API key is set in this environment, unauthenticated) that have an
  arXiv id. **Window: first arXiv version from 2023-04-01 to 2023-12-31**, decided by the arXiv API's `published`
  field (= v1 date) for every candidate whose new-style id starts with 2303-2401 (the id's YYMM is the v1 submission
  month; the one-month margin on each side catches boundary cases, the `published` date decides). At most 50 per
  context, seeded subsample (`default_rng(20261030 + j)`, j = context index, as in `followups_fetch.py`). Title and
  abstract from the arXiv API.
- Embedding (kernel `kaggle/review-olmo-2023-embed`, a copy of `kaggle/embed-continuation/embed.py` with the same
  models, fp16, `max_seq_length = 512`, normalised): text `item_text(title, abstract)` (abstract cut at 1,200
  characters), exactly the follow-ups' recipe. Both Qwen3-Embedding-0.6B and BGE-M3. Kaggle rather than local so the
  vectors come from the same encoder settings as the existing follow-ups (and local RAM is in use).

## Measures

For generator g ∈ {OLMo-7B-0424 (primary), gpt-4o (comparison)} × embedder ∈ {Qwen3, BGE-M3}, per context i:

- gap₂₀₂₃(i) = idea step(i) − median cosine of the context's 2023 follow-ups to the context, for contexts with ≥ 1
  2023 follow-up (the same ≥ 1 rule as the by-year analysis).
- gap_Y(i) for Y = 2024, 2025, 2026 against the existing follow-ups whose **first arXiv version** falls in year Y
  (primary dating: the arXiv id's YYMM prefix, which is the v1 month; consistent with the 2023 window). The same
  quantities with Semantic Scholar's `publicationDate` year are reported as a reproduction of
  `step_gap_by_year.json`.
- Means over contexts with percentile bootstrap CIs over contexts, B = 10,000, `default_rng(20261115)` per
  generator × embedder.
- Paired: on contexts with both a 2023 and a 2024 follow-up, mean gap₂₀₂₃, mean gap₂₀₂₄, the difference and the ratio
  mean gap₂₀₂₃ / mean gap₂₀₂₄, with paired bootstrap CIs (same resampled contexts for both).
- Descriptive: number of 2023 follow-ups and contexts covered; within-context slope of follow-up step on v1 year over
  2023-2026 (context fixed effects, cluster bootstrap 2,000), as in `step_gap_by_year.py`.

## Decision rule (fixed now)

Primary cells: OLMo × {Qwen3, BGE-M3}. In each cell the short step is **"not a lag artefact"** if both hold:

1. mean gap₂₀₂₃ over all contexts with a 2023 follow-up is positive with its 95 % CI excluding 0, and
2. on the paired contexts, mean gap₂₀₂₃ ≥ 0.5 × mean gap₂₀₂₄ (point estimates; "not smaller than the 2024 gap by
   more than half").

Reading across the two cells: both pass → "not a lag artefact"; one passes → "mixed", reported per embedder; neither
→ "a lag contribution cannot be ruled out". If fewer than 30 contexts have a 2023 follow-up, the test is declared
underpowered and the paper says so, whatever the estimates.

gpt-4o is a comparison only, with no decision: its training cutoff (October 2023) overlaps the 2023 window, so 2023
citing papers may be in its training data and a smaller gap there is not evidence of drift.

Multiplicity: one decision per primary cell, two cells, combined by the conjunction above (no correction needed for a
conjunction); all other numbers are descriptive.

## What the paper says under each outcome

- **Not a lag artefact**: Section 5.1 (and the M4 answer in Appendix B) adds that against citing papers first posted
  April–December 2023, after OLMo's data cutoff, OLMo's gap is X (CI), against Y for 2024, so the restriction to 2024+
  follow-ups does not create the short step.
- **Mixed**: both numbers per embedder, with the statement that the evidence against a lag contribution holds in one
  embedder only.
- **Lag cannot be ruled out**: the limitation is stated in Section 5.1 and the discussion: part of the step gap may
  reflect how far a field has moved by the time the measured follow-ups appear; the gap against 2023 papers is
  reported.
- **Underpowered**: the counts are reported and the limitation stays as written.

## Result (2026-11-15)

Run: fetch 29 min (Semantic Scholar unauthenticated + arXiv API), embedding on Kaggle (`kaggle/review-olmo-2023-embed`,
the `embed-continuation` recipe unchanged), analysis 1 min locally. Output `docs/prereg/review_olmo_2023_followups.json`.

**Fetch.** 1,659 citing papers with first arXiv version 2023-04-01 to 2023-12-31; capped at 50 in 6 contexts → 1,204
follow-ups; 143 of 150 contexts have ≥ 1 (70 have ≥ 5).

**Step gap (mean over contexts, 10,000-resample percentile CI).** Year = first arXiv version.

| Generator / space | 2023 (n = 143) | 2024 (145) | 2025 (148) | 2026 (132) | paired 2023 vs 2024 (139): diff; ratio |
|---|---|---|---|---|---|
| OLMo / Qwen3 | 0.215 [0.200, 0.231] | 0.222 [0.208, 0.235] | 0.224 | 0.221 | −0.006 [−0.020, 0.009]; 0.975 [0.913, 1.042] |
| OLMo / BGE-M3 | 0.184 [0.175, 0.194] | 0.183 [0.173, 0.193] | 0.190 | 0.195 | +0.003 [−0.004, 0.010]; 1.015 [0.976, 1.056] |
| gpt-4o / Qwen3 (comparison) | 0.147 [0.133, 0.162] | 0.153 [0.139, 0.166] | 0.155 | 0.155 | −0.006 [−0.020, 0.009]; 0.963 [0.875, 1.062] |
| gpt-4o / BGE-M3 (comparison) | 0.166 [0.157, 0.175] | 0.164 [0.154, 0.173] | 0.170 | 0.177 | +0.003 [−0.004, 0.010]; 1.017 [0.974, 1.062] |

Year by Semantic Scholar date reproduces `step_gap_by_year.json` for gpt-4o/Qwen3 (0.153 / 0.154 / 0.155).
Within-context slope of follow-up step on v1 year, 2023-2026 (it involves follow-ups only, so it is the same for both
generators): −0.0041 per year [−0.0079, −0.0004] in Qwen3 and −0.0062 [−0.0083, −0.0041] in BGE-M3. Later citing
papers sit slightly *farther* from the context; this drift is about 0.004-0.006 per year, against a step gap of
0.15-0.22.

**Decision.** OLMo/Qwen3: c1 passes (CI excludes 0), c2 passes (0.216 ≥ 0.5 × 0.221). OLMo/BGE-M3: c1 and c2 pass.
Not underpowered (143 ≥ 30). **Verdict: not a lag artefact.** For gpt-4o (cutoff October 2023, overlapping the window)
the 2023 gap is also unchanged; that is descriptive only.

Deviations: none from the registration. Unregistered note: the measured drift (negative slope) is small, and it runs
the way a lag artefact would predict, but it is about a fortieth of the gap per year.
