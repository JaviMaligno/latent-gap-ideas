# Review follow-up: overlap between the gpt-4o replication and the OLMo-direction contexts

Registered 2026-11-14, before any re-analysis, in response to a finding made while designing
`review_posctrl.md`: `strengthen_critique_replication.json` records
`overlap_with_olmo_direction_eligible: 0`, but the code computes it as
`set(json.loads(eligible.json))`, and `eligible.json` is now a dict, so the set holds its keys, not context ids.
The designing agent reports that 17 of the 150 replication contexts are among the 150 OLMo-eligible contexts.
draft.md says the OLMo contexts are disjoint from the gpt-4o replication.

## Computation (script `scripts/review_sample_overlap.py`, output `docs/prereg/review_sample_overlap.json`)

1. Recompute the overlap correctly: the set of arXiv ids of the 150 replication contexts against the ids of the 150
   OLMo-eligible contexts actually generated (from the OLMo run's context list, not from `eligible.json`'s keys), and
   against the original discovery sample. Report counts and ids.
2. If the overlap is non-zero: rerun the registered confirmatory contrasts of both studies on the non-overlapping
   contexts only, with the same code paths and seeds as the originals (`strengthen_critique_replication.py`
   analyse/pooled for E1–E6 and R1/R2; `olmo_direction_analysis.py final` for the three primary contrasts), both
   embedders. Report the estimates, intervals and readings next to the published ones.

## Reading rule

- The paper's "disjoint" wording is corrected in every outcome to the true overlap.
- If every registered reading of both studies is unchanged on the non-overlapping contexts, the paper says so in one
  sentence. If any reading changes (detected ↔ insufficient ↔ bounded), the paper reports both and states that the
  replication is partly not independent of the OLMo test for those contexts; the abstract wording follows the more
  conservative reading.

No paid calls, no GPU; a few minutes locally.
