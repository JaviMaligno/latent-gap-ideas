# BGE-M3 replication of the confirmatory results marked "not run" in Table 6

Pre-registered 2026-11-05, before any BGE-M3 vector of the texts below was computed or any BGE-M3 statistic of
these tests was looked at. No new text is generated: every test re-measures saved outputs in the replication space.
The Qwen3 results of all these tests are known (they are in the paper); what is not known is how they come out in
BGE-M3. Code: `scripts/strengthen_bge_embed.py` (texts and local embedding), `scripts/strengthen_bge_replication.py`
(analysis). Output: `docs/prereg/strengthen_bge_replication.json`.

## Encoder

BGE-M3 (`BAAI/bge-m3`), sentence-transformers 3.0.1, max_seq_length 512, L2-normalised, float32, run locally on
Apple MPS (checked against CPU on a probe batch: cosine 1.0). Corpus vectors (past and future arXiv / Steam items)
are the existing Kaggle BGE-M3 vectors (`prepared/<domain>/emb_bgem3.npy`); the continuation study's gpt-4o texts
(contexts, follow-ups, one-shot ideas, GEO / reviewer / ORACLE rounds) use the existing Kaggle vectors
(`kaggle-output/embed-continuation-v1/emb_bgem3.npy`), as in `continuation_reanalysis.py`. Texts embedded locally
here: the 720 decoded texts of the causal anchoring runs, all gpt-5.4-mini texts of `runs/continue-modern`
(150 contexts, 2,331 follow-ups, 1,500 one-shot ideas, 5,400 critique ideas) and the gpt-4o ORACLE and ORACLE-H
ideas (2,700; ORACLE is re-embedded so that H1 compares the two arms within one runtime). Local–Kaggle agreement is
reported on the texts embedded both ways (gpt-4o ORACLE ideas; gpt-5.4-mini contexts against their corpus rows).

## Part 1: causal anchoring (Section 3, `anchoring_causal.md`)

Same 12 cells (2 domains × 2 decoders × REAL/GAP/FAR), 20 targets each, frozen donors, same saved decoded texts.
Shown neighbours are the rows saved in each item (checked against the frozen Qwen3 neighbour sets); C = normalised
mean of their BGE-M3 vectors; e = BGE-M3 vector of the decoded text.

- **A (primary replication endpoint):** donor attraction `A_k = (e_swap − e_true)·(C_donor − C_true)`, mean over
  the 20 targets, t interval at the frozen one-sided level 0.05/24. **Replicates in a cell** if the lower bound is
  > 0. The frozen minimum of 0.05 is a Qwen3-scale number; whether A ≥ 0.05 is reported but is not the criterion.
- **D for REAL cells:** the target of a REAL cell is a real held-out item, so it has a BGE-M3 vector (the corpus row
  whose Qwen3 vector equals the target, cos ≥ 0.999, asserted). `D = e_true·t − e_swap·t`, same interval;
  replicates if lower bound > 0 and mean ≥ the frozen 0.04. The full frozen verdict ("causal anchoring supported")
  is applied to REAL cells.
- **GAP and FAR** targets are points of the Qwen3 space with no text, so D is not defined in BGE-M3 and is not run.
- **Overall:** the paper's claim "swapping moved the decoded text towards the donor's neighbours in 12 of 12
  cells" replicates if A's lower bound is > 0 in 12 of 12 BGE-M3 cells; 10–11 is reported as partial; fewer as
  not replicated. Descriptive: e·t true / swap / none in REAL cells.

## Part 2: gpt-5.4-mini and ORACLE-H (Section 5.4, `modern_and_oracle_history.md`)

Same estimands and code path as `modern_step_gap.py` and `oracle_history_analysis.py`, with every step, fit and
tangent recomputed from BGE-M3 vectors (the Qwen3 analysis used the step/fit stored at generation; here they are
recomputed). Tangent reference pool: BGE-M3 corpus items dated on or after the earliest follow-up of the run, minus
the context's follow-ups, the nearest |F_i| to the context. Bootstrap 2,000, seed 20261030.

- **M1:** gpt-5.4-mini CTX (and CTX+MEM) step gap > 0, Wilcoxon one-sided. Replicates if p < 0.05 after Holm.
- **M2:** GEO lengthens the step from round 0 to round 2. Replicates if mean > 0 and p < 0.05 after Holm.
- **M3:** ORACLE's follow-up-specific change is not significant at 0.05 (two-sided). Replicates if p ≥ 0.05;
  also reported: whether its 95 % CI lies within ±0.015 (the post-hoc bound of the paper).
- **H1:** gpt-4o ORACLE-H's fit gain over ORACLE, paired by context, one-sided. Qwen3: fails. The failure
  replicates if p ≥ 0.05.
- **H2:** ORACLE-H's follow-up-specific change (gpt-4o and gpt-5.4-mini), two-sided, with CI and the ±0.015 check.
- **M4:** H1 for gpt-5.4-mini.
- Holm over {M1, M2, M3, H1}, as pre-registered.
- Reported for every arm of Table 4: step lengthened, fit, alignment and follow-up-specific change.

## Part 3: continuation, A1–A3, B2, B4, C2, C4 (Section 5, A.8, `continue_from_context.md`, `iterative_critique.md`)

gpt-4o only (the generator of these pre-registered tests), all follow-ups, existing Kaggle BGE-M3 vectors; code
identical to `continue_from_context.part_a/part_b` and `iterative_critique.analyse` with the corpus and the texts
in BGE-M3 (directions from the 50 nearest past BGE-M3 neighbours; distractors and shown neighbours as BGE-M3 corpus
rows; same seed).

- **A1:** median follow-up step, with quartiles (descriptive).
- **A2:** momentum − hole projection > 0, Wilcoxon one-sided. Qwen3: −0.320, fails (design flaw). The failure
  replicates if the mean difference is < 0.
- **A3:** centre projection, two-sided. Qwen3: +0.427. Replicates if positive with p < 0.05.
- **B2:** CTX+NB ideas closer to the shown neighbours' centroid than to the follow-ups' centroid, one-sided.
  Qwen3: +0.008, p 0.023, Holm 0.068 (not significant). Reported with Holm within {A2, B1, B2, B3, B4}
  recomputed in BGE-M3; "replicates" means the same Holm decision (not significant at 0.05).
- **B4:** CTX+MOM hit > CTX hit, McNemar on context majorities plus bootstrap CI. Qwen3: −0.041, fails. The
  failure replicates if the mean difference is ≤ 0; the reversal replicates if additionally the CI excludes 0.
- **C2:** GEO hit change ≥ −0.03 (non-inferiority, round 0 → round 2). Qwen3: −0.11, fails. The failure
  replicates if the mean change is < −0.03 and the non-inferiority test is not significant.
- **C4:** reviewer hit change, two-sided, CI. Qwen3: −0.05 (CI −0.09 to −0.02). Reported; same sign with a CI
  excluding 0 counts as replicated.
- Holm over {C1, C2, C3, C5} recomputed in BGE-M3 and reported.

Anything else is exploratory and labelled as such. No parameter will be changed after the first result.
