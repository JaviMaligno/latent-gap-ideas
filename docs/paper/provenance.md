# Provenance of the numbers in draft.md

Every number in the paper, by section, with the repository file it comes from. Moved out of the paper body on 2026-11-04 so the preprint carries no process notes.

## Abstract

- src: docs/prereg/anchoring_causal.md (Result: 12/12 supported); docs/geometry.md §12 and docs/design.md ("arXiv FAR ceiling, parameter-free": predicted 0.062, observed 0.058)
- src: docs/prereg/where_future_lands_controls.json (auc_no_cv: hole 0.449-0.540, momentum 0.728-0.920); docs/prereg/combination_vs_momentum_result.json (arms.V200_k3.N_candidates 1,313,400; diffs.FITMC_minus_FITM_auc_all 0.00015)
- src: docs/prereg/continue_from_context_result.json (A.n_contexts 150; B.B1.CTX.mean_step_gap 0.1494; iterative_critique.C5.fit_gain −0.0115); docs/prereg/direction_tangent.json (alignment_change > 0 in 12/12 cells; followups_loo_alignment 0.468 / 0.525)
## 1 Introduction

- src: docs/prereg/anchoring_causal.md; docs/prereg/where_future_lands.md; docs/prereg/combination_vs_momentum.md; docs/prereg/continue_from_context.md; docs/prereg/iterative_critique.md; docs/prereg/direction_tangent.json; docs/paper/related_work_step.md (Tang and Yang 2026, arXiv:2605.27905)
## 2 Setup

- src: docs/prereg/where_future_lands.md ("Data (frozen)"); docs/prereg/where_future_lands_result.json (sizes)
- src: docs/prereg/continue_from_context.md ("Data", amendment 2026-10-30, "Analysis rules"); docs/prereg/continue_from_context_result.json (A.n_contexts 150, A.n_followups 2883); scripts/continuation_reanalysis.py (docstring: "continuing" = influential or methodology/result intents; MIN_CONT = 3); docs/prereg/continuation_reanalysis.json (gpt-4o/qwen3e06/continuing.n_contexts 68); docs/paper/followup_check_s2.json (label_continuing 3, label_continuing_and_s2_continuing 0)
- src: docs/paper/pending.md (Second generator: OLMo, Dolma cutoff); docs/prereg/memory_probe.md (gpt-4o published cutoff October 2023)
## 3 Empty points are not ideas

- src: docs/prereg/anchoring_none_margin.json (range_all 0.029-0.075: condition none, own target minus donor target; scripts/anchoring_none_margin.py)
- src: docs/design.md (Decoder: 8 neighbours, score-only feedback, 4 iterations × 3 candidates)
- src: docs/prereg/anchoring_causal.md ("Frozen targets and interventions"; Result: 36 cells, 20/20, 720 target-runs, 12/12 supported; FAR e·t TRUE 0.019-0.066, SWAP/NONE −0.022 to −0.007; REAL NONE 0.362-0.465 vs TRUE 0.691-0.809); docs/prereg/anchoring_causal_result.json (cells[].quantitative.*.means.target)
- src: docs/prereg/coherent_search.md ("What the run does show": search vs random, arXiv FAR 0.022 vs 0.004, GAP 0.369 vs 0.350; Steam GAP 0.403 vs 0.383)
- src: docs/geometry.md §1 (0 ± 0.031; 0.41), §12 (0.058 vs 0.062); docs/design.md ("arXiv FAR ceiling, parameter-free", 0.062); docs/bounds.md §4b; docs/prereg/fluent_search.md (Rev 3 result, "Absolute scale": 0.09-0.14)
- src: docs/geometry.md §12 (PLAIND − INTERP Steam +1.16 to +3.19, arXiv +1.07, +3.01); docs/design.md (Patents outcome (6) +2.09, +4.83)
## 4 Gaps do not forecast where the future grows

- src: docs/prereg/where_future_lands.md ("Measures", K = 256, 200,000 draws, far threshold 0.15); scripts/where_future_lands.py (KMeans refitted per embedder)
- src: docs/prereg/where_future_lands_controls.json (auc_no_cv: n_top 26; hole 0.449-0.540, all CIs include 0.5; momentum 0.728-0.920; exploratory); docs/prereg/where_future_lands_result.json (F3: auc_momentum 0.705-0.917, gain −0.021 to +0.006, patents/bgem3 gain 0.0311, gain_ci [−0.058, 0.125])
- src: docs/prereg/where_future_lands_result.json (F2.partial_rho, F2.ci, F2.raw_rho_hole); docs/prereg/where_future_lands_controls.json (f2_numerators); docs/paper/related_work.md (Luo and Evans 2026)
- src: docs/prereg/where_future_lands_result.json (sizes.future 40,989 + 41,000 + 40,342 = 122,331; F1.far_count 0; F1.max_cos_future_min 0.41-0.57; F1.median_iso_diff 0.002-0.029); docs/prereg/where_future_lands_controls.json (cross_domain: share_far 0; out-of-domain min_max_cos 0.165-0.495; out-of-domain median_iso minus in-domain median_iso of the same past, 0.083-0.321)
- src: docs/prereg/combination_vs_momentum.md (Result: M within-novel 30.9x, AUC 0.962; POP 23.5x, AUC 0.953; COH AUC 0.164; H5 −0.0003 Qwen3); docs/prereg/combination_vs_momentum_result.json (arms.V200_k3.N_candidates 1,313,400; diffs.FITMC_minus_FITM_auc_all 0.00015); docs/paper/related_work.md (Alien Space: no momentum or popularity baseline)
## 5.1 The short step replicates

- src: docs/prereg/continue_from_context_result.json (B.B1.CTX mean_step_gap 0.1494, ci 0.138-0.160; B.B1.CTX+NB 0.140; B.holm B1 9.3e-26); docs/prereg/continuation_reanalysis.json (all 8 cells: B.CTX.mean_step_gap 0.129-0.218; B1_step_gap_CTX_p ≤ 1.9e-12)
## 5.2 Iterative critique (pre-registered)

- src: docs/prereg/iterative_critique.md (Design, Predictions; Result header: spend $16.06, 4,950 calls; correction 2026-11-01)
- src: docs/prereg/continue_from_context_result.json (iterative_critique.{GEO,LLM,ORACLE}.round0/round2.{step,fit,hit}; followup_step_median 0.620; C1 step_reduction 0.0868, ci 0.0795-0.0940, gap_closed 0.0868, step_gap_round0 0.148; C2 hit_change −0.109, ci −0.16 to −0.06; C3 0.0629, ci 0.0561-0.0693; C4 −0.053, ci −0.089 to −0.020; C5 fit_gain −0.0115, ci −0.0162 to −0.0067)
- src: docs/prereg/continuation_reanalysis.json (gpt-4o/qwen3e06/all.I.GEO: matched-paper fit = fit − excess, round0 0.6770 + 0.0377 = 0.7147, round2 0.6363 + 0.0450 = 0.6812, difference −0.034; the matching is out of support at round 0, see Appendix A.10, so this is indicative only)
## 5.3 Direction, measured independently of the step (exploratory)

- src: docs/prereg/direction_tangent.json (per cell: share_above_pool_max.round0 0.692 / 0.918 / 0.964 / 0.971, round2 0.400 / 0.736 / 0.790 / 0.840; followups_loo_alignment 0.468 qwen3e06, 0.525 bgem3; {GEO,LLM,ORACLE}.alignment_round0, alignment_round2, alignment_change_p ≤ 0.0054, towards_nearest_noncitingdir(_p), towards_followup_specific(_p)); scripts/direction_tangent.py (docstring: exploratory, raised in review; paired Wilcoxon; all follow-ups)
## 5.1 (continued) Second generator, OLMo

- src: docs/prereg/continuation_reanalysis.json (all 8 cells: B.CTX.mean_step_gap, B1_step_gap_CTX_p, I.{GEO,LLM,ORACLE}.step_reduction, fit_change, fit_change_p: 17 of 24 below 0.05; olmo-7b-0424/bgem3/continuing.I.GEO.fit_change +0.0027, p 0.57); docs/prereg/direction_tangent.json (alignment_change, towards_followup_specific per cell and arm); docs/paper/pending.md (Second generator: gpt-4o-mini ServiceModelDeprecating, Llama 3.1 access disabled, OLMo 2,048 window, prepended system text); kaggle/llama-continuation/run.py (parse: OLMo writes prose; N_IDEAS_ITER = 1)
## 5.4 A 2026 generator, and score feedback with history (pre-registered)

- src: docs/prereg/modern_and_oracle_history.md (design, pre-registered predictions M1-M4, H1-H2, Result section); docs/prereg/oracle_history.json (continue-from-context: ORACLE, ORACLE-H, H1_fit_gain_H_minus_plain; continue-modern: M1.CTX.mean_step_gap 0.211, M1.CTX+MEM 0.198, GEO, LLM, ORACLE, ORACLE-H step/fit/align/spec, H1_fit_gain_H_minus_plain 0.0006); scripts/oracle_history_analysis.py; scripts/modern_step_gap.py
## 6 The distance of inspiration does not help (pre-registered)

- src: docs/prereg/inspiration_from_anything.md (design frozen at 7ea8ae8; P1-P5; Result; BGE-M3 replication; Coherence; Stress test); docs/prereg/inspiration_from_anything_result.json (qwen3e06: INCIDENTAL/INSTRUCTED P1_margin_slope 0.0038/0.0053, p_pos 0.034/0.008; P1_two_lines hump false; P3_iso_slope 0.0216/0.0242; P2_eta2 0.0409 vs 0.0042, 0.0598 vs 0.0070; P4_uptake 0.218 vs 0.193, p 5.6e-27; P5 cells_beating_both 6; baselines NONE −0.0011, MOMENTUM −0.0020; by_source steam_game INSTRUCTED −0.0053; bgem3: P1 0.0015/−0.0008, P2 0.0172 vs 0.0074, 0.0156 vs 0.0075, P5 none; coherence: kappa_quadratic 0.623, INSTRUCTED slope −0.267 p_neg 0.014, INCIDENTAL −0.036; cli_codex/cli_claude margin_slope 0.0045/0.0016, steam_game −0.0090/−0.0101 p 0.028/0.032, time_share_none 0.13/0.0); isolation-controlled slopes 0.0015 / 0.0027 from the exploratory regression recorded in the prereg; docs/paper/related_work_inspiration.md (Fu et al. 2013; Gonçalves et al. 2013; Chan, Dow and Schunn 2015; Liu, Dubova et al. 2026, arXiv:2603.19087; Simonsohn 2018)
## 7 Related work

- src: docs/paper/related_work.md and docs/paper/related_work_step.md (annotated bibliographies with URLs, what was read for each item, threat ratings, and searches that found nothing). Reference list to be built from them.
- src: docs/paper/related_work_step.md (Claim 1: Tang and Yang 0.322 vs 0.410, 219,655 ideas; Claim 2: Tang and Yang AgentLaboratory gap −0.027, frontier alignment 0.250 vs 0.377, "does not substantially improve"; Liu et al. arXiv:2610.17439, 12-17 %, wild judge; SciMON; MOOSE-Chem 2.730 vs 2.578; reviewer-loop systems; Wang et al. 2026 arXiv:2603.27146; "Searched for and not found")
- src: docs/paper/related_work.md (HindSight ρ = −0.29; IdeaForecastBench; future-aligned proposals); docs/paper/related_work_step.md (PreScience; GIScholarBench; Kumar et al. 2025)
- src: docs/paper/related_work.md (Luo and Evans, Son et al., Science4Cast, Impact4Cast, Alien Space)
- src: docs/paper/related_work_inspiration.md (items 1-3, 5, 11, 13, 14)
- src: docs/paper/related_work.md (ELM, EAGLE, cone literature, homogenisation, "Searched for and not found"); docs/paper/related_work_step.md (vec2text: Table 7 noise result)
- TODO before submission (docs/paper/related_work.md caveats): confirm EAGLE and ELM venues; read Ball (2026) in full; primary sources for GPT-1900 and Talkie.
## 8 Limitations

- src: docs/paper/followup_check.md ("Claude's counts": CONTINUES 3, USES 0, BACKGROUND 25, UNCLEAR 2; "Second labeller and Semantic Scholar": Codex 2/30 plus 10 UNCLEAR, 27 of 30); docs/paper/followup_check_sample.json and docs/paper/followup_check_codex.json (κ computed from the 30 paired labels, binary CONTINUES vs not: observed 0.900, expected 0.847, κ = 0.348); docs/paper/followup_check_s2.json (s2_continuing 6; s2_continuing_by_label BACKGROUND 5, UNCLEAR 1; label_continuing_and_s2_continuing 0)
- src: docs/prereg/iterative_critique.md (GEO band leakage); docs/prereg/direction_tangent.json (12 cells × 3 tests)
## 9 Conclusion

- src: docs/prereg/where_future_lands_result.json; docs/prereg/where_future_lands_controls.json; docs/prereg/anchoring_causal.md; docs/prereg/continue_from_context_result.json; docs/prereg/direction_tangent.json; docs/prereg/oracle_history.json; docs/prereg/inspiration_from_anything_result.json
## A.1 Setup of the gaps study

- src: docs/prereg/where_future_lands.md ("Data (frozen)"); docs/prereg/where_future_lands_result.json (sizes: Steam reference 61,400 + heldout 6,822 = 68,222; future 40,989 / 41,000 / 40,342); docs/design.md (Domains table)
- src: docs/prereg/where_future_lands.md (4,225 adult rows; held-out slice); docs/design.md (Models; change log 2026-10-22)
- src: docs/design.md (Arms table; Decoder; change log decoder comparison 11/15; change log 2026-10-23 generator grid; Patents pre-registration; "Owner's criterion matrix": 80 ratings, 64 with judges); docs/glossary.md (Arms; pitch); docs/geometry.md §12 (two generators on arXiv)
- src: docs/prereg/where_future_lands.md ("Hypotheses and predictions"; "Analysis rules"); scripts/where_future_lands.py (cv_auc; F3 gain bootstrap)
## A.2 Full table of the future test

- src: docs/prereg/where_future_lands_result.json (all cells: F1.far_count, F1.max_cos_future_min, F1.median_iso_diff, F2.partial_rho, F3.auc_momentum, F3.auc_hole, F3.gain, F4.rank_biserial, F5.rho, F6.median_percentile); rounded as in the Result table of docs/prereg/where_future_lands.md, except patents/Qwen3 max-cos 0.4649, which that table rounds to 0.47
- src: docs/prereg/where_future_lands_result.json (F3.gain_ci; F4.rank_biserial; F5.n, F5.rho; F6.median_percentile, F6.matched_within_001 = 0, F6.median_gap_density 0.406-0.525, F6.median_heldout_density 0.636-0.696); docs/prereg/where_future_lands_controls.json (f2_numerators: steam/qwen 0.216, arxiv/qwen 0.186, steam/bge 0.314, arxiv/bge 0.315; patents/qwen 0.092 [−0.03, 0.22], patents/bge 0.144 [0.00, 0.28]); docs/prereg/where_future_lands.md (30,865 pending; 7,081 before the adult filter)
## A.3 Causal anchoring and the cap

- src: docs/geometry.md §5 (0.63-0.72 vs 0.06; 100 % FAR), §12 (arXiv 100 %); docs/design.md (Patents outcome (3)); docs/bounds.md §3 (identity)
- src: docs/prereg/anchoring_causal.md (Result table); docs/prereg/anchoring_causal_result.json (cells[].drop, cells[].attraction)
- src: docs/prereg/anchoring_causal.md (Result "Reading": fails in 7/12 cells, all FAR, 3 of 4 GAP); docs/geometry.md §11 ("0 of 300 items"), §12 (0 / 100 arXiv)
- src: docs/geometry.md §1 (0 ± 0.031; 0.41; ‖μ‖ 0.64; k = 232; 0.95, 0.48, 0.47, 0.957), §8 (10^44; fitted 10^3-3·10^4), §12 (arXiv 0.29, 0.54, 10^38, 0.058 vs 0.062); docs/design.md (patents ‖μ‖ 0.511; arXiv prediction 0.062; Steam 0.044 vs 0.046; patents 0.0625 ± 0.005, observed 0.062); docs/bounds.md §4 (formula), §4b (32/32, 30/32, 0.0024; patents 16/16, 9/16)
## A.4 Searching without neighbours

- src: docs/prereg/coherent_search.md (Result: C fails in every run; 44 evaluations); docs/prereg/fluent_search.md (Result: FAR 0.216 / 0.229; λ = 0.2 0.046 / 0.043; C 7/10 arXiv; revision 2 4/10 Steam; Revision 3: C powered, 10/10 REAL)
- src: docs/prereg/fluent_search.md (Rev 3 result table)
- src: docs/prereg/fluent_search.md (Rev 3 verdicts; "Sensitivity (not pre-registered): self-preference": 3/20 = 15 % with Opus; "Absolute scale")
## A.5 Non-certifiability

- src: docs/bounds7_impossibility.md §7.2 (two_worlds, certifies_iff, query_lower_bound), §7.3 (hardness reduction, prose), "Formal scope", "EMPIRICAL / STATED LIMIT — is M_b astronomical here?"; lean/LatentGap/Impossibility.lean
## A.6 Combination vs momentum: design and table

- src: docs/prereg/combination_vs_momentum.md ("Atoms (frozen)", "Two windows", "Scores per candidate triple", "Outcomes and metrics"); docs/prereg/combination_vs_momentum_result.json (arms.V200_k3: N_candidates, K, n_targets, n_novel_targets)
- src: docs/prereg/combination_vs_momentum.md (Result, primary-arm table); docs/prereg/combination_vs_momentum_result.json (arms.V200_k3.scores.*)
- src: docs/prereg/combination_vs_momentum.md (Decisions table H1-H5; Reading; "Secondary arm (pairs ...)": 2.93x; Limitations); docs/prereg/combination_vs_momentum_result.json (diffs.FITMC_minus_FITM_auc_all 0.00015; FITMCH_*_minus_FITMC_auc_all; coefs.FIT_MC.COH −0.2194, RAR −0.0694; reductions_applied; scores.POP.enr_all value 28.2, ci 31.1-31.8); Alien Space figures from docs/paper/related_work.md item 2
## A.7 Asking without a map, and my ratings against the judges'

- src: docs/geometry.md §12 (PLAIND − INTERP Steam +1.16 to +3.19, arXiv CIs; PLAIND pair cos 0.534 / 0.504); docs/design.md (Patents outcome (6) with CIs; "PLAIN collapses": 0.864, 0.404, 0.35-0.49, 45/50; GAP outcome 4.91 vs 5.52)
- src: docs/design.md ("Item agreement is weak everywhere after batch 1": −0.32, 0.36, 0.34, 0.13; patents −1.2 to −3.0; "Human batch P1": 3/4 FAR, 1/3 GAP, 0/5 PLAIND; "Reading: the judges' novelty rewards ..."); docs/geometry.md "Human batch 3" (7.00 vs 5.75, p = 0.105)
- src: docs/prereg/memory_probe.md ("Style-free anticipation ... exploratory": 0.58, 0.50, p 0.002 / 0.020, gpt-4o 0.38 p 0.198, published cutoff October 2023); scripts/anticipation_normalised.py (--future-from 2024-01-01)
## 8 Limitations (gaps-study paragraphs, formerly an appendix)

- src: docs/design.md ("Judge criteria are not independent": coherence ~ feasibility 0.35 / 0.62 / 0.63, ~ appeal 0.41 / 0.66 / 0.79); docs/prereg/where_future_lands_result.json (F5.n patents 9,477); docs/prereg/where_future_lands.md (30,865 pending); docs/paper/pending.md (A: 8,640 calls, ≈ 12 h; B: 16 items, ≥ 3 raters); docs/prereg/combination_vs_momentum.md (Limitations)
## A.8 Continuation study: how real follow-ups move, and the one-shot arms

- src: docs/prereg/continue_from_context_result.json (A: A1_step_median 0.620, A1_step_quartiles 0.574-0.661, A2_mean_proj_momentum 0.0162, A2_mean_proj_hole 0.3364, A2_mean_diff −0.320, A2_diff_ci, A2_holds false, A3_mean_proj_centre 0.427, A3_ci; A_baseline: median_abs_step_mismatch 0.00049, centre 0.427 vs 0.471 p 3e-23, hole 0.336 vs 0.350 p 5.8e-5, momentum 0.0162 vs 0.0147 p 0.61); scripts/continue_from_context.py (directions()); scripts/continue_from_context_baseline.py (docstring: exploratory)
- src: docs/prereg/continue_from_context_result.json (B.{CTX,CTX+NB,CTX+MEM,CTX+MOM}.mean_step_gap, mean_fit, mean_hit, mean_div)
- src: docs/prereg/continue_from_context_result.json (B.holm B1 9.3e-26, B2 0.068, B3 2.7e-20, B4 1.0; B.B2 mean_diff 0.0080, ci, p 0.023; B.B3 0.767 vs 0.824; B.B4 hit_MOM 0.617, hit_CTX 0.659, mean_diff −0.041, ci, mcnemar_b 8, mcnemar_c 15, fit_MOM 0.685, fit_CTX 0.680, best_arm_by_hit CTX+NB; B_centre: followups 0.427, CTX 0.129 to CTX+MOM 0.152); docs/prereg/continuation_reanalysis.json (B3_div_MEM_vs_CTX_p ≤ 1.5e-9 in the four gpt-4o cells; olmo-7b-0424/qwen3e06/all.B.CTX.mean_div 0.826, CTX+MEM 0.808); scripts/continue_from_context.py (Part B scoring)
- src: docs/paper/pending.md (Second generator: what was tried — gpt-4o-mini ServiceModelDeprecating, Llama 3.1 access disabled, OLMo 2,048 window, prepended system text); kaggle/llama-continuation/run.py (parse: OLMo writes prose; N_IDEAS_ITER = 1)
- src: docs/prereg/encoder_agreement.json (ctx 150, fup 2883, partB 3000 texts; min_cos 0.999994-0.999996), produced by scripts/encoder_agreement.py; reproduction: docs/prereg/continuation_reanalysis.json gpt-4o/qwen3e06/all.I.GEO.step_reduction 0.08684 vs docs/prereg/continue_from_context_result.json iterative_critique.C1.step_reduction 0.08682
## A.9 Why a distance-matched baseline is not identified

- src: docs/prereg/direction_tangent.json (share_above_pool_max.round0 / round2 per cell: 0.692/0.400, 0.964/0.790, 0.918/0.736, 0.971/0.840); scripts/direction_tangent.py (docstring: "the distance-matched baseline of iterative_critique is out of support at round 0"); docs/prereg/continuation_reanalysis.json (I.*.excess, excess_change, excess_change_p, kept for the record); docs/prereg/continue_from_context_result.json (A_baseline.median_abs_step_mismatch 0.00049)

## Added in the preprint pass (2026-11-04)

- Section 3, far targets vs future: docs/prereg/far_targets_vs_future.json (scripts/far_targets_vs_future.py; future_max_cos_to_target median 0.046 / 0.066 / 0.064, max 0.069 / 0.100 / 0.080; past ceiling median 0.046 / 0.058 / 0.062).
- Section 3, transport law 5 of 12: docs/prereg/anchoring_causal.md (Result: holds in all REAL cells and one GAP cell).
- Section 5 data, 2025 contexts: runs/followups-2025/followups.jsonl (680 contexts fetched; first 150 with >= 5 follow-ups, 2,331 follow-ups); docs/prereg/modern_and_oracle_history.md ("Contexts", "Follow-ups").
- Section 5.3 / Table 3 alignment p-values: docs/prereg/direction_tangent.json (alignment_change_p); decomposition from scripts/direction_tangent.py (tang, unit, ts = unit(t - (t·tp) tp)).
- Section 6 / Table 5 Holm p and outcomes: docs/prereg/inspiration_from_anything_result.json (holm_INCIDENTAL, holm_INSTRUCTED, P4_uptake, P5); predictions from docs/prereg/inspiration_from_anything.md ("Predictions"); steam_game ranks from by_source; coherence sample from scripts/inspiration_coherence.py.
- Table A1 direct AUCs: docs/prereg/where_future_lands_controls.json (auc_no_cv).
- Table A2 CIs: docs/prereg/anchoring_causal_result.json (cells[].drop, cells[].attraction).
- Figures 1-5: scripts/paper_figures.py, from the files above and runs/inspiration/gpt-4o-0806/scored_{qwen3e06,bgem3}.parquet.

## Added in the review pass (2026-11-04, Codex review)

- Section 3, REAL margin 0.036-0.071: docs/prereg/anchoring_none_margin.json (REAL own_minus_other 0.03556-0.07057).
- Section 3, future above past ceiling 58 / 86 / 58 %, max 0.1002: docs/prereg/far_targets_vs_future.json (share_targets_future_above_past; arxiv.future_max_cos_to_target.max 0.100156).
- Section 4 and A.2, F3 cluster-bootstrap intervals, raw vs partial ρ, F6 fallback: docs/prereg/revision_future_cells.json (scripts/revision_future_cells.py).
- Section 5.1, first-version dates and versions: docs/prereg/revision_first_public.json (scripts/revision_arxiv_versions.py, scripts/revision_first_public.py); one-shot narrowing 0.022 of 0.149: docs/prereg/revision_null_effects.json.
- Section 5.2, NEUTRAL arm (N1-N6, Table 2 row): docs/prereg/neutral_revision.md (Result), docs/prereg/revision_neutral.json (scripts/revision_neutral.py; generation scripts/iterative_critique.py with LG_ITER_ARMS=NEUTRAL, LG_ITER_R0_FROM=GEO, LG_ITER_OUT=iter_ideas_neutral.jsonl).
- Section 5.3, Table 3 weighted contributions and Holm-36, cap counts: docs/prereg/revision_direction_tangent.json (scripts/revision_direction_tangent.py); MDE, intervals and TOST bounds: docs/prereg/revision_null_effects.json (scripts/revision_null_effects.py).
- Section 5.4, intervals and bounds: docs/prereg/revision_null_effects.json.
- Section 6, date filter, appid check, estimands, CLI clustering, CIs and isolation adjustment: docs/prereg/revision_inspiration.json (scripts/revision_inspiration.py).
- A.6, repaired bootstrap and H1-H5 wording: docs/prereg/revision_bootstrap_combination.json (scripts/revision_bootstrap_combination.py).
- Appendix B commits: `git log --diff-filter=A -- docs/prereg/<file>.md`.
