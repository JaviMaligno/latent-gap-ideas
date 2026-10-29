# Pre-registration: fluency-constrained GPU search (powered h_coh lower bound)

**Frozen 2026-10-27, before any non-smoke run.** Code as committed with this file: `kaggle/fluent-search/search.py`
(kernel `javieraguilarmartin/latent-gap-fluent-search`, Kaggle GPU T4 x2), `scripts/fluent_prepare.py`,
`scripts/fluent_gate.py`. Inputs: private dataset `javieraguilarmartin/latent-gap-fluent-inputs`.

**Question.** How far can COHERENT text reach towards far embedding-space targets when the optimiser is strong and no
corpus neighbour is used? The blind LLM search ([coherent_search.md](coherent_search.md)) failed its REAL positive
control: a writer steered only by a cosine score cannot even recover a hidden real item, so its FAR/GAP misses say
nothing about coherence. The gradient token search ([bounds2.md](../bounds2.md) §1) reaches far (h_any FAR 0.161 Steam /
0.174 arXiv, GAP 0.669 / 0.553) but only with multilingual token soup. This arm puts a fluency penalty into that
gradient search and traces reach against fluency.

**What makes the result interpretable.** A lower bound is informative only if the same optimiser demonstrably climbs
where coherent text is known to exist. The REAL control (hidden real items, masked) must be recovered: judge-passing
h_coh ≥ corpus q99 on most REAL targets. Without that, a FAR miss is "underpowered", not "unreachable". An exhibit
(a judge-passing FAR witness far above the ceiling) needs no power check.

## Design

| Item | Fixed choice |
|---|---|
| Targets | `runs/<d>/pilot/targets.npz`, FAR k=1..20, GAP k=1..20, REAL k=1..10, d ∈ {steam, arxiv}: the coherent-search test set. k=0 is smoke only |
| REAL masking | as `scripts/coherent_search.py`: `REAL_hidden[k]` plus every visible item with cos > 0.95 removed from the corpus for ceiling, q99 and witness corpus-max; held-out items with cos > 0.95 dropped (code copied into `fluent_prepare.py`; values identical to coherent_search on all 100 targets) |
| References | ceiling = max cos over the visible corpus; ceiling* = max(ceiling, held-out ceiling); corpus q99 = 99th percentile of cos(t, visible corpus) |
| No neighbours | the kernel receives only the target vector. It never sees a corpus text (100 random corpus texts per domain, REAL items excluded, are used only to report the corpus's own LM NLL) |
| Encoder | Qwen3-Embedding-0.6B (fp16 on GPU, last-token pooling, explicit EOS, document without instruction), identical to the corpus embedding |
| LM | Qwen3-1.7B-Base (same tokenizer). Mean NLL in nats per text token of the rendered text, conditioned on a fixed content-free header (Steam: `A video game page on the Steam store:`; arXiv: one sentence saying that a title is followed by an abstract summarising problem, approach and contribution). Header 4-grams are banned from the text |
| Objective | J(s) = cos(enc(s), t) − λ · meanNLL_LM(s) |
| λ grid | {0, 0.05, 0.2, 0.5, 1.0, 2.0}. λ = 0: unconstrained reach in this format (full ASCII vocabulary for the gradient proposals); λ > 0: every gradient proposal is among the LM's top-100 next tokens |
| Format | exactly `DOMAINS[d].render`: Steam `Genres: <list>. Tags: <list>. About: <prose>`, arXiv `Title: <t>. Abstract: <prose>`. Spans start with a word token. List/title spans exclude `.:;()`; every token is printable ASCII and not whitespace-only |
| Phase 1 (generation) | spans written left to right. Candidates per step: LM top-32 next tokens ∪ encoder-gradient top-32 (gradient at an appended placeholder; for λ > 0 restricted to the LM's top-100 next tokens, AutoDAN-style). Each candidate scored exactly (encoder KV cache) and the best J appended. Span lengths (tokens): genres 1-10, tags 3-40, About 40-110, title 4-30, abstract 60-130, +15 overrun allowed. A list/title span may close only where closing beats every extension and the LM ranks the next scaffold token in its top 5; prose closes only at a sentence end. No repeated token 3-gram |
| Proposal constraints | list items non-empty, unique, ≤ 4 words, and each adds a word not used by earlier items of the list (no recombined keyword stuffing); ≤ 4 genres, ≤ 12 tags (inside the corpus p99 of 7 / 20); title ≤ 18 words; prose within the corpus p99 word bound |
| Phase 2 (refinement) | 35 s per target × λ: at 6 random editable whole-word positions, replacements from the gradient top-6 (λ > 0: within the LM top-100) and LM top-6, a deletion and 2 LM insertions; every proposal decoded, re-parsed, re-tokenised and scored exactly with both models; best J accepted if it improves. No new repeated 3-grams |
| Saved per target × λ | the generated text, every accepted state and the highest-cos proposal of each step (the pool) |
| Compute | two workers, worker i: encoder on GPU i, LM on GPU 1−i; Kaggle free quota only |
| Re-verification | every saved text re-embedded locally with the ONNX encoder (`LocalQwen`); only ONNX scores are reported |
| Validity | parses back exactly (render(parse(text)) == text) and within `latent_gap.coherent.BOUNDS` and 511 tokens |
| Coherence gate | per target and λ: distinct valid candidates without the gibberish flag (`latent_gap.coherent`), in descending ONNX score; judge 1 `claude:claude-haiku-5-5`, then judge 2 `codex:gpt-6-luna` if judge 1 accepts; accepted if both ≥ 7; walk stops at the first accepted, at most 15 candidates. Same blind prompt as coherent_search. Subscription CLIs only |
| Outcome per target | h_coh(λ) = best gate-passing ONNX score at λ; **h_coh = max over λ** (lower bounds combine by max); h_any(λ=0) = best valid ONNX score at λ = 0 |

**Judge pair calibration** (from `runs/coherent_calibration/`, same texts as coherent_search): the Haiku + Codex gate at
7 accepts 80% of Steam and 92% of arXiv corpus texts, 0% of the 100 constructed degenerate texts per domain and 0% of
the GPU token-soup witnesses. h_coh is therefore conservative (real texts fail 8-20% of the time).

## Decision rules (per domain)

| | Rule |
|---|---|
| C, REAL control | **powered** if h_coh ≥ corpus q99 on ≥ 80% (8 of 10) of REAL targets (unmeasured counts as a miss); otherwise **underpowered** |
| P1, FAR (the owner's objection: coherent ideas cannot reach far points) | **refuted in its strong form** if h_coh ≥ ceiling* + 0.05 on ≥ 20% of FAR targets (an exhibit; no power check needed). Otherwise, if coverage (FAR targets with any gate-passing candidate) < 90%: no verdict. Otherwise, if C underpowered: "uninformative". Otherwise **supported** if h_coh ≤ ceiling* + 0.03 on ≥ 80% of FAR targets; else "intermediate" |
| P2, GAP | **reachable** if h_coh ≥ ceiling − 0.02 on > 50% of GAP targets (coverage ≥ 90% required); else, if C powered, "not reached by a powered fluent search"; else "inconclusive". Also reported: share with h_coh ≥ ceiling* + 0.05 |

Reported without a rule: the frontier per λ (median ONNX h_any, median LM NLL of the states, median h_coh and its
coverage); the corpus NLL reference; h_coh of the blind Sonnet search and the anchored decoder landing on the same
targets; h_any of the unconstrained support search.

**Our forecast** (to be checked, not a rule). From the k=0 smoke: the search climbs (REAL k=0 at λ = 0.2 reaches
0.86-0.89, far above q99 0.45-0.55), but the judges accept only high-λ text, and only on arXiv. We expect **arXiv C
powered but not by much** (judge-passing REAL h_coh above q99: smoke 0.525 vs q99 0.448, at λ = 0.5) and **Steam: insufficient coverage** (no Steam
smoke text passed the gate at any λ; Haiku rates the cos-steered Steam pages 2-6 for stuffed or contradictory
genre/tag lists), hence Steam C underpowered and P1/P2 without verdict on Steam. arXiv P1: no strong refutation
(judge-passing FAR h_coh ≈ ceiling* − 0.06 in the smoke), "supported" only if C passes. P2 not reached on either
domain. If so, the honest summary is: the encoder can be pushed far with fluent-*looking* text, but text that two
judges accept as a real description stays at or below the corpus frontier, and on Steam even real-item recovery fails
the coherence bar, so the objection is neither refuted nor confirmed there.

## Smoke tests (k=0 only, excluded from the test set)

Nine Kaggle smoke runs on the k=0 targets (excluded from the test set), each followed by ONNX re-verification and
the gate (`scripts/fluent_gate.py`). Outputs in `runs/fluent-search/smoke*/`. Each run changed one thing because of a
failure the previous one showed; nothing was tuned on test targets. ONNX and fp16 GPU scores agree within 0.0002.

| Smoke | Change | What it showed |
|---|---|---|
| 1 | Qwen3-0.6B-Base LM, header `Steam store page:`, λ = 0.2 | climbs: REAL 0.858 Steam / 0.811 arXiv (ceilings 0.803 / 0.699, q99 0.546 / 0.448), Steam FAR 0.107 (ceiling 0.030). Gate 1/6: Haiku 1-6 for gradient sub-word junk (".DataBindings", "Linearencodedations"), duplicated tags, cut-off endings |
| 2 | gradient proposals within LM top-2000 for λ > 0; item/sentence-end rules; λ 0.5, 1.0 | 2/12 pass (arXiv GAP 0.378, 0.310). Genre span hit its cap after a comma; a title broke the 18-word bound; λ = 1.0 ≈ LM-only text |
| 3 | +15-token overrun, word bounds during generation, ≤ 4 words per item | titles still truncated ("...with Time Constraints on."); off-domain Steam prose |
| 4-5 | invented exemplar in the LM header; header 4-gram ban; LM memory fix | the exemplar became a topic attractor (every text paraphrased it; Steam FAR fell to −0.04): dropped |
| 6 | content-free header describing the fields; close only where the LM ranks the scaffold token top-5; whole-word refinement edits; LM top-100 | 0/12 pass: Haiku 2-6, stuffed/contradictory lists, generic or circular prose |
| 7 | Qwen3-1.7B-Base; ≤ 4 genres, ≤ 12 tags | arXiv 6/6 pass (REAL 0.525 at λ = 0.5, q99 0.448; GAP 0.385; FAR −0.013); Steam 0/6: the LM echoed the field descriptions of the header |
| 8 | minimal Steam header | Steam 0/6 (Haiku 1-6), REAL 0.838 / 0.816 at λ 0.2 / 0.5; tag lists of recombined genres |
| 9 | each list item must add a new word | Steam 0/6, ratings 2-6 (more 5-6); REAL 0.887 / 0.816 |

Not changed after smoke 9 except the whitespace-only token ban (a vocabulary mask) and the λ grid (2.0 added so that
the frontier reaches the most fluent end).

## Budget

Per target × λ: 50-100 s on T4 (generation 30-65 s, refinement 35 s). 100 targets × 6 λ = 600 runs on two workers
≈ 7-8 h of one T4 x2 session (watchdog: no new job after 10 h, kill at 10 h 40). Kaggle free quota: 26.0 h left this
week before the run (smokes used 1.35 h). Gate: ≤ 15 candidates per target × λ, Haiku then Codex, subscription CLIs
(free); estimated 2-4 h with 6 workers. No company spend and no gateway.

## Result (2026-10-27; `scripts/fluent_gate.py --run full`, output in runs/fluent-search/full/summary.md)

Full run: kernel version 10, 600 target × λ records (100 targets × 6 λ), 7.37 GPU-hours on T4 x2, no failure
(3 generations at λ ≤ 0.05 broke the format and skipped refinement; their generated text is scored). 3,187 distinct
candidates re-embedded with ONNX (max |ONNX − GPU| = 0.0006; 99.9% valid; gibberish flag 88-91% at λ = 0, 14-25% at
λ = 0.05, ≤ 17% above). Gate: 2,111 Haiku and 148 Codex ratings, no walk reached the 15-candidate cap, ~5 failed
ratings counted as not accepted.

Medians over targets; h_coh(λ) = best gate-passing ONNX score at λ with (measured / targets); h_coh = max over λ.

| domain | arm | n | ceiling | corpus q99 | anchored landing | blind h_coh (Sonnet) | h_any support-search | h_any λ=0 (this format) | h_coh λ=0.05 | λ=0.2 | λ=0.5 | λ=1 | λ=2 | h_coh (max over λ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| steam | FAR | 20 | 0.042 | -0.002 | 0.031 | 0.007 | 0.161 | 0.216 | - (0/20) | 0.028 (4/20) | 0.020 (1/20) | - (0/20) | - (0/20) | 0.020 (5/20) |
| steam | GAP | 20 | 0.533 | 0.457 | 0.523 | 0.435 | 0.669 | 0.710 | - (0/20) | - (0/20) | - (0/20) | 0.457 (2/20) | 0.342 (1/20) | 0.443 (3/20) |
| steam | REAL | 10 | 0.729 | 0.589 | 0.763 | 0.590 | - | 0.874 | - (0/10) | - (0/10) | 0.685 (1/10) | 0.605 (1/10) | - (0/10) | 0.645 (2/10) |
| arxiv | FAR | 20 | 0.055 | 0.010 | 0.071 | 0.008 | 0.174 | 0.229 | - (0/20) | 0.025 (4/20) | -0.013 (11/20) | -0.023 (15/20) | -0.018 (13/20) | -0.011 (18/20) |
| arxiv | GAP | 20 | 0.428 | 0.346 | 0.476 | 0.370 | 0.553 | 0.652 | - (0/20) | 0.407 (8/20) | 0.344 (14/20) | 0.305 (12/20) | 0.258 (13/20) | 0.392 (20/20) |
| arxiv | REAL | 10 | 0.739 | 0.527 | 0.819 | 0.447 | - | 0.864 | 0.817 (1/10) | 0.829 (3/10) | 0.686 (4/10) | 0.428 (7/10) | 0.351 (9/10) | 0.686 (10/10) |

No candidate at λ = 0 passed the gate (token soup, as in the support search).

Frontier (medians over targets of the best ONNX score per run, all candidates, and of the LM NLL of the run's states;
corpus NLL under the same LM and header: Steam 3.04 [p10 2.74, p90 3.51], arXiv 2.49 [1.94, 2.95] nats/token):

| λ | Steam FAR h_any / NLL | Steam GAP | Steam REAL | arXiv FAR | arXiv GAP | arXiv REAL |
|---|---|---|---|---|---|---|
| 0 | 0.216 / 7.58 | 0.710 / 7.42 | 0.874 / 7.61 | 0.229 / 7.28 | 0.652 / 8.03 | 0.864 / 6.83 |
| 0.05 | 0.158 / 3.07 | 0.658 / 3.41 | 0.833 / 3.21 | 0.143 / 2.97 | 0.621 / 3.11 | 0.869 / 3.15 |
| 0.2 | 0.046 / 1.76 | 0.565 / 2.14 | 0.759 / 2.24 | 0.043 / 1.46 | 0.441 / 1.72 | 0.799 / 1.79 |
| 0.5 | -0.007 / 1.44 | 0.502 / 1.97 | 0.690 / 2.03 | -0.006 / 1.20 | 0.370 / 1.30 | 0.530 / 1.46 |
| 1.0 | -0.033 / 1.49 | 0.470 / 1.65 | 0.566 / 1.87 | -0.023 / 1.19 | 0.311 / 1.42 | 0.459 / 1.38 |
| 2.0 | -0.027 / 1.46 | 0.411 / 1.47 | 0.477 / 1.47 | -0.024 / 1.26 | 0.266 / 1.25 | 0.399 / 1.34 |

Shares per arm (h_coh = max over λ; unmeasured counts as a miss):

| domain | arm | coverage | ≥ corpus q99 | > ceiling | within ceiling* + 0.03 | ≥ ceiling* + 0.05 | ≥ ceiling − 0.02 | max (h_coh − ceiling*) |
|---|---|---|---|---|---|---|---|---|
| steam | FAR | 0.25 | 0.25 | 0.05 | 0.25 | 0.00 | 0.15 | -0.012 |
| steam | GAP | 0.15 | 0.05 | 0.00 | 0.15 | 0.00 | 0.00 | -0.052 |
| steam | REAL | 0.20 | 0.20 | 0.00 | 0.20 | 0.00 | 0.00 | -0.040 |
| arxiv | FAR | 0.90 | 0.25 | 0.00 | 0.90 | 0.00 | 0.20 | -0.015 |
| arxiv | GAP | 1.00 | 0.65 | 0.20 | 0.90 | 0.05 | 0.30 | +0.092 |
| arxiv | REAL | 1.00 | 0.70 | 0.30 | 0.70 | 0.30 | 0.30 | +0.154 |

**Verdicts as pre-registered:**

- **C (REAL control): UNDERPOWERED on both domains.** arXiv: judge-passing h_coh ≥ corpus q99 on 7 of 10 REAL
  targets (the rule needs 8); Steam: 2 of 10 (only 2 REAL targets have any gate-passing text).
- **P1 FAR. arXiv: "uninformative: REAL control failed"** (coverage 90%; 18 of 20 FAR targets within ceiling* + 0.03,
  i.e. 90%, and 0 of 20 at ceiling* + 0.05; best judge-passing FAR text 0.015 *below* ceiling*). **Steam: no verdict,
  insufficient coverage (25%).** The strong-form refutation (≥ 20% of FAR targets at ceiling* + 0.05) fails on both
  domains: no judge-passing FAR witness above the corpus ceiling + 0.05 exists in 600 runs.
- **P2 GAP. arXiv: "inconclusive"** (reached ceiling − 0.02 on 30%, rule > 50%). **Steam: insufficient coverage
  (15%).** One arXiv GAP target (k=11) has a judge-passing text at 0.521 vs ceiling* 0.429 (+0.092; Haiku 8, Codex 8).

**Against the forecast:** C on arXiv came one target short of "powered" (forecast: "powered but not by much"); Steam
was, as forecast, without coverage; P1 arXiv had no strong refutation (forecast), and the FAR numbers sit where a
"supported" verdict would have been (90% within +0.03); P2 not reached (forecast), GAP h_coh medians 0.44 Steam
(forecast 0.40-0.45) and 0.39 arXiv (forecast 0.30-0.35).

**What the run shows.**

1. **The optimiser is strong in reach.** Without a fluency term it beats the support search in the same format
   (h_any FAR 0.22-0.23 vs 0.16-0.17, GAP 0.65-0.71 vs 0.55-0.67), and with fluent-looking text (λ = 0.2, NLL 1.5-2.2,
   below the corpus's own 2.5-3.0) it recovers hidden REAL items well beyond their masked ceiling (median 0.76-0.80).
   So coherence-looking text *under the LM* is not the limit for REAL targets.
2. **The judges are the limit.** Texts that two blind judges accept are fewer and lower: arXiv REAL judge-passing
   median 0.686 (vs ceiling 0.739); on Steam the judges accept almost nothing the search produces (Haiku rates
   cos-steered Steam pages 2-6 for stuffed or contradictory genre/tag lists), so Steam is not measured.
3. **Where coherent text is measured, it does go beyond the corpus near the data, not far from it.** arXiv: 3 of 10
   REAL targets and 4 of 20 GAP targets have a judge-passing text above the *visible* corpus ceiling (best REAL
   +0.154 over ceiling*: a coherent MIMO channel-prediction survey at 0.874 vs 0.720), but no FAR target does (best
   −0.015 vs ceiling*). FAR reach collapses as soon as λ > 0.05 (h_any 0.14 → 0.04 at λ = 0.2), i.e. the FAR cosines
   of the unconstrained search are bought entirely with high-NLL tokens.
4. **Read together** (not a pre-registered rule): on arXiv the FAR result is what the objection predicts (90% within
   +0.03, nothing at +0.05, best coherent FAR text below the ceiling) from an optimiser that recovered 7 of 10 hidden
   real items to the top 1% and beat the masked ceiling on 3; the pre-registered bar for calling it powered (8 of 10)
   was missed by one target, so the verdict stays "uninformative". On Steam nothing can be said: a 1.7B base LM
   steered by the encoder does not write Steam pages the judges accept.

Kaggle GPU: 8.72 h in total for this arm (9 smoke runs 1.35 h, full run 7.37 h; quota 11.36 / 30 h this week). No
company spend, no gateway; judges via subscription CLIs.

## Revision 2: Steam with Qwen3-4B-Base (new arm, frozen 2026-10-28 before its non-smoke run)

**Why.** In rev 1, Steam was unmeasured (coverage 25% / 15% / 20% on FAR / GAP / REAL): the judges rejected the
pages a 1.7B base LM writes under encoder steering. The owner approved one further arm with a larger open base LM to
make Steam measurable. arXiv is not rerun. Results of rev 1 stand as reported above and are not combined with rev 2.

**Arm.** Identical to rev 1 (same targets, references, masking, encoder, objective, λ grid {0, 0.05, 0.2, 0.5, 1, 2},
budget per target × λ, re-verification, gate: Haiku then Codex, both ≥ 7, cap 15) except:

| Item | Rev 2 |
|---|---|
| Targets | Steam only: FAR k=1..20, GAP k=1..20, REAL k=1..10 (50 targets, 300 target × λ runs) |
| LM | Qwen3-4B-Base, fp16; worker i: encoder on GPU i, LM on GPU 1−i (as before) |
| List items (Steam genres / tags) | in addition to rev 1: an item is a plain label (letters, digits, apostrophes, `+`, single spaces, in-word hyphens), no `and` / `or`, ≤ 30 characters, and shares at most one word with earlier items of the list |
| Closing a list span | the J comparison and the item rules only (the rev-1 "scaffold token in the LM top 5" rule now applies to arXiv titles only) |
| Run name | `steam4b` (kernel version after this commit); gate output `runs/fluent-search/steam4b/` |

The list rules come from the four rev-2 smokes (k=0 only): the 4B LM chained items with `and`, `&`, `/`, `|` and
` - `, glued them into spaceless runs, or kept a genre list open because it rarely ranks the `.` of `. Tags:` in its
top 5. They restrict proposals only; validity (render round trip, corpus-p99 bounds) is unchanged.

**Decision rules.** Exactly the rev-1 rules for Steam, not relaxed: C powered iff judge-passing h_coh ≥ corpus q99 on
≥ 8 of 10 REAL targets; P1 FAR refuted (strong form) iff h_coh ≥ ceiling* + 0.05 on ≥ 20%; else no verdict below 90%
coverage; else "uninformative" if C underpowered; else supported iff ≥ 80% within ceiling* + 0.03, otherwise
intermediate. P2 GAP reachable iff h_coh ≥ ceiling − 0.02 on > 50% (coverage ≥ 90%); else "not reached" if C powered,
"inconclusive" if not.

**Smoke tests (k=0, λ 0.2 and 0.5; ONNX-verified and gated).**

| Smoke | Change | Gate | Notes |
|---|---|---|---|
| 4B-1 | Qwen3-4B-Base, rev-1 rules | 1/6 (FAR λ 0.5) | REAL 0.790 / 0.743; lists chained with `and`, `&`, `/`, ` - ` |
| 4B-2 | item shape rule (plain labels, no and/or, ≤ 30 chars) | 0/6 | genre lists could not close: last item grew into "Strategy Description A spacef" |
| 4B-3 | close_top only for titles | 2/6 (FAR 0.2, 0.5) | REAL 0.849 / 0.835 rejected for recombined tags ("Hunting Game, Hunting Simulator Game, ...") |
| 4B-4 | ≤ 1 shared word per item | **4/6**: FAR −0.015, −0.043; REAL λ 0.2 0.837; GAP λ 0.5 0.457 | REAL k=0: q99 0.546, masked ceiling 0.803 |

**Predictions** (to be checked; not rules). C: borderline: REAL k=0 passes above its masked ceiling, but in rev 1 REAL
pages were the hardest to get accepted (stuffed tags); we expect 5-8 of 10 REAL targets ≥ q99, i.e. powered with
probability ≈ 0.4. P1 FAR: coverage ≥ 90% (smoke FAR texts pass at the first candidate); no strong refutation
(smoke FAR h_coh ≤ ceiling − 0.045); if C is powered, "supported". P2 GAP: not reached (median h_coh ≈ 0.45-0.48
vs ceiling 0.533).

**Budget.** 4B smoke: 496 s for 6 runs on two workers (≈ 85 s per target × λ). Full: 300 runs / 2 workers × 85 s ≈
3.6 h (watchdog: no new job after 5.5 h, kill at 6 h). Quota before the full run: see the commit; the rev-2 smokes
used ≈ 0.6 h, and the arm stays far under the ~14 h cap, so no λ level is cut.

## Result, revision 2 (Steam, Qwen3-4B-Base; 2026-10-28; runs/fluent-search/steam4b/summary.md)

Kernel version 15: 300 target × λ records, 3.65 GPU-hours on T4 x2, no failure and no watchdog stop. 1,309 distinct
candidates re-embedded with ONNX (max |ONNX − GPU| = 0.0007; valid 100%; gibberish flag 16%, almost all at λ ≤ 0.05).
Gate: 773 candidates judged, no walk at the cap.

| arm | n | ceiling | corpus q99 | h_any λ=0 | h_coh λ=0.2 | λ=0.5 | λ=1 | λ=2 | h_coh (max over λ) | coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| FAR | 20 | 0.042 | -0.002 | 0.202 | 0.025 (10/20) | -0.014 (16/20) | -0.027 (20/20) | -0.027 (11/20) | 0.000 | 20/20 |
| GAP | 20 | 0.533 | 0.457 | 0.703 | 0.536 (5/20) | 0.499 (4/20) | 0.450 (8/20) | 0.420 (15/20) | 0.449 | 19/20 |
| REAL | 10 | 0.729 | 0.589 | 0.885 | 0.834 (2/10) | 0.687 (4/10) | 0.560 (4/10) | 0.521 (7/10) | 0.615 | 8/10 |

(λ = 0 and 0.05: no gate-passing candidate. Medians over targets; h_coh(λ) medians over measured targets.)

| arm | ≥ corpus q99 | > ceiling | within ceiling* + 0.03 | ≥ ceiling* + 0.05 | ≥ ceiling − 0.02 | max (h_coh − ceiling*) |
|---|---|---|---|---|---|---|
| FAR | 0.55 | 0.05 | 1.00 | 0.00 | 0.35 | +0.012 |
| GAP | 0.40 | 0.15 | 0.90 | 0.05 | 0.25 | +0.063 |
| REAL | 0.40 | 0.20 | 0.60 | 0.20 | 0.30 | +0.133 |

REAL per target (h_coh vs q99): k=1 0.701 / 0.616, k=2 0.810 / 0.595, k=7 0.693 / 0.588, k=8 0.857 / 0.577 reach q99
(k=2 and k=8 above the masked ceiling); k=5, 6, 9, 10 pass the gate only at 0.50-0.54 (below q99); k=3, 4 have no
gate-passing text although their λ = 0 reach is 0.90-0.92.

**Verdicts as pre-registered (rules not relaxed):**

- **C (REAL control): UNDERPOWERED**: 4 of 10 REAL targets ≥ corpus q99 (needs 8).
- **P1 FAR: "uninformative: REAL control failed"**. Coverage is now 100%; all 20 FAR targets stay within
  ceiling* + 0.03, none reaches ceiling* + 0.05 (best +0.012). No strong-form refutation.
- **P2 GAP: "inconclusive"** (reached ceiling − 0.02 on 25%; one target, k=18, has a judge-passing text at 0.604 vs
  ceiling* 0.542, +0.063).

**Against the forecast:** coverage ≥ 90% on FAR as forecast (100%); C fell below the forecast range (4 vs 5-8 of 10);
no strong refutation (as forecast); GAP not reached (median h_coh 0.449 vs forecast 0.45-0.48).

**What revision 2 adds.** The larger LM makes Steam measurable: coverage went from 25 / 15 / 20% (1.7B) to
100 / 95 / 80%. The picture is now the same on both domains:

- coherent FAR text stays at the corpus ceiling (Steam: 20/20 within +0.03; arXiv rev 1: 18/20);
- coherent text does go beyond the visible corpus near the data (Steam REAL k=8: 0.857 vs masked ceiling 0.724;
  GAP k=18 +0.063);
- but the search cannot yet recover enough hidden real items as text that both judges accept (Steam 4/10, arXiv 7/10
  at q99), so the FAR result stays formally uninformative.

The gap between h_any at λ = 0.2 (REAL median 0.757) and the judge-passing median (0.615) is the coherence bar as
the judges apply it, mostly to tag lists, not a lack of reach.

**GPU.** Revision 2: 4.22 h (four smokes 0.57 h, full run 3.65 h). Both arms together: 12.94 h (rev 1 8.72 h).
Quota after the run: 15.58 h used, 14.42 h left this week (resets 2026-10-31).

## Revision 3: seed-and-rewrite (frozen 2026-10-28, after tuning on separate targets, before the test run)

**Why.** The two earlier searches failed the control for opposite reasons. The blind LLM loop
([coherent_search.md](coherent_search.md)) writes coherent pages but does not get near a hidden real item. The GPU
search gets near it (λ = 0 REAL reach 0.86-0.89) but its text fails the judges. The gate itself accepts 80-92% of real
corpus pages. This arm combines the two: an LLM rewrites the GPU search's best texts into coherent pages and moves them
towards the target with the cosine as its only feedback.

**Method (frozen; `scripts/seed_rewrite.py`, prompt version v2).**

| Item | Fixed choice |
|---|---|
| Seeds | per target, the S = 2 distinct texts with the highest ONNX score among all saved states (generated + pool) of the GPU search at λ ∈ {0, 0.05}: test targets from `runs/fluent-search/full` (arXiv, 1.7B LM) and `runs/fluent-search/steam4b` (Steam, 4B LM). The GPU search never saw a corpus text |
| Rewriters | `claude:claude-sonnet-5-5` (Claude Code CLI) and `codex:gpt-6-luna` (Codex CLI, stdin closed); subscription only, no gateway. Each is a separate run |
| Rewriter input | only the seed, its own best rewrite so far and that rewrite's cosine (a number). No corpus text, neighbour or direction |
| Chain per seed | round 0: rewrite the draft into a real, sensible page keeping its specific content; rounds 1-3 (R = 3): rewrite again to score higher, bringing back dropped content, every named term fitting the rest; then one polish round on the chain's best text (light coherence edit, the seed not shown). Body under 170 words so that it renders untruncated. 10 calls per target and rewriter |
| Scoring | local ONNX encoder; validity (render round trip, corpus-p99 bounds), gibberish flag |
| Gate | unchanged: valid, non-gibberish candidates in descending score; claude-haiku-5-5, then codex:gpt-6-luna; both ≥ 7; cap 15 |
| References, masking | unchanged (`runs/fluent-search/refs.*`) |
| Test set | FAR k=1..20, GAP k=1..20, REAL k=1..10 per domain, run once per rewriter |

**Tuning (REAL k=11..20 per domain, built and masked like the test REAL targets; never test targets).** Seeds came
from a λ ∈ {0, 0.05} GPU run on those targets (kernel versions 16-17: Steam with the 4B LM, arXiv with the 1.7B LM,
0.29 + 0.31 GPU-h). Judge-passing h_coh ≥ q99:

| Version | Steam Sonnet | Steam Codex | arXiv Sonnet | arXiv Codex |
|---|---|---|---|---|
| v1 (S=2, R=3, no length cap, no polish) | 10/10 | 10/10 | 8/10 | 7/10 |
| **v2 (frozen)**: body < 170 words, "every named term fits", polish round | **10/10** | **10/10** | **9/10** | **10/10** |

In v1 the arXiv failures were rewrites at 0.88-0.94 that the judges rated 4-6 for carried-over stuffed jargon or a
body cut by the 1200-character render limit. Under v2, tuning REAL h_coh medians are 0.816 / 0.821 (Steam, ceiling
0.741) and 0.880 / 0.875 (arXiv, ceiling 0.723); 38 of 40 runs beat the masked ceiling.

**Decision rules.** As rev 1, per domain, applied to the **max over the two rewriters** (lower bounds combine by max);
per-rewriter verdicts are reported as well. C powered iff h_coh ≥ q99 on ≥ 8 of 10 REAL test targets (not relaxed).
P1 FAR: refuted (strong form) iff h_coh ≥ ceiling* + 0.05 on ≥ 20%; else no verdict below 90% coverage; else
uninformative if C fails; else supported iff ≥ 80% within ceiling* + 0.03; else intermediate. P2 GAP: reachable iff
h_coh ≥ ceiling − 0.02 on > 50% (coverage ≥ 90%); else "not reached" if C powered, "inconclusive" if not.
Self-preference: Codex is both a rewriter and judge 2. Its rewrites are also judged first by Haiku. Per-rewriter
results are reported so the Sonnet-only verdict can be read on its own.

**Predictions** (not rules). C powered on both domains (tuning 9-10/10). FAR: the λ = 0 FAR seeds score 0.20-0.23
only through token soup with no semantic content, so rewriting should lose most of it. Expected: coverage ≥ 90%,
judge-passing FAR h_coh close to the ceiling, i.e. **P1 supported** on both domains, with no or few targets at
ceiling* + 0.05. P2 GAP: **reachable** on both domains (the GAP seeds are 0.65-0.71 and the REAL rewrites kept about
95% of their seed's score).

**Budget.** 100 test targets × 2 rewriters × 10 calls ≈ 2,000 rewriter calls plus about 300-600 judge calls; tuning
took 7-10 min per 10 targets with 4 workers, so about 1-2 h wall with the four runs in parallel. No GPU.

### Rev 3 result (2026-10-28; runs/seed-rewrite/test, `scripts/seed_rewrite.py --label test --summary`)

All 200 target-runs completed (Steam then arXiv, 2 processes at a time to relieve memory), no failures. Medians
over targets; h_coh = best judge-passing rewrite; ≥ q99 counts targets.

| Domain | Rewriter | Arm | ceiling | q99 | seed best | h_coh | coverage | ≥ q99 |
|---|---|---|---|---|---|---|---|---|
| Steam | max | REAL | 0.729 | 0.589 | 0.885 | 0.842 | 10/10 | 10/10 |
| Steam | max | FAR | 0.042 | −0.002 | 0.202 | 0.069 | 20/20 | 20/20 |
| Steam | max | GAP | 0.533 | 0.457 | 0.711 | 0.567 | 20/20 | 19/20 |
| arXiv | max | REAL | 0.739 | 0.527 | 0.884 | 0.874 | 10/10 | 10/10 |
| arXiv | max | FAR | 0.055 | 0.010 | 0.229 | 0.057 | 20/20 | 18/20 |
| arXiv | max | GAP | 0.428 | 0.346 | 0.656 | 0.472 | 19/20 | 19/20 |

Pre-registered verdicts (max over rewriters):
- **C powered on both domains** (10/10 REAL ≥ q99; 10/10 above the masked ceiling). Per rewriter: 9-10/10.
- **P1 FAR, arXiv: objection supported** (90% within ceiling* + 0.03, no escape; max h_coh − ceiling* +0.038), and
  supported for each rewriter.
- **P1 FAR, Steam: objection refuted in its strong form** (5/20 = 25% of FAR targets at ≥ ceiling* + 0.05, max
  +0.095). Per rewriter: Codex/luna refuted (25%), Sonnet intermediate (1/20 escape, 75% within).
- **P2 GAP reachable on both domains** (≥ ceiling − 0.02 on 85% Steam, 75% arXiv; ≥ ceiling* + 0.05 on 25% / 35%).

Against the predictions: C and GAP as forecast; arXiv FAR as forecast; Steam FAR not (forecast "supported").

**Sensitivity (not pre-registered): self-preference.** The Steam escape rests on luna rewrites that luna itself passed
as judge 2. Four of the five carry invented tokens from the GPU seeds ("qucheessa", "P-ras", "The Sword Rend
BoydlationLEYON", "Space Mah", "Aardaphila") and passed the gate at 7. All 148 gate-passing FAR/GAP texts were
re-judged by claude-opus-5-5, which neither wrote nor gated any of them (`scripts/seed_rewrite_opus_check.py`,
runs/seed-rewrite/test/opus_check.jsonl, 0 errors). Opus accepts (≥ 7) 89-90% of the Sonnet-written FAR texts but 68-70%
of the luna-written ones. Requiring Opus ≥ 7 as well, the Steam escapes left are 3/20 = 15% (luna k=13 and k=19,
Sonnet k=12; Opus 8, 7, 9), below the 20% threshold, and the verdict would be "intermediate" (60% within ceiling* +
0.03); arXiv is unchanged (supported). The pre-registered verdict stands as written; it is fragile to the judge.

**Absolute scale.** Even the escapes stay far from the target: Steam FAR h_coh 0.09-0.14 against a perfect 1 and a
corpus ceiling of 0.04-0.05. Coherent text can be written slightly closer to a far point than any existing game, on a
minority of targets, but no coherent text gets near the point; the decoded "idea" is a normal game.
