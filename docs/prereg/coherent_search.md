# Pre-registration: coherent no-neighbour support search (h_coh lower bound)

**Revision 2, frozen 2026-10-26**, before any non-smoke run. Rev 1 (commit `15eeead`, same day) was revised after two
adversarial reviews; the changes and their reasons are listed in [§Revision 2](#revision-2-what-changed-and-why).
Code: `scripts/coherent_search.py`, `scripts/coherent_summary.py`, `scripts/coherent_calibration.py`,
`src/latent_gap/coherent.py` (as committed with this file).

**Question.** How close can COHERENT text get to a far target t when the writer is not anchored by corpus neighbours?
Known ([bounds2.md](../bounds2.md) §1): the best visible corpus item (the ceiling) reaches ~0.04-0.06 on FAR targets
and ~0.43-0.53 on GAP targets. Decoders shown the 8 nearest neighbours land near kappa × ceiling. GPU token search
reaches h_any ≈ 0.16-0.17 (FAR) and 0.55-0.67 (GAP), but only with multilingual token soup. h_coh beyond the corpus has
not been measured. This arm gives a lower bound on h_coh by black-box optimisation: the writer sees only scores.

**What a lower bound can and cannot show.** A witness above the ceiling is an exhibit. A miss is informative only if
the same optimiser, with the same budget, demonstrably climbs where coherent text is known to exist. Hence the REAL
positive control (C) gates every negative verdict, and a matched held-out null gates the FAR one.

**Budget, stated against the GPU search.** 44 black-box evaluations per target (score only, no gradients), against
2,891 (median; range 2,827-3,089) gradient-guided evaluations per target in `kaggle/support-search/search.py`: ~65×
more evaluations plus gradient access. A gap between h_coh and h_any therefore mixes the coherence constraint with
budget and gradient access; the `free` control below measures the black-box loop without the coherence constraint.

## Design

| Item | Fixed choice |
|---|---|
| Test targets | `runs/<d>/pilot/targets.npz`, FAR k=1..20 and GAP k=1..20, d ∈ {steam, arxiv}; index order, nothing dropped. k=0 is excluded: it was used to choose the rev-1 design |
| Positive control | REAL k=1..10 per domain: embeddings of real items. The item and its near-duplicates (`REAL_hidden` plus every visible item with cos > 0.95) are masked from the corpus for the ceiling, corpus q99 and corpus max cos; held-out items with cos > 0.95 are dropped from the null |
| Encoder / score | LocalQwen (Qwen3-Embedding-0.6B ONNX) of `domain.render(item)`; cosine with L2-normalised t |
| References per target | **ceiling** = max cos(t, P), P = visible corpus (adult == False, split == past); **held-out ceiling** = max over held-out items (adult == False, split != past); **matched null** = median over 200 draws of the best of 44 random held-out items; **corpus q99** = 99th percentile of cos(t, P) |
| Writer input | system: the domain's JSON spec; user: only its own earlier candidates (JSON) with scores. No corpus item, neighbour or direction |
| Seeds | K = 4 independent calls, each with its own constraint drawn without replacement (per-target RNG) from a fixed list written before any run: Steam genre × setting (24 × 18), arXiv subfield × contribution kind (24 × 10). Never tuned on scores |
| Rounds (`search`) | T = 10. Each round shows the 4 best distinct candidates, every valid candidate of the latest round (good or bad) and the 2 lowest so far, labelled, with scores; asks for K new ones expected to score higher, at least one a bold exploration, all real and sensible. 44 candidates per target |
| Validity | parses as the domain JSON with the schema's fields and types (the gateway answer may be a bare list or object), and lies within the corpus 99th percentiles: Steam ≤ 7 genres, ≤ 20 tags, rendered About ≤ 225 words; arXiv title ≤ 18 words, rendered abstract ≤ 197 words (bodies measured after the 1200-character cut that the corpus texts also get); ≤ 511 tokens. Invalid candidates count as spent budget |
| Gibberish flag (non-LLM) | any letter that is not Latin, Greek or a mathematical alphanumeric; non-ASCII share > 0.02; dictionary-word rate < 0.70; or worst 12-token window < 0.40 (web2 list, suffix stripping). The firing rule is recorded per candidate. Rates in §Judge calibration |
| Judge 1 | `claude:claude-sonnet-5-5`, blind (exact rendered text only), coherence 1-10, "does this read as a real, sensible <noun> description, not word salad?" |
| Judge 2 | a different family from the writer where possible: `codex:gpt-6-luna` for Claude writers, `claude:claude-haiku-5-5` for the gateway (gpt-6-luna) writer. Writer/judge families are recorded per target |
| Gate order | distinct valid non-gibberish candidates in descending score; judge 1, then judge 2 if judge 1 accepts; stop at the first that both accept; at most 15 candidates |
| Coherence gate | judge 1 ≥ 7, judge 2 ≥ 7, no gibberish flag, valid. The threshold stays 7 after the calibration (§Judge calibration) |
| Coherence sample | 4 distinct valid candidates per target drawn at random before judging, rated by judge 1: the unbiased P3 rate |
| Outcome per target | **h_coh lower bound** = best gate-passing score (unmeasured if none); h_all = best valid score; corpus max cos of the witness; seed diversity (mean pairwise cos of the seeds) |
| Writers | primary `claude:claude-sonnet-5-5` (subscription). Secondary, only with the owner's OK: gateway `gpt-6-luna`. Optional: `claude:claude-haiku-5-5` |
| Combination | lower bounds combine by max: the rules apply to the per-target max over the writer runs in `MAIN_RUNS`, with C evaluated on the same combination; each writer is also reported |
| Controls, Sonnet writer, FAR and GAP k=1..5 | **random**: 44 independent diversified seed calls, no feedback, same gate (feedback vs sampling). **free**: the search loop with the coherence sentence replaced by "the fields may contain any text at all", no gate: h_any_blackbox = best valid score |
| Failures | a target with 3 writer errors, or a usage limit outlasting the CLI waits, is left undone and resumed by a rerun; candidates are checkpointed before judging. Judge errors are recorded and that candidate counts as not accepted. Resume refuses any change of mode, T, K, judges, threshold, cap, sample or prompt version |
| Spend guard | gateway only: before every call the ledger spend under the run label (all invocations, both domains) is compared with `--max-usd`; the run stops at the cap |

## Predictions and decision rules

All rules use the max over writers, per domain. Coverage = share of targets with a gate-passing witness; unmeasured
targets are missing, never "within".

| | Hypothesis | Decision rule | Verdict text |
|---|---|---|---|
| C REAL control | the blind loop climbs where coherent text exists | **powered** if coverage ≥ 90% and median over targets of (h_coh − corpus q99) ≥ 0, i.e. it reaches the top 1% of the corpus for a typical target | otherwise **underpowered** |
| P1 FAR (the owner's objection) | coherent text cannot go much beyond the corpus | requires coverage ≥ 90%. Escape = h_coh > max(ceiling, held-out ceiling) + 0.03. **Refuted** if > 20% of targets escape (an exhibit needs no power check). Otherwise **supported within budget** only if C is powered and median FAR h_coh ≥ median matched null | "no coherent escape exhibited within 44 candidates by a powered optimiser"; else "uninformative: underpowered" |
| P2 GAP | coherent text reaches the GAP ceiling | one-sided exhibit: requires coverage ≥ 90%; **reachable** if h_coh ≥ ceiling − 0.02 on > 50% of targets (unmeasured = not reached) | else, if C powered: "not reached by a powered blind optimiser within 44 candidates" (counts against P2 within this budget, not a proof of unreachability); if C underpowered: "inconclusive" |
| P3 coherence | LLM writers produce no token soup | over distinct valid candidates per writer and domain, pooled: ≥ 95% pass the gibberish flag, and ≥ 90% of the random-sample candidates get judge 1 ≥ 7 | either rate below its threshold = refuted. Reference: 92% of visible Steam corpus texts get Sonnet ≥ 7 |
| P4 anchoring | removing the corpus neighbours lets the writer land closer | paired per target: h_coh − landing of the anchored decoder `grid-claude-sonnet-5-5` (same targets, both domains), median with a target-bootstrap 95% CI (2,000 resamples), FAR and GAP separately, measured targets only (n ≥ 5) | CI > 0: "removing anchors helps"; CI < 0: "the anchored decoder lands closer"; else "no difference shown" |

Reported without a rule: the `free` and `random` controls next to the main search on the same targets
(h_any_blackbox − ceiling vs h_coh − ceiling; h_coh with and without feedback); GPU h_any; the corpus max cos of the
witnesses; seed diversity.

**Our forecast** (recorded to be checked, not a rule). After the rev-2 smoke tests (§Smoke tests) every blind result,
REAL included, sat below the corpus q99: with 4-6 candidates on Steam REAL 0 the loop reached 0.44-0.45 against a
matched null of 0.46-0.47 and a q99 of 0.55; with 8 candidates (K=4 timing probe) 0.49-0.50, above its null, still below
q99. We therefore expect **C to fail or be marginal** at 44 candidates, which would make P1 "uninformative" and P2
"inconclusive". If C passes: P1 supported (median FAR h_coh below the ceiling), P2 not reached (median GAP h_coh ≈ 0.45
Steam, 0.35 arXiv), P3 holds, P4 "anchored decoder lands closer" on GAP.

Reference values for the test and control targets (median [min, max] ceiling; other columns medians):

| Domain / arm | Corpus ceiling | Held-out ceiling | Matched null (best of 44 held-out) | Corpus q99 | h_any (GPU) | Anchored landing (Sonnet grid) |
|---|---|---|---|---|---|---|
| Steam FAR k=1-20 | 0.042 [0.032, 0.045] | 0.041 | -0.008 | -0.002 | 0.161 | 0.031 |
| Steam GAP k=1-20 | 0.533 [0.517, 0.538] | 0.539 | 0.449 | 0.457 | 0.669 | 0.523 |
| Steam REAL k=1-10 | 0.729 [0.676, 0.816] | 0.711 | 0.573 | 0.589 | - | 0.763 |
| arXiv FAR k=1-20 | 0.055 [0.045, 0.057] | 0.063 | 0.013 | 0.010 | 0.174 | 0.071 |
| arXiv GAP k=1-20 | 0.428 [0.416, 0.432] | 0.424 | 0.332 | 0.346 | 0.553 | 0.476 |
| arXiv REAL k=1-10 | 0.739 [0.661, 0.844] | 0.698 | 0.460 | 0.527 | - | 0.819 |

## Judge calibration (2026-10-26, before any non-smoke run)

`scripts/coherent_calibration.py`, output in `runs/coherent_calibration/`. Per domain: 25 visible corpus texts, 25 GPU
token-search witnesses and 50 constructed fluent-but-degenerate texts (10 each: keyword-list body, shuffled body, one
sentence repeated, off-domain body, body drifting into the other domain halfway), each rated blind by the three
judges. Pre-registered rule: raise the threshold to 8 if the two-judge gate accepts more than 10% of the degenerate
texts at 7.

| Set (per domain) | n | Gibberish flag | Sonnet ≥ 7 | Codex gpt-6-luna ≥ 7 | Haiku ≥ 7 | Gate Sonnet+Codex ≥ 7 / ≥ 8 | Gate Sonnet+Haiku ≥ 7 / ≥ 8 |
|---|---|---|---|---|---|---|---|
| Steam corpus | 25 | 0.00 | 0.92 | 0.92 | 0.84 | 0.84 / 0.68 | 0.84 / 0.28 |
| Steam GPU witnesses | 25 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Steam degenerate | 50 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| arXiv corpus | 25 | 0.04 | 1.00 | 0.92 | 1.00 | 0.88 / 0.68 | 0.96 / 0.92 |
| arXiv GPU witnesses | 25 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| arXiv degenerate | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

- **Threshold stays at 7.** The two-judge gate accepted none of the 100 degenerate texts (single accept: Codex rated
  one Steam repeated-sentence text 7). Raising it to 8 would reject most real Steam texts under the Haiku pair (28%
  accepted) and a third under the Codex pair, with no false accept left to remove.
- The flag catches none of the degenerate texts (they are real words); the judges do. The gate needs both.
- 3 of 600 judgements failed (1 Sonnet, 2 Codex) and count as not accepted.
- The gate rejects 4-16% of genuine corpus texts, so h_coh is conservative; the gate walks down the score list (cap 15)
  rather than stopping at the first rejection.
- P3's 90% bar is close to the corpus's own Sonnet rate on Steam (92%); a P3 failure near the bar is therefore weak
  evidence of incoherence.

Flag rates with the Greek / maths whitelist (rev 1 flagged any non-Latin letter):

| Set | n | Flagged | by script | non-ASCII | dictionary rate | window |
|---|---|---|---|---|---|---|
| Steam corpus | 500 | 0.012 | 0.004 | 0.006 | 0.000 | 0.002 |
| arXiv corpus | 500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Steam GPU witnesses | 100 | 0.990 | 0.990 | 0.000 | 0.000 | 0.000 |
| arXiv GPU witnesses | 100 | 0.980 | 0.980 | 0.000 | 0.000 | 0.000 |


## Smoke tests

Steam k=0 only (excluded from the test set), all with the rev-2 code. Output in
`runs/steam/coherent_search/*-smoke-r2/`. The rev-1 smoke outputs stay in `*-smoke`, `*-smoke-j2` and `smoke-v1/`;
`coherent_summary.py` never reads them. The null is matched to the smoke budget (best of 6 or 4 held-out items).

| Run | Target | Ceiling | Matched null | Corpus q99 | Seeds | h_all | h_coh | Judge 1 / 2 | Corpus max | Seed div. | Tokens in / out | Write / judge s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gateway gpt-6-luna, T=2 K=2 | FAR 0 | 0.030 | -0.038 | -0.010 | -0.078 | -0.065 | -0.065 | 10 / 9 (Haiku) | 0.694 | 0.30 | 2487 / 2581 | 240 / 32 |
| gateway gpt-6-luna, T=2 K=2 | GAP 0 | 0.516 | 0.394 | 0.440 | 0.290-0.291 | 0.362 | 0.362 | 8 / 9 (Haiku) | 0.789 | 0.46 | 2548 / 2923 | 204 / 49 |
| gateway gpt-6-luna, T=2 K=2 | REAL 0 | 0.803 | 0.469 | 0.546 | 0.391-0.408 | 0.446 | 0.446 | 9 / 9 (Haiku) | 0.698 | 0.26 | 2550 / 2724 | 218 / 49 |
| claude-haiku-5-5, T=1 K=2 | FAR 0 | 0.030 | -0.043 | -0.010 | -0.083 to -0.068 | -0.068 | -0.068 | 9 / 9 (Codex) | 0.671 | 0.31 | - | 204 / 66 |
| claude-haiku-5-5, T=1 K=2 | GAP 0 | 0.516 | 0.384 | 0.440 | 0.212-0.298 | 0.316 | 0.316 | 9 / 8 (Codex) | 0.678 | 0.57 | - | 211 / 516 |
| claude-haiku-5-5, T=1 K=2 | REAL 0 | 0.803 | 0.456 | 0.546 | 0.412-0.439 | 0.439 | 0.439 | 9 / 9 (Codex) | 0.661 | 0.35 | - | 235 / 188 |
| haiku `random`, 4 seeds | FAR 0 | 0.030 | -0.043 | -0.010 | -0.101 to -0.052 | -0.052 | -0.052 | 9 / 10 | 0.701 | 0.30 | - | 138 / 470 |
| haiku `random`, 4 seeds | GAP 0 | 0.516 | 0.384 | 0.440 | 0.245-0.309 | 0.309 | 0.309 | 9 / 10 | 0.653 | 0.52 | - | 185 / 86 |
| haiku `free`, T=1 K=2 | FAR 0 | 0.030 | -0.043 | -0.010 | -0.067 to -0.022 | -0.022 | (no gate) | 10 / - | 0.670 | 0.26 | - | 150 / 78 |
| haiku `free`, T=1 K=2 | GAP 0 | 0.516 | 0.384 | 0.440 | 0.218-0.262 | 0.267 | (no gate) | 9 / - | 0.755 | 0.50 | - | 171 / 72 |

- All 46 candidates were valid and inside the bounds, none was flagged, and all 41 judged ones got judge 1 ≥ 8. In a
  first rev-2 draft, 3 of 8 Sonnet and 2 of 12 Haiku candidates broke an About bound counted on the uncut text; the
  bound now counts the rendered (1200-character) body, like the corpus.
- The diversified seeds do differ (party game underwater, flight sim in a wasteland, Wild West tactical shooter...;
  mean pairwise cos 0.26-0.57), but the later "bold exploration" slots still drift back to atmospheric narrative games.
- Every smoke run, REAL included, ended **below the corpus q99 and below the matched null**, i.e. below the best of
  4-6 random real items. The K=4 timing probe (8 candidates, REAL 0) did beat its 8-item null (0.49-0.50 vs 0.47) but
  not q99 (0.546). With 4-8 candidates this says little; it is the reason for the forecast above.
- `coherent_summary.py --runs <smoke runs> --no-check` prints every table and marks C "UNDERPOWERED" (median
  h_coh − q99 = −0.100), P1 "uninformative" and P2 "inconclusive", labelled as not binding. Without `--runs` it reads
  only `MAIN_RUNS` and reports "no runs found".
- Timing probe, K=4 T=1 on Steam REAL 0 (`*-timing-k4`): gateway 61 s writing (a K=4 round ≈ 1400 in / 1215 out
  tokens, ≈ 40 s), Sonnet 81 s writing (≈ 50 s per K=4 round) and 385 s judging with Codex under load. These and the
  smoke times were measured while the 12-worker calibration ran, so they are upper bounds.
- Gateway spend in rev 2 (smoke + timing, 33 calls): 0.012 USD. Rev 1: 0.0053 USD.

## Cost and time of the full run

100 main targets (2 domains × (20 FAR + 20 GAP + 10 REAL)) and 40 control targets (2 domains × 2 modes × 10).

| Writer | Spend | Wall clock |
|---|---|---|
| gateway gpt-6-luna (PAID), main targets only | per target ≈ 4 seeds × 112 / 405 + round 1 ≈ 1400 / 1215 + 9 rounds ≈ 2800 / 1215 (8-10 shown) ≈ 27k in / 14k out ≈ 0.0096 USD; **≈ 1.0 USD in total**. The script's projection is conservative (0.016 per target); pass `--max-usd 1.5`, a hard total over both domains | ≈ 20 s seeds + 10 × 40-45 s + ~1 min judging (Sonnet + Haiku) ≈ 9 min per target; 8 workers ≈ **2 h** |
| claude-sonnet-5-5 (subscription, primary) | 0 | main: ≈ 30 s seeds + 10 × 55 s + 2-4 min judging (Codex 30-60 s per call) ≈ 13 min per target → 100 targets, 4 workers ≈ **5.5 h**. Controls: random ≈ 8.5 min, free ≈ 11 min per target → 40 targets, 4 workers ≈ **1.6 h**. Together ≈ **7 h** (≈ 5 h with 6 workers, longer if usage limits hit) |
| claude-haiku-5-5 (subscription, optional) | 0 | ≈ 9 min per target → 100 targets, 4 workers ≈ **3.8 h** |

The gateway run is above the 0.50 USD threshold and needs the owner's explicit OK.

## Revision 2: what changed and why

Two adversarial reviews of rev 1 returned 18 findings (2 were the same smoke-filter bug). All were accepted.

| Finding (severity) | Change in rev 2 |
|---|---|
| P1 cannot fail informatively: a weak optimiser and a coherence wall look the same; unmeasured counted as "within" (blocker) | REAL positive control C gates every negative verdict; coverage ≥ 90% required; unmeasured = missing; verdict text "no coherent escape exhibited within 44 candidates" |
| No control separates coherence from optimiser weakness; budget gap to the GPU search unstated (major) | budget stated (44 vs ~2,891 gradient-guided); `free` (no coherence) and `random` (no feedback) controls on 5 + 5 targets per domain |
| Summary let `*-smoke-j2` through and never checked run configs (major, reported twice) | summary reads only the pre-registered `MAIN_RUNS` and asserts each record's config (mode, T, K, judges, threshold, cap, sample, prompt version) |
| P2 unfalsifiable; anchoring comparison had no rule (major) | P2 is a one-sided exhibit whose miss is informative only when C passes; P4 adds a paired rule against `grid-claude-sonnet-5-5` with a bootstrap CI |
| Judge gate never calibrated against fluent degenerate text; no length bounds (major) | calibration of the three judges on corpus, GPU and constructed degenerate texts (§Judge calibration); corpus-p99 field bounds and the 511-token window. Scaled to 25 + 25 + 50 texts per domain (not 50 + 50 + 50) to fit the CLI time |
| Seeds collapse onto the model's mode; history shows only the best (major) | per-seed constraints from fixed lists; each round also shows the whole latest round and the 2 lowest; seed diversity reported |
| Errors kill a target and lose its writing; `ex.map` re-raises (major) | per-target try/except (left undone, resumable), candidate checkpoint before judging, timeouts and CLI errors are failed candidates (3 per target fail the target), usage limits and the spend cap stop cleanly |
| P3 had no verdict, used medians of per-target rates and counted duplicates (major) | pooled rates over distinct candidates; the judge rate uses only the random sample; P3 verdict printed |
| k=0 chose the design (minor) | test set is k=1..20 (REAL k=1..10); k=0 is smoke only |
| FAR ceiling biased low by target selection (minor) | held-out ceiling and matched held-out null per target; an escape must beat max(ceiling, held-out ceiling) + 0.03, and a FAR "supported" verdict needs h_coh ≥ the null |
| h_coh unmeasured when the top candidates fail; P3 weighted to the top (minor) | gate walks down the score list until both judges accept, cap 15; P3 on the random sample |
| Greek / maths letters flagged as gibberish (minor) | whitelisted; firing rule recorded; rates rechecked (arXiv corpus 1.2% → 0.0%, witnesses still 98-99%) |
| Gateway writer judged by its own family (minor) | judge 2 is Haiku for the gateway writer; families recorded per target |
| Spend guard projected per invocation only (minor) | ledger spend under the run label is checked before every call, as a total over invocations and domains |
| One Gateway per target, unshared ledger lock (minor) | one shared Gateway |
| Resume checked only T and K (minor) | full config compared on resume and on checkpoints |
| Gateway answers not in `{"candidates": [...]}` were lost silently (minor) | a bare list or object is accepted; a round with 0 parseable candidates logs a warning |

Also changed: the decision targets, reference values and forecast above; the rev-1 forecast (GAP medians 0.45 / 0.35)
is kept as the conditional forecast.

## Commands

```bash
# main search, primary writer (free)
uv run python scripts/coherent_search.py --domain steam --backend claude:claude-sonnet-5-5 --workers 4
uv run python scripts/coherent_search.py --domain arxiv --backend claude:claude-sonnet-5-5 --workers 4
# controls (free)
for m in random free; do for d in steam arxiv; do
  uv run python scripts/coherent_search.py --domain $d --mode $m --backend claude:claude-sonnet-5-5 --workers 4
done; done
# only after the owner's OK (cap is the total over both domains):
uv run python scripts/coherent_search.py --domain steam --backend gateway --workers 8 --max-usd 1.5
uv run python scripts/coherent_search.py --domain arxiv --backend gateway --workers 8 --max-usd 1.5
uv run python scripts/coherent_summary.py   # reads only MAIN_RUNS and checks their config
```

## Result (2026-10-26; `scripts/coherent_summary.py`, full output in runs/coherent_logs/summary.txt)

All runs complete, no failures: 50 targets per main run (FAR 20, GAP 20, REAL 10) per domain and writer, controls 10
per domain. Medians over targets:

| domain | arm | ceiling | matched null | h_coh sonnet | h_coh luna | anchored decoder | h_any (GPU) |
|---|---|---|---|---|---|---|---|
| steam | FAR | 0.042 | -0.006 | 0.007 | 0.021 | 0.031 | 0.161 |
| steam | GAP | 0.533 | 0.449 | 0.435 | 0.380 | 0.523 | 0.669 |
| steam | REAL (control) | 0.729 | 0.573 | 0.590 | 0.504 | 0.763 | - |
| arxiv | FAR | 0.055 | 0.012 | 0.008 | 0.017 | 0.071 | 0.174 |
| arxiv | GAP | 0.428 | 0.335 | 0.370 | 0.348 | 0.476 | 0.553 |
| arxiv | REAL (control) | 0.739 | 0.461 | 0.447 | 0.425 | 0.819 | - |

Verdicts as pre-registered: **C fails in every run** (the blind loop cannot recover a hidden real item: REAL h_coh sits
0.15-0.31 below that item's own ceiling and at or below the corpus q99), so **P1 (FAR) is "uninformative:
optimiser underpowered" and P2 (GAP) "inconclusive"** for both writers and domains. P3 holds (100% of judged
candidates coherent, >= 99% pass the gibberish flag). P4: the anchored decoder lands closer than the blind search on
every arm, domain and writer (paired CI excludes 0 everywhere).

What the run does show: (1) score feedback adds little over independent sampling (search vs `random`, same targets:
arXiv FAR 0.022 vs 0.004, GAP 0.369 vs 0.350; Steam GAP 0.403 vs 0.383); (2) dropping the coherence requirement does
not help (`free` is no better than `search`), so in this black-box regime coherence is not the binding constraint,
the optimiser is: a scalar score in a 1024-d space with 44 evaluations is not enough to steer an LLM writer; (3) the
corpus neighbours are what make decoders reach the ceiling (anchoring law), and without them an LLM writer does not
find its way even to a known real item. The owner's objection is therefore neither supported nor refuted here. A
powered test of h_coh needs an optimiser that is strong without neighbours, e.g. the GPU token search of
kaggle/support-search with a fluency penalty (maximise cos - lambda * LM perplexity) and the same judge gate.

Gateway cost (gpt-6-luna writer, ledger tags `cohsearch-*`): 1.10 USD (cap 1.5).
