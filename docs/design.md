# Latent gaps as a source of new ideas — design and pre-registration

Status: **pre-registered before any idea was generated** (2026-10-22). Changes after data are logged at the end.

## Claim under test

A viral post argued that AI *can* create brand-new ideas: embed every Steam game, pick a point in vector
space far from all neighbours, convert it back to a description, and rank the results by how unlike
existing ideas they are.

Three assumptions are hidden in that recipe, and each can fail on its own:

1. **Far = new.** In 1536 dimensions almost every point is far from every other (concentration of
   measure), and real texts occupy a narrow cone of the embedding space. A point far from all items may
   simply be off the manifold of text.
2. **A point can be turned back into text.** If most of the space contains no coherent text, the decoder
   will land somewhere else: we expect *regression to the manifold*, the decoded text re-embedding much
   closer to existing items than the target was.
3. **Unlike = good.** A region can be empty because nobody thought of it or because it makes no sense.
   The only outside judge of that is time: did the world later fill the gap?

## Domains

| Domain | Corpus | Past (space) | Future (validation) | Quality signal |
|---|---|---|---|---|
| Steam (pilot) | Kaggle `fronkongames/steam-games-dataset`, English, about ≥ 200 chars | release < 2024-01-01 (71,939) | 2024–2026 (41,497) | reviews, positive ratio, owners |
| arXiv cs.* (main) | Kaggle `Cornell-University/arxiv`, title + abstract | first version < 2023-01-01 | 2023–2026 | citations (Semantic Scholar), later |

Patents (HUPD, grant/reject label) are a candidate third domain **only after** Steam and arXiv are closed.

## Models

- Embedding space: `Qwen3-Embedding-0.6B` (open source), L2-normalised, last-token pooling, 512 tokens,
  documents embedded without an instruction. Corpus embedded on Kaggle (GPU, fp16, sentence-transformers);
  decoder outputs embedded locally with the ONNX export (CPU, fp32). Parity checked on a corpus sample.
  Robustness space: `bge-m3`, used only to re-score finished outputs.
- Generator / decoder: `gpt-6-luna` (gateway alias `gpt-6-luna`).
- Judge: `gpt-5.6-sol` (`gpt-5.6-sol`), a different model than the generator. Same vendor family —
  a stated limitation.

## Arms (50 ideas each, per domain)

Targets are built from the **past** corpus only.

| Arm | Target | Role |
|---|---|---|
| `REAL` | embedding of a held-out past item (itself and near-duplicates, cos > 0.95, hidden from the decoder) | ceiling: how well can the decoder land on a point we know is text? |
| `INTERP` | normalised midpoint of two past items with cosine below the corpus 10th percentile | "combine distant concepts" in geometry |
| `NOISE` | ladder: from a held-out real item x towards a random direction r, `norm((1-l)x + l r)`, with l solved so isolation hits {0.35, 0.45, 0.55, 0.70, 0.85} (10 targets per level) | where decoding breaks as the target leaves the data cone |
| `CROWD` | ladder: from x away from the centroid c of its 20 nearest items, `norm(x + l(x - c))`, same levels (0.85 is unreachable along this path; the reached isolation is recorded) | peripheral points that stay in the span of real items |
| `FAR` | from 20k random Gaussian directions, the 50 with the lowest max-cosine to the corpus | the post's literal recipe: far from everything |
| `PLAIN` | no target: "invent a brand-new game unlike any existing one" | LLM without the space |
| `PAIR` | no target: two random distant past items, "invent one new game that combines them" | the human heuristic (astronaut rides a horse) |

**Decoder (closed loop, 4 iterations × 3 candidates, best kept).** The LLM gets the 8 nearest visible past items
to the target with their similarity, writes a new item in the domain's canonical format, the item is embedded,
and the next prompt reports only the similarity reached and the best so far (score-only feedback). The
residual-guided variant first planned ("move towards / away from" lists) lost the A/B; see the change log.
Every label used in results is defined in [glossary.md](glossary.md).

## Measurements

- **E1 Landing**: cos(decoded, target), per arm.
- **E2 Regression**: isolation = 1 − max cos to the past corpus, for the target and for the decoded text.
- **E3 Judged quality**: every item, including 50 real future items as calibration, is first rewritten by a
  normaliser prompt into the same 120-word pitch (so style cannot reveal provenance), then scored blind by
  the judge, 1–10: coherence, feasibility, appeal; novelty is judged against the 5 nearest past items.
- **E4 Time**: for each decoded item, `past_sim` = max cos to past, `fut_sim` = max cos to future;
  *anticipation* = `fut_sim − past_sim`. Base rate: the same quantity for held-out past items.
  For targets: growth of the region (future items with cos > τ divided by past items with cos > τ) against
  the global ratio (41,497 / 71,939). Quality of the nearest future items (reviews) for items that
  anticipated.

Tests: Mann–Whitney between arms with bootstrap 95% CIs on medians; Holm correction within each
hypothesis family.

## Pre-registered predictions

- **H1** landing: `REAL > INTERP > FAR`, and landing falls monotonically with the ladder level in `NOISE` and `CROWD`.
- **H2** regression: on the ladders and `FAR` the decoded isolation stays near that of real items whatever
  the target isolation (aiming further buys nothing).
- **H3** the post's claim: decoded ladder items are more isolated than `PLAIN` at equal judged coherence. We
  predict *more isolated, less coherent*; the claim holds only if coherence does not drop.
- **H4** time: ladder target regions do **not** grow faster than `REAL` regions, and decoded ladder ideas
  do not anticipate future items more than `PAIR` ideas.

## Study 2 (after the pilot, same infrastructure): coherence as a landscape

A counter-argument raised in discussion: most of the space is junk, but coherence behaves like an extra
dimension with *local optima* — pockets that are both unlike everything and meaningful — and image
diffusion shows that such pockets exist (an astronaut riding a horse) but must be *reached*. The difference
we flagged: a diffusion model samples on the data manifold because it is trained to denoise towards it,
and it starts from a concept and moves by similarity; the post's recipe seeks dissimilarity with no prior.
Study 2 makes the two procedures analogous:

- learn a coherence probe `c(x)` on the embedding (real items vs. judged-incoherent decodes from the pilot);
- **guided search**: maximise isolation subject to `c(x)` high, as classifier guidance does in diffusion;
- **rooted walks**: start from a real concept and step outward along residual directions, monitoring
  coherence at every step, as diffusion starts from a prompt concept;
- compare against the ladders and `PAIR` with the same E1–E4.

Added after the pilot's geometry analysis ([geometry.md](geometry.md)), to run before the guided search:

- **`GAP` arm (Steam first)**: targets drawn from the corpus's own Gaussian N(μ, Σ), keeping the most isolated of
  20k draws, so they sit inside the cap but where no game is. Pre-registered from §6–§8 of geometry.md, before any
  decoding: ceiling ≈ 0.54, landing ≈ κ · ceiling ≈ 0.50–0.55 per generator (κ from the pilot), decoded items
  closer to the target than to the shown neighbours' centroid (the opposite of FAR), and judged novelty between
  INTERP's and PLAIN's. If landing tracks κ · ceiling but novelty stays at the ladders' level, in-distribution gaps
  are reachable but not new; if novelty approaches PLAIN, they are the guided version of the tweet that works.
- **Cross-domain laws (arXiv, then patents)**: the same geometry script per domain. Predictions: κ ≈ 1 for every
  generator; ceiling ≈ μ·t + √(tᵀΣt) z_n for targets away from the cap's centre; the random-direction slope equals
  √(tr Σ / d); and the corpus size needed to cover gaps scales with the domain's effective dimension (participation
  ratio), so a lower-dimensional domain should make far targets and gaps markedly easier to fill.
- **arXiv FAR ceiling, parameter-free (bounds.md §4), written 2026-10-24 after embedding the arXiv corpus and
  before building any arXiv target.** Corpus statistics only: N = 72,000, d = 1024, ‖μ‖ = 0.5428. Prediction for the
  median corpus ceiling of the 50 FAR targets (most isolated of 20k uniform directions):
  (z_N √(1 − ‖μ‖²) + ‖μ‖ Φ⁻¹(50/40,000)) / √d = **0.062** (Steam, same formula: 0.044, observed 0.046). Success:
  absolute error ≤ 0.02 (Astra's criterion); strict: ≤ 0.005 (Steam's error). Also predicted for arXiv: κ ≈ 1 per
  generator on REAL; FAR decodes closer to the shown neighbours' centroid than to the target (anchoring law), with
  landing ≈ anchoring × centroid score.
- **Patents pre-registration, written 2026-10-26 before building any patents target** (HUPD, filed 2012-15 visible,
  N = 71,521, ‖μ‖ = 0.511; same protocol and generators as arXiv: sonnet and luna-Codex, 8 arms x 50 + PLAIND,
  pitches without results). Predictions: (1) median corpus ceiling of the 50 FAR targets (most isolated of 20k uniform
  directions) = 0.0625 ± 0.005 (the battery's measurement on this corpus with other directions,
  docs/prereg/conjecture_battery_results_patents.csv; the parameter-free formula says 0.068 and is expected to
  overpredict by ~0.005 here); (2) landing ≈ κ · ceiling with κ in [0.9, 1.1] per generator on REAL; (3) FAR and GAP
  decodes closer to the shown neighbours' centroid than to the target for ≥ 95% of items (anchoring); (4) GAP novelty
  not above INTERP (Steam: below; arXiv: equal); (5) PLAIN collapses for at least one generator (median pairwise cos >
  0.6) and PLAIND brings it within 0.1 of the guided arms; (6) PLAIND novelty > every guided arm for both generators
  (bootstrap CI vs INTERP excludes 0). No anticipation test: the future split (2016-17) is inside every generator's
  training data.
- **Patents outcome (2026-10-27; runs/patents/geometry.log, prereg_checks.log from scripts/patents_prereg.py).**
  Decoded 450/450 (luna) and 442/450 (sonnet: 8 items refused twice by Claude's biology safeguard, mostly FAR); four
  subscription judges cover 97-100% of items (the gaps are Haiku refusals). (1) FAR ceiling 0.062: holds (formula
  0.068, overpredicts by 0.006 as expected). (2) κ = 1.01 (luna), 1.09 (sonnet): holds. (3) FAR and GAP decodes closer
  to the shown neighbours' centroid than to the target in 100% of items for both generators: holds. (4) GAP − INTERP
  novelty (mean of four judges, bootstrap 95% CI): luna −0.21 [−0.74, +0.33] holds; sonnet +0.72 [+0.29, +1.15]
  **fails**. Sonnet's guided decodes stay very close to the shown patents (INTERP landing 0.87, novelty 3.2, coherence
  3.7), so on patents its INTERP is a near-copy that judges rate as neither new nor coherent; GAP is less anchored to one
  patent. (5) PLAIN collapses for sonnet (median pairwise cos 0.735; luna 0.566): holds; PLAIND brings it to 0.394
  (luna 0.382), within 0.1 of the least diverse guided arm (GAP 0.317 / 0.332) but 0.12 above the median guided arm for
  sonnet (0.08 for luna): holds in the weaker reading only. (6) PLAIND novelty above every guided arm, CI vs INTERP
  +2.09 [+1.65, +2.54] (luna), +4.83 [+4.46, +5.18] (sonnet): holds. Score: 4 hold, 1 holds in the weaker reading, 1
  fails.
- **GAP outcome (2026-10-24, five subscription generators; gateway luna added 2026-10-26; geometry.md §11):**
  ceiling 0.541 and landing ≈ κ · ceiling hold; "closer to the target than to the shown neighbours' centroid" fails
  (0/300, predicted instead by the anchoring law of bounds.md §3); novelty 4.91, below INTERP (5.52) for all six
  generators, so the in-distribution version of the tweet is reachable but not new.
- **PLAIN collapses towards one idea per generator (found 2026-10-24).** Each PLAIN call is independent and sends
  the same prompt (only "request number k" changes), so the generator keeps returning its modal answer. Sampling
  temperature is not a variable here: these reasoning models accept only the default (Azure rejects any other value for
  gpt-6-luna), so no claim about it is made. Why luna repeats more via Codex than via the gateway was measured afterwards
  (scripts/collapse_probe.py, exploratory, 50 PLAIN items per condition, median pairwise cosine [bootstrap 95% CI]):
  gateway as in the grid 0.589 [0.564, 0.613], identical repeat 0.572 (run-to-run noise ~0.02); reproducing each Codex
  difference on the gateway changes nothing: reasoning effort medium 0.592, strict output schema 0.573, flattened
  prompt 0.584, all three 0.593. Through Codex: effort medium 0.824 [0.790, 0.848] and low 0.824 (45/50 "Museum of…"),
  high 0.601 (20/50 museums). The collapse is the Codex harness itself (its own system prompt and context, the one
  difference that cannot be removed) at low or medium effort; at high effort it disappears.
  Median pairwise cosine among a generator's 50 PLAIN items: sonnet 0.864 (45/50 named "Tidewright…"), luna-Codex
  0.811 (46/50 "Museum of…"), sol61 0.632, luna-gateway 0.574, haiku 0.569, opus 0.520; 50 random corpus games 0.404;
  guided arms 0.35–0.49. Judged novelty is per item against the catalogue, so "PLAIN is the most novel" holds item by
  item but not as a source of many distinct ideas. Any claim about PLAIN must report within-arm diversity alongside
  novelty; a diversity-controlled PLAIN (e.g. previous titles shown, or one call asking for N distinct ideas) is the
  fix if PLAIN is to be compared as an idea generator.
- **Caveat for every judged result**: when landing is ≈ 0 (FAR), the judged text does not represent its target;
  FAR's novelty scores are those of texts written from its shown neighbours, not of the far point.

## Budget (pilot, Steam)

Corpus embedding on Kaggle (free GPU quota). Decoder ≈ 1,000 calls of gpt-6-luna (≈ 3k in / 0.6k out)
≈ 0.6 USD. Normaliser + judge ≈ 700 calls of gpt-5.6-sol ≈ 10–16 USD. All on a gateway
(Azure), in one or two days.

## Change log

- 2026-10-22, before any data: primary space moved from `text-embedding-3-small` (gateway) to
  `Qwen3-Embedding-0.6B`. Reason: the shared embedding deployment is capped at 120k TPM and shared with other
  services; the corpus would have taken ~4.5 h at saturation.
- 2026-10-22, smoke tests (n=2 and n=5 per arm, not part of the pilot):
  - `VOID` dropped. Normalised mixtures fall towards the centre of the data cone: their isolation (0.27) was
    *below* that of a real item (0.31), so the arm measured the opposite of its name. `FAR` targets sit at
    isolation 0.97, entirely outside the cone, and decode to items as isolated as ordinary games (0.29):
    nothing was measured between the two. Replaced by the `NOISE` and `CROWD` ladders.
  - Items tagged sexual/nudity/NSFW/mature (4,225, 3.7 %) excluded from the space: Azure's content filter
    rejects prompts that show them as neighbours.
  - Decoder: best-of-3 per iteration, 4 iterations. The residual "towards/away" lists made the second
    iteration worse on all 15 smoke targets; a three-way comparison (residual, score-only feedback, one-shot
    best-of-12; same 12 calls) on identical targets decides the pilot decoder before the pilot runs.
  - Decoder comparison result (15 targets, paired): score-only feedback best on 11/15 (mean landing 0.536 vs
    0.515 one-shot best-of-12 vs 0.501 residual lists). Pilot decoder: score-only feedback, 4 × best-of-3.
- 2026-10-22, pilot: the luna normaliser sometimes spent its token budget on reasoning and returned empty pitches;
  33 of 392 items were scored 1/1 by every judge. Fixed before any reported number: inputs validated (empty pitches
  rejected, retried with larger budgets, Codex luna fallback), prompt hashes stored with every judgment, affected
  items re-judged. Five judges (gpt-5.6-sol on the gateway; gpt-6.1-sol via Codex; Opus, Sonnet, Haiku 5.5 via
  Claude Code).
- 2026-10-22, explicit content: ~3 % of past rows read as explicit despite the adult-tag exclusion; flagged post hoc
  and kept (excluding them would shift target indices), with a sensitivity table that leaves them out.
- 2026-10-23, generator grid: the pilot's targets decoded by gpt-6-luna and gpt-6.1-sol (Codex) and Opus, Sonnet,
  Haiku 5.5 (Claude Code), 350 items each, judged by the four subscription judges. Same geometry with every
  generator; PLAIN − best guided arm on novelty +1.1 to +2.6, CI excluding 0 for all.
- 2026-10-23, geometry analysis and human check: geometry.md (cap, efficiency law, two-moment ceiling,
  extrapolation, bge-m3 replication, Lean proofs); 16 blind human ratings (novelty ρ = +0.59 with the judges'
  mean, appeal ρ = −0.75). `GAP` and the cross-domain laws added to Study 2 as pre-registered predictions.

## Inspiration for the anticipation test (2026-10-25)

Cite: P. Ball, "The Einstein test: what happens when AI tries to rediscover relativity?", *Nature* 657, 338-340
(2026), doi:10.1038/d41586-026-02804-x. It reports Demis Hassabis's proposal (India AI Summit, New Delhi, February
2026): train a model only on knowledge before a cutoff (he suggested 1911) and see whether it reaches general
relativity. Later press gives a 1901 / 1905-papers variant; no primary transcript found, so cite Ball, not a paraphrase.
Related: T. Zahavy, "Position: LLMs can't jump", ICML 2026, PMLR 306:172863-172872 (LLMs lack abduction; general
relativity as case study); D. Shi, X. Li, J. M. Hernández-Lobato, arXiv:2608.26187 (formalises and measures the jump).

Our version (part 1) is weaker and measurable: does an idea land closer to items published after the split than to any
item before it (scripts/anticipation.py), with memory handled by MEM (docs/prereg/memory_probe.md). Part 2, planned:
the experiment proper, a generator with a controlled training cutoff (open model, with and without later data).

## Results in paper pitches (2026-10-25)

The owner noticed that real abstracts report measured results and generated ones can only invent them: pitches kept
them, so results unblinded real items and rewarded fabrication. Pitches with a number: FUTURE_REAL 18-20%, decoded
arms 0-8%, PLAIN 14% (luna) and 52% (sonnet, invented). Fix: `Domain.pitch_rule` for arXiv and patents (problem,
approach, what is distinct; no results, numbers, benchmarks or claims of what was shown); Steam prompt unchanged byte
for byte. All arXiv items re-pitched and re-judged (old judgements in runs/arxiv/grid-*/legacy/with-results/).
Effect, mean of 4 judges x 2 generators: FUTURE_REAL novelty 6.16 -> 5.94, appeal 6.66 -> 6.48; PLAIN appeal
6.24 -> 6.00; every other arm moves < 0.12; the ranking is unchanged except that PLAIN (7.36 -> 7.26) and PLAIND (7.34 -> 7.37) swap at the top, within noise. Human arXiv batch A1 (with results) is void;
A2 has the same 16 items with the new pitches.
- **Judge criteria are not independent (raised by the owner, 2026-10-28; `scripts/criteria_dependence.py`,
  runs/criteria_dependence.log).** Item-level Spearman (mean of judges): coherence ~ feasibility 0.35 (Steam) / 0.62
  (arXiv) / 0.63 (patents), coherence ~ appeal 0.41 / 0.66 / 0.79; each coherence point moves feasibility by +0.6-0.8
  and appeal by +0.4-0.7. Novelty is nearly independent of coherence (ρ 0.02 / 0.05 / 0.18; slope −0.26 to +0.0) and
  trades off against feasibility (ρ −0.27 to −0.33). Consequences: (a) every novelty comparison stands, raw,
  coherent-only (coherence ≥ 7) and coherence-adjusted (OLS criterion ~ arm + coherence) give the same order; (b) the
  low feasibility/appeal of the incoherent guided arms (Sonnet INTERP/FAR/GAP on arXiv and patents) is mostly
  incoherence: patents Sonnet INTERP feasibility 5.4 raw → 6.8 adjusted, at REAL's level; (c) PLAIN/PLAIND's low
  feasibility is not (patents Sonnet PLAIN 4.2 raw, 3.6 adjusted). Feasibility and appeal are reported raw and
  coherence-adjusted from here on; the adjusted value is a direct-effect reading, since the arm itself causes the
  incoherence.
- **Human batch P1, patents (owner, non-expert, n = 16, 12 luna + 4 real 2016-17; rated 2026-10-28;
  runs/patents/humanP1).** Novelty per arm, owner vs mean of four judges: PLAIND 5.8 vs 7.95, FUTURE_REAL 4.5 vs 5.7,
  GAP 4.3 vs 6.25, FAR 4.0 vs 7.0. Same top arm as the judges (PLAIND), but everything is less new to the owner and
  the item-level agreement is weak (novelty ρ = 0.13, coherence −0.19; feasibility 0.64). "Do you know this or
  something that does the same?": yes on 3/4 FAR and 1/3 GAP generated items ("un recommendation system de toda la
  vida"), on 3/4 real applications, and on 0/5 PLAIND (1 unsure). So the guided generated patents read as known prior
  art to a reader, which the judges, comparing with the 5 shown neighbours only, did not see. Owner's criterion
  matrix (scripts/human_criteria.py): coherence ~ feasibility 0.78, ~ appeal 0.48, ~ novelty −0.42; pooled over all
  80 owner ratings (Steam 48, arXiv 16, patents 16): 0.51 / 0.35 / −0.28, feasibility ~ novelty −0.43.
- **Where the owner's novelty departs from the judges' (scripts/human_novelty_gap.py, 64 items with judges).**
  Item agreement is weak everywhere after batch 1 (ρ Steam B2 −0.32, B3 0.36, arXiv 0.34, patents 0.13). The gap is
  not uniform: patents −1.2 to −3.0 on every arm (FAR −3.0), Steam PLAIN −2.4 (the repeated museums the judges
  never saw side by side), arXiv near zero except real future papers (+1.7, the owner finds them newer than the
  judges do). In patents the gap grows with the owner's "known" answer (no −1.6, unsure −2.6, yes −2.3), and the
  largest disagreements are coherent, ordinary items the judges call new (FAR-34: owner coherence 10, novelty 2;
  judges 6.2 / 7.0; GAP-23 "un recommendation system de toda la vida"). Reading: the judges' novelty rewards
  specific, unfamiliar wording against five shown neighbours; the owner's asks whether the concept already exists.
  The few large positive gaps are Steam GAP items whose core idea survives the oddities ("quitándole algunas cosas
  excesivamente estrafalarias puedes tener un buen juego").
