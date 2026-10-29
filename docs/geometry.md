# Why far points do not decode: the geometry

Numbers from `scripts/geometry.py` (Steam, Qwen3-Embedding-0.6B, d = 1024, past corpus n = 68,222, vectors
renormalised) over the pilot and the five generator-grid runs, and from `scripts/geometry_bgem3.py` for the bge-m3
check (§9). Logs: `runs/steam/geometry.log`, `geometry_bgem3.log`; per-item values: `geometry_items.csv`. Labels:
[glossary](glossary.md). Exact identities and bounds are proved in Lean (`lean/`, §10). Unless stated otherwise a
number is a median over items, per arm and generator, for the decoded arms (REAL, INTERP, NOISE, CROWD, FAR).

Revised 2026-10-23 after an adversarial review (gpt-6-astra, `runs/steam/astra_review.md`): the corpus ceiling is a
benchmark, not a universal limit; the extrapolation is conditional; the effective-dimension covering argument and
the "in-slab only" wording were wrong and are removed.

## 0. Terms

- **Target t**: the point of the space we ask the generator to fill. FAR targets are uniform random directions: 20k
  draws, keeping the 50 whose most similar past game is least similar.
- **Shown neighbours**: the 8 past games nearest to t, listed in the prompt with their similarity to t; for a FAR
  target they are the closest games to a point far from all of them (cos ≈ 0.05).
- **Landing**: cos(e, t) between the decoded item e and its target; 1 would be exactly on it.
- **Ceiling(t)**: max over the visible past corpus (rows not hidden for t) of cos(x, t). For REAL the hidden rows
  include the game itself, so its full-corpus maximum would be ≈ 1. It is a **benchmark**: what the best existing
  game reaches, not a bound on what any text could reach (decoders sometimes exceed it, §4).

## 1. Texts fill a narrow cap

Two independent random directions in d = 1024 have cos 0 ± 0.031. Two Steam games have a median cos of **0.41**
(5–95 %: 0.28–0.53). The corpus mean has norm 0.64 and the median game sits at cos 0.64 from the mean direction μ̂.
The top second-moment direction carries 41 % of the energy; 90 % fits in k = 232 of 1024 dimensions (centred
participation ratio ≈ 112). With U_k those 232 directions and m(v) = ‖U_kᵀ v‖:

| | m(v), median |
|---|---|
| a real game | 0.95 |
| a uniform random direction | √(232/1024) = 0.48 |
| REAL / INTERP / CROWD / NOISE targets | 0.95 / 0.96 / 0.87 / 0.68 |
| FAR targets | **0.47** |
| decoded items, decoded arms, every generator | 0.957 (per item 0.881–0.982, 5–95 %: 0.934–0.974) |
| PLAIN / PAIR items | 0.95–0.96 by generator |

Whatever the target, decoded items have the in-slab fraction of a real game and its latitude around μ̂
(cos(e, μ̂) 0.53–0.65; corpus 0.64). FAR targets have cos(t, μ̂) = −0.08.

## 2. The split of landing

cos(e, t) = ⟨U_kᵀe, U_kᵀt⟩ + ⟨rest⟩ exactly (Lean: `inner_split`), and Cauchy-Schwarz gives landing ≤ m(e)m(t) +
√(1−m(e)²)√(1−m(t)²) (`landing_le`), which always holds but is loose (0.73 for FAR). The out-of-slab part is small in
absolute terms (per item −0.02 to 0.135, median 0.016) although for FAR, where 88 % of the target's norm lies outside
the slab, it is about **37 %** of a landing that is itself ≈ 0.03. So: both parts contribute, and the part of a target
outside the region where texts live adds at most a few hundredths. Across arms landing ≈ 1.24–1.42 · m(t) − 0.55 (R²
0.80–0.86 per generator); that fit is between arms (within REAL the pooled correlation is 0.42; within INTERP 0.10).

## 3. The ladder approximation

A NOISE target is t = norm((1 − l)x + l r), x a real game, r an independent uniform direction, ε = ⟨x, r⟩. Exactly
(`ladder_norm_sq`, `ladder_inner`):

  cos(t, x) = (1 − l + lε) / √((1 − l)² + l² + 2l(1 − l)ε)

and since E ε = 0, Var ε = 1/d (observed SD of ε 0.031), cos(t, x) ≈ (1 − l)/√((1 − l)² + l²) (`ladder_cos` proves
the ε = 0 case): 0.88 at l = 0.35, 0.17 at l = 0.85. The heuristic landing ≈ c · cos(t, x), c the generator's median
REAL landing, tracks the fall across levels (r = 0.96–0.98 pooled over levels, error 0.04); within a level it ranks
items weakly (r 0.14–0.74). CROWD moves a game away from its neighbourhood's centroid, not along independent noise, so
the approximation does not apply there (error 0.12–0.15) and its failure says nothing against it.

## 4. Decoders match the best existing game

For FAR the ceiling is **0.046** and decoders land at 0.026–0.046. Opus exceeds the ceiling on 46 % of FAR items and
70–84 % of REAL, INTERP, CROWD; luna on the gateway on 14–25 %. A permutation baseline (a decoded item against another
target of the same arm) gives −0.05 for FAR, so decoders pick up the weak signal there is. What this shows is that
**these decoders, with this prompting and a 4 × 3 search budget, land about where the best of 68k existing games
already is**; it does not show that no text could land closer.

## 5. Anchoring to the shown neighbours, and the direction decoders move

For FAR the shown neighbours' centroid has cos 0.06 with the target, but the decoded item has cos **0.63–0.72** with
it; the decoded item is closer to that centroid than the target is in 100 % of FAR, 98 % of NOISE and 96 % of CROWD
items. The stronger generators anchor more (Opus/Sonnet 0.72 versus luna 0.63 on FAR), in line with the judges: their
guided items get the lowest novelty and their PLAIN items the highest.

Under a Gaussian surrogate x ~ N(μ, Σ), E[x | ⟨x, t⟩ = s] = μ + Σt (s − μ·t)/(tᵀΣt): the corpus can move towards a
target only along Σt. Decoded FAR items' residuals e − μ align positively with their own target's Σt (cos 0.31–0.38;
another target's Σt: 0.01–0.08), more than the corpus's best item for the target (0.27). This is a directional
association, not a proof that decoded texts are nearest points of the cap.

## 6. Comparing two heuristics: κ · ceiling and the ladder

Fit κ = median(landing / ceiling) on REAL only, per generator (κ uses REAL outcomes only; ceilings exclude hidden
rows; no response leakage), then predict the other arms: κ = 0.92 (luna), 0.95 (luna gateway), 0.97 (Haiku), 1.00
(gpt-6.1-sol), 1.02 (Sonnet), 1.03 (Opus).

| arm | error, κ·ceiling | error, ladder | r within level, κ·ceiling | r within level, ladder |
|---|---|---|---|---|
| NOISE (5 levels) | **0.017–0.033** | 0.034–0.058 | 0.36–0.54 | 0.14–0.74 |
| CROWD (5 levels) | **0.040–0.057** | 0.077–0.213 | 0.30–0.63 | 0.22–0.81 |
| FAR | 0.018 | – | 0.37 (whole arm) | – |

The pooled correlations (0.93–0.98) mostly reflect the ordering of the five construction levels. Within a level
κ·ceiling has the lower error at every level of both ladders and a moderate correlation. It is a descriptive
regularity of these runs, to be validated on fresh targets (Study 2, other domains), not a law of decoders.

## 7. The two-moment approximation to the ceiling

Over the visible rows, a target's scores cos(x, t) have mean m_t and SD s_t. If they behaved like n independent
Gaussian draws, the best would sit near m_t + s_t z_n, with z_n ≈ √(2 ln n) − (ln ln n + ln 4π)/(2√(2 ln n)) +
γ/√(2 ln n) = 4.317 for n = 68k (γ the Euler-Mascheroni constant; an asymptotic approximation to the expected maximum
of iid standard normals). Cosines are bounded and the scores are neither Gaussian nor independent, so this is an
empirical approximation, checked three ways:

| arm | ceiling | predicted | error | r within arm | held-out half: error, bias | z at q = 0.99 / 0.999 / 0.9999 |
|---|---|---|---|---|---|---|
| FAR | 0.046 | 0.048 | **0.005** | 0.69 | 0.005, −0.001 | 2.35 / 3.12 / 3.72 |
| NOISE | 0.450 | 0.424 | 0.034 | 0.98 | 0.032, +0.021 | 2.35 / 3.14 / 3.83 |
| CROWD | 0.450 | 0.391 | 0.056 | 0.95 | 0.057, +0.053 | 2.46 / 3.41 / 4.36 |
| REAL | 0.736 | 0.694 | 0.071 | −0.18 | 0.059, +0.046 | 2.44 / 3.39 / 4.08 |
| INTERP | 0.800 | 0.715 | 0.087 | 0.09 | 0.077, +0.067 | 2.26 / 2.99 / 3.68 |

(Gaussian quantiles: 2.33 / 3.09 / 3.72.) The held-out column predicts the maximum of a random half from the other
half's moments, so the fit is not using the maximum it predicts. Far from the populated centre (FAR) the
approximation is close; elsewhere it under-predicts consistently, and REAL and CROWD have heavier upper tails than a
Gaussian. Why (near-duplicates, clusters) is not tested here.

The formal counterpart (`expected_max_le`, `expected_max_le_of_bounds`) is an upper bound under an explicit MGF
hypothesis E exp(sX_i) ≤ exp(s m_i + σ_i² s²/2) with variance proxies σ_i, not the observed SD: E max_i X_i ≤
max_i m_i + max_i σ_i √(2 ln n), with no independence needed.

## 8. What corpus size would make a far point reachable

For a uniform random direction t and unit corpus vectors, E_t[tᵀΣt] = tr Σ / d and tr Σ = mean‖x‖² − ‖μ‖² = 1 − ‖μ‖²
(`sum_norm_sub_mean_sq`, `sum_directional_variance` prove the basis-sum identity; the expectation over t adds
isotropy): any corpus of unit vectors shows a random direction on average at most 1/d of variance. That bounds the
spread of scores, not the maximum of a heavy tail.

Growth with corpus size, measured without selection leakage: the corpus is split in halves; FAR and GAP targets are
selected against half A, and ceiling(n) is measured on subsamples of half B (n up to 34,111; 5 replicates). Fitting
a + b √(2 ln n) per replicate:

| target class | n = 1k | 8k | 34k | slope b [range] | fitted curve reaches 0.3 at |
|---|---|---|---|---|---|
| RANDOM: unselected uniform directions | 0.070 | 0.085 | 0.093 | 0.027 [0.024, 0.029] | n ~ 10³¹ [10²⁹, 10³⁹] |
| FAR: most isolated vs half A | 0.022 | 0.036 | 0.044 | 0.027 [0.025, 0.028] | n ~ 10⁴⁴ [10⁴¹, 10⁴⁷] |
| GAP: most isolated normalised N(μ, Σ) draws vs half A | 0.499 | 0.525 | 0.536 | 0.045 | already |
| TYPICAL: normalised N(μ, Σ) draws | 0.586 | 0.610 | 0.622 | 0.042 | already |

(Gaussian-model slope for a random direction: √(tr Σ / d) = 0.024; bge-m3: fitted 0.022 vs 0.020.) The powers are
**conditional extrapolations** of a curve fitted between 10³ and 3·10⁴ texts, not measurements: bounded cosines
eventually saturate, and the tail may change with n. An independent benchmark that needs no fit: if texts were
uniform on the sphere of d = 1024, one text within cos ≥ 0.3 of a given direction is expected only once
n ≈ 1/p_d(0.3) = 2.2 · 10²² (p_d(c) = ½ I_{1−c²}((d−1)/2, ½), the exact cap fraction); for cos ≥ 0.5, 3.2 · 10⁶⁵.
Both routes put a random direction's nearest text near zero similarity for any catalogue that exists.

What this suggests, as hypotheses for Study 2 rather than conclusions:

- **GAP targets** (isolated samples from a normalised Gaussian surrogate of the corpus) have a benchmark ceiling of
  0.54, like a mid-level NOISE target; by §6 they would land near 0.5. Normalisation and selection change the
  surrogate's distribution, and nothing here shows that texts exist there; that is what decoding them would test.
- **Domain dimension**: corpora that spread their variance over fewer directions show a random direction more
  variance per direction. Whether that makes gaps easier to fill is a prediction for arXiv and patents; the
  earlier effective-dimension covering numbers were not a valid argument and are withdrawn.

## 9. bge-m3 check

Targets live in the Qwen3 space, so bge-m3 can re-measure only what needs no target: corpus shape, the decoded items'
in-slab fraction and latitude, anchoring to the (Qwen-selected) shown neighbours, κ on REAL (whose target is a real
game present in both spaces) and ceiling growth for directions drawn in bge-m3. It does not replicate the landing
laws for the other arms; that would need targets built and decoded in bge-m3.

bge-m3 is more anisotropic (pairs of games at median cos 0.59, ‖μ‖ = 0.77, k_90 = 179, participation ratio 153).
Decoded items have in-slab fraction 0.951 (per item 0.886–0.983; PLAIN/luna median 0.926) and latitude 0.73–0.80
(corpus 0.95 and 0.77). FAR decoded items sit at cos 0.79–0.81 from their shown neighbours' centroid, while a shown
neighbour sits at 0.72 from the centroid of the other seven: the decoded item is closer to the group than its own
members are to each other. κ on REAL is 0.90–0.98 (Qwen3: 0.92–1.03). This is robustness across two embedding
spaces, not model independence.

## 10. Formal proofs

`lean/LatentGap/Geometry.lean` (Lean 4 + Mathlib, compiled on Kaggle by `kaggle/lean-check`; no `sorry`, only the
standard axioms) proves `inner_split`, `landing_le`, `ladder_norm_sq`, `ladder_inner`, `ladder_cos` (ε = 0),
`sum_norm_sub_mean_sq`, `sum_directional_variance`, `expected_max_le` and `expected_max_le_of_bounds`. The empirical
regularities (κ, the two-moment approximation, the extrapolation) are measurements, not theorems; the cap fraction is
the standard formula, not formalised.

## 11. GAP arm: in-distribution gaps (Study 2, pre-registered)

Targets: the 50 most isolated of 20k normalised draws from the corpus's own Gaussian N(μ, Σ) (`targets.gap`), the
same in every run. Decoded by six generators (the five subscription ones in the grid plus gateway gpt-6-luna in
the pilot run); judged by the same four judges, pitches written by gpt-6-luna through Codex (`--pitch-codex`). Predictions were
written in design.md before any decoding.

| Prediction | Observed | Verdict |
|---|---|---|
| ceiling ≈ 0.54 | 0.541 [0.516, 0.547] | holds |
| landing ≈ κ · ceiling | landing / ceiling 0.89 (luna) – 1.02 (opus), landing 0.48–0.55 | holds |
| decoded closer to the target than to the shown neighbours' centroid | 0 of 300 items; cos to centroid 0.72–0.81 | fails |
| judged novelty between INTERP and PLAIN | 4.91 vs INTERP 5.52, PLAIN 7.77; below INTERP for all six generators (bootstrap CI excludes 0 for gateway luna, luna, sol61, sonnet) | fails |

**Gateway-luna pilot (added 2026-10-26; 50/50 items, 0.31 USD).** Same targets, so ceiling 0.541; landing 0.485,
landing / ceiling 0.897; 0 of 50 closer to the target than to the centroid (cos 0.740); novelty GAP 5.50 vs INTERP
6.30, PLAIN 7.67, GAP − INTERP bootstrap 95% CI [−1.29, −0.33]. It behaves like the Codex-luna grid run (ratio
0.887, novelty 5.46) and changes none of the verdicts. Its PLAIN items are the collapsed gateway-luna ones of the
batch 2 caveat below, so its PLAIN mean is inflated in the same way.

The decoders reach in-distribution gaps as well as the best existing game does, but they get there by writing a
synthesis of the eight shown neighbours (bounds.md §3), and the judges rate the result less novel than a midpoint
of two distant games. This is the first of the two outcomes written in design.md: in-distribution gaps are
reachable but not new. Coherence stays high (7.22, between CROWD and FAR), so the arm does not fail on sense.

**Human check (batch 2, 16 blind luna items: 6 GAP, 5 INTERP, 5 PLAIN; artifact SbxFeHmvDtAWX5d1snxiD7, ratings in
runs/steam/human2).** Mean novelty, human vs four judges: INTERP 7.0 vs 6.6, GAP 6.2 vs 5.8, PLAIN 5.2 vs 7.6. The
human agrees that GAP is less novel than INTERP; PLAIN is where they part: rated one after another, the luna PLAIN
items repeat ("another museum of things that never happened"), which the human penalised and per-item judges cannot
see (design.md, PLAIN collapse). Item-level ρ with the judges' mean: novelty −0.32, appeal −0.30, coherence +0.15,
feasibility +0.12 (n = 16, descriptive).

## 12. arXiv (cs.*): what replicates and what does not

Same pipeline on arXiv cs.* (title + abstract, past = first version < 2023-01-01, n = 72,000 visible rows), same
arms and targets construction, two generators (gpt-6-luna via Codex, Sonnet 5.5 via Claude Code), same four
judges, pitches without results. Sources: `runs/arxiv/geometry.log` (`scripts/geometry.py --domain arxiv`, re-run
2026-10-26, identical to the 2026-10-24 log) and `geometry_items.csv`; `docs/prereg/conjecture_battery_results.csv`;
judge means, diversity, isolation and bootstrap CIs (10k resamples over items) computed 2026-10-26 from
`runs/<domain>/grid-*/items.jsonl` and `judged_{gpt61sol,opus,sonnet,haiku}.jsonl` (throwaway script, not
committed). One arXiv item (Sonnet PAIR-40) has 3 judges instead of 4. "Steam" values are from §1–§11 or the same
computation on the five Steam grid runs.

### Laws, Steam vs arXiv

| Law / quantity | Steam (5 generators) | arXiv (luna, Sonnet) | Replicates? |
|---|---|---|---|
| corpus: median pair cos, ‖μ‖, k_90, participation ratio | 0.41, 0.64, 232, 112 | 0.29, 0.54, 281, 117 | – (arXiv wider, not lower-dimensional) |
| decoded in-slab fraction m(e) (real items) | 0.957 (0.95) | 0.955 (0.951) | yes |
| FAR ceiling: observed vs parameter-free formula (bounds.md §4) | 0.046 vs 0.044 (retrospective) | 0.058 vs **0.062 pre-registered** (error 0.004, strict ≤ 0.005) | yes |
| FAR ceiling, two-moment approximation (§7) | 0.048, error 0.005 | 0.064, error 0.007 | yes |
| random-direction growth slope vs √(tr Σ / d) (§8) | 0.027 vs 0.024 | 0.0266 vs 0.0262 | yes |
| FAR fitted curve reaches cos 0.3 at | n ~ 10⁴⁴ | n ~ 10³⁸ | yes (conditional extrapolation) |
| κ on REAL | 0.92–1.03 (luna-Codex 0.92, Sonnet 1.02) | 1.02, 1.09 | roughly; Sonnet above 1 |
| κ·ceiling error: NOISE / CROWD / FAR | 0.017–0.033 / 0.040–0.057 / 0.018 | 0.027 / 0.056 / 0.019 (mean of 2) | yes |
| ladder-law error: NOISE / CROWD | 0.034–0.058 / 0.077–0.213 | 0.052 / 0.107 | yes (κ·ceiling still better) |
| FAR landing; items above the ceiling | 0.026–0.046; luna 10 %, Sonnet 44 % | 0.038, 0.073; 22 %, 78 % | yes, landing ≈ 0 |
| anchoring: FAR decoded vs shown centroid (centroid vs target) | 0.63–0.72 (0.06) | 0.67, 0.78 (0.08) | yes |
| decoded closer to the centroid than the target is: FAR / NOISE / CROWD | 100 / 98 / 96 % | 100 / 98 / 100 % | yes |
| FAR residual alignment with Σt (best corpus item) | 0.31–0.38 (0.27) | 0.27, 0.35 (0.24) | yes |
| GAP ceiling; landing / ceiling | 0.541; 0.89–1.02 | 0.433; 1.04, 1.14 | yes |
| GAP decoded closer to the target than to the shown centroid | 0 / 250 | 0 / 100 | yes (prediction fails again) |
| GAP − INTERP novelty | below INTERP for all 6 (4.91 vs 5.52) | luna −0.13 [−0.55, +0.30], Sonnet +0.30 [−0.21, +0.81] | **no**: GAP ≈ INTERP |
| PLAIND − INTERP novelty | +1.16 (luna) to +3.19 (Sonnet), all CIs exclude 0 | luna +1.07 [+0.71, +1.45], Sonnet +3.01 [+2.54, +3.48] | yes |
| most isolated decoded arm (1 − max cos to past) | PLAIN | PLAIN 0.351 / 0.343 (FAR 0.305 / 0.291) | yes |
| PLAIN collapse: median pair cos (50 random past items) | luna-Codex 0.811, Sonnet 0.864 (0.406) | luna 0.524, Sonnet 0.759 (0.283) | yes, weaker for luna |
| PLAIN modal title | 46/50 "Museum of…", 45/50 "Tidewright…" | 16/50 "Proof-Carrying Silence", 43/50 "Tidal…" | yes |
| PLAIND pair cos (INTERP) | luna 0.534, Sonnet 0.504 (0.428, 0.444) | 0.455, 0.415 (0.345, 0.334) | yes: back near guided-arm diversity |

Random-pair baselines are the mean over 20 draws of 50 visible past items (Steam 0.406, range 0.389–0.424; arXiv
0.283, 0.269–0.297). The pre-registered arXiv anticipation that landing ≈ anchoring × centroid score was not
evaluated here.

### Judges per arm (arXiv, mean of the 4 judges, pitches without results)

| arm | novelty luna / Sonnet | coherence luna / Sonnet | appeal luna / Sonnet | Steam novelty / coherence (5-gen mean) |
|---|---|---|---|---|
| FUTURE_REAL | 5.96 / 5.94 | 7.36 / 7.36 | 6.50 / 6.46 | 4.59 / 7.04 |
| REAL | 4.54 / 2.98 | 7.39 / 6.59 | 6.10 / 5.52 | 3.07 / 7.82 |
| INTERP | 6.36 / 4.28 | 5.92 / 4.53 | 4.99 / 4.33 | 5.37 / 6.70 |
| NOISE | 5.16 / 3.39 | 7.16 / 6.19 | 6.19 / 5.32 | 3.61 / 7.71 |
| CROWD | 5.15 / 3.78 | 6.98 / 5.99 | 6.02 / 5.40 | 4.18 / 7.42 |
| GAP | 6.24 / 4.58 | 6.45 / 4.80 | 5.86 / 4.97 | 4.79 / 7.22 |
| FAR | 6.32 / 4.92 | 6.80 / 4.38 | 5.98 / 4.40 | 5.75 / 7.27 |
| PAIR | 6.44 / 6.09 | 5.98 / 6.24 | 5.08 / 5.18 | 6.61 / 7.27 |
| PLAIN | 7.50 / 7.02 | 7.01 / 7.24 | 6.46 / 5.54 | 7.79 / 7.86 |
| PLAIND | 7.44 / 7.30 | 7.16 / 7.00 | 6.30 / 6.69 | 7.80 / 7.99 |

Two differences from Steam. (1) Real future papers are judged more novel than every guided arm (5.94 vs ≤ 5.62 as
a two-generator mean, though luna's INTERP, FAR and GAP, 6.24–6.36, exceed it; on Steam future games score 4.59, below INTERP, GAP and FAR). (2) Guided arms lose coherence
on arXiv, Sonnet's most (INTERP / GAP / FAR 4.38–4.80, Steam 6.4–7.2 for the same generator): a synthesis of eight
distant abstracts reads as an overloaded paper more easily than a mixed game does. PLAIN and PLAIND keep coherence
≈ 7 in both domains, so the PLAIN-family lead does not come at a cost in sense.

### Results-leak fix (design.md, "Results in paper pitches")

Old pitches kept reported results (FUTURE_REAL 18–20 % of pitches, Sonnet PLAIN 52 %, invented). Re-pitched without
results and re-judged (old judgements in `legacy/with-results/`). Effect, mean of the 2 generators, old → new:
FUTURE_REAL novelty 6.16 → 5.94, appeal 6.66 → 6.48; PLAIN appeal 6.24 → 6.00 (Sonnet 5.98 → 5.54, luna 6.50 →
6.46); every other arm and criterion moves < 0.12. The novelty order is unchanged except at the top, where PLAIN
(7.36 → 7.26) and PLAIND (7.34 → 7.37) swap, a difference of 0.02–0.11, within noise. All numbers above use the new
judgements.

### Humans, anticipation and memory

- **Human batch A2** (non-expert owner, 16 blind luna items): summarised below under "Human arXiv batch A2"; the
  human puts real recent papers on a par with PLAIND, where the judges put them 2.5 points lower. Batch A1 is void
  (pitches with results).
- **Memory (MEM)**: no item-level memory step at any generator's cutoff on arXiv (gpt-4o step 0.030 vs unseen-window
  noise 0.057). **Style**: gpt-4o's rewording alone moves real past abstracts to the future (hit 0.224 → 0.363);
  luna's and Sonnet's do not (0.237, 0.217). **Style-free anticipation** (one normaliser for everything): PLAIND
  Sonnet 0.58 and luna 0.50 vs real past 0.28 (Fisher p 0.002, 0.020), real future 0.62; gpt-4o 0.38 (p 0.20); FAR
  not distinguishable from the baseline for any generator. Read as topic-level knowledge of what came later, not
  foresight. Details and n: docs/prereg/memory_probe.md. arXiv is the domain where this test works: Steam loses the
  past/future separation after normalising.

### Status: closed

Done on arXiv: corpus and targets; seven decoded arms, PLAIN, PAIR and PLAIND for luna and Sonnet; four judges on
result-free pitches; the geometry laws (table above, FAR ceiling pre-registered and within the strict criterion);
the falsification battery (bounds.md §4b, 8/8 Qwen-arXiv cells within 0.02); support search (bounds2.md: token
search reaches FAR 0.185 only through token soup, coherent candidates stay at 0.058–0.059); MEM, style and
style-free anticipation; non-expert human batch A2.

Left open, explicitly:

- expert raters for arXiv (pending); the only human reading is non-expert, n = 16;
- 2 generators on arXiv vs 5 on Steam, and no gateway-luna arXiv run, so generator-level claims (κ > 1 for Sonnet,
  luna's weaker PLAIN collapse) rest on one model each;
- the anchoring × centroid prediction and a causal test of anchoring (randomised or removed neighbours, bounds.md
  §4) are not run;
- GAP vs INTERP novelty is the one Steam result that does not replicate (equal on arXiv, not lower); with two
  generators it is not resolved which domain is typical.

## Consequence for the tweet

The post's recipe asks for a text at a random far direction. In this space the best existing game reaches cos 0.05
there, the generators land at that level, and what they write drifts towards the examples they were shown. The guided
version where texts plausibly exist and the corpus is thin (GAP, §11) lands as well as the best game does but is
judged less novel than a midpoint of two games (on arXiv, §12, no more novel than the midpoint); PLAIN reaches the
most isolated and most novel items without any target.

### Human batch 3 (2026-10-25): PLAIND vs INTERP, gpt-6-luna, 8 + 8 blind items, none reused from batch 2

| arm | human novelty | judges' novelty | human coherence | human appeal |
|---|---|---|---|---|
| PLAIND | 7.00 | 7.25 | 7.12 | 6.38 |
| INTERP | 5.75 | 5.66 | 6.38 | 6.12 |

Mann-Whitney PLAIND vs INTERP on human novelty p = 0.105 (n = 8 + 8: direction, not proof). Item-level Spearman with
the judges' mean: novelty 0.36, coherence 0.37, appeal -0.23. Across batches the human's PLAIN-family novelty goes
from 5.2 (batch 2, collapsed PLAIN, "otro museo... cansino") to 7.0 once repetition is removed, while INTERP goes 7.0
-> 5.75 on a fresh sample: the batch-2 penalty on PLAIN was the repetition, and the judges' ranking (PLAIN-family >
INTERP) is what the human gives when the ideas do not repeat. Two of the eight PLAIND items were still museums (drawn
by lot) and scored 7 and 8. Data: runs/steam/human3/.

### Human arXiv batch A2 (2026-10-26): luna, pitches without results; 16 blind items

Owner's own ratings (not a domain expert; 6 items carried over from the void batch A1, two of them rated on a pitch
that still had results: FUTURE_REAL-3, GAP-15). Means per arm, human vs the 4 judges:

| arm | n | human novelty | judges' novelty | human coherence | judges' coherence |
|---|---|---|---|---|---|
| FUTURE_REAL (real, 2023+) | 4 | 6.75 | 5.06 | 6.50 | 7.06 |
| PLAIND | 5 | 6.60 | 7.55 | 8.20 | 7.25 |
| FAR | 4 | 6.00 | 6.06 | 8.75 | 7.38 |
| GAP | 3 | 5.67 | 5.83 | 6.33 | 6.75 |

Item-level Spearman with the judges: novelty 0.34, coherence 0.53, appeal 0.43, feasibility 0.01. The human's novelty
range is narrow (5-8 except one 3) and puts real recent papers on a par with PLAIND, where the judges put them 2.5
points lower. Recognition (asked after the fact): item 9 (FUTURE_REAL-46, real) "sounded familiar"; item 16 (FAR-30,
generated) "felt familiar, could not say from what" and got the lowest novelty, 3. Without FUTURE_REAL-3 (leak) the
real items average 6.33. n = 16: a non-expert reading, not a verdict. Data: runs/arxiv/humanA2/.
