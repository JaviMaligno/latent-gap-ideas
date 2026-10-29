# Round 5, track C2: a covered language and a bounded directional tail

2026-10-28. Continuation of [bounds5 §7](bounds5_support.md#7-conjecture-the-useful-next-support-model-is-a-covered-language-not-a-fitted-n)
and [bounds4](bounds4_neff.md). All new encoder measurements use
`src/latent_gap/embedder.py:LocalQwen(threads=4)`, locally, without LLM or judge
calls. Existing verified local embeddings supply the saved rewrite witnesses.

**Result: the specified cell-specific uniform residual model is refuted.** Four of 384
independent grammar checks exceed their cell radii; the separately tested
free-rewrite radius fails on 23/187 coherent witnesses. A smaller directional
cone envelope is exceeded by five Steam FAR witnesses, including **two
Opus-accepted witnesses**. The conditional support theorem and the calibrated
one-draw probability theorem are valid, but the tested hypotheses do not give
a support certificate for coherent text. The new bounded beta tail has valid
cosine support and still fails out-of-set transfer: only arXiv FAR passes the
main future-corpus criterion, and none of the twelve search cells passes.

**Disclosure.** The test results are already known. The task, previous reports,
and examples in those reports revealed both aggregate and individual results.
Future-corpus extrema also appear in the existing reference files. This is
an out-of-set reuse of data, **not a prospective confirmation**. The code fits
only generated corpus compositions, past corpus rows and
`seed-rewrite/tune-v2/REAL-11..20`; its measurement phase opens the test
outcomes and future projection arrays only after a hash freeze. Encoding a
held-out composition before that freeze does not put its residual into the fit.
Fresh target-blind random directions provide an additional future-corpus check.
One existing Opus record was also inspected for its schema before the freeze;
its rating does not enter any fit. No claim of blindness to test outcomes is made.

## 1. A concrete language, its cover, and exactly what is conditional

Fix the encoder, tokenizer, last-token pooling, EOS, and L2 normalization from
round 4. Treat the mathematical encoder as unit-valued. The floating-point
experiment is an audit of that model, not an interval proof of ONNX arithmetic.
No newly generated text is asserted to pass a coherence judge.

### 1.1 Real source bank and operations

For each domain, the source bank \(A_D\) consists of all non-adult **past**
corpus rows satisfying the following deterministic predicate. Remove the union
of all `refs.jsonl` hidden REAL source positions, including calibration targets;
this makes one bank per domain, independent of which target is being bounded.
Use the exact stored text, never a cleaned or shortened version.

* Steam has its `Genres: … . Tags: … . About: …` fields, at most seven genres
  and twenty tags. arXiv has `Title: … . Abstract: …`, at most eighteen title
  words.
* The body has 70–170 whitespace-separated words and at most 1,200 characters;
  the entire text has at most 511 content tokens. No encoder truncation occurs.
* Split sentences with `(?<=[.!?])\s+`. Require at least three sentences and
  require joining them with one space to reproduce the original body exactly.
  This is a deterministic syntactic splitter, not a claim about linguistic
  sentence boundaries.
* The middle sentence has an eligible term: an ASCII alphabetic/hyphen token
  of length at least six, excluding the fixed stop list in the script. Select
  the longest term, breaking ties lexicographically. These are **terms**, not
  entities identified by a semantic classifier.

The recorded banks contain **26,193 Steam / 45,944 arXiv** sources. They are
listed by original past-corpus position and text hash in
[cover_bank.csv](prereg/cover_bank.csv).

For distinct ordered sources \((a,b)\), apply a subset of the following operations
in the fixed order F, S, T:

| Operation | Exact action |
|---|---|
| F: fields | Replace the entire header of \(a\) with that of \(b\): genres and tags together for Steam, title for arXiv. |
| S: sentence | Replace the middle sentence of \(a\) with the middle sentence of \(b\). |
| T: term | Replace at most the first two whole-word occurrences of the selected middle-sentence term with the donor's selected term. With S, use the original source's term as the replacement, so ST is not automatically a no-op. |

Reject no-ops, missing/equal substitution terms, and outputs failing the same
format/length predicate. Do not recursively use generated texts as sources.
Every output retains the source's first and last sentences verbatim.

Define \(L_D\) to contain **all valid changed outputs** of
F, S, T, FS, FT, ST on every allowed ordered source pair. This is a large,
finite, reproducible language, not just the measured sample. Before rejection
it has at most \(6|A_D|(|A_D|-1)\) derivations. There is no identity operation.
The three-operation FST
language is a separate depth stress test, outside \(L_D\).

Work on derivations if two pairs produce the same text. Taking a maximum over
derivations and then rendering gives the same text support; duplicate derivations
do not change a supremum. This also makes the sampling law unambiguous.
For an independently fixed coherence predicate \(J_D\), the object bounded is
\(C_D\cap L_D\). Original corpus provenance and these operations do not prove
\(J_D(x)=1\). The language intentionally imposes stronger restrictions than
free paraphrase.

### 1.2 Finite cover and cell residual model

There is one input cell for each admissible derivation \((o,a,b)\).
Its predicted unit anchor is

\[
 q_{oab}=\frac{(1-\lambda_o)p_a+\lambda_o p_b}
                   {\|(1-\lambda_o)p_a+\lambda_o p_b\|},\qquad
 E(g_o(a,b))=q_{oab}+r_{oab}. \tag{1}
\]

Here \(p_a\) is the prepared corpus vector, renormalized to unit length. It is
a fixed reference anchor; equality to the local encoding \(E(a)\) is **not**
assumed. Select \(\lambda_o\in\{0,.1,\ldots,1\}\)
by minimum mean squared residual on **16 training compositions** per domain
and operation, ties going to the smaller coefficient. The complete input cover is explicit:
every language derivation occurs in one listed type of cell. The missing
encoder hypothesis is also explicit:

\[
 \mathcal H_D:\quad
 \|r_{oab}\|\le\epsilon_{D,o}
 \quad\text{for EVERY admissible pair, for each of the six operations}. \tag{2}
\]

If (2) holds, the balls \(B(q_{oab},\epsilon_{D,o})\) are an embedding cover.
Measuring a sample maximum does **not** prove
(2). Merely enumerating the input grammar does not certify its encoder image.

### 1.3 THEOREM: the conditional support certificate

Let \(K_D=\operatorname{cone}\{p_a:a\in A_D\}\). A positive mean-axis latitude
\(u^Tp_a\ge a_0>0\) ensures every mixture in (1) has norm at least \(a_0\).
Its normalization lies in \(K_D\cap S^{d-1}\), because both weights are
nonnegative. For each operation choose any polar vector
\(z_o\in K_D^\circ\), i.e. \(p_a^Tz_o\le0\) for every source.

For unit target \(t\), nonempty coherent restricted language, and hypothesis (2),

\[
 \boxed{S_{C_D\cap L_D}(t)
 \le \min\left(1,\max_o[\|t-z_o\|+\epsilon_{D,o}]\right).} \tag{3}
\]

A simpler common-polar form, used in the forecast CSV, is

\[
 U_D(t)=\min\{1,B_{K_D}(t)+\max_o\epsilon_{D,o}\},\qquad
 B_{K_D}(t)=\|t-z\|. \tag{4}
\]

**Proof.** The convex pair has positive norm by its latitude. For its unit
normalization \(q\), polarity gives \(q^Tz\le0\). Therefore
\(q^Tt=q^T(t-z)+q^Tz\le\|t-z\|\). Cauchy–Schwarz gives
\(r^Tt\le\|r\|\le\epsilon_o\). Add the inequalities, maximize over
**every** cell, and intersect with the unit-sphere bound 1. Restricting a
pointwise bound to the fixed coherence gate preserves it. No assumption on
target randomness or independence from the corpus is needed.

The cone is a computational relaxation of the pair anchors: it permits more
mixtures than the grammar, but this only enlarges the upper bound. FAR uses
the round-4 column-generation NNLS dual, checks every bank constraint and adds
the same numerical polar-feasibility correction. Other arms use its looser
axis-ratio dual. Optimizer convergence is not needed for an upper certificate.
The duals, active positions, weights and gaps are saved in
[cover_cones.csv](prereg/cover_cones.csv); positions there index the selected
bank for `grammar`, the whole past corpus for `extension`.

Lean proves the pair-polar, nonzero-norm, per-cell, covered-language, and
coherence-restriction steps in `LatentGap.Cover`. It also proves that a score
violation forces failure of its claimed residual; the converse is false.

## 2. Calibration: a probability result, not a support result

### 2.1 Frozen sampling law and independent sets

For each domain/operation, independently draw an ordered source pair uniformly
with replacement from \(A_D^2\), reject equal sources and invalid/unchanged
outputs, and repeat until accepted. Thus the law is uniform on **valid
derivations**, not uniform on distinct texts. Its fixed RNG streams and exact
draws appear in [cover_plan.csv](prereg/cover_plan.csv). Rejection depends only
on the specified syntax and length checks, never on target score, residual or
coherence ratings. Separate streams supply 16 training, 64 calibration and
32 check draws per cell type. No forced disjointness changes the sampling law;
accidental reuse of a source or output is permitted by independent draws.

Fit \(\lambda_o\) on training only. Then set
\(\widehat\epsilon_o=\max_{j\le64}\|r_j\|\) from the independent calibration
set. Test its residual coverage on the 32 fresh check draws. FST has 32 extra
draws per domain and uses the frozen FS predictor and radius, with **no**
calibration guarantee for this changed law. There are 1,408 generated
compositions and four original-source encoder parity checks altogether.

### 2.2 THEOREM: what the sample maximum guarantees

Conditional on the training set, let calibration residuals \(R_1,\ldots,R_m\)
and a new residual \(R\) be iid from this specified law, and let
\(\widehat\epsilon=\max_jR_j\). For strict exceedance, including distributions
with atoms,

\[
 \Pr\{R>\widehat\epsilon\}\le\frac1{m+1},\qquad
 \Pr_{\rm cal}\{\Pr(R>\widehat\epsilon\mid\rm cal)>\alpha\}
       \le(1-\alpha)^m. \tag{5}
\]

The first follows by exchangeability: the new draw is strictly largest at
most one of \(m+1\) exchangeable positions. For the second, if the residual
mass above the chosen maximum exceeds \(\alpha\), all calibration draws lie
in an initial segment with mass at most \(1-\alpha\); use the generalized
quantile to handle atoms. This is the distribution-free one-sided tolerance
bound, conditional on the fitted predictor, not a guarantee on the support.

For twelve domain/operation cells, set
\(\alpha=1-(.05/12)^{1/64}\) and union-bound their calibration failures. With
95% probability over calibration, every cell's true failure rate is at most
this \(\alpha\). A fixed mixture of the six cell laws inherits the same bound.
The marginal one-new-draw error is
\(1/65\). Equations (3)–(4) hold on the residual-success event for **all targets
simultaneously**, even if a target was selected using the corpus.

These are weak guarantees for a large adaptive search. For \(B\) draws from
the same fixed law, a union bound gives at most \(B\alpha\) (clipped at 1)
on the simultaneous calibration event, or \(B/65\) marginally. An arbitrary
target-guided rewrite distribution is not that law. Conditioning on coherence
also changes the law: if the original law's acceptance probability is \(q>0\),
the failure probability after filtering is at most \(\alpha/q\), not
automatically \(\alpha\). No new judge is run to estimate \(q\).

To turn (5) into a support claim using a minimum atom mass \(\pi_{\min}\),
one would need \(\alpha<\pi_{\min}\), or an equivalent exhaustive/regularity
certificate. Uniform valid-pair sampling has extremely small atom masses in
these billion-derivation languages. Sixty-four calibration draws cannot do
this. This is precisely the quantifier gap identified in round 4.

### 2.3 Separately falsifiable transfer to free rewrites

Membership in \(L_D\) is tested on the exact rendered saved winner. First
index source rows by their unchanged first/last sentence pair. For any match,
enumerate all allowed donors and operations and compare the complete text.
Failing the boundary-sentence condition is already a proof of nonmembership.
This is a syntactic coverage test, independent of the embedding score.

A distinct extension hypothesis, designed to allow arbitrary wording, uses
\(q_8(x)\), the normalized mean of the eight nearest visible corpus vectors
to \(E(x)\). This is a deterministic reconstruction model, not a predictor
available before encoding \(x\). It always lies in the full visible corpus
cone. Set the domain radius to the larger of all six grammar calibration
radii and all measured \(q_8\) residuals from tune-v2 REAL 11–20, pooling both
writers. This is a deliberately conservative empirical transfer model:

\[
 \mathcal H_{\rm rewrite}:\quad
 \|E(x)-q_8(x)\|\le\widehat\epsilon_{\rm ext}
 \quad\text{for all relevant coherent rewrites}. \tag{6}
\]

Given (6), the full-corpus cone plus this radius is a valid conditional upper
bound, with REAL's hidden rows removed as in the references. The nearest-eight
rule and radius freeze before test. Tuning rewrites are adaptive and correlated;
there is **no** iid tolerance guarantee for (6). A test residual exceeding this
radius refutes this particular reconstruction hypothesis even if the score is
below its loose upper bound. It does not refute an existential tube around the
entire cone, since a different mixture could approximate the witness better.

### 2.4 A smaller directional envelope, separately hypothesized

Before any residuals were fitted or checked, the protocol was supplemented
with a directional candidate (timestamped as such in `cover_protocol.csv`).
For each fixed test direction, let \(d_D(t)\) be zero or the largest
\(r^Tt\) among all six operations' calibration residuals, whichever is larger.
Let \(d_{\rm ext}(t)\) enlarge it by the maximum projection of the tuning
REAL rewrite residuals \(E(x)-q_8(x)\). These vectors are never test rewrites.

**THEOREM, conditional directional version.** If every relevant text's
residual satisfies \(r(x)^Tt\le d(t)\), then
\(E(x)^Tt\le\min(1,B_K(t)+d(t))\). The proof is the same polar inequality
plus the stated scalar residual bound; `directional_cell_certificate`
formalizes it. The global support statement requires this hypothesis for
every text, not only the observed residual vectors. It does not follow from
the norm-calibration result, and it is not a law for arbitrary new directions.

For a fixed direction and one new grammar draw, pooling maxima only enlarges
the relevant cell's independent calibration maximum, so the marginal
\(1/65\) guarantee still applies. Simultaneous 95% PAC coverage over all
12 cells and 50 fixed test directions per domain requires the larger
\(\alpha=1-(.05/600)^{1/64}\). The extension with adaptive rewrites again has
no such sampling guarantee. Both target-score violations and actual
directional residual violations are checked, without refitting the envelope.

## 3. A new bounded-support directional tail

### 3.1 EMPIRICAL LAW candidate: a beta exceedance tail with endpoint 1

For each fixed direction \(t\), fit only its visible past-corpus projections
\(s_i=p_i^Tt\), removing its exact REAL hidden positions. Put
\(u_t=\widehat q_{.95}\), \(p_u=\#\{s_i>u_t\}/n\). For exceedances,
\(Z=(s-u_t)/(1-u_t)\in(0,1)\), fit a two-shape beta distribution
\(\operatorname{Beta}(a_t,b_t)\) by maximum likelihood with location 0 and scale 1
**fixed**. There is no endpoint fit or later clipping of an unbounded law.

\[
 S_t(h)=p_u\,\overline I_{(h-u_t)/(1-u_t)}(a_t,b_t),\quad u_t\le h<1;
 \qquad S_t(h)=0\quad(h\ge1). \tag{7}
\]

Below the threshold, join the empirical body's 101 evenly spaced quantile
knots by monotone interpolation, scaled to total mass \(1-p_u\). Extend the
CDF by 0 below its minimum and 1 at/above 1. This gives a proper CDF on
\([-1,1]\); shape positivity and normalized beta mass ensure the tail is
monotone and joins continuously at \(u_t\). Repeated body knots use the
generalized-quantile convention. All extreme calculations occur above \(u_t\).

Unlike the previous GPD, no score below 1 contradicts this tail's support.
That is an algebraic property of the proposed law, **not** evidence for its
accuracy or for reachability of its endpoint by coherent text. Both shapes
retain direction-specific curvature. The law is falsifiable by frequencies
and maxima on rows not used to fit it.
It models a hypothetical population of new corpus items. It is not the law
of resampling the finite observed catalogue, whose maximum cannot exceed its
known ceiling.

### 3.2 Frozen out-of-set checks and search extrapolation

Use all non-adult future corpus rows, 40,989 Steam / 41,000 arXiv. For the
existing REAL/GAP/FAR directions and 24 additional uniform unit directions
per domain (fixed seed, no score-based selection), freeze (7) before projecting
future rows. Report future exceedance counts at the fitted one-row q99 and
q999, their observed/expected ratios, and future maximum predictions at the
actual future sample size. REAL 11–20 directions are reported separately from
REAL 1–10. Original FAR directions were selected against the past corpus, so
past maxima are descriptive; future rows are an out-of-set check. Temporal
shift and dependence between rows can also invalidate iid maximum predictions.

The frozen practical acceptance criterion for each domain/arm is: median
q99 and q999 ratios in [.5,2], absolute paired median maximum bias at most
.03 for REAL/GAP or .01 for FAR/RANDOM, and at least 80% coverage by nominal
95% maximum intervals. Directions share rows, so this is an operational
calibration criterion, not a collection of independent hypothesis tests.

**THEOREM, conditional iid maximum.** If \(N\) projections are iid from
\(F_t\), their maximum CDF is \(F_t(h)^N\), for any CDF. At the continuous,
strictly increasing tail quantiles used here, with \(0<F_t(h)<1\),

\[
 h_{p,t}(N)=F_t^{-1}(p^{1/N}),\qquad
 N_t(h)=\frac{\log(1/2)}{\log F_t(h)}. \tag{8}
\]

Fit the median of \(\log_{10}N_t(h)\) only on measured tune-v2 REAL 11–20,
separately by domain and writer. Then freeze medians and 95% plug-in maximum
intervals for all 200 test records. These intervals condition on the fitted
size; they do not include parameter-estimation uncertainty. A domain/writer
model must pass all three arms: at least 90% measured winners, absolute paired
median score error at most .03, and at least 80% maximum-interval coverage.
Also compare predicted/observed excess over the original ceiling and escape
frequency at `ceiling_star + .05`. No test-arm size is fitted.

If the tail fails future calibration, it cannot be used to attribute a search
failure specifically to the iid search-size assumption. If it passes, that
still validates only the checked corpus range, not extrapolation to enormous
REAL-equivalent sizes or the replacement of corpus draws by coherent rewrites.

## 4. Measurements and verdicts

The 200 forecasts froze at **2026-10-28 18:35:56 UTC**. Prediction SHA-256:
`f44bd3a765147c7f17da49cc51e286fa2a8fabfc424ce01caf7e1b28fe1010cc`.
All subsequent checks use those forecasts without changing any parameter.

### 4.1 Reference-vector audit and fitted cell models

The four fresh local encodings differ from their prepared reference vectors
by Euclidean distances **.00201683, .00187835** (Steam) and
**.00191121, .00187992** (arXiv). Thus the prepared vectors cannot silently be
identified with the exact local outputs. The first fit stopped at the parity
guard before writing fitted artifacts. A timestamped pre-freeze amendment
removed the initially proposed zero-residual identity cell. The six operation
models, draws and decision criteria stayed unchanged. Some Steam training and
calibration residuals had been computed internally before the stop; no
held-out residual or test outcome had been measured. This adjustment was
triggered by encoder parity, not by a test failure.

All remaining new outputs are embedded locally; their residuals are measured
against the explicitly defined prepared reference anchors. The support theorem
only requires those anchors to be unit vectors, so it needs no equality to
\(E(\text{source})\). The tail experiment likewise fits/checks the supplied
prepared past/future arrays; comparison to local rewrite embeddings is an
empirical transfer to that reference-vector population.

| Domain | Operation | Fitted λ | Calibration maximum ε | Training maximum |
|---|---|---:|---:|---:|
| Steam | F | .2 | .570277 | .463121 |
| Steam | S | .1 | .556776 | .297324 |
| Steam | T | 0 | .214066 | .095606 |
| Steam | FS | .2 | .715476 | .540835 |
| Steam | FT | .2 | .736558 | .700878 |
| Steam | ST | 0 | .608226 | .367451 |
| arXiv | F | .3 | .543734 | .414064 |
| arXiv | S | .1 | .522458 | .343904 |
| arXiv | T | 0 | .182099 | .125132 |
| arXiv | FS | .4 | .592846 | .552283 |
| arXiv | FT | .3 | .519916 | **.576525** |
| arXiv | ST | .1 | .485882 | .324409 |

Already the arXiv FT training maximum exceeds its independently calibrated
radius by **.056610**. A calibration maximum need not cover the training set.
This alone refutes that radius as a uniform bound for the assigned cell model;
it does not contradict the one-new-draw probability theorem.
The common grammar radii in (4) are **.736558 / .592846**. The free-rewrite
extension radii, after including the 39 measured tuning winners, are
**.736558 / .828220**.

For the calibrated grammar law, the simultaneous 95% PAC failure bound is
**8.2071% per cell**; its marginal one-new-draw bound is **1.5385%**. These are
probability bounds with the stated iid law, not numerical encoder-support
certificates. The unseen residual checks below determine whether the stronger
uniform model actually survives.

### 4.2 Independent grammar checks: probability coverage survives, uniformity does not

Every entry below has 32 independent check draws. These are syntactically in
the registered language but have **no new coherence ratings**.

| Domain | Op | Maximum check residual | Frozen ε | Violations |
|---|---|---:|---:|---:|
| Steam | F | .567269 | .570277 | 0/32 |
| Steam | S | .391577 | .556776 | 0/32 |
| Steam | T | .215228 | .214066 | **1/32** |
| Steam | FS | .537937 | .715476 | 0/32 |
| Steam | FT | .490283 | .736558 | 0/32 |
| Steam | ST | .620031 | .608226 | **1/32** |
| arXiv | F | .508771 | .543734 | 0/32 |
| arXiv | S | .473480 | .522458 | 0/32 |
| arXiv | T | .321117 | .182099 | **2/32** |
| arXiv | FS | .550826 | .592846 | 0/32 |
| arXiv | FT | .484732 | .519916 | 0/32 |
| arXiv | ST | .434890 | .485882 | 0/32 |

The largest counterexample is arXiv T, check draw 0: source past position
**59914**, donor **40219**, λ=0; residual **.321117** against **.182099**,
an excess of **.139018**. Its full generated text and its 1,024-dimensional
local embedding are linked by SHA-256
`254d3d5e222088a0fd35d7885a3b46b2d872cc3f1128f70d041ea8efc8d0dd7b`
in the plan/embedding/check CSVs. The other check failures are arXiv T draw 24,
Steam ST draw 7 and Steam T draw 23. They refute (2) for these **assigned
derivations**; they do not prove the text is outside the union of every other
possible anchor ball, or outside an alternative cover.

The **coarsened common radius** \(\max_o\widehat\epsilon_o\) does cover all
384 in-language checks (and all 64 depth checks). That weaker hypothesis is
not refuted by the cell-specific failures. Its FAR envelopes are the loose
.784481/.673069 values reported below; the finite check does not establish
that it covers every unencoded derivation, and syntactic nonmembership still
prevents applying this language theorem to the saved free rewrites.

Overall **4/384 = 1.04%** fail. This is compatible with the marginal one-draw
result (its expected number of failures is at most 384/65 ≈ 5.91); shared
calibration sets mean these are not 384 independent Bernoulli trials. The
probability claim is useful and has not been refuted by observing a few
exceptions. The deterministic claim that there are **no** exceptions is false.
Because the synthetic failures are unjudged, they do not establish that the
same assigned-cell failure occurs inside the unknown coherent sublanguage.

The held-out depth FST has **0/32 failures in each domain**, with maximum
residual .578239 Steam / .582369 arXiv against the FS radii .715476/.592846.
That is a finite stress-test success under a different law, not calibration
or support coverage at arbitrary composition depth.

### 4.3 Existing coherent witnesses: membership and residual transfer

All **187/200 measured judge-passing winners are outside \(L_D\)**. Therefore
the restricted-language theorem cannot be applied to any of them by syntax
alone. This is a direct failure of using this particular grammar as a cover
of the existing free rewrites, not evidence against the witnesses' coherence.

The following compares the separately frozen extension model against every
domain/writer/arm. “Norm fails” tests (6). “Directional residual fails” tests
the actual \((E-q_8)^Tt\), while “directional score fails” compares the witness
with the **full-corpus cone** upper bound plus \(d_{\rm ext}(t)\).

| Domain | Writer | Arm | Measured | Norm fails | Directional residual fails | Directional score fails |
|---|---|---|---:|---:|---:|---:|
| Steam | Sonnet | REAL | 10/10 | 0 | 1 | 0 |
| Steam | Sonnet | GAP | 20/20 | 0 | 3 | 0 |
| Steam | Sonnet | FAR | 20/20 | **5** | 20 | 0 |
| Steam | Luna | REAL | 10/10 | 0 | 0 | 0 |
| Steam | Luna | GAP | 20/20 | **1** | 2 | 0 |
| Steam | Luna | FAR | 20/20 | **15** | 19 | **5** |
| arXiv | Sonnet | REAL | 9/10 | 0 | 2 | 0 |
| arXiv | Sonnet | GAP | 12/20 | 0 | 5 | 0 |
| arXiv | Sonnet | FAR | 18/20 | 0 | 16 | 0 |
| arXiv | Luna | REAL | 10/10 | 0 | 0 | 0 |
| arXiv | Luna | GAP | 19/20 | 0 | 3 | 0 |
| arXiv | Luna | FAR | 19/20 | **2** | 16 | 0 |

Thus the specified nearest-eight norm model fails on **21 Steam / 2 arXiv
records**. Opus filtering leaves **15 Steam / 2 arXiv** such failures. For
example, Steam/Luna FAR-17 has residual **.848178 > .736558**, Opus 8;
arXiv/Luna FAR-4 has **.894098 > .828220**, Opus 7. These are actual coherent
witnesses under both saved gates. They reject the fixed nearest-eight
reconstruction radius, not every possible norm tube around the corpus cone.

All 187 witness **scores** lie below the much looser norm envelopes. Their
FAR medians show why that is insufficient:

| Domain | Max-over-writers h median | Grammar norm envelope | Extension norm envelope | Grammar directional envelope | Extension directional envelope |
|---|---:|---:|---:|---:|---:|
| Steam | .069255 | .784481 | .792656 | .092194 | .107741 |
| arXiv | .056810 | .673069 | .915830 | .120645 | .132705 |

Norm envelopes equal 1 on REAL and GAP. Their lack of score violations cannot
validate a containment hypothesis that fails direct residual checks. The
directional extension is much tighter and is itself refuted by Steam/Luna
FAR **8, 12, 13, 18, 19**. The largest score excess is .027699 at k=12.
Two survive Opus:

| Witness | Opus | Local score | Frozen directional extension upper | Excess |
|---|---:|---:|---:|---:|
| Steam/Luna FAR-13 | 8 | .093236 | .092731 | +.000505 |
| Steam/Luna FAR-19 | 7 | .104821 | .099776 | **+.005045** |

These score violations reject even an **existential** cone anchor with that
small directional remainder, because every unit vector in the cone obeys the
dual upper bound. They do not depend on nearest-eight being the best anchor.
The full-cone duality gap is at most **4.21e−9**, far below either excess;
the grammar-cone gap is at most **6.85e−7**. All checked polar products are
negative. This is a numerical float64 certificate for the reference geometry,
not an interval proof of the encoder implementation.

For max-over-writers FAR, the directional extension has **5/20** original-gate
violations and **2/19 measured** Opus-filtered violations in Steam; arXiv has
**0/20** and **0/19**. Its arXiv score envelope survives this finite check,
while its nearest-eight directional residual hypothesis fails on 19/20
original and 16/19 Opus-filtered combined winners. The zero score count is not
a validation of that hypothesis.

Opus filtering uses only the recorded original winners, choosing the larger
score among the two available accepted writer winners. It does not rerank
unjudged lower-scoring candidates. There are **no Opus REAL ratings**; the
CSV's zero measured REAL rows mean unassessed, not rejected by Opus.

### 4.4 Bounded beta tail: valid endpoint, failed out-of-set calibration

Past-corpus median maxima are descriptive, because these rows fitted the law
and FAR selection used them. They already show a consistent overprediction:

| Domain | Arm | Recorded past ceiling | Beta median maximum at past N |
|---|---|---:|---:|
| Steam | REAL | .7294 | .7426 |
| Steam | GAP | .5331 | .5669 |
| Steam | FAR | .0419 | .0563 |
| arXiv | REAL | .7386 | .7674 |
| arXiv | GAP | .4280 | .4684 |
| arXiv | FAR | .0550 | .0739 |

The primary out-of-set check uses future corpus rows. REAL below means test
directions 1–10; RANDOM comprises the 24 fresh target-blind directions. Ratios
are observed/expected exceedance frequencies at **fixed past-fitted thresholds**.
Bias is median(predicted maximum − observed maximum) paired by direction;
it need not equal the difference of the two displayed medians.

| Domain | Arm | q99 ratio | q999 ratio | Future max median | Predicted median | Paired bias | 95% interval coverage | Criterion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Steam | REAL | .890 | .732 | .7109 | .7344 | +.0204 | 60% | FAIL |
| Steam | GAP | 1.011 | .671 | .5391 | .5602 | +.0203 | 60% | FAIL |
| Steam | FAR | 1.115 | .854 | .0409 | .0520 | +.0103 | 65% | FAIL |
| Steam | RANDOM | 1.065 | .781 | .1003 | .1153 | +.0157 | 45.8% | FAIL |
| arXiv | REAL | .499 | .256 | .6979 | .7535 | +.0518 | 50% | FAIL |
| arXiv | GAP | .723 | .366 | .4239 | .4594 | +.0340 | 40% | FAIL |
| arXiv | FAR | 1.689 | 1.817 | .0631 | .0691 | +.0072 | 90% | **Pass** |
| arXiv | RANDOM | 1.043 | .671 | .1095 | .1239 | +.0159 | 45.8% | FAIL |

Only **one of eight** primary domain/arm cells passes. Separately, future
projections on the REAL tuning directions pass for arXiv (80% maximum-interval
coverage) and fail for Steam (70%). That contrast is another reason not to
validate the model only on REAL tuning directions.

The arXiv REAL/GAP q999 frequency defects occur before search extrapolation.
The random-direction maximum failures also occur without FAR selection.
Temporal shift, within-corpus dependence and unmeasured more-extreme tail
shape can contribute; these checks do not identify which mechanism caused
every failure. They reject the specified past-fit/future-transfer model under
the frozen practical criterion, not the mathematical validity of a beta CDF.

### 4.5 Search-level tail extrapolation and witnesses' excess

The bounded-tail REAL-only fits give \(\log_{10}N\) **7.647 / 7.046** for
Steam Sonnet/Luna and **10.265 / 9.730** for arXiv, from 10/10/9/10 measured
tuning targets. The smaller sizes relative to the previous Gaussian mixture
are a change of surrogate, not a count of searched or reachable texts.

Predicted medians below use the same measured targets as the observed median.
Intervals condition on the point fit and exclude size-estimation uncertainty.

| Domain | Writer | Arm | Measured | Predicted h | Observed h | 95% maximum-interval coverage |
|---|---|---|---:|---:|---:|---:|
| Steam | Sonnet | REAL | 10/10 | .8288 | .8396 | 30% |
| Steam | Luna | REAL | 10/10 | .8127 | .8122 | 20% |
| Steam | Sonnet | GAP | 20/20 | .6468 | .5500 | 10% |
| Steam | Luna | GAP | 20/20 | .6307 | .5359 | 5% |
| Steam | Sonnet | FAR | 20/20 | .1091 | .0529 | 5% |
| Steam | Luna | FAR | 20/20 | .0984 | .0684 | 20% |
| arXiv | Sonnet | REAL | 9/10 | .9322 | .8664 | 11.1% |
| arXiv | Luna | REAL | 10/10 | .9268 | .8498 | 10% |
| arXiv | Sonnet | GAP | 12/20 | .6263 | .4743 | 0% |
| arXiv | Luna | GAP | 19/20 | .6145 | .4343 | 0% |
| arXiv | Sonnet | FAR | 18/20 | .1812 | .0309 | 0% |
| arXiv | Luna | FAR | 19/20 | .1715 | .0505 | 0% |

**All twelve cells fail** the joint search criterion. Steam REAL medians are
close, but their distributions are not calibrated. All 200 forecasts are
within [−1,1] (range .08257–.96473), so the endpoint defect from round 4 is
repaired. Pooled measured-target MAE remains **.10121**. This valid support
alone does not repair the tail/search model.

For FAR, observed versus predicted median excess above the visible ceiling
is **.0122 vs .0692** (Steam Sonnet), **.0305 vs .0582** (Steam Luna),
**−.0237 vs .1285** (arXiv Sonnet), and **−.0022 vs .1186** (arXiv Luna).
The predicted excess and the gate-passing witnesses' excess do not match.

| Domain / writer | Predicted FAR escape at ceiling*+.05 | Original gate observed | Opus-filtered observed |
|---|---:|---:|---:|
| Steam / Sonnet | 89.3% | 5% (1/20) | 5% (1/20) |
| Steam / Luna | 62.9% | 25% (5/20) | 10% (2/20) |
| Steam / either writer | **89.3–95.3%** | **25% (5/20)** | **15% (3/20)** |
| arXiv / either writer | ≈100% | **0%** | **0%** |

The interval for either writer is the mean of per-target union-probability
lower/upper bounds, with **no writer independence assumption**. Opus filtering
leaves the same conclusion. No Opus-specific model was fitted; those entries
are gate sensitivity of the same saved witnesses and forecasts.

**Decision on the tail question:** the bounded beta candidate does not
reproduce REAL/GAP/FAR ceilings and coherent-search excess jointly. Since it
already fails future corpus calibration, this round still cannot separate
tail/temporal/dependence error from failure of iid corpus equivalence for
guided search. The endpoint problem is solved for this candidate; the
calibration and search-transfer problems are not.

**Decision on §7:** a grammar, cover, and precise conditional theorem are now
concrete, and an honest distributional residual guarantee is available. The
specific deterministic per-cell radius proposal is refuted; the broad grammar
coverage of saved coherent witnesses fails; and the small directional
extension is refuted even under Opus. None of these findings establishes a
nontrivial unconditional upper bound for the full coherent set. That bound
remains **1**, with the saved coherent witnesses providing the lower bounds.

## 5. Reproduction, computational limits, and formal scope

```sh
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
uv run --no-sync --no-cache python scripts/bounds/cover.py --plan
uv run --no-sync --no-cache python scripts/bounds/cover.py --embed --limit 8
uv run --no-sync --no-cache python scripts/bounds/cover.py --embed --limit 300
# Repeat until all rows are checkpointed; each invocation is independently bounded.
uv run --no-sync --no-cache python scripts/bounds/cover.py --embed --limit 240
uv run --no-sync --no-cache python scripts/bounds/cover.py --fit
uv run --no-sync --no-cache python scripts/bounds/cover.py --freeze
uv run --no-sync --no-cache python scripts/bounds/cover.py --measure
uv run --no-sync --no-cache python scripts/bounds/cover.py --check
# In lean/, one full build at the end:
lake build
```

This is the execution sequence for a fresh set of artifacts. In the current
checkout, use `--check`; immutable fitted/result files deliberately refuse
overwrite. The scientific sampling plan is independent of chunk size.

No LLM calls, git commits, dependency installation, network downloads, or
source-data writes. The script sets OpenMP/BLAS/Accelerate thread caps to four,
uses one ONNX process at a time, handles one corpus domain at a time, and checkpoints
each new embedding in [cover_embeddings.csv](prereg/cover_embeddings.csv).
Every embedding is tagged with the existing local-encoder fingerprint.
Original-source parity checks compare local ONNX with prepared corpus vectors;
every accepted rewrite is checked against its stored score. An interrupted
chunk resumes by text SHA-256. A partial CSV row fails visibly for inspection.
All non-embedding artifacts refuse overwrite; the freeze hashes code, inputs,
calibration artifacts and forecasts. `--measure` and `--check` verify them.

The initial eight-text benchmark loaded ONNX in 13.87 seconds and completed
in 20.84 seconds. Before the full run, this implied roughly 20–25 minutes for
the remaining 1,404 texts. Each chunk uses a conservative twice-maximum-time
estimate and a 35-minute default stop (hard CLI maximum 40 minutes); 300-text
chunks are far shorter. All other stages also have deadline checks.

**Measured cost.** All **1,412 distinct local embeddings** completed, with
96–432 content tokens each, **1,046.58 seconds (17.44 minutes)** of forward
time, median .693 seconds, maximum 3.455 seconds. Seven chunks including model
loads took about **18.9 minutes** in total; the longest was **277.45 seconds**.
The successful fit took **33.56 s**, freeze **56.82 s**, measurement **93.07 s**.
The operational [runtime ledger](prereg/cover_runtime.csv), added after the
first two chunks, records encoder peak RSS 1.88–2.04 GiB and an overall
recorded peak **2.48 GiB** during the cone forecast stage. No run approached
45 minutes. No encoder was rerun after the forecasts froze.

**Numerical/audit checks passed:** bounded-tail inverse/quantile round trips
through \(N=10^{40}\), endpoint/interval order, elementary cell inequality,
all forecast/source hashes, 200 test records with 187 measured winners,
448 composition checks (384 in-language plus 64 depth stress), 168 future
direction checks, score reproduction, and every numerical polar constraint.
The final test audit completed at **2026-10-28 18:37:37 UTC**.

Lean's ten new lemmas cover pair polarity, positive mixture norm, cell and
language support, coherence restriction, score-failure implication, residual
inflation and finite probability-budget algebra. The grammar implementation,
beta fit, tolerance-interval probability proof, iid maximum identity and
numerical cone optimization are **not** claimed Lean-formalized. No new axioms
or `sorry` are used. Each lemma has a `#print axioms` command in the new file.

**VERIFIED:** the **single final `lake build` passed (18,120 jobs)**. All ten
new lemmas print only `propext`, `Classical.choice`, `Quot.sound`; there are
no compilation errors or added axioms. The build emits non-blocking style and
module warnings. The [verification ledger](prereg/cover_verification.csv)
records the theorem names, axiom sets, successful checks and source hashes.

Artifacts: [protocol](prereg/cover_protocol.csv), [plan](prereg/cover_plan.csv),
[models](prereg/cover_models.csv), [calibration](prereg/cover_calibration.csv),
[tuning](prereg/cover_tuning.csv), [tails](prereg/cover_tails.csv),
[size fits](prereg/cover_fits.csv), [input hashes](prereg/cover_sources.csv),
[predictions](prereg/cover_predictions.csv), [freeze](prereg/cover_manifest.csv),
[residual checks](prereg/cover_checks.csv), [test results](prereg/cover_results.csv),
[test summary](prereg/cover_summary.csv), [combined writers](prereg/cover_combined.csv),
[future checks](prereg/cover_future.csv),
[future summary](prereg/cover_future_summary.csv), [audit](prereg/cover_audit.csv).

## Open

The Round 5 questions are resolved below as an empirical decision, explicit
limits, or the specified external judgment handoff; none remains an
unspecified research task.

| Item | Round 6 disposition |
|---|---|
| Replacement uniform residual model | **STATED LIMIT.** The fresh replacement **FAILS (4/384 checks)** after prospective calibration (§6.1, §6.4, §6.6). Finite coverage cannot certify the full grammar; radius 2 is the unconditional fallback. |
| Coherence of synthetic counterexamples | **CLOSED by applied judgments.** Round 5 refuted on coherent text in both domains (original gate 2/3 arXiv, 2/2 Steam; Opus gate 2/3, 1/2); the Round-6 failures are all incoherent. Coverage of all coherent text remains a stated limit. |
| Tail/search separation | **CLOSED by theorem (§6.2).** Even exact iid marginals do not identify adaptive maxima without a dependence/selection law. The available corpus cannot certify the extrapolated extreme-tail range. |
| Corpus identity cell | **CLOSED by exact job specification (§6.3).** The Kaggle re-encoding job is prepared, not run. Prepared/local equality remains unlicensed; the job states precisely which new numerical-map claims its output would permit. |
| Support question | **CLOSED as an evidence-qualified answer** in the final paragraph. |

# Round 6, track C3: closing the scope of the cover experiment

## 6.1 Prospective protocol and frozen judgment decision rule

This section is written before Round 6 measurements and before any synthetic
judgments. The old results motivate the new model, but enter neither its
predictor fit nor radius calibration. Historical Round 5 code is preserved in
`runs/cover/round5_cover.py`; its original hash manifest remains unchanged.

The replacement predictor uses the token edit fraction
`f = max(1, token_Levenshtein(source, output)) / max(source_tokens, output_tokens)`.
For each domain/operation, choose `gamma` from {0, .5, 1, 1.5, 2} to minimize
mean squared embedding residual on 16 fresh training compositions, breaking
ties toward smaller gamma. Set `lambda = min(.75, gamma*f)` and
`q = unit((1-lambda)*p_source + lambda*p_donor)`. Independently calibrate
`c = max(residual/sqrt(f))` on 64 fresh compositions, and propose the
input-dependent radius `epsilon(x) = min(2, c*sqrt(f))`. Prepared vectors
remain arbitrary reference anchors. This is a new, falsifiable **uniform
hypothesis over derivations**, not a claim that calibration proves uniformity.

Training and calibration use separate streams of seed 2026102861. Only after
freezing their plan, embeddings and fitted model, generate 32 checks per cell
with new seed 2026102862. All draws follow the original valid-pair law
conditioned on avoiding the frozen historical text-hash inventory. That
inventory includes corpus texts, recorded compositions/caches, and every
Round-4 deterministic edit of every past source (a conservative exclusion).
The new streams use replacement, without rejection based on residuals or on
other new draws. An audit must find no duplicate/new-versus-old text hash;
a collision aborts the prospective claim rather than quietly resampling.
Source reuse is permitted; a new composition means a new complete encoded
text. Disjointness is an observed audit result, not an iid assumption.

**Frozen empirical decision:** PASS iff all 384 fresh check residuals are at
most their frozen radii plus 1e-6; otherwise FAIL, reporting every failing
cell and text. A PASS is finite-check survival only. The trivial uniform
radius 2 is always valid for unit vectors; no tighter uniform certificate
will be inferred from a PASS. No model rescue/refit uses check outcomes.

**Frozen judgment rule `cover-r6-v1` (before judgments).** Queue every Round-5
training/check assigned-cell counterexample and every Round-6 fresh-check
counterexample used here, deduplicated by exact text SHA-256, capped at 120
across both domains. If this exceeds 120, stop instead of silently omitting a
counterexample. Each JSONL record contains `id`, `domain`, the full `text`,
`text_sha256`, `decision_rule`, and its model/cell/residual/radius claims.
Judges receive the text/domain without scores, residuals or predictions.
Use the unchanged revision-3 prompt and validity/gibberish gate: Haiku
`claude-haiku-5-5`, then `gpt-6-luna`, both coherence >=7. Opus
`claude-opus-5-5` is an additional sensitivity gate, also >=7.

A queued text is an original-gate coherent counterexample iff its locally
checked validity holds, its fixed gibberish rule does not flag it, both
original scores are integers in [1,10] and >=7, and its recorded residual
exceeds its recorded radius by >1e-6. It is a stricter counterexample iff
Opus also scores >=7. **One accepted counterexample refutes the specified
assigned-cell uniform hypothesis restricted to that gate.** Zero accepted
counterexamples means only "no coherent refutation in this queue"; it never
validates a cover, an alternative anchor, or all coherent text. Missing,
failed, refused or malformed ratings remain unresolved, not rejected.
A Haiku score <7 settles original rejection even if luna is skipped; an
original rejection settles stricter rejection even if Opus is absent.
Report model/domain counts and both gates separately; never fit radii or
choose a new claim after looking at judgments.

Return one record per ID to
`~/Documents/experiment-data/latent-gap-ideas/runs/cover/judge_results.jsonl`:
`{"id":"...","text_sha256":"...","haiku":8,"luna":7,"opus":8}`.
Scores may be null when unavailable; optional error/reason/model metadata is
preserved. Unknown/duplicate IDs, mismatched hashes and out-of-range scores
are errors. `scripts/bounds/cover.py --apply-judgments` will verify the frozen
queue/rule and write an immutable result keyed by the input file hash. It
makes no LLM calls and leaves unresolved IDs explicitly pending.

## 6.2 THEOREM: a marginal tail does not identify a search maximum

**Hypotheses.** Let F be any continuous strictly increasing CDF on a
nondegenerate score interval in [-1,1], and fix budget B>=2. Suppose only F
and B are known; no joint law, conditional proposal kernel, dependence,
acceptance or stopping law is specified. Even granting that **every** proposed
score has marginal F, the law of the maximum is not identified. This is
stronger information than the corpus supplies for target-guided search.

**Proof.** With U uniform on [0,1), the completely dependent scores
`X_i=F^{-1}(U)` have maximum CDF F(h). Independent uniforms give maximum CDF
F(h)^B. The cyclic scores `X_i=F^{-1}((U+(i-1)/B) mod 1)` have the same
marginals but maximum CDF `max(0, B*F(h)-(B-1))`: the B equally spaced points
can all lie below u only over an interval of starting phases of total length
`max(0,B*u-(B-1))`. These CDFs differ for interior h. Nonadaptive proposals
are special cases of adaptive algorithms, so nonidentifiability in this
subclass proves nonidentifiability for the larger class. These examples
need not be realizable by this fixed encoder: they prove that a derivation
from F and B **alone** is impossible. For a finite discrete encoder law the
first two constructions still differ whenever 0<F(h)<1. QED.

Here is the information a valid search model would need. Fix the target,
initial retrieved seeds, writer/gate versions and budget/stopping policy.
Let H_(i-1) contain all previous proposals, scores, gate outcomes and feedback;
let K_i(dx|H_(i-1),t) be the conditional proposal kernel. Include any random
gate in this kernel or in a specified conditional observation law. Assign
rejected proposals score -infinity. For threshold h define
`p_i(h)=P(accepted score at step i > h | no earlier accepted score > h)`.
Pad early stopping with rejected proposals to a deterministic maximum budget B.
Then, whenever the preceding survival probability is positive,

\[
 P(M_B\le h)=\prod_{i=1}^B(1-p_i(h)).
\]

This is the chain rule, **not** an independence claim. A zero survival event
makes subsequent factors immaterial. Only if the conditional accepted-score
law is the same F at every history do the factors reduce to F(h). If a
justified density-ratio bound `K_i(A|H,t)<=kappa_i*P(A)` held for every history
and measurable proposal set, then a fixed-gate base exceedance probability p
would give `P(M_B>h)<=1-product_i(1-min(1,kappa_i*p))`, hence also the union
bound `min(1,p*sum_i kappa_i)`. No such bounds or kernels have been established
for the saved writers. Gate filtering, retrieval, feedback and stopping all
matter; fitting one REAL median with an effective N does not identify them.
The saved maxima and even finite full transcripts do not identify kernels
on histories never observed. This establishes the **stated limit**, without
claiming that every possible search model fails.

**Extreme-tail resolution limit.** No fresh extreme-tail validation is
claimed in this round; this item takes the theorem alternative. Round 5's
q999 tests do not validate extrapolation to the search range: its fitted
N are about 10^7--10^10.3, whose median-maximum one-draw exceedance
probabilities are approximately log(2)/N. Even *granting iid sampling* and no
temporal shift, zero exceedances among 41,000 future rows gives only the
one-sided 95% upper bound `1-.05^(1/41000)`, about 7.31e-5. To distinguish a
probability p from zero at that confidence requires at least
`ceil(log(.05)/log(1-p))` draws, approximately 3/p (4.32*N at the median
maximum). Real temporal/dependence problems weaken, not strengthen, that
idealized calculation. Fresh directions share the same rows and generally
have different F_t; their counts cannot be pooled as iid samples of a
particular direction's extreme tail. The rejected beta candidate supplies
neither this validation nor a search law. There is no defensible prediction
of adaptive maxima from the available iid-tail information alone.

## 6.3 Corpus identity: exact re-encoding job, not a sampled certificate

The prepared files were produced by the Kaggle GPU script using
SentenceTransformer, float16 computation and float16 storage; the present
local map uses the saved ONNX graph, CPUExecutionProvider, explicit EOS and
float32 output. Matching model names does not equate these maps. The four
~.002 differences cannot certify a maximum discrepancy over the corpus.

**THEOREM (finite-population sampling limitation).** Fix a threshold 0<=tau<2. For N fixed unknown
nonnegative discrepancies and a simple random sample of m distinct indices,
a population with exactly one discrepancy above the proposed bound escapes
detection with probability (N-m)/N. Therefore a procedure that certifies
"none exceed the bound" whenever a sample has no exceedances cannot have a
uniform false-certificate probability <=delta unless m>=(1-delta)*N, absent
additional structure. Proof: the exceptional index is sampled with
probability m/N. An iid tolerance bound controls a proportion, not the
maximum. The four convenience comparisons have neither this coverage nor a
random-sampling confidence certificate. The unconditional unit-vector bound
2 remains valid but uninformative.

The exact replacement job is specified by
`runs/cover/reencode_job.json` and `runs/cover/kaggle_reencode.py`, with input
SHA-256s and pinned local package versions in the job manifest. **It has not
been run or submitted.** Upload exactly the model files listed in the manifest (graph,
external weights, tokenizer/config files), the manifest, this runner, and
both original `prepared/{domain}/{corpus.parquet,emb_qwen3e06.npy}` files to a
private Kaggle dataset preserving those relative paths. Use a Kaggle CPU
session (a GPU-enabled session is also acceptable but the provider stays
CPUExecutionProvider). No model download, SentenceTransformer conversion,
GPU provider substitution, text cleaning or float16 output is permitted.
This deliberately runs the local encoder contract on Kaggle; re-running the
old GPU encoder would leave the identity mismatch unresolved.

Use Python 3.12 and install the exact packages listed in the manifest in that Kaggle session,
then run one bounded invocation at a time, for example:

```sh
OMP_NUM_THREADS=4 python kaggle_reencode.py \
  --input /kaggle/input/cover-reencode \
  --output /kaggle/working/cover-reencode --domain steam --limit 200 --minutes 35
# Repeat for steam, then arxiv, until the completion audit reports no missing rows.
```

Each invocation verifies input hashes, resumes at the first missing row, uses
one process and four ONNX/BLAS threads, and stops before 35 minutes. Inputs
are **every row, in original parquet order**, including adult/future rows;
there is no sampling, filtering or reordering in the output. For each exact
text: encode with the frozen tokenizer; append EOS 151643 if absent; if over
512 tokens retain the first 511 plus EOS; use attention mask ones, positions
0..n-1, empty KV tensors (1,8,0,128), last-token pooling, L2 normalization,
and float32 storage. Duplicate exact texts reuse their first-row vector, which the census audit
checks; this makes the saved reference a function of text within each domain.
Per-row checkpoints include row index, text/token hashes,
vector hash, norm and prepared-vector discrepancy. Atomic writes and a final
index audit prevent interrupted rows from appearing complete. A completed
census supplies the maximum prepared/re-encoded discrepancy over each
specified subset; it is a census of saved numerical outputs, not a confidence
interval or a hardware-roundoff theorem.

After completion, use the new vectors as the reference corpus of a **named
Kaggle ONNX numerical map**, and add an identity cell for each original text:
its stored encoding is exactly its stored anchor, so residual zero is an
identity by construction for that map. To call it the existing Mac numerical
map requires either a full Mac census or a certified cross-platform numerical
error bound; CPUExecutionProvider on different hardware is not a proof of
bitwise equality. Alternatively keep prepared anchors and use the completed
census maximum as identity radius for the Kaggle ONNX map. Recompute bank
latitudes, mixtures/cones, corpus ceilings and tails; re-encode synthetic
calibration/check and saved witness texts on the same frozen map before
repeating any cross-map residual or score claims. Retain old targets as
explicit fixed numerical directions for comparisons; generating targets
from the new corpus defines a different experiment and needs a new split.
An identity cell does not establish coherence: the gate must concern the
actual encoded text, including any truncation.
Existing cached local judgments transfer to the identical text, but embeddings
and radii do not transfer without rechecking. Historical results remain
labelled with their original geometry. None of this would make the grammar
cover all coherent text or certify its unencoded compositions.

## 6.4 What a replacement residual experiment can establish

Conditional on the frozen historical exclusion set and fitted predictor,
calibration and check derivations have the same specified sampling law.
Applying §2.2 to `R/sqrt(f)` gives the same marginal strict-failure bound
1/65 and simultaneous 95% calibration confidence bound 8.2071% per cell.
Clipping the radius at 2 introduces no failures for exact unit vectors.
These guarantees concern the reduced sampling law, not the excluded texts,
all syntactically valid texts, coherent conditioning, or adaptive rewrites.
The streams are pseudorandom implementations of independent draws; their
observed nonoverlap is audited, not imposed by conditioning the sampling law.
The probabilistic statements are over the full sampling procedure, not an
extra conditional guarantee given a successful novelty audit.

The positive-mixture property is retained even though lambda varies by input.
If the proposed residual inequality held for every derivation, §1.3 would
apply with the radius replaced by the maximum of `min(2,c_op*sqrt(f))` over
**all** valid derivations. Its range on the checks is not that maximum, and
calibration does not establish the inequality at unmeasured derivations.

**THEOREM (remaining uniform limitation).** Fix the measured transcript,
reference anchors and fitted predictor q(x). If an unobserved grammar text z
has a proposed radius rho(z)<2, the class of unit-valued maps consistent with
the transcript contains one mapping z to -q(z). Its residual is 2>rho(z),
with every observed value and calibration statistic unchanged. If z belongs
to an independently fixed coherent set, the same argument applies within
that set. Proof: alter the map at z only; its output remains unit norm.
For multiple derivations it suffices to violate one assigned cell. Thus
finite residual observations alone cannot certify any proposed sub-2 radius
at an unobserved cell. This is a statement about inference from observations
and unit normalization, **not** about what the known Qwen weights can or
cannot realize. An exhaustive evaluation of the finite grammar or a global
bound derived from those weights could escape it; neither has been supplied.
A finite-check PASS would not supply either missing ingredient.

The always-valid fallback for this predictor is rho=2 at every cell, by the
triangle inequality. Substituting it into the cone certificate yields only
support <=1. Calling a larger sample maximum a "uniform calibration" would
hide the missing quantifier. The replacement's numerical pass/fail is
reported below; the nontrivial global uniform claim is a **stated limit**.

## 6.5 Reproduction and handoff

Run these stages sequentially, one encoder process at a time. Each embedding
invocation is checkpointed and bounded; repeat it until its phase is complete.
Chunk size is operational only and does not change the frozen sampling plan.

```sh
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python scripts/bounds/cover.py --r6-plan
.venv/bin/python scripts/bounds/cover.py --r6-embed --limit 8
.venv/bin/python scripts/bounds/cover.py --r6-embed --limit 64 --minutes 40
# Repeat --r6-embed until all 960 train/cal texts are checkpointed.
.venv/bin/python scripts/bounds/cover.py --r6-fit
.venv/bin/python scripts/bounds/cover.py --r6-plan-check
.venv/bin/python scripts/bounds/cover.py --r6-embed --phase check --limit 32 --minutes 40
# Repeat until all 384 checks are checkpointed.
.venv/bin/python scripts/bounds/cover.py --r6-measure
.venv/bin/python scripts/bounds/cover.py --r6-queue
.venv/bin/python scripts/bounds/cover.py --r6-check
# After the experimenter returns the actual judgments:
.venv/bin/python scripts/bounds/cover.py --apply-judgments
```

In this checkout the plan/fit/result stages refuse overwrite; use `--r6-check`
to audit completed artifacts. `--apply-judgments --results <path>` supports an
explicit returned-results file. Its output is
`runs/cover/judgment_application_<results-sha256>.json`, including preserved
raw ratings, per-claim gate decisions and model/domain summaries. The queue's
`decision_rule` is the frozen ID `cover-r6-v1`; its complete definition is
stored in neighboring `protocol.json` and §6.1. Both files accompany the
experimenter's queue. Never present the claim/residual fields to the judges. This selected queue
cannot estimate a population coherence rate or the acceptance probability q.

The Round 5 source hash is checked against the archived
`runs/cover/round5_cover.py`, **not** silently replaced by the new program's
hash. All original frozen data hashes are still checked. Round 6 separately
pins the new code, predictor/calibration files, source data and check plan.
No Lean source was changed and no `lake build` was run: the new statements
here are prose proofs, not claimed Lean formalizations. The re-encoding job
has only been syntax-checked and specified; no Kaggle execution is implied.

## 6.6 Fresh-check outcome and final audit

**Replacement result: FAIL — 4/384 fresh checks (1.0417%) violate the
frozen radii, two per domain.** The fit was frozen at `2026-10-28T19:46:53.268632+00:00`;
the separate-seed check plan was frozen at `2026-10-28T19:48:31.712956+00:00`.
The 192 training and 768 calibration compositions are fresh; the 384 checks
are distinct from those 960 texts and from the frozen 680,750-hash historical
inventory. The final audit also found no overlap with the encoder-cache text hashes
stored at the end of this run. No check residual entered fitting or radius calibration.

The table gives the fitted gamma and c from §6.1. The radius range is over
**checks only**, not the whole grammar. Each row has 32 checks; all decisions
use full precision and the frozen 1e-6 numerical tolerance.

| Domain | Op | gamma | c | Check radius range | Failures | Maximum residual minus its radius |
|---|---|---:|---:|---:|---:|---:|
| steam | F | 0.5 | 1.461518 | 0.451034–0.906395 | 0/32 | -0.110763 |
| steam | S | 0.5 | 1.329319 | 0.281499–0.751186 | 1/32 | +0.033039 |
| steam | T | 1 | 2.182988 | 0.140911–0.482141 | 0/32 | -0.038770 |
| steam | FS | 1 | 1.166081 | 0.403942–0.834787 | 0/32 | -0.070377 |
| steam | FT | 1 | 1.437668 | 0.491057–0.915185 | 0/32 | -0.086651 |
| steam | ST | 0.5 | 1.471845 | 0.283915–0.771574 | 1/32 | +0.067322 |
| arxiv | F | 2 | 2.393549 | 0.529234–1.003224 | 1/32 | +0.021969 |
| arxiv | S | 0.5 | 1.260966 | 0.304637–0.803963 | 0/32 | -0.124942 |
| arxiv | T | 1 | 2.163308 | 0.142028–0.424260 | 0/32 | -0.095597 |
| arxiv | FS | 1.5 | 1.436071 | 0.521900–0.938993 | 0/32 | -0.077144 |
| arxiv | FT | 2 | 2.330676 | 0.533419–0.914166 | 0/32 | -0.071329 |
| arxiv | ST | 0.5 | 1.118921 | 0.367715–0.687622 | 1/32 | +0.044997 |

All four new counterexamples are explicit:

| Domain/op | Draw | Source / donor past positions | Residual | Frozen radius | Excess | Text SHA-256 prefix |
|---|---:|---|---:|---:|---:|---|
| steam/ST | 1 | 39340 / 34584 | 0.550479 | 0.483157 | +0.067322 | `a28f0af8f72b063e` |
| steam/S | 2 | 61137 / 367 | 0.338738 | 0.305699 | +0.033039 | `689383053dc14ce8` |
| arxiv/ST | 9 | 38822 / 30929 | 0.439648 | 0.394651 | +0.044997 | `304bcc365faab29d` |
| arxiv/F | 31 | 29379 / 50769 | 0.637253 | 0.615283 | +0.021969 | `66c58380bd2e5ebf` |

These failures refute the new assigned-derivation uniform hypothesis on the
syntactic grammar. They do not refute the calibrated one-draw probability
bound, prove failure of every possible anchor, or establish coherence before
judgment. The largest new excess is .067322 (Steam ST); arXiv's largest is
.044997 (ST). The replacement is not rescued or refitted after this result.
A useful global residual certificate remains unavailable; the unconditional
radius-2/support-1 fallback and the precise limitation in §6.4 close that
claim as a stated limit.

**Judgment handoff (now applied, below).** The published queue contains **9
complete texts: 4 Steam / 5 arXiv**. It contains all four Round-5 check
failures, the arXiv FT Round-5 training failure, and all four new failures.
All nine pass the frozen local format and gibberish checks. None has new
Haiku, luna or Opus ratings in this round. The queue and complete rule are at
`~/Documents/experiment-data/latent-gap-ideas/runs/cover/judge_queue.jsonl`
and neighboring `protocol.json`; byte-identical repository copies are in
`runs/cover/`. Queue SHA-256:
`820dea7921443f1744b2b64d2f21169f398dded1df77c7a720c76d0b2d7aae5f`.
Full text hashes and model-specific residual claims are in every record.

**Judgment outcome (applied 2026-10-28, after the freeze).** The nine texts
were judged once with the unchanged revision-3 prompt through the subscription
CLIs (Haiku, then luna only when Haiku >=7; Opus on all nine) and the results
were applied with `--apply-judgments`
(`runs/cover/judgment_application_cdaffe73….json`; ratings in
`runs/cover/judge_results.jsonl`). Nothing is pending.

| Model | Domain | Queued | Original gate accepted | Opus gate accepted | Verdict |
|---|---|---|---|---|---|
| Round 5 | arXiv | 3 | 2 | 2 | REFUTED |
| Round 5 | Steam | 2 | 2 | 1 | REFUTED |
| Round 6 | arXiv | 2 | 0 | 0 | no coherent refutation in queue |
| Round 6 | Steam | 2 | 0 | 0 | no coherent refutation in queue |

So the Round-5 covered-language residual model is refuted on coherent
synthetic text in both domains, under both gates. The Round-6 replacement
still FAILS its frozen residual check (4/384), but none of its four failing
compositions is coherent: under the gate its refutation is a refutation of
the grammar-wide claim only, not of a coherent-text claim.

**Verification passed.** `--r6-check` verifies the old frozen artifacts via
the original code archive, all new freezes, all 1,344 text identities and
embeddings, the 384 checks, and accept/reject/pending judgment semantics.
The application step was exercised with an explicitly empty smoke-test
results file: every original/Opus decision remained pending, with zero
accepted claims. Unknown IDs, mismatched text hashes and duplicate IDs were
rejected. These are software tests, **not returned judgments**; the default
external `judge_results.jsonl` has not been created. Every measured vector
and scalar was checked finite, and the external published bytes match the
frozen repository queue and rule. The Kaggle runner passed syntax checks;
its manifest still says `SPECIFIED_NOT_RUN`.

**Measured cost.** There were 24 successful encoder chunks, totaling
57.05 minutes including model loads,
and 43.30 minutes of recorded forward time.
The longest successful invocation took 351.07
seconds (5.85 minutes); peak recorded
RSS across Round 6 stages was 1.90 GiB.
Three conservative preflight refusals ran no new forwards; chunk sizes were
reduced instead of increasing concurrency. Every successful measurement
remained below 45 minutes. One encoder process, `LocalQwen(threads=4)`, and
OpenMP/BLAS caps of four were used throughout. No LLM calls, model downloads,
Kaggle submission, Lean changes/builds or git commits were made.

Artifacts: [models](prereg/cover_r6_models.csv),
[calibration](prereg/cover_r6_calibration.csv),
[model freeze](prereg/cover_r6_freeze.csv),
[fresh check plan](prereg/cover_r6_check_plan.csv),
[checks](prereg/cover_r6_checks.csv), [summary](prereg/cover_r6_summary.csv),
[judgment freeze](prereg/cover_r6_judge_freeze.csv),
[verification](prereg/cover_r6_verification.csv),
[local queue](../runs/cover/judge_queue.jsonl),
[re-encoding job](../runs/cover/reencode_job.json).

## Status of the support question

We proved how a guarantee that every allowed text stays near known examples
would limit how far its embedding can reach, and why our samples and ordinary
corpus tails cannot, by themselves, supply that guarantee or predict a guided
search. Accepted texts refute the claim that coherent embeddings stay inside
mixtures of corpus vectors; proposed bounds meant to cover every grammar
composition, including the fresh replacement, also fail direct tests. Judged for
coherence, four of the five Round-5 synthetic failures are coherent (three
under Opus too), while the four Round-6 failures are not. The stated limits
are that this grammar misses known coherent rewrites, sampling does not
certify its untested compositions, and GPU corpus vectors cannot simply be
equated with local encodings. **Text accepted by the fixed gate can go beyond
the sampled corpus boundary in some FAR directions**: it exceeds the stricter
reference by .05 cosine on 5/20 Steam targets (3/20 with Opus) and 0/20 arXiv
targets. This demonstrates some escape beyond the corpus; whether coherent
text can reach arbitrary FAR targets, or get close to them, remains
unestablished. For all coherent text, the only unconditional upper bound
proved here is the universal maximum similarity of 1; accepted texts give
lower bounds on what can be reached.

Continuation: [Round 7 — finite maxima, coherence-query impossibility, and restrictions that rescue certification](bounds7_impossibility.md).
