# Round 4, track C: coherent encoder support

2026-10-28. Inputs: [bounds §3](bounds.md), [bounds2](bounds2.md),
[bounds3](bounds3.md), [fluent search revisions 1–3](prereg/fluent_search.md),
and their existing local data. No LLM calls, new targets, or git commits.

**THEOREM — main limitation.** Corpus vectors, finitely many encoder probes,
and their local slopes/ranks cannot certify a nontrivial upper bound on support
over a fixed coherent set containing an unobserved text. This remains true when
the observations include every successful search witness. A global cover or a
global bound on the error of a composition model is an additional hypothesis,
not a consequence of those observations.

**EMPIRICAL LAW — new falsification.** A useful, unusually tight geometric
relaxation can nevertheless be computed: the entire normalized positive cone
of the corpus. Its FAR support has median **0.0561 Steam / 0.0876 arXiv**.
The recorded judge-passing rewrites exceed its certified numerical upper bound
on **12/20 Steam / 3/20 arXiv targets**. Thus even arbitrary positive mixtures
of corpus embeddings, followed by normalization, do not contain all the
observed coherent embeddings. This is stronger than showing that a particular
eight-neighbour mixture fits poorly. It refutes the zero-residual composition
model; it does not contradict the cone theorem.

## 1. Definitions and what the existing searches measure

Fix the Qwen encoder weights, tokenizer, document rendering, last-token pooling,
EOS and L2 normalization of `src/latent_gap/embedder.py`. Let \(V\) be the
canonical, round-trippable texts with at most 511 content tokens and terminal
EOS; exclude undefined/zero embeddings. Write \(E(x)\in S^{1023}\).
The encoder truncates longer inputs; for unrestricted support one can instead
work with their finitely many effective token sequences. Coherence must be
judged on the **actual encoded text**, not an untruncated original.

For domain \(D\), let \(V_D\subseteq V\) impose the recorded domain format and
field bounds. Define a fixed target-blind gate \(J_D\): valid, no recorded
gibberish flag, Haiku and luna coherence scores both at least 7 using the
revision-3 prompt. Freeze judge versions and randomness/tie handling for a
mathematical set; the actual results below use the **saved ratings**, not a new
rating. A third-judge gate defines a different, smaller set. Then

\[
 C_D=\{x\in V_D:J_D(x)=1\},\qquad
 S_c(t)=\sup_{x\in C_D}E(x)^Tt,\qquad
 S(t)=\sup_{x\in V}E(x)^Tt,\quad \|t\|=1.
\]

These finite nonempty sets have maxima. If a search finds no accepted witness,
report its lower bound as missing, not zero. The Lean statements require a
nonempty set explicitly. Neither a corpus entry nor an edited corpus entry is
automatically a member of this particular judge-defined set.

A formal alternative is \(C_{h,D}=\{x\in V_D:-\log Q(x)/n(x)\le h\}\) for a
specified autoregressive LM, header, tokenization and EOS convention. This is
a **different operational definition**, not an established equivalent of
\(C_D\). Rev 1–2 already show that low NLL and judge acceptance diverge.
No LM is run in this round.

**THEOREM — witnesses.** For a finite evaluated set \(W\),
\[
 \underline S_c(t):=\max_{x\in W\cap C_D}E(x)^Tt
 \le S_c(t)\le S(t)\le1.
\]
The last inequality is Cauchy–Schwarz. Adaptation, selecting FAR targets and
maximizing across writers preserve the witness inequality. They do not turn
it into an upper bound. The preregistration's `h_coh` is \(\underline S_c\),
not \(S_c\). Likewise the token-soup scores lower-bound \(S\).
Lean: `witness_le_support`, `support_le_one`.

**EMPIRICAL LAW — reading the controls correctly.** Revision 3 recovers REAL
targets with combined medians .842/.874 and 10/10 above q99 in both domains.
This repairs the earlier optimizer control. It cannot establish that the
same budget exhausts coherent support on FAR. Steam's 25% and arXiv's 0%
refer to reaching **ceiling* + .05**, not to simply exceeding the corpus
ceiling. In fact the combined FAR witnesses exceed the visible ceiling on
16/20 and 11/20 targets, respectively. The arXiv visible median is .055,
not .04; its ceiling* median is .063. These reference choices matter.

## 2. THEOREM: the obstruction survives restriction to coherent text

Let \(A\subseteq V\) contain all queried inputs, including corpus entries,
perturbation endpoints and search candidates. Suppose a fixed coherent set
\(C\) contains \(z\notin A\). Let \(f:V\to S^{d-1}\) be any encoder compatible
with the transcript. For every unit target \(t\), there is another unit-valued
encoder \(g\) with

\[
 g|_A=f|_A,\qquad g(z)=t,\qquad S_{c,g}(t)=1.
\]

**Proof.** Set \(g(z)=t\) and leave every other value unchanged. Membership
\(z\in C\) is unchanged because the gate is fixed independently of the encoder
score. All corpus statistics and all statistics of observed differences are
identical. The witness at \(z\) gives support at least 1; unit normalization
gives at most 1. The argument also preserves an entire observed subset of
local neighbourhoods if \(z\) is outside their union, and works for adaptive
queries: unchanged answers produce the same query transcript. Lean:
`unseen_coherent_extension` (even \(A\) need not be finite).

Consequently any data-only rule returning \(U(t)<1\) for that transcript fails
for a compatible encoder. This is an **information obstruction in the class
of unit-valued encoders**. It does not assert that the fixed Qwen architecture
can implement the altered map, that its actual weights are unknown, or that
an unseen Qwen text at \(t\) exists. A certificate using the actual weights and
the entire constrained language can escape the obstruction; finite measured
statistics alone cannot. An exhaustive transcript of \(C\) also escapes it.

**THEOREM — coherent contamination.** If \(P\) is a distribution on coherent
texts and \(z\) is coherent, \(P_\epsilon=(1-\epsilon)P+\epsilon\delta_z\)
still has only coherent support. If \(E(z)=t\), the pushforward support contains
\(t\), while \(n\) iid text draws miss the added component with probability
\((1-\epsilon)^n\). The mean/covariance changes satisfy the same bounds as
[bounds §1](bounds.md). Thus even a measured distribution over coherent texts
does not determine its support. This statement is conditional on such a text;
the encoder-extension argument above supplies the indistinguishable-world
construction. It does not conjure a witness for the actual encoder.

**THEOREM — dimension without extent is insufficient.** A rank-one smooth
curve on the unit sphere can connect any two unit vectors in dimension at
least two. Parametrize their great-circle arc by
\(e(s)=\cos(s/R)a+\sin(s/R)v\), over the required angular length times \(R\).
Its speed is \(1/R\); choosing large \(R\) makes its Lipschitz constant as small
as desired while retaining the endpoint \(t\). Smooth starts/stops can be
inserted with a longer parameter interval. Therefore a tangent rank and a
local speed bound without a bound on covered extent do not bound support.
For disconnected charts even rank-zero components can sit at unrelated
points. This continuous example explains the missing hypothesis; text itself
is discrete, and finite secants are not literal encoder derivatives.

## 3. THEOREMS: what additional hypotheses would suffice

### 3.1 Global covers and local tangent tubes

Let \(a_1,\ldots,a_M\) be unit anchors and
\(C_A(t)=\max_i a_i^Tt\). If **every** coherent embedding is within \(\rho\)
of an anchor, then

\[
 \boxed{S_c(t)\le\min\{1,C_A(t)+\rho\}.} \tag{1}
\]

**Proof.** \(e^Tt\le a_i^Tt+\|e-a_i\|\) by Cauchy–Schwarz; take the supremum.
Lean: `score_le_score_add_distance`, `cover_certificate`. A sharper version
is \(\max_i H_{1-\rho^2/2}(a_i^Tt)\), using the cap function of bounds §1
when its stated latitude assumptions hold (in particular \(\rho\le\sqrt2\)).
If a genuine global Lipschitz constant \(L\) is known for a specified input
metric and the anchors cover \(C_D\) at input radius \(\eta\), take
\(\rho=L\eta\). Both the constant and the input cover are required.

For a chart with an orthogonal projection \(P_i\), suppose all its embeddings
admit

\[
 e=a_i+v+r,\quad v\in V_i,\quad \|v\|\le R_i,\quad\|r\|\le\epsilon_i.
\]

If the charts cover **all** of \(E(C_D)\), then

\[
 \boxed{S_c(t)\le\min\{1,\max_i[a_i^Tt+R_i\|P_it\|+\epsilon_i]\}.} \tag{2}
\]

**Proof.** \(v^Tt=v^TP_it\le R_i\|P_it\|\), and \(r^Tt\le\epsilon_i\).
Lean: `tangent_certificate`, with the orthogonality equation explicit.
For a continuously extended encoder on a chart, a certified Jacobian bound
\(L_i\), Jacobian-Lipschitz constant \(M_i\), and input radius \(\eta_i\)
give \(R_i=L_i\eta_i\), \(\epsilon_i=M_i\eta_i^2/2\) by integrating Taylor's
remainder, with \(V_i\) containing the Jacobian image. This is conditional
calculus, not a measured property of the discrete Qwen map.

For an independent uniform target, \(\mathbb E\|P_iT\|^2=q_i/d\) if the
projection rank is \(q_i\), by spherical symmetry and \(\operatorname{tr}P_i=q_i\).
It is not valid to replace \(\|P_it\|\) by \(\sqrt{q_i/d}\) for a fixed or
corpus-selected FAR target, nor to substitute \(q_i\) for the ambient dimension
in the spherical cap theorem. Multiple charts also require a maximum/union
bound, not one favourable chart.

**THEOREM — direction of inference.** A maximum observed secant slope is a
lower bound on the smallest global Lipschitz constant, not an upper bound.
A measured secant rank bounds the rank of that sample matrix; unprobed
directions can increase the image dimension. A held-out reconstruction
residual tests a chart on those held-out points, not on its full domain.
These facts follow directly from taking a supremum over a larger set.

### 3.2 Normalized mixtures: the normalization term cannot disappear

Let \(D=\{p_1,\ldots,p_N\}\) be the visible unit corpus and
\(C_D(t)=\max_i p_i^Tt\). A composition hypothesis must concern **embeddings**:

\[
 E(x)=\frac{m_x}{\|m_x\|}+r_x,\qquad
 m_x=\sum_i w_i(x)p_i,\quad w_i\ge0,\quad\sum_iw_i=1,
 \quad\|m_x\|\ge s_0>0,\quad\|r_x\|\le\epsilon. \tag{3}
\]

If (3) holds for **every** coherent text and \(C_D(t)\ge0\), then

\[
 \boxed{S_c(t)\le\min\{1,C_D(t)/s_0+\epsilon\}
 =\min\{1,C_D(t)+(s_0^{-1}-1)C_D(t)+\epsilon\}.} \tag{4}
\]

**Proof.** \(m_x^Tt\le C_D(t)\), divide by its positive norm and add the
residual bound. Lean: `convex_mixture_score`, `normalized_mixture_certificate`.
For negative ceilings, \(\|m_x\|\le1\) instead gives \(C_D(t)+\epsilon\);
a unified expression is \(C_D(t)+(s_0^{-1}-1)\max(C_D(t),0)+\epsilon\).
If a unit axis satisfies \(u^Tp_i\ge a_0>0\) for every corpus vector,
\(\|m_x\|\ge u^Tm_x\ge a_0\); thus \(s_0=a_0\) is a genuine corpus-derived
choice. It supplies no bound on \(r_x\).

Lexical recombination, shared topics, attention's convex weights and a Gaussian
mixture fitted to corpus moments do **not** imply (3). Signed projections,
nonlinear layers and normalization break that implication. In particular
the successful corpus-ceiling law of bounds3 concerns dictionary extremes,
not a mixture law for coherent text embeddings.

### 3.3 A substantially tighter, computable cone certificate

Let \(K=\operatorname{cone}(p_1,\ldots,p_N)\). Its unit directions are exactly
the normalized positive mixtures in (3) with zero residual. Define

\[
 U_K(t)=\sup_{q\in K,\|q\|=1}q^Tt,\qquad
 K^\circ=\{z:p_i^Tz\le0\text{ for every }i\}.
\]

**THEOREM — dual bound.** For any \(z\in K^\circ\),

\[
 U_K(t)\le\|t-z\|,\qquad
 S_c(t)\le\min(1,\|t-z\|+\epsilon) \tag{5}
\]

provided every coherent embedding lies within \(\epsilon\) of a unit direction
of \(K\). Proof: \(q^Tt=q^T(t-z)+q^Tz\le\|t-z\|\); then use (1)'s
single-vector inequality. Lean: `cone_dual_certificate`,
`cone_tube_certificate`, `normalized_mixture_polar`.

A cheap choice is
\[
 \beta(t)=\max\left(0,\max_i\frac{p_i^Tt}{u^Tp_i}\right),\qquad
 z=t-\beta u,\qquad U_K(t)\le\min(1,\beta). \tag{6}
\]
Every denominator is positive. Then \(z\) is polar and \(\|t-z\|=\beta\).
This uses the joint score/latitude information, unlike \(C_D/a_0\).
Lean also proves `cone_ratio_certificate` directly.

**THEOREM — sharpness for the cone.** Solve nonnegative least squares
\(q=\arg\min_{q\in K}\|t-q\|^2\). When \(q\ne0\), the projection optimality
conditions give \(z=t-q\in K^\circ\), \(q^Tz=0\), hence
\(U_K(t)=\|q\|\), attained by \(q/\|q\|\). One can derive those conditions
by comparing the optimum with \(q+sp_i\), \(s\ge0\), and with positive
rescalings of \(q\). The finite generated cone is closed. The zero-projection
case gives the upper bound 0; equality need not hold, and is not asserted.
Here every FAR target has a positive corpus score, so its projection is nonzero.

The script performs NNLS column generation, then **checks every visible
corpus constraint**. Convergence is not required for an upper bound. At any
iterate with residual \(r=t-q\), set
\[
 \nu=\max_i p_i^Tr,\qquad
 z=r-\frac{\max(0,\nu)+10^{-9}}{a_0}u. \tag{7}
\]
This is polar in exact arithmetic, including the positive numerical buffer.
The output includes the dual vector, active corpus positions, nonnegative
weights, primal attainable mixture score, upper bound and feasibility check.
The arithmetic is float64, **not an interval-arithmetic certificate of ONNX**.
The observed violations are far larger than the numerical gap/buffer.

A further necessary residual is independently checkable: for any polar \(z\),
\[
 \operatorname{dist}(e,K\cap S^{d-1})
 \ge\max(0,e^Tz)/\|z\| \quad(z\ne0). \tag{8}
\]
Subtract any feasible \(q\), use \(q^Tz\le0\) and Cauchy–Schwarz. Thus a
separator can rule out **every** corpus mixture, unlike a large reconstruction
error against just eight selected neighbours.

## 4. THEOREMS: what `N_eff` could legitimately mean

A valid choice is a certified covering number
\(N_{\rm eff}(\rho)=N(E(C_D),\|\cdot\|,\rho)\), with a specified finite net
of **unit** anchors. Then (1) is exactly a `ceiling(t; N_eff) + slack` bound,
where the ceiling is that net's maximum and slack is \(\rho\). The net's
cardinality alone does not determine its fixed-target ceiling.

If the net is fixed independently of a uniform spherical target and
\(0<c-\rho<1\), the existing cap union theorem gives
\[
 \Pr\{S_c(T)\ge c\}\le
 \min\{1,N_{\rm eff}(\rho)\,p_d(c-\rho)\}. \tag{9}
\]
For selection from \(M\) initially independent uniform targets, a union bound
adds a factor \(M\); alternatively retain explicit target projections. This
does not treat selected FAR targets as unconditional uniform draws. A fitted
participation ratio, duplicate-adjusted corpus count, number of probes or
number of restarts is not such a covering number.

There is also a genuine **text-distribution** model: for a fully specified LM
law \(Q\), let \(P_c(x)=Q(x)1_{C_D}(x)/Q(C_D)\). Its pushforward consists
only of reachable embeddings. If \(Q\) has positive mass on every accepted
text, its support maximum is still \(S_c\), independently of how tiny those
masses are. This construction answers what a distribution on reachable texts
is, but supplies no small geometric support by itself.

**THEOREM — an LM mass floor gives a cardinality, usually a useless one.**
If every permitted complete text has \(Q(x)\ge p_{\min}>0\), then
\(|C|p_{\min}\le1\), by summing masses. Lean: `mass_floor_cardinality`.
For an EOS-inclusive NLL threshold \(-\log Q(x)/n(x)\le h\), with \(n\le L\),
\(|C|\le e^{hL}\). If the measured NLL omits EOS, use the separate fixed-length
prefix bounds \(|C_n|\le e^{hn}\) and sum over lengths: \(|C|\le\sum_{n\le L}e^{hn}\).
It is not legitimate to silently treat the saved mean NLL as a complete-text
probability including EOS.

**EMPIRICAL LAW — scale check.** An illustrative threshold \(h=2.5\) nats
and \(L=511\) permits \(\log |C|\) around 1278; the d=1024 cap at cosine .3
costs only about 51.4 nats (bounds §2). Inserting that count in (9) is vacuous,
even before selection and before checking whether this LM proxy equals the
judge set. A bound on average perplexity/entropy without a mass floor does
not even bound the cardinality of rare support.

**THEOREM — probability is easier than support, but is a different claim.**
For iid residuals \(R_1,\ldots,R_m\) from a specified coherent-text law, the
sample maximum \(R_{(m)}\) has
\[
 \Pr_{\rm sample}\!\left\{P(R>R_{(m)})>1-\delta^{1/m}\right\}\le\delta.
\]
Proof: the event requires all draws below the appropriate upper population
quantile; its probability is at most \((\delta^{1/m})^m\), with generalized
quantiles handling atoms. This gives a distributional tail guarantee, never
support containment. The present adaptive candidates and correlated edits
are not iid draws from a known coherent population, so this calibration is
**not applied** to them. A support conclusion would need additional minimum
mass/coverage assumptions on every possible violating region.

## 5. EMPIRICAL LAW: Jacobian-free probes and a failed extrapolation

The new `--audit` mode in `scripts/bounds/support_probe.py` is read-only and
prints JSON. It checks all 200 saved revision-3 records against their
encoder-fingerprinted ONNX caches and reproduces target-specific masking.
Corpus sizes are 68,222 Steam / 72,000 arXiv, dimension 1024. Cache fingerprint:
`6ae3845eab14497b`. It does not import or invoke a rewriter/judge workflow.

The optional local probe uses `LocalQwen(threads=4)`. Fixed RNG seed 20261028
selects eight past-corpus texts per domain independently of target scores,
with 70–170 body words, at least three sentences, valid round-trip format and
token/field bounds. Four training edits: drop first sentence, drop last,
swap first two, lowercase body. Two held-out edit types: drop middle sentence,
replace sentence spaces by newlines. Each edit is format/length checked;
**none is newly judge-rated or asserted coherent**. These are discrete
perturbations of real texts, not infinitesimal derivatives or certified
paraphrases. The final numbers include all 112 local embeddings.

Project differences \(\Delta=E(x')-E(x)\) onto the sphere tangent at \(E(x)\).
Use the first four to estimate the 95%-energy rank and participation ratio,
and reconstruct the two held-out tangent differences. Slopes use raw-token
Levenshtein distance, including the same EOS contract.

| Domain | Corpus min latitude a0 | Train rank95 median | Train PR median | Maximum train embedding radius | Median held-out tangent residual | Max held-out tangent residual | Median per-anchor max secant slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Steam | .241058 | 3 | 1.973 | .689799 | .224924 | .322034 | .038520 |
| arXiv | .199724 | 3 | 2.350 | .243751 | .210001 | .344295 | .026874 |

Rank95 cannot exceed four with four training secants. Therefore these ranks
are **not evidence for intrinsic dimension three**. The held-out residuals
are substantial even within the same source text; arXiv's largest held-out
displacement is .355650, already larger than its .243751 training radius.
The measured slopes are in embedding distance per edited token and are
lower estimates of the Lipschitz supremum.

To make a falsifiable attempted extrapolation, use the 32 training differences
per domain to define
\[
 \widehat\delta(t)=\max(0,\max_j\Delta_j^Tt),\qquad
 \widehat U_{\rm dir}(t)=\min(1,C_D(t)+\widehat\delta(t)). \tag{10}
\]
This would be a theorem if every coherent embedding were a corpus convex
mixture plus a convex mixture of these differences (allowing the zero vector).
That global transfer assumption is **not established**. In the experiment,
(10) is an **EMPIRICAL LAW candidate, falsified** by the following rows.
For comparison use \(\widehat U_{\rm rad}=\min(1,C_D+\widehat\rho)\), where
\(\widehat\rho\) is the maximum training displacement in the table above.

| Domain | Arm | Combined h_coh median (measured/n) | Directional plug-in median | Violations / measured | Radius plug-in median |
|---|---|---:|---:|---:|---:|
| Steam | FAR | .0693 (20/20) | .0589 | 12/20 | .7317 |
| Steam | GAP | .5672 (20/20) | .5835 | 8/20 | 1.0000 |
| Steam | REAL | .8416 (10/10) | .8297 | 5/10 | 1.0000 |
| arXiv | FAR | .0568 (20/20) | .0623 | 9/20 | .2988 |
| arXiv | GAP | .4724 (19/20) | .4469 | 10/19 | .6717 |
| arXiv | REAL | .8742 (10/10) | .7601 | 9/10 | .9819 |

The large radius plug-in has no score violations, but is very loose and lacks
the cover hypothesis of (1). This is checked separately against each
rewrite's **nearest** visible corpus embedding, not just its generating seed.
For example FAR median nearest distances .8855/.9047 exceed the calibrated
radii .6898/.2438. Absence of a score violation is not validation of coverage.
Indeed the proposed cover fails on **all 20/20 FAR witnesses in each domain**,
16/20 Steam GAP and 5/10 Steam REAL, and all 19/19 arXiv GAP and 10/10 arXiv
REAL witnesses. Thus this plug-in's containment hypothesis is empirically
false, even though its score envelope is not violated.
The small directional plug-in is visibly violated even though its median
sometimes exceeds the median witness: inequalities must be tested per target.

## 6. EMPIRICAL LAW: comparison with seed-and-rewrite, including the tight cone

The corpus cap uses \(a_0\) above and the domain mean axis, as in bounds §1.
All tables use the original visible corpus, with REAL hidden and near-duplicate
positions excluded exactly as in `refs.jsonl`. The stricter \(C^*\) reference
is the maximum of visible and already saved held-out ceilings. **The cone
does not include held-out corpus vectors**; it is a statement about the
visible-corpus composition hypothesis, not every possible corpus extension.
References are read, not remeasured on future data.
Tables round values; all inequality decisions use unrounded float64 values.
Bound medians include every target, while witness medians exclude missing
accepted texts and report their coverage explicitly.

| Domain | Arm | Visible ceiling median | Cap upper median | Mixture C/a0 median | Ratio-cone upper median | Optimized cone upper median |
|---|---|---:|---:|---:|---:|---:|
| Steam | FAR | .0419 | .9451 | .1740 | .0941 | .0561 |
| Steam | GAP | .5331 | 1.0000 | 1.0000 | .9993 | not computed |
| Steam | REAL | .7294 | 1.0000 | 1.0000 | 1.0000 | not computed |
| arXiv | FAR | .0550 | .9584 | .2755 | .1327 | .0876 |
| arXiv | GAP | .4280 | 1.0000 | 1.0000 | .9962 | not computed |
| arXiv | REAL | .7386 | 1.0000 | 1.0000 | 1.0000 | not computed |

The cap and loose normalized-mixture certificates are numerically loose
against every arm. They are upper bounds only for their stated containment
models. The optimized cone is **numerically tight for that geometric model**:
maximum primal–dual gaps are 3.49e-9 Steam / 4.21e-9 arXiv, with at most
16/27 NNLS column additions. All checked polar inner products are negative
(largest below −1.18e-9). This is optimization tightness, not text reachability.
The attaining positive combination need not be an encoding of any text.

| Domain | Writer | FAR h_coh median | Coverage | Violates ratio cone | Violates optimized cone | Largest h_coh − optimized cone |
|---|---|---:|---:|---:|---:|---:|
| Steam | Sonnet | .0529 | 20/20 | 1 | 9 | +.04016 |
| Steam | luna | .0684 | 20/20 | 4 | 12 | +.08398 |
| Steam | max | .0693 | 20/20 | 4 | 12 | +.08398 |
| arXiv | Sonnet | .0309 | 18/20 | 0 | 1 | +.00976 |
| arXiv | luna | .0505 | 19/20 | 0 | 2 | +.00917 |
| arXiv | max | .0568 | 20/20 | 0 | 3 | +.00976 |

Concrete reproductions: Steam k=12, Sonnet score .093979 versus cone upper
.053818 (Opus 9); Steam k=12, luna .137799 against the same upper (fails Opus).
arXiv k=20, Sonnet .106745 versus .096982 (Opus accepts). These establish
positive residual requirements without relying solely on luna's self-gating.
The test with Opus concerns the already rated winning texts; it does not
impute Opus acceptance for other candidates.

Requiring the existing Opus rating as well leaves **11 Steam / 2 arXiv FAR
targets** above the optimized cone, with 19/20 targets measured in each domain.
The largest score excesses are .04670/.00976. Thus the zero-residual model
fails even under that stricter gate. Separators from (8) force a **uniform
Euclidean residual of at least .11036/.06397** for the original gate, or
.08243/.05989 for the Opus-filtered recorded winners. These distance lower
bounds can be larger than score excesses and are still not sufficient errors
for a global tube.

**EMPIRICAL LAW — minimum slack forced by witnesses.** Any bound of the form
\(S_c(t)\le C_D(t)+\epsilon\) with one slack per domain/arm must have at least
the largest observed excess below. These are **lower bounds on necessary
slack**, never sufficient slack estimates. Substituting them back would be
fitting an envelope to known maxima, not deriving a support bound.

| Domain | Arm | Combined h_coh median | Minimum slack over visible ceiling | Minimum slack over ceiling* |
|---|---|---:|---:|---:|
| Steam | FAR | .0693 | .09541 | .09541 |
| Steam | GAP | .5672 | .12899 | .12458 |
| Steam | REAL | .8416 | .15232 | .14479 |
| arXiv | FAR | .0568 | .05278 | .03835 |
| arXiv | GAP | .4724 | .09461 | .09068 |
| arXiv | REAL | .8742 | .18769 | .16720 |

In particular a universal \(C^*+.03\) upper bound is **violated** on FAR by
8 Steam and 2 arXiv targets, even though the preregistered arXiv population-share
rule says “supported”. That experimental verdict never asserted a universal
inequality. Requiring Opus as well leaves the preregistration's large FAR
escapes at 3/20 Steam and 0/20 arXiv; it changes the gate, not the theorem.

**EMPIRICAL LAW — anchoring audit.** For each accepted rewrite, select its
eight nearest visible corpus vectors in embedding space (not the eight
target-nearest texts shown in the original decoder). Let \(m\) be their mean,
\(c=m/\|m\|\), \(a=e^Tc\). The exact §3 identity still holds:
\(e^Tt=a(c^Tt)+(e-ac)^Tt\). The present neighbourhoods are selected after
observing the rewrite, so this is a diagnostic, not an independent prediction.

| Domain | Arm | Median ||m|| | Median ||e−unit(m)|| | Median a | Median directional remainder |
|---|---|---:|---:|---:|---:|
| Steam | FAR | .8210 | .7735 | .7009 | +.0914 |
| Steam | GAP | .8458 | .6101 | .8139 | +.1130 |
| Steam | REAL | .8374 | .5607 | .8427 | +.1673 |
| arXiv | FAR | .7441 | .7229 | .7387 | +.0865 |
| arXiv | GAP | .7818 | .6465 | .7910 | +.1361 |
| arXiv | REAL | .8227 | .5419 | .8532 | +.1538 |

Medians of terms do not add. The new FAR remainders are positive and material;
the old decoder's approximately cancelling anchoring factors and small
remainder cannot be promoted to a constraint on all coherent text. Large
eight-neighbour residuals alone would not exclude other mixtures; the cone
separators above do.

## 7. CONJECTURE: the useful next support model is a covered language, not a fitted N

A testable restricted model would specify a domain grammar/composition law
over real texts, a cover of its allowed edits/compositions, and an independently
bounded encoder residual for every cover cell. Equations (1), (2) or (5)
would then give an actual upper bound; corpus-aligned composition with a
small but nonzero residual might explain the modest FAR excess. This is a
**CONJECTURE about a restricted language**, not evidence that the whole
judge-defined coherent set has such a cover.

The necessary next measurements are: independent source texts and edit types,
held-out composition depths, directional residual tails, and witnesses that
maximize separation from the full corpus cone rather than just target score.
For a probabilistic claim, freeze the text sampling law and use independent
calibration; for a support claim, add a certified global cover or a justified
minimum-mass/regularity assumption. Increasing the number of sampled edits
improves diagnostics but does not remove the quantifier gap in §2.

The result of this round is therefore a sharper obstruction plus a falsified,
computable composition model. The only unconditional numerical support upper
bound established for the entire coherent set remains **1**, bracketed below
by the actual judge-passing witnesses. No fitted `N_eff` is offered as a
substitute for the missing global hypothesis.

## 8. Reproduction, runtime and formal scope

```sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
UV_CACHE_DIR=/tmp/latent-gap-uv uv run python scripts/bounds/support_probe.py --self-check
UV_CACHE_DIR=/tmp/latent-gap-uv uv run python scripts/bounds/support_probe.py \
  --audit --local-edits 8 --cone-steps 32 > /tmp/latent-gap-support-final.json
# From lean/, one full library build after completing the sources:
lake build
```

The script preserves the old no-flags/`--probe` preparation mode used in
bounds2; it is **not invoked** here. `--audit` writes only stdout, hashes its
input reference and witness records, preserves missing scores, validates the
fingerprinted cache, checks masking and score reproduction, and emits
per-target measurements and cone certificates. Optional probes use four
ONNX threads. A conservative throughput check and a 28-minute default deadline
guard the optional work; `--minutes` cannot exceed 30. No output is written
back to the source data. A cache miss fails visibly rather than downloading
a model or silently changing the encoder.

**EMPIRICAL LAW — cost and checks.** The initial local benchmark loaded ONNX
in 13.9 s and embedded 191–203 token texts in .87–1.10 s. The corpus/cache plus
112-embedding audit took 140.6 s; a separate 40-target cone audit took 52.9 s.
The exact orthant self-checks passed, including a case where normalization
raises support above the dictionary ceiling. The final combined audit took
**365.7 s (6.1 minutes)**, below the 30-minute limit. It checked all 200 runs,
112 new local embeddings and 40 FAR cone certificates. The last metadata-only
script refinement additionally hashes normalized past vectors and texts and
keeps uncomputed-bound violation counts missing; it changes no measurements.

**Formal scope.** `lean/LatentGap/Support.lean` proves the supremum/witness
rules, cover and tangent inequalities, convex/normalized-mixture certificates,
cone dual/tube/polar/ratio inequalities, necessary scalar residual, exact
coherent-set oracle extension, and probability-mass cardinality inequality.
It includes its own `#print axioms` commands and is imported by the library.
The smooth example, probability arguments, projection optimality proof and
empirical containment tests are prose/Python, **not claimed Lean formalized**.
**VERIFIED: the single final `lake build` passed (18,118 jobs).** The focused
compiler check and full build both passed all **15 theorems**; every
`#print axioms` reports only `propext`, `Classical.choice`, `Quot.sound`.
No `sorry`, added axioms or compilation errors; only style/module warnings.
The full log is `/tmp/latent-gap-support-lake-build.log`, and the focused
axiom log is `/tmp/latent-gap-support-lean-check.log`. A final cache-only run
also checked the added provenance hashes and reproduced all 200 recorded
witness scores exactly, including missing values.

Support source SHA-256:
`7fba7be669da1d20d38b3ee1f827d22cb624d6e3ca5150918d3c45f6391485fb`.
