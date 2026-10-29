# Round 7: what cannot be certified about coherent support

2026-10-29. Read with [Round 4 §§2–3](bounds5_support.md),
[Rounds 5–6](bounds6_cover.md), and [fluent search, revisions 1–3](prereg/fluent_search.md).
This round uses only existing text, CSV and JSON/JSONL files. No encoder,
LLM, network, model download, or local Lean build was run.

**THEOREM — answer, with the necessary qualifications.** A finite coherent
set has a maximum. What fails is **uniform, zero-error certification from
incomplete coherence labels**, over the unrestricted class of predicates.
Keep the actual encoder fixed, even reveal its entire finite lookup table.
If an unlabelled admissible text scores above a proposed bound, and all
known coherent texts score at most that bound, two coherence predicates can
match every observation while placing their maxima on opposite sides of it.
Certification requires ruling out **every** such text, or a restriction
that logically determines their labels together.

This is not a theorem that coherent text reaches every target, that its
maximum equals 1, or that no upper bound exists. Those stronger readings
are false in general. In particular, **b < 1 does not imply that a fixed
encoder has a text scoring above b**. The relevant endpoint is the encoder's
actual maximum over the allowed finite language, which may itself be below 1.

## 7.1 Existence and the lower bounds already in hand

### THEOREM — finite maximum and witness certification

Fix a domain, a tokenizer with finite vocabulary size v, and content cap L.
Let V be a specified set of canonical, round-trippable rendered texts within
that cap, with defined nonzero embeddings. Enforce known syntactic eligibility
rules in V, rather than charging oracle queries for deterministically invalid
strings. Append the prescribed terminal EOS. There are at most

\[
 |V|\le\sum_{n=0}^{L}v^n
\]

such texts. We do not include arbitrarily long original strings whose suffixes
the encoder discards: their coherence need not equal the coherence of the
encoded prefix. Fix the encoder E, its numerical or idealized mathematical
contract, a unit target t, and the score s(x) = cos(E(x),t). In the mathematical
unit-vector model s(x) = E(x)ᵀt. For a fixed predicate C: V → {0,1}, put

\[
 h_C^*(t)=\max_{x\in V:C(x)=1}s(x),\qquad
 S_E(t)=\max_{x\in V}s(x).
\]

**PROOF.** A nonempty finite subset of the reals has an attained maximum.
For any nonempty set W of accepted witnesses,

\[
 \max_{x\in W}s(x)\le h_C^*(t)\le S_E(t)\le1.
\]

The first inequality is set inclusion; the last is Cauchy–Schwarz. If C is
empty there is no real maximum: use −∞ in the extended reals, or phrase the
certificate as ∀x, C(x)=1 ⇒ s(x)≤b. A missing search witness is **missing**,
never an observed maximum of zero. Lean uses nonempty finite sets for
`hStar` and the pointwise `UpperBound` for the empty case.

Consequently one accepted text certifies a lower bound. Searching adaptively,
selecting FAR targets, failing a positive control, or comparing several
writers does not invalidate that witness inequality. None supplies an upper
bound. Throughout this document the historical `h_coh` column is an observed
lower bound, not h*. A universal bound is falsified by one accepted violation;
the preregistration's population-share verdict is a different statement.

**THEOREM — a finite set has a strict gap unless it hits the target exactly.**
For nonempty V and unit embeddings, if t∉E(V), then S_E(t)<1 and there is
some ε(t)>0 with h*≤1−ε(t), for every nonempty coherent subset.
**PROOF.** Equality of a unit-vector cosine to 1 requires E(x)=t. Every
score is therefore strictly below 1, and the maximum of finitely many such
scores is strictly below 1. Take ε(t)=1−S_E(t). In an idealized atomless
target law independent of the fixed encoder, a finite union of singletons
has probability zero, so this strict gap exists almost surely. This is not
a numerical lower bound on ε, and it is not an exact-hit audit of the saved
floating-point targets. It further rules out the literal claim that no
nontrivial bound can exist; the challenge is certifying a specified value.

### EMPIRICAL — seed-and-rewrite test, both coherence gates

The original gate requires valid rendered text, no fixed gibberish flag,
Haiku ≥7 and luna ≥7. The stricter gate additionally requires Opus ≥7.
For mathematical purposes freeze a target-blind predicate and its randomness
per text; empirically these are the **saved decisions**, not assertions of
human consensus or newly repeated judgments. Corpus provenance alone does
not imply either gate: the recorded calibration accepted 80% of Steam and
92% of arXiv corpus texts under the original gate
([preregistration, Judge pair calibration](prereg/fluent_search.md)).

Each cell below is **median of target-specific witness lower bounds
(targets with a witness / test targets)**. “Either” takes the maximum across
the two writers *at each target before taking a median*. These medians are
not bounds for an arbitrary individual target; the individual records are
the certificates. Missing targets are excluded from medians. Filtering can
therefore raise a displayed median without improving any individual witness.

| Domain / arm | Original: Sonnet | Original: luna | Original: either | +Opus: Sonnet | +Opus: luna | +Opus: either |
|---|---:|---:|---:|---:|---:|---:|
| Steam FAR | .052878 (20/20) | .068409 (20/20) | .069255 (20/20) | .047210 (18/20) | .056494 (14/20) | .071912 (19/20) |
| Steam GAP | .549962 (20/20) | .535917 (20/20) | .567247 (20/20) | .555788 (17/20) | .536270 (12/20) | .551210 (18/20) |
| Steam REAL | .839621 (10/10) | .812212 (10/10) | .841620 (10/10) | unassessed | unassessed | unassessed |
| arXiv FAR | .030878 (18/20) | .050524 (19/20) | .056810 (20/20) | .028890 (16/20) | .054575 (13/20) | .034005 (19/20) |
| arXiv GAP | .474327 (12/20) | .434259 (19/20) | .472435 (19/20) | .474327 (8/20) | .435679 (17/20) | .472435 (17/20) |
| arXiv REAL | .866371 (9/10) | .849830 (10/10) | .874205 (10/10) | unassessed | unassessed | unassessed |

Sources: `DATA/runs/seed-rewrite/test/{domain}/{writer}/{arm}-{k}.json`,
`gate.h_coh`, `gate.coh_text`, `gate.j1`, `gate.j2`; join Opus by
`{domain}/{writer}/{arm}-{k}` in `test/opus_check.jsonl`.
Here DATA = `~/Documents/experiment-data/latent-gap-ideas`.
The repository's [cover_results.csv](prereg/cover_results.csv) records the
per-writer witnesses and text hashes;
[cover_combined.csv](prereg/cover_combined.csv) independently reproduces the
combined medians and coverage, up to numerical rounding. The lightweight
read in this round checked all **200 records**, the **187 original accepted
winners**, their saved ≥7 ratings, valid/non-gibberish candidate flags,
exact-text hashes and matching scores.
There are **148 Opus records**, all FAR/GAP; there are **no Opus REAL ratings**.
Opus filtering reranks only the original accepted writer winners; it does
not judge or search lower-scoring alternatives.

Concrete certificate: Steam/luna FAR-13 has score **.0932356268**, Haiku **8**,
luna **10**, Opus **8**, in the above `steam/gpt-6-luna/FAR-13.json` and Opus
record. Thus both relevant h* values at that target are at least .0932356268
under the saved scoring contract. Relative to ceiling* + .05, the combined
FAR escape counts are **5/20 Steam, 0/20 arXiv** under the original gate and
**3/20 Steam, 0/20 arXiv** with Opus. The Opus denominator of 20 includes one
target without a surviving witness in each domain; it does not impute a score.

### EMPIRICAL — GPU fluent-search witnesses

These are the original two-judge gate, maximized over the saved λ runs per
target. Scores were reverified with local ONNX. Opus was not applied to these
GPU-run winners. Rev 1 and Rev 2 are separate historical arms, as preregistered.

| Run / LM | Domain | FAR lower-bound median | GAP lower-bound median | REAL lower-bound median |
|---|---|---:|---:|---:|
| Rev 1 / 1.7B | Steam | .019767 (5/20) | .443037 (3/20) | .644795 (2/10) |
| Rev 1 / 1.7B | arXiv | −.011375 (18/20) | .392050 (20/20) | .685664 (10/10) |
| Rev 2 / 4B | Steam | .000159 (20/20) | .448500 (19/20) | .614866 (8/10) |

Sources: `DATA/runs/fluent-search/{full,steam4b}/targets.csv`, `h_coh`;
the corresponding `gate.csv` retains each λ winner's actual text and two
ratings. [Fluent-search results](prereg/fluent_search.md) give the experiment
and failed REAL-control verdicts (2/10 Steam and 7/10 arXiv ≥q99 in Rev 1;
4/10 Steam in Rev 2). These failures limit conclusions from *misses*, not
the lower bounds from accepted texts. Under an identical frozen gate, the
union of any witness collections gives their pointwise maximum as a further
lower bound; no pooled experimental verdict or pooled median is inferred here.

All reported decimals are historical numerical evaluations, not interval
certificates for ideal real arithmetic. Round 6 §6.3 also established that
prepared GPU corpus vectors and local ONNX outputs cannot be silently
identified. That limits geometric comparisons with the corpus; it does not
turn an accepted local witness into an upper bound or affect the combinatorial
theorem below, which uses one fixed score function in both worlds.

## 7.2 The non-certifiability theorem

### Model of a certifier

Let K ⊆ V be texts with initially supplied coherence labels. To strengthen
the observer, K may include **every corpus text**, with its actual label,
and every previously settled judge decision. If the corpus is only provided
as texts and embeddings, its unknown coherence labels are not free: either
include only its genuinely known labels in K or explicitly grant them as
extra information. No assumption that every corpus item is coherent is needed.

A certifier receives the corpus, its reference embeddings, t and b or a rule
for selecting b. It may inspect weights and query E on any texts, without a
budget. It adaptively queries C on at most q texts, then returns a claimed
upper bound (or abstains). Let Q be the set of distinct coherence queries,
A = K ∪ Q, and P = {x∈A: C(x)=1}. A transcript is consistent: repeated
settled labels agree. Missing/failed ratings are not negative labels.
Queries outside V carry no additional information about unrestricted C|V.

The compatible worlds are

\[
 \mathcal V(A,P)=\{C'\subseteq V:C'\cap A=P\}.
\]

“Certified” means the bound holds for **every** compatible predicate. This
is zero-error, worst-case soundness, not a Bayesian posterior under an
unstated prior. Randomized zero-error certification has the same obstruction
on any fixed random tape on which a bound is issued. A procedure may always
abstain or return 1; we do not claim every procedure must issue b.

### THEOREM — fixed-encoder two-world indistinguishability

Suppose P is nonempty, every x∈P satisfies s(x)≤b, and there exists
z∈V\A with s(z)>b. Then there are two nonempty predicates C₀,C₁ agreeing
on all corpus labels and all queried labels, with the same encoder everywhere,
such that

\[
 h_{C_0}^*(t)\le b<h_{C_1}^*(t).
\]

**PROOF.** Take C₀=P and C₁=P∪{z}. Both intersect A in P. The low-world
maximum is the maximum of the known positive scores, hence at most b;
the high world contains z. Every encoder answer is identical, including at z
if its embedding was queried. For adaptive queries, induct on the transcript:
equal initial information and previous answers give the same next query;
all queries on the first path have unchanged labels, hence the same next
answer and eventual output. This remains true after fixing a random tape.
Lean: `two_worlds`, `Certifier.run_eq_of_agreement`.

Once chosen, C₀ and C₁ are functions of text alone. Choosing the adversarial
pair after seeing a target does not make the predicates take that target as
an input; it witnesses failure of a guarantee quantified over all predicates.
No claim is made that these arbitrary predicates are linguistically plausible
or implementable by the particular frozen judges. Such constraints are exactly
the additional hypotheses needed to leave this oracle model.

The qualifications matter:

* If a known coherent text already exceeds b, the low world is impossible
  and the proposed bound is already **refuted**, not unidentified.
* The exceptional text must be outside **K ∪ Q**, not merely absent from
  the newest queries. Its label may already be fixed by the corpus transcript.
* If P is empty, the same construction proves pointwise bound versus
  violation. For two nonempty real-max worlds one additionally needs a
  permissible positive text scoring ≤b that is not labelled negative.
* If b≥S_E(t), no high world exists. Even with b<1 this can happen: a
  constant encoder orthogonal to t has S_E(t)=0. Conversely, when an
  unlabelled text attains 1, every safe b<1 is obstructed.

This differs from [Round 4 §2](bounds5_support.md): that argument changed an
unobserved **encoder value** while keeping coherence fixed. Here the encoder,
architecture, weights and every embedding remain fixed; only unobserved
coherence labels change. Neither argument proves an actual Qwen text at t.

### THEOREM — exact certificate characterization and query lower bound

Define the dangerous set and its initially unknown part by

\[
 H_b=\{x\in V:s(x)>b\},\qquad M_b=|H_b\setminus K|.
\]

For the unrestricted predicate class a transcript certifies the pointwise
upper bound b **if and only if**

\[
 P\cap H_b=\varnothing\quad\text{and}\quad H_b\subseteq K\cup Q. \tag{1}
\]

Equivalently, every dangerous text has a known negative label. Hence any
successful sound certificate requires

\[
 q\ \ge\ |Q|\ \ge\ |H_b\setminus K|=M_b. \tag{2}
\]

**PROOF.** An observed positive in H_b is an immediate counterexample. An
unobserved z∈H_b can be added to the minimal compatible predicate P,
invalidating b. Conversely, if all of H_b is known negative, no compatible
coherent text can score above b. Thus H_b\K ⊆ Q, giving (2) by cardinality.
Lean: `certifies_iff`, `query_lower_bound`, `budget_impossibility`;
`Certifier.sound_run_certifies` and `Certifier.sound_run_query_lower_bound`
connect universal algorithmic soundness to the actual adaptive run and count
repeated queries as calls. The theorem applies to the returned b even if
the algorithm chose it adaptively.

The bound is **tight as an oracle-query bound**: given H_b, query its unknown
members; if all answers are negative, issue b, otherwise exhibit the positive
counterexample. Thus the worst-case all-negative branch needs exactly M_b
new coherence queries. This does not charge the potentially enormous work
needed to compute H_b. It is not a requirement to query low-scoring texts.

### THEOREM — finite-sample non-identifiability, precisely stated

At any incomplete transcript satisfying the two-world hypotheses, no statistic
of that transcript can decide h*≤b with certainty over all predicates: the
same statistic has the same value in both worlds. This is the meaning of
**UNTESTABLE from finite samples** used below: not identifiable from the
specified *non-exhaustive* observations without a structural assumption.
It is not literally untestable from **every** finite sample. V is finite;
an exhaustive census, or the targeted census of H_b, settles it.

Even a probabilistic error allowance remains costly in an unstructured class.
Suppose a randomized decision procedure both certifies the all-negative
dangerous world with probability ≥1−δ and falsely certifies any world with
one dangerous positive with probability ≤δ. Couple the same random tape in
the base world and the world with exceptional text z. They can differ before
returning only if z is queried. Therefore each z is queried in the base world
with probability at least 1−2δ. Summing over its M_b choices yields expected
calls ≥(1−2δ)M_b for δ<1/2. Without the completeness requirement, always
abstaining satisfies soundness and has no such lower bound. This probabilistic
extension is a prose proof, not included in Lean.

### EMPIRICAL / STATED LIMIT — is M_b astronomical here?

The repo records **v=151,669** encoder token IDs in
[`kaggle/fluent-search/search.py`](../kaggle/fluent-search/search.py) (`W.shape[0]`),
also in `DATA/models/qwen3e06-onnx/config.json`. The LM's padded 151,936 logits
are not the encoder vocabulary. The operative cap is **511 content tokens
+ EOS**, not the architecture's 32,768-position capacity
([embedder](../src/latent_gap/embedder.py), [gate](../src/latent_gap/coherent.py)).
Dimension is 1,024. Arithmetic on these saved quantities gives

| Raw token length | log10(v^length) | Scale |
|---|---:|---:|
| 128 (earlier support-search cap) | 663.1548 | about 1.43 × 10^663 sequences |
| 511 (current content cap) | 2647.4383 | about 2.74 × 10^2647 sequences |

The sum over lengths has essentially the same scale. These are **raw token
sequence counts / upper envelopes**, not lower bounds on canonical admissible
texts, coherent texts, distinct embeddings, or H_b. Round-trip, special-token,
ASCII, field, and gibberish rules remove sequences; many may share embeddings.
Even the finite grammar has at most **4,116,282,336 Steam** and
**12,664,831,152 arXiv derivations** before rejection/deduplication, from
6N(N−1) with the saved banks N=26,193 and 45,944
([Round 5 §1.1](bounds6_cover.md)). These too are upper counts, not M_b.

Thus an unstructured *worst case* over score maps can demand a query for
essentially every admissible text (set all unknown scores above b). The raw
search envelope is astronomical. **For this fixed encoder, target and b,
the repository does not establish an astronomical lower bound on M_b.**
Dimension, vocabulary, and a handful of token-soup witnesses do not establish
how many distinct eligible texts exceed b. M_b could be zero, one, or huge;
if b is below a saved accepted witness, b is already false. Multiplying raw
text counts by a random-sphere cap probability is unjustified for this encoder
and these selected FAR targets. No new encoding was used or is proposed as
an accomplished result here. The exact conditional answer is (2).

## 7.3 Restrictions that can rescue certification, and what survives

### (a) Coherence regularity in embedding space

**THEOREM — known Boolean Lipschitz coherence collapses the query space.**
Let d(x,y)=‖E(x)−E(y)‖ and assume a *known global* L₀>0 with
|C(x)−C(y)|≤L₀d(x,y). Form the graph on all V joining pairs with
d(x,y)<1/L₀. C must be constant on every connected component.
Conversely any component-constant Boolean labelling is L₀-Lipschitz:
opposite labels lie in different components, hence at distance ≥1/L₀.
This includes exact embedding collisions. For L₀=0 all labels are equal.

**PROOF / certificate.** Along each edge a Boolean difference cannot be 1;
propagate along paths. Between components the Lipschitz inequality follows
from their separation. Replace individual texts in (1) by components:
every component touching H_b must have a known negative label. One query per
previously unlabelled dangerous component is necessary and sufficient when
all such labels are negative. A positive dangerous component refutes b.
Building and certifying the graph still requires global embedding information.
Positive anchors alone do not exclude a remote positive component.

For a real coherence score r with threshold C(x)=1 iff r(x)≥τ, a known
L₀-Lipschitz bound and observed real scores give the exclusion certificate
r(a)+L₀d(x,a)<τ for every dangerous x, with a suitable anchor a. The strict
margin matters. Binary “rejected” labels alone need not give a margin for r.
Continuity or smoothness without quantitative constants is insufficient:
on distinct finitely many embedding points, arbitrary real values admit
smooth interpolations using disjoint bump functions. Boolean functions are
continuous in that finite discrete image topology. Collisions are the only
automatic constraint if coherence must factor through E.

**Evidential status: global regularity remains UNTESTABLE from an ambiguous
incomplete sample; the saved-data falsification test is now CLOSED (§8.1).** No coherence Lipschitz constant
or margin has been certified. Observed slopes give lower, not upper, bounds
on the required constant. If a transcript admits a regular completion and an
unlabelled point has a distinct constrained neighbour, changing that point's
label alone can violate the proposed law without changing the sample; thus
those samples cannot certify that law in the unrestricted class.
If the chosen constant makes the law vacuous, it supplies no query reduction.
This does not claim that every incomplete transcript is ambiguous once the
law is *assumed*: component propagation can determine all relevant labels.

**Concrete test, executed in Round 8 (§8.1):** freeze the gate and exact-text
embedding rule, then compute the minimum opposite-label distance. The original
gate requires L₀≥81.990194 in Steam and L₀≥97.473911 in arXiv; the latter is
also the pooled lower bound. No opposite-label exact embedding collision was
found. The Opus gate, its smaller labelled sample, and the conditional FAR
component/query diagnostics are reported separately below. Surviving constants
are sample-compatible, not certified global constants. The encoder edit slopes in [Round 4 §5](bounds5_support.md) are not
measurements of coherence regularity. The attempted corpus-neighbour cover
using radii .689799 Steam / .243751 arXiv is separately **REFUTED** by
**20/20 FAR witnesses in each domain**; the directional plug-in fails on
**12/20 Steam, 9/20 arXiv FAR**. Those are specific cover failures, not a
refutation of every possible regularity constant.

### (b) Coherent text is generated by a corpus grammar

**THEOREM — complete containment, not grammar syntax alone.** If a proved
grammar containment C⊆L_G holds, an upper bound on max_{x∈L_G}s(x) bounds h*.
Equality C=L_G additionally makes membership a syntactic decision. A finite
grammar can be exhaustively evaluated; alternatively a complete cover with
certified per-cell encoder residuals gives Round 5 equation (3), formalized
by `LatentGap.Cover.covered_language_certificate`. For unit cone anchors q,
polar certificates z_i and global residuals ε_i, the bound is
min(1,max_i(‖t−z_i‖+ε_i)). The proof is polarity plus Cauchy–Schwarz in
every cell. C∩L_G is a different object from all C when containment fails.

**Evidential status: the tested containment and specific residual models are
REFUTED.** The Round-5 grammar has **187/187 saved accepted rewrite winners
outside L_G** ([Round 5 §4.3](bounds6_cover.md),
[cover_results.csv](prereg/cover_results.csv), `in_language`). This grammar
is a fixed-depth pair-operation language, not an established closure of all
natural-language composition. Adding identity corpus productions does not
by itself prove coverage of free rewrites.

Its assigned-cell radii fail on **4/384** independent checks; arXiv T draw 0
has residual **.321117 > .182099**, excess **.139018**. The later returned
judgments, not the earlier “unjudged” paragraphs, settle coherence: among
the five Round-5 queued failures (including one training failure), **4 pass
the original gate and 3 pass the Opus gate**, with a surviving refutation in
each domain ([Round 6 §6.6](bounds6_cover.md),
[`runs/cover/judge_results.jsonl`](../runs/cover/judge_results.jsonl)).
The Round-6 replacement also fails **4/384** checks (largest excess **.067322**),
but **0/4** failing texts pass either coherence gate. Its grammar-wide claim
is refuted; its coherent-only version is **not refuted by that queue**, and
is not certified by it. The coarsened common Round-5 radius surviving all
checks is likewise not certified globally.

**UNTESTABLE from a proper sample as a universal containment claim:** if an
unlabelled z∉L_G remains eligible, add it to C while preserving all labels.
If every such label is determined, this objection disappears. For a proposed
new grammar the concrete coverage test is to run its exact membership
decision on every frozen accepted text (both gates separately); a single
miss refutes containment, whereas no miss does not prove it. A grammar equal
to V covers everything but gives no smaller support without further encoder
structure. Finite grammar membership alone does not make its optimization easy.

### (c) Coherent embeddings lie in corpus support or its positive cone

**THEOREM — exact support and cone containment suffice conditionally.**
If E(C)⊆{p_i}, then h*≤max_i p_iᵀt. This also holds for the convex hull
before normalization. For the positive cone K=cone(p_i), a polar vector
z satisfying p_iᵀz≤0 gives h*≤min(1,‖t−z‖), provided
E(C)⊆K∩S^{d−1}. With a globally proved distance ε to that unit cone, add ε.
Proof: qᵀt=qᵀ(t−z)+qᵀz≤‖t−z‖, then bound the residual by ε.
These are [Round 4 §3](bounds5_support.md) and the existing Lean cone lemmas.

**Evidential status: exact empirical-support / zero-residual cone containment
is REFUTED in the specified reference geometry.** Round 4 §6 reports
optimized FAR cone upper medians **.0561 Steam / .0876 arXiv**, exceeded by
accepted witnesses on **12/20 Steam / 3/20 arXiv** targets. Positive mixtures
of *any* visible corpus vectors fail, not merely a selected neighbour average.
These are numerical certificates for the prepared reference cone; they are
not a census of a newly re-encoded corpus cone. The exact-local-corpus variant
still needs the Round-6 §6.3 census/identity treatment.

The nearest-eight tube with radii **.736558 Steam / .828220 arXiv** fails on
**21 Steam / 2 arXiv** writer records, and **15 / 2** survive Opus. That
refutes that reconstruction rule, not every existential cone tube. The small
directional extension *does* exclude every cone anchor and is **REFUTED**:
Steam/luna FAR-19, Opus **7**, scores **.104821 > .099776**, excess **.005045**;
FAR-13, Opus **8**, also exceeds it by **.000505**
([Round 5 §4.3](bounds6_cover.md), [cover_results.csv](prereg/cover_results.csv)).
Larger residual allowances remain conditional possibilities; ε=2 recovers
only the trivial bound 1 in this certificate.

**UNTESTABLE from incomplete labels** for an unrefuted proposed region B if
an unlabelled text has E(z)∉B: C=P versus P∪{z} preserves the observations
and flips containment, assuming the known positives lie in B. If E(V)⊆B has
been proved from the weights, no coherence assumption is needed instead.
“Population support” of an unspecified distribution is not the finite corpus
support. A positive mass at a rare unseen text changes it while leaving any
particular finite sample possible; see (e) below.

### (d) Coherence is a known reference-LM threshold

**THEOREM — a fixed, fully accessible LM makes the predicate identifiable.**
For a specified conditional LM Q, header, tokenizer, length and EOS convention,
define C_{Q,h}(x)=1 iff −log Q(x)/n(x)≤h, together with fixed validity rules.
There is now one prescribed predicate, not all unobserved label extensions.
Exact evaluation/enumeration settles the finite maximum; branch bounds can
sometimes prune the enumeration. A bound for this set applies to judge
coherence only if their equality or the required containment is separately
proved. If Q is itself accessible *only* through budgeted point queries,
unseen-value ambiguity has not been removed merely by renaming the oracle.

**CONJECTURE — proxy equality.** For a proposed, independently fixed threshold
h, equality of this LM-defined set with the judge-defined coherent set is a
separate hypothesis. It is not assumed in the theorem and has not been proved.

**EMPIRICAL — this is a proxy, not the actual fluent-search gate.** The saved
GPU objective was s(x)−λ meanNLL(x), with proposal restrictions; it was not
exhaustive optimization over an NLL threshold set. The final gate still used
two judges. Its NLL excludes terminal EOS and conditions on a fixed header
([`lm_launch`](../kaggle/fluent-search/search.py)); therefore the saved NLL
is not automatically a complete-string probability.
Rev 1 λ=.2 has REAL h_any medians **.759 Steam / .799 arXiv** and median
state NLL **2.24 / 1.72**, yet accepted coverage is **0/10 / 3/10** for that
λ ([Rev-1 frontier and gate tables](prereg/fluent_search.md)). These aggregates
demonstrate divergence of the objectives and gate, **not** a proof that no
NLL threshold can separate the individual labels. No equality threshold was
specified and validated in those runs.

**Concrete test for equivalence, executed in Round 8 (§8.2).** Exact-text
joins, kept separate by domain and LM/header/EOS convention, refute **every
single NLL threshold on each of the three labelled samples**: the largest
accepted NLL exceeds the smallest rejected NLL in Steam/1.7B, arXiv/1.7B and
Steam/4B. The best thresholds still make 10/1,169, 110/929 and 100/762 errors,
respectively. These are saved numeric NLLs and saved judge decisions, not a
claim about every LM, an ideal-real probability calculation, or human consensus.
No refitting followed the frozen analysis. Status: **CLOSED — sample threshold
equality REFUTED for the three tested conventions**; other proxies and global
containment claims require their own assumptions or tests.

### THEOREM — computational hardness for the general LM-threshold problem

The fixed-predicate escape from the information obstruction does **not**
generally make optimization efficient. Here is an explicit reduction, with
rational probabilities and a unit-vector encoder; no external hardness claim
about neural networks is being assumed.

Input is a Boolean formula φ in n variables. Allowed texts are n assignment
bits a followed by a flag y∈{0,1}, then deterministic EOS. Define an
autoregressive LM Q_φ: the first n bits are uniform; at the flag position,

\[
 Q_\varphi(y=1\mid a)=
 \begin{cases}3/4&\varphi(a)\text{ true},\\1/4&\varphi(a)\text{ false}.
 \end{cases}
\]

The other flag has the complementary probability; EOS has probability 1.
This is a normalized distribution with a polynomial-size, polynomial-time
conditional-probability circuit. Let τ=2^{−(n+1)}, and accept texts iff
Q_φ(a,y,EOS)≥τ. High-probability texts have mass 3·2^{−(n+2)}, low ones
2^{−(n+2)}. Thus the accepted flag is exactly y=φ(a). Every assignment
has one accepted text, so the coherent set is always nonempty. Equivalently
use a mean-NLL threshold h=log 2 when counting the n+1 non-EOS tokens, or
h=((n+1)/(n+2))log 2 when including EOS.

Take two orthogonal unit vectors e₀,e₁, E(a,y)=e_y and t=e₁. The encoder is
fixed and trivial; padding zeros embeds this construction in dimension 1024.
Then

\[
 h^*_{C_{Q_\varphi,h}}(t)=
 \begin{cases}1&\varphi\text{ is satisfiable},\\0&\varphi\text{ is unsatisfiable}.
 \end{cases}
\]

**PROOF / complexity consequence.** A text of score >1/2 passing the threshold
exists iff φ has a satisfying assignment. In the finite problem represented
by polynomial-time rational circuits (length cap unary/polynomially bounded),
existence of such a witness is in NP and this reduction makes it NP-complete;
the universal assertion h*≤1/2 is coNP-complete. Computing h* exactly, or
approximating it with guaranteed additive error <1/2, decides SAT. A uniformly
efficient algorithm that always decides this upper-bound question would imply
P=NP; an algorithm allowed to abstain or return only 1 is not ruled out.
This concerns deciding correctness or finding an exact maximum, not the cost
of checking a supplied proof. A polynomially checkable, polynomial-size proof
for every valid upper bound in this family would instead imply NP=coNP.
These are conditional computational obstructions, not an unconditional proof
that every instance is difficult.

The difficulty can also reside entirely in the encoder: keep Q uniform so
every assignment passes, and let E_φ(a)=e₁ iff φ(a), else e₀. The same gap
reduction works with a varying polynomial-time encoder circuit. Conversely,
fixed-order Markov LMs with additive finite-state scores and bounded length
admit dynamic programming over the joint state (and, where needed, a bounded
integer likelihood-cost budget). An explicit full lookup table can simply be
scanned in time linear in its table size. Representation and structure matter.

**STATED LIMIT — no hardness theorem for the actual checkpoints.** The first
reduction varies Q with φ; the second varies E. Neither constructs those
functions inside the fixed pretrained Qwen architecture/weights, the real
Steam/arXiv renderer, or the saved h setting. One fixed 511-token instance
has no growing input family on which to assert NP-hardness. Exact real
softmax/log comparisons also require a specified arithmetic model or proven
margins; the reduction deliberately avoids that issue with rational masses
and the exact scores 0 and 1. Exhaustive computation is finite under a
decidable numerical scoring contract, but its efficiency for this instance
is not established. The SAT reduction closes the **general** hardness question
without pretending to settle this checkpoint's tractability.

### (e) Other restrictions worth separating

**THEOREM — a specified hypothesis class / logical version space.** For any
class ℋ of coherence predicates, certification is equivalent to

\[
 \forall C'\in\mathcal H\cap\mathcal V(A,P),\quad C'\cap H_b=\varnothing.
\]

This is necessary and sufficient by the definition of soundness. A known
program for the frozen judge, a sound symbolic verifier, an equivalence
partition of texts, or explicit semantic rules can fix unqueried labels and
invalidate the two-world construction. The missing condition is precisely
that both worlds remain in ℋ. Merely saying “natural language” supplies no
specified ℋ or implication to verify. Small class size or low VC dimension
alone does not guarantee a short support certificate: the class consisting
of a base predicate P and its M single-dangerous-text extensions has M+1
members (VC dimension 1 on that unknown region), but the base-world branch
requires all M queries. Proof: any unqueried singleton extension survives.
Status: **STATED LIMIT** unless a concrete class and its consequences are
provided; testing such a class against saved labels can refute it, not
establish its universal applicability from an ambiguous sample.

**THEOREM — a trusted encoder-only range certificate.** If verified analysis
of the fixed weights proves ∀x∈V, s(x)≤b, then every coherence predicate
obeys b with zero C queries: H_b is empty. A finite branch-and-bound proof
with sound upper bounds and all branches closed suffices, as does a full
enumeration with exact comparisons. A single close token-soup text may
refute this *all-text* bound without being a coherent witness. Status:
**OPEN, concrete certificate test**: submit a finite proof covering every
token/grammar branch, checked against the frozen encoder and target with a
specified arithmetic error bound; any uncovered branch or unchecked upper
bound is failure to certify. No such proof has been supplied here. This is
why neither Round 4 nor Round 7 excludes weight-based verification.

**THEOREM — known minimum mass yields probabilistic support exclusion.**
Fix b before sampling. Suppose iid texts are drawn from a fully specified law and every possible
coherent dangerous text has mass ≥p_min>0. If any violation exists, m draws
miss it with probability at most (1−p_min)^m. Thus issuing b after no
violations gives false-certificate probability ≤δ for
m≥log δ/log(1−p_min), rounded up (0<p_min<1). This is a confidence guarantee,
not a zero-error certificate. Without a mass floor, adding a coherent
dangerous atom of mass ε yields miss probability (1−ε)^m arbitrarily close
to 1. This proves the finite-sample support limitation even when observed
frequencies fit perfectly. Status: **UNTESTABLE from the present samples**
as a universal mass-floor assumption; the adaptive rewrites are not iid
draws from the grammar/corpus law in any case.

A complete-text LM mass floor p_min bounds cardinality by 1/p_min, but not
the orientation of those embeddings: a singleton can score 1. Round 4's
illustration h=2.5 nats and L=511 permits log cardinality about **1278**,
versus about **51.4** nats for the d=1024 cap at cosine .3; the proposed
union bound is vacuous even under its independent-target assumptions.
An NLL definition omitting EOS instead uses a separate prefix bound for
each length. [Round 5 §4.4–4.5](bounds6_cover.md) also records **7/8** primary
future-corpus cells failing the beta transfer criterion and **12/12** search
cells failing. This **REFUTES those fitted transfer models**, not the valid
conditional mass-floor or iid-maximum theorems. Finite validation at q999
does not license an arbitrarily extreme support extrapolation.

## Open

Here CLOSED refers to the specified mathematical or empirical question, not
to a claim that the actual coherent maximum has been computed. A concrete
test is an unexecuted protocol, not a request to run models in this round.

| Item | Disposition |
|---|---|
| Does h* exist? | **CLOSED — theorem.** Attained for nonempty finite C; empty case handled explicitly (§7.1). |
| Existing witness lower bounds, including Opus | **CLOSED — local audit.** All domain/arm tables above; Opus REAL is unassessed, not rejected. |
| Unrestricted coherence non-certifiability | **CLOSED — theorem.** Safe known positives plus an unlabelled high-score text give two worlds with fixed E (§7.2). |
| Exact coherence-query complexity | **CLOSED — theorem.** Exactly M_b unknown dangerous labels on the negative branch; adaptive-query formalization included. |
| Does b<1 always imply obstruction? | **CLOSED — false.** Need an actual text above b; S_E(t) may be <1. |
| Is M_b astronomical for our Qwen FAR targets? | **STATED LIMIT.** Raw space is enormous; saved quantities do not lower-bound the size of its high-score subset. |
| Coherence regularity | **CLOSED — saved-data falsification test (§8.1).** Original gate refutes L₀<81.990194 Steam / <97.473911 arXiv (pooled <97.473911), with no exact opposite-label collision. Surviving constants save zero queries on the saved original-gate FAR graph. Opus admits conditional cached query reduction at larger constants; no global regularity law is certified. |
| Grammar containment / residuals | **CLOSED for tested claims.** 187/187 free winners missed; Round 5 coherent failures under both gates; Round 6 grammar claim fails but its coherent-only restriction has no accepted failure in the queue. Global surviving variants are a **STATED LIMIT**. |
| Exact reference-corpus cone and small directional tube | **CLOSED — REFUTED**, including Opus witnesses for the directional tube. The different exact-local-corpus version awaits the already specified Round-6 census. |
| Judge coherence equals a reference-LM threshold | **CLOSED — REFUTED on all three tested samples (§8.2).** No separating threshold for Steam/1.7B, arXiv/1.7B or Steam/4B; best errors 10/1,169, 110/929, 100/762. This does not refute every other LM/convention or establish global containment. |
| Complexity of LM-threshold maximum | **CLOSED for the general circuit problem.** SAT reduction / coNP-complete upper-bound decision. **STATED LIMIT** for fixed pretrained Qwen. |
| A nontrivial encoder-only upper certificate | **CONCRETE TEST.** A complete finite branch proof, checked with sound numerical bounds for the frozen map; no such artifact supplied. |
| Statistical tails versus support | **CLOSED — theorem / STATED LIMIT.** Known mass floor can give probabilistic exclusion; samples without it cannot rule out arbitrarily rare violations. |
| Lean compilation | **CLOSED — compiled on Kaggle 2026-10-29** (kernel latent-gap-lean-check v7, pinned Mathlib, `lake build` exit 0, 0 errors). All 14 `LatentGap.Impossibility` theorems print only `[propext, Classical.choice, Quot.sound]`; no `sorryAx`. |
| Can an actual coherent text approach every FAR target? | **STATED LIMIT.** These results neither prove nor disprove it. A witness settles a specified achievable score; exclusion requires one of the complete certificates above. |

## Formal scope and verification performed

[`Impossibility.lean`](../lean/LatentGap/Impossibility.lean) defines the attained
finite maximum, witness inequality, compatible predicates, exact transcript
certificate, dangerous-set cardinality bound, and finite adaptive certifiers.
It is imported by [`LatentGap.lean`](../lean/LatentGap.lean). Every theorem has
an axiom-print command; there are no `sorry`, `admit`, or added axioms. The
token counting, probability extensions, regularity graph, and complexity
reductions are prose proofs, not claimed Lean formalizations. Compiled on Kaggle (2026-10-29, `lake build` exit 0); every axiom print lists only
`propext`, `Classical.choice` and `Quot.sound`.

Verification in this round is limited to reading existing local evidence,
recomputing displayed CSV/JSON aggregates with Python's standard library,
checking the certificate equivalence/two-world/query-count claims on all
405 four-text transcript/threshold cases and the rational LM construction
on all 276 Boolean truth tables with one to three variables,
checking adaptive-path agreement on 37,632 small tree/world pairs,
reviewing the new Lean against the locally available Mathlib source signatures,
and checking the patch. No result file was changed. Earlier reports remain
unchanged apart from a one-line pointer at the end of Round 6.

For non-mathematicians: **we cannot prove that no bound exists**. With a fixed
length limit there is a best coherent text for each target, and similarity
never exceeds 1. What we can prove is that an unchecked text prevents a
guaranteed tighter answer whenever it could beat the proposed bound and its
coherence is unconstrained by what we already know. One accepted example
proves that a score can be reached; ruling out every better example requires
checking them all or proving a rule that covers them all. Our experiments
have defeated several proposed rules. Round 8 finds opposite saved decisions
on almost identical descriptions and shows that no single saved NLL threshold
matches all decisions in any of the three tested settings. The remaining
regularity assumptions still need proof; the Opus sample allows conditional
query savings, while the original-gate FAR cache gains none. These results
have not shown that coherent text can get arbitrarily close to every far target.


## Round 8: saved-data tests of regularity and NLL equivalence

### 8.0 Frozen rules — recorded before either outcome

**Freeze UTC: `2026-10-29T08:44:05.668499+00:00`.** Analysis code:
[`scripts/bounds/round8_saved.py`](../scripts/bounds/round8_saved.py),
SHA-256 **`be85e291684a02b7c0fbc1dce93dd1793943f461ada19cf09075fe9bca12997a`**. The same timestamp and hash are
in [`runs/bounds8/freeze.json`](../runs/bounds8/freeze.json). Only synthetic
unit checks were run before this freeze; no opposite-label distance,
NLL separation, optimal threshold or AUC was computed. Schema/provenance
inspection was performed first. No rule will be refitted after outcomes.

**Common rules.** Work offline, using saved files only; no ONNX, encoder,
LLM, download or network. Set OMP/BLAS thread limits to 2. Stream JSONL/CSV,
mmap NPY files, use float64 distance blocks of 64 rows. Open new outputs
exclusively in `runs/bounds8/`; never overwrite historical results. Preserve
an input hash manifest and an exact-text audit. Match UTF-8 text byte-for-byte
via verified SHA-256, with no stripping, normalization or fuzzy matching.
Deduplicate within domain; a pooled statistic includes cross-domain pairs
under the additional interpretation of one combined embedding-space gate.

**8.1 rule.** Collect exact-description Haiku `claude-haiku-5-5` and luna
`gpt-6-luna` ratings from all saved seed-rewrite tuning/test sets, fluent-search
runs with judgments, the cover queue, coherence calibration, and compatible
individual ratings in `*/coherent_search/*/items.jsonl`. Other judge identities
are ignored. A positive requires both scores >=7; either score <7 is a
negative; missing or failed decisions are unresolved. Repeated scores on the
same side of 7 agree; any judge with scores on both sides makes that text
unresolved, with the conflict retained in the audit. This is the user's
score-defined gate: do not manufacture negative labels from missing ratings
or invalidity flags. Retain eligibility flags and check the minimizing pairs'
eligibility when interpreting the original admissible-language claim.
The Opus sensitivity requires a saved Opus score and applies the three-judge
conjunction; absent Opus is excluded, including otherwise settled negatives.

Use only fingerprint `6ae3845eab14497b` local caches in fluent-search and
support-search, and the local vector checkpoints `cover_embeddings.csv` and
`cover_r6_embeddings.csv`. Pilot/grid inline local vectors may supply exact-text
matches to genuine original-gate ratings, but their normalized-pitch judgments
and different rubric cannot label the rendered source descriptions. Count
these exclusions. Do not substitute prepared corpus/GPU/BGE embeddings.
Validate 1,024 finite coordinates and norm within 1e-5 of 1; normalize once in
float64 to fix the unit-vector distance convention. Multiple vectors for the
same text must agree within 1e-12 after normalization or stop.

For Steam, arXiv and pooled, compute **all opposite-label Euclidean distances**
and report the minimum, its two full texts/hashes/provenance and 1/d_min.
Every L0 < 1/d_min is refuted; equality and larger constants are only
sample-compatible. Test exact vector identity separately; an opposite-label
collision refutes every finite constant. No decimal rounding is used in tests.
Count same-label graph edges at radii d_min and d_min/2 (L_min and 2 L_min)
to distinguish lack of certification from a literally edgeless regularity law.

For the saved FAR targets k=1..20 in each domain, test the pre-existing bounds
b=ceiling*+.03 and ceiling*+.05. Under each of those **assumed** constants,
count saved, eligible, embedded, as-yet-unsettled dangerous texts within a
negative anchor's strict radius; also count positive-ball implications,
known positive violations, and overlaps of positive/negative balls (which
would refute the assumed constant via an intermediate cached text). This is
a conditional, finite-cache query-saving diagnostic, never a census of H_b
or proof of the global law. Report both domain gates separately. No hypothetical
query saving can rescue a bound already exceeded by a known positive.

**8.2 rule.** Use only `fluent-search/full` and `fluent-search/steam4b` for this
test, joining their own `judgements.jsonl` and `candidates.csv` to the exact
texts in `kaggle-out/fluent_gpu*.jsonl` (generated, pool and final states).
The CSV omits text, so reconstruct and verify its hash from these pools.
Recover a generated state's omitted CSV NLL only from a saved occurrence of
that identical text. Labels for 8.2 remain run-local. One observation per
(domain, run, exact text), never one per target, lambda or repeated state.
Exclude unresolved labels and missing/nonfinite NLL; report their counts.
Repeated finite NLLs differing by >1e-4 nats/token are excluded as numerical
conflicts; otherwise use the smallest saved value, independently of labels,
and retain all distinct values in the audit. No new NLL is evaluated.

Keep three strata separate: Steam/full and arXiv/full use
`Qwen/Qwen3-1.7B-Base`; Steam/steam4b uses `Qwen/Qwen3-4B-Base`. The tokenizer
is Qwen3-Embedding-0.6B's, with at most 511 text tokens, mean NLL in nats per
text token, header conditioned and terminal EOS excluded. The fixed headers
are `A video game page on the Steam store:\n` and
`arXiv computer science paper. The title is followed by the abstract, a paragraph that summarises the problem, the approach and the contribution.\n`.
The run metadata, preregistration and `lm_launch` establish this convention;
early smoke headers/checkpoints are never pooled into 8.2.

With both classes present, a threshold accepting NLL<=h exists on the sample
**iff max accepted NLL < min rejected NLL**. Report both extremes and their
texts. Enumerate every distinct NLL threshold plus the all-reject choice;
minimize FP+FN, treating ties as a block and selecting the smallest h on equal
error. This is the requested descriptive best threshold, not a held-out fit
or an independently validated rule. AUC uses **-NLL** to predict acceptance,
with half credit for equal NLL. If no separator exists, every single threshold
is refuted on this sample; otherwise report sample compatibility only.
One-class strata get no separability or AUC claim. No global proxy equality
is inferred from a successful sample.


**Pre-outcome implementation correction, UTC `2026-10-29T08:45:13.842097+00:00`.** The first
attempt stopped during source inventory on a pilot record with `best: null`,
before entering `analyse` or writing any outcome. Treat null best records as
missing embeddings. Also enforce the already specified FAR k=1..20 restriction
explicitly. The statistical rules above are unchanged. The corrected code
SHA-256 is **`b2e263fd42f7bfdb09693a2bfc264630af50b6669833933caf1ebd7de0a9c142`**, frozen in
[`freeze-v2.json`](../runs/bounds8/freeze-v2.json) before restarting. The first
code is preserved as [`round8_saved_v1.py`](../runs/bounds8/round8_saved_v1.py)
with its original freeze; synthetic checks pass again.


**FAR graph diagnostic freeze, UTC `2026-10-29T08:49:29.834736+00:00`.** After the primary
outcomes, before computing graph components, complete the already frozen
query-saving question using the same data, labels, gates, radii d_min and
d_min/2, FAR targets and bounds; **no parameter, threshold or gate is refitted**.
Direct-anchor counts do not test propagation through unknown intermediates.
[`round8_graph.py`](../scripts/bounds/round8_graph.py), SHA-256
**`4a4357f017e487e494149b34888ed079653b11a2bdb084acdbb5c9aff90b7537`**, therefore forms the entire graph on
saved embedded texts with unambiguous saved eligibility=True, separately by
domain. Edges use strict float64 Euclidean distance < radius. The base code
remains `b2e263fd42f7bfdb09693a2bfc264630af50b6669833933caf1ebd7de0a9c142`. Freeze artifact:
[`graph-freeze.json`](../runs/bounds8/graph-freeze.json).

For each graph report vertices/components, every mixed-label component and
a spanning-tree path witnessing its inconsistency. A mixed component refutes
that proposed global constant even if its intermediate texts are unlabelled.
For each FAR bound compare cached unknown dangerous texts M_cache with unknown
dangerous components; known-negative components require no new query, while
positive dangerous components refute the bound under the assumption.
Report savings only conditionally, and never count a mixed graph or a refuted
bound as a successful certificate. If the largest sample-compatible radius
has no savings, smaller radii cannot add any. Zero savings on this cache does
not prove zero savings on the full unknown V. Verify the primary input hashes
before loading; write a new graph result, preserving all primary outcomes.

### 8.1 Coherence regularity: results and source accounting

**The original gate is sharply discontinuous at the scale of these saved
embeddings.** The minimum pairs are single-word variants with opposite saved
Haiku decisions. This refutes small Boolean Lipschitz constants, not every
finite constant and not linguistic coherence as a human concept. All four
distinct minimizing pairs below have saved validity=True and gibberish=False.

| Gate | Domain | Accepted / rejected embedded texts | d_min | Refuted L₀ range | Labelled-sample-compatible range |
|---|---|---:|---:|---:|---:|
| Original | steam | 263 / 2163 | 0.012196580463 | 0 ≤ L₀ < 81.990194140 | L₀ ≥ 81.990194140 |
| Original | arxiv | 246 / 1355 | 0.010259155447 | 0 ≤ L₀ < 97.473910512 | L₀ ≥ 97.473910512 |
| Original | pooled | 509 / 3518 | 0.010259155447 | 0 ≤ L₀ < 97.473910512 | L₀ ≥ 97.473910512 |
| +Opus | steam | 62 / 22 | 0.500055089292 | 0 ≤ L₀ < 1.999779667 | L₀ ≥ 1.999779667 |
| +Opus | arxiv | 56 / 17 | 0.497127220691 | 0 ≤ L₀ < 2.011557522 | L₀ ≥ 2.011557522 |
| +Opus | pooled | 118 / 39 | 0.497127220691 | 0 ≤ L₀ < 2.011557522 | L₀ ≥ 2.011557522 |

The thresholds in the table are rounded for display; the operative endpoints
are the unrounded reciprocals in [`results.json`](../runs/bounds8/results.json).
Equality is allowed because the graph uses **strict** distance <1/L₀.
The pooled minima are the arXiv pairs; pooling presumes a single combined
embedding-space predicate across the two domain prompts. **No exact
opposite-label embedding collision was found**, either within or across
domains, under either gate. No conflicting repeated scores crossed 7.

**Minimizing pairs.** Full texts, all rating sources and embedding paths are
in `results.json`; exact texts are also expanded below.

| Gate / domain | Positive text | Negative text | Ratings (Haiku, luna, Opus) |
|---|---|---|---|
| Original / Steam | `5d552bcdeea3…`, “truly in the game world” | `031b93c44bd7…`, “really in the game world” | (7,9,—) vs (6,missing,—) |
| Original / arXiv | `c1d5cb47ace2…`, “We start by providing” | `1d09226a7595…`, “We begin by providing” | (7,7,—) vs (6,missing,—) |
| +Opus / Steam | `88b531b2c070…`, Sonnet FAR-20, Slim Ball Classics | `cd3ba9df4443…`, luna FAR-20, ball workshop | (7,7,8) vs (7,7,5) |
| +Opus / arXiv | `796ccd5b5131…`, Sonnet FAR-15, Triangle Closing for Recruiter Search | `2959579a0fb0…`, luna FAR-15, Triangles Conquer Recruiter via LI | (8,9,9) vs (7,8,6) |

The original pairs come from `fluent-search/full/judgements.jsonl` and its
exact-text GPU pools. Both Opus pairs are original accepted seed-rewrite test
winners, joined to `test/opus_check.jsonl`; their rejection is specifically
the additional Opus decision. Skipped luna on Haiku rejection is **settled
negative**, not missing-decision exclusion.

**Sources actually used.** Counts are per-source distinct (domain,text)
records; overlapping sources must not be added as independent observations.
“Settled” uses the original gate after exact-text rating joins; “unresolved”
includes unjudged candidates. Every source below supplies embedded settled
labels. Sources and hashes are in the
[`input manifest`](../runs/bounds8/input_manifest.json); all candidate label
and exclusion states are in [`text_audit.jsonl`](../runs/bounds8/text_audit.jsonl).

| Source under DATA/runs | Distinct texts | Settled | With local embedding | Settled, no local embedding | Unresolved |
|---|---:|---:|---:|---:|---:|
| `coherent_calibration` | 200 | 200 | 50 | 150 | 0 |
| `cover/queue` | 9 | 9 | 9 | 0 | 0 |
| `fluent-search/cputest` | 2 | 2 | 2 | 0 | 0 |
| `fluent-search/full` | 3166 | 2106 | 2106 | 0 | 1060 |
| `fluent-search/smoke` | 69 | 62 | 62 | 0 | 7 |
| `fluent-search/smoke2` | 49 | 41 | 41 | 0 | 8 |
| `fluent-search/smoke6` | 68 | 68 | 68 | 0 | 0 |
| `fluent-search/smoke7` | 64 | 43 | 43 | 0 | 21 |
| `fluent-search/smoke8` | 38 | 38 | 38 | 0 | 0 |
| `fluent-search/smoke9` | 43 | 43 | 43 | 0 | 0 |
| `fluent-search/steam4b` | 1300 | 769 | 769 | 0 | 531 |
| `fluent-search/steam4b-smoke` | 33 | 31 | 31 | 0 | 2 |
| `fluent-search/steam4b-smoke2` | 22 | 22 | 22 | 0 | 0 |
| `fluent-search/steam4b-smoke3` | 30 | 23 | 23 | 0 | 7 |
| `fluent-search/steam4b-smoke4` | 25 | 16 | 16 | 0 | 9 |
| `seed-rewrite/test` | 2144 | 552 | 552 | 0 | 1592 |
| `seed-rewrite/tune-v1` | 358 | 126 | 126 | 0 | 232 |
| `seed-rewrite/tune-v2` | 429 | 92 | 92 | 0 | 337 |

After deduplication: **4,027** embedded settled original-gate texts
(2,426 Steam; 1,601 arXiv), **151 settled texts excluded for missing local
embeddings** (75 Steam; 76 arXiv), and 14,017 unresolved candidate texts
(3,597 embedded; 10,420 without saved embeddings). The 151 comprise 150
coherence-calibration texts and one arXiv Sonnet coherent-search text with a
decisive saved luna rejection. No model was run to fill those gaps.

The +Opus sample is **157 texts, all embedded**: 148 seed-rewrite test
winners plus the nine cover-queue texts (84 Steam; 73 arXiv). No Opus score
exists for the remaining candidates in this inventory. In particular,
original rejections without Opus were not silently added to this sensitivity
sample. The cover queue includes the five Round-5 and four Round-6 texts;
its DATA and repository copies were checked identical and counted once.

Embedding sources are the shared local fingerprint `6ae3845eab14497b`
NPY caches (`fluent-search/verified_cache`, with `support-search/verified_cache`
for exact matches), plus `docs/prereg/cover{,_r6}_embeddings.csv`. The Round-6
fit checkpoint is a duplicate subset, so it is not another sample. Prepared
corpus vectors and GPU/BGE arrays are excluded because they do not establish
the same saved local-encoder mapping. Corpus provenance is not a gate label.

**Other inspected sources and exclusions.**

- `*/coherent_search/*/items.jsonl`: all saved runs inspected, including
  Sonnet search/free/random, gateway-luna and timing/smoke variants. Their
  judge configurations mainly use Sonnet→luna or Sonnet→Haiku. Sonnet
  scores cannot substitute for Haiku; exact required-judge scores were retained
  when available. Only one settled original decision resulted, and its exact
  local embedding is absent, as counted above. Per-run counts are in `results.json`.
- Fluent `smoke3`, `smoke4`, `smoke5` and `tune-seeds-{steam,arxiv}` have GPU
  pools but no own saved `judgements.jsonl`; they supply no additional settled
  gate labels. Seed texts that recur in judged rewrites are already joined by
  exact text. Early smoke LM/header conventions are used for **8.1 geometry
  only**, never mixed into 8.2 NLL strata.
- Pilot/grid: 6,081 item records inspected; 6,071 have inline best embeddings
  and ten do not. Their judgments rate normalized pitches with neighbours and
  a different rubric, and use gpt-6.1-sol rather than the original luna gate.
  Even a saved Haiku score <7 on that pitch is not a rejection of the embedded
  rendered description. Hence no pilot/grid **judgment** is substituted.
  The inventory is explicit below; some runs have no judgment files at all.

| Pilot/grid source | Item records | Inline best embedding present |
|---|---:|---:|
| `arxiv/grid-claude-sonnet-5-5` | 650 | 650 |
| `arxiv/grid-gpt-4o-0806` | 450 | 450 |
| `arxiv/grid-gpt-6-luna` | 650 | 650 |
| `patents/grid-claude-sonnet-5-5` | 442 | 442 |
| `patents/grid-gpt-6-luna` | 450 | 450 |
| `steam/grid-claude-haiku-5-5` | 450 | 450 |
| `steam/grid-claude-opus-5-5` | 450 | 450 |
| `steam/grid-claude-sonnet-5-5` | 630 | 630 |
| `steam/grid-gpt-4o-0806` | 430 | 424 |
| `steam/grid-gpt-6-1-sol` | 450 | 450 |
| `steam/grid-gpt-6-luna` | 630 | 630 |
| `steam/pilot` | 399 | 395 |

No patents stratum has settled original-gate labels on the exact embedded
descriptions, so patents is excluded rather than assigned negatives.

**Does a surviving constant reduce FAR queries?** The answer depends on the
gate and on whether the statement is conditional or certified. The complete
cached graph uses only unambiguously eligible texts: 3,904 Steam and 2,800
arXiv vertices. These are **not all V**. Component results and explicit
opposite-label path witnesses are in
[`graph_results.json`](../runs/bounds8/graph_results.json). The graph rule was
additionally frozen before that computation, transparently after the primary
pair/NLL results; neither the radii nor any label/threshold rule was changed.

| Gate / domain | Assumed L₀ | Radius | Components / vertices | Mixed-label components |
|---|---:|---:|---:|---:|
| original / steam | 81.990194140 (1×L_min) | 0.012196580 | 3848 / 3904 | 0 |
| original / steam | 163.980388280 (2×L_min) | 0.006098290 | 3895 / 3904 | 0 |
| opus / steam | 1.999779667 (1×L_min) | 0.500055089 | 419 / 3904 | 5 |
| opus / steam | 3.999559334 (2×L_min) | 0.250027545 | 1297 / 3904 | 0 |
| original / arxiv | 97.473910512 (1×L_min) | 0.010259155 | 2784 / 2800 | 0 |
| original / arxiv | 194.947821025 (2×L_min) | 0.005129578 | 2794 / 2800 | 0 |
| opus / arxiv | 2.011557522 (1×L_min) | 0.497127221 | 321 / 2800 | 2 |
| opus / arxiv | 4.023115043 (2×L_min) | 0.248563610 | 938 / 2800 | 0 |

**Original gate:** at the largest surviving radius (d_min), same-label
neighbours do merge, but **no unknown dangerous texts merge with each other
or a labelled component** for any FAR target at either frozen margin. Across
the 20 separate target questions, M_cache→unknown dangerous components is
41→41 and 14→14 for Steam (+.03, +.05), and 2→2 and 0→0 for arXiv.
All of those nonempty cached unknown sets occur on targets whose proposed
bound already has a known positive violation (Steam 8/20 and 5/20; arXiv
2/20 and 0/20). Smaller radii cannot create edges. Thus **every surviving
original-gate L₀ gives zero query reduction on this saved FAR graph**.
This is a vacuous reduction on the cache, not a proof that regularity is
globally vacuous or that the real H_b is empty on the remaining targets.

**+Opus:** the pairwise lower endpoints themselves are additionally
**refuted** by unlabelled intermediate texts: the d_min graphs contain five
mixed components in Steam and two in arXiv. Opposite labels connected by a
path of edges <1/L₀ cannot admit any Boolean Lipschitz completion.
For example, the first stored Steam path has three edges, each ≤.473107,
and the first arXiv path has two edges, each ≤.476084, below their respective
~.5 radii. The path hashes and distances permit direct inspection.

At the **predeclared** 2×L_min (3.999559334 Steam, 4.023115043 arXiv),
there are no mixed components. These constants, and all larger constants,
admit a completion on the eligible cached cloud; this is not a proof over V.
The interval between the pairwise lower endpoints and these tested constants
is not exhaustively classified by this two-radius graph diagnostic. At
2×L_min there is genuine **conditional** query compression:

| +Opus domain | Bound | Targets without a positive dangerous component | Sum M_cache → sum new component queries | Queries saved | Targets with savings |
|---|---|---:|---:|---:|---:|
| steam | ceiling* + 0.03 | 13/20 | 120 → 27 | 93 | 12 |
| steam | ceiling* + 0.05 | 17/20 | 142 → 37 | 105 | 16 |
| arxiv | ceiling* + 0.03 | 18/20 | 90 → 18 | 72 | 13 |
| arxiv | ceiling* + 0.05 | 20/20 | 86 → 18 | 68 | 13 |

These sums count separate target questions; they are not a joint optimal
query budget, and a component query still has to return negative. Seven/three
Steam and two/zero arXiv targets (+.03/+.05) already violate their bounds
under the saved Opus positives, so are excluded from this savings table.
No new labels are obtained by this analysis: it shows what a **proved**
constant could save. No global constant or complete embedding graph has been
certified, so **neither gate gains a certified upper bound** here. The much
weaker Opus lower bounds reflect its small selected sample, not evidence
that Opus is intrinsically smoother.

<details>
<summary>Exact minimizing texts (positive then negative)</summary>

**original / steam**

positive_pair: `5d552bcdeea363385e88b9630e9b2e8d1d95f3380061da24c4f756158e769b10`

```text
Genres: Action, Adventure, Puzzle, Simulation. Tags: Action-Adventure, Adventure-Quest, Puzzle-Adventure. About: This is a first-person adventure game where you will be able to explore a beautiful world and solve a variety of puzzles to progress through the story. The game is developed with the help of a team of talented artists, programmers, and sound designers who have worked on many successful games in the past. The story is based on a classic adventure game, but with a modern soundtrack that will keep you engaged and entertained for hours on end. The soundtrack is a mix of orchestral and electronic music that will enhance the atmosphere of the game and make you feel like you are truly in the game world.
```

negative_pair: `031b93c44bd71a05f14f8288e2cf47bc5b8ea3922e9888901b6977322648ad7f`

```text
Genres: Action, Adventure, Puzzle, Simulation. Tags: Action-Adventure, Adventure-Quest, Puzzle-Adventure. About: This is a first-person adventure game where you will be able to explore a beautiful world and solve a variety of puzzles to progress through the story. The game is developed with the help of a team of talented artists, programmers, and sound designers who have worked on many successful games in the past. The story is based on a classic adventure game, but with a modern soundtrack that will keep you engaged and entertained for hours on end. The soundtrack is a mix of orchestral and electronic music that will enhance the atmosphere of the game and make you feel like you are really in the game world.
```

**original / arxiv**

positive_pair: `c1d5cb47ace2e15d3e38f4eff4e9b0cfee7564013b05a1d5499b11f5e2a72a8e`

```text
Title: A Survey on Quantum Machine Learning Algorithms and Applications. Abstract: Quantum machine learning algorithms have emerged as a promising research area in recent years. This paper reviews recent advances in quantum machine learning research. We start by providing a brief overview of quantum machine algorithms. We then review quantum optimization algorithms. Next, we review quantum machine and hybrid machine learning systems. We also review quantum and hybrid quantum machine systems.
```

negative_pair: `1d09226a759531714efc93be5029b17402b4967ea068dc2d62f18e432cfed331`

```text
Title: A Survey on Quantum Machine Learning Algorithms and Applications. Abstract: Quantum machine learning algorithms have emerged as a promising research area in recent years. This paper reviews recent advances in quantum machine learning research. We begin by providing a brief overview of quantum machine algorithms. We then review quantum optimization algorithms. Next, we review quantum machine and hybrid machine learning systems. We also review quantum and hybrid quantum machine systems.
```

**opus / steam**

positive_pair: `88b531b2c070c21e60296d43cd0d43fc68f0097548ef2e53e95cbdb75fa3ae27`

```text
Genres: Sports,Simulation,Casual. Tags: Sports,Ball Games,Crafting,Simulation,Building,Casual,Singleplayer. About: Slim Ball Classics is a relaxed crafting and sports simulation about designing, assembling and testing handmade balls. Start in a small workshop with a few basic materials and learn which ones suit each kind of ball. Every ball has its own construction, features and advantages, and you will need to balance strength, weight and bounce to get a perfect result.

Choose from a collection of classic designs, each with a history and a traditional use. Work through the assembly step by step, then try optional upgrades and extra parts to improve performance. Watch for construction complications, because a poor choice of material can cause energy loss and ruin a ball's flight.

Test your creations on the practice field, where every ball behaves according to its materials and build. Compare your results against earlier designs and see how each new version improves on its predecessors.

Features:
- Craft 16 different balls, from traditional handmade types to modern functional designs
- Collect and combine a wide range of materials
- Learn the strengths and properties of each material
- Unlock new workshop tools and a power-holder station
- Test balls in varied conditions and accelerate your d
```

negative_pair: `cd3ba9df4443dc276ed11d7b630e36ba17410b7e6cf53cae5e96075b241d7871`

```text
Genres: Simulation,Sports,Casual. Tags: Physics,Ball,Crafting,Sandbox. About: Step into a hands-on workshop devoted to the design, history, and practical use of balls. Choose materials, compare their strengths and properties, and build equivalent balls to see how construction changes their performance. Begin with classic forms, including traditional horned designs, or try a slim B-ball and the distinctive Baxter model. Each design has trade-offs: handmade assembly and familiar construction methods can offer useful qualities while bringing their own complications.

Create and compare a collection of up to 16 balls, then test them with a 250 W power-holder. Explore how regional traditions and real-world applications shape material choices, from everyday play to specialized uses. Study the features that made earlier designs popular, refine your own functional builds, and weigh each advantage against the demands of production. Experiment, test, and uncover how thoughtful design turns simple materials into objects made to move.
```

**opus / arxiv**

positive_pair: `796ccd5b513152ce26273839970606fd47ac69048044a2fdd533e8b8eac26384`

```text
Title: Triangle Closing for Recruiter Search on LinkedIn: Location-Aware Candidate Retrieval Across Cities and Metropolitan Areas. Abstract: Recruiters on LinkedIn search for candidates across many cities and metropolitan areas, yet location and job-title fields are noisy and inconsistent. We study recruiter search as link prediction on a heterogeneous graph of recruiters, candidates, jobs, companies, and locations, using triangles as the central structural signal. Our method scores a recruiter-candidate pair by counting closed triples through shared employers, industries, and areas. We normalize location strings and job-title variants and feed the triangle counts into a ranking model. On a large offline dataset of recruiter interactions, the approach improves ranking quality over text-only and embedding baselines, especially for rare titles and small areas. Ablations show that location-aware triangles account for most of the gain.
```

negative_pair: `2959579a0fb090f246314d5890b48f20d03ff3c5b64ff149b2842f363005aac2`

```text
Title: Triangles Conquer Recruiter via LI. Abstract: We study triangle-based candidate discovery for recruiters on LinkedIn (LI). Recruiters, candidates, and organizations form a heterogeneous graph, where a triangle represents mutually reinforcing connections among these entities. We score recruiter–candidate links using local triangle patterns alongside profile and employment signals, then rank candidates for outreach. This approach favors candidates supported by multiple paths through the network while retaining useful evidence from sparse neighborhoods. Recommendations can be interpreted through the shared connections and organizational context contributing to each score. The source draft contains no recoverable dataset, benchmark, or quantitative findings; accordingly, this description makes no performance claims and frames the contribution as a graph-based method and evaluation protocol for recruiter–candidate interactions.
```

</details>

### 8.2 NLL threshold equivalence: results

**Every single acceptance threshold is refuted on each of the three saved
samples.** Each accepted extreme has a higher NLL than a rejected extreme,
so no h can accept all positives and reject all negatives. This tests equality,
not one-sided containment. The NLL convention and the direction of AUC were
frozen; early smokes, different LMs and domains are never pooled.

| Domain / run / LM | Accepted / rejected | Max accepted NLL | Min rejected NLL | Separable? | Minimum violations (FP + FN) | Best h | AUC of −NLL |
|---|---:|---:|---:|---|---:|---:|---:|
| steam / full / 1.7B | 10 / 1159 | 2.240147352 | 1.014606118 | No | 10/1169 (0 + 10) | < 1.014606118 (all reject) | 0.720880 |
| arxiv / full / 1.7B | 112 / 817 | 2.351627350 | 0.825668037 | No | 110/929 (0 + 110) | 0.794605613 | 0.807593 |
| steam / steam4b / 4B | 105 / 657 | 2.268039703 | 0.862740874 | No | 100/762 (6 + 94) | 1.043018103 | 0.767196 |

The best thresholds are descriptive in-sample minima, not a proposed
validated substitute gate. Class imbalance matters: rejecting everything
already makes only ten errors in Steam/full. AUC above .5 shows ranking
information, not threshold equality. The corresponding AUCs using **+NLL**
as the positive score would be 1 minus those shown.

| Stratum | Distinct exact texts | Unresolved labels excluded | Settled labels | Settled, missing NLL | NLL conflicts excluded | Used |
|---|---:|---:|---:|---:|---:|---:|
| steam/full | 1601 | 429 | 1172 | 0 | 3 | 1169 |
| arxiv/full | 1565 | 631 | 934 | 0 | 5 | 929 |
| steam/steam4b | 1300 | 531 | 769 | 0 | 7 | 762 |

There were no conflicting settled judge labels. The 15 NLL-conflict texts
had repeated saved values differing by >1e-4 nats/token, so the frozen rule
excluded them; all values and labels remain in
[`nll_audit.json`](../runs/bounds8/nll_audit.json). Their discrepancies range
from about .000116 to .001835 nats/token. None of the six separating-extreme
witnesses has an NLL conflict. Missing CSV NLL on generated states was joined
from identical pool/final texts, leaving **zero settled texts without NLL**.
Numerical values are saved fp16-model evaluations, not interval certificates
for ideal exact log probabilities.

**Contradicting extremes:** the exact texts below, their complete hashes,
ratings and NLL values identify an explicit threshold impossibility witness
for each stratum. The full audit preserves all retained text-level records.

<details>
<summary>Exact maximum-accepted and minimum-rejected NLL texts</summary>

**steam / full**

max_accepted: NLL=2.240147352219; haiku=[7], luna=[8];
SHA-256 `a95a372191ece8846d7c3682629444ba0404093bad9f06790a1dc64368ac40aa`.

```text
Genres: Action, Adventure, Puzzle, Simulation. Tags: Action Adventure Puzzle Simulation. About: Play as a hero in a post-apocalyptic world. Use your puzzle-solving skills to unlock new areas. Collect items and upgrade your character. Play through multiple endings. Is description: Play through a post apocalypse world.
```

min_rejected: NLL=1.014606118202; haiku=[4];
SHA-256 `7f7ac12fe782e3a0eed8f1a14b3e76ebc5def43e272f2b00443fbe772be7502b`.

```text
Genres: Action, Adventure, Role-Playing, Strategy. Tags: Action-Adventure, Action-Strategy, Action-Simulation, Action-RPG, Action-Puzzle, Action-Casual, Action-Fighting, Action-Shooter, Action-Tower Defense, Action-Multiplayer. About: A new action-adventure game with a unique and challenging gameplay. The game is set in a fantasy world where you can explore, fight, and solve puzzles. The gameplay is fast-paced and requires quick thinking and reflexes. The graphics are colorful and detailed, and the sound effects are immersive. The story is engaging and the characters are well-developed. The multiplayer mode allows you to play with friends or online. The controls are easy to learn and the game is accessible to all players. The developers have created a game that is both fun and challenging, and it is sure to become a hit among gamers.
```

**arxiv / full**

max_accepted: NLL=2.351627349854; haiku=[7], luna=[8];
SHA-256 `1d3b94c6e07e606e844012b557dca7996affe07306dd66364c92f1bc3c8dd4b0`.

```text
Title: Simulating Continuous-Time Systems with Discontinuous Semiflows with Applications to Digital Simulation Methods Summary. Abstract: We develop methods for simulating continuous-time differential equations with discontinuous semiflows. Our framework applies to differential equations arising in semiconductor device modeling. Our contributions include methods based on meshless Gaussians. These methods have several advantages. First, they scale well with the device geometry. Second, they converge rapidly. Third, they are stable.
```

min_rejected: NLL=0.825668036938; haiku=[5];
SHA-256 `1c84d6cc2a8c7d038e50dd797c25cba1f4fe52f080e7833ee21510e5684a9ffa`.

```text
Title: A new approach to the problem of finding the maximum vertex in a graph. Abstract: The problem of determining the maximum cliques in a given graph is a fundamental problem in graph theory. In this paper, we propose a new approach for solving this problem. Our approach is based on a combination of graph theory and machine learning techniques. We first use graph theory to identify the most important nodes in the graph, and then use machine learning to predict the maximum number of cliques that can be formed by these nodes. We evaluate our approach on a variety of graphs and compare it to existing methods. Our results show that our approach is significantly more accurate than existing methods, and can be used to solve the maximum-clique problem in a fraction of the time.
```

**steam / steam4b**

max_accepted: NLL=2.268039703369; haiku=[7], luna=[7];
SHA-256 `daacbf669aae790411f88dcb78c0e7ff1ccb9439e6285a9bc54fd5c0f1d6bcf3`.

```text
Genres: Action, Adventure, Indie, RPGs. Tags: Adventure, Free, Indie RPG, Adventure Game with. About: The first part of the story of the mysterious and ancient city of Zanath, where the fate of the entire world is at stake. The city is under the control of the evil and powerful Zanadu, who is trying to destroy the world.
```

min_rejected: NLL=0.862740874290; haiku=[6];
SHA-256 `1ed5b9ec89892c766f9b13f08a8d1e4b2fb7ca7726944e40ce9ad23f3e799b9e`.

```text
Genres: Action, Adventure, RPG, Strategy. Tags: Indie, Singleplayer, Steam Achievements, Steam Cloud, Steam Workshop, Steam Trading Cards, Steam Community Market, Steam Leaderboards, Steam Group, Steam Friends, Steam Reviews, Steam Sales. About: The game is set in a post-apocalyptic world where the player takes on the role of a survivor. The player must navigate through a dangerous world filled with mutated creatures and other survivors. The game features a variety of weapons and equipment, as well as the ability to craft new items. The story is told through a series of cutscenes and dialogue with other characters. The gameplay is fast-paced and action-packed, with a focus on combat and exploration. The graphics are detailed and realistic, with an emphasis on creating a sense of atmosphere and immersion. The soundtrack is also a key part of the game, with music that is both
```

</details>

### 8.3 Verification and scope

Primary outcomes completed at `2026-10-29T08:45:47.054738+00:00`; graph outcomes at
`2026-10-29T08:50:21.564281+00:00`. Primary peak resident memory was
392.4 MiB; the graph process used 217.7 MiB.
All Python processes explicitly limited OMP/BLAS threads to 2. Saved NPY
inputs were memory-mapped and CSV/JSONL inputs streamed. No ONNX/encoder,
LLM, network call, model download or Lean build was used.

Synthetic checks cover rejection/missing/conflicting judge decisions,
exhaustive small tied-threshold/AUC cases and distances. An independent
[`verification script`](../scripts/bounds/verify_round8.py) checks all six
opposite-label minima with SciPy `cdist`, checks the three threshold minima
by direct enumeration and the three AUCs by pair counting, and rehashes every
consumed input: **7,989 files unchanged**, all independent checks passed.
The verification process peaked at **217.2 MiB**. Its record is
[`verification.json`](../runs/bounds8/verification.json). The primary and graph
code hashes remain frozen; no outcome-driven refit or threshold change was
made. Existing result files and the compiled Lean artifact are unchanged.
Only this report, new analysis scripts and new `runs/bounds8/` artifacts were
written; no commit was made.
