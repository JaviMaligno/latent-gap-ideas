# Related work for the planned inspiration-distance experiment (video-game ideation)

Javier Aguilar. Written 2026-11-02, before pre-registration. Scope: what is already known about the planned
experiment in which an LLM (gpt-4o, cutoff October 2023; Claude Code and Codex as contaminated stress tests) proposes
a new video game after being shown inspiration material graded from near (other games) to far (other cultural
products, science and technology, everyday information such as recipes, sports rules, news and random Wikipedia
articles), with distance graded both by type and by embedding distance to the target. Outcomes: anticipation of
Steam games released from 2024, isolation from the pre-2024 catalogue, judged coherence. Working hypothesis: an
inverted U (moderate distance best).

Already covered elsewhere and not repeated here: SciMON, MOOSE-Chem, Alien Space of Science, Tang and Yang (2026),
Liu, Butler et al. (2026, arXiv:2610.17439), Si et al. (2024), EAGLE, ELM, HindSight, Luo and Evans
(`related_work.md`, `related_work_step.md`).

Threat-to-novelty scale, as in `related_work.md`: **high** = already runs the planned experiment or makes its main
claim; **medium** = makes a neighbouring claim that a reviewer will cite against the plan unless the design positions
against it; **low** = background or a different question; **none** = cited only for method. "Read" says what was
actually read: full = whole paper; HTML = arXiv HTML read through a summarising fetcher, numbers checked in the
relevant sections; abstract = abstract (OpenAlex or publisher) or secondary coverage only.

---

## 1 Design cognition: near vs far inspiration and its effect on novelty and quality

The human literature is sixty years old and its verdict is split three ways: far helps novelty (Chan et al. 2011;
Srinivasan et al. 2017), near helps creativity overall (Chan, Dow and Schunn 2015; Jia et al. 2019), and moderate is
best (Fu et al. 2013 as a proposal; Gonçalves et al. 2013 and Wang and Nickerson 2019 as findings). The inverted-U
hypothesis is therefore not new; what this literature lacks is an LLM ideator and any ground truth other than judges.

1. **Fu, K., Chan, J., Cagan, J., Kotovsky, K., Schunn, C. and Wood, K. (2013). The meaning of "near" and "far": the
   impact of structuring design databases and the effect of distance of analogy on design output.** *Journal of
   Mechanical Design* 135(2), 021007. https://doi.org/10.1115/1.4023158 (author copy formerly at
   https://www.lrdc.pitt.edu/Schunn/papers/fuchanetal-near-far-2013.pdf, now 404). Read: abstract, plus the
   characterisation of its results in Chan et al. (2015, full text). Measured: engineering students' concepts after
   near or far patent stimuli, where distance comes from a patent map built with latent semantic analysis and a
   Bayesian structural-form algorithm; the stimuli of their 2011 study are re-mapped onto the same structure. Finding:
   "near" and "far" are relative to the candidate pool; far stimuli led to lower novelty and quality than near ones,
   and "there is such a thing as too far". Chan et al. (2015) attribute to it the argument for an inverted U in which
   moderate distance is best. **Threat: medium.** It is the origin of the planned hypothesis, so the inverted U
   cannot be presented as a new idea, only as a new test (LLM ideator, future ground truth).

2. **Chan, J., Fu, K., Schunn, C., Cagan, J., Wood, K. and Kotovsky, K. (2011). On the benefits and pitfalls of
   analogies for innovative design: ideation performance based on analogical distance, commonness, and modality of
   examples.** *Journal of Mechanical Design* 133(8), 081004. https://doi.org/10.1115/1.4004396;
   https://edrl.engr.wisc.edu/wp-content/uploads/sites/142/2022/02/2011-jmd-distancecommonness.pdf. Read: abstract
   and introduction (PDF). Measured: senior engineering students, U.S. patent examples crossed by near/far field,
   more/less common, picture/text, plus a no-example control. Finding: far-field and less-common examples raise
   novelty and the variability of quality; far + less common beats control on novelty; modality does not matter.
   **Threat: low.** Binary distance, human ideators, judged outcomes.

3. **Chan, J., Dow, S. P. and Schunn, C. D. (2015). Do the best design ideas (really) come from conceptually distant
   sources of inspiration?** *Design Studies* 36, 31-58. https://doi.org/10.1016/j.destud.2014.08.001;
   https://joelchan.github.io/files/2014-desstudies-nearfar-in-press.pdf. Read: full (PDF). Measured: OpenIDEO
   concepts that cite their inspirations (12 challenges; the analysed subset is several hundred concepts, 16 %
   shortlisted); distance = 1 − cosine between LDA topic vectors (400 topics) of the inspiration and the challenge
   brief, z-scored within challenge, validated against human similarity ratings (r = 0.51); outcome = shortlisting by
   an expert panel; multilevel logistic models with challenge and author nesting. Finding: citing any source helps,
   but each unit of maximum source distance lowers the log-odds of shortlisting by 0.33 (95 % CI roughly 0.06 to 0.60)
   and mean distance by about 0.40; concepts citing far sources converge on the no-citation rate; the slope does not
   vary across challenges. The paper names curvilinear effects as a reason for continuous measures but reports only
   the linear term. **Threat: medium.** The strongest field evidence, and it predicts a monotone decline, not an
   inverted U: the plan must pre-register the monotone alternative and a valid shape test.

4. **Chan, J., Schunn, C. D. and Dow, S. P. (2014). Overreliance on conceptually far sources decreases the creativity
   of ideas.** *Proceedings of CogSci 2014*. https://escholarship.org/uc/item/2bm91572. Read: abstract.
   Conference version of item 3. **Threat: low** (duplicate).

5. **Gonçalves, M., Cardoso, C. and Badke-Schaub, P. (2013). Inspiration peak: exploring the semantic distance between
   design problem and textual inspirational stimuli.** *International Journal of Design Creativity and Innovation*
   1(4), 215-232. https://doi.org/10.1080/21650349.2013.799309. Read: abstract. Measured: novice designers given three
   types of written stimuli at increasing semantic distance from the problem; fluency, flexibility, originality.
   Finding: the intermediate type yields more, more flexible and more original ideas; the too close and the too
   distant are less useful. Related: Gonçalves, Cardoso and Badke-Schaub (2014), "What inspires designers?",
   *International Journal of Industrial Ergonomics* 44(4); "How far is too far?" (ICDC 2012). **Threat: medium.** A
   direct human inverted U with textual stimuli of graded distance: the hypothesis has prior support, and three
   ordinal levels are what the plan must exceed (continuous distance, many types).

6. **Srinivasan, V., Song, B., Luo, J., Subburaj, K., Elara, M. R., Blessing, L. and Wood, K. (2017). Understanding
   effects of analogical distance on performance of ideation: key observations and findings.** ASME IDETC/CIE 2017,
   DETC2017-67752. https://doi.org/10.1115/detc2017-67752; journal version: **Srinivasan, V. et al. (2018). Does
   analogical distance affect performance of ideation?** *Journal of Mechanical Design* 140(7), 071101. Read:
   abstract of the conference version. Measured: 105 students, 138 concepts of spherical rolling robots generated
   from patents; distance = knowledge distance between the patent's technology classes and the problem's classes.
   Finding: near classes are used more often; novelty rises with distance; quality falls with distance. **Threat:
   low-medium.** Its two monotone trends are the trap for the plan: any composite of a rising and a falling outcome
   can show a hump without any outcome being non-monotone.

7. **Jia, L., Becattini, N., Cascini, G. and Tan, R. (2019). Testing ideation performance on a large set of designers:
   effects of analogical distance.** *International Journal of Design Creativity and Innovation* 8(1), 31-45.
   https://doi.org/10.1080/21650349.2019.1618736. Read: abstract. 84 mechanical-engineering students, near/medium/far
   analogies and a control; near analogies give higher novelty and quality for untrained designers. **Threat: low.**
   A three-level design that found no hump.

8. **Tseng, I., Moss, J., Cagan, J. and Kotovsky, K. (2008). The role of timing and analogical similarity in the
   stimulation of idea generation in design.** *Design Studies* 29(3), 203-221.
   https://doi.org/10.1016/j.destud.2008.01.003; **Christensen, B. T. and Schunn, C. D. (2007). The relationship of
   analogical distance to analogical function and preinventive structure: the case of engineering design.** *Memory &
   Cognition* 35(1), 29-38. https://doi.org/10.3758/bf03195939. Read: known work and its characterisation in item 3.
   Far sources matter more after ideation has begun (Tseng); designers draw mostly on within-domain analogies
   (Christensen and Schunn). **Threat: low.** Tseng's timing effect suggests a design arm: stimulus before vs after a
   first draft.

9. **Goucher-Lambert, K. and Cagan, J. (2019). Crowdsourcing inspiration: using crowd generated inspirational stimuli
   to support designer ideation.** *Design Studies* 61, 1-29. https://doi.org/10.1016/j.destud.2019.01.001. Read:
   abstract via search coverage. 96 participants; near stimuli raise feasibility and usefulness, far stimuli raise
   uniqueness; near/far validated by cosine similarity. **Threat: low.** Same trade-off as item 6.

10. **Wang, K. and Nickerson, J. V. (2019). A Wikipedia-based method to support creative idea generation: the role of
    stimulus relatedness.** *Journal of Management Information Systems* 36(4), 1284-1312.
    https://doi.org/10.1080/07421222.2019.1661095. Read: abstract. Measured: stimuli at graded relatedness obtained by
    following Wikipedia hyperlinks from the focal concept, one hop at a time. Finding: relatedness raises quantity and
    usefulness; remotely related, not unrelated, stimuli raise novelty. **Threat: medium.** A graded Wikipedia distance
    with a peak short of "unrelated" in humans; the plan's "random Wikipedia article" level is exactly the level this
    paper found worst for novelty.

11. **Malthouse, E., Russell, S., Liang, Y. and Hills, T. T. (2021). The influence of exposure to randomness on
    lateral thinking in divergent, convergent, and creative search.** *Cognition* 218, 104937.
    https://doi.org/10.1016/j.cognition.2021.104937; https://wrap.warwick.ac.uk/id/eprint/159029. Read: abstract.
    Pre-registered, 592 British participants, extraneous pages from Wikipedia's random-page generator during one
    forecasting and two fluency tasks. Finding: no improvement and often significant impairment; Bayesian
    meta-analysis supports the null; stimuli may need to be "optimally random", i.e. task-related. **Threat: low-medium.**
    The only pre-registered test of incidental random everyday stimuli I found, and it is negative in humans: the plan
    has a human prior to beat at its far end.

12. **Zhao, E. Y., Ishihara, M., Jennings, P. D. and Lounsbury, M. (2018). Optimal distinctiveness in the console video
    game industry: an exemplar-based model of proto-category evolution.** *Organization Science* 29(4), 588-611.
    https://doi.org/10.1287/orsc.2017.1194. Read: abstract. 6,544 U.S. console games in 78 proto-categories formed
    around hit exemplars. Finding: early in a proto-category, conformity to the exemplar raises sales; as it matures,
    moderate differentiation is best; critics' reviews mediate. **Threat: low** for the plan, **useful** for its
    interpretation: in games, the payoff of distance is a moving inverted U that depends on category age, which is
    what a Steam-anticipation target will inherit (2024 releases cluster around recent exemplars).

Background on recombination distance, not read for this note: Fleming (2001), "Recombinant uncertainty in
technological search", *Management Science* 47(1) (distant recombination raises variance); Uzzi, Mukherjee, Stringer
and Jones (2013), "Atypical combinations and scientific impact", *Science* 342 (conventional core plus an atypical
tail); Kittur et al. (2019), "Scaling up analogical innovation with crowds and AI", *PNAS* 116(6),
https://doi.org/10.1073/pnas.1807185116, and Hope, Chan, Kittur and Shahaf (2017), "Accelerating innovation through
analogy mining", KDD, https://arxiv.org/abs/1706.05585 (structural rather than surface analogies). **Threat: low.**

## 2 LLM ideation with random or cross-domain stimuli

13. **Liu, Q. E., Dubova, M., Conklin, H., Harada, T. and Griffiths, T. L. (2026). Assessing the effect of
    cross-domain mapping on creativity in humans and large language models.** arXiv:2603.19087 (v3, 15 Sep 2026; an
    earlier version circulated as "Serendipity by Design"). https://arxiv.org/abs/2603.19087. Pre-registered on OSF.
    Read: HTML, about 96 % of the text. Measured: one new feature for each of 10 everyday products (backpack, sofa,
    car...), either by transferring a property of a randomly assigned source concept (26 sources from animals, plants,
    natural systems, technology, food) or by naming an unmet user need; 100 target-source pairs; 140 Prolific humans
    and seven LLMs including **gpt-4o**, o3 and Claude-Sonnet-4; ideas rephrased by o3 to remove style; 1,002 lay
    raters, about 2,800 ideas, originality (split-half 0.81), feasibility, utility, investment-worthiness; distance
    between Wikipedia-sentence distributions embedded with all-mpnet-base-v2. Findings: random cross-domain mapping
    raises human originality (2.93 vs 2.57); LLMs are more original than humans on average (o3 3.83, Claude-Sonnet-4
    3.76) and show no cross-domain effect overall (b = 0.02, p = 0.64) except on the more distant half (b = 0.12) and
    the most distant 10 % (b = 0.24); distance predicts originality in humans (β = 0.10) and LLMs (β = 0.18); the
    trend is described as monotone, no curve is fitted; humans' best cross-domain ideas lose on utility and
    feasibility. No test against real later products. **Threat: medium-high.** It is the closest existing design (an
    LLM, gpt-4o among them, randomly assigned sources graded by embedding distance, a no-source condition) and its
    shape is monotone increasing for originality; the plan differs by the domain (games), the stimulus types
    (everyday information rather than curated concepts), the incidental framing and above all the ground truth
    (later releases rather than lay ratings).

14. **Choi, Y. and Song, M. (2026). Read, grep, and synthesize: diagnosing cross-domain seed exposure for LLM research
    ideation (PaperGym).** arXiv:2605.11532 (12 May 2026). https://arxiv.org/abs/2605.11532. Read: HTML. Measured: 30
    ML research problems, GPT-5 generator, Claude Sonnet 4.6 judge; four arms: no retrieval, same-domain top-21 seeds,
    cross-domain seeds (≤ 21, seven ML domains), and 21 seeds drawn uniformly at random from the 1,167-seed library.
    Findings: cross-domain beats same-domain on pairwise novelty 20-9 (p ≈ 0.06) and no retrieval 18-12 (p ≈ 0.36),
    but ties the random-seed control 14-16 (p ≈ 0.86); single-pass novelty C 4.13 = D 4.13; the model leaves the
    attribution unused in 8/30 random vs 1/30 cross-domain problems. Authors: systems benefit from diverse seed
    exposure but do not exploit why a seed was retrieved. No distance measured, no shape. **Threat: medium.** It is
    the content-not-score result of our own Section 3 seen from the retrieval side, and it shows that the plan needs
    a random-content control at matched distance or "distance" will be confounded with "something different was
    shown".

15. **Mehrotra, P., Parab, A. and Gulwani, S. (2024). Enhancing creativity in large language models through
    associative thinking strategies.** arXiv:2405.06715. https://arxiv.org/abs/2405.06715. Read: HTML. GPT-4,
    temperature 1; product improvement (25 objects), fairy-tale twists (15), marketing (15); prompt to associate with
    a random object, with one-hop and attribute variants. Findings: originality +40 % in product design (originality
    = no matching product on Amazon, Etsy or eBay) and +28 % in stories, usefulness lower; one-hop behaves like the
    control because the model picks a close final association; incongruity noted qualitatively. No distance measure.
    **Threat: low-medium.** The LLM version of the novelty-up, usefulness-down trade-off; its "no existing product
    matches" check is a crude present-day version of the plan's isolation measure.

16. **Shen, A., Druckmann, S. and Zou, J. (2026). Unlocking LLM creativity in science through analogical reasoning.**
    arXiv:2605.11258. https://arxiv.org/abs/2605.11258. Read: HTML, first 70 %. Claude Sonnet 4.5, GPT-5.2, Gemini 3
    Flash; no-domain vs cross-domain vs structured analogy (extract objects and relations, map them to a domain);
    binary novelty e.g. Gemini 8.6 % / 37.9 % / 68.8 %, but human-rated reasonableness 67 % for analogy vs 86 % for
    cross-domain; judged domain distance highest for analogy; four biomedical case studies implemented. **Threat:
    low.** Science, model-chosen domains, no graded stimulus; again novelty rises while soundness falls with distance.

17. **Radensky, M., Shahid, S., Fok, R., Siangliulue, P., Hope, T. and Weld, D. S. (2024-2026). Scideator /
    Human-LLM compound system for scientific ideation through facet recombination and novelty evaluation.**
    arXiv:2409.14634 (v7, May 2026). https://arxiv.org/abs/2409.14634. Read: abstract and search coverage. Retrieves
    analogous papers in distance tiers (near-1, near-2, far-1, far-2) and recombines purpose/mechanism/evaluation
    facets; user study with 19-22 CS researchers on creativity support. **Threat: low.** Distance tiers are a design
    feature, not a manipulated variable with an outcome curve.

18. **Kang, H. B., Mysore, S., Huang, K., Chang, H.-S., Prein, T., McCallum, A., Kittur, A. and Olivetti, E. (2022).
    Augmenting scientific creativity with retrieval across knowledge domains.** NLP+HCI workshop at NAACL 2022.
    arXiv:2206.01328. https://arxiv.org/abs/2206.01328. Read: abstract. Aspect-based retrieval of papers from other
    domains, case studies with scientists. **Threat: low.**

19. **Ding, Z., Srinivasan, A., MacNeil, S. and Chan, J. (2023). Fluid transformers and creative analogies: exploring
    large language models' capacity for augmenting cross-domain analogical creativity.** ACM C&C 2023.
    arXiv:2302.12832. https://arxiv.org/abs/2302.12832. Read: abstract. LLM-generated cross-domain analogies judged
    helpful (median 4/5) for problem reformulation and frequently lead to changes in it. **Threat: low.**

20. **Zhu, Q. and Luo, J. (2023). Generative transformers for design concept generation.** *Journal of Computing and
    Information Science in Engineering* 23(4), 041003. https://doi.org/10.1115/1.4056220; arXiv:2211.03468; and
    **Zhu, Q. and Luo, J. (2022). Generative design ideation: a natural language generation approach.**
    arXiv:2204.09658. Read: abstracts. GPT-2/GPT-3 concept generation with domain-knowledge, problem-driven and
    analogy-driven synthesis; near-field vs far-field source knowledge in a rolling-toy case. **Threat: low.**

21. **Ma, K., Grandi, D., McComb, C. and Goucher-Lambert, K. (2024). Do large language models produce diverse design
    concepts? A comparative study with human-crowdsourced solutions.** *JCISE*. https://doi.org/10.1115/1.4067332.
    Read: abstract. 4,000 GPT-4 solutions over 5 topics, 8 parameter settings and 8 prompting techniques; humans are
    always more diverse. **Threat: low.** Relevant to the plan's isolation measure: the no-stimulus baseline will be
    homogeneous.

22. Method systems that use other material as stimulus without varying its distance: **Yasunaga, M. et al. (2023).
    Large language models as analogical reasoners** (ICLR 2024), arXiv:2310.01714 (self-generated exemplars for
    reasoning); **Yuan, S. et al. (2023). AnalogyKB**, arXiv:2305.05994 (million-scale analogy KB from knowledge
    graphs); **Bhavya, B., Xiong, J. and Zhai, C. (2022). Analogy generation by prompting large language models**,
    arXiv:2210.04186; **Sanyal, A. et al. (2025). Spark: a system for scientifically creative idea generation**,
    arXiv:2504.20090 (retrieval plus a reviewer model trained on 600K OpenReview reviews); **Li, L. et al. (2024).
    Chain of Ideas**, arXiv:2410.13185 (EMNLP Findings 2025; chains of related papers); **Ma, R. et al. (2025).
    MetaMuse**, arXiv:2510.03851 (dictionary keywords as external stimuli for algorithm design; many keywords are of
    little value and get used as variable names); **Wang, B. et al. (2025). From analogy to innovation**, *Advanced
    Engineering Informatics*, https://doi.org/10.1016/j.aei.2025.103427. Read: abstracts. **Threat: low** for all.

## 3 LLM creativity benchmarks that vary stimulus distance

I found no benchmark whose independent variable is the distance of a stimulus shown to the model; item 13 is the only
graded-distance design with LLMs. What exists measures distance in the *output*:

23. **Divergent Association Task and its LLM uses**, e.g. "Divergent creativity in humans and large language models",
    *Scientific Reports* (2025), https://doi.org/10.1038/s41598-025-25157-3; **lechmazur/divergent**,
    https://github.com/lechmazur/divergent (25 words unrelated to each other and to 50 random seed words, relatedness
    judged by LLMs). Read: abstracts / README. Random output scores high on distance-based creativity metrics, a
    caveat raised in this literature. **Threat: none**; method caveat for the isolation measure.

24. **Padmakumar, V., Chen, Y.-H., Pan, J., Chen, V. and He, H. (2025). Measuring LLM novelty as the frontier of
    original and high-quality output.** arXiv:2504.09389. https://arxiv.org/abs/2504.09389. Read: abstract. Novelty =
    harmonic mean of the share of n-grams unseen in pretraining data and task quality (OLMo, Pythia); novel in-context
    examples raise originality at the cost of quality. **Threat: low.** A template for the plan's composite (isolation
    × coherence as a harmonic mean) and one more instance of the trade-off.

25. **Carichon, F., Sharma, S., Girard, M., Rampa, R. and Farnadi, G. (2026). IDEAFix: evaluation framework for
    creative defixation prompting in LLMs.** arXiv:2606.00875. Read: abstract. Defixation prompts raise originality;
    homogenisation persists across models. **Threat: low.**

## 4 LLM ideas evaluated against later releases in creative (non-science) domains

Searched with general and extended web queries on games, films, products, patents, Kickstarter and post-cutoff
anticipation (2026-11-02). I found nothing that scores LLM-generated game, film or product ideas against items
released after the model's cutoff. The time-split recipe exists only for science (HindSight; IdeaForecastBench;
future-aligned proposals; PreScience, all in `related_work.md`) and for events (forecasting benchmarks). Adjacent:

26. **Terwiesch, C., Meincke, L., Girotra, K., Mollick, E., Nave, G. and Ulrich, K. T. (2023). AI and its impact on
    creativity and diversity: an empirical study of LLM-generated product ideas** (SSRN 4526071,
    https://doi.org/10.2139/ssrn.4526071) and **Girotra, K., Meincke, L., Terwiesch, C. and Ulrich, K. T. (2023).
    Ideas are dimes a dozen: large language models for idea generation in innovation** (Wharton working paper,
    https://mackinstitute.wharton.upenn.edu/wp-content/uploads/2023/08/LLM-Ideas-Working-Paper.pdf). Read: known work.
    GPT-4 product ideas beat students' on purchase intent; LLM pools are less diverse. Present-day consumer judgments,
    not later releases. **Threat: low.**

27. **Large language models as narrative-driven recommenders** (2024), arXiv:2410.13604. Read: search coverage. Checks
    leakage by comparing films before and after the model's cutoff; the drop is small, partly because post-cutoff
    films were relevant. **Threat: none**; it is a reminder that "released after the cutoff" is not "unknown at the
    cutoff".

28. Cutoff leakage: "Simulated ignorance fails" (IJCAI 2026 preprint) and OracleProto, arXiv:2605.03762 (temporal
    masking leaves 1.37 % residual leakage); a model with an October 2023 cutoff described a November 2023 event when
    prompted. Read: search coverage. **Threat: none**; design constraint for the gpt-4o arm.

Games-specific LLM work (GAVEL, NeurIPS 2024, Ludii board games with a concept-based novelty score; LLM quest and
narrative generation; LLM and games surveys, e.g. arXiv:2402.18659) generates or evaluates game content but never
against later releases. **Threat: low.**

---

## What remains open, and how the design should change to be novel

**Nothing pre-empts the plan as a whole.** No study scores LLM ideas against later releases in games or any other
creative domain, and none crosses stimulus *type* with continuous stimulus *distance* for an LLM ideator. But two of
its pieces are not new and must not be sold as such: the inverted-U hypothesis (proposed by Fu et al. 2013, found in
humans by Gonçalves et al. 2013 and Wang and Nickerson 2019, contradicted in field data by Chan et al. 2015) and the
LLM design with randomly assigned sources graded by embedding distance (Liu et al. 2026, with gpt-4o, monotone for
originality). The novelty lies in (a) the ground truth (later releases, not judges), (b) the separation of content
from distance, which our own anchoring result says is the variable that matters, and (c) the everyday, incidental
stimuli and the game domain.

Concrete changes:

1. **Pre-register competing shapes, not one.** H-mono-down (Chan et al. 2015: nearer is better for anticipation and
   coherence), H-mono-up (Liu et al. 2026: farther is better for isolation and originality), H-hump (Fu, Gonçalves).
   Test the hump with the two-lines test (Simonsohn 2018, *Advances in Methods and Practices in Psychological Science*
   1(4)) or an interrupted-regression equivalent, not with a significant quadratic term, which detects curvature on a
   monotone concave trend too.
2. **Test the shape per outcome, and the composite only as secondary.** Isolation will rise with distance and
   coherence will fall (items 6, 9, 15, 16, 24); their product or harmonic mean will show a hump by construction. The
   informative claim is a hump in *anticipation* alone, the only outcome that is not mechanically tied to distance.
3. **Separate content from distance.** Add (i) a matched-distance random-content control: for each stimulus, another
   item of a different type at the same embedding distance from the target (PaperGym's random seeds tied cross-domain
   retrieval); (ii) a shuffled control: the same stimulus set assigned to a different target; (iii) a donor swap as in
   `docs/prereg/anchoring_causal.md`, to measure whether the game the model writes moves towards the stimulus shown.
   Without these, "moderate distance is best" cannot be told apart from "this content is useful".
4. **Cross type with within-type distance.** Fu et al. (2013) show that near and far are relative to the candidate
   pool: grade distance within each type (near and far recipes, near and far papers) so that type and distance are not
   collinear, and measure distance in two embedders (as in the paper) plus one non-embedding measure (e.g. tag or genre
   overlap for games) so that the curve is not an artefact of one space.
5. **Manipulate the framing: incidental vs instructed.** The plan's premise is material "a human would have around
   without thinking it relevant". Human evidence is that extraneous random pages do not help (Malthouse et al. 2021)
   while instructed mapping does (Liu et al. 2026). An arm that presents the material as background context, crossed
   with an arm that says "use this as inspiration", is new and tests the premise directly.
6. **Measure uptake as a mediator.** Code or judge whether the output uses the stimulus and how (surface vs structural
   transfer, as Liu et al. 2026 code by hand; attribution use, as PaperGym), so that a null curve can be read as
   "ignored" or "used and unhelpful".
7. **Date the ground truth by first public appearance, not release.** Many 2024 Steam releases were announced, had
   store pages, demos or trailers before October 2023 and are therefore in gpt-4o's training data (item 27, 28). Use
   Steam page creation or first-announcement dates, keep only games first visible after the cutoff, and report the
   effect on the excluded set as a contamination check. Date the stimuli too: nothing shown may postdate the cutoff.
8. **Calibrate anticipation against baselines that need no stimulus.** Include no-stimulus and near-games-only arms
   and a momentum or trend baseline (the paper's Section 4: momentum is what forecasts where the future grows). A
   stimulus level counts as helping only if it beats these, since the no-stimulus arm will already sit near popular
   2024 genres (item 21: LLM pools are homogeneous; IdeaForecastBench: what matches the future is what is popular).
9. **Keep the far end honest.** Include the "random Wikipedia article" and "news" levels as pre-registered extreme
   points, since that is where the human literature predicts failure (items 10, 11) and where the LLM literature
   predicts the largest originality gain (item 13). A clean divergence between the two literatures there would itself
   be a result.
10. **Optional timing arm.** Stimulus before vs after a first draft (Tseng et al. 2008), cheap to add if the budget
    allows; drop it otherwise rather than widen the design before the main arms are closed.

### Searched for and not found

- An LLM study whose outcome is a curve over continuous stimulus distance for anything other than judged originality.
- Any test of LLM-generated game, film or product ideas against later releases (queries on post-cutoff anticipation,
  time-split, Steam, Kickstarter, later patents; 2026-11-02).
- Any LLM study with "oblique strategies" prompts; only random-object and random-keyword stimuli (items 15, 22).
- The full text of Fu et al. (2013): the author copy returns 404 and the publisher copy is closed; its results are
  cited here from its abstract and from Chan et al. (2015). Re-check before citing numbers.
