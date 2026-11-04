# Related work for the continuation-step results (claims 1-3) and embedding inversion

Javier Aguilar. Written 2026-10-30. Novelty check for the results of `docs/prereg/continue_from_context.md` (Part B)
and `docs/prereg/iterative_critique.md` before they go into the paper. Complements `related_work.md`, which already
covers SciMON, PreScience, HindSight, IdeaForecastBench, Future-aligned proposals, Alien Space, Luo & Evans,
Science4Cast, EAGLE, ELM and Si et al.; those are only revisited here where a detail bears on claims 1-3.

Threat scale as in `related_work.md`: **high** = already makes the claim; **medium** = a neighbouring claim a reviewer
will cite against me unless I position against it; **low** = background or a different question; **none** = nothing
found. "Read" says what was read: HTML = arXiv HTML read in sections through a summarising fetcher, numbers checked
against the relevant sections; abstract = abstract page only.

My results, for reference (Qwen3-Embedding-0.6B, 150 arXiv contexts from 2021-2022, citing follow-ups from 2024+,
gpt-4o-2024-08-06 with training cutoff before every follow-up):
- **Claim 1:** real follow-ups sit at median cos 0.620 from their context; one-shot gpt-4o continuations at 0.765
  (gap ~0.145), with or without neighbours shown.
- **Claim 2:** rule-based (GEO) and LLM-reviewer critique lengthen the step (0.087 and 0.063 over two revisions);
  hit rate and fit fall; ORACLE feedback (similarity to the real follow-ups) does not raise fit (−0.012).
- **Claim 3 (exploratory):** against a later non-citing arXiv paper matched on cosine to the context, gpt-4o ideas
  have 0.037 less fit to the real follow-ups, before any critique; critique leaves this excess unchanged.

---

## Claim 1: LLM continuations take a shorter step than real follow-ups

### LOUD: pre-empted

**Tang, Y. and Yang, Y. (2026). AI Research Agents Narrow Scientific Exploration.** arXiv:2605.27905 (v1 27 May
2026, v2 11 Jul 2026). https://arxiv.org/abs/2605.27905. Read: HTML, main text and supplement S1-S2.
Five agent frameworks (Zero-shot, AIScientist, ResearchAgent, AgentLaboratory, Co-Scientist) × five LLMs (incl.
GPT-5.4, Llama-3.1-8B, Qwen3), 219,655 ideas from seed sets of five Semantic Scholar papers (an anchor plus four
related papers, titles and abstracts), 155 areas in 12 fields, seeds 2020-2025. Embedding: Qwen3-Embedding-4B.
"Exploration distance" = cosine distance from an idea to the centroid of its five seed papers; compared with
"follow-on human papers" = papers that directly cite at least one seed. Result: **AI 0.322 vs human follow-ons 0.410**
(diff −0.088, 95 % CI −0.092 to −0.083), significant in every field and every year, for every framework (Zero-shot
largest gap −0.144, AgentLaboratory smallest −0.027) and every LLM. Their wording: "AI-generated ideas remain closer
to the seed literature than subsequent human research." They also report lower breadth (0.554 vs 0.599), lower
"frontier alignment" (share of next-year top-10 % keywords, 0.285 vs 0.365) and lower neighbourhood impact.
**Threat: high.** This is my claim 1, with the same operationalisation (embedding cosine from the source to
citing follow-ups vs to LLM ideas), the same family of encoder, and a sample three orders of magnitude larger. The
size of my gap in similarity units (0.145) is in the same range as theirs in distance units (0.088 overall, 0.144 for
zero-shot, which is the condition closest to my CTX arm). Claim 1 can only be presented as an independent
replication with three differences: (i) a single-paper context, not a five-paper centroid; (ii) a generator whose
training cutoff (Oct 2023) precedes every follow-up (≥ 2024-01-01), whereas Tang and Yang pair 2020-2025 seeds with
models such as GPT-5.4 and Qwen3 trained after many follow-ons appeared, and do not discuss leakage; (iii) the
neighbour-shown arm (CTX+NB), which shows the gap does not depend on what is in the prompt.

### Other close work

- **Liu, Z., Butler, K., Machlanski, D., Bueno, D. and Tsaftaris, S. A. (2026). Large Language Models Extend Known
  Research But Do Not Create New Research Directions.** arXiv:2610.17439 (12 Oct 2026).
  https://arxiv.org/abs/2610.17439. Read: HTML. 26.9M OpenAlex papers, 1,325 topics, eight LLMs, 524,308
  pseudo-papers from 20 focal papers per topic; SPECTER2. LLM pseudo-papers are *farther* from the nearest
  published paper than real ones (up to 16 % when given the papers farthest from the centroid), but have 15-23 %
  lower "orthogonal novelty" (share of variance outside the subspace spanned by the topic's literature). Comparison set
  is same-topic papers, not citing follow-ups, and there is no temporal holdout. **Threat: medium.** It is a
  counterpoint on distance (a topic-level prompt gives longer, not shorter, steps), and the reconciliation matters:
  their distance is to the nearest published paper, mine is to the one paper being continued.
- **Ajith et al. (2026), PreScience,** arXiv:2602.20459, already in `related_work.md`. Detail relevant here (HTML,
  sections 4 and appendices B, D): the contribution-generation task gives a model the target paper's influential
  references and asks for the target's title and abstract (the reverse direction of mine: from the cited papers to the
  citing one). Scored by an LLM judge (LACER, 1-10): GPT-4o 4.71, a random influential reference 4.31, GPT-5 5.64.
  Generated contributions beat the "just copy a reference" baseline only modestly, which is consistent with staying
  near the inputs, but the paper never measures generation-to-input distance. The repo's `task_followup_prediction/`
  directory (https://github.com/allenai/prescience) holds only code folders (`generate`, `evaluate`, `analysis`) with
  ROUGE-L/BERTScore scripts and no README describing a separate task; I could not confirm a distinct follow-up task
  beyond contribution generation. **Threat: low.**
- **Wang, Q., Downey, D., Ji, H. and Hope, T. (2024). SciMON: Scientific Inspiration Machines Optimized for
  Novelty.** ACL 2024. arXiv:2305.14259. https://arxiv.org/abs/2305.14259. Read: HTML (v7). Input = background
  context of a paper; target = the same paper's idea; temporal split (test = 2022). Qualitative: "All models often
  copied and rephrased directly from the context." No embedding step measured, no follow-ups. **Threat: low-medium**
  (the qualitative form of claim 1, within one paper).
- **Li, Z. et al. (2026). GIScholarBench.** arXiv:2606.08036 (SIGSPATIAL 2026). https://arxiv.org/abs/2606.08036.
  Read: HTML, task 3. LLM proposes 8-10 directions from a seed paper; ground truth = concatenated titles and
  keywords of later citing papers; TF-IDF topic coverage: models recover 0.09-0.12 of citing-paper topics and miss
  88-91 %, and cluster more (HHI 0.70-0.87 vs 0.64). No step measured. **Threat: low.**
- **Chen, Z., Zhao, Y. and Cohan, A. (2026). Measuring the Gap Between Human and LLM Research Ideas.**
  arXiv:2607.01233. https://arxiv.org/abs/2607.01233. Read: HTML. Ideas generated from the prior works a real paper
  drew on, compared with that paper (Qwen3-Embedding-4B): model proposals are less broadly positioned over the prior
  works and over-produce "bridge" and "synthesis" ideas (47-64 % vs 12 %). No follow-ups. **Threat: low.**
- **Kumar, S., Ghosal, T., Goyal, V. and Ekbal, A. (2025). Can Large Language Models Unlock Novel Scientific
  Research Ideas?** EMNLP 2025. arXiv:2409.06185. https://arxiv.org/abs/2409.06185. Read: HTML. Future-research
  ideas from a paper scored against the authors' own future-work statements (IAScore), not later papers. **Threat: low.**
- Science-of-science distance between a citing paper and what it cites exists as a novelty measure (e.g. Shibayama,
  Yin and Matsumoto, 2021, PLOS ONE, https://doi.org/10.1371/journal.pone.0254034; arXiv:2510.03240), but not as a
  benchmark for LLM continuations. **Threat: low.**

## Claim 2: critique lengthens the step but does not orient it; score feedback towards the real follow-ups does not raise fit

- **Liu et al. (2026), arXiv:2610.17439** (above). A five-turn ideator-judge dialogue keeps pseudo-papers 12-17 %
  below real papers in orthogonal novelty at every turn; a "wild" judge raises distance but *lowers* orthogonal
  novelty relative to a conservative judge. **Threat: medium.** This is the closest published form of "feedback
  lengthens the step but not in a new direction", measured against same-topic literature rather than against what
  was actually published next, with no temporal holdout and no score-towards-the-truth arm.
- **Tang and Yang (2026), arXiv:2605.27905** (above). Observational across frameworks: the framework with the longest
  step (AgentLaboratory, gap −0.027) has among the lowest frontier alignment (0.250 vs 0.377), and they state literature
  search "does not substantially improve" frontier alignment or impact. Every framework already includes novelty
  prompts and, for AIScientist/ResearchAgent/Co-Scientist, reflection or reviewer modules; there is no controlled
  critique arm and no per-follow-up direction measure. **Threat: medium-low** for the "longer but not oriented" half.
- **SciMON (Wang et al., 2024)** (above). Iterative novelty boosting: ideas whose SentenceBERT cosine to retrieved
  literature exceeds 0.6 are sent back with "make it significantly different"; humans judge 88.9 % of updated ideas
  substantially different and 55.6 % more novel. The boosted ideas are **never** scored against the ground truth
  (ROUGE/BERTScore are reported only for the retrieval variants). **Threat: low-medium:** the same feedback mechanism,
  but the consequence for matching the real target is not measured.
- **Yang, Z. et al. (2025). MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific
  Hypotheses.** ICLR 2025. arXiv:2410.07076. https://arxiv.org/abs/2410.07076. Read: HTML (v6), sections 5-6 and
  Table 10. Evolutionary unit = mutations refined with feedback on validness, novelty, clarity, significance, then
  recombined; on 51 post-cutoff 2024 chemistry papers, removing it lowers the average GPT-4o Matched Score against
  the real hypothesis from 2.730 to 2.578 (top-of-beam MS 3.765 → 2.863). **Threat: low, but a counterpoint to cite.**
  Feedback-plus-recombination there *raises* the match to the real hypothesis. Differences that reconcile it: the target
  is the same paper's hypothesis, not later citing work; inspirations are retrieved from a corpus that contains the
  ground-truth inspirations in most settings; the effect mixes feedback with mutation, recombination and LLM ranking;
  and the score is a Likert judge, not geometry.
- **Yang, Z. et al. (2024). MOOSE: Large Language Models for Automated Open-domain Scientific Hypotheses Discovery.**
  Findings of ACL 2024. arXiv:2309.02726. https://arxiv.org/abs/2309.02726. Read: abstract and secondary summaries.
  Present/past/future feedback improves LLM-judged and expert-judged novelty and validity; no comparison with later
  papers. **Threat: low.**
- Reviewer-loop systems: **Baek, J., Jauhar, S. K., Cucerzan, S. and Hwang, S. J. (2025). ResearchAgent.** NAACL 2025.
  arXiv:2404.07738. https://arxiv.org/abs/2404.07738; **Lu, C. et al. (2024). The AI Scientist.** arXiv:2408.06292.
  https://arxiv.org/abs/2408.06292; **Li, L. et al. (2024). Chain of Ideas.** arXiv:2410.13185.
  https://arxiv.org/abs/2410.13185. Read: known work. All refine ideas with LLM reviewers and evaluate with LLM or
  human judges; none checks refined ideas against later papers. **Threat: low.**
- **Score feedback towards the real follow-ups (ORACLE arm):** not found anywhere. Wang et al. (2026,
  arXiv:2603.27146, already in `related_work.md`; HTML re-read) *train* on a future-alignment reward and gain up to
  +10.6 % FAS, which is the trained counterpart; inference-time score-only feedback is not tested there.
  **Threat: none** for the oracle result itself.

## Claim 3: at equal distance from the source, LLM continuations point towards the real follow-ups less than an arbitrary later paper

- No work found that conditions on the step and compares direction. Searched (2026-10-30): distance-matched or
  random-later-paper baselines for generated ideas, follow-up retrieval with distractors, direction of displacement.
- Nearest neighbours: Tang and Yang's frontier alignment (keyword coverage of next-year papers, not distance-matched);
  Liu et al.'s orthogonal novelty (geometric direction relative to the topic subspace, not relative to the realised
  future); PreScience's "random influential reference" and "same-category random paper" baselines (LACER 4.31 and
  1.27, not matched on distance to the input); HindSight's vanilla-generation control (no distance matching).
- **Threat: low.** The claim is the most novel of the three. Its weakness is internal, not prior art: the matched papers
  share the follow-ups' era (2023-2026) and the encoder mismatch between ONNX and Kaggle embeddings; both caveats are
  stated in `iterative_critique.md` and must travel with the claim.

## Decoding a point of embedding space (vec2text and relatives)

- **Morris, J. X., Kuleshov, V., Shmatikov, V. and Rush, A. M. (2023). Text Embeddings Reveal (Almost) As Much As Text
  (vec2text).** EMNLP 2023. arXiv:2310.06816. https://arxiv.org/abs/2310.06816. Read: HTML. Iterative correction
  (re-embed the hypothesis, correct towards the target) recovers 92 % of 32-token inputs exactly (GTR-base, 50 steps
  plus beam); 8 % at 128 tokens with ada-002. Out of domain (BEIR) BLEU falls with length (Quora 95.5, NQ 32.7,
  trec-news 14.5). Most relevant to "far from data": Gaussian noise on the embedding destroys inversion long before it
  hurts retrieval (Table 7: λ = 0.01 leaves NDCG@10 at 0.296 vs 0.302 but BLEU falls from 80.4 to 10.3). The paper
  does not invert interpolations, averages or arbitrary points. **Threat: low.** It shows that a small displacement off
  the text manifold already breaks the strongest decoder, which supports my far-point argument rather than
  pre-empting it; it remains the stronger decoder I did not test.
- **Zhang, C., Morris, J. X. and Shmatikov, V. (2025). Universal Zero-shot Embedding Inversion (ZSInvert).**
  arXiv:2504.00147. https://arxiv.org/abs/2504.00147. Read: abstract. Encoder-agnostic adversarial decoding plus a
  correction model; recovers semantic content, degrades under large noise. **Threat: low.** A cheaper stronger decoder
  to name in the limitation.
- **Wang, Y. et al. (2026). Denoising-Aware Inversion: Revealing Privacy Risks in Noise-Protected Text Embeddings.**
  arXiv:2608.18610. https://arxiv.org/abs/2608.18610. Read: HTML excerpt via search. States the off-manifold failure
  explicitly: noise "pushes the target embedding off the clean text manifold", and inversion then chases off-manifold
  directions; their fix is to project back onto the manifold first. **Threat: low**, but useful: projecting onto the
  manifold before decoding is exactly the step that, for a far point, returns the nearest existing region (my cap result).
- **Ondov, B., Chang, C.-H., Zhou, Y., Giuffrè, M. and Xu, H. (2026). ctELM: Decoding and Manipulating Embeddings of
  Clinical Trials with Embedding Language Models.** arXiv:2601.18796. https://arxiv.org/abs/2601.18796. Read:
  abstract. Claims "plausible clinical trials from novel vectors" and controllable shifts along concept vectors; no
  numbers on quality vs distance from data. **Threat: low-medium** for the decoding half: it is the ELM recipe applied to
  generating new items, and a reviewer may cite it as evidence that novel vectors decode well; it reports no test
  against later trials.
- **Zhou, Feng and Yang (2025). Provably Secure Retrieval-Augmented Generation.** arXiv:2508.01084. Appendix E.2.9:
  feeding uniformly random vectors to a vec2text-style decoder yields fluent text unrelated to any source. Read:
  search excerpt. **Threat: low.** Anecdotal support that decoders produce fluent text from anywhere, so fluency is
  not evidence of reaching the point.
- Text-VAE latent holes: **Bowman, S. R. et al. (2016). Generating Sentences from a Continuous Space.** CoNLL 2016.
  https://arxiv.org/abs/1511.06349; **Xu, P., Cheung, J. C. K. and Cao, Y. (2020). On Variational Learning of
  Controllable Representations for Text without Supervision.** ICML 2020.
  https://proceedings.mlr.press/v119/xu20a.html (manipulated codes fall in vacant regions where the decoder fails);
  **Li, R., Peng, X. and Lin, C. (2021). On the Latent Holes of VAEs for Text Generation.** arXiv:2110.03318.
  Read: known work / abstracts. **Threat: low.** The qualitative fact that decoders fail away from the data is old;
  what I add is the pre-registered quantitative ceiling from the cap.

---

## Paragraph ready to merge into Section 6 (after "Benchmarks that score ideas against the future")

**Continuing a given paper.** That LLM research ideas stay closer to their starting literature than the human work
that follows it has been shown at scale by Tang and Yang (2026): over 219,655 ideas from five agent frameworks and five
LLMs, the cosine distance from an idea to the centroid of its five seed papers is 0.322, against 0.410 for the human
papers that cite those seeds, in every field and framework. My Part B result (median cosine 0.765 for gpt-4o
continuations vs 0.620 for real citing follow-ups) is an independent replication of that finding, not a discovery,
with a single-paper context and a generator whose training cutoff precedes every follow-up, which removes the leakage
their 2020-2025 design leaves open. Liu et al. (2026) find, at the topic level, that LLM pseudo-papers can sit farther
from the nearest published paper yet still inside the subspace of existing work, and that an ideator-judge dialogue
does not change this; SciMON's novelty boosting (Wang et al., 2024) similarly makes ideas more different from the
literature without testing whether they match the real target better, while MOOSE-Chem's feedback-and-recombination
loop raises the match to a rediscovered hypothesis (Yang et al., 2025). What I add is the controlled version against
the realised future: critique, whether rule-based or from an LLM reviewer, lengthens the step to near the real one,
yet the fit to the real follow-ups falls, and feedback that reports the similarity to those very follow-ups does not
raise it. A distance-matched comparison, which I found in no prior work, localises the deficit: at the same distance
from the context, an arbitrary later paper of the same category points towards the real follow-ups more than gpt-4o's
continuation does (by 0.037 in cosine), before any critique, and critique leaves that gap unchanged. The short step is
known; the finding is that it is the wrong direction, not only the wrong length, and that neither critique nor score
feedback fixes the direction.

<!-- src: docs/prereg/iterative_critique.md (Result and Exploratory sections); docs/prereg/continue_from_context_result.json -->

## Searched for and not found

- A distance-matched (step-conditioned) direction test of generated continuations against citing follow-ups.
- Inference-time score-only feedback towards the real future (my ORACLE arm) in any ideation paper.
- A quantitative curve of embedding-inversion quality against distance from the data manifold for arbitrary points
  (only noise-level results in vec2text and its privacy follow-ups).
- A README or paper section for PreScience's `task_followup_prediction/`; only code folders are visible.
