# Related work (for Section 6 of docs/paper/draft.md)

Javier Aguilar. Written 2026-10-30. Two parts: a paper-ready Related Work section in prose, then an annotated
bibliography with URLs, what I read, how each item relates to each finding, and a threat-to-novelty rating.
Numbers about my own results come from files in this repo (HTML comments name them); numbers about other work come
from the cited paper, read as stated in the bibliography.

---

## 6 Related work

**Decoding embedding points into items.** The closest technical ancestors of gap-based ideation are ELM
(Tennenholtz et al., 2024), which trains an adapter so an LLM can read a domain embedding and describe it, and EAGLE
(Tennenholtz et al., 2024b), which steers an LLM with reinforcement learning so that the embedding of a generated item
maximises a utility; its showcase task is filling "content gaps" in a MovieLens embedding, with a utility that rewards
distance from the three nearest existing movies. ELM explicitly leaves out "arbitrary hypothetical points in embedding
space" and reports that semantic consistency falls as extrapolations move further from real items; EAGLE motivates
itself by the observation that decoders can go "off-manifold". Neither evaluates against items that appeared later.
My Section 4 makes the off-manifold worry quantitative for the untrained version of the recipe (an LLM shown the
nearest neighbours): the decoded text is a blend of the neighbours shown, causally (12 of 12 cells), and far points lie
at a predictable distance from the cap where texts live. Embedding inversion (Morris et al., 2023) recovers text
from embeddings of real texts; it is the natural stronger decoder, and my limitations section notes that I did not
test it.

<!-- src: docs/prereg/anchoring_causal.md (12/12 cells); docs/geometry.md §12 (0.062 predicted, 0.058 measured) -->

**The narrow cap.** That learned text representations occupy a narrow cone is well established for contextual word
representations (Ethayarajh, 2019; Gao et al., 2019), motivated post-processing such as removing the common mean and
top directions (Mu and Viswanath, 2018) or whitening (Su et al., 2021; Li et al., 2020), and has been argued to be
inherent to self-attention (Godey et al., 2024). For sentence-embedding models trained contrastively, Ren et al.
(2025) report a mean vector shared by all sentences across 38 MMTEB models, and Ait-Saada and Nadif (2023) find
anisotropy has little effect on downstream quality. I do not claim the cap as a discovery. What I add is its
consequence for ideation: the cap's geometry predicts, from corpus statistics alone and before any target is built,
how close any text gets to a "far" point (0.062 predicted, 0.058 measured on arXiv), and so where far-point decoding
must fail.

<!-- src: docs/geometry.md §1 and the pre-registered prediction table; corpus ‖μ‖ 0.64 / 0.54 Steam / arXiv (docs/geometry.md table "corpus: median pair cos, ‖μ‖ ..."), 0.511 patents (docs/design.md) -->

**Does the future land in the gaps?** The idea that unoccupied regions of a knowledge map are where novelty lies
predates LLMs, in patent "white space" and "vacancy" maps (e.g. Son et al., 2012) and in the literature-based
discovery tradition, but these maps were rarely validated against later filings. The work closest to my Section 3
is Luo and Evans (2026), who embed OpenAlex concepts per discipline and year with word2vec, find topological holes of
order 0-2 with persistent homology, and track which later works fill them. Their results anticipate part of mine:
larger holes are monotonically less likely to be closed (at order 1 the annual closure probability falls from 0.062
in the smallest decile to 0.004 in the largest), fill rates collapse with order (73.5 %, 8.1 %, 0.2 %), and among nine
null models only those in which simulated papers combine concepts already close in the embedding reproduce the
observed fill rates. In their words, scientists fill "modest holes within and near" their regions. My test differs in
three ways that I think matter for the ideation claim: the map is of items (documents, games, patents) embedded by
general text encoders, the very maps the recipe uses, not of concepts; I test the recipe's literal "far" region
(none of 122,331 future items reaches it); and I pit the hole score against a momentum baseline as a forecaster of
growth, which they do not do. Their positive finding, that hole-filling works draw outsized citations, sits alongside
my mixed one (more isolated future games are slightly better reviewed, more isolated patent applications slightly less
often granted). Kedrick et al. (2025) study the converse event, papers that *open* gaps, and find them cited less at
five years and more at twenty.

<!-- src: docs/prereg/where_future_lands_result.json (F1 far_count 0 in all six cells; F5 rho +0.11/+0.13 Steam, −0.08/−0.08 patents); docs/paper/draft.md abstract (122,331) -->

**Forecasting where science goes.** Link prediction on concept networks is the established form of "where does the
future appear". In Science4Cast (Krenn et al., 2023), the strongest of ten methods used hand-crafted network features,
above all node degree and its recent change, and parameter-free preferential attachment was competitive with learned
models, with AUC up to 0.93 for single new links. Gu and Krenn (2025) extend this to forecasting high-impact concept
pairs. Tshitoyan et al. (2019) and Sourati and Evans (2023) show that embedding proximity and the distribution of
researchers anticipate discoveries. He and Chen (2018) report that a temporal-embedding novelty of MeSH topics
predicts their publication growth up to ten years ahead, controlling for one lag of growth, with R² of 0.003-0.017.
My finding that momentum forecasts growth (AUC 0.71-0.92) is therefore an expected result in the link-prediction
literature, and I present it as such: it is the baseline a gap-based forecaster must beat. The contribution is the
head-to-head in item-embedding geometry, where the hole score alone is at or below chance (0.39-0.48) and adds
nothing to momentum in five of six domain-embedder cells, together with the modest but replicated partial effect of
holes once momentum is fixed (ρ 0.18-0.34 in Steam and arXiv), which agrees in size and sign with He and Chen.

<!-- src: docs/prereg/where_future_lands_result.json (F2 partial_rho, F3 auc_momentum, auc_hole, gain); docs/prereg/where_future_lands.md (Result table) -->

The closest recent work on generation is The Alien Space of Science (Artiles et al., 2026). It represents 16,068 LLM
papers as sets of 273 LLM-extracted "idea atoms", scores atom triples for coherence (an autoregressive model over atom
sets) and for availability (a dual encoder over 39,791 author repertoires), and in a one-year temporal holdout ranks
all 1,143,135 triples without LLM calls: the top 0.875 % recovers 2025 targets at 35.9× chance by coherence alone, and
weighting unavailability raises recovery of strictly novel triples from 2.35× to 3.24× chance. Two points bear on
my claims. First, their unavailability is defined against people, not geometry, and their own density-estimator
baseline fails to separate paper-supported from author-only triplets (AUC 0.445); this agrees with my finding that
geometric emptiness is not the signal. Second, they report no momentum or recency baseline for the 2025 targets
(an appendix checks only that methods pick atoms of similar age), so it remains open how much of the coherence
signal is what is already growing. My Section 7 runs that comparison on my corpora.

**Benchmarks that score ideas against the future.** HindSight (Jiang, 2026) matches 200 generated ideas to 27,589
later papers with SPECTER2 and finds that LLM-judged novelty is negatively correlated with future realisation
(ρ = −0.29) for a retrieval-augmented generator. IdeaForecastBench (Li et al., 2026) asks LLMs to forecast the next
quarter's cs.ML topics across 624 topic-cutoff episodes and finds that how history is compressed changes hit rates; its
"Topic Trend" variant, which ranks clusters by recent activity, is the closest it comes to a momentum baseline, and it
is generator-dependent. Wang et al. (2026) train models to write "future-aligned" proposals scored against 2025
papers and note that their metric favours incremental, popular directions. PreScience (Ajith et al., 2026) builds a
forecasting benchmark around 98K AI papers and finds a synthetic year of research less diverse and less novel than the
real one. All four judge generated ideas by whether the future matches them; none asks where the real future sits on
the past map, or whether a gap score forecasts it. HindSight's negative correlation is consistent with my
observation that my ratings and the LLM judges' measure different things.

<!-- src: docs/design.md ("Where the owner's novelty departs from the judges'") -->

**Homogenisation and memory.** That LLMs return to a narrow set of ideas is well documented: Si et al. (2025) found
only 200 non-duplicates among 4,000 seed ideas per topic, even though the same study's expert reviewers rated the AI
ideas more novel than human ones; Jiang et al. (2025) show intra-model repetition and inter-model homogeneity on open
prompts; Doshi and Hauser (2024) find that AI help raises individual creativity but lowers collective diversity; Deng
et al. (2026) attribute part of the gap to fixation on early outputs and to knowledge merged into one distribution.
My PLAIN collapse (45 of 50 Steam games from one generator sharing a name stem) is an instance of this, and I do not
claim it as new. Showing a generator its earlier proposals is close to the deduplicated archives used by AI-scientist
systems and to population-referential instructions (Ibrahim et al., 2026); my contribution is only the comparison in
which this cheap fix, with no map at all, is judged more novel than every map-guided arm in three corpora.

<!-- src: docs/design.md ("PLAIN collapses towards one idea per generator": 45/50; patents outcome (6)); docs/paper/draft.md §5.2 -->

**Science of science and the Einstein test.** Novelty as atypical combination predicts impact (Uzzi et al., 2013;
Wang et al., 2017; Shi and Evans, 2023), and patents whose text differs from earlier patents but resembles later ones
mark breakthroughs (Kelly et al., 2021). These works measure novelty of items that exist; my F5 asks the narrower
question of whether isolation on the past map tracks reception, and finds small effects of opposite sign in games and
patents. The "adjacent possible" view of novelty (Tria et al., 2014) and the principle of relatedness in economic
development (Hidalgo et al., 2007; 2018), where new activity appears next to existing capabilities, are the general
form of what my data show. Finally, the temporal-holdout "Einstein test" (Benrimoh et al., 2025; Hassabis's 1911
proposal as reported by Ball, 2026) asks whether a model trained before a discovery can reach it, and has been
attempted with GPT-1900 (Hla, 2026; a 3.3B model on pre-1900 text) and Talkie (Levine, Duvenaud and Radford, 2026; 13B,
pre-1931). Those projects test a model's reach; mine tests a map's. They share the logic that only a frozen past and
the real future can say whether something was new.

---

## Annotated bibliography

Threat-to-novelty scale: **high** = already makes one of my claims; **medium** = makes a neighbouring claim that a
reviewer will cite against me unless I position against it; **low** = background or a different question.
"Read" says what I actually read: full = whole paper; HTML = arXiv HTML read in sections through a summarising
fetcher, with the numbers checked against the relevant sections; abstract = abstract or secondary coverage only.

My findings, for reference: **F1** no future item in the far region; **F3** momentum forecasts growth, hole score
does not and adds nothing; **F2** holes fill modestly once momentum is fixed; **F4** edge vs centre is
embedder-dependent; **F5** isolation vs quality, + Steam, − patents; **A** anchoring (decoding blends the shown
neighbours); **S** neighbour-free search stays at cos 0.09-0.14 from far points; **C** the cap and its pre-registered
ceiling; **L** non-certifiability (Lean); **P** PLAIN/PLAIND; **J** owner vs judges.

### Closest work

1. **Luo, J. and Evans, J. A. (2026). Filling holes in science draws collective attention, but most higher-order holes
   remain unexplored.** arXiv:2608.28822 (v2, 14 Oct 2026). https://arxiv.org/abs/2608.28822. Read: full (PDF).
   Word2vec concept embeddings per discipline-year from OpenAlex (19 disciplines), persistent homology of order 0-2,
   later works that fill each hole. Larger holes are less often closed (Table S4: order-1 annual closure 0.062 → 0.004
   from smallest to largest death decile); fill rates collapse with order (73.5 %, 8.1 %, 0.2 %); only null models
   that combine nearby concepts reproduce real fill rates; hole-filling works are more cited and "prescient".
   Relation: supports F2 (holes fill, modestly) and the spirit of F1/F3 (the future is near what exists); contrasts
   with F5 (they find a citation premium; I find mixed reception). No momentum baseline, no item-level map, no
   "far" region, no generation. **Threat: medium-high**, the most important item to cite. The claim "the future
   fills small, nearby holes and rarely large ones" is theirs in concept space. My novelty rests on item embeddings
   (the recipe's actual map), the literal far region, the momentum head-to-head, three non-identical domains
   (games, papers, patents), and the explanation of why decoding cannot reach far points.

2. **Artiles, A. H., Weiss, M., Brinkmann, L., Rahwan, I., Schölkopf, B., Pal, C., Larochelle, H., Goyal, A. and
   Rahaman, N. (2026). The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research
   Directions.** arXiv:2603.01092 (v3). https://arxiv.org/abs/2603.01092. Read: HTML, all sections and appendices
   A.8, C, D.5, E. 273 atoms from 16,068 LLM papers (2017-2025); coherence model plus author-availability dual
   encoder; temporal holdout trained to 2024, tested on 2,424 2025 triples (1,411 strictly novel); top 10,000 of
   1,143,135 triples: 35.9× chance overall at β = 0, strictly novel 2.35× → 3.24× at β = 0.4; Vendi 225.2 vs 56.5 for
   independent Claude sampling; density estimator fails (AUC 0.445). No momentum, recency or growth baseline for
   the 2025 targets (D.5 only compares atom age profiles across methods). Relation: the strongest published
   counterpoint to "gap-based ideation fails", but its gap is social (unavailability to authors), not geometric;
   its density baseline failing agrees with F3; its diversity result agrees with P (independent LLM sampling is
   narrow). **Threat: medium.** It does not test where the future lands on a geometric map or compare with
   momentum, so it does not pre-empt F1-F3; a reviewer will ask why my negative result does not contradict its
   positive one, and the answer is that we test different notions of "gap". My Section 7 (combination vs momentum)
   should state whether momentum explains their kind of signal in my corpora.

3. **Krenn, M., Buffoni, L., Coutinho, B., Eppel, S., Foster, J. G., Gritsevskiy, A., Lee, H., Lu, Y., Moutinho, J.
   P., Sanjabi, N., Sonthalia, R., Tran, N. M., Valente, F., Xie, Y., Yu, R. and Kopp, M. (2023). Forecasting the
   future of artificial intelligence with machine learning-based link prediction in an exponentially growing
   knowledge network (Science4Cast).** Nature Machine Intelligence 5, 1326-1335. arXiv:2210.00881.
   https://arxiv.org/abs/2210.00881. Read: full (PDF). 64k-concept semantic network from >100k AI papers; predict
   new concept links 1-5 years ahead. Winners used hand-crafted features; degree centrality and its first and
   second time derivatives (popularity and its change) were the most important; preferential attachment, parameter
   free, competitive; AUC up to 0.93. Relation: F3's "momentum wins" is the known result in link prediction.
   **Threat: medium for F3 as a method claim; low for F3 as a test of the gap premise.** I must present momentum as
   the established baseline, not as a discovery.

4. **He, J. and Chen, C. (2018). Predictive effects of novelty measured by temporal embeddings on the growth of
   scientific literature.** Frontiers in Research Metrics and Analytics 3:9. arXiv:1801.09121.
   https://doi.org/10.3389/frma.2018.00009. Read: HTML (ar5iv). Novelty = semantic change of MeSH topic vectors
   across aligned yearly word2vec embeddings; random-effects panel regression of growth t+1..t+10 on novelty with
   one lag of growth; coefficient ≈ 0.12 at every horizon, R² 0.003-0.017. Relation: a small positive novelty effect
   on growth under a weak momentum control, close in size to my F2 partial ρ; it does not test isolation on a map
   or a far region. **Threat: low-medium.** Cite as prior evidence that novelty adds a little beyond growth.

### Decoding embeddings

5. **Tennenholtz, G., Chow, Y., Hsu, C.-W., Jeong, J., Shani, L., Tulepbergenov, A., Ramachandran, D., Mladenov, M.
   and Boutilier, C. (2024). Demystifying Embedding Spaces using Large Language Models (ELM).** ICLR 2024.
   arXiv:2310.04475. https://arxiv.org/abs/2310.04475. Read: HTML. Adapter maps domain embeddings into an LLM
   (PaLM 2-XS); MovieLens 25M; semantic and behavioural consistency; interpolations and CAV extrapolations only;
   "we do not consider arbitrary hypothetical points"; consistency decreases with extrapolation. Relation: trained
   precursor of the decoder; its falling consistency is the trained-decoder version of A and C. **Threat: low.**

6. **Tennenholtz, G., Chow, Y., Hsu, C.-W., Shani, L., Liang, E. and Boutilier, C. (2024). Embedding-Aligned Language
   Models (EAGLE).** NeurIPS 2024. arXiv:2406.00024. https://arxiv.org/abs/2406.00024. Read: HTML (main text and
   appendices up to Amazon results). RL agent edits text so its encoded embedding maximises a utility that combines
   user relevance with distance from the three nearest movies ("content gaps"); MovieLens 25M with Gemini-generated
   descriptions; beats ELM on utility (0.74 vs 0.57) and on human ratings (124 raters); no evaluation against items
   released later. Relation: the most principled version of gap-based ideation; it optimises for gaps without asking
   whether the future goes there (F1, F3). **Threat: low** (different question), but cite as the method family.

7. **Morris, J. X., Kuleshov, V., Shmatikov, V. and Rush, A. M. (2023). Text Embeddings Reveal (Almost) As Much As
   Text (vec2text).** EMNLP 2023. https://arxiv.org/abs/2310.06816. Read: abstract (known work). Iterative inversion
   of text embeddings. Relation: a stronger decoder I did not test (limitation for A and S). **Threat: low.**

### Anisotropy and the cap

8. **Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?** EMNLP-IJCNLP 2019.
   https://aclanthology.org/D19-1006/. Read: known work. Contextual representations occupy a narrow cone.
   Relation: background for C. **Threat: low.**
9. **Mu, J. and Viswanath, P. (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations.**
   ICLR 2018. https://arxiv.org/abs/1702.01417. Read: known work. Common mean and dominant directions; removing them
   helps. Relation: C (my ‖μ‖ is this mean). **Threat: low.**
10. **Gao, J., He, D., Tan, X., Qin, T., Wang, L. and Liu, T.-Y. (2019). Representation Degeneration Problem in
    Training Natural Language Generation Models.** ICLR 2019. https://arxiv.org/abs/1907.12009. Read: known work.
    **Threat: low.**
11. **Li, B., Zhou, H., He, J., Wang, M., Yang, Y. and Li, L. (2020). On the Sentence Embeddings from Pre-trained
    Language Models (BERT-flow).** EMNLP 2020. https://arxiv.org/abs/2011.05864; **Su, J., Cao, J., Liu, W. and
    Ou, Y. (2021). Whitening Sentence Representations for Better Semantics and Faster Retrieval.**
    https://arxiv.org/abs/2103.15316. Read: known work. **Threat: low.**
12. **Godey, N., de la Clergerie, É. and Sagot, B. (2024). Anisotropy Is Inherent to Self-Attention in
    Transformers.** EACL 2024. https://aclanthology.org/2024.eacl-long.3. Read: abstract. **Threat: low.**
13. **Ren, X., Sun, Y. and Liang, H. (2025). Correcting Mean Bias in Text Embeddings: A Refined Renormalization with
    Training-Free Improvements on MMTEB.** arXiv:2511.11041. https://arxiv.org/abs/2511.11041. Read: abstract. A
    mean vector "near-identical across all sentences" in 38 modern sentence-embedding models. Relation: the modern
    sentence-embedder form of the cap in C. **Threat: low.** No one I found predicts the far-point-to-text distance
    from the cap, which is what C pre-registers.
14. **Ait-Saada, M. and Nadif, M. (2023). Is Anisotropy Truly Harmful? A Case Study on Text Clustering.** ACL 2023
    (short). https://aclanthology.org/2023.acl-short.103. Read: abstract. **Threat: low.**

### Scoring generated ideas against the future

15. **Jiang, B. (2026). HindSight: Evaluating LLM-Generated Research Ideas via Future Impact.** arXiv:2603.15164.
    https://arxiv.org/abs/2603.15164. Read: HTML. Cutoff June 2023; 27,589 later papers; SPECTER2 match at θ = 0.96;
    retrieval-augmented ideas score 0.297 vs 0.119; LLM-judged novelty vs HindSight ρ = −0.29. Relation: J (judges'
    novelty is not future realisation); my arXiv anticipation test is a similar style of check. **Threat: low.**
16. **Li, F., Tang, Z., Yu, H., Zhao, Y. and You, J. (2026). Can Large Language Models Forecast What Researchers
    Study Next? (IdeaForecastBench).** arXiv:2609.00747. https://arxiv.org/abs/2609.00747. Read: HTML, main text and
    tables 5-9. 63,855 cs.ML papers, 52 topics × 12 cutoffs; Hit@5 up to 0.95 (Qwen2.5-14B, Summary); embedding
    novelty only as a diagnostic, never related to realisation; Topic Trend (clusters ranked by recent activity) is
    the nearest thing to momentum, LLM-mediated; title reuse 7-47 % per backbone. Relation: F3 (no isolated momentum
    baseline there), P (reuse = repetition). **Threat: low.**
17. **Wang, H., Jiang, P., Sun, J., Shi, Z., Yu, H., Han, J. and Ji, H. (2026). Learning to Predict Future-Aligned
    Research Proposals with Language Models.** arXiv:2603.27146. https://arxiv.org/abs/2603.27146. Read: HTML.
    Train 2024 → test 2025 ML venues; future-alignment score via text-embedding-3-large retrieval plus GPT-4.1-mini
    rubric; +10.6 % over prompting; the authors note it rewards incremental, popular directions; no trend baseline.
    Relation: indirect support for F3 (what is aligned with the future is what is popular). **Threat: low.**
18. **Ajith, A., Singh, A., DeYoung, J., Kunievsky, N., Kozlowski, A. C., Tafjord, O., Evans, J., Weld, D. S., Hope,
    T. and Downey, D. (2026). PreScience: A Dataset and Benchmark for Scientific Forecasting.** arXiv:2602.20459.
    https://arxiv.org/abs/2602.20459. Read: abstract. 98K AI papers, seven forecasting tasks including future
    combination prediction; a synthetic year is less diverse and less novel than the real one. Relation: P.
    **Threat: low**; its future-combination task is a neighbour of my Section 7.

### Forecasting science and novelty vs growth

19. **Gu, X. and Krenn, M. (2025). Forecasting high-impact research topics via machine learning on evolving knowledge
    graphs (Impact4Cast).** Machine Learning: Science and Technology. arXiv:2402.08640.
    https://arxiv.org/abs/2402.08640. Read: abstract. Concept graph from >21M papers, forecasts high-impact pairs,
    AUC > 0.9 in most settings. Relation: F3 family. **Threat: low.**
20. **Krenn, M. and Zeilinger, A. (2020). Predicting research trends with semantic and neural networks with an
    application in quantum physics (SemNet).** PNAS 117(4), 1910-1916. https://doi.org/10.1073/pnas.1914370116.
    Read: known work. **Threat: low.**
21. **Tshitoyan, V., Dagdelen, J., Weston, L., et al. (2019). Unsupervised word embeddings capture latent knowledge
    from materials science literature.** Nature 571, 95-98. https://doi.org/10.1038/s41586-019-1335-8. Read: known
    work. Proximity in a word embedding anticipates later discoveries. Relation: proximity, not gaps, carries signal
    (F1, F3). **Threat: low.**
22. **Sourati, J. and Evans, J. A. (2023). Accelerating science with human-aware artificial intelligence.** Nature
    Human Behaviour 7, 1682-1696. https://doi.org/10.1038/s41562-023-01648-z. Read: known work. Random walks on a
    researcher-concept hypergraph anticipate discoveries; also proposes "alien" hypotheses. Relation: the social
    notion of gap behind item 2. **Threat: low.**
23. **Shi, F. and Evans, J. (2023). Surprising combinations of research contents and contexts are related to impact
    and emerge with scientific outsiders from distant disciplines.** Nature Communications 14, 1641.
    https://doi.org/10.1038/s41467-023-36741-4. Read: abstract. Embedding of the content hypergraph forecasts
    next-year combinations; surprising combinations relate to impact. Relation: F5. **Threat: low.**
24. **Kedrick, K., Yang, W., Gebhart, T., Wang, Y. and Funk, R. J. (2025). Opening Knowledge Gaps Drives Scientific
    Progress.** arXiv:2509.21899. https://arxiv.org/abs/2509.21899. Read: abstract. Gap-opening papers (topology of
    concept networks, MAG 1900-2020) are cited less at five years, more at twenty. Relation: horizon caveat for F5 and
    for my one-to-three-year windows. **Threat: low.**
25. **Uzzi, B., Mukherjee, S., Stringer, M. and Jones, B. (2013). Atypical combinations and scientific impact.**
    Science 342, 468-472. https://doi.org/10.1126/science.1240474. **Wang, J., Veugelers, R. and Stephan, P. (2017).
    Bias against novelty in science.** Research Policy 46(8), 1416-1436.
    https://doi.org/10.1016/j.respol.2017.06.006. Read: known work. Relation: F5. **Threat: low.**
26. **Kelly, B., Papanikolaou, D., Seru, A. and Taddy, M. (2021). Measuring Technological Innovation over the Long
    Run.** American Economic Review: Insights 3(3), 303-320. https://doi.org/10.1257/aeri.20190499. Read: abstract.
    Important patents are textually unlike earlier patents and like later ones. Relation: F5 in patents (my
    isolation alone is slightly negative for grant; theirs combines backward novelty with forward impact).
    **Threat: low.**
27. **Wu, C. (2025). Novelty and Impact of Economics Papers.** arXiv:2511.01211. https://arxiv.org/abs/2511.01211.
    Read: abstract. Spatial (isolation in embedding space) vs temporal novelty predict different outcomes. Relation:
    F5. **Threat: low.**
28. **Tria, F., Loreto, V., Servedio, V. D. P. and Strogatz, S. H. (2014). The dynamics of correlated novelties.**
    Scientific Reports 4, 5890. https://doi.org/10.1038/srep05890. **Hidalgo, C. A., Klinger, B., Barabási, A.-L.
    and Hausmann, R. (2007). The product space conditions the development of nations.** Science 317, 482-487.
    https://doi.org/10.1126/science.1144581; **Hidalgo, C. A. et al. (2018). The principle of relatedness.**
    Springer Proceedings in Complexity. https://doi.org/10.1007/978-3-319-96661-8_46. Read: known work. Novelty
    arrives in the adjacent possible / next to existing capabilities. Relation: general frame for F1-F3.
    **Threat: low.**
29. **Son, C., Suh, Y., Jeon, J. and Park, Y. (2012). Development of a GTM-based patent map for identifying patent
    vacuums.** Expert Systems with Applications 39(3), 2489-2500. https://doi.org/10.1016/j.eswa.2011.08.101. Read:
    abstract. Patent "vacuum" maps; I found no study validating that mapped vacuums are later filled. Relation: the
    practitioner form of the premise F1-F3 test. **Threat: low.**

### Homogenisation, fixation and memory

30. **Si, C., Yang, D. and Hashimoto, T. (2025). Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study
    with 100+ NLP Researchers.** ICLR 2025. arXiv:2409.04109. https://arxiv.org/abs/2409.04109. Read: HTML,
    sections 3, 5, 7. AI ideas judged more novel (5.64 vs 4.84, p < 0.01); 200 non-duplicates among 4,000 seed ideas
    (cos > 0.8 counts as duplicate). Relation: P (collapse) and J (novelty judgements). **Threat: low** for P's
    collapse, which is theirs; my claim is the comparison with map-guided arms.
31. **Jiang, L., et al. (2025). Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond).**
    NeurIPS 2025 Datasets and Benchmarks (best paper). arXiv:2510.22954. https://arxiv.org/abs/2510.22954. Read:
    abstract. Intra-model repetition, inter-model homogeneity, 26K open queries. Relation: P. **Threat: low.**
32. **Doshi, A. R. and Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective
    diversity of novel content.** Science Advances 10(28), eadn5290. https://doi.org/10.1126/sciadv.adn5290. Read:
    known work. Relation: P. **Threat: low.**
33. **Deng, Y., Brucks, M. and Toubia, O. (2026). Examining and Addressing Barriers to Diversity in LLM-Generated
    Ideas.** arXiv:2602.20408. https://arxiv.org/abs/2602.20408. Read: abstract. LLMs fixate on early outputs and
    merge knowledge into one distribution; CoT and ordinary personas raise diversity above humans. Relation: P;
    possible tension (fixation on own earlier outputs vs my PLAIND, where showing them reduces repetition): the
    difference is that PLAIND shows titles to avoid, not ideas to continue. **Threat: low-medium for P**; worth one
    sentence.
34. **Ibrahim, F. N., Azad, N. S. and Baten, R. A. (2026). Anchorless Diversification for Parallel LLM Ideation.**
    arXiv:2605.30150. https://arxiv.org/abs/2605.30150. Read: abstract. Population-referential instructions and
    semantic stratification diversify idea pools; anchored regeneration compared. Relation: P and A (anchoring to
    shown seeds). **Threat: low.**
35. **Padmakumar, V. and He, H. (2024). Does Writing with Language Models Reduce Content Diversity?** ICLR 2024.
    https://arxiv.org/abs/2309.05196; **Kirk, R., et al. (2024). Understanding the Effects of RLHF on LLM
    Generalisation and Diversity.** ICLR 2024. https://arxiv.org/abs/2310.06452. Read: known work. **Threat: low.**

### The Einstein test

36. **Ball, P. (2026). [Einstein-test feature], Nature news feature, September 2026.**
    https://www.nature.com/articles/d41586-026-02804-x. Read: excerpt only (3 Quarks Daily repost, 10 Sep 2026) and
    secondary coverage; the original headline and full text were not accessible. Reports Hassabis's proposal (India
    AI Summit, February 2026) to train an LLM on knowledge up to 1911 and test whether it reproduces general
    relativity. Must be re-checked against the original before citing details. **Threat: low.**
37. **Benrimoh, D., Mikus, N. and Rosenfeld, A. (2025). The Einstein Test: Towards a Practical Test of a Machine's
    Ability to Exhibit Superintelligence.** arXiv:2501.06948. https://arxiv.org/abs/2501.06948. Read: abstract.
    **Threat: low.**
38. **Hla, M. (2026). GPT-1900.** Model cards, https://huggingface.co/mhla/gpt1900-d34-physics-clm-pre1900. Read:
    model cards and secondary coverage; no paper. 3.29B (D34) on pre-1900 English text, physics fine-tune and RL;
    coverage reports Planck-like statements but no relativity. **Threat: low.**
39. **Levine, N., Duvenaud, D. and Radford, A. (2026). Talkie-1930.** Open-weight 13B model on ~260B tokens of
    pre-1931 English text. Read: secondary coverage only (e.g. https://the-decoder.com/here-is-what-an-llm-that-knows-nothing-after-1930-thinks-our-world-looks-like-in-2026/);
    the primary release page must be found before citing. **Threat: low.**

### Searched for and not found

- A test of whether real future items land in sparse or far regions of a document or item embedding built from the
  past, against a momentum baseline. Searched: arXiv/web queries on future papers in sparse embedding regions,
  patent white-space validation, embedding novelty vs growth with momentum baselines, preferential attachment vs
  novelty on embeddings (2026-10-30). The nearest are items 1 (concept-space holes, no momentum), 2 (social gap,
  no momentum), 3 (link prediction, no geometry) and 4 (topic novelty with a one-lag control).
- Any work predicting, before building targets, the distance from far points to the nearest coherent text (C), or a
  formal non-certifiability result for coherence bounds (L). None found; I did not search the formal-methods
  literature exhaustively.
- Steam-specific novelty-vs-success studies: only informal analyses of rare tag combinations.
