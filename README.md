# latent-gap-ideas

Does a point far from everything in an embedding space decode to a new idea? This repo tests that recipe (embed a
catalogue, pick points far from every item, turn them back into text) on three catalogues: Steam games, arXiv
computer-science papers and US patent applications.

Write-up: [The Gaps in the Map Are Not Ideas](https://www.javieraguilar.ai/en/blog/far-points-are-not-ideas/)
([español](https://www.javieraguilar.ai/es/blog/far-points-are-not-ideas/)).

## Findings

- **Decoding a point with an LLM blends its neighbours.** In a pre-registered causal test (true, swapped or no
  neighbours; Steam and arXiv; two generators; REAL, GAP, FAR) the listing follows the neighbours it is shown in all
  12 cells. The cosine score alone moves it only 0.04–0.07 closer to its own point than to another point.
- **Far points stay out of reach without neighbours too.** A GPU search that maximises cosine directly, followed by a
  coherence-preserving rewrite, recovers hidden items (10/10 per domain) but leaves FAR points near zero: on arXiv 90%
  stay within the nearest existing paper; on Steam 25% beat the nearest game by more than 0.05 (15% with an
  independent judge), and none passes cos 0.14.
- **Why:** texts occupy a narrow cap of the sphere (two Steam games: cos 0.41; two random directions: 0 ± 0.03). The
  FAR ceiling was predicted from corpus statistics before building any arXiv target (0.062 predicted, 0.058 observed).
- **Interior gaps are reachable but not newer**; the most novel ideas came from asking without a map (PLAIN), once the
  generator sees its own earlier ideas (PLAIND).
- **Bounds:** no certifiable ceiling for coherent text without an explicit coherence rule; proofs checked in Lean
  (`lean/`).

## Paper and later studies

The preprint draft is [`docs/paper/draft.md`](docs/paper/draft.md) (figures in `docs/paper/figures/`, every number
traced to its source file in `docs/paper/provenance.md`). Besides the decoding results above it reports three
pre-registered studies run against what was published or released later:

- **Gaps do not forecast growth** (`docs/prereg/where_future_lands.md`, `combination_vs_momentum.md`): a hole score
  ranks the fastest-growing cells of a past-only map at chance (AUC 0.45–0.54) while momentum reaches 0.73–0.92; no
  future item lands in the far region.
- **Distances do not steer a continuation** (`continue_from_context.md`, `iterative_critique.md`,
  `modern_and_oracle_history.md`): LLM continuations of arXiv papers stay closer to their source than the papers that
  later cite it, and feeding back the similarity to those papers does not raise it.
- **Inspiration distance has no sweet spot** (`inspiration_from_anything.md`): material from nine sources, graded by
  distance, gives games that anticipate the post-cutoff Steam catalogue no better at middle distances.

## Layout

- `docs/design.md` — design, arms, decisions log. `docs/geometry.md` — geometry laws and cross-domain replication.
  `docs/bounds*.md` — the bound attempts and the impossibility result. `docs/glossary.md` — terms.
- `docs/prereg/` — pre-registrations written before each run, with their results (`anchoring_causal.md`,
  `fluent_search.md`, `coherent_search.md`, `memory_probe.md`, …).
- `src/latent_gap/` — corpus, targets, decoder, embedder, judges. `scripts/` — runs and analyses.
- `kaggle/` — GPU kernels (embedding, fluent search, continuation, Lean check). `lean/` — Lean 4 proofs.
- `docs/paper/` — preprint draft, figures, provenance and annotated related-work notes.

## Reproducing

Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/): `uv sync`. Embeddings use Qwen3-Embedding-0.6B (ONNX). Model
calls go through a LiteLLM-compatible gateway (`LITELLM_BASE_URL`, `LITELLM_KEY_FILE`) or through the Codex and Claude
Code CLIs. Data live outside the repo (`~/Documents/experiment-data/latent-gap-ideas`): Steam from Kaggle
`fronkongames/steam-games-dataset`, arXiv cs.* metadata, patents from HUPD.

Three large embedding tables referenced by `docs/prereg/` (`cover_embeddings.csv`, `cover_r6_embeddings.csv`,
`cover_r6_fit_embeddings.csv`) are attached to the [v1.0 release](../../releases/tag/v1.0) instead of the tree.
