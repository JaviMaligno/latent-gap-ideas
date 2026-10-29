# Glossary

Every label that appears in tables, logs and result files (`items.jsonl`, `metrics.csv`, `summary.csv`,
`judged*.jsonl`). Design and hypotheses: [design.md](design.md).

## Space and corpus

| Term | Meaning |
|---|---|
| space | Qwen3-Embedding-0.6B embeddings (1024-d, L2-normalised; cosine = dot product) of each item's canonical text `Genres: … Tags: … About: …` (about truncated to 1,200 chars). bge-m3 is the robustness space. |
| past / future | Items released before / from the cutoff (Steam: 2024-01-01). Targets and the decoder only ever see the past; the future is used only to check anticipation. |
| hidden | Corpus rows the decoder may not see for a given target: the held-out item itself and its near-duplicates (cos > 0.95). |
| `adult` | Steam items whose tags/genres match `Sexual Content`, `Nudity`, `NSFW`… Excluded from the space, because Azure's content filter rejects prompts that contain them. |
| `explicit_src` / `explicit_nn` | Flag only, never an exclusion: the item's source rows (`explicit_src`) or the neighbours shown to the decoder (`explicit_nn`) match the broader `EXPLICIT` regex on name + text. Used for a sensitivity check. |

## Arms (where an idea comes from)

| Arm | Target / prompt | What it stands for |
|---|---|---|
| `REAL` | A held-out real past game; it and its near-duplicates are hidden. | Ceiling: how close the decoder gets to a point we know is a real game. |
| `INTERP` | Normalised midpoint of two past games whose cosine is below the corpus 10th percentile. | "Combine two distant concepts", done in geometry. |
| `NOISE` | Ladder: a real game x mixed with a random direction r, `norm((1-λ)x + λr)`. | Leaving the data cone step by step. |
| `CROWD` | Ladder: x pushed away from the centroid c of its 20 nearest games, `norm(x + λ(x - c))`. | Going to the periphery while staying inside the span of real games. |
| `FAR` | From 20,000 random directions, the 50 farthest from every past game. | The post's literal recipe: "a point far from all neighbours". |
| `PLAIN` | No target. Prompt: "invent a brand-new game, unlike any game that exists". | The LLM without the space. |
| `PAIR` | No target. Two random distant past games, "invent one new game that combines them". | The human heuristic (astronaut riding a horse), done in words. |
| `FUTURE_REAL` | Not generated: 50 random real games from the future split. | Judge calibration: what real, new games score. |

**Ladder level**: the target isolation that λ is solved for by bisection, one of 0.35, 0.45, 0.55, 0.70, 0.85
(10 targets per level). Along the `CROWD` path 0.85 is unreachable; the farthest point is taken (≈0.79).

**Decoder** (arms with a target): the LLM sees the 8 nearest visible past games with their similarity to the
target and writes a new game as JSON. The text is embedded, and the next turn only reports the similarity
reached and the best so far (score-only feedback). 4 iterations × 3 candidates; the best candidate is kept.

## Geometric metrics (`metrics.csv`, `summary.csv`)

| Column | Meaning |
|---|---|
| `landing` | cos(decoded text, target) for the best candidate. 1 = landed exactly on the target. |
| `first_landing` | The same for the first iteration: how much the feedback loop added. |
| `iso_target` | Isolation of the target = 1 − max cos to the visible past corpus. Real games ≈ 0.2–0.3. |
| `iso_decoded` | Isolation of the decoded text, same definition. |
| regression | The pattern `iso_target` ≫ `iso_decoded`: the text falls back into the populated region whatever was asked. |
| `past_sim` / `fut_sim` | Max cos of the decoded text to the past / future corpus. |
| `anticipation` | `fut_sim − past_sim`: > 0 means the idea resembles something released later more than anything earlier. Base rate = the same quantity for held-out real past games. |
| `growth_target` / `growth_decoded` | Region growth around a point: future items within the radius of the point's 50th nearest past item, divided by 50, divided by the global future/past ratio. 1 = grew like the whole catalogue, > 1 = faster. |
| `level`, `lam` | Ladder level and the λ that reaches it. |
| `iters` | Iterations recorded for the item. |
| intervals | `[lo, hi]` = 95% bootstrap interval of the median (geometry) or the mean (judges). |

## Judged quality (`judged.jsonl`, `judged_<judge>.jsonl`)

| Term | Meaning |
|---|---|
| pitch | Every item, real or generated, is first rewritten by `gpt-6-luna` as a neutral ~120-word pitch with no name and no marketing, so style cannot reveal where it came from. Judges only see pitches. |
| `coherence` | 1–10: the idea makes sense as a whole; its parts fit together. |
| `feasibility` | 1–10: a small studio could actually build it. |
| `appeal` | 1–10: a meaningful audience would want to play it. |
| `novelty` | 1–10: how different it is from the 5 nearest past games, which are shown as pitches too (10 = nothing like them). |
| judges | `sol` = gpt-5.6-sol through the gateway (the first judge, `judged.jsonl`); `opus`, `sonnet`, `haiku` = Claude Opus/Sonnet/Haiku 5.5 through Claude Code; `gpt61sol` = gpt-6.1-sol through Codex. Same rubric, pitch and neighbours for all of them. |
