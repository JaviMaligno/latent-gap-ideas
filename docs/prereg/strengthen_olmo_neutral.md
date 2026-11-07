# OLMo NEUTRAL revision arm: is OLMo's follow-up-specific turn feedback-specific? (pre-registered)

Written and committed 2026-11-05, before any generation of this arm. Kernel: `kaggle/olmo-neutral/run.py`
(Kaggle `javieraguilarmartin/latent-gap-olmo-neutral`). Analysis: `scripts/strengthen_olmo_neutral.py`. Output:
`docs/prereg/strengthen_olmo_neutral.json`. Mirrors `docs/prereg/neutral_revision.md` (gpt-4o), whose design, tests and
results are already known to me; nothing of this arm has been generated or looked at.

## Why

The paper (Sections 5.2-5.3, Table 3, limitations) says OLMo's follow-up-specific changes (+0.012 to +0.032; the
reviewer's in Qwen3 survives Holm) are untested against plain revision, because only gpt-4o has a NEUTRAL arm. For
gpt-4o, NEUTRAL reproduced the whole turn, follow-up-specific part included, and no critique arm turned the ideas
detectably more (N5, N6). Whether that holds for OLMo is open.

## Design (fixed)

- **Generator and settings: identical to OLMo's critique arms** (`kaggle/llama-continuation/run.py`, output
  `kaggle-output/llama-continuation-v1/iter_llama.jsonl`): `allenai/OLMo-7B-0424-Instruct-hf`, float16,
  `device_map="auto"` over 2 × T4, transformers sampling at temperature 1, top_p 1, max 450 new tokens, 2,048-token
  window (prompt truncated from the left as there), batch 12, system text prepended to the user turn, the same parser
  (`parse`) and up to 2 retries of a failed parse. Prompt functions (`prompt`, `revise_prompt`) copied verbatim.
- **Contexts:** the same 150 contexts (`latent-gap-llama-inputs/contexts.json`, unchanged dataset).
- **One idea per context, 3 rounds**, as OLMo's critique arms (N_IDEAS_ITER = 1; gpt-4o used 3).
- **NEUTRAL feedback, every round:** the same content-free text as gpt-4o: "Revise your proposal: write an improved
  version." (`scripts/iterative_critique.py`, `NEUTRAL_FB`), inserted with the same `revise_prompt`.
- **Round 0 reused from GEO**, as for gpt-4o: NEUTRAL's round-0 idea of context i is OLMo's GEO round-0 idea of
  context i (k = 0), copied from `iter_llama.jsonl` into the new dataset `latent-gap-olmo-neutral-inputs`
  (`geo_r0.jsonl`) and marked `r0_from: GEO`. Only the 2 revisions are new: 300 generations. Consequence, stated in
  advance: NEUTRAL and GEO share round 0, so NEUTRAL vs GEO compares round 2 from the same starting ideas; NEUTRAL vs
  Reviewer and vs ORACLE compare changes from independent round-0 draws.
- **Empty parse:** as in OLMo's critique run, a round whose output fails all parses gets an empty record; the next
  revision then has empty feedback. Rows without an abstract are dropped; a context missing round 0 or round 2 in an
  arm is dropped from that arm's contrasts (NaN), as in `revision_neutral.py`.
- **Seed:** `torch.manual_seed(20261105)` (the critique run used 20261030; the seed only fixes which samples are drawn).
- **Embedding in the kernel**, with the critique run's settings (sentence-transformers, float16 on T4,
  `max_seq_length` 512, normalised, `item_text` with the abstract cut at 1,200 characters): Qwen3-Embedding-0.6B
  (primary) and BGE-M3 (robustness). Embedded: the 150 context texts and every NEUTRAL row with an abstract (rounds 0-2).
  Analysis uses the critique run's context and follow-up vectors for every arm, its idea vectors for GEO / Reviewer /
  ORACLE, and the new vectors for all NEUTRAL rounds (so NEUTRAL's round 0 → round 2 change is within one encoder run,
  as for gpt-4o). Reported: agreement of the new and old vectors on the 150 shared round-0 texts and on the 150
  contexts (min and mean cos). If the minimum is below 0.999 in an embedder, that is recorded as a deviation and the
  embedder's results are reported with that caveat.
- **Run rule:** no separate smoke test. The first kernel version that completes all 300 generations and both
  embeddings is the data. A version that fails before writing its outputs is fixed and re-pushed (logged in
  `deviations`); outputs of a completed version are never discarded.
- **Outputs (new kernel, existing outputs untouched):** `iter_olmo_neutral.jsonl`, `neutral_emb_qwen3e06.npy`,
  `neutral_emb_bgem3.npy`, `ctx_emb_{space}.npy`, `raw_generations_neutral.jsonl`, downloaded to
  `~/Documents/experiment-data/latent-gap-ideas/kaggle-output/olmo-neutral-v1/`.

## Measures (as `neutral_revision.md` and Table 3)

Per context and arm (one idea): step = cos(idea, context); fit = max cos to the context's follow-ups; hit = the
nearest neighbour among follow-ups ∪ same-primary-category non-past, non-follow-up papers is a follow-up; tangent
alignment with the follow-ups' unit mean tangent f; change along p (unit mean tangent of the nearest future-split papers
outside the saved follow-ups, as many as the context has follow-ups) and along the follow-up-specific direction
s = unit(f − (f·p)p). Δ_A = round 2 − round 0 (step as reduction, round 0 − round 2: positive = longer step).

## Confirmatory tests (14 per embedder, Holm within embedder; Qwen3 primary, BGE-M3 robustness with its own Holm)

Paired over contexts, d = Δ_A − Δ_NEUTRAL; Wilcoxon signed-rank; bootstrap 95 % CI (2,000 resamples over contexts,
seed 20261030, as there).

- **O1:** GEO lengthens the step more than NEUTRAL (one-sided, greater).
- **O2:** the Reviewer lengthens the step more than NEUTRAL (one-sided, greater).
- **O3 (two-sided):** fit change, each critique arm vs NEUTRAL (GEO, Reviewer, ORACLE).
- **O4 (two-sided):** hit change, three tests.
- **O5 (two-sided):** tangent-alignment change, three tests.
- **O6 (two-sided):** change along s (follow-up-specific), three tests. This is the question the paper leaves open.

Predictions: O1 and O2 hold (critique lengthens the step beyond revision, as for gpt-4o). O3-O6 open; the gpt-4o
reading ("no feedback-specific turn detected") predicts no O6 contrast significant after Holm.

Also reported (not in the 14): NEUTRAL's own round 0 → 2 changes per measure (Wilcoxon, as Tables 2-3); for each
critique arm the share of its change reproduced by NEUTRAL (Δ_NEUTRAL / Δ_A, step and along s); exploratory contrasts
along p and ORACLE − NEUTRAL on the step (two-sided); and the minimum detectable mean difference for each O6 contrast
(80 % power, two-sided alpha 0.05, realised SD and n; t-based and Wilcoxon-approximate, as in
`strengthen_continuations_full`).

## Interpretation rules (fixed now)

- **O6, per critique arm (Qwen3 primary):**
  - Holm p < 0.05 and d > 0: that arm's follow-up-specific turn for OLMo is feedback-specific; the paper must say the
    gpt-4o null does not carry over to OLMo for that arm.
  - Holm p < 0.05 and d < 0: the critique turns OLMo's ideas less than plain revision does (as ORACLE did for gpt-4o).
  - Not significant and the 95 % CI inside ±0.015 (the post-hoc bound of Section 5.3): consistent with no
    feedback-specific turn of that size, as for gpt-4o.
  - Not significant and the CI crosses ±0.015: insufficient; the paper reports the CI and the MDE and says the
    OLMo turn is not resolved at that size.
  - BGE-M3 is read with the same rules; a result that holds in Qwen3 but not BGE-M3 is reported as encoder-dependent.
- **O1/O2:** hold if Holm p < 0.05; otherwise "critique did not lengthen OLMo's step beyond revision (detected)".
- **O3-O5:** Holm p < 0.05 = the critique arm differs from plain revision on that measure; otherwise not detected.
- Whatever the outcome, the paper's sentences "OLMo has no NEUTRAL arm, so its follow-up-specific changes are untested
  against plain revision" (Section 5.3) and the limitation "The neutral control covers gpt-4o only" are replaced by
  this result.

## Descriptive: F4 on contexts with continuations (no confirmatory reading)

Same construction as F4 of `strengthen_continuations_full.md` (labels from
`docs/prereg/strengthen_continuations_full_labels.json`): the 17 contexts with >= 2 consensus continuations
(both labellers CONTINUES; contexts 6, 8, 10, 23, 30, 37, 53, 58, 59, 64, 83, 114, 118, 133, 141, 143, 148). f from
the continuations' tangents, p from the nearest future-split papers outside the saved follow-ups (as many as
continuations; robustness: as many as follow-ups), s = unit(f − (f·p)p). For OLMo, each embedder: change along s from
round 0 to round 2 for GEO, Reviewer, ORACLE and NEUTRAL; each critique arm minus NEUTRAL with mean, bootstrap 95 % CI
and uncorrected two-sided Wilcoxon p; positive control = leave-one-out alignment of the continuations; the original
all-follow-up s change on the same contexts. Also the lenient set (either labeller CONTINUES, >= 2: 36 contexts).
The prior pre-registration made only gpt-4o / Qwen3 confirmatory; with one OLMo idea per context and n = 17 this is
reported as description only, with its CIs read against ±0.015 but no claim drawn from a null.

## Deviations from the gpt-4o NEUTRAL arm (known in advance)

1. One idea per context instead of 3 (OLMo's critique arms have one), so per-context changes are noisier; the MDE is
   reported.
2. Generated on Kaggle (2 × T4, transformers) instead of the gateway; embedded on Kaggle T4 float16 with the critique
   run's settings instead of local ONNX / CPU. BGE-M3 is therefore available with the same settings and is reported.
3. Different sampling seed (above).
4. New kernel and dataset; the critique run's kernel, dataset and outputs are not modified.

## Cost

No paid API. Kaggle GPU only: 300 OLMo generations (the critique run's revisions took about 6 s each on 2 × T4,
batch 12: about 15 min per round of 150) plus model download and load (about 4 min) and two embedding passes of
about 600 texts (about 2 min): about 40-50 min of one 2 × T4 session. Analysis runs locally on saved vectors (no model
loaded; it reads the corpus embeddings, so it runs when no other heavy local process is running).
