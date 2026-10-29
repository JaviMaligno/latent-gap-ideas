# Fluent search (two T4s; no corpus text, no LLM API)

Fluency-constrained version of `kaggle/support-search`: maximise cos(encoder(text), t) - lambda * mean LM NLL,
encoder Qwen3-Embedding-0.6B and LM Qwen3-1.7B-Base (same tokenizer). Pre-registration and results:
`docs/prereg/fluent_search.md`.

1. `uv run python scripts/fluent_prepare.py` writes `runs/fluent-search/inputs/` (targets only, plus 100 corpus texts
   per domain used only as an NLL reference) and `runs/fluent-search/refs.*` (ceilings, q99, REAL masks; local only).
2. Upload the inputs as the **private** dataset `javieraguilarmartin/latent-gap-fluent-inputs`:
   `kaggle datasets create -p ~/Documents/experiment-data/latent-gap-ideas/runs/fluent-search/inputs`
   (`kaggle datasets version -p ... -m <msg>` for a new version).
3. Edit only the `RUN` block of `search.py` (smoke vs full), then `kaggle kernels push -p kaggle/fluent-search`.
   `machine_shape: NvidiaTeslaT4` gives T4 x2; the script refuses to run with fewer than two GPUs. Two workers:
   worker i has the encoder on GPU i and the LM on GPU 1-i. Progress is printed per block (generation, every 10
   refinement steps, every saved target x lambda, ETA per target).
4. `kaggle kernels status javieraguilarmartin/latent-gap-fluent-search`; when complete,
   `kaggle kernels output javieraguilarmartin/latent-gap-fluent-search -p runs/fluent-search/<run>/kaggle-out`.
5. `uv run python scripts/fluent_gate.py --run <run>`: ONNX re-verification of every saved text, validity, gibberish
   flag, two-judge coherence gate (claude-haiku-5-5 then codex:gpt-6-luna, subscription CLIs) and the tables.

Local CPU check of the kernel code (no GPU): `FLUENT_CPU=1 FLUENT_IN=<dir> FLUENT_OUT=<dir> FLUENT_RUN='{...}'
python kaggle/fluent-search/search.py` in an environment with torch and transformers 4.57.1.
