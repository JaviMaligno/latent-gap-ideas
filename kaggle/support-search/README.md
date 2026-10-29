# Support search (two T4s; no decoder calls)

1. Run `uv run python scripts/bounds/support_probe.py` from the repository root.
2. Upload `~/Documents/experiment-data/latent-gap-ideas/runs/support-search/` as the
   **private** dataset `javieraguilarmartin/latent-gap-support-inputs` (metadata is generated).
   The kernel reads only `support_inputs.json`; it does not need the full corpora.
3. Push this directory with `kaggle kernels push -p kaggle/support-search` and select
   **GPU T4 x2**. The script explicitly checks for two GPUs and uses both. Metadata follows
   the repository's embed-steam format; the accelerator choice must be T4 x2 in Kaggle.
4. Download `witness_gpu0.jsonl` and `witness_gpu1.jsonl`. Run:

   ```sh
   uv run python scripts/bounds/verify_support.py /path/to/witness_gpu0.jsonl /path/to/witness_gpu1.jsonl
   ```

   Verification defaults to five-minute batches and caches by text and encoder identity.
   Rerun the same command to resume until it reports all supplied witnesses verified.

5. Two independent raters fill `rater1` / `rater2` in a copy of `coherence_blind.csv`,
   judging the exact displayed text as an intelligible description of the indicated
   domain. Use 1=yes, 0=no; leave missing ratings blank. Rerun verification with
   `--judgments /path/to/completed.csv`. Unrated witnesses never become h_coh bounds.

Expected allocation: 200 targets, 45 seconds each, divided between two workers (~75
minutes plus loading and in-flight blocks). Workers stop starting targets after 100
minutes from launch; a parent watchdog kills remaining work at 110 minutes. Partial
JSONL files are usable; compare coverage by domain and arm. A hard kill can lose the
current target but preserves previous targets. Model download / CPU startup / OOM can
still fail: the kernel has not been run on this machine's unavailable GPUs.

FP16 gradient search is a proposal mechanism. Only local ONNX scores of the saved,
round-tripped strings are reported as empirical lower bounds. No upper bound, human
coherence label, or language-generator likelihood is inferred from search failure.
