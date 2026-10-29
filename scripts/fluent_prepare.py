"""Inputs and reference values for the fluency-constrained GPU search (docs/prereg/fluent_search.md).

Writes runs/fluent-search/inputs/ (uploaded as the private Kaggle dataset javieraguilarmartin/latent-gap-fluent-inputs):
  fluent_inputs.json  jobs = FAR/GAP k=0..20 and REAL k=0..10 per domain (k=0 is smoke only): target vector only.
                      The kernel never sees a corpus item that could act as a neighbour: no seeds, no texts.
  nll_reference.json  100 random visible corpus texts per domain (REAL items and their near-duplicates removed), only
                      to place the witnesses' LM NLL against the corpus's own fluency. Never used as seeds.
and runs/fluent-search/refs.csv (local only): per target ceiling, held-out ceiling, corpus q99, with the REAL item and
its near-duplicates masked exactly as scripts/coherent_search.py does.

Usage: uv run python scripts/fluent_prepare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.targets import DUP_COS

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
KS = {"FAR": range(0, 21), "GAP": range(0, 21), "REAL": range(0, 21)}  # REAL k=11..20: seed_rewrite tuning set
N_NLL_REF = 100


def main() -> None:
    out = DATA / "runs/fluent-search"
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    jobs, refs, nll_ref = [], [], []
    for domain in ("steam", "arxiv"):
        root = DATA / "prepared" / domain
        meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "text"])
        E = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        keep_idx = np.flatnonzero((~meta.adult & (meta.split == "past")).to_numpy())
        held_idx = np.flatnonzero((~meta.adult & (meta.split != "past")).to_numpy())
        P, H = E[keep_idx], E[held_idx]
        texts = meta.text.to_numpy()[keep_idx]
        del E
        with np.load(DATA / f"runs/{domain}/pilot/targets.npz", allow_pickle=True) as z:
            tg = {a: z[a] for a in KS}
            hidden = {k: np.asarray(v, dtype=int) for k, v in enumerate(z["REAL_hidden"])}
        banned = set()
        for arm, ks in KS.items():
            for k in ks:
                t = tg[arm][k].astype(np.float32)
                t /= np.linalg.norm(t)
                # Masking copied from scripts/coherent_search.py (one()): REAL_hidden = positions in P (the item and
                # its near-duplicates); also every visible item with cos > DUP_COS; held-out near-duplicates dropped.
                sims = P @ t
                hs = H @ t
                if arm == "REAL":
                    hidden_pos = np.union1d(hidden[k], np.flatnonzero(sims > DUP_COS)).astype(int)
                    hs = hs[hs <= DUP_COS]
                    banned |= set(hidden_pos.tolist())
                else:
                    hidden_pos = np.array([], dtype=int)
                sims[hidden_pos] = -np.inf
                refs.append({"domain": domain, "arm": arm, "k": k, "ceiling": float(sims.max()),
                             "corpus_q99": float(np.quantile(sims[np.isfinite(sims)], 0.99)),
                             "heldout_ceiling": float(hs.max()), "n_masked": len(hidden_pos),
                             "hidden_pos": hidden_pos.tolist()})
                jobs.append({"domain": domain, "arm": arm, "k": k, "target": t.tolist()})
        rng = np.random.default_rng(20261026)
        pool = np.setdiff1d(np.arange(len(P)), np.array(sorted(banned), dtype=int))
        for i in rng.choice(pool, N_NLL_REF, replace=False):
            nll_ref.append({"domain": domain, "text": str(texts[i])})
        print(domain, "jobs", sum(j["domain"] == domain for j in jobs), flush=True)
    (inp / "fluent_inputs.json").write_text(json.dumps(jobs))
    (inp / "nll_reference.json").write_text(json.dumps(nll_ref))
    (inp / "dataset-metadata.json").write_text(json.dumps(
        {"title": "latent-gap-fluent-inputs", "id": "javieraguilarmartin/latent-gap-fluent-inputs",
         "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    pd.DataFrame(refs).to_json(out / "refs.jsonl", orient="records", lines=True)
    pd.DataFrame(refs).drop(columns="hidden_pos").to_csv(out / "refs.csv", index=False)
    print(f"wrote {inp} and {out / 'refs.csv'}", flush=True)


if __name__ == "__main__":
    main()
