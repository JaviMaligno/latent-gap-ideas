"""Exploratory: how close does any future item get to the saved FAR targets (Qwen3)?

F1 of where_future_lands.md measures each future item's maximum cosine to the past (none <= 0.15). This measures the
quantity the abstract's claim is about: for each saved FAR target, the maximum cosine of any future item to it, next to
the same statistic for the past corpus it was built against (its corpus ceiling).
    uv run python scripts/far_targets_vs_future.py
Output: docs/prereg/far_targets_vs_future.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/far_targets_vs_future.json"


def norm(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def main() -> None:
    res = {}
    for dom in ["steam", "arxiv", "patents"]:
        meta = pd.read_parquet(DATA / f"prepared/{dom}/corpus.parquet")
        E = norm(np.load(DATA / f"prepared/{dom}/emb_qwen3e06.npy").astype(np.float32))
        keep = ~meta["adult"].to_numpy() if "adult" in meta else np.ones(len(meta), bool)
        past = (meta["split"] == "past").to_numpy() & keep
        fut = (meta["split"] != "past").to_numpy() & keep
        T = norm(np.load(DATA / f"runs/{dom}/grid-gpt-6-luna/targets.npz")["FAR"].astype(np.float32))
        cf = (E[fut] @ T.T).max(0)
        cp = (E[past] @ T.T).max(0)
        res[dom] = {"n_targets": int(len(T)), "n_future": int(fut.sum()), "n_past": int(past.sum()),
                    "future_max_cos_to_target": {"median": float(np.median(cf)), "max": float(cf.max())},
                    "past_max_cos_to_target": {"median": float(np.median(cp)), "max": float(cp.max())},
                    "share_targets_future_above_past": float((cf > cp).mean())}
        print(dom, json.dumps(res[dom]), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
