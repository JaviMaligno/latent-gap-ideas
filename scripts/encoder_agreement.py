"""Row-by-row agreement of the local ONNX and the Kaggle Qwen3-Embedding-0.6B vectors on the continuation texts.

The iterative_critique matched baseline compared ideas embedded locally with corpus papers embedded on Kaggle and
flagged a possible encoder mismatch. This checks the texts embedded both ways (contexts, follow-ups, Part B ideas).
Output: docs/prereg/encoder_agreement.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402

D = M.DATA
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/encoder_agreement.json"


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    kinds = pd.read_parquet(D / "kaggle-upload/continuation-texts/texts.parquet")["kind"].to_numpy()
    kag = norm(np.load(D / "kaggle-output/embed-continuation-v1/emb_qwen3e06.npy").astype(np.float32))
    z = np.load(M.RUN / "emb.npz")
    local = {"ctx": z["C"], "fup": z["F"], "partB": np.load(M.RUN / "idea_emb.npy")}
    out = {}
    for kind, L in local.items():
        c = (norm(L.astype(np.float32)) * kag[kinds == kind]).sum(1)
        out[kind] = {"n": int(len(c)), "min_cos": float(c.min()), "median_cos": float(np.median(c))}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
