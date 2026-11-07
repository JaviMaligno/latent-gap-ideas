"""Joint call bootstrap: historical 25-minus-5 NONE shift (25-per-call run vs 5-per-call run) against the within-run
call-size effect C1 (docs/prereg/strengthen_cli_callsize.md). Computed after Codex review round 6; not pre-registered.
    OMP_NUM_THREADS=1 uv run python scripts/strengthen_cli_callsize_joint.py
Output: docs/prereg/strengthen_cli_callsize_joint.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs/prereg/revision_inspiration.json").read_text())["cli_clustered"]
out = {}
for g in ("codex", "claude"):
    old25 = np.array(R[g]["none_call_means"])  # 4 calls of 25 ideas
    p5 = pd.read_parquet(A.DATA / f"runs/inspiration/{g}-cli5/scored_qwen3e06.parquet")
    old5 = {c: x.margin.to_numpy() for c, x in p5[p5.source == "NONE"].groupby("call")}
    n = pd.read_parquet(A.DATA / f"runs/inspiration/{g}-clisize/scored_qwen3e06.parquet")
    n = n[n.source == "NONE"]
    new = {s: {c: x.margin.to_numpy() for c, x in n[n["size"] == s].groupby("call")} for s in (5, 25)}

    def m(d, ks):
        return np.concatenate([d[k] for k in ks]).mean()
    hist = old25.mean() - m(old5, list(old5))
    cur = m(new[25], list(new[25])) - m(new[5], list(new[5]))
    rng = np.random.default_rng(20261109)
    bs = []
    for _ in range(10000):
        h = old25[rng.integers(0, 4, 4)].mean() - m(old5, rng.choice(list(old5), len(old5)))
        c = m(new[25], rng.choice(list(new[25]), len(new[25]))) - m(new[5], rng.choice(list(new[5]), len(new[5])))
        bs.append((h, c, c - h))
    bs = np.array(bs)
    out[g] = {"historical_shift_25_minus_5": float(hist), "hist_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
              "current_C1": float(cur), "current_minus_historical": float(cur - hist),
              "ci": np.percentile(bs[:, 2], [2.5, 97.5]).tolist(), "n_old25_calls": 4, "n_old5_calls": len(old5)}
(ROOT / "docs/prereg/strengthen_cli_callsize_joint.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
