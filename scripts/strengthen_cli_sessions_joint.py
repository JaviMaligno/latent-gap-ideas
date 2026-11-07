"""Post-results implementation of the pre-registered interpretation rule for S1 (docs/prereg/strengthen_cli_sessions.md, "Interpretation fixed in advance"): the
session-averaged 25-minus-5 NONE effect against the historical 25-minus-5 shift and its own uncertainty, with a joint
call bootstrap (historical calls resampled as in strengthen_cli_callsize_joint.py; session calls resampled within
session x size, as in the registered S1 bootstrap).
    uv run python scripts/strengthen_cli_sessions_joint.py
Also writes the registered descriptive comparison: earlier runs' NONE baselines next to the eight session means.
Output: docs/prereg/strengthen_cli_sessions_joint.json
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
P = json.loads((ROOT / "docs/prereg/strengthen_cli_powered.json").read_text())
C = json.loads((ROOT / "docs/prereg/strengthen_cli_callsize.json").read_text())
out = {}
for g in ("codex", "claude"):
    old25 = np.array(R[g]["none_call_means"])
    p5 = pd.read_parquet(A.DATA / f"runs/inspiration/{g}-cli5/scored_qwen3e06.parquet")
    old5 = [x.margin.to_numpy() for _, x in p5[p5.source == "NONE"].groupby("call")]
    d = pd.read_parquet(A.DATA / f"runs/inspiration/{g}-clisessions/scored_qwen3e06.parquet")
    groups = {k: [x["margin"].to_numpy() for _, x in gg.groupby("cl")] for k, gg in d.groupby(["session", "size"])}
    S = sorted({k[0] for k in groups})

    def sess(gr):
        return np.mean([np.concatenate(gr[(k, 25)]).mean() - np.concatenate(gr[(k, 5)]).mean() for k in S])
    hist = old25.mean() - np.concatenate(old5).mean()
    cur = sess(groups)
    rng = np.random.default_rng(20261111)
    bs = []
    for _ in range(10000):
        h = old25[rng.integers(0, len(old25), len(old25))].mean() - \
            np.concatenate([old5[i] for i in rng.integers(0, len(old5), len(old5))]).mean()
        c = sess({k: [v[i] for i in rng.integers(0, len(v), len(v))] for k, v in groups.items()})
        bs.append((h, c, c - h))
    bs = np.array(bs)
    out[g] = {"historical_shift_25_minus_5": float(hist), "hist_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
              "sessions_effect": float(cur), "sessions_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
              "sessions_minus_historical": float(cur - hist), "ci": np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
              "n_old25_calls": len(old25), "n_old5_calls": len(old5), "n_sessions": len(S),
              "descriptive_runs": {
                  "run1_original_25_per_call": float(old25.mean()),
                  "run2_powered_5_per_call": float(P[g]["none_margin"]),
                  "run3_within_run_5": float(C[g]["cells"]["NONE-5"]["mean"]),
                  "run3_within_run_25": float(C[g]["cells"]["NONE-25"]["mean"]),
                  "sessions_5": {int(k): float(x.margin.mean()) for k, x in d[d["size"] == 5].groupby("session")},
                  "sessions_25": {int(k): float(x.margin.mean()) for k, x in d[d["size"] == 25].groupby("session")},
                  "run2_minus_run3_at_5": float(P[g]["none_margin"] - C[g]["cells"]["NONE-5"]["mean"])}}
(ROOT / "docs/prereg/strengthen_cli_sessions_joint.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
