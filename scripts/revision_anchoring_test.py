"""Exploratory test of anchoring in the inspiration study (gpt-4o): steam_game stimuli vs NONE and vs other sources.

  uv run python scripts/revision_anchoring_test.py  ->  docs/prereg/revision_anchoring_test.json
"""
import json
from pathlib import Path

import pandas as pd
from scipy import stats

D = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/inspiration/gpt-4o-0806"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/revision_anchoring_test.json"


def holm(ps: dict) -> dict:
    order, out, run = sorted(ps, key=ps.get), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


res = {}
for e in ["qwen3e06", "bgem3"]:
    d = pd.read_parquet(D / f"scored_{e}.parquet")
    none = d[d.arm == "NONE"].margin
    for fr in ["INCIDENTAL", "INSTRUCTED"]:
        x = d[d.arm == fr]
        ps = {s: float(stats.mannwhitneyu(g.margin, none).pvalue) for s, g in x.groupby("source")}
        means = x.groupby("source").margin.mean().sort_values()
        g, o = x[x.source == "steam_game"].margin, x[x.source != "steam_game"].margin
        res[f"{e}/{fr}"] = {
            "steam_mean": float(g.mean()), "none_mean": float(none.mean()),
            "steam_vs_none_p": ps["steam_game"], "steam_vs_none_holm9": holm(ps)["steam_game"],
            "steam_minus_others": float(g.mean() - o.mean()),
            "steam_vs_others_p": float(stats.mannwhitneyu(g, o).pvalue),
            "rank_from_worst_of_9": int(list(means.index).index("steam_game") + 1),
        }
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
