"""Compare judges on the same items: per-arm means per judge, item-level agreement, and the arm ranking.

    uv run python scripts/judge_compare.py --run pilot
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from judge import DATA  # noqa: E402

CRITERIA = ("coherence", "feasibility", "appeal", "novelty")
ORDER = ["FUTURE_REAL", "REAL", "INTERP", "NOISE", "CROWD", "GAP", "FAR", "PAIR", "PLAIN", "PLAIND"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--run", default="pilot")
    args = ap.parse_args()
    run = DATA / "runs" / args.domain / args.run

    frames = [pd.read_json(p, lines=True) for p in sorted(run.glob("judged_*.jsonl"))]
    df = pd.concat(frames)[["arm", "k", "judge", *CRITERIA]]
    judges = list(dict.fromkeys(df["judge"]))
    common = df.groupby(["arm", "k"])["judge"].nunique()
    common = common[common == len(judges)].index  # items every judge scored
    df = df.set_index(["arm", "k"]).loc[common].reset_index()
    print(f"judges: {judges}; items scored by all: {len(common)}\n")

    for c in CRITERIA:
        t = df.pivot_table(index="arm", columns="judge", values=c, aggfunc="mean").reindex(ORDER)[judges]
        t["mean"] = t.mean(axis=1)
        print(f"{c} (mean per arm)")
        print(t.round(2).to_string(), "\n")

    print("item-level Spearman between judges")
    for c in ("coherence", "novelty"):
        w = df.pivot_table(index=["arm", "k"], columns="judge", values=c)[judges]
        pairs = {f"{a}~{b}": w[a].corr(w[b], method="spearman") for a, b in itertools.combinations(judges, 2)}
        print(f"  {c}: " + ", ".join(f"{k} {v:.2f}" for k, v in pairs.items()))

    print("\narm ranking by novelty per judge (highest first)")
    t = df.pivot_table(index="arm", columns="judge", values="novelty", aggfunc="mean")
    for j in judges:
        print(f"  {j:9s} " + " > ".join(t[j].sort_values(ascending=False).index))

    # Is PLAIN the most novel generated arm for every judge? Bootstrap over items, per judge.
    rng = np.random.default_rng(0)
    print("\nPLAIN minus best guided arm in novelty (bootstrap 95% CI), per judge")
    for j in judges:
        g = df[df["judge"] == j]
        plain = g.loc[g["arm"] == "PLAIN", "novelty"].to_numpy(float)
        best = max(("INTERP", "FAR", "NOISE", "CROWD", "GAP"), key=lambda a: g.loc[g["arm"] == a, "novelty"].mean())
        other = g.loc[g["arm"] == best, "novelty"].to_numpy(float)
        d = [rng.choice(plain, len(plain)).mean() - rng.choice(other, len(other)).mean() for _ in range(4000)]
        print(f"  {j:9s} vs {best:6s}: {plain.mean() - other.mean():+.2f} "
              f"[{np.quantile(d, .025):+.2f},{np.quantile(d, .975):+.2f}]")


if __name__ == "__main__":
    main()
