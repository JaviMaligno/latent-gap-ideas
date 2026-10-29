"""Agreement of the human ratings (artifact `ratings` collection, exported as one JSON per item) with each judge.

    uv run python scripts/human_agreement.py --ratings <dir with ARM-k.json files>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
CRITERIA = ["coherence", "feasibility", "appeal", "novelty"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--run", default="pilot")
    args = ap.parse_args()
    h = pd.DataFrame([(lambda r: r.get("data", r))(json.loads(f.read_text())) for f in Path(args.ratings).glob("*.json")])
    rows = []
    for f in (DATA / "runs/steam" / args.run).glob("judged_*.jsonl"):
        for line in f.read_text().splitlines():
            x = json.loads(line)
            s = x.get("scores", x)
            rows.append({"judge": f.stem.removeprefix("judged_"), "arm": x["arm"], "k": x["k"], **{c: s[c] for c in CRITERIA}})
    j = pd.DataFrame(rows).merge(h[["arm", "k"]], on=["arm", "k"])
    m = h.merge(j.groupby(["arm", "k"])[CRITERIA].mean().reset_index(), on=["arm", "k"], suffixes=("_h", "_j"))
    print(f"{len(h)} human ratings")
    for c in CRITERIA:
        print(f"  {c:11s} human {m[c + '_h'].mean():.2f}  judges {m[c + '_j'].mean():.2f}  "
              f"rho {spearmanr(m[c + '_h'], m[c + '_j']).correlation:+.2f}  MAE {np.abs(m[c + '_h'] - m[c + '_j']).mean():.2f}")
    print("per judge, spearman with the human (" + " / ".join(CRITERIA) + "); judge vs the other judges on novelty")
    for name, g in j.groupby("judge"):
        x = g.merge(h, on=["arm", "k"], suffixes=("_j", "_h"))
        others = j[j.judge != name].groupby(["arm", "k"]).novelty.mean()
        loo = spearmanr(g.set_index(["arm", "k"]).novelty, others.loc[g.set_index(["arm", "k"]).index]).correlation
        print(f"  {name:9s} " + " ".join(f"{spearmanr(x[c + '_h'], x[c + '_j']).correlation:+.2f}" for c in CRITERIA)
              + f" | {loo:+.2f}")
    print(m.groupby("arm")[[c + s for c in CRITERIA for s in ("_h", "_j")]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
