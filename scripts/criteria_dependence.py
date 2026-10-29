"""How much coherence drives the other judge criteria, and arm comparisons adjusted for it.

Item level = mean of the judges on one item. Per domain and generator run: Spearman between criteria; then, per arm,
the raw mean of feasibility / appeal / novelty, the mean among coherent items only (coherence >= 7), and the
coherence-adjusted mean from OLS  criterion ~ arm + coherence  (evaluated at the domain's mean coherence).
Adjusting for coherence estimates the arm's effect at equal coherence; coherence is itself caused by the arm, so the
adjusted numbers are a "direct effect" reading, not a replacement for the raw ones.

    uv run python scripts/criteria_dependence.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs"
CRIT = ["coherence", "feasibility", "appeal", "novelty"]

rows = []
for d in ("steam", "arxiv", "patents"):
    for run in sorted((DATA / d).glob("grid-*")):
        files = [f for f in run.glob("judged_*.jsonl") if f.stem.count("_") == 1 or "noresults" in f.stem]
        if not files:
            continue
        J = pd.concat([pd.read_json(f, lines=True) for f in files])
        J = J[J.judge.notna()] if "judge" in J else J
        it = J.groupby(["arm", "k"])[CRIT].mean().reset_index()
        it["domain"], it["run"] = d, run.name.removeprefix("grid-")
        rows.append(it)
df = pd.concat(rows)

print("Spearman between criteria, item level (pooled arms), per domain")
for d, g in df.groupby("domain"):
    c = g[CRIT].corr("spearman").round(2)
    print(f"\n{d} (n={len(g)})\n{c.to_string()}")

print("\nPer arm: raw mean | coherent-only (coh>=7) mean [n] | coherence-adjusted mean")
for (d, r), g in df.groupby(["domain", "run"]):
    X = pd.get_dummies(g.arm, drop_first=False).astype(float)
    X["coh"] = g.coherence - g.coherence.mean()
    out = []
    for c in ("feasibility", "appeal", "novelty"):
        beta, *_ = np.linalg.lstsq(X.to_numpy(), g[c].to_numpy(), rcond=None)
        adj = dict(zip(X.columns[:-1], beta[:-1]))
        for a, ga in g.groupby("arm"):
            co = ga[ga.coherence >= 7]
            out.append({"crit": c, "arm": a, "raw": ga[c].mean(), "coh7": co[c].mean() if len(co) else np.nan,
                        "n7": len(co), "adj": adj[a], "slope": beta[-1]})
    o = pd.DataFrame(out)
    print(f"\n== {d} / {r}  (coherence of arm: " +
          ", ".join(f"{a} {v:.1f}" for a, v in g.groupby('arm').coherence.mean().sort_values().items()) + ")")
    for c, oc in o.groupby("crit", sort=False):
        oc = oc.sort_values("raw", ascending=False)
        print(f"  {c:11s} slope/coh pt {oc.slope.iloc[0]:+.2f} | " +
              "  ".join(f"{a} {x.raw:.1f}|{x.coh7:.1f}[{x.n7}]|{x.adj:.1f}" for a, x in oc.set_index('arm').iterrows()))
