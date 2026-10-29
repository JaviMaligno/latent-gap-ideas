"""Correlation matrices between criteria for the owner's ratings, beside the judges' on the same items.

Batches: Steam B1 (human1, no items.json kept: human only), B2, B3; arXiv A2; patents P1 (when rated). Each rating file
is ratings/<ARM>-<k>.json or <BATCH>-<ARM>-<k>.json; judges come from the batch's items.json (mean of judges).
n is 16 per batch, so per-batch matrices are noisy; the pooled matrix per domain is the one to read.

    uv run python scripts/human_criteria.py
"""
import json
from pathlib import Path

import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs"
CRIT = ["coherence", "feasibility", "appeal", "novelty"]
BATCHES = [("steam", "human1"), ("steam", "human2"), ("steam", "human3"), ("arxiv", "humanA2"), ("patents", "humanP1")]


def load(domain: str, batch: str) -> pd.DataFrame:
    root = DATA / domain / batch
    rdir = root / "ratings" if (root / "ratings").exists() else root
    files = sorted(f for f in rdir.glob("*.json") if f.name != "items.json")
    if not files:
        return pd.DataFrame()
    h = pd.DataFrame([(lambda r: r.get("data", r))(json.loads(f.read_text())) for f in files])
    h = h.dropna(subset=CRIT)
    h["key"] = h.arm + "-" + h.k.astype(int).astype(str)
    if (root / "items.json").exists():
        items = json.loads((root / "items.json").read_text())
        jm = {f"{i['arm']}-{i['k']}": {c: pd.Series([v[c] for v in i["judges"].values() if c in v]).mean()
                                         for c in CRIT} for i in items if i.get("judges")}
        for c in CRIT:
            h[f"j_{c}"] = h.key.map(lambda k: jm.get(k, {}).get(c))
    h["domain"], h["batch"] = domain, batch
    return h


def show(title: str, h: pd.DataFrame) -> None:
    print(f"\n== {title} (n={len(h)})")
    print("owner:\n" + h[CRIT].corr("spearman").round(2).to_string())
    jc = [f"j_{c}" for c in CRIT]
    if all(c in h for c in jc) and h[jc].notna().all(axis=None):
        j = h[jc].rename(columns=lambda c: c[2:])
        print("judges, same items:\n" + j.corr("spearman").round(2).to_string())


all_h = [h for d, b in BATCHES if len(h := load(d, b))]
H = pd.concat(all_h)
for (d, b), h in H.groupby(["domain", "batch"]):
    show(f"{d} {b}", h)
for d, h in H.groupby("domain"):
    if h.batch.nunique() > 1:
        show(f"{d} pooled", h)
show("all domains pooled", H)
