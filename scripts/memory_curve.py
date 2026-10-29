"""MEM curves (docs/prereg/memory_probe.md): reconstruction of hidden real items per half-year window, per generator.

landing = cos(best decoded, target). Anchoring covariate: cos(centroid of the shown neighbours, target), which
predicts landing for an item the generator does not know (bounds.md §3). excess = landing - anchoring fit, the fit
taken on the windows the generator cannot have seen is not available for the newest models, so the fit is pooled
over all generators' 2022H1 windows... no: the fit is per generator on all its items (a level shift at the cutoff
survives a per-generator fit, since the fit only absorbs the anchoring slope).

    uv run python scripts/memory_curve.py --domain arxiv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
GENS = ("gpt-4o-0806", "gpt-6-luna", "claude-sonnet-5-5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="arxiv")
    args = ap.parse_args()
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    P = emb[(meta["split"] == "past").to_numpy()]

    rows = []
    for g in GENS:
        run = DATA / "runs" / args.domain / f"grid-{g}"
        tg = np.load(run / "targets.npz", allow_pickle=True)
        for line in (run / "items.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["arm"] != "MEM" or not r.get("best"):
                continue
            t = tg["MEM"][r["k"]]
            c = P[r["neighbours"]].mean(0)
            rows.append({"gen": g, "k": r["k"], "window": str(tg["MEM_window"][r["k"]]), "landing": r["best"]["sim"],
                         "anchor": float(c @ t / np.linalg.norm(c)), "nn1": float((P[r["neighbours"]] @ t).max())})
    df = pd.DataFrame(rows)
    for g, gg in df.groupby("gen"):
        a, b = np.polyfit(gg["anchor"], gg["landing"], 1)
        df.loc[gg.index, "excess"] = gg["landing"] - (a * gg["anchor"] + b)
    pd.set_option("display.width", 200)
    for col in ("landing", "excess", "nn1"):
        t = df.pivot_table(index="window", columns="gen", values=col, aggfunc="mean")[list(GENS)]
        print(f"\n{col} (mean per window)\n" + t.round(3).to_string())
    se = df.groupby(["gen", "window"])["excess"].sem().unstack(0)[list(GENS)]
    print("\nexcess s.e.m. per window (typical)\n" + se.median().round(3).to_string())
    out = DATA / "runs" / args.domain / "memory_curve.csv"
    df.to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
