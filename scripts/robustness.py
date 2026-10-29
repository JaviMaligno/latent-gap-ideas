"""Target-free metrics of a run's outputs re-measured in the robustness space (bge-m3): isolation, anticipation,
region growth. Landing cannot be re-measured: targets only exist in the primary space.

    uv run python scripts/robustness.py --run pilot --emb ~/.../kaggle-output/embed-outputs-v1/emb_bgem3.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import DATA, boot_median, max_sim, region_growth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--emb", required=True, help="output embeddings, rows in the order of the outputs texts.parquet")
    ap.add_argument("--space", default="bgem3")
    args = ap.parse_args()

    root = DATA / "prepared" / args.domain
    run = DATA / "runs" / args.domain / args.run
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
    emb = np.load(root / f"emb_{args.space}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    ratio = len(F) / len(P)
    tg = np.load(run / "targets.npz", allow_pickle=True)

    outs = pd.read_parquet(DATA / "kaggle-input" / f"{args.domain}-{args.run}-outputs" / "texts.parquet")
    E = np.load(Path(args.emb).expanduser()).astype(np.float32)
    rows = []
    for (arm, k), e in zip(zip(outs["arm"], outs["k"]), E):
        hk = f"{arm}_hidden"
        hidden = np.asarray(tg[hk][k], dtype=np.int64) if hk in tg.files else None
        ps, fs = max_sim(P, e, hidden), max_sim(F, e)
        rows.append({"arm": arm, "k": k, "iso_decoded": 1 - ps, "anticipation": fs - ps,
                     "growth_decoded": region_growth(P, F, e, ratio, hidden)})
    df = pd.DataFrame(rows)
    df.to_csv(run / f"metrics_{args.space}.csv", index=False)

    real_iso, base = [], []
    for k, i in enumerate(tg["REAL_idx"]):
        h = np.asarray(tg["REAL_hidden"][k], dtype=np.int64)
        ps = max_sim(P, P[i], h)
        real_iso.append(1 - ps)
        base.append(max_sim(F, P[i]) - ps)

    rng = np.random.default_rng(0)
    out = []
    for arm, g in df.groupby("arm", sort=False):
        rec = {"arm": arm, "n": len(g)}
        for c in ("iso_decoded", "anticipation", "growth_decoded"):
            m, lo, hi = boot_median(g[c].to_numpy(dtype=float), rng)
            rec[c] = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
        out.append(rec)
    print(f"space: {args.space}")
    print(pd.DataFrame(out).to_string(index=False))
    for name, x in (("isolation of the held-out real items", real_iso), ("base-rate anticipation", base)):
        m, lo, hi = boot_median(np.array(x), rng)
        print(f"{name}: {m:.3f} [{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
