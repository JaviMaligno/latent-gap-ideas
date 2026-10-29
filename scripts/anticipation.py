"""Anticipation: is a generated idea closer to something published AFTER the split than to anything before it?

For each idea e: past = max cos to a random past subsample of the same size as the future corpus (so neither side
wins by size), fut = max cos to the future corpus. hit = fut > past. Averaged over B subsamples.
References from real items: held-out PAST items (timeless baseline: what "no anticipation" looks like) and held-out
FUTURE items (ceiling). Caveat: the generators were trained after the split, so a hit can be memory, not foresight;
compare arms with each other, or use a generator whose training cutoff precedes the future split.

    uv run python scripts/anticipation.py --domain arxiv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
SEED = 20261025
B = 5
N_REF = 500


def nearest(E: np.ndarray, C: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = np.full(len(E), -1.0, np.float32)
    for i in range(0, len(C), chunk):
        out = np.maximum(out, (E @ C[i:i + chunk].T).max(1))
    return out


def score(E: np.ndarray, P: np.ndarray, F: np.ndarray, rng) -> dict:
    fut = nearest(E, F)
    hits, margins = [], []
    for _ in range(B):
        past = nearest(E, P[rng.choice(len(P), len(F), replace=False)])
        hits.append((fut > past).mean())
        margins.append(np.median(fut - past))
    return {"n": len(E), "hit": float(np.mean(hits)), "margin": float(np.mean(margins)), "fut_cos": float(np.median(fut))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="arxiv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--future-from", default=None,
                    help="YYYY-MM-DD: only future items from this date count (set it after the generators' training cutoff); "
                         "future items before it are dropped from both sides")
    args = ap.parse_args()
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    past_i = np.flatnonzero((meta["split"] == "past").to_numpy())
    fut = (meta["split"] == "future").to_numpy().copy()
    if args.future_from:
        fut &= (pd.to_datetime(meta["date"], errors="coerce") >= args.future_from).to_numpy()
    fut_i = np.flatnonzero(fut)
    rng = np.random.default_rng(SEED)

    rows = []
    # references: real held-out items scored against the rest
    ref_p, ref_f = rng.choice(past_i, N_REF, replace=False), rng.choice(fut_i, N_REF, replace=False)
    P_rest, F_rest = emb[np.setdiff1d(past_i, ref_p)], emb[np.setdiff1d(fut_i, ref_f)]
    rows.append({"gen": "real", "arm": "REF_PAST", **score(emb[ref_p], P_rest, F_rest, rng)})
    rows.append({"gen": "real", "arm": "REF_FUTURE", **score(emb[ref_f], P_rest, F_rest, rng)})

    P, F = emb[past_i], emb[fut_i]
    for run in sorted((DATA / "runs" / args.domain).glob("grid-*/items.jsonl")):
        items = [json.loads(l) for l in run.read_text().splitlines()]
        for arm in sorted({x["arm"] for x in items}):
            E = np.array([x["best"]["emb"] for x in items if x["arm"] == arm and x.get("best")], np.float32)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            rows.append({"gen": run.parent.name.removeprefix("grid-"), "arm": arm, **score(E, P, F, rng)})

    df = pd.DataFrame(rows)
    print(f"{args.domain}: past {len(past_i)}, future {len(fut_i)} (past subsampled to {len(fut_i)})")
    print(df.pivot_table(index="arm", columns="gen", values="hit").round(2).to_string())
    print("\nmedian margin fut-past\n" + df.pivot_table(index="arm", columns="gen", values="margin").round(3).to_string())
    tag = f"_from{args.future_from}" if args.future_from else ""
    out = Path(args.out) if args.out else DATA / "runs" / args.domain / f"anticipation{tag}.csv"
    df.to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
