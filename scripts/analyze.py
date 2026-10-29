"""E1 landing, E2 regression to the manifold, E4 time (docs/design.md). Judge-based E3 lives in judge.py.

    uv run python scripts/analyze.py --domain steam --run pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.steam import EXPLICIT
from latent_gap.targets import sources

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
K_REGION = 50
DECODED_ARMS = ("REAL", "INTERP", "FAR", "NOISE", "CROWD", "GAP")


def boot_median(x: np.ndarray, rng: np.random.Generator, b: int = 2000) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    m = np.median(rng.choice(x, (b, len(x))), axis=1)
    return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def max_sim(M: np.ndarray, v: np.ndarray, hidden: np.ndarray | None = None) -> float:
    s = M @ v
    if hidden is not None and len(hidden):
        s[hidden] = -np.inf
    return float(s.max())


def region_growth(P: np.ndarray, F: np.ndarray, v: np.ndarray, global_ratio: float, hidden=None) -> float:
    s = P @ v
    if hidden is not None and len(hidden):
        s[hidden] = -np.inf
    radius = np.partition(s, -K_REGION)[-K_REGION]
    return float((F @ v >= radius).sum() / K_REGION / global_ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--run", default="pilot")
    args = ap.parse_args()

    root = DATA / "prepared" / args.domain
    run = DATA / "runs" / args.domain / args.run
    meta = pd.read_parquet(root / "corpus.parquet", columns=["name", "split", "text", "adult"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    pm = meta[past].reset_index(drop=True)
    explicit = (pm["name"] + " " + pm["text"]).str.contains(EXPLICIT).to_numpy()
    global_ratio = len(F) / len(P)
    tg = np.load(run / "targets.npz", allow_pickle=True)

    rows = []
    for line in (run / "items.jsonl").read_text().splitlines():
        r = json.loads(line)
        best = r.get("best")
        if not best:
            rows.append({"arm": r["arm"], "k": r["k"], "failed": True})
            continue
        e = np.array(best["emb"], dtype=np.float32)
        hk = f"{r['arm']}_hidden"
        hidden = np.asarray(tg[hk][r["k"]], dtype=np.int64) if hk in tg.files else None
        row = {"arm": r["arm"], "k": r["k"], "failed": False,
               "iters": len(r.get("history", [])) or 1,
               "past_sim": max_sim(P, e, hidden), "fut_sim": max_sim(F, e),
               "explicit_src": bool(explicit[sources(tg, r["arm"], r["k"])].any()),
               "explicit_nn": bool(explicit[r.get("neighbours", [])].any())}
        row["iso_decoded"] = 1 - row["past_sim"]
        row["anticipation"] = row["fut_sim"] - row["past_sim"]
        row["growth_decoded"] = region_growth(P, F, e, global_ratio, hidden)
        if r["arm"] in DECODED_ARMS:
            t = tg[r["arm"]][r["k"]]
            row["landing"] = float(e @ t)
            row["first_landing"] = next((h["sim"] for h in r["history"] if h["sim"] is not None), np.nan)
            row["iso_target"] = 1 - max_sim(P, t, hidden)
            row["growth_target"] = region_growth(P, F, t, global_ratio, hidden)
            if f"{r['arm']}_level" in tg.files:
                row["level"] = float(tg[f"{r['arm']}_level"][r["k"]])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(run / "metrics.csv", index=False)

    # Base rate for anticipation: the held-out REAL items themselves (a real past item vs. the rest of the past).
    base = []
    for k, i in enumerate(tg["REAL_idx"]):
        v = P[i]
        base.append(max_sim(F, v) - max_sim(P, v, np.asarray(tg["REAL_hidden"][k], dtype=np.int64)))
    base = np.array(base)

    rng = np.random.default_rng(0)
    cols = ["landing", "first_landing", "iso_target", "iso_decoded", "anticipation", "growth_target", "growth_decoded"]
    out = []
    ok = ~df["failed"]
    for arm, g in df[ok].groupby("arm", sort=False):
        rec = {"arm": arm, "n": len(g)}
        for c in cols:
            if c in g:
                m, lo, hi = boot_median(g[c].to_numpy(dtype=float), rng)
                rec[c] = f"{m:.3f} [{lo:.3f},{hi:.3f}]" if not np.isnan(m) else ""
        out.append(rec)
    summary = pd.DataFrame(out)
    m, lo, hi = boot_median(base, rng)
    print(summary.to_string(index=False))
    print(f"\nbase-rate anticipation (held-out real past items): {m:.3f} [{lo:.3f},{hi:.3f}]")
    flagged = df["explicit_src"].fillna(False).astype(bool) | df["explicit_nn"].fillna(False).astype(bool)
    print("failed:", df["failed"].sum(), "| flagged explicit (source or neighbours, kept above):", int(flagged.sum()))
    clean = ok & ~flagged
    print("\nsensitivity, excluding flagged explicit items (median landing / iso_decoded):")
    print(df[clean].groupby("arm", sort=False)[["landing", "iso_decoded"]].median().round(3).to_string())
    summary.to_csv(run / "summary.csv", index=False)


if __name__ == "__main__":
    main()
