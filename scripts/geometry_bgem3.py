"""Replicate the target-free parts of geometry.py in the robustness space (bge-m3). Targets were built in the Qwen3
space, so only what needs no target is re-measured: corpus shape (G1), the decoded items' in-slab fraction and
latitude, anchoring to the shown neighbours (G5), the efficiency law on REAL (G6: its target is a real game, which
exists in both spaces) and the ceiling extrapolation with random directions of this space (G8).

    uv run python scripts/geometry_bgem3.py --domain steam
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from geometry import DATA, RUNS, corpus_shape, extrapolation, unit

OUTPUTS = {"pilot": ("steam-pilot-outputs", "embed-outputs-v1"), "grid": ("steam-grid-outputs", "embed-grid-v1")}


def outputs() -> pd.DataFrame:
    frames = []
    for key, (inp, out) in OUTPUTS.items():
        df = pd.read_parquet(DATA / "kaggle-input" / inp / "texts.parquet")
        df["e"] = list(unit(np.load(DATA / "kaggle-output" / out / "emb_bgem3.npy").astype(np.float32)))
        if "run" not in df:
            df["run"] = "pilot"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["gen"] = df.run.map(RUNS)
    return df


def neighbours() -> dict:
    nn = {}
    for run in RUNS:
        for line in (DATA / "runs" / "steam" / run / "items.jsonl").read_text().splitlines():
            r = json.loads(line)
            nn[(run, r["arm"], r["k"])] = r.get("neighbours") or []
    return nn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
    emb = unit(np.load(root / "emb_bgem3.npy").astype(np.float32))  # renormalised: stored norms are not exactly 1
    keep = ~meta["adult"].to_numpy()
    P = emb[keep][(meta["split"][keep] == "past").to_numpy()]
    mu_hat, Uk, stats = corpus_shape(P, rng)
    print("G1 corpus shape (bge-m3):", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}))

    tg = np.load(DATA / "runs" / args.domain / "pilot" / "targets.npz", allow_pickle=True)
    df = outputs()
    E = np.stack(df.e.to_list())
    df["m_e"] = np.linalg.norm(E @ Uk, axis=1)
    df["e_mu"] = E @ mu_hat
    nn = neighbours()
    anchor = []
    for i, r in df.iterrows():
        idx = nn.get((r.run, r.arm, r.k), [])[:8]
        anchor.append(float(E[i] @ unit(P[idx].mean(0))) if idx else np.nan)
    df["anchor_decoded"] = anchor
    print(f"\ncorpus items: in-slab {np.median(np.linalg.norm(P[rng.integers(0, len(P), 5000)] @ Uk, axis=1)):.3f}, "
          f"latitude {np.median(P @ mu_hat):.3f} | random direction in-slab {np.sqrt(Uk.shape[1] / P.shape[1]):.3f}")
    print("decoded items, median in-slab fraction / latitude / cos to shown neighbours' centroid:")
    print(df.groupby(["arm", "gen"])[["m_e", "e_mu", "anchor_decoded"]].median().unstack().round(3).to_string())

    # The 8 neighbours shown for a FAR target, against each other: how typical a game they are as a group.
    far_nn = [v[:8] for (run, arm, k), v in nn.items() if arm == "FAR" and run == "pilot"]
    loo = [np.median([P[i[j]] @ unit(np.delete(P[i], j, 0).mean(0)) for j in range(len(i))]) for i in far_nn]
    print("FAR shown neighbours: median cos of each to the centroid of the other seven (leave-one-out)",
          round(float(np.median(loo)), 3))
    print("m(e) per item, all arms and generators, min/p05/median/p95/max:",
          " ".join(f"{v:.3f}" for v in np.quantile(df.m_e, [0, 0.05, 0.5, 0.95, 1])))

    real = df[df.arm == "REAL"].copy()
    x = P[tg["REAL_idx"][real.k.to_numpy()]]
    real["landing"] = np.einsum("nd,nd->n", np.stack(real.e.to_list()), x)
    S = P @ x.T
    for j, k in enumerate(real.k):
        S[np.asarray(tg["REAL_hidden"][k], dtype=np.int64), j] = -np.inf
    real["ceiling"] = S.max(0)
    print("\nG6 on REAL in bge-m3: landing / ceiling / kappa per generator")
    print(real.groupby("gen").apply(lambda g: pd.Series({"landing": g.landing.median(), "ceiling": g.ceiling.median(),
                                                         "kappa": (g.landing / g.ceiling).median()}),
                                    include_groups=False).round(3).to_string())

    extrapolation(P, rng)


if __name__ == "__main__":
    main()
