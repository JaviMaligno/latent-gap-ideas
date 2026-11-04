"""Draw the stimuli and the MOMENTUM baseline for docs/prereg/inspiration_from_anything.md.

dist(s) = 1 - max cos(s, past Steam games), primary embedder (Qwen3), adult rows removed; a steam_game stimulus is
not compared with itself. BGE-M3 dist is stored too. Quintiles of dist within source; 40 stimuli per source x quintile
(seed 20261103). MOMENTUM: the K=256 cells of scripts/where_future_lands.py rebuilt identically (same seed, held-out
draw and k-means); 200 past Steam games released in 2023 from the top quartile of cells by momentum.
Inputs: prepared/inspiration/pool.parquet, kaggle-output/embed-inspiration-v1/emb_*.npy
Output: prepared/inspiration/stimuli.parquet, momentum.parquet, pool_dist.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_future_lands as W  # noqa: E402

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
INS = DATA / "prepared/inspiration"
EMB = DATA / "kaggle-output/embed-inspiration-v1"
SEED, PER_CELL, N_MOM = 20261103, 40, 200


def norm(x):
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def max_cos_excluding(Q, P, q_ids, p_ids, chunk=1000):
    out = np.empty(len(Q), np.float32)
    pid = pd.Index(p_ids)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ P.T
        for r, qid in enumerate(q_ids[i:i + chunk]):
            if qid is not None and qid in pid:
                S[r, pid.get_loc(qid)] = -1
        out[i:i + chunk] = S.max(1)
    return out


def main() -> None:
    pool = pd.read_parquet(INS / "pool.parquet")
    meta = pd.read_parquet(DATA / "prepared/steam/corpus.parquet", columns=["id", "name", "split", "adult", "date"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    past = (meta["split"] == "past").to_numpy()
    q_ids = np.where(pool["source"] == "steam_game", pool["id"], None)
    for e in ("qwen3e06", "bgem3"):
        P = norm(np.load(DATA / f"prepared/steam/emb_{e}.npy")[keep][past])
        Q = norm(np.load(EMB / f"emb_{e}.npy"))
        pool[f"dist_{e}"] = 1 - max_cos_excluding(Q, P, q_ids, meta.loc[past, "id"].to_numpy())
        print(e, pool.groupby("source")[f"dist_{e}"].median().round(3).to_dict(), flush=True)
    pool["rank"] = pool.groupby("source")["dist_qwen3e06"].rank(pct=True)
    pool["quintile"] = np.minimum((pool["rank"] * 5).apply(np.ceil).astype(int), 5)
    pool[["source", "id", "dist_qwen3e06", "dist_bgem3", "rank", "quintile"]].to_parquet(INS / "pool_dist.parquet")

    rng = np.random.default_rng(SEED)
    fut_names = set(meta.loc[~past, "name"].str.lower())
    cand = pool[~((pool["source"] == "steam_game") & pool["title"].str.lower().isin(fut_names))]
    st = pd.concat([g.iloc[rng.choice(len(g), PER_CELL, replace=False)] for _, g in cand.groupby(["source", "quintile"])])
    st = st.reset_index(names="pool_row")
    st.to_parquet(INS / "stimuli.parquet")
    print("stimuli", len(st), st.groupby("source").size().to_dict(), flush=True)

    # MOMENTUM: rebuild the cells of where_future_lands exactly
    emb = norm(np.load(DATA / "prepared/steam/emb_qwen3e06.npy")[keep])
    Pm, pm = emb[past], meta[past].reset_index(drop=True)
    r0 = np.random.default_rng(W.SEED)
    held = np.zeros(len(Pm), bool)
    held[r0.choice(len(Pm), len(Pm) // 10, replace=False)] = True
    R, rm = Pm[~held], pm[~held].reset_index(drop=True)
    cell = KMeans(n_clusters=W.K_CELLS, random_state=W.SEED, n_init=1).fit(R).labels_
    lo, hi = W.MOMENTUM_YEARS["steam"]
    ry = W.years(rm["date"], "steam")
    mom = np.bincount(cell, weights=((ry >= lo) & (ry <= hi)).astype(float), minlength=W.K_CELLS) / np.bincount(cell, minlength=W.K_CELLS)
    top = mom >= np.quantile(mom, 0.75)
    ok = np.flatnonzero(top[cell] & (ry == 2023))
    pick = ok[rng.choice(len(ok), N_MOM, replace=False)]
    m = rm.iloc[pick][["id", "name"]].assign(cell=cell[pick], momentum=mom[cell[pick]])
    m.to_parquet(INS / "momentum.parquet")
    print("momentum", len(m), "from", len(ok), "candidates; cells", int(top.sum()), flush=True)


if __name__ == "__main__":
    main()
