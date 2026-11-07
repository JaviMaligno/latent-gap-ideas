"""Powered test: is building on a past game worse than no material? (docs/prereg/strengthen_anchoring_none.md)

  uv run python scripts/strengthen_anchoring_none.py draw                       # 1,000 new steam_game stimuli
  ALIAS=gpt-4o-0806 uv run python scripts/strengthen_anchoring_none.py generate
  OMP_NUM_THREADS=6 uv run python scripts/strengthen_anchoring_none.py score-qwen
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow python scripts/strengthen_anchoring_none.py embed-bge
  uv run python scripts/strengthen_anchoring_none.py analyse
Raw data: runs/inspiration/gpt-4o-anchoring/. Output: docs/prereg/strengthen_anchoring_none.json
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
INS = DATA / "prepared/inspiration"
RUN = DATA / "runs/inspiration/gpt-4o-anchoring"
ORIG = DATA / "runs/inspiration/gpt-4o-0806"
OUT = ROOT / "docs/prereg/strengthen_anchoring_none.json"
PER_Q, N_NONE = 200, 1000
SEED_DRAW, SEED_ORDER, SEED_BOOT, N_BOOT = 20261107, 20261108, 20261107, 2000


def draw() -> None:
    pool = pd.read_parquet(INS / "pool.parquet")
    pdist = pd.read_parquet(INS / "pool_dist.parquet")
    st = pd.read_parquet(INS / "stimuli.parquet")
    meta = pd.read_parquet(DATA / "prepared/steam/corpus.parquet", columns=["name", "split", "adult"])
    meta = meta[~meta["adult"].to_numpy()]
    fut_names = set(meta.loc[meta["split"] != "past", "name"].str.lower())
    pool = pool.assign(rank=pdist["rank"].to_numpy(), quintile=pdist["quintile"].to_numpy())
    used = set(st["pool_row"])
    cand = pool[(pool["source"] == "steam_game") & ~pool.index.isin(used)
                & ~pool["title"].str.lower().isin(fut_names)]
    rng = np.random.default_rng(SEED_DRAW)
    new = pd.concat([g.iloc[rng.choice(len(g), PER_Q, replace=False)] for _, g in cand.groupby("quintile")])
    new = new.reset_index(names="pool_row")
    RUN.mkdir(parents=True, exist_ok=True)
    new.to_parquet(RUN / "stimuli.parquet")
    print("drawn", len(new), new.groupby("quintile").size().to_dict(), "candidates", len(cand), flush=True)


def generate() -> None:
    import inspiration_generate as G
    from latent_gap.gateway import Gateway
    gw = Gateway(ledger=DATA / "ledger.jsonl", model=os.environ["ALIAS"])
    st = pd.read_parquet(RUN / "stimuli.parquet")
    out = RUN / "ideas.jsonl"
    jobs = [(i, "INSTRUCTED", 0) for i in range(len(st))] + [(-1, "NONE", k) for k in range(N_NONE)]
    jobs = [jobs[j] for j in np.random.default_rng(SEED_ORDER).permutation(len(jobs))]
    done = {(r["stim"], r["arm"], r["k"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [j for j in jobs if j not in done]
    if os.environ.get("LG_LIMIT"):
        todo = todo[: int(os.environ["LG_LIMIT"])]
    lock = threading.Lock()

    def call(text: str) -> dict:
        for _ in range(3):
            raw = gw.chat([{"role": "system", "content": G.SYSTEM}, {"role": "user", "content": text}], max_tokens=400,
                          json_mode=True, tag="anchoring-none", extra={"temperature": 1.0})
            try:
                d = json.loads(raw)
                if d.get("title") and d.get("short_description"):
                    return {k: str(d.get(k, "")) for k in ("title", "short_description", "genres", "tags")}
            except json.JSONDecodeError:
                pass
        return {"title": "", "short_description": "", "genres": "", "tags": ""}

    def job(j) -> None:
        i, arm, k = j
        s = None if i < 0 else G.stimulus_block(st.loc[i, "title"], st.loc[i, "text"])
        rec = {"stim": i, "arm": arm, "k": k, **call(G.prompt(arm, s))}
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"{len(todo)} calls", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(job, todo), 1):
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


def ideas() -> pd.DataFrame:
    d = pd.DataFrame([json.loads(l) for l in (RUN / "ideas.jsonl").read_text().splitlines()])
    return d.sort_values(["arm", "stim", "k"]).reset_index(drop=True)


def idea_texts(d: pd.DataFrame) -> list[str]:
    from latent_gap.steam import item_text
    return [item_text(g, t, s) for g, t, s in zip(d["genres"], d["tags"], d["short_description"])]


def score_qwen() -> None:
    from latent_gap.embedder import LocalQwen
    d = ideas()
    d = d[d["title"] != ""].reset_index(drop=True)
    V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6"))).embed(idea_texts(d))
    np.save(RUN / "ideas_emb_qwen3e06.npy", V.astype(np.float32))
    print("qwen", V.shape, flush=True)


def embed_bge() -> None:
    """Settings of revision_neutral.embed_bge (max_seq_length 512, normalised, CPU float32)."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    d = ideas()
    d = d[d["title"] != ""].reset_index(drop=True)
    m = SentenceTransformer("BAAI/bge-m3", device="cpu")
    m.max_seq_length = 512
    T, out = idea_texts(d), []
    for b in range(0, len(T), 64):
        out.append(m.encode(T[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        print(f"bge {min(b + 64, len(T))}/{len(T)}", flush=True)
    np.save(RUN / "ideas_emb_bgem3.npy", np.concatenate(out).astype(np.float32))


def analyse() -> None:
    import inspiration_analysis as A
    from scipy import stats
    d0 = ideas()
    n_failed = int((d0["title"] == "").sum())
    d = d0[d0["title"] != ""].reset_index(drop=True)
    st = pd.read_parquet(RUN / "stimuli.parquet")
    meta = pd.read_parquet(DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    res = {"n_calls": int(len(d0)), "n_failed": n_failed,
           "n": {a: int((d["arm"] == a).sum()) for a in ("INSTRUCTED", "NONE")}}
    for e in ("qwen3e06", "bgem3"):
        vp = RUN / f"ideas_emb_{e}.npy"
        if not vp.exists():
            res[e] = {"skipped": f"{vp.name} missing"}
            continue
        rng = np.random.default_rng(A.SEED)
        E = A.norm(np.load(DATA / f"prepared/steam/emb_{e}.npy")[keep])
        P, Fc = E[past], E[(~past) & (appid > A.APPID_CUT)]
        V = A.norm(np.load(vp))
        assert len(V) == len(d)
        fut = A.maxcos(V, Fc)
        marg = np.mean([fut - A.maxcos(V, P[rng.choice(len(P), len(Fc), replace=False)]) for _ in range(A.B)], 0)
        x = d.assign(margin=marg, iso=1 - A.maxcos(V, P))
        has = (x["stim"] >= 0).to_numpy()
        S = A.norm(np.load(DATA / f"kaggle-output/embed-inspiration-v1/emb_{e}.npy"))[st["pool_row"].to_numpy()]
        x.loc[has, "uptake"] = [V[n] @ S[i] for n, i in zip(np.flatnonzero(has), x.loc[has, "stim"].to_numpy())]
        x.loc[has, "quintile"] = st.loc[x.loc[has, "stim"].to_numpy(), "quintile"].to_numpy()
        g, h = x.loc[has, "margin"].to_numpy(), x.loc[~has, "margin"].to_numpy()
        diff = float(g.mean() - h.mean())
        brng = np.random.default_rng(SEED_BOOT)
        bd = [g[brng.integers(0, len(g), len(g))].mean() - h[brng.integers(0, len(h), len(h))].mean() for _ in range(N_BOOT)]
        o = pd.read_parquet(ORIG / f"scored_{e}.parquet")
        og, oh = o[(o["arm"] == "INSTRUCTED") & (o["source"] == "steam_game")]["margin"].to_numpy(), o[o["arm"] == "NONE"]["margin"].to_numpy()
        pg, ph = np.concatenate([g, og]), np.concatenate([h, oh])
        res[e] = {
            "steam_instructed_mean": float(g.mean()), "none_mean": float(h.mean()), "diff": diff,
            "welch_t_one_sided_p": float(stats.ttest_ind(g, h, equal_var=False, alternative="less").pvalue),
            "mannwhitney_one_sided_p": float(stats.mannwhitneyu(g, h, alternative="less").pvalue),
            "boot_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "sd": {"steam": float(g.std(ddof=1)), "none": float(h.std(ddof=1))},
            "by_quintile": {int(q): float(v) - float(h.mean()) for q, v in x[has].groupby("quintile")["margin"].mean().items()},
            "uptake_mean": float(x.loc[has, "uptake"].mean()),
            "iso": {"steam": float(x.loc[has, "iso"].mean()), "none": float(x.loc[~has, "iso"].mean())},
            "pooled_with_original": {"n": [int(len(pg)), int(len(ph))], "diff": float(pg.mean() - ph.mean()),
                                     "welch_t_one_sided_p": float(stats.ttest_ind(pg, ph, equal_var=False, alternative="less").pvalue)},
        }
        x.to_parquet(RUN / f"scored_{e}.parquet")
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    {"draw": draw, "generate": generate, "score-qwen": score_qwen, "embed-bge": embed_bge, "analyse": analyse}[sys.argv[1]]()
