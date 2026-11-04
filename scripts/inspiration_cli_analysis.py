"""Exploratory analysis of the Claude Code / Codex stress test (docs/prereg/inspiration_from_anything.md).

Same measures as scripts/inspiration_analysis.py (primary embedder, local Qwen3), read only within each generator:
each source and the within-source distance slope against the generator's own NONE arm.
    uv run python scripts/inspiration_cli_analysis.py codex claude
Output: key "cli_<gen>" in docs/prereg/inspiration_from_anything_result.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402
from latent_gap.steam import item_text  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    st = pd.read_parquet(A.INS / "stimuli.parquet")
    meta = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    E = A.norm(np.load(A.DATA / "prepared/steam/emb_qwen3e06.npy")[keep])
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    P, Fc = E[past], E[(~past) & (appid > A.APPID_CUT)]
    S = A.norm(np.load(A.DATA / "kaggle-output/embed-inspiration-v1/emb_qwen3e06.npy"))[st["pool_row"].to_numpy()]
    from latent_gap.embedder import LocalQwen
    enc = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6")))
    allres = json.loads(A.OUT.read_text())
    for gen in sys.argv[1:]:
        run = A.DATA / "runs/inspiration" / f"{gen}-cli"
        d = pd.DataFrame([json.loads(l) for l in (run / "ideas.jsonl").read_text().splitlines()])
        d = d[d["title"] != ""].reset_index(drop=True)
        cache = run / "ideas_emb_qwen3e06.npy"
        if cache.exists() and len(np.load(cache)) == len(d):
            V = A.norm(np.load(cache))
        else:
            V = A.norm(enc.embed([item_text(g, t, s) for g, t, s in zip(d["genres"], d["tags"], d["short_description"])]))
            np.save(cache, V)
        fut = A.maxcos(V, Fc)
        marg = np.mean([fut - A.maxcos(V, P[rng.choice(len(P), len(Fc), replace=False)]) for _ in range(A.B)], 0)
        d["margin"], d["iso"] = marg, 1 - A.maxcos(V, P)
        has = (d["stim"] >= 0).to_numpy()
        si = d.loc[has, "stim"].to_numpy()
        d.loc[has, "source"] = st.loc[si, "source"].to_numpy()
        d.loc[has, "rank"] = st.loc[si, "rank"].to_numpy()
        d.loc[has, "uptake"] = [V[n] @ S[i] for n, i in zip(np.flatnonzero(has), si)]
        g, none = d[has], d[~has]
        r = A.slope_within(g["margin"].to_numpy(), g["rank"].to_numpy(float), g["source"].to_numpy())
        res = {"n": int(len(d)), "n_none": int(len(none)), "none_margin": float(none["margin"].mean()),
               "none_iso": float(none["iso"].mean()), "margin_slope": r,
               "iso_slope": A.slope_within(g["iso"].to_numpy(), g["rank"].to_numpy(float), g["source"].to_numpy()),
               "by_source": {s: {"margin": float(h["margin"].mean()), "iso": float(h["iso"].mean()),
                                 "p_vs_none": float(stats.mannwhitneyu(h["margin"], none["margin"]).pvalue)}
                             for s, h in g.groupby("source")},
               "distinct_titles_none": int(none["title"].nunique()),
               "time_share_none": float(none["short_description"].str.contains("time", case=False).mean())}
        allres[f"cli_{gen}"] = res
        d.to_parquet(run / "scored_qwen3e06.parquet")
        print(gen, json.dumps(res, indent=1, default=float), flush=True)
    A.OUT.write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
