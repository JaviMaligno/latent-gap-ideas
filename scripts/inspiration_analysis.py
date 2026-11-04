"""Confirmatory analysis of docs/prereg/inspiration_from_anything.md (P1-P5) plus the exploratory checks.

Ideas are embedded locally (Qwen3 ONNX; agrees with the Kaggle encoder to cos >= 0.99999, encoder_agreement.json)
with the Steam item template (genres, tags, short description). LG_EMB=bgem3 reuses Kaggle BGE-M3 embeddings of the
ideas (ideas_emb_bgem3.npy, produced by the embedding kernel) for the replication.
    uv run python scripts/inspiration_analysis.py
Output: docs/prereg/inspiration_from_anything_result.json (key = embedder)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latent_gap.steam import item_text  # noqa: E402

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
INS = DATA / "prepared/inspiration"
RUN = DATA / "runs/inspiration/gpt-4o-0806"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/inspiration_from_anything_result.json"
SEED, B, N_BOOT, APPID_CUT = 20261103, 5, 2000, 2_680_000
SOURCES = ["steam_game", "board_game", "movie", "sports_rules", "paper", "patent", "news", "recipe", "wikipedia"]


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def maxcos(Q, P, chunk=1000, exclude=None):
    """max cos of each row of Q to P; exclude[i] = row of P to ignore for Q[i] (or -1)."""
    out = np.empty(len(Q), np.float32)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ P.T
        if exclude is not None:
            for r, j in enumerate(exclude[i:i + chunk]):
                if j >= 0:
                    S[r, j] = -1
        out[i:i + chunk] = S.max(1)
    return out


def ols(y, X):
    """OLS with classical SEs; returns coefficients, SEs, residual dof."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    s2 = r @ r / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), dof


def dummies(cat, levels):
    return np.column_stack([(cat == l).astype(float) for l in levels])


def slope_within(y, x, src):
    X = np.column_stack([x, dummies(src, SOURCES)])
    b, se, dof = ols(y, X)
    t = b[0] / se[0]
    return {"slope": float(b[0]), "se": float(se[0]), "p_pos": float(stats.t.sf(t, dof)),
            "p_neg": float(stats.t.cdf(t, dof))}


def two_lines(y, x, src):
    """Simonsohn (2018) two-lines test with source fixed effects. Step 1: breakpoint at the maximum of a within-source
    quadratic fit, clipped to the 10th-90th percentile of x. Step 2 (Robin Hood): the breakpoint is moved so that the
    segment whose slope has the smaller |z| gets the share |z_other| / (|z1| + |z2|) of the observations. Hump = slope
    below > 0 and slope above < 0, each one-sided p < 0.05."""
    D = dummies(src, SOURCES)

    def fit(xc):
        lo, hi = np.where(x < xc, x - xc, 0.0), np.where(x >= xc, x - xc, 0.0)
        X = np.column_stack([lo, hi, (x >= xc).astype(float), D])
        b, se, dof = ols(y, X)
        return b[0] / se[0], b[1] / se[1], b[0], b[1], dof

    bq, _, _ = ols(y, np.column_stack([x, x ** 2, D]))
    xc = -bq[0] / (2 * bq[1]) if bq[1] < 0 else np.median(x)
    xc = float(np.clip(xc, *np.percentile(x, [10, 90])))
    z1, z2, *_ = fit(xc)
    share_low = (x < xc).mean()
    if abs(z1) < abs(z2):  # low segment weaker: give it more observations
        share_low = abs(z2) / (abs(z1) + abs(z2))
    else:
        share_low = 1 - abs(z1) / (abs(z1) + abs(z2))
    xc2 = float(np.clip(np.quantile(x, share_low), *np.percentile(x, [10, 90])))
    z1, z2, b1, b2, dof = fit(xc2)
    p1, p2 = float(stats.t.sf(z1, dof)), float(stats.t.cdf(z2, dof))
    return {"breakpoint": xc2, "slope_low": float(b1), "slope_high": float(b2), "p_low_pos": p1, "p_high_neg": p2,
            "hump": bool(p1 < 0.05 and p2 < 0.05)}


def eta2(y, src, q):
    """Partial eta^2 of source and of quintile in y ~ C(source) + C(quintile)."""
    full = np.column_stack([dummies(src, SOURCES), dummies(q, [2, 3, 4, 5])])
    no_s = np.column_stack([np.ones(len(y)), dummies(q, [2, 3, 4, 5])])
    no_q = dummies(src, SOURCES)

    def sse(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ b) ** 2).sum())
    e = sse(full)
    return (sse(no_s) - e) / (sse(no_s)), (sse(no_q) - e) / (sse(no_q))


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, ps[k] * (len(keys) - i)))
        out[k] = run
    return out


def main() -> None:
    emb_name = os.environ.get("LG_EMB", "qwen3e06")
    rng = np.random.default_rng(SEED)
    ideas = pd.DataFrame([json.loads(l) for l in (RUN / "ideas.jsonl").read_text().splitlines()])
    n_raw = len(ideas)
    ideas = ideas[ideas["title"] != ""].reset_index(drop=True)
    st = pd.read_parquet(INS / "stimuli.parquet")
    mom = pd.read_parquet(INS / "momentum.parquet")
    meta = pd.read_parquet(DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult", "text"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    E = norm(np.load(DATA / f"prepared/steam/emb_{emb_name}.npy")[keep])
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    clean = (~past) & (appid > APPID_CUT)
    P, Fc, Fd = E[past], E[clean], E[(~past) & ~(appid > APPID_CUT)]
    pid = meta.loc[past, "id"].to_numpy()

    texts = [item_text(g, t, d) for g, t, d in zip(ideas["genres"], ideas["tags"], ideas["short_description"])]
    cache = RUN / f"ideas_emb_{emb_name}.npy"
    if cache.exists() and len(np.load(cache)) == len(ideas):
        V = norm(np.load(cache))
    elif emb_name == "qwen3e06":
        from latent_gap.embedder import LocalQwen
        V = norm(LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6"))).embed(texts))
        np.save(cache, V)
    else:
        raise SystemExit(f"missing {cache}: embed the ideas with {emb_name} on Kaggle first")
    pool_emb = norm(np.load(DATA / f"kaggle-output/embed-inspiration-v1/emb_{emb_name}.npy"))
    S = pool_emb[st["pool_row"].to_numpy()]

    # MOMENTUM games: their own embedding, compared with the past set minus themselves
    m_rows = pd.Index(pid).get_indexer(mom["id"])
    assert (m_rows >= 0).all()
    M = P[m_rows]

    def anticip(Q, excl=None, Fut=Fc):
        fut = maxcos(Q, Fut)
        hits, marg = [], []
        for _ in range(B):
            sub = rng.choice(len(P), len(Fut), replace=False)
            ex = None
            if excl is not None:
                pos = {j: n for n, j in enumerate(sub)}
                ex = np.array([pos.get(j, -1) for j in excl])
            pc = maxcos(Q, P[sub], exclude=ex)
            hits.append(fut > pc)
            marg.append(fut - pc)
        return np.mean(hits, 0), np.mean(marg, 0)

    hit, margin = anticip(V)
    iso = 1 - maxcos(V, P)
    m_hit, m_margin = anticip(M, excl=m_rows)
    m_iso = 1 - maxcos(M, P, exclude=m_rows)
    d = ideas.assign(hit=hit, margin=margin, iso=iso)
    stim = d["stim"].to_numpy()
    has = stim >= 0
    d.loc[has, "source"] = st.loc[stim[has], "source"].to_numpy()
    d.loc[has, "quintile"] = st.loc[stim[has], "quintile"].to_numpy()
    d.loc[has, "rank"] = st.loc[stim[has], "rank"].to_numpy()
    d.loc[has, "dist"] = st.loc[stim[has], f"dist_{emb_name}"].to_numpy()
    # uptake: own stimulus vs 5 others of the same source and quintile (fixed draw per idea)
    up = np.full(len(d), np.nan)
    cell_rows = st.groupby(["source", "quintile"]).indices
    for n in np.flatnonzero(has):
        i = stim[n]
        others = [j for j in cell_rows[(st.loc[i, "source"], st.loc[i, "quintile"])] if j != i]
        o = rng.choice(others, 5, replace=False)
        up[n] = V[n] @ S[i] - (S[o] @ V[n]).mean()
    d["uptake"] = up

    res = {"n_ideas": int(len(d)), "n_calls": int(n_raw), "n_failed": int(n_raw - len(ideas)),
           "n_future_clean": int(clean.sum()), "n_future_dirty": int(len(Fd))}
    framings = ("INCIDENTAL", "INSTRUCTED")
    for fr in framings:
        g = d[d["arm"] == fr]
        y, x, src = g["margin"].to_numpy(), g["rank"].to_numpy(float), g["source"].to_numpy()
        # rank re-ranked within source among the drawn stimuli, in the chosen embedder
        xr = g.groupby("source")["dist"].rank(pct=True).to_numpy()
        r = {"P1_margin_slope": slope_within(y, xr, src), "P1_two_lines": two_lines(y, xr, src),
             "P3_iso_slope": slope_within(g["iso"].to_numpy(), xr, src),
             "hit_slope": slope_within(g["hit"].to_numpy(), xr, src)}
        es, eq = eta2(y, src, g["quintile"].to_numpy(int))
        bs = []
        idx = np.arange(len(g))
        for _ in range(N_BOOT):
            b = rng.choice(idx, len(idx))
            a1, a2 = eta2(y[b], src[b], g["quintile"].to_numpy(int)[b])
            bs.append(a1 - a2)
        r["P2_eta2"] = {"source": es, "quintile": eq, "diff": es - eq,
                        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "p_boot": float((np.array(bs) <= 0).mean())}
        r["by_source"] = {s: {"margin": float(h["margin"].mean()), "hit": float(h["hit"].mean()),
                              "iso": float(h["iso"].mean()), "uptake": float(h["uptake"].mean()),
                              "by_quintile_margin": h.groupby("quintile")["margin"].mean().round(4).to_dict()}
                          for s, h in g.groupby("source")}
        res[fr] = r
    a = d[d["arm"] == "INSTRUCTED"].set_index("stim")["uptake"]
    b_ = d[d["arm"] == "INCIDENTAL"].set_index("stim")["uptake"]
    both = a.index.intersection(b_.index)
    w = stats.wilcoxon(a[both], b_[both], alternative="greater")
    res["P4_uptake"] = {"instructed": float(a.mean()), "incidental": float(b_.mean()),
                        "diff": float((a[both] - b_[both]).mean()), "p": float(w.pvalue), "n": int(len(both))}
    none = d[d["arm"] == "NONE"]
    res["baselines"] = {"NONE": {"margin": float(none["margin"].mean()), "hit": float(none["hit"].mean()),
                                 "iso": float(none["iso"].mean()), "n": int(len(none))},
                        "MOMENTUM": {"margin": float(m_margin.mean()), "hit": float(m_hit.mean()),
                                     "iso": float(m_iso.mean()), "n": int(len(M))}}
    # P5: per stimulus mean of the two framings; each cell vs each baseline, one-sided, Holm over 45 cells per baseline
    ps_ = d[has].groupby("stim").agg(margin=("margin", "mean"), source=("source", "first"), quintile=("quintile", "first"))
    p_none, p_mom = {}, {}
    for (s, q), h in ps_.groupby(["source", "quintile"]):
        p_none[f"{s}/Q{int(q)}"] = float(stats.mannwhitneyu(h["margin"], none["margin"], alternative="greater").pvalue)
        p_mom[f"{s}/Q{int(q)}"] = float(stats.mannwhitneyu(h["margin"], m_margin, alternative="greater").pvalue)
    hn, hm = holm(p_none), holm(p_mom)
    beats = [k for k in p_none if hn[k] < 0.05 and hm[k] < 0.05]
    res["P5"] = {"cells_beating_both": beats, "holds": not beats,
                 "cells_beating_none": [k for k in hn if hn[k] < 0.05], "cells_beating_momentum": [k for k in hm if hm[k] < 0.05]}
    # Holm over {P1, P2, P3, P4} per framing. P1 predicts that the positive-slope test FAILS, so its adjusted p is
    # reported and P1 holds when it stays >= 0.05. P2 uses the one-sided bootstrap p of the eta^2 difference.
    for fr in framings:
        res[f"holm_{fr}"] = holm({"P1": res[fr]["P1_margin_slope"]["p_pos"], "P2": res[fr]["P2_eta2"]["p_boot"],
                                  "P3": res[fr]["P3_iso_slope"]["p_pos"], "P4": res["P4_uptake"]["p"]})
    # exploratory: contamination check against future games visible before the cutoff
    _, mdirty = anticip(V, Fut=Fd)
    res["exploratory_dirty_future_margin"] = {fr: float(mdirty[(d["arm"] == fr).to_numpy()].mean()) for fr in framings + ("NONE",)}
    d.to_parquet(RUN / f"scored_{emb_name}.parquet")
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres[emb_name] = res
    OUT.write_text(json.dumps(allres, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
