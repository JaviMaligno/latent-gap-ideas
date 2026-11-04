"""Revision reanalysis of Section 6 (inspiration from anything), answering review points 4, 7, 9, 18 and 19.

No generation and no embedding model: reuses the saved idea embeddings (runs/inspiration/*/ideas_emb_*.npy), the
saved pool/stimuli/momentum tables and the Steam corpus embeddings. The scoring of scripts/inspiration_analysis.py is
replayed with the same seed and call order, and checked against the saved scored_*.parquet before anything is used.
    OMP_NUM_THREADS=2 uv run python scripts/revision_inspiration.py
Output: docs/prereg/revision_inspiration.json
  (a) "date_filter": P1-P5 and the steam_game headline without the steam_game stimuli released on/after the gpt-4o
      cutoff (2023-10-01), also without the late-2023 MOMENTUM games; "appid_threshold": appid vs release date.
  (b) "estimands": which distance variable each prediction uses; P2/P5 with BGE-M3 quintiles; P1 on quintiles.
  (c) "cli_clustered": stress-test comparisons and slopes with the 25-idea call as the cluster, Holm over sources.
  (d) "ci": 95 % CIs of the P1 slope, the two-lines slopes, the quadratic term, and the isolation-adjusted slope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402

CUTOFF = pd.Timestamp("2023-10-01")
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/revision_inspiration.json"
FR = ("INCIDENTAL", "INSTRUCTED")
T975 = stats.t.ppf(0.975, 10_000)


def ci(b, se, dof):
    t = stats.t.ppf(0.975, dof)
    return [float(b - t * se), float(b + t * se)]


def slope_ci(y, x, src, extra=None):
    X = np.column_stack([x] + ([extra] if extra is not None else []) + [A.dummies(src, A.SOURCES)])
    b, se, dof = A.ols(y, X)
    t = b[0] / se[0]
    out = {"slope": float(b[0]), "se": float(se[0]), "ci95": ci(b[0], se[0], dof),
           "p_pos": float(stats.t.sf(t, dof)), "p_two": float(2 * stats.t.sf(abs(t), dof))}
    if extra is not None:
        out["extra_coef"], out["extra_se"] = float(b[1]), float(se[1])
    return out


def two_lines_ci(y, x, src):
    """Same algorithm as A.two_lines, also returning SEs and 95 % CIs of both segment slopes."""
    D = A.dummies(src, A.SOURCES)

    def fit(xc):
        lo, hi = np.where(x < xc, x - xc, 0.0), np.where(x >= xc, x - xc, 0.0)
        b, se, dof = A.ols(y, np.column_stack([lo, hi, (x >= xc).astype(float), D]))
        return b, se, dof

    bq, _, _ = A.ols(y, np.column_stack([x, x ** 2, D]))
    xc = -bq[0] / (2 * bq[1]) if bq[1] < 0 else np.median(x)
    xc = float(np.clip(xc, *np.percentile(x, [10, 90])))
    b, se, _ = fit(xc)
    z1, z2 = b[0] / se[0], b[1] / se[1]
    share_low = abs(z2) / (abs(z1) + abs(z2)) if abs(z1) < abs(z2) else 1 - abs(z1) / (abs(z1) + abs(z2))
    xc2 = float(np.clip(np.quantile(x, share_low), *np.percentile(x, [10, 90])))
    b, se, dof = fit(xc2)
    lo_ci, hi_ci = ci(b[0], se[0], dof), ci(b[1], se[1], dof)
    return {"breakpoint": xc2,
            "slope_low": float(b[0]), "se_low": float(se[0]), "ci95_low": lo_ci,
            "p_low_pos": float(stats.t.sf(b[0] / se[0], dof)),
            "slope_high": float(b[1]), "se_high": float(se[1]), "ci95_high": hi_ci,
            "p_high_neg": float(stats.t.cdf(b[1] / se[1], dof)),
            # largest fall over the segment above the breakpoint compatible with the CI (margin units)
            "max_fall_above_excluded_beyond": float(-hi_ci[0] * (x.max() - xc2)),
            "max_rise_below_compatible": float(lo_ci[1] * (xc2 - x.min()))}


def quad_ci(y, x, src):
    """Quadratic term c of y ~ x + x^2 + source FE. A symmetric inverted U peaking at 0.5 with height h above the
    ends has c = -4h, so the CI of c bounds h: h_max = -ci_low / 4."""
    b, se, dof = A.ols(y, np.column_stack([x, x ** 2, A.dummies(src, A.SOURCES)]))
    c = ci(b[1], se[1], dof)
    return {"coef_x2": float(b[1]), "se": float(se[1]), "ci95": c, "hump_height_excluded_above": float(-c[0] / 4)}


def replay(emb: str):
    """Replay inspiration_analysis.main up to the uptake loop with the same rng; return the idea table (checked
    against the saved scored parquet), the MOMENTUM margins/ids and the rng in the state the P2 bootstrap used."""
    rng = np.random.default_rng(A.SEED)
    ideas = pd.DataFrame([json.loads(l) for l in (A.RUN / "ideas.jsonl").read_text().splitlines()])
    ideas = ideas[ideas["title"] != ""].reset_index(drop=True)
    st = pd.read_parquet(A.INS / "stimuli.parquet")
    mom = pd.read_parquet(A.INS / "momentum.parquet")
    meta = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    E = A.norm(np.load(A.DATA / f"prepared/steam/emb_{emb}.npy")[keep])
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    P, Fc = E[past], E[(~past) & (appid > A.APPID_CUT)]
    del E
    pid = meta.loc[past, "id"].to_numpy()
    V = A.norm(np.load(A.RUN / f"ideas_emb_{emb}.npy"))
    assert len(V) == len(ideas)
    S = A.norm(np.load(A.DATA / f"kaggle-output/embed-inspiration-v1/emb_{emb}.npy"))[st["pool_row"].to_numpy()]
    m_rows = pd.Index(pid).get_indexer(mom["id"])
    M = P[m_rows]

    def anticip(Q, excl=None):
        fut = A.maxcos(Q, Fc)
        marg = []
        for _ in range(A.B):
            sub = rng.choice(len(P), len(Fc), replace=False)
            ex = None
            if excl is not None:
                pos = {j: n for n, j in enumerate(sub)}
                ex = np.array([pos.get(j, -1) for j in excl])
            marg.append(fut - A.maxcos(Q, P[sub], exclude=ex))
        return np.mean(marg, 0)

    margin = anticip(V)
    m_margin = anticip(M, excl=m_rows)
    saved = pd.read_parquet(A.RUN / f"scored_{emb}.parquet")
    assert np.allclose(margin, saved["margin"].to_numpy(), atol=1e-6), "replayed margins differ from saved"
    # consume the uptake draws exactly as the original loop did
    stim = saved["stim"].to_numpy()
    cell_rows = st.groupby(["source", "quintile"]).indices
    for n in np.flatnonzero(stim >= 0):
        i = stim[n]
        others = [j for j in cell_rows[(st.loc[i, "source"], st.loc[i, "quintile"])] if j != i]
        rng.choice(others, 5, replace=False)
    iso_V = 1 - A.maxcos(V, P)
    assert np.allclose(iso_V, saved["iso"].to_numpy(), atol=1e-6)
    return saved, pd.Series(m_margin, index=mom["id"].to_numpy()), rng, st


def analyse(d, m_margin, rng, emb, keep_stim=None, keep_mom=None, qcol="quintile", xmode="rerank"):
    """P1-P5 + Holm as in inspiration_analysis, on a subset of stimuli (keep_stim: set of stim indices) and of
    MOMENTUM games. qcol: quintile variable for P2/P5. xmode: 'rerank' (drawn stimuli re-ranked by this embedder's
    dist, as in the paper), 'pool_rank' (Qwen pool rank) or 'quintile' ((q-1)/4 of qcol)."""
    if keep_stim is not None:
        d = d[(d["stim"] < 0) | d["stim"].isin(keep_stim)]
    mm = m_margin if keep_mom is None else m_margin[m_margin.index.isin(keep_mom)]
    res = {"n_stimuli": int(d.loc[d["stim"] >= 0, "stim"].nunique()), "n_momentum": int(len(mm))}
    for fr in FR:
        g = d[d["arm"] == fr]
        y, src = g["margin"].to_numpy(), g["source"].to_numpy()
        if xmode == "rerank":
            x = g.groupby("source")["dist"].rank(pct=True).to_numpy()
        elif xmode == "pool_rank":
            x = g["rank"].to_numpy(float)
        else:
            x = (g[qcol].to_numpy(float) - 1) / 4
        q = g[qcol].to_numpy(int)
        r = {"P1": slope_ci(y, x, src), "two_lines": two_lines_ci(y, x, src), "quadratic": quad_ci(y, x, src),
             "P3": slope_ci(g["iso"].to_numpy(), x, src),
             "P1_iso_adjusted": slope_ci(y, x, src, extra=g["iso"].to_numpy()),
             "r_margin_iso": float(np.corrcoef(y, g["iso"])[0, 1])}
        es, eq = A.eta2(y, src, q)
        bs = []
        for _ in range(A.N_BOOT):
            b = rng.choice(len(g), len(g))
            a1, a2 = A.eta2(y[b], src[b], q[b])
            bs.append(a1 - a2)
        r["P2"] = {"source": es, "quintile": eq, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   "p_boot": float((np.array(bs) <= 0).mean())}
        sg = g[g["source"] == "steam_game"]
        r["steam_game"] = {"margin": float(sg["margin"].mean()), "uptake": float(sg["uptake"].mean()),
                           "rank_of_9_lowest_first": int(g.groupby("source")["margin"].mean().rank()["steam_game"])}
        res[fr] = r
    a = d[d["arm"] == "INSTRUCTED"].set_index("stim")["uptake"]
    b_ = d[d["arm"] == "INCIDENTAL"].set_index("stim")["uptake"]
    both = a.index.intersection(b_.index)
    res["P4"] = {"instructed": float(a.mean()), "incidental": float(b_.mean()),
                 "p": float(stats.wilcoxon(a[both], b_[both], alternative="greater").pvalue)}
    none = d[d["arm"] == "NONE"]
    ps_ = d[d["stim"] >= 0].groupby("stim").agg(margin=("margin", "mean"), source=("source", "first"),
                                                 q=(qcol, "first"))
    p_none, p_mom = {}, {}
    for (s, q), h in ps_.groupby(["source", "q"]):
        p_none[f"{s}/Q{int(q)}"] = float(stats.mannwhitneyu(h["margin"], none["margin"], alternative="greater").pvalue)
        p_mom[f"{s}/Q{int(q)}"] = float(stats.mannwhitneyu(h["margin"], mm.to_numpy(), alternative="greater").pvalue)
    hn, hm = A.holm(p_none), A.holm(p_mom)
    res["P5"] = {"cells_beating_both": sorted(k for k in p_none if hn[k] < 0.05 and hm[k] < 0.05),
                 "n_cells": len(p_none), "momentum_margin": float(mm.mean()), "none_margin": float(none["margin"].mean())}
    for fr in FR:
        h = A.holm({"P1": res[fr]["P1"]["p_pos"], "P2": res[fr]["P2"]["p_boot"], "P3": res[fr]["P3"]["p_pos"],
                    "P4": res["P4"]["p"]})
        res[fr]["holm"] = h
        res[fr]["verdicts"] = {"P1_holds": h["P1"] >= 0.05, "P2_holds": h["P2"] < 0.05, "P3_holds": h["P3"] < 0.05,
                               "P4_holds": h["P4"] < 0.05}
    res["P5"]["holds"] = not res["P5"]["cells_beating_both"]
    return res


def appid_threshold() -> dict:
    m = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult", "date"])
    m["appid"] = pd.to_numeric(m["id"], errors="coerce")
    na = m[~m["adult"]]
    above = na["appid"] > A.APPID_CUT
    fut = na["split"] == "future"
    out = {"note": "release dates bound page creation from above only; a pre-cutoff release proves the appid "
                   "existed before the cutoff, a post-cutoff release does not prove the page was created after it",
           "n_future_nonadult": int(fut.sum()), "n_future_above_cut": int((fut & above).sum()),
           "n_future_at_or_below_cut": int((fut & ~above).sum()),
           "n_all_above_cut_nonadult": int(above.sum()),
           "n_past_split_above_cut": int((~fut & above).sum()),
           "n_above_cut_released_before_cutoff": int((above & (na["date"] < CUTOFF)).sum()),
           "n_above_cut_released_2023_or_earlier": int((above & (na["date"] < "2024-01-01")).sum()),
           "earliest_release_above_cut": str(na.loc[above, "date"].min().date())}
    ab = na[above].sort_values("date")
    out["earliest_10_above_cut"] = [{"appid": int(a), "date": str(t.date())} for a, t in zip(ab["appid"][:10], ab["date"][:10])]
    bins = {}
    for lo, hi in [(2_600_000, 2_640_000), (2_640_000, 2_680_000), (2_680_000, 2_700_000), (2_700_000, 2_720_000),
                   (2_720_000, 2_760_000), (2_760_000, 2_800_000)]:
        b = na[(na["appid"] > lo) & (na["appid"] <= hi)]["date"]
        bins[f"({lo},{hi}]"] = {"n": int(len(b)), "min": str(b.min().date()), "p10": str(b.quantile(0.1).date()),
                                "median": str(b.median().date()),
                                "share_released_before_cutoff": float((b < CUTOFF).mean()),
                                "share_released_2023_or_earlier": float((b < "2024-01-01").mean())}
    out["release_dates_by_appid_bin"] = bins
    # the largest appid released before the cutoff, and the smallest appid released in 2024+ (how sharp the order is)
    out["max_appid_released_before_cutoff"] = int(na.loc[na["date"] < CUTOFF, "appid"].max())
    out["n_future_below_cut_by_year_of_release"] = na[fut & ~above]["date"].dt.year.value_counts().sort_index().to_dict()
    return out


def cli_clustered() -> dict:
    """Stress test with the call (25 ideas) as cluster. Calls: rows are written per call in pick order (checked:
    INSTRUCTED rows equal the pick order, 450 rows = 18 calls of 25, then 4 NONE calls)."""
    st = pd.read_parquet(A.INS / "stimuli.parquet")
    picks = [i for _, g in st.groupby(["source", "quintile"]) for i in g.index[:10]]
    rng = np.random.default_rng(A.SEED)
    out = {}
    for gen in ("codex", "claude"):
        d = pd.read_parquet(A.DATA / "runs/inspiration" / f"{gen}-cli/scored_qwen3e06.parquet")
        assert d.loc[d["stim"] >= 0, "stim"].tolist() == picks and (d["arm"].iloc[450:] == "NONE").all()
        d["call"] = np.arange(len(d)) // 25
        d.loc[d["stim"] < 0, "source"] = "NONE"
        calls_per_source = d.groupby("source")["call"].unique().apply(lambda c: [int(x) for x in c]).to_dict()
        none = d[d["source"] == "NONE"]
        cm = d.groupby("call").agg(margin=("margin", "mean"), source=("source", "first"))
        comp = {}
        for s in A.SOURCES:
            h = d[d["source"] == s]
            p_idea = float(stats.mannwhitneyu(h["margin"], none["margin"]).pvalue)
            hc, nc = cm[cm["source"] == s]["margin"], cm[cm["source"] == "NONE"]["margin"]
            p_call = float(stats.mannwhitneyu(hc, nc, method="exact").pvalue)
            # cluster bootstrap of the mean difference: resample calls within each group, then ideas within call
            diffs = []
            gs = {c: g["margin"].to_numpy() for c, g in d.groupby("call")}
            sc, ncl = list(hc.index), list(nc.index)
            for _ in range(4000):
                a = np.concatenate([gs[c] for c in rng.choice(sc, len(sc))])
                b = np.concatenate([gs[c] for c in rng.choice(ncl, len(ncl))])
                diffs.append(a.mean() - b.mean())
            diffs = np.array(diffs)
            # exact permutation of call labels over the 2 + 4 calls (15 splits), idea-weighted mean difference
            allc = sc + ncl
            obs = h["margin"].mean() - none["margin"].mean()
            from itertools import combinations
            perm = []
            for pick in combinations(allc, len(sc)):
                a = np.concatenate([gs[c] for c in pick])
                b = np.concatenate([gs[c] for c in allc if c not in pick])
                perm.append(a.mean() - b.mean())
            perm = np.array(perm)
            comp[s] = {"diff": float(obs), "n_calls": len(sc), "p_idea_mw": p_idea, "p_callmeans_mw_exact": p_call,
                       "call_means": [float(x) for x in hc], "boot_ci95": [float(np.percentile(diffs, 2.5)),
                                                                            float(np.percentile(diffs, 97.5))],
                       "p_perm_calls_two_sided": float((np.abs(perm) >= abs(obs) - 1e-12).mean()),
                       "n_permutations": int(len(perm))}
        out_g = {"none_call_means": [float(x) for x in cm[cm["source"] == "NONE"]["margin"]],
                 "calls_per_source": calls_per_source, "comparisons": comp}
        for key in ("p_idea_mw", "p_callmeans_mw_exact", "p_perm_calls_two_sided"):
            out_g[f"holm9_{key}"] = A.holm({s: comp[s][key] for s in A.SOURCES})
        # distance slope (pool rank, as in the paper) with call clustering
        g = d[d["stim"] >= 0]
        y, x, src, cl = g["margin"].to_numpy(), g["rank"].to_numpy(float), g["source"].to_numpy(), g["call"].to_numpy()
        for name, yy in (("margin", y), ("iso", g["iso"].to_numpy())):
            X = np.column_stack([x, A.dummies(src, A.SOURCES)])
            b, se, dof = A.ols(yy, X)
            # CR1 cluster-robust SE, 18 clusters
            XtXi = np.linalg.pinv(X.T @ X)
            r = yy - X @ b
            meat = sum(np.outer(X[cl == c].T @ r[cl == c], X[cl == c].T @ r[cl == c]) for c in np.unique(cl))
            G, n, k = len(np.unique(cl)), len(yy), np.linalg.matrix_rank(X)
            V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)
            se_cr = float(np.sqrt(V[0, 0]))
            # wild cluster bootstrap-t (Rademacher, null slope = 0 imposed), 9,999 draws
            X0 = X[:, 1:]
            b0, *_ = np.linalg.lstsq(X0, yy, rcond=None)
            f0, r0 = X0 @ b0, yy - X0 @ b0
            uc = np.unique(cl)
            idx = np.searchsorted(uc, cl)
            tb = []
            for _ in range(9999):
                w = rng.choice([-1.0, 1.0], len(uc))[idx]
                ys = f0 + w * r0
                bb, *_ = np.linalg.lstsq(X, ys, rcond=None)
                rr = ys - X @ bb
                mt = sum(np.outer(X[cl == c].T @ rr[cl == c], X[cl == c].T @ rr[cl == c]) for c in uc)
                vv = XtXi @ mt @ XtXi * G / (G - 1) * (n - 1) / (n - k)
                tb.append(bb[0] / np.sqrt(vv[0, 0]))
            tb = np.abs(np.array(tb))
            # within-call slope: call fixed effects (absorb source), classical SE
            Xc = np.column_stack([x, A.dummies(cl, list(uc))])
            bc, sec, dofc = A.ols(yy, Xc)
            out_g[f"{name}_slope"] = {
                "slope": float(b[0]), "se_classical": float(se[0]), "ci95_classical": ci(b[0], se[0], dof),
                "se_cr1_18_calls": se_cr, "ci95_cr1_t17": ci(b[0], se_cr, G - 1),
                "p_two_cr1_t17": float(2 * stats.t.sf(abs(b[0] / se_cr), G - 1)),
                "p_two_wild_cluster_boot": float((tb >= abs(b[0] / se_cr)).mean()),
                "within_call_slope": float(bc[0]), "within_call_se": float(sec[0]),
                "within_call_ci95": ci(bc[0], sec[0], dofc)}
        out[gen] = out_g
    # Holm over the two generators' steam_game comparisons (the two claims the paper makes)
    for key in ("p_idea_mw", "p_callmeans_mw_exact", "p_perm_calls_two_sided"):
        out[f"holm2_steam_game_{key}"] = A.holm({gen: out[gen]["comparisons"]["steam_game"][key]
                                                 for gen in ("codex", "claude")})
    return out


def main() -> None:
    res = {"cutoff": str(CUTOFF.date())}
    meta = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "date"])
    st = pd.read_parquet(A.INS / "stimuli.parquet")
    mom = pd.read_parquet(A.INS / "momentum.parquet")
    sg = st[st["source"] == "steam_game"].reset_index(names="stim").merge(meta, on="id", how="left")
    late = sg[(sg["date"] >= CUTOFF) | sg["date"].isna()]
    momd = mom.merge(meta, on="id", how="left")
    mom_late = momd[(momd["date"] >= CUTOFF) | momd["date"].isna()]
    res["stimulus_dates"] = {
        "n_steam_game_stimuli": int(len(sg)), "n_unknown_date": int(sg["date"].isna().sum()),
        "n_on_or_after_cutoff": int((sg["date"] >= CUTOFF).sum()),
        "late_by_qwen_quintile": late.groupby("quintile").size().to_dict(),
        "late_stimuli": [{"stim": int(r.stim), "id": r.id, "date": str(r.date.date()), "quintile": int(r.quintile)}
                         for r in late.itertuples()],
        "steam_game_release_range": [str(sg["date"].min().date()), str(sg["date"].max().date())],
        "other_sources": "board_game <=2022, movie 2018 dump, sports_rules/wikipedia 2022-03-01 dump, paper <=2022, "
                         "patent 2012-2015, news 2004-05, recipe <=2019: all before the cutoff by construction",
        "momentum_n_on_or_after_cutoff": int(len(mom_late)), "momentum_n": int(len(mom))}
    keep_stim = set(st.index) - set(late["stim"])
    keep_mom = set(mom["id"]) - set(mom_late["id"])
    res["appid_threshold"] = appid_threshold()
    print(json.dumps(res["stimulus_dates"], indent=1)[:800], json.dumps(res["appid_threshold"], indent=1), flush=True)

    full, filt, filt2, est, cis = {}, {}, {}, {}, {}
    for emb in ("qwen3e06", "bgem3"):
        d, m_margin, rng_orig, st_ = replay(emb)
        # reproduction with the original rng state: must equal the saved result
        full[emb] = analyse(d, m_margin, rng_orig, emb)
        orig = json.loads(A.OUT.read_text())[emb]
        for fr in FR:
            assert abs(full[emb][fr]["P1"]["slope"] - orig[fr]["P1_margin_slope"]["slope"]) < 1e-9
            assert abs(full[emb][fr]["P2"]["p_boot"] - orig[fr]["P2_eta2"]["p_boot"]) < 1e-9, "P2 bootstrap not replayed"
        assert abs(full[emb]["P5"]["momentum_margin"] - orig["baselines"]["MOMENTUM"]["margin"]) < 1e-9
        r1 = lambda: np.random.default_rng(A.SEED + 1)  # noqa: E731 - same fresh bootstrap stream for every variant
        full_fresh = analyse(d, m_margin, r1(), emb)
        filt[emb] = {"full_same_bootstrap_stream": {fr: full_fresh[fr]["P2"] for fr in FR},
                     "excl_late_stimuli": analyse(d, m_margin, r1(), emb, keep_stim=keep_stim),
                     "excl_late_stimuli_and_late_momentum": analyse(d, m_margin, r1(), emb, keep_stim=keep_stim,
                                                                    keep_mom=keep_mom)}
        # (b) estimands
        pdist = pd.read_parquet(A.INS / "pool_dist.parquet")
        d = d.copy()
        has = d["stim"] >= 0
        prow = st_.loc[d.loc[has, "stim"].to_numpy(), "pool_row"].to_numpy()
        pdist["q_bge_pool"] = np.minimum(np.ceil(pdist.groupby("source")["dist_bgem3"].rank(pct=True) * 5), 5)
        d.loc[has, "q_bge_pool"] = pdist["q_bge_pool"].to_numpy()[prow]
        # re-binned among the drawn stimuli, as in paper_figures.fig5 (40 per cell when there are no ties)
        sd = st_.assign(dq=st_[f"dist_{emb}"])
        sd["q_rebin"] = np.ceil(sd.groupby("source")["dq"].rank(pct=True) * 5).clip(1, 5)
        d.loc[has, "q_rebin"] = sd["q_rebin"].to_numpy()[d.loc[has, "stim"].to_numpy()]
        e = {"rebin_cell_sizes": sd.groupby(["source", "q_rebin"]).size().describe()[["min", "max"]].to_dict()}
        if emb == "bgem3":
            e["P2_P5_bge_rebinned_drawn"] = analyse(d, m_margin, r1(), emb, qcol="q_rebin")
            e["P2_P5_bge_pool_quintiles"] = analyse(d, m_margin, r1(), emb, qcol="q_bge_pool")
            e["P1_on_bge_rebinned_quintile"] = {fr: slope_ci(d[d["arm"] == fr]["margin"].to_numpy(),
                                                             (d[d["arm"] == fr]["q_rebin"].to_numpy(float) - 1) / 4,
                                                             d[d["arm"] == fr]["source"].to_numpy()) for fr in FR}
        else:
            e["qwen_rebin_equals_quintile"] = bool((d.loc[has, "q_rebin"] == d.loc[has, "quintile"]).all())
            e["P1_on_qwen_quintile"] = {fr: slope_ci(d[d["arm"] == fr]["margin"].to_numpy(),
                                                     (d[d["arm"] == fr]["quintile"].to_numpy(float) - 1) / 4,
                                                     d[d["arm"] == fr]["source"].to_numpy()) for fr in FR}
            e["P1_on_qwen_pool_rank"] = {fr: slope_ci(d[d["arm"] == fr]["margin"].to_numpy(),
                                                      d[d["arm"] == fr]["rank"].to_numpy(float),
                                                      d[d["arm"] == fr]["source"].to_numpy()) for fr in FR}
        est[emb] = e
        cis[emb] = {fr: {k: full[emb][fr][k] for k in ("P1", "two_lines", "quadratic", "P1_iso_adjusted",
                                                        "r_margin_iso")} for fr in FR}
        print(emb, "done", flush=True)
    res["reproduction_full"] = full
    res["date_filter"] = filt
    res["estimands"] = {
        "map": {"P1": "continuous within-source rank of the drawn stimuli, re-ranked by each embedder's dist",
                "P3": "same as P1", "two_lines": "same as P1",
                "P2": "Qwen3 pool quintile (stimuli.parquet 'quintile') in BOTH embedders",
                "P5": "Qwen3 pool quintile cells in BOTH embedders",
                "fig5": "quintiles re-binned among the drawn stimuli by each embedder's dist (Qwen: identical to the "
                        "pool quintile)",
                "cli_slope": "Qwen3 pool rank (stimuli 'rank'), not re-ranked"},
        **est}
    res["ci"] = cis
    res["cli_clustered"] = cli_clustered()
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
