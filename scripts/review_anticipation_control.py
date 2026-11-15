"""Validation of the Section 6 anticipation margin (positive and format controls). Never calls an LLM.

Pre-registered in docs/prereg/review_anticipation_control.md (committed before this was run).

  # free arms A1, A2, A5, A6 and the A3 parts without A4 (one embedder at a time, ~1.5-2 GB peak)
  OMP_NUM_THREADS=4 uv run python scripts/review_anticipation_control.py free
  # after scripts/review_anticipation_rewrite.py has written runs/review-anticipation/rewrites.jsonl:
  OMP_NUM_THREADS=4 uv run python scripts/review_anticipation_control.py rewrite_score      # writes texts, embeds Qwen3
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 uv run --no-project --with sentence-transformers==3.0.1 \
      --with transformers==4.44.2 --with "numpy<2" python scripts/review_anticipation_control.py embed_bge
  OMP_NUM_THREADS=4 uv run python scripts/review_anticipation_control.py rewrite_score      # scores A4, Holm, outcome

Smoke test (tiny subset, temp outputs, never the registered JSON):
  uv run python scripts/review_anticipation_control.py free --smoke --out /tmp/x.json --work /tmp/work

Output: docs/prereg/review_anticipation_control.json; per-item margins and held-out ids in runs/review-anticipation/.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/inspiration/gpt-4o-0806"
INS = DATA / "prepared/inspiration"
WORK = DATA / "runs/review-anticipation"
OUT = REPO / "docs/prereg/review_anticipation_control.json"
SEED, B_PAST, N_BOOT, APPID_CUT, DEDUP = 20261114, 5, 4000, 2_680_000, 0.97
N_HELD, N_SPLITS, N_A5 = 200, 100, 20
EMBS = ("qwen3e06", "bgem3")
FR = ("INCIDENTAL", "INSTRUCTED")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# --------------------------------------------------------------------------------------------- data
def load_meta():
    import pandas as pd
    meta = pd.read_parquet(DATA / "prepared/steam/corpus.parquet", columns=["id", "date", "tags", "split", "adult", "text"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    clean = (~past) & (appid > APPID_CUT)
    dirty = (~past) & ~(appid > APPID_CUT)
    tagged = (meta["tags"].fillna("").str.strip() != "").to_numpy()
    y2023 = (meta["date"].dt.year == 2023).to_numpy()
    m = {"past": past, "clean": clean, "dirty": dirty, "tagged": tagged, "past23": past & y2023}
    return meta, keep, m


def load_E(emb: str, keep: np.ndarray) -> np.ndarray:
    """Corpus embeddings, adult rows removed, L2-normalised float32, filled in chunks (no full-size temporaries)."""
    raw = np.load(DATA / f"prepared/steam/emb_{emb}.npy", mmap_mode="r")
    rows = np.flatnonzero(keep)
    E = np.empty((len(rows), raw.shape[1]), np.float32)
    for i in range(0, len(rows), 8192):
        x = np.asarray(raw[rows[i:i + 8192]], np.float32)
        E[i:i + 8192] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return E


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def draws(meta, m, path: Path) -> dict:
    """All held-out sets, drawn once from the corpus index with default_rng(SEED), in the order of the prereg:
    A1 H0, A1 past-2023 arm, A1 splits H_1..H_100, A2 H_T, A2 past-2023 tagged arm, A5 clean subsets
    (20 of size |dirty|, 20 tagged-clean of size |dirty tagged|, 20 of the larger untagged set at the size of the
    smaller). Held-out ids are written to `path` before any margin is computed; an existing file must match."""
    rng = np.random.default_rng(SEED)
    idx = {k: np.flatnonzero(v) for k, v in m.items()}
    ct = np.flatnonzero(m["clean"] & m["tagged"])
    cu = np.flatnonzero(m["clean"] & ~m["tagged"])
    dt = np.flatnonzero(m["dirty"] & m["tagged"])
    du = np.flatnonzero(m["dirty"] & ~m["tagged"])
    p23t = np.flatnonzero(m["past23"] & m["tagged"])
    d = {"H0": rng.choice(idx["clean"], N_HELD, replace=False),
         "P23": rng.choice(idx["past23"], N_HELD, replace=False),
         "H_r": [rng.choice(idx["clean"], N_HELD, replace=False) for _ in range(N_SPLITS)],
         "H_T": rng.choice(ct, N_HELD, replace=False),
         "P23_T": rng.choice(p23t, N_HELD, replace=False)}
    d["A5_clean"] = [rng.choice(idx["clean"], len(idx["dirty"]), replace=False) for _ in range(N_A5)]
    d["A5_clean_tagged"] = [rng.choice(ct, len(dt), replace=False) for _ in range(N_A5)]
    small = min(len(du), len(cu))
    big_is_clean = len(cu) >= len(du)
    d["A5_untagged_big_is_clean"] = big_is_clean
    d["A5_untagged_sub"] = [rng.choice(cu if big_is_clean else du, small, replace=False) for _ in range(N_A5)]
    d["A5_untagged_small_full"] = du if big_is_clean else cu
    ids = meta["id"].to_numpy()
    rec = {"seed": SEED, "H0": ids[d["H0"]].tolist(), "past23": ids[d["P23"]].tolist(),
           "splits": [ids[h].tolist() for h in d["H_r"]], "H_T": ids[d["H_T"]].tolist(),
           "past23_tagged": ids[d["P23_T"]].tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        assert old == rec, f"{path} exists and differs from the redraw"
    else:
        path.write_text(json.dumps(rec))
    return d


def past_pools(m):
    return {"P": np.flatnonzero(m["past"]), "PT": np.flatnonzero(m["past"] & m["tagged"]),
            "PU": np.flatnonzero(m["past"] & ~m["tagged"])}


def draw_samples(rng, pool: np.ndarray, size: int) -> list[np.ndarray]:
    return [np.sort(pool[rng.choice(len(pool), size, replace=False)]) for _ in range(B_PAST)]


# --------------------------------------------------------------------------------------------- scoring
def margins(E, Q, excl, refs, chunk=400, dedup_counts=None):
    """Margin of every row of Q for each reference set in refs = [(fut_idx, past_samples, dedup), ...].
    excl[i] = corpus row of query i (its own row is set to -1 in every reference set) or -1. With dedup, every
    reference item with cos > DEDUP to the query is excluded, in the future set and in each past sample alike."""
    out = [np.empty(len(Q), np.float32) for _ in refs]
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ E.T
        for r, j in enumerate(excl[i:i + chunk]):
            if j >= 0:
                S[r, j] = -1
        Sd = None
        for k, (fi, ps, dd) in enumerate(refs):
            M = S
            if dd:
                if Sd is None:
                    Sd = np.where(S > DEDUP, np.float32(-1), S)
                    if dedup_counts is not None:
                        dedup_counts.append((S[:, fi] > DEDUP).sum(1))
                M = Sd
            fut = M[:, fi].max(1)
            pc = np.mean([M[:, p].max(1) for p in ps], 0)
            out[k][i:i + chunk] = fut - pc
        del S, Sd
    return out


# --------------------------------------------------------------------------------------------- statistics
def pct(a, q):
    return [float(v) for v in np.percentile(np.asarray(a, float), q)]


def boot(a, b, rng, n, paired=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        dd = a - b
        return dd[rng.integers(0, len(dd), (n, len(dd)))].mean(1)
    return a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)


def contrast(a, b, rng, n, sd, sd_none, paired=False):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a.mean() - b.mean()) if not paired else float((a - b).mean())
    bs = boot(a, b, rng, n, paired)
    ci = np.array(pct(bs, [2.5, 97.5]))
    r = {"diff": d, "ci95": ci.tolist(), "snr": d / sd, "snr_ci95": (ci / sd).tolist(),
         "snr_none_sd": d / sd_none, "snr_none_sd_ci95": (ci / sd_none).tolist(), "n": [len(a), len(b)]}
    if not paired:
        r["welch_p_greater"] = float(stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue)
    return r


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, ps[k] * (len(keys) - i)))
        out[k] = run
    return out


def scales(emb: str) -> dict:
    """sd_idea = pooled within-source residual SD of the Table 5 margins (y ~ source FE) per framing, averaged over
    the two framings (as strengthen_inspiration_shape.py); NONE SD as a second scale; SD of source means per framing."""
    import pandas as pd
    d = pd.read_parquet(RUN / f"scored_{emb}.parquet")
    per, spread = {}, {}
    for fr in FR:
        g = d[d["arm"] == fr]
        y, src = g["margin"].to_numpy(float), g["source"].to_numpy()
        resid = y - pd.Series(y).groupby(src).transform("mean").to_numpy()
        per[fr] = float(np.sqrt((resid ** 2).sum() / (len(y) - 9)))
        spread[fr] = float(pd.Series(y).groupby(src).mean().std(ddof=1))
    return {"sd_idea": float(np.mean(list(per.values()))), "sd_idea_by_framing": per,
            "sd_none": float(d[d["arm"] == "NONE"]["margin"].std(ddof=1)), "sd_source_means": spread}


def table5_refs(emb: str) -> dict:
    t5 = json.loads((REPO / "docs/prereg/inspiration_from_anything_result.json").read_text())[emb]
    anch = json.loads((REPO / "docs/prereg/strengthen_anchoring_none.json").read_text())[emb]["diff"]
    band = json.loads((REPO / "docs/prereg/strengthen_shape_band_observed.json").read_text())["cells"]
    U = {f"{df}/{fr}": band[f"{emb}/{fr}"][df]["B4000"]["U_band"] for df in ("df4", "df8") for fr in FR}
    return {"P1_slopes": {fr: t5[fr]["P1_margin_slope"]["slope"] for fr in FR}, "anchoring": abs(anch),
            "U_band": U, "U4": max(U[f"df4/{fr}"] for fr in FR), "U8": max(U[f"df8/{fr}"] for fr in FR),
            "dirty_none_obs": t5["exploratory_dirty_future_margin"]["NONE"], "clean_none_obs": t5["baselines"]["NONE"]["margin"]}


def yardstick(delta: float, sc: dict, ref: dict) -> dict:
    return {"delta": delta,
            "P1_slope_over_delta": {fr: ref["P1_slopes"][fr] / delta for fr in FR},
            "anchoring_over_delta": ref["anchoring"] / delta,
            "source_spread_over_delta": {fr: sc["sd_source_means"][fr] / delta for fr in FR},
            "U_band_over_delta": {k: v / delta for k, v in ref["U_band"].items()},
            "U4_over_delta": ref["U4"] / delta, "U8_over_delta": ref["U8"] / delta}


# --------------------------------------------------------------------------------------------- A6
def a6_isolation_range(emb: str) -> dict:
    import pandas as pd
    sys.path.insert(0, str(REPO / "scripts"))
    import inspiration_analysis as A
    from scipy import stats
    pool = pd.read_parquet(INS / "pool_dist.parquet")
    st = pd.read_parquet(INS / "stimuli.parquet")
    sc = pd.read_parquet(RUN / f"scored_{emb}.parquet")
    sc = sc[sc["arm"].isin(FR)]
    col = f"dist_{emb}"
    q = [0, 5, 25, 50, 75, 95, 100]
    lo_all, hi_all = np.percentile(st[col], [5, 95])
    width_all = hi_all - lo_all

    def desc(x):
        v = np.percentile(np.asarray(x, float), q)
        return {"min": v[0], "p5": v[1], "p25": v[2], "p50": v[3], "p75": v[4], "p95": v[5], "max": v[6],
                "iqr": v[4] - v[2], "range_5_95": v[5] - v[1], "ratio_5_95_to_pooled_stimuli": (v[5] - v[1]) / width_all,
                "n": int(len(x))}
    res = {"pooled_stimuli_5_95": [float(lo_all), float(hi_all)], "by_source": {}}
    ivs = []
    for s in A.SOURCES:
        ps, ss, ii = pool.loc[pool["source"] == s, col], st.loc[st["source"] == s, col], sc.loc[sc["source"] == s, "iso"]
        r = {"pool": desc(ps), "stimuli": desc(ss), "ideas_iso": desc(ii),
             "pool_quintile_cuts": np.percentile(ps, [20, 40, 60, 80]).tolist()}
        r["range_restricted"] = bool(r["stimuli"]["ratio_5_95_to_pooled_stimuli"] < 0.5)
        res["by_source"][s] = r
        ivs.append((r["stimuli"]["p5"], r["stimuli"]["p95"]))
    # share of the pooled 5-95 range covered by the union of the nine within-source 5-95 ranges
    ivs = sorted((max(a, lo_all), min(b, hi_all)) for a, b in ivs if min(b, hi_all) > max(a, lo_all))
    cov, cur = 0.0, None
    for a, b in ivs:
        if cur is None or a > cur[1]:
            if cur is not None:
                cov += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur is not None:
        cov += cur[1] - cur[0]
    res["union_coverage_of_pooled_5_95"] = float(cov / width_all)
    res["n_range_restricted"] = int(sum(r["range_restricted"] for r in res["by_source"].values()))
    # exploratory: P1 slope per 0.1 of raw isolation (within-source OLS of margin on dist_e, source FE)
    res["P1_slope_per_0.1_raw_isolation"] = {}
    for fr in FR:
        g = sc[sc["arm"] == fr]
        r = A.slope_within(g["margin"].to_numpy(float), g["dist"].to_numpy(float), g["source"].to_numpy())
        tq = stats.t.ppf(0.975, len(g) - 10)
        res["P1_slope_per_0.1_raw_isolation"][fr] = {"slope": 0.1 * r["slope"], "se": 0.1 * r["se"],
                                                    "ci95": [0.1 * (r["slope"] - tq * r["se"]),
                                                             0.1 * (r["slope"] + tq * r["se"])]}
    return json.loads(json.dumps(res, default=float))


# --------------------------------------------------------------------------------------------- free
def free(args) -> None:
    import pandas as pd
    sys.path.insert(0, str(REPO / "scripts"))
    import inspiration_analysis as A
    from scipy import stats
    work, out = Path(args.work), Path(args.out)
    smoke = args.smoke
    n_boot = 200 if smoke else N_BOOT
    n_split_score = 3 if smoke else N_SPLITS
    n_a5_score = 2 if smoke else N_A5
    meta, keep, m = load_meta()
    log("corpus", {k: int(v.sum()) for k, v in m.items()}, "tagged clean", int((m["clean"] & m["tagged"]).sum()))
    dr = draws(meta, m, work / "heldout_ids.json")
    log("held-out ids written", work / "heldout_ids.json")
    pools = past_pools(m)
    clean_idx = np.flatnonzero(m["clean"])
    ct = np.flatnonzero(m["clean"] & m["tagged"])
    dirty_idx = np.flatnonzero(m["dirty"])
    dt = np.flatnonzero(m["dirty"] & m["tagged"])
    pid = pd.Index(meta["id"].to_numpy())
    mom = pd.read_parquet(INS / "momentum.parquet")
    mom_rows = pid.get_indexer(mom["id"])
    assert (mom_rows >= 0).all() and m["past"][mom_rows].all()
    mom_tag = m["tagged"][mom_rows]
    res = json.loads(out.read_text()) if out.exists() else {}
    res.update({"prereg": "docs/prereg/review_anticipation_control.md", "seed": SEED, "n_boot": n_boot,
                "smoke": smoke, "sd_idea": {}, "A5_dirty": {}, "A6_isolation_range": {}})
    for emb in (args.emb or EMBS):
        log("==", emb)
        sc, ref = scales(emb), table5_refs(emb)
        res["sd_idea"][emb] = sc
        sd, sdn = sc["sd_idea"], sc["sd_none"]
        E = load_E(emb, keep)
        scored = pd.read_parquet(RUN / f"scored_{emb}.parquet")
        V = norm(np.load(RUN / f"ideas_emb_{emb}.npy"))
        assert len(V) == len(scored)
        none_i = np.flatnonzero((scored["arm"] == "NONE").to_numpy())
        VN = V[none_i]
        rng1 = np.random.default_rng(SEED + 1)  # past samples, per reference set in a fixed order
        rngb = np.random.default_rng(SEED + 2)  # bootstrap
        R = {}
        rows_out = []

        # ---- A1: H0 vs F0 = clean \ H0
        F0 = np.setdiff1d(clean_idx, dr["H0"])
        S0 = draw_samples(rng1, pools["P"], len(F0))
        Q = np.vstack([E[dr["H0"]], VN, E[mom_rows], E[dr["P23"]]])
        ex = np.concatenate([dr["H0"], -np.ones(len(VN), int), mom_rows, dr["P23"]])
        cnt = []
        mg, mgd = margins(E, Q, ex, [(F0, S0, False), (F0, S0, True)], dedup_counts=cnt)
        cut = np.cumsum([0, N_HELD, len(VN), len(mom_rows), N_HELD])
        arm = lambda x, k: x[cut[k]:cut[k + 1]]  # noqa: E731
        fut, non, mo, p23 = (arm(mg, k) for k in range(4))
        tg = m["tagged"][dr["H0"]]
        R["A1_heldout"] = {
            "n_reference": int(len(F0)), "means": {"fut_asis": float(fut.mean()), "NONE": float(non.mean()),
                                                  "MOMENTUM": float(mo.mean()), "past23_asis": float(p23.mean())},
            "delta_F": contrast(fut, non, rngb, n_boot, sd, sdn),
            "delta_P23": contrast(p23, non, rngb, n_boot, sd, sdn),
            "fut_minus_past23": contrast(fut, p23, rngb, n_boot, sd, sdn),
            "momentum_minus_none": float(mo.mean() - non.mean()),
            "by_tag_presence": {"tagged": {"n": int(tg.sum()), "fut_asis": float(fut[tg].mean()),
                                           "delta_vs_none": float(fut[tg].mean() - non.mean())},
                                "untagged": {"n": int((~tg).sum()), "fut_asis": float(fut[~tg].mean()),
                                             "delta_vs_none": float(fut[~tg].mean() - non.mean())}},
            "tag_share_heldout": float(tg.mean())}
        fd, nd, md, pd_ = (arm(mgd, k) for k in range(4))
        c = np.concatenate(cnt)
        R["A1_dedup"] = {"threshold": DEDUP, "delta_F": float(fd.mean() - nd.mean()),
                         "means": {"fut_asis": float(fd.mean()), "NONE": float(nd.mean()),
                                   "MOMENTUM": float(md.mean()), "past23_asis": float(pd_.mean())},
                         "queries_with_future_neardup": {"fut_asis": int((arm(c, 0) > 0).sum()),
                                                         "NONE": int((arm(c, 1) > 0).sum()),
                                                         "MOMENTUM": int((arm(c, 2) > 0).sum()),
                                                         "past23_asis": int((arm(c, 3) > 0).sum())}}
        for name, k in (("fut_asis", 0), ("NONE", 1), ("MOMENTUM", 2), ("past23_asis", 3)):
            ids_ = (meta["id"].to_numpy()[ex[cut[k]:cut[k + 1]]] if name != "NONE"
                    else [f"idea{j}" for j in none_i])
            rows_out += [{"scoring": "F0", "arm": name, "id": str(i_), "margin": float(v)}
                         for i_, v in zip(ids_, arm(mg, k))]
        log("A1", json.dumps(R["A1_heldout"]["means"]), "dF", round(R["A1_heldout"]["delta_F"]["diff"], 5),
            "snr", round(R["A1_heldout"]["delta_F"]["snr"], 3), "dedup dF", round(R["A1_dedup"]["delta_F"], 5))

        # ---- A1 split variability
        sp = []
        for r_, H in enumerate(dr["H_r"]):
            Fr = np.setdiff1d(clean_idx, H)
            Sr = draw_samples(rng1, pools["P"], len(Fr))
            if r_ >= n_split_score:
                continue
            Q = np.vstack([E[H], VN])
            ex = np.concatenate([H, -np.ones(len(VN), int)])
            (mr,) = margins(E, Q, ex, [(Fr, Sr, False)])
            sp.append(float(mr[:N_HELD].mean() - mr[N_HELD:].mean()))
            if (r_ + 1) % 10 == 0 or r_ == n_split_score - 1:
                log(f"split {r_ + 1}/{n_split_score}")
        R["A1_splits"] = {"n_splits": len(sp), "mean": float(np.mean(sp)), "q2.5_q97.5": pct(sp, [2.5, 97.5]),
                          "values": sp}

        # ---- A2: tagged universe, H_T vs F_T \ H_T, past samples from tagged past
        FT0 = np.setdiff1d(ct, dr["H_T"])
        ST0 = draw_samples(rng1, pools["PT"], len(FT0))
        Q = np.vstack([E[dr["H_T"]], VN, E[mom_rows], E[dr["P23_T"]]])
        ex = np.concatenate([dr["H_T"], -np.ones(len(VN), int), mom_rows, dr["P23_T"]])
        (mt,) = margins(E, Q, ex, [(FT0, ST0, False)])
        futT, nonT, moT, p23T = (arm(mt, k) for k in range(4))
        R["A2_tagged"] = {
            "n_reference": int(len(FT0)), "n_past_tagged": int(len(pools["PT"])),
            "means": {"fut_tagged": float(futT.mean()), "NONE": float(nonT.mean()),
                      "MOMENTUM_tagged": float(moT[mom_tag].mean()), "MOMENTUM_all": float(moT.mean()),
                      "past23_tagged": float(p23T.mean())},
            "n_momentum_tagged": int(mom_tag.sum()),
            "delta_T": contrast(futT, nonT, rngb, n_boot, sd, sdn),
            "delta_P23_T": contrast(p23T, nonT, rngb, n_boot, sd, sdn),
            "fut_minus_past23_tagged": contrast(futT, p23T, rngb, n_boot, sd, sdn),
            "momentum_tagged_minus_none": float(moT[mom_tag].mean() - nonT.mean())}
        for name, k in (("fut_tagged", 0), ("NONE", 1), ("MOMENTUM", 2), ("past23_tagged", 3)):
            ids_ = (meta["id"].to_numpy()[ex[cut[k]:cut[k + 1]]] if name != "NONE"
                    else [f"idea{j}" for j in none_i])
            rows_out += [{"scoring": "FT0", "arm": name, "id": str(i_), "margin": float(v)}
                         for i_, v in zip(ids_, arm(mt, k))]
        log("A2", json.dumps(R["A2_tagged"]["means"]), "dT", round(R["A2_tagged"]["delta_T"]["diff"], 5))

        # ---- A2 exploratory: Table 5 rescored in the tagged universe (full F_T, no hold-out)
        STf = draw_samples(rng1, pools["PT"], len(ct))
        stim_i = np.flatnonzero(scored["arm"].isin(FR).to_numpy())
        if smoke:
            stim_i = np.sort(np.random.default_rng(SEED + 99).choice(stim_i, 360, replace=False))
        Q = np.vstack([V[stim_i], VN, E[mom_rows]])
        ex = np.concatenate([-np.ones(len(stim_i) + len(VN), int), mom_rows])
        (mf,) = margins(E, Q, ex, [(ct, STf, False)], chunk=500)
        g = scored.iloc[stim_i].assign(margin_T=mf[:len(stim_i)])
        rs = {"n_reference": int(len(ct)), "n_ideas": int(len(stim_i)),
              "NONE": float(mf[len(stim_i):len(stim_i) + len(VN)].mean()),
              "MOMENTUM_all": float(mf[len(stim_i) + len(VN):].mean()),
              "MOMENTUM_tagged": float(mf[len(stim_i) + len(VN):][mom_tag].mean())}
        for fr in FR:
            h = g[g["arm"] == fr]
            xr = h.groupby("source")["dist"].rank(pct=True).to_numpy()
            s = A.slope_within(h["margin_T"].to_numpy(float), xr, h["source"].to_numpy())
            tq = stats.t.ppf(0.975, len(h) - 10)
            rs[fr] = {"P1_slope": s["slope"], "se": s["se"], "ci95": [s["slope"] - tq * s["se"], s["slope"] + tq * s["se"]],
                      "table5_slope": ref["P1_slopes"][fr],
                      "sign_reversed": bool(np.sign(s["slope"]) != np.sign(ref["P1_slopes"][fr])),
                      "by_source": h.groupby("source")["margin_T"].mean().to_dict()}
        R["A2_rescored_table5"] = rs
        log("A2 rescored", {fr: round(rs[fr]["P1_slope"], 5) for fr in FR})
        del Q
        gc.collect()

        # ---- A5: dirty future set, NONE ideas only
        ex = -np.ones(len(VN), int)
        S_d = draw_samples(rng1, pools["P"], len(dirty_idx))
        refs, labels = [(dirty_idx, S_d, False)], ["dirty"]
        for k, sub in enumerate(dr["A5_clean"]):
            s_ = draw_samples(rng1, pools["P"], len(sub))
            if k < n_a5_score:
                refs.append((np.sort(sub), s_, False)); labels.append("clean_sub")
        refs.append((dt, draw_samples(rng1, pools["PT"], len(dt)), False)); labels.append("dirty_tagged")
        for k, sub in enumerate(dr["A5_clean_tagged"]):
            s_ = draw_samples(rng1, pools["PT"], len(sub))
            if k < n_a5_score:
                refs.append((np.sort(sub), s_, False)); labels.append("clean_tagged_sub")
        small = dr["A5_untagged_small_full"]
        refs.append((np.sort(small), draw_samples(rng1, pools["PU"], len(small)), False))
        labels.append("small_untagged_full")
        for k, sub in enumerate(dr["A5_untagged_sub"]):
            s_ = draw_samples(rng1, pools["PU"], len(sub))
            if k < n_a5_score:
                refs.append((np.sort(sub), s_, False)); labels.append("big_untagged_sub")
        out5 = margins(E, VN, ex, refs)
        mean_of = lambda lab: [float(o.mean()) for o, l in zip(out5, labels) if l == lab]  # noqa: E731
        dirty_m, clean_sub = mean_of("dirty")[0], mean_of("clean_sub")
        dt_m, ct_sub = mean_of("dirty_tagged")[0], mean_of("clean_tagged_sub")
        su, bu = mean_of("small_untagged_full")[0], mean_of("big_untagged_sub")
        g_size = dirty_m - float(np.mean(clean_sub))
        g_tag = dt_m - float(np.mean(ct_sub))
        if dr["A5_untagged_big_is_clean"]:
            g_untag = su - float(np.mean(bu))
        else:
            g_untag = float(np.mean(bu)) - su
        dates = meta["date"]
        largest = max([abs(v) for v in ref["P1_slopes"].values()] + [ref["anchoring"]])
        g_obs = ref["dirty_none_obs"] - ref["clean_none_obs"]
        res["A5_dirty"][emb] = {
            "G_obs_table5": g_obs, "none_dirty": dirty_m, "none_clean_size_matched": clean_sub,
            "G_size": g_size, "n_dirty": int(len(dirty_idx)),
            "none_dirty_tagged": dt_m, "none_clean_tagged_matched": ct_sub, "G_tag": g_tag, "n_dirty_tagged": int(len(dt)),
            "untagged": {"n_matched": int(len(small)), "dirty_untagged": su if dr["A5_untagged_big_is_clean"] else None,
                         "clean_untagged_subsets": bu, "G_untag": g_untag},
            "median_release": {k: str(dates[m_].median().date()) for k, m_ in
                               (("clean", m["clean"]), ("dirty", m["dirty"]),
                                ("clean_tagged", m["clean"] & m["tagged"]), ("dirty_tagged", m["dirty"] & m["tagged"]))},
            "reading": ("format" if abs(g_tag) < 0.5 * g_size else "not_format"),
            "G_obs_over_largest_table5_effect": g_obs / largest, "largest_table5_effect": largest}
        log("A5", {k: round(v, 5) for k, v in (("G_size", g_size), ("G_tag", g_tag), ("G_untag", g_untag))},
            res["A5_dirty"][emb]["reading"])

        # ---- A3 (parts without A4)
        R["A3_snr_yardstick"] = {"delta_F": {"snr": R["A1_heldout"]["delta_F"]["snr"],
                                             "snr_ci95": R["A1_heldout"]["delta_F"]["snr_ci95"],
                                             "snr_none_sd": R["A1_heldout"]["delta_F"]["snr_none_sd"],
                                             **yardstick(R["A1_heldout"]["delta_F"]["diff"], sc, ref)},
                                 "delta_T": {"snr": R["A2_tagged"]["delta_T"]["snr"],
                                             "snr_ci95": R["A2_tagged"]["delta_T"]["snr_ci95"],
                                             "snr_none_sd": R["A2_tagged"]["delta_T"]["snr_none_sd"],
                                             **yardstick(R["A2_tagged"]["delta_T"]["diff"], sc, ref)},
                                 "table5_refs": ref}
        R["outcome"] = "pending A4 (rewrite_score)"
        res["A6_isolation_range"][emb] = a6_isolation_range(emb)
        res[emb] = {**res.get(emb, {}), **R}
        work.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_parquet(work / f"margins_{emb}.parquet")
        out.write_text(json.dumps(res, indent=1, default=float))
        log("wrote", out)
        del E, V, VN
        gc.collect()


# --------------------------------------------------------------------------------------------- A4
def read_rewrites(work: Path):
    import pandas as pd
    rw = pd.DataFrame([json.loads(l) for l in (work / "rewrites.jsonl").read_text().splitlines() if l.strip()])
    rw = rw.drop_duplicates(["id", "group"], keep="last")
    return rw


def rewrite_texts(work: Path):
    """Texts of the successful rewrites, in item_text form (the idea template)."""
    sys.path.insert(0, str(REPO / "src"))
    from latent_gap.steam import item_text
    rw = read_rewrites(work)
    ok = rw[(rw["title"] != "") & (rw["short_description"] != "")].reset_index(drop=True)
    texts = [item_text(g, t, d) for g, t, d in zip(ok["genres"], ok["tags"], ok["short_description"])]
    return rw, ok, texts


def embed_bge(args) -> None:
    """Run in the sentence-transformers env (see module docstring). Same settings as strengthen_bge_embed.py."""
    import os
    import torch
    from sentence_transformers import SentenceTransformer
    work = Path(args.work)
    rows = json.loads((work / "rewrites_texts.json").read_text())
    T = [r["text"] for r in rows]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl = SentenceTransformer("BAAI/bge-m3", device=device)
    mdl.max_seq_length = 512
    order = sorted(range(len(T)), key=lambda i: -len(T[i]))
    Vb = np.zeros((len(T), 1024), dtype=np.float32)
    for b in range(0, len(order), 32):
        idx = order[b:b + 32]
        Vb[idx] = mdl.encode([T[i] for i in idx], batch_size=4, normalize_embeddings=True, convert_to_numpy=True)
        if device == "mps":
            torch.mps.empty_cache()
        print(f"bge {min(b + 32, len(T))}/{len(T)} ({device})", flush=True)
    np.save(work / "rewrites_emb_bgem3.npy", Vb)
    print("done", Vb.shape, flush=True)


def outcome_one(r: dict) -> str:
    sig = lambda k: r["holm"][k] < 0.05  # noqa: E731
    if sig("delta_R") and r["L_snr_R"] >= 0.5 and sig("D_R"):
        return "V"
    if sig("delta_R"):
        return "W"
    if sig("delta_F") or sig("delta_T"):
        return "F"
    return "N"


def rewrite_score(args) -> None:
    import pandas as pd
    work, out = Path(args.work), Path(args.out)
    res = json.loads(out.read_text())
    rw, ok, texts = rewrite_texts(work)
    tpath = work / "rewrites_texts.json"
    rec = [{"id": i, "group": g, "text": t} for i, g, t in zip(ok["id"], ok["group"], texts)]
    if not tpath.exists() or json.loads(tpath.read_text()) != rec:
        tpath.write_text(json.dumps(rec))
        for e in EMBS:
            (work / f"rewrites_emb_{e}.npy").unlink(missing_ok=True)
    qp = work / "rewrites_emb_qwen3e06.npy"
    if not qp.exists():
        import os
        sys.path.insert(0, str(REPO / "src"))
        from latent_gap.embedder import LocalQwen
        log("embedding", len(texts), "rewrites with Qwen3 (ONNX)")
        np.save(qp, LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4"))).embed(texts))
    if not (work / "rewrites_emb_bgem3.npy").exists():
        raise SystemExit("run the embed_bge subcommand (module docstring), then rewrite_score again")
    n_fail = {g: int(((rw["group"] == g) & ((rw["title"] == "") | (rw["short_description"] == ""))).sum())
              for g in ("fut", "past23")}
    ids_ = json.loads((work / "heldout_ids.json").read_text())
    n_missing = {"fut": len(set(ids_["H0"]) - set(rw.loc[rw.group == "fut", "id"])),
                 "past23": len(set(ids_["past23"]) - set(rw.loc[rw.group == "past23", "id"]))}
    res["n_failed_rewrites"] = {"failed": n_fail, "not_generated": n_missing,
                                "over_10_in_a_group": any(v > 10 for v in n_fail.values())}
    meta, keep, m = load_meta()
    dr = draws(meta, m, work / "heldout_ids.json")  # asserts the A1 draw is unchanged
    pools = past_pools(m)
    clean_idx = np.flatnonzero(m["clean"])
    pid = pd.Index(meta["id"].to_numpy())
    grp = ok["group"].to_numpy()
    orig_rows = pid.get_indexer(ok["id"].astype(str))
    assert (orig_rows >= 0).all()
    ex = np.where(grp == "past23", orig_rows, -1)  # a past23 rewrite's original game is excluded from its samples
    pvals_all, per = {}, {}
    for emb in EMBS:
        log("==", emb)
        sc, ref = scales(emb), table5_refs(emb)
        sd, sdn = sc["sd_idea"], sc["sd_none"]
        E = load_E(emb, keep)
        rng1 = np.random.default_rng(SEED + 1)
        F0 = np.setdiff1d(clean_idx, dr["H0"])
        S0 = draw_samples(rng1, pools["P"], len(F0))  # first reference set of `free`: the same samples (CRN)
        W = norm(np.load(work / f"rewrites_emb_{emb}.npy"))
        assert len(W) == len(ok)
        (mw,) = margins(E, W, ex, [(F0, S0, False)])
        mg = pd.read_parquet(work / f"margins_{emb}.parquet")
        g = lambda sc_, a: mg[(mg.scoring == sc_) & (mg.arm == a)].set_index("id")["margin"]  # noqa: E731
        fut_asis, none, p23_asis = g("F0", "fut_asis"), g("F0", "NONE").to_numpy(), g("F0", "past23_asis")
        futT, noneT = g("FT0", "fut_tagged").to_numpy(), g("FT0", "NONE").to_numpy()
        fut_rw = pd.Series(mw[grp == "fut"], index=ok.loc[grp == "fut", "id"].astype(str).to_numpy())
        p23_rw = pd.Series(mw[grp == "past23"], index=ok.loc[grp == "past23", "id"].astype(str).to_numpy())
        rngb = np.random.default_rng(SEED + 2)
        pair = fut_rw.index.intersection(fut_asis.index)
        A4 = {"n": {"fut_rw": int(len(fut_rw)), "past23_rw": int(len(p23_rw))},
              "means": {"fut_rw": float(fut_rw.mean()), "past23_rw": float(p23_rw.mean()), "NONE": float(none.mean()),
                        "fut_asis": float(fut_asis.mean()), "past23_asis": float(p23_asis.mean())},
              "delta_R": contrast(fut_rw.to_numpy(), none, rngb, N_BOOT, sd, sdn),
              "D_R": contrast(fut_rw.to_numpy(), p23_rw.to_numpy(), rngb, N_BOOT, sd, sdn),
              "style_cost_paired": contrast(fut_rw[pair].to_numpy(), fut_asis[pair].to_numpy(), rngb, N_BOOT, sd, sdn,
                                            paired=True),
              "past23_rw_minus_none": contrast(p23_rw.to_numpy(), none, rngb, N_BOOT, sd, sdn)}
        from scipy import stats
        welch = lambda a, b: float(stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue)  # noqa: E731
        ps = {"delta_F": welch(fut_asis.to_numpy(), none), "delta_T": welch(futT, noneT),
              "delta_R": welch(fut_rw.to_numpy(), none), "D_R": welch(fut_rw.to_numpy(), p23_rw.to_numpy())}
        pvals_all.update({f"{emb}/{k}": v for k, v in ps.items()})
        R = res[emb]
        R["A4_rewrite"] = A4
        R["A3_snr_yardstick"]["delta_R"] = {"snr": A4["delta_R"]["snr"], "snr_ci95": A4["delta_R"]["snr_ci95"],
                                            "snr_none_sd": A4["delta_R"]["snr_none_sd"],
                                            "snr_none_sd_ci95": A4["delta_R"]["snr_none_sd_ci95"],
                                            **yardstick(A4["delta_R"]["diff"], sc, ref)}
        dR = A4["delta_R"]["diff"]
        per[emb] = {"L_snr_R": A4["delta_R"]["snr_ci95"][0], "delta_R": dR, "U4": ref["U4"], "U8": ref["U8"],
                    "mom": {"delta_P23": R["A1_heldout"]["delta_P23"]["diff"], "past23_rw_minus_none":
                            A4["past23_rw_minus_none"]["diff"], "delta_F": R["A1_heldout"]["delta_F"]["diff"],
                            "delta_R": dR}}
        log("A4", json.dumps(A4["means"]), "dR", round(dR, 5), "snrR", round(A4["delta_R"]["snr"], 3))
        del E
        gc.collect()
    adj = holm(pvals_all)
    res["holm_family"] = {"raw": pvals_all, "adjusted": adj}
    labels = {}
    for emb in EMBS:
        r = per[emb]
        r["holm"] = {k: adj[f"{emb}/{k}"] for k in ("delta_F", "delta_T", "delta_R", "D_R")}
        labels[emb] = outcome_one(r)
        u4 = r["U4"] / r["delta_R"] if r["delta_R"] > 0 else float("inf")
        info = "informative" if u4 < 0.25 else ("uninformative" if u4 >= 0.5 else "in_between")
        mo = r["mom"]
        res[emb]["holm"] = r["holm"]
        res[emb]["outcome"] = {"label": labels[emb], "L_snr_R": r["L_snr_R"],
                               "null_informativeness": {"U4_over_delta_R": u4, "U8_over_delta_R":
                                                        (r["U8"] / r["delta_R"] if r["delta_R"] > 0 else float("inf")),
                                                        "reading": info},
                               "momentum_reading_consistent": bool(mo["delta_P23"] <= 0 and mo["past23_rw_minus_none"] <= 0
                                                                   and mo["delta_F"] > 0 and mo["delta_R"] > 0)}
    order = "NFWV"
    overall = min(labels.values(), key=order.index)
    infos = [res[e]["outcome"]["null_informativeness"]["reading"] for e in EMBS]
    res["outcome"] = {"label": overall, "by_embedder": labels,
                      "null_informative_both": all(i == "informative" for i in infos),
                      "null_uninformative_any": any(i == "uninformative" for i in infos),
                      "rule": "per-embedder label (V > W > F > N per the prereg table); the weaker embedder governs"}
    out.write_text(json.dumps(res, indent=1, default=float))
    log("outcome", json.dumps(res["outcome"]), "wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["free", "rewrite_score", "embed_bge"])
    ap.add_argument("--smoke", action="store_true", help="3 splits, 2 A5 subsets, 360 rescored ideas, 200 resamples")
    ap.add_argument("--emb", nargs="*", choices=EMBS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--work", default=str(WORK))
    args = ap.parse_args()
    if args.smoke and (Path(args.out).resolve() == OUT.resolve() or Path(args.work).resolve() == WORK.resolve()):
        raise SystemExit("--smoke must write to a temporary --out and --work")
    {"free": free, "rewrite_score": rewrite_score, "embed_bge": embed_bge}[args.cmd](args)


if __name__ == "__main__":
    main()
