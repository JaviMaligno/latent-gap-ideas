"""Part B analysis of docs/prereg/review_tangent_confounds.md (Section 11.3-11.4): P1 (rewrite gap, M4) and P2
(paraphrase / nearby-paper null for the tangent at matched step, M3(b)). Free; reads the rewrites produced by
scripts/review_rewrite_generate.py (DISC, REP; paid) and kaggle/review-rewrite (OLMO-C; Kaggle).

  OMP_NUM_THREADS=4 uv run python scripts/review_rewrite_null.py analyse          # -> docs/prereg/review_rewrite_null.json
  uv run python scripts/review_rewrite_null.py analyse --synthetic --cells DISC-4o --spaces qwen3e06 --n-ctx 10 \
      --n-boot 200 --out /tmp/x.json      # smoke on fake rewrites built from existing vectors (no generation)

Inputs: runs/review-rewrite/disc/rewrites.jsonl + kaggle-output/review-rewrite-disc-embed/emb_{space}.npy (rows =
parsed rewrites sorted by id, the order of texts.parquet); runs/review-rewrite/rep/{rewrites.jsonl,
rewrites_emb_{space}.npy} (same order); kaggle-output/review-rewrite-olmo/{rewrites.jsonl, rw_{space}.npy} (rows =
items with an abstract in file order). f, p, s come from the original follow-ups through the Part A engine
(review_tangent_confounds.engine, keep_geom); rewritten follow-ups in the f-set use their leave-one-out s_-j.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import review_tangent_confounds as RTC  # noqa: E402
from review_tangent_confounds import CRIT, change, fe_ols, holm, strip, summ, unit, tang  # noqa: E402

DATA = RTC.DATA
OUT = ROOT / "docs/prereg/review_rewrite_null.json"
SEED = RTC.SEED
CELLS = {"DISC-4o": "disc", "REP-4o": "rep", "OLMO-C": "olmo"}
CONTENT_MIN = 0.80


def load_rewrites(study: str, space: str):
    if study == "disc":
        rows = [json.loads(l) for l in (DATA / "runs/review-rewrite/disc/rewrites.jsonl").read_text().splitlines()]
        rows = sorted((r for r in rows if r["abstract"]), key=lambda r: r["id"])
        V = np.load(DATA / f"kaggle-output/review-rewrite-disc-embed/emb_{space}.npy")
    elif study == "rep":
        rows = [json.loads(l) for l in (DATA / "runs/review-rewrite/rep/rewrites.jsonl").read_text().splitlines()]
        rows = sorted((r for r in rows if r["abstract"]), key=lambda r: r["id"])
        V = np.load(DATA / f"runs/review-rewrite/rep/rewrites_emb_{space}.npy")
    else:
        K = DATA / "kaggle-output/review-rewrite-olmo"
        rows = [r for r in map(json.loads, (K / "rewrites.jsonl").read_text().splitlines()) if r["abstract"]]
        V = np.load(K / f"rw_{space}.npy")
    assert len(rows) == len(V), (study, space, len(rows), len(V))
    return rows, unit(V.astype(np.float32))


def all_rows_count(study: str) -> dict:
    p = {"disc": DATA / "runs/review-rewrite/disc/rewrites.jsonl", "rep": DATA / "runs/review-rewrite/rep/rewrites.jsonl",
         "olmo": DATA / "kaggle-output/review-rewrite-olmo/rewrites.jsonl"}[study]
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return {s: {"items": sum(r["set"] == s for r in rows), "dropped_after_retry": sum(r["set"] == s and not r["abstract"] for r in rows),
                "needed_retry": sum(r["set"] == s and r.get("attempts", 1) > 1 for r in rows)} for s in ("RW-F", "NA", "NB")}


def synthetic_rewrites(cell, corp, rng):
    """Smoke only: fake rewrites from existing vectors (RW-F = follow-up + noise, NA = context + noise, NB = 3 corpus
    papers from ranks k+1..k+200), texts copied from the originals."""
    rows, V = [], []
    d = cell.C.shape[1]
    for i in range(cell.n):
        c = cell.C[i]
        for j, f in enumerate(cell.ctx[i]["followups"]):
            if cell.name == "OLMO-C" and j not in set(cell.fset[i].tolist()) and j % 3:
                continue
            rows.append({"id": f"rwf/{i}/{j}", "set": "RW-F", "ctx": i, "j": j, "orig_id": f["arxiv_id"],
                         "continuation": j in set((cell.labels or {}).get(i, {}).get("cont", [])),
                         "title": f["title"], "abstract": f["abstract"]})
            V.append(unit(cell.F[i][j] + 0.25 * unit(rng.normal(size=d)) + 0.1 * c))
        for r in range(3):
            rows.append({"id": f"na/{i}/{r}", "set": "NA", "ctx": i, "r": r, "orig_id": cell.ctx[i]["arxiv_id"],
                         "title": "x", "abstract": "y"})
            V.append(unit(c + 0.35 * unit(rng.normal(size=d))))
        sc = corp.E @ c
        order = np.argsort(-sc)[len(cell.F[i]):len(cell.F[i]) + 200]
        for r, k in enumerate(rng.choice(200, 3, replace=False)):
            rows.append({"id": f"nb/{i}/{r}", "set": "NB", "ctx": i, "r": r, "orig_id": str(corp.ids[order[k]]),
                         "title": "x", "abstract": "y"})
            V.append(corp.E[order[k]])
    return rows, np.array(V, np.float32)


def item_geometry(cell, eng, rows, V):
    """Per rewrite: ctx, set, step, T.f, T.p, T.s (rewritten follow-ups in the f-set: LOO f_-j, s_-j)."""
    out = []
    for n, r in enumerate(rows):
        i = r["ctx"]
        g = eng["geom"][i]
        if g is None:
            continue
        c = g["c"]
        x = V[n]
        t = tang(x[None], c)[0]
        f, s, p = g["f"], g["s"], g["p"]
        if r["set"] == "RW-F":
            js = g["js"].tolist()
            if r["j"] in js and len(js) > 1:
                TC = g["TF"][g["js"]]
                fm = unit(TC.sum(0) - g["TF"][r["j"]])
                f = fm
                s = unit(fm - (fm @ p) * p)
        out.append([i, ["RW-F", "NA", "NB"].index(r["set"]), float(x @ c), float(t @ f), float(t @ p), float(t @ s)])
    return np.array(out, float)


def content_check(cell, corp, rows, V) -> tuple[dict, dict]:
    cos = {}
    for n, r in enumerate(rows):
        i = r["ctx"]
        if r["set"] == "RW-F":
            o = cell.F[i][r["j"]]
        elif r["set"] == "NA":
            o = cell.C[i]
        else:
            if r["orig_id"] not in corp.row:
                continue
            o = corp.E[corp.row[r["orig_id"]]]
        cos[r["id"]] = float(V[n] @ o)
    res = {}
    for s in ("RW-F", "NA", "NB"):
        x = np.array([v for k, v in cos.items() if k.startswith(s.lower().replace("-", "") + "/")])
        if len(x):
            res[s] = {f"q{q}": float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95)} | {
                "mean": float(x.mean()), "share_below_0.80": float((x < CONTENT_MIN).mean()), "n": int(len(x))}
    return res, cos


def p1(cell, eng, rows, V, rng, exclude: set) -> dict:
    """gap_RW = median step of the ideas - median step of the rewritten follow-ups (per context)."""
    out = {}
    if cell.pb is not None:
        idea_step = {i: cell.pb[1][cell.pb_index[i]] @ cell.C[i] for i in cell.pb_index}
        idea_set = "one-shot CTX ideas"
    else:
        idea_step = {}
        for i in range(cell.n):
            idx = [n for a in CRIT for n in cell.index.get((a, i, 0), [])]
            if idx:
                idea_step[i] = cell.V[idx] @ cell.C[i]
        idea_set = "round-0 critique ideas"
    rw = {}
    for n, r in enumerate(rows):
        if r["set"] == "RW-F" and r["id"] not in exclude:
            rw.setdefault(r["ctx"], []).append((r["j"], float(V[n] @ cell.C[r["ctx"]]), bool(r.get("continuation"))))
    lab = cell.labels or {}
    g_all, g_cont, d_rw, g_orig_cont = [], [], [], []
    for i in range(cell.n):
        if i not in rw or i not in idea_step:
            continue
        mi = float(np.median(idea_step[i]))
        st = np.array([s for _, s, _ in rw[i]])
        g_all.append(mi - np.median(st))
        orig = np.array([float(cell.F[i][j] @ cell.C[i]) for j, _, _ in rw[i]])
        d_rw.append(float(np.median(st) - np.median(orig)))
        cont = set(lab.get(i, {}).get("cont", []))
        sc = [s for j, s, _ in rw[i] if j in cont]
        if sc:
            g_cont.append(mi - np.median(sc))
            g_orig_cont.append(mi - np.median([float(cell.F[i][j] @ cell.C[i]) for j in cont]))
    out["idea_set"] = idea_set
    for nm, x, alt in (("gap_RW", g_all, "greater"), ("gap_RW_cont", g_cont, "greater"), ("delta_RW", d_rw, "two-sided"),
                       ("gap_original_vs_unrewritten_continuations", g_orig_cont, "greater")):
        x = np.array(x, float)
        if len(x) >= 2:
            out[nm] = summ(x, rng) | {"p_wilcoxon": RTC.wil(x, alt), "alternative": alt}
    return out


def p1_echo(cell, rows, V, rng, F4) -> dict:
    """T4 echo-adjusted model refitted with the rewritten follow-ups as the real items."""
    import continue_from_context as CFC  # noqa: F401
    ct = RTC.corpus_texts([c["arxiv_id"] for c in cell.ctx])
    feats = []
    for n, r in enumerate(rows):
        if r["set"] != "RW-F":
            continue
        name, ab = ct[cell.ctx[r["ctx"]]["arxiv_id"]]
        tk = RTC.toks(name + " " + ab[:1200])
        g = {k: RTC.ngrams(tk, k) for k in (1, 2, 3, 4)}
        feats.append([r["ctx"], float(V[n] @ cell.C[r["ctx"]])] + RTC.overlap_feats(r["title"] + " " + r["abstract"], tk, g))
    Rw = np.array(feats, float)
    I = F4["ideas"]
    si = RTC.idea_steps(cell, F4)
    ctxs = set(Rw[:, 0].astype(int))
    im = np.isin(I[:, 0].astype(int), list(ctxs))
    y = np.concatenate([si[im], Rw[:, 1]])
    isidea = np.concatenate([np.ones(im.sum()), np.zeros(len(Rw))])
    c3 = np.concatenate([I[im, 4], Rw[:, 4]])
    gid = np.concatenate([I[im, 0], Rw[:, 0]]).astype(int)
    f0 = fe_ols(y, isidea, gid, rng=rng, L={"gamma0": [1.0]})
    f1 = fe_ols(y, np.column_stack([isidea, c3]), gid, W=f0["_W"], L={"gamma": [1, 0], "beta_C3": [0, 1]})
    sh = 1 - f1["_draws"][:, 0] / f0["_draws"][:, 0]
    return {"gamma0": strip(f0)["gamma0"], "with_C3": strip(f1), "echo_share": float(1 - f1["gamma"]["est"] / f0["gamma0"]["est"]),
            "echo_share_ci": RTC.ci(sh), "mean_C3_rewrites": float(np.nanmean(Rw[:, 4])), "mean_C3_ideas": float(np.nanmean(I[im, 4]))}


def p2(cell, eng, G, rng) -> dict:
    """G: item geometry of the rewrites. (a) projections; (b) matched-step delta on r2 critique ideas U NB;
    (c) null dose-response on NA U NB and the turn it predicts per arm."""
    M = eng["M"]
    # idea items at rounds 0 and 2 (critique arms), same geometry
    ideas = {0: [], 2: []}
    for i in range(cell.n):
        g = eng["geom"][i]
        if g is None:
            continue
        for r in (0, 2):
            for a in CRIT:
                idx = cell.index.get((a, i, r))
                if idx:
                    X = cell.V[idx]
                    T = tang(X, g["c"])
                    ok = np.isfinite(T).all(1)
                    for x, t in zip(X[ok], T[ok]):
                        ideas[r].append([i, 3 + r // 2, float(x @ g["c"]), float(t @ g["f"]), float(t @ g["p"]), float(t @ g["s"])])
    I0, I2 = np.array(ideas[0]), np.array(ideas[2])
    res = {"projections": {}}
    for nm, X in (("RW-F", G[G[:, 1] == 0]), ("NA", G[G[:, 1] == 1]), ("NB", G[G[:, 1] == 2]), ("ideas_r0", I0), ("ideas_r2", I2)):
        if not len(X):
            continue
        e = {}
        for col, m in ((2, "step"), (3, "on_f"), (4, "on_p"), (5, "on_s")):
            per = np.array([X[X[:, 0] == i, col].mean() for i in np.unique(X[:, 0])])
            e[m] = summ(per, rng)
        res["projections"][nm] = e
    # (b) matched-step comparison
    NB = G[G[:, 1] == 2]
    Z = np.concatenate([I2, NB])
    isidea = np.concatenate([np.ones(len(I2)), np.zeros(len(NB))])
    b, pv = {}, {}
    for col, m in ((5, "s"), (4, "p"), (3, "align")):
        fit = fe_ols(Z[:, col], np.column_stack([1 - Z[:, 2], isidea]), Z[:, 0].astype(int), rng=rng,
                     L={"beta_1_minus_step": [1, 0], "delta_idea": [0, 1]})
        b[m] = strip(fit)
        pv[m] = fit["delta_idea"]["p_cluster_t"] if fit["delta_idea"]["p_cluster_t"] is not None else 1.0
    for m, h in holm(pv).items():
        b[m]["delta_idea"]["p_holm3"] = h
    res["matched_step"] = b
    # (c) null dose-response on NA U NB, predicted turn per arm
    N0 = G[np.isin(G[:, 1], (1, 2))]
    nulls = {}
    for col, m in ((5, "s"), (3, "align")):
        fit = fe_ols(N0[:, col], 1 - N0[:, 2], N0[:, 0].astype(int), rng=rng, L={"beta0": [1.0]})
        W, uniq = fit["_W"], np.unique(N0[:, 0].astype(int))
        b0, b0d = fit["beta0"]["est"], fit["_draws"][:, 0]
        e = {"beta0": strip(fit)["beta0"], "n_obs": fit["n_obs"], "arms": {}}
        for a in CRIT:
            ds, dy = change(M, a, "step"), change(M, a, m)
            # contexts in the null fit's clusters, weighted by the same draw counts
            ok = np.array([np.isfinite(ds[i]) and np.isfinite(dy[i]) for i in uniq])
            dsu, dyu = np.array([ds[i] for i in uniq]), np.array([dy[i] for i in uniq])
            Wk = W[:, ok]
            mds = (Wk @ dsu[ok]) / Wk.sum(1)
            mdy = (Wk @ dyu[ok]) / Wk.sum(1)
            pred, obs = b0 * dsu[ok].mean(), dyu[ok].mean()
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_b = b0d * mds / mdy
            e["arms"][a] = {"predicted_turn": float(pred), "observed_turn": float(obs), "mean_dstep": float(dsu[ok].mean()),
                            "ratio_predicted_over_observed": float(pred / obs) if obs != 0 else None, "ratio_ci": RTC.ci(ratio_b),
                            "predicted_ci": RTC.ci(b0d * mds)}
        nulls[m] = e
    res["null_dose_response"] = nulls
    # reading flags (per cell)
    pa = nulls["align"]["arms"]
    ds_ = b["s"]["delta_idea"]
    lo, hi = ds_["boot_ci"]
    res["reading"] = {
        "null_predicts_at_least_observed_align_majority": sum(pa[a]["predicted_turn"] >= pa[a]["observed_turn"] for a in CRIT) >= 2,
        "delta_s_ci_includes_0": bool(lo <= 0 <= hi), "delta_s_positive_ci_excl_0": bool(lo > 0),
        "predicted_below_half_observed_align_all": all(pa[a]["predicted_turn"] < 0.5 * pa[a]["observed_turn"] for a in CRIT)}
    r = res["reading"]
    r["category"] = ("against: rise in alignment = leaving the paraphrase regime" if r["null_predicts_at_least_observed_align_majority"]
                     and r["delta_s_ci_includes_0"] else "turn exceeds the null; claim stands" if r["delta_s_positive_ci_excl_0"]
                     and r["predicted_below_half_observed_align_all"] else "not separable from a step effect with these controls")
    return res


def p1_reading(p1res: dict, original_gap: float) -> dict:
    g = p1res.get("gap_RW")
    out = {}
    if g:
        lo = g["ci"][0]
        out["followups"] = ("survives genre-and-echo control" if lo > 0 and g["mean"] >= 0.5 * original_gap else
                            f"about {100 * (1 - g['mean'] / original_gap):.0f} % of the short step is genre and echo; the rest remains"
                            if lo > 0 and g["mean"] >= 0.03 else "against: not distinguishable from genre and prompt echo")
    gc = p1res.get("gap_RW_cont")
    if gc:
        out["continuations"] = ("no gap detected against rewritten continuations" if gc["ci"][0] <= 0 <= gc["ci"][1]
                                else "claim stands with the rewritten number")
    out["original_gap"] = original_gap
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyse"])
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--spaces", default=",".join(RTC.SPACES))
    ap.add_argument("--n-ctx", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=RTC.N_BOOT)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--synthetic", action="store_true", help="smoke: fake rewrites from existing vectors")
    a = ap.parse_args()
    RTC.CFG["n_boot"], RTC.CFG["n_ctx"] = a.n_boot, a.n_ctx
    res = {"prereg": "docs/prereg/review_tangent_confounds.md (Part B)", "seed": SEED, "n_boot": a.n_boot,
           "synthetic": a.synthetic, "n_ctx_subset": a.n_ctx}
    p1_primary = {}
    excl_ids = {}
    for space in a.spaces.split(","):
        corp = RTC.Corpus(space)
        for cname in a.cells.split(","):
            study = CELLS[cname]
            cell = RTC.load_cell(cname, space, corp)
            key = f"{cname}/{space}"
            rng = np.random.default_rng(SEED)
            eng = RTC.engine(cell, corp, keep_geom=True)
            if a.synthetic:
                rows, V = synthetic_rewrites(cell, corp, np.random.default_rng(0))
            else:
                rows, V = load_rewrites(study, space)
                rows_V = [(r, v) for r, v in zip(rows, V) if r["ctx"] < cell.n]
                rows, V = [r for r, _ in rows_V], np.array([v for _, v in rows_V])
            cc, cos = content_check(cell, corp, rows, V)
            if space == "qwen3e06":
                excl_ids[cname] = {k for k, v in cos.items() if k.startswith("rwf/") and v < CONTENT_MIN}
            e = {"counts": all_rows_count(study), "content_check": cc}
            e["P1"] = p1(cell, eng, rows, V, rng, set())
            if cname in excl_ids:
                e["P1_sensitivity_excluding_rwf_below_0.80_qwen"] = p1(cell, eng, rows, V, rng, excl_ids[cname]) | {
                    "n_excluded": len(excl_ids[cname])}
            F4 = RTC.t4_texts(cell)
            e["P1_echo_refit"] = p1_echo(cell, rows, V, rng, F4)
            orig_gap = float(np.nanmean(eng["gap_b1"])) if cell.pb is not None else float(np.nanmean(eng["gap_r0"]))
            e["P1_reading"] = p1_reading(e["P1"], orig_gap)
            G = item_geometry(cell, eng, rows, V)
            e["P2"] = p2(cell, eng, G, rng)
            res[key] = e
            if key == "DISC-4o/qwen3e06":
                p1_primary = {k: e["P1"][k]["p_wilcoxon"] for k in ("gap_RW", "gap_RW_cont") if k in e["P1"]}
            RTC.save(res, Path(a.out))
            RTC.log(f"{key} done")
        del corp
    if p1_primary:
        res["P1_primary_holm2"] = holm({k: (v if v is not None else 1.0) for k, v in p1_primary.items()})
    RTC.save(res, Path(a.out))
    RTC.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
