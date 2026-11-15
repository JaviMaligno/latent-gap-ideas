"""Positive control for steering and a fixed-target oracle (docs/prereg/review_posctrl.md).

  uv run python scripts/review_posctrl.py holdout                 # Block A hold-out, written once, BEFORE generation
  (generation, paid: scripts/iterative_critique.py with LG_ITER_ARMS=HINT / ORACLE-C, commands in the prereg)
  OMP_NUM_THREADS=4 uv run python scripts/review_posctrl.py embed-qwen
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow python scripts/review_posctrl.py embed-bge
  uv run python scripts/review_posctrl.py analyse                 # Block A, B1 if present, M1(b) gap fraction
  uv run python scripts/review_posctrl.py oracle-switch           # Block B0 (free)
  uv run python scripts/review_posctrl.py olmo-inputs             # Block C: hold-out + Kaggle dataset
  uv run python scripts/review_posctrl.py olmo-analyse            # Block C, after the kernel output is downloaded

Smoke options: --limit N (first N contexts), --boot B (resamples instead of 20,000), --raw DIR (where the hold-out,
posctrl ideas and their vectors are read/written instead of runs/critique-replication), --out FILE (JSON).
Output: docs/prereg/review_posctrl.json (sections holdout, qwen3e06, bgem3, oracle_switch, gap_fraction, olmo).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/critique-replication"
OLMO_RUN = DATA / "runs/olmo-direction"
OUT = ROOT / "docs/prereg/review_posctrl.json"
SEED, N_BOOT, BOUND = 20261114, 20000, 0.015
SPACES = ("qwen3e06", "bgem3")
OLD = ("GEO", "LLM", "ORACLE")
NEW = ("HINT", "ORACLE-C")
MEAS = ("step", "s", "p", "align", "fit", "hinted", "cos_h", "cen", "s_orig")

A = argparse.Namespace(limit=None, boot=N_BOOT, raw=RUN, out=OUT)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def unit(v):
    return v / np.linalg.norm(v)


def tang(X, c):
    T = X - np.outer(X @ c, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return T / np.linalg.norm(T, axis=-1, keepdims=True)


def s_of(f, p):
    return unit(f - (f @ p) * p)


def save(key, value):
    res = json.loads(A.out.read_text()) if A.out.exists() else {}
    res[key] = value
    A.out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------- data

def ctx_rows():
    import strengthen_critique_replication as SCR
    return SCR.ctx_rows()[: A.limit or None]


def holdout_path():
    return Path(A.raw) / "posctrl_holdout.json"


def holdout() -> None:
    """h_i = rng.integers(n_i), one draw per context in file order, default_rng(20261114). Written once."""
    import strengthen_critique_replication as SCR
    ctx = SCR.ctx_rows()  # always all 150: the draws are consumed in file order
    out = holdout_path()
    assert not out.exists(), f"{out} already written (the hold-out is written once)"
    rng = np.random.default_rng(SEED)
    recs = []
    for i, c in enumerate(ctx):
        h = int(rng.integers(len(c["followups"])))
        f = c["followups"][h]
        recs.append({"ctx": i, "h_index": h, "arxiv_id": f["arxiv_id"], "title": f["title"],
                     "ctx_arxiv_id": c["arxiv_id"], "n_followups": len(c["followups"])})
    Path(A.raw).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs, indent=1))
    print(f"hold-out written to {out}: {len(recs)} contexts; min n_i - 1 = {min(r['n_followups'] for r in recs) - 1}")


def load_holdout(ctx):
    hold = json.loads(holdout_path().read_text())
    for i, c in enumerate(ctx):
        h = hold[i]
        assert h["ctx"] == i and h["ctx_arxiv_id"] == c["arxiv_id"] and c["followups"][h["h_index"]]["arxiv_id"] == h["arxiv_id"]
    return hold[:len(ctx)]


def posctrl_rows():
    """New rows (HINT, ORACLE-C) with an abstract, sorted (arm, ctx, k, round): the order of posctrl_emb_*.npy."""
    p = Path(A.raw) / "iter_ideas_posctrl.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return sorted((r for r in rows if r["abstract"] and r["arm"] in NEW), key=lambda r: (r["arm"], r["ctx"], r["k"], r["round"]))


def row_key(r):
    return f"{r['arm']}/{r['ctx']}/{r['k']}/{r['round']}"


def posctrl_texts(rows):
    from latent_gap.arxiv import item_text
    return [item_text(r["title"], r["abstract"]) for r in rows]


def embed_qwen() -> None:
    """Settings of strengthen_critique_replication.embed_qwen (local ONNX Qwen3-Embedding-0.6B)."""
    from latent_gap.embedder import LocalQwen
    rows = posctrl_rows()
    V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4"))).embed(posctrl_texts(rows))
    np.save(Path(A.raw) / "posctrl_emb_qwen3e06.npy", V.astype(np.float32))
    (Path(A.raw) / "posctrl_emb_qwen3e06.keys.json").write_text(json.dumps([row_key(r) for r in rows]))
    print("qwen posctrl ideas", V.shape, flush=True)


def embed_bge() -> None:
    """Settings of strengthen_critique_replication.embed_bge (sentence-transformers, max_seq_length 512, normalised, CPU)."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    m = SentenceTransformer("BAAI/bge-m3", device=os.environ.get("LG_BGE_DEVICE", "cpu"))
    m.max_seq_length = 512
    rows = posctrl_rows()
    T = posctrl_texts(rows)
    parts = []
    for b in range(0, len(T), 64):
        parts.append(m.encode(T[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        print(f"bge posctrl {min(b + 64, len(T))}/{len(T)}", flush=True)
    np.save(Path(A.raw) / "posctrl_emb_bgem3.npy", np.concatenate(parts).astype(np.float32))
    (Path(A.raw) / "posctrl_emb_bgem3.keys.json").write_text(json.dumps([row_key(r) for r in rows]))


def corpus_meta():
    import pandas as pd
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    return meta, keep


def future_pool(space, meta, keep):
    """Unit vectors of the future split (adult rows dropped), read through a memmap: one float32 copy only."""
    fut = (meta["split"] == "future").to_numpy()
    rows = np.flatnonzero(keep)[fut]
    E = np.load(DATA / f"prepared/arxiv/emb_{space}.npy", mmap_mode="r")
    X = np.asarray(E[rows], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, meta.loc[fut, "id"].to_numpy()


def citing_sets():
    out = {}
    for l in (OLMO_RUN / "citing_all.jsonl").read_text().splitlines():
        r = json.loads(l)
        out.setdefault(r["arxiv_id"], set()).update(r.get("citing_arxiv_all") or [])
    return out


def replication_vectors(space, it, nu, ctx):
    import strengthen_critique_replication as SCR
    full = SCR.ctx_rows()
    starts = np.cumsum([0] + [len(c["followups"]) for c in full])
    if space == "qwen3e06":
        z = np.load(RUN / "emb.npz")
        C, Fall, V = norm(z["C"]), norm(z["F"]), norm(np.load(RUN / "ideas_emb_qwen3e06.npy"))
    else:
        z = np.load(RUN / "emb_bgem3.npz")
        C, Fall, V = norm(z["ctx"]), norm(z["fup"]), norm(z["ideas"])
    assert len(V) == len(it) + len(nu) and len(Fall) == starts[-1]
    F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]
    return C[:len(ctx)], F, V[:len(it)], V[len(it):]


def posctrl_vectors(space, rows):
    p = Path(A.raw) / f"posctrl_emb_{space}.npy"
    if not rows or not p.exists():
        return None
    keys = json.loads((Path(A.raw) / f"posctrl_emb_{space}.keys.json").read_text())
    assert keys == [row_key(r) for r in rows], f"posctrl_emb_{space}.npy does not match iter_ideas_posctrl.jsonl: re-embed"
    return norm(np.load(p).astype(np.float32))


# ---------------------------------------------------------------- measures

def index_rows(sources, arms):
    index = {}
    for a in arms:
        rows, _ = sources[a]
        for n, x in enumerate(rows):
            if x["arm"] == a and x.get("abstract"):
                index.setdefault((a, x["ctx"], x["round"]), []).append(n)
    return index


def per_context(ctx, C, F, sources, arms, pool, pool_ids, hold, pool_exclude=None):
    """Block A construction (held-out direction) plus the published all-follow-up s for the leaky/published readings.
    M[a][m]: (n_ctx, 3); K: per-context constants."""
    M = {a: {m: np.full((len(ctx), 3), np.nan) for m in MEAS} for a in arms}
    K = {k: np.full(len(ctx), np.nan) for k in ("f_dot_p", "th_s", "th_s_orig")}
    index = index_rows(sources, arms)
    for i in range(len(ctx)):
        c, Fi = C[i], F[i]
        h = hold[i]["h_index"]
        Fm = np.delete(Fi, h, axis=0)
        excl = {f["arxiv_id"] for f in ctx[i]["followups"]} | (pool_exclude(i) if pool_exclude else set())
        sims = pool @ c
        sims[np.isin(pool_ids, list(excl))] = -np.inf
        order = np.argsort(-sims)
        f = unit(tang(Fm, c).mean(0))
        p = unit(tang(pool[order[:len(Fm)]], c).mean(0))
        s = s_of(f, p)
        s_orig = s_of(unit(tang(Fi, c).mean(0)), unit(tang(pool[order[:len(Fi)]], c).mean(0)))
        th = tang(Fi[h:h + 1], c)[0]
        cen = unit(Fm.mean(0))
        K["f_dot_p"][i], K["th_s"][i], K["th_s_orig"][i] = f @ p, th @ s, th @ s_orig
        for a in arms:
            _, V = sources[a]
            for r in range(3):
                idx = index.get((a, i, r))
                if not idx:
                    continue
                X = V[idx]
                T = tang(X, c)
                ok = np.isfinite(T).all(1)
                M[a]["step"][i, r] = (X @ c).mean()
                M[a]["fit"][i, r] = (X @ Fm.T).max(1).mean()
                M[a]["cos_h"][i, r] = (X @ Fi[h]).mean()
                M[a]["cen"][i, r] = (X @ cen).mean()
                if ok.any():
                    T = T[ok]
                    for m, v in (("s", s), ("p", p), ("align", f), ("hinted", th), ("s_orig", s_orig)):
                        M[a][m][i, r] = (T @ v).mean()
    return M, K


def change(M, a, m):
    return M[a][m][:, 0] - M[a][m][:, 2] if m == "step" else M[a][m][:, 2] - M[a][m][:, 0]


def dcon(M, a, b, m):
    return change(M, a, m) - change(M, b, m)


def est(d, rng, alt="two-sided"):
    """Mean, 95 % paired t interval, 95 % percentile bootstrap (A.boot resamples over contexts), Wilcoxon p."""
    n_all = len(d)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "n_dropped": n_all - n, "mean": float(d.mean()) if n else None}
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    tq = stats.t.ppf(0.975, n - 1)
    b = d[rng.integers(0, n, (A.boot, n))].mean(1)
    p = float(stats.wilcoxon(d, alternative=alt).pvalue) if np.any(d != 0) else 1.0
    return {"mean": m, "t_ci95": [m - tq * se, m + tq * se], "boot_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "p_wilcoxon": p, "n": n, "n_dropped": n_all - n, "sd": float(d.std(ddof=1))}


def ratio(num, den, rng):
    """Ratio of means with a joint percentile bootstrap (same index draws for numerator and denominator)."""
    ok = ~np.isnan(num) & ~np.isnan(den)
    x, y = num[ok], den[ok]
    n = len(x)
    if n < 2 or y.mean() == 0:
        return {"n": n, "ratio": None}
    idx = rng.integers(0, n, (A.boot, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = x[idx].mean(1) / y[idx].mean(1)
    return {"ratio": float(x.mean() / y.mean()), "boot_ci95": [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))],
            "num_mean": float(x.mean()), "den_mean": float(y.mean()), "n": n}


def holm(p: dict) -> dict:
    keys, out, run = sorted(p, key=p.get), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        out[k] = run
    return out


def detected(e):
    lo, hi = e["t_ci95"]
    if e["p_holm"] < 0.05 and lo > 0:
        return "positive"
    if e["p_holm"] < 0.05 and hi < 0:
        return "negative"
    return None


def words(t):
    return set(re.findall(r"\w+", t.lower()))


# ---------------------------------------------------------------- Block A / B1 cell

def block_a_cell(ctx, C, F, sources, arms, pool, pool_ids, hold, rows_new, citing, olmo_ids, rng):
    M, K = per_context(ctx, C, F, sources, arms, pool, pool_ids, hold)
    cell = {"n_contexts": len(ctx)}
    if "HINT" not in arms:
        cell["skipped"] = "no HINT rows / vectors"
        return cell, M
    # fixed order of rng consumption: MC, PC1-PC3, ratios, B1, secondaries
    cell["MC:HINT-NEUTRAL:hinted"] = est(dcon(M, "HINT", "NEUTRAL", "hinted"), rng)
    mc = cell["MC:HINT-NEUTRAL:hinted"]
    pc = {"PC1:HINT-NEUTRAL:s": est(dcon(M, "HINT", "NEUTRAL", "s"), rng),
          "PC2:HINT-GEO:s": est(dcon(M, "HINT", "GEO", "s"), rng),
          "PC3:HINT-ORACLE:s": est(dcon(M, "HINT", "ORACLE", "s"), rng)}
    for k, v in holm({k: v["p_wilcoxon"] for k, v in pc.items()}).items():
        pc[k]["p_holm"] = v
        pc[k]["detected"] = detected(pc[k])
    cell["confirmatory_PC"] = pc
    pc1 = pc["PC1:HINT-NEUTRAL:s"]
    ratios = {}
    if pc1["detected"] == "positive":
        dh = dcon(M, "HINT", "NEUTRAL", "s")
        for a in OLD:
            ratios[f"R_{a}"] = ratio(dcon(M, a, "NEUTRAL", "s"), dh, rng)
        for a in OLD:
            ratios[f"Q_{a}"] = ratio(change(M, a, "s"), change(M, "HINT", "s"), rng)
        for k in [k for k in ratios if k.startswith("R_")]:
            lo, hi = ratios[k]["boot_ci95"]
            ratios[k]["reading"] = ("barely supported (upper bound < 0.5)" if hi < 0.5 else
                                    "counts against 'barely' (lower bound >= 0.5)" if lo >= 0.5 else "straddles 0.5: report the number")
    else:
        ratios["note"] = "not estimated: PC1 not detected positive"
    cell["yardstick"] = ratios
    # reading rule (fixed order)
    if not (mc.get("t_ci95") and mc["t_ci95"][0] > 0):
        outcome = "1: manipulation check failed, positive control uninformative"
    elif pc1["detected"] == "positive":
        outcome = "4: PC1 detected positive, read R_A"
    elif pc1["detected"] == "negative":
        outcome = "5: PC1 detected negative"
    elif -BOUND < pc1["t_ci95"][0] and pc1["t_ci95"][1] < BOUND:
        outcome = "2: MC passes, PC1 not detected and inside ±0.015 (counts against the current claim)"
    else:
        outcome = "3: MC passes, PC1 not detected, interval crosses ±0.015 (insufficient)"
    cell["reading"] = outcome
    if pc["PC2:HINT-GEO:s"]["detected"] is None:
        cell["reading_note"] = "PC2 not detected: the text may not claim content steers more than distance feedback"
    # Block B1
    if "ORACLE-C" in arms:
        oc = {"OC1:ORACLE-C-NEUTRAL:cen": est(dcon(M, "ORACLE-C", "NEUTRAL", "cen"), rng),
              "OC2:ORACLE-C-NEUTRAL:s": est(dcon(M, "ORACLE-C", "NEUTRAL", "s"), rng)}
        for k, v in holm({k: v["p_wilcoxon"] for k, v in oc.items()}).items():
            oc[k]["p_holm"] = v
            oc[k]["detected"] = detected(oc[k])
        o1, o2 = oc["OC1:ORACLE-C-NEUTRAL:cen"]["detected"], oc["OC2:ORACLE-C-NEUTRAL:s"]["detected"]
        oc["reading"] = ("fixed target steers where the max did not" if o1 == "positive" and o2 == "positive" else
                         "score optimisable, move not follow-up-specific (read Δp)" if o1 == "positive" else
                         "fixed-target scalar fails like the max")
        if pc1["detected"] == "positive":
            oc["R_ORACLE-C"] = ratio(dcon(M, "ORACLE-C", "NEUTRAL", "s"), dcon(M, "HINT", "NEUTRAL", "s"), rng)
        oc["descriptive:HINT-ORACLE-C:s"] = est(dcon(M, "HINT", "ORACLE-C", "s"), rng)
        oc["descriptive:ORACLE-C-NEUTRAL:p"] = est(dcon(M, "ORACLE-C", "NEUTRAL", "p"), rng)
        cell["B1_oracle_c"] = oc
    # secondaries (descriptive)
    sec = {"per_arm_change": {}}
    for a in arms:
        sec["per_arm_change"][a] = {m: est(change(M, a, m), rng) for m in ("s", "p", "align", "step", "fit", "hinted", "cos_h", "cen", "s_orig")}
        sec["per_arm_change"][a]["round_means"] = {f"round{r}": {m: float(np.nanmean(M[a][m][:, r])) for m in MEAS} for r in range(3)}
    a_, b_ = K["f_dot_p"], np.sqrt(np.clip(1 - K["f_dot_p"] ** 2, 0, None))
    cp, cs, da = a_ * change(M, "HINT", "p"), b_ * change(M, "HINT", "s"), change(M, "HINT", "align")
    sec["HINT_align_split"] = {"align_change": float(np.nanmean(da)), "contrib_p_weighted": float(np.nanmean(cp)),
                               "contrib_s_weighted": float(np.nanmean(cs)), "weight_a_fp_mean": float(np.nanmean(a_)),
                               "weight_b_mean": float(np.nanmean(b_)),
                               "share_p": float(np.nanmean(cp) / np.nanmean(da)), "share_s": float(np.nanmean(cs) / np.nanmean(da))}
    sec["HINT-NEUTRAL:p"] = est(dcon(M, "HINT", "NEUTRAL", "p"), rng)
    sec["HINT-NEUTRAL:step"] = est(dcon(M, "HINT", "NEUTRAL", "step"), rng)
    sec["hinted_paper_projection_th_s"] = {"mean": float(np.nanmean(K["th_s"])), "mean_on_published_s_leaky": float(np.nanmean(K["th_s_orig"]))}
    sec["hinted_gap_closed"] = ratio(change(M, "HINT", "s"), K["th_s"] - M["HINT"]["s"][:, 0], rng)
    if citing:
        M2, _ = per_context(ctx, C, F, sources, arms, pool, pool_ids, hold,
                            pool_exclude=lambda i: citing.get(ctx[i]["arxiv_id"], set()))
        sec["variant_pool_excluding_all_citing"] = {
            "PC1:HINT-NEUTRAL:s": est(dcon(M2, "HINT", "NEUTRAL", "s"), rng),
            "PC2:HINT-GEO:s": est(dcon(M2, "HINT", "GEO", "s"), rng),
            "PC3:HINT-ORACLE:s": est(dcon(M2, "HINT", "ORACLE", "s"), rng),
            "s_change": {a: float(np.nanmean(change(M2, a, "s"))) for a in arms}}
    sec["leaky_original_s"] = {"label": "published s contains h_i: leaky for HINT",
                               "HINT-NEUTRAL:s_orig": est(dcon(M, "HINT", "NEUTRAL", "s_orig"), rng),
                               "s_orig_change": {a: float(np.nanmean(change(M, a, "s_orig"))) for a in arms}}
    t2 = [r for r in rows_new if r["arm"] == "HINT" and r["round"] == 2 and r["ctx"] < len(ctx)]
    jac = [len(words(r["title"]) & words(hold[r["ctx"]]["title"])) / max(1, len(words(r["title"]) | words(hold[r["ctx"]]["title"]))) for r in t2]
    sec["title_echo_round2"] = {"share_jaccard_ge_0.5": float(np.mean(np.array(jac) >= 0.5)) if jac else None,
                                "mean_jaccard": float(np.mean(jac)) if jac else None, "n": len(jac)}
    sub = np.array([c["arxiv_id"] in olmo_ids for c in ctx])
    sec["olmo_eligible_overlap"] = {"n": int(sub.sum()), **{
        k: float(np.nanmean(dcon(M, "HINT", b, m)[sub])) if sub.any() else None
        for k, b, m in (("MC", "NEUTRAL", "hinted"), ("PC1", "NEUTRAL", "s"), ("PC2", "GEO", "s"), ("PC3", "ORACLE", "s"))}}
    cell["secondary"] = sec
    return cell, M


# ---------------------------------------------------------------- M1(b): fraction of the gap

def gap_cell(ctx, C, F, sources, arms, pool, pool_ids, rng):
    """Published s (all follow-ups, n pool neighbours); G_c = mean_j t_j·s_{c,-j} with p from n-1 neighbours."""
    n = len(ctx)
    G = np.full(n, np.nan)
    S = {a: np.full((n, 3), np.nan) for a in arms}
    index = index_rows(sources, arms)
    for i in range(n):
        c, Fi = C[i], F[i]
        excl = {f["arxiv_id"] for f in ctx[i]["followups"]}
        sims = pool @ c
        sims[np.isin(pool_ids, list(excl))] = -np.inf
        order = np.argsort(-sims)
        TF = tang(Fi, c)
        k = len(Fi)
        p_n, p_n1 = unit(tang(pool[order[:k]], c).mean(0)), unit(tang(pool[order[:k - 1]], c).mean(0))
        s = s_of(unit(TF.mean(0)), p_n)
        tot = TF.sum(0)
        G[i] = float(np.mean([TF[j] @ s_of(unit(tot - TF[j]), p_n1) for j in range(k)]))
        for a in arms:
            _, V = sources[a]
            for r in range(3):
                idx = index.get((a, i, r))
                if idx:
                    T = tang(V[idx], c)
                    T = T[np.isfinite(T).all(1)]
                    if len(T):
                        S[a][i, r] = (T @ s).mean()
    out = {"G_mean": float(np.nanmean(G))}
    for a in arms:
        gap = G - S[a][:, 0]
        ds = S[a][:, 2] - S[a][:, 0]
        e = {"gap_mean": float(np.nanmean(gap)), "s_round0": float(np.nanmean(S[a][:, 0])),
             "frac_of_gap_change": ratio(ds, gap, rng)}
        if a != "NEUTRAL" and "NEUTRAL" in arms:
            e["frac_of_gap_minus_neutral"] = ratio(ds - (S["NEUTRAL"][:, 2] - S["NEUTRAL"][:, 0]), gap, rng)
        out[a] = e
    return out


def discovery_sample(space):
    """As revision_neutral.analyse: Kaggle vectors for contexts, follow-ups and GEO/LLM/ORACLE; local NEUTRAL vectors."""
    import continuation_reanalysis as R
    import revision_neutral as RV
    ctx, C, F, it, IT = RV.load_gpt4o(R, space)
    nrows = RV.neutral_rows()
    NV = norm(np.load(RV.RUN / f"neutral_emb_{space}.npy").astype(np.float32))
    L = A.limit or len(ctx)
    return ctx[:L], C[:L], F[:L], {a: (it, IT) for a in OLD} | {"NEUTRAL": (nrows, NV)}


def replication_sample(space):
    import strengthen_critique_replication as SCR
    ctx = ctx_rows()
    it, nu = SCR.idea_rows()
    C, F, IT, NV = replication_vectors(space, it, nu, ctx)
    return ctx, C, F, {a: (it, IT) for a in OLD} | {"NEUTRAL": (nu, NV)}


def analyse() -> None:
    meta, keep = corpus_meta()
    citing = citing_sets()
    el = json.loads((OLMO_RUN / "eligible.json").read_text())
    olmo_ids = {c["arxiv_id"] for c in el["contexts"] if c.get("eligible")}
    ctx = ctx_rows()
    hold = load_holdout(ctx)
    rows_new = posctrl_rows()
    save("holdout", {"file": str(holdout_path()), "seed": SEED, "n": len(hold),
                     "mean_n_followups": float(np.mean([h["n_followups"] for h in hold]))})
    gap = {}
    for space in SPACES:
        pool, pool_ids = future_pool(space, meta, keep)
        ctx_, C, F, sources = replication_sample(space)
        PV = posctrl_vectors(space, rows_new)
        arms = OLD + ("NEUTRAL",)
        if PV is not None:
            for a in NEW:
                if any(r["arm"] == a for r in rows_new):
                    sources[a] = (rows_new, PV)
                    arms += (a,)
        rng = np.random.default_rng(SEED)
        cell, _ = block_a_cell(ctx_, C, F, sources, arms, pool, pool_ids, hold, rows_new, citing, olmo_ids, rng)
        save(space, cell)
        print(space, json.dumps({k: cell[k] for k in ("reading",) if k in cell}),
              json.dumps({k: [round(v.get("mean") or 0, 4), round(v.get("p_holm", v.get("p_wilcoxon", 1)), 4)]
                          for k, v in cell.get("confirmatory_PC", {}).items()}), flush=True)
        # M1(b) on both gpt-4o samples, published s
        rng = np.random.default_rng(SEED)
        g = {"replication": gap_cell(ctx_, C, F, {a: sources[a] for a in OLD + ("NEUTRAL",)}, OLD + ("NEUTRAL",), pool, pool_ids, rng)}
        dctx, dC, dF, dsrc = discovery_sample(space)
        g["discovery"] = gap_cell(dctx, dC, dF, dsrc, OLD + ("NEUTRAL",), pool, pool_ids, rng)
        gap[space] = g
        print(space, "gap fraction", json.dumps({s: {a: round(v[a]["frac_of_gap_change"]["ratio"] or 0, 4) for a in OLD + ("NEUTRAL",)}
                                                 for s, v in g.items()}), flush=True)
        del pool
    save("gap_fraction", gap)


# ---------------------------------------------------------------- Block B0: argmax switching

def switch_cell(ctx, C, F, sources, rng):
    arms = ("ORACLE", "GEO", "NEUTRAL")
    out, per = {}, {}
    for a in arms:
        rows, V = sources[a]
        chains = {}
        for n, x in enumerate(rows):
            if x["arm"] == a and x.get("abstract") and x["ctx"] < len(ctx):
                chains.setdefault((x["ctx"], x["k"]), {})[x["round"]] = n
        sw = {0: [], 1: []}
        dcen = {i: [] for i in range(len(ctx))}
        dfix = {i: [] for i in range(len(ctx))}
        for (i, k), ch in chains.items():
            Fi = F[i]
            arg = {r: int(np.argmax(Fi @ V[n])) for r, n in ch.items()}
            for r in (0, 1):
                if r in arg and r + 1 in arg:
                    sw[r].append(arg[r] != arg[r + 1])
            if 0 in ch and 2 in ch:
                cen = unit(Fi.mean(0))
                x0, x2 = V[ch[0]], V[ch[2]]
                dcen[i].append(float(x2 @ cen - x0 @ cen))
                dfix[i].append(float(x2 @ Fi[arg[0]] - x0 @ Fi[arg[0]]))
        allsw = sw[0] + sw[1]
        out[a] = {"switch_rate": float(np.mean(allsw)) if allsw else None, "n_revisions": len(allsw),
                  "switch_rate_0_1": float(np.mean(sw[0])) if sw[0] else None, "switch_rate_1_2": float(np.mean(sw[1])) if sw[1] else None,
                  "mean_n_followups": float(np.mean([len(f) for f in F]))}
        per[a] = {"cen": np.array([np.mean(v) if v else np.nan for v in dcen.values()]),
                  "fix": np.array([np.mean(v) if v else np.nan for v in dfix.values()])}
        out[a]["change_cos_centroid"] = float(np.nanmean(per[a]["cen"]))
        out[a]["change_cos_round0_argmax"] = float(np.nanmean(per[a]["fix"]))
    out["ORACLE-NEUTRAL:cos_centroid"] = est(per["ORACLE"]["cen"] - per["NEUTRAL"]["cen"], rng)
    out["ORACLE-NEUTRAL:cos_round0_argmax"] = est(per["ORACLE"]["fix"] - per["NEUTRAL"]["fix"], rng)
    return out


def oracle_switch() -> None:
    res = {}
    for sample, loader in (("discovery", discovery_sample), ("replication", replication_sample)):
        res[sample] = {}
        for space in SPACES:
            ctx, C, F, sources = loader(space)
            res[sample][space] = switch_cell(ctx, C, F, sources, np.random.default_rng(SEED))
            r = res[sample][space]
            print(sample, space, {a: round(r[a]["switch_rate"], 3) for a in ("ORACLE", "GEO", "NEUTRAL")},
                  "ORACLE-NEUTRAL centroid", round(r["ORACLE-NEUTRAL:cos_centroid"]["mean"], 4),
                  "fixed", round(r["ORACLE-NEUTRAL:cos_round0_argmax"]["mean"], 4), flush=True)
    save("oracle_switch", res)


# ---------------------------------------------------------------- Block C: OLMo (Kaggle)

OLMO_UP = DATA / "kaggle-upload/olmo-posctrl-inputs"
OLMO_OUT = DATA / "kaggle-output/olmo-posctrl"
OLMO_KDIR = ROOT / "kaggle/olmo-posctrl"
OLMO_DATASET = "javieraguilarmartin/latent-gap-olmo-posctrl-inputs"
OLMO_KERNEL = "javieraguilarmartin/latent-gap-olmo-posctrl"


def olmo_saved():
    """Eligible contexts of the OLMo-direction run in eligible_pos order: {pos: (part dir, ctx index in that part)}."""
    loc = {}
    for d in sorted((DATA / "kaggle-output").glob("olmo-direction-w*p*")):
        info = json.loads((d / "run_info.json").read_text())
        for i, (p, role) in enumerate(zip(info["positions"], info["roles"])):
            if role == "eligible":
                loc[p] = (d, i)
    return loc


def olmo_inputs() -> None:
    import continue_from_context as M
    import iterative_critique as IC
    from latent_gap.arxiv import item_text
    import pandas as pd
    # verbatim prompt functions and feedback strings
    src = (OLMO_KDIR / "run.py").read_text()
    import inspect
    for fn, repl in ((M.prompt, {}), (IC.revise_prompt, {"M.prompt(": "prompt("}), (IC.hint_feedback, {})):
        s = inspect.getsource(fn)
        for a, b in repl.items():
            s = s.replace(a, b)
        assert s in src, f"{fn.__name__} in kaggle/olmo-posctrl/run.py is not a verbatim copy"
    el = json.loads((OLMO_RUN / "eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    pos = el["eligible_pos"]
    rows = {}
    for l in (OLMO_RUN / "followups.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["n_followups"] >= 5:
            rows[len(rows)] = r
    loc = olmo_saved()
    assert set(pos) <= set(loc), "some eligible contexts have no saved OLMo-direction output"
    by_id = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"]).set_index("id")
    hp = OLMO_RUN / "posctrl_holdout_olmo.json"
    rng = np.random.default_rng(SEED)
    cache, ctx, hold = {}, [], []
    for i, p in enumerate(pos):
        r = rows[p]
        assert r["arxiv_id"] == recs[p]["arxiv_id"]
        h = int(rng.integers(len(r["followups"])))
        d, j = loc[p]
        if d not in cache:
            cache[d] = [json.loads(l) for l in (d / "iter_olmo.jsonl").read_text().splitlines()]
        r0 = sorted((x for x in cache[d] if x["ctx"] == j and x["arm"] == "GEO" and x["round"] == 0), key=lambda x: x["k"])
        assert [x["k"] for x in r0] == [0, 1, 2] and all(x["pos"] == p for x in r0)
        name, abstract = by_id.loc[r["arxiv_id"], "name"], by_id.loc[r["arxiv_id"], "abstract"]
        f = r["followups"][h]
        hold.append({"ctx": i, "pos": p, "h_index": h, "arxiv_id": f["arxiv_id"], "title": f["title"],
                     "ctx_arxiv_id": r["arxiv_id"], "n_followups": len(r["followups"]),
                     "h_is_consensus": h in recs[p]["consensus"], "k_consensus": len(recs[p]["consensus"])})
        ctx.append({"ctx": i, "pos": p, "arxiv_id": r["arxiv_id"], "ctx_block": M.paper_block(name, abstract),
                    "ctx_text": item_text(name, abstract), "fups": [item_text(x["title"], x["abstract"]) for x in r["followups"]],
                    "hint_title": f["title"], "h_index": h,
                    "round0": [{k: x[k] for k in ("k", "title", "abstract", "step", "fit")} for x in r0],
                    "source": f"{d.name}/ctx{j}"})
    if hp.exists():
        assert json.loads(hp.read_text()) == hold, f"{hp} differs from the seeded draw"
    else:
        hp.write_text(json.dumps(hold, indent=1))
    OLMO_UP.mkdir(parents=True, exist_ok=True)
    (OLMO_UP / "contexts.json").write_text(json.dumps(ctx))
    (OLMO_UP / "templates.json").write_text(json.dumps({"system": M.SYSTEM}))
    info = {"seed": SEED, "positions": pos, "n": len(ctx), "prereg": "docs/prereg/review_posctrl.md Block C",
            "round0_from": "GEO round 0 of olmo-direction-w*p* iter_olmo.jsonl"}
    (OLMO_UP / "run_info.json").write_text(json.dumps(info))
    (OLMO_UP / "dataset-metadata.json").write_text(json.dumps({"title": "latent-gap olmo posctrl inputs", "id": OLMO_DATASET,
                                                               "licenses": [{"name": "other"}]}))
    n_empty = sum(not x["abstract"] for c in ctx for x in c["round0"])
    print(f"{len(ctx)} contexts, {sum(len(c['fups']) for c in ctx)} follow-ups, {n_empty} empty round-0 ideas; "
          f"{len(ctx) * 6} OLMo generations (HINT, rounds 1-2); hold-out {hp}; "
          f"h is a consensus continuation in {sum(h['h_is_consensus'] for h in hold)} contexts")
    print("next:\n"
          f"  kaggle datasets create -p {OLMO_UP}     # later: kaggle datasets version -p {OLMO_UP} -m '...'\n"
          f"  kaggle kernels push -p {OLMO_KDIR}\n"
          f"  kaggle kernels status {OLMO_KERNEL}\n"
          f"  kaggle kernels output {OLMO_KERNEL} -p {OLMO_OUT}")


def olmo_analyse() -> None:
    """Block C: HINT (new kernel) against the saved NEUTRAL, GEO, ORACLE (olmo-direction) on the all-follow-up s_{-h};
    secondary: the continuation direction without h_i for contexts with k >= 3 consensus continuations."""
    import olmo_direction_analysis as OA
    meta, keep = corpus_meta()
    citing = citing_sets()
    el = json.loads((OLMO_RUN / "eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    hold = json.loads((OLMO_RUN / "posctrl_holdout_olmo.json").read_text())[: A.limit or None]
    saved = OA.load_outputs()
    out_dir = Path(os.environ.get("LG_OLMO_POSCTRL_OUT", OLMO_OUT))
    hrows = [json.loads(l) for l in (out_dir / "iter_olmo_posctrl.jsonl").read_text().splitlines()]
    hab = [r for r in hrows if r["abstract"]]
    res = {"n_contexts": len(hold), "n_hint_rows": len(hrows), "n_hint_empty": len(hrows) - len(hab)}
    for space in SPACES:
        pool, pool_ids = future_pool(space, meta, keep)
        HV = norm(np.load(out_dir / f"iter_{space}.npy").astype(np.float32))
        assert len(HV) == len(hab)
        fr = {}  # saved follow-ups (h_index and pool exclusion), indexed as olmo_direction_kaggle.build
        for l in (OLMO_RUN / "followups.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["n_followups"] >= 5:
                fr[len(fr)] = r
        ctx, C, F, rows, Vs = [], [], [], [], []
        for i, h in enumerate(hold):
            o = saved[h["pos"]]
            ctx.append({"arxiv_id": h["ctx_arxiv_id"], "followups": fr[h["pos"]]["followups"]})
            assert fr[h["pos"]]["arxiv_id"] == h["ctx_arxiv_id"] and len(ctx[i]["followups"]) == len(o[space]["F"])
            C.append(o[space]["C"]); F.append(o[space]["F"])
            for n, x in enumerate(o["rows"]):
                rows.append({**x, "ctx": i}); Vs.append(o[space]["V"][n])
        C = np.array(C)
        V = np.array(Vs)
        hidx = [n for n, x in enumerate(hab) if x["ctx"] < len(hold)]
        sources = {a: (rows, V) for a in ("GEO", "LLM", "ORACLE", "NEUTRAL")}
        sources["HINT"] = ([hab[n] for n in hidx], HV[hidx])
        arms = ("GEO", "LLM", "ORACLE", "NEUTRAL", "HINT")
        rng = np.random.default_rng(SEED)
        cell, M = block_a_cell(ctx, C, F, sources, arms, pool, pool_ids, hold, [hab[n] for n in hidx], citing, set(), rng)
        # secondary: continuation direction rebuilt without h_i (k >= 3 contexts only), as olmo_direction_analysis
        cont = {a: [] for a in ("HINT", "GEO", "ORACLE")}
        idxr = index_rows(sources, arms)
        nk = 0
        for i, h in enumerate(hold):
            js = [j for j in recs[h["pos"]]["consensus"] if j != h["h_index"]]
            if len(recs[h["pos"]]["consensus"]) < 3:
                continue
            nk += 1
            c = C[i]
            TF = tang(F[i], c)
            f = unit(TF[js].mean(0))
            excl = citing.get(h["ctx_arxiv_id"], set()) | {h["ctx_arxiv_id"]}
            sims = pool @ c
            sims[np.isin(pool_ids, list(excl))] = -np.inf
            p = unit(tang(pool[np.argsort(-sims)[:len(js)]], c).mean(0))
            s = s_of(f, p)
            val = {}
            for a in ("HINT", "GEO", "ORACLE", "NEUTRAL"):
                _, Va = sources[a]
                v = []
                for r in (0, 2):
                    ix = idxr.get((a, i, r))
                    T = tang(Va[ix], c) if ix else np.zeros((0, len(c)))
                    T = T[np.isfinite(T).all(1)]
                    v.append((T @ s).mean() if len(T) else np.nan)
                val[a] = v[1] - v[0]
            for a in cont:
                cont[a].append(val[a] - val["NEUTRAL"])
        cell["secondary_continuation_direction_k_ge_3"] = {"n": nk, **{f"{a}-NEUTRAL:s_cont": est(np.array(v), rng) for a, v in cont.items()}}
        res[space] = cell
        print(space, cell.get("reading"), flush=True)
        del pool
    save("olmo", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["holdout", "embed-qwen", "embed-bge", "analyse", "oracle-switch", "olmo-inputs", "olmo-analyse"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--raw", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=OUT)
    A = ap.parse_args()
    {"holdout": holdout, "embed-qwen": embed_qwen, "embed-bge": embed_bge, "analyse": analyse,
     "oracle-switch": oracle_switch, "olmo-inputs": olmo_inputs, "olmo-analyse": olmo_analyse}[A.cmd]()
