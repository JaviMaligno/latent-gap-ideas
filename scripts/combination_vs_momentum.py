"""Does novel combination beat momentum? (docs/prereg/combination_vs_momentum.md)

Local CPU only, no LLM calls. Every parameter is fixed by the pre-registration.
  uv run python scripts/combination_vs_momentum.py            # full run
  uv run python scripts/combination_vs_momentum.py --smoke    # small sample, code-path check only
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
REPO = Path(__file__).resolve().parents[1]
SEED = 20261030
SMOKE = "--smoke" in sys.argv
OUT = REPO / "docs/prereg" / ("combination_vs_momentum_smoke.json" if SMOKE else "combination_vs_momentum_result.json")
K_CELLS = 32 if SMOKE else 256
N_DRAWS = 20_000 if SMOKE else 200_000
N_BOOT = 50 if SMOKE else 2000
NMF_ITER = 200
V_PRIMARY, V_ROBUST = (40, 30) if SMOKE else (200, 100)
TOPK_FRAC = 0.00875
# Pre-registered reduction (1), applied because the full-scale timing test projected ~1.5 h or more:
# NMF V=200 at ~10.6 s/iteration on the 72k x 20k past TF-IDF (211.9 s for 20 iterations).
DROP_ROBUST = not SMOKE
REDUCTIONS = ["(1) dropped the V=100 robustness arm: NMF timing projected >= 1.5 h total"] if DROP_ROBUST else []
BETA = 0.4
EMBEDDERS = ("qwen3e06", "bgem3")
WINDOWS = {  # name: (last feature year, recent years, outcome years or None = future split)
    "pseudo": (2020, (2019, 2020), (2021, 2022)),
    "test": (2022, (2021, 2022), None),
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- data and atoms
def load():
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["split", "adult", "date", "text"])
    keep = ~meta["adult"].to_numpy()
    idx = np.where(keep)[0]
    if SMOKE:
        rng = np.random.default_rng(SEED)
        sp = meta["split"].to_numpy()[idx]
        idx = np.sort(np.concatenate([rng.choice(idx[sp == "past"], 8000, replace=False),
                                      rng.choice(idx[sp == "future"], 4000, replace=False)]))
    meta = meta.iloc[idx].reset_index(drop=True)
    year = pd.to_datetime(meta["date"]).dt.year.to_numpy()
    past = (meta["split"] == "past").to_numpy()
    return meta, year, past, idx


def atom_sets(texts: pd.Series, past: np.ndarray, V: int, tfidf_cache: dict) -> np.ndarray:
    """(n_papers, 3) sorted atom indices, -1 rows where fewer than 3 non-zero weights."""
    if "X" not in tfidf_cache:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5 if SMOKE else 20, max_df=0.5,
                              max_features=20000, sublinear_tf=True)
        vec.fit(texts[past])
        tfidf_cache["X"] = vec.transform(texts).astype(np.float32)
        tfidf_cache["vocab"] = vec.get_feature_names_out()
    X = tfidf_cache["X"]
    t0 = time.time()
    nmf = NMF(n_components=V, init="nndsvda", max_iter=NMF_ITER, random_state=SEED)
    nmf.fit(X[past])
    W = nmf.transform(X)
    log(f"  NMF V={V}: {time.time() - t0:.0f}s, iters {nmf.n_iter_}")
    top = np.argsort(-W, axis=1)[:, :3]
    ok = np.take_along_axis(W, top, 1)[:, 2] > 0
    top = np.sort(top, axis=1)
    top[~ok] = -1
    vocab = tfidf_cache["vocab"]
    words = [[str(vocab[j]) for j in np.argsort(-nmf.components_[i])[:6]] for i in range(V)]
    return top, words


# ---------------------------------------------------------------- combinatorics
def candidates(V: int, k: int):
    C = np.array(list(combinations(range(V), k)), dtype=np.int32)
    keys = encode(C, V)
    lookup = np.full(V ** k, -1, np.int64)
    lookup[keys] = np.arange(len(C))
    return C, lookup


def encode(C: np.ndarray, V: int) -> np.ndarray:
    key = np.zeros(len(C), np.int64)
    for j in range(C.shape[1]):
        key = key * V + C[:, j]
    return key


def paper_combos(top: np.ndarray, k: int) -> list[np.ndarray]:
    """For each paper, its k-subsets of the 3 atoms (as (n, k) arrays per subset position)."""
    return [top[:, list(s)] for s in combinations(range(3), k)]


def counts_comb(top: np.ndarray, mask: np.ndarray, k: int, V: int, lookup: np.ndarray, N: int) -> np.ndarray:
    out = np.zeros(N, np.int64)
    for sub in paper_combos(top[mask], k):
        out += np.bincount(lookup[encode(sub, V)], minlength=N)
    return out


def pair_matrix(top: np.ndarray, mask: np.ndarray, V: int) -> np.ndarray:
    M = np.zeros((V, V), np.int64)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        np.add.at(M, (top[mask, a], top[mask, b]), 1)
    return M + M.T


# ---------------------------------------------------------------- embedding hole
def fit_cells(E: np.ndarray, rng: np.random.Generator):
    km = KMeans(n_clusters=K_CELLS, random_state=SEED, n_init=1).fit(E)
    past_n = np.bincount(km.labels_, minlength=K_CELLS)
    mu = E.mean(0)
    w, U = np.linalg.eigh(np.cov(E.T, bias=True))
    sd = np.sqrt(np.clip(w, 0, None)).astype(np.float32)
    draw_n = np.zeros(K_CELLS, np.int64)
    step = 20_000
    for _ in range(0, N_DRAWS, step):
        Z = rng.standard_normal((step, E.shape[1])).astype(np.float32) * sd
        P = mu + Z @ U.T.astype(np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        draw_n += np.bincount(km.predict(P), minlength=K_CELLS)
    hole = (draw_n / draw_n.sum()) / (past_n / len(E))
    return km, hole


def hole_feature(E, top, feat_mask, C, km, hole, V) -> np.ndarray:
    ok = feat_mask & (top[:, 0] >= 0)
    rows = np.repeat(np.where(ok)[0], 3)
    M = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, top[ok].ravel())), shape=(len(E), V))
    cent = np.asarray(M.T @ E)
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
    out = np.empty(len(C), np.float32)
    for i in range(0, len(C), 20_000):
        P = cent[C[i:i + 20_000]].sum(1)
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        out[i:i + 20_000] = hole[km.predict(P)]
    return out


# ---------------------------------------------------------------- features per window
def window_features(top, year, past, C, lookup, V, k, window):
    last, (r0, r1), outc = WINDOWS[window]
    valid = top[:, 0] >= 0
    feat = past & (year <= last) & valid
    rec = feat & (year >= r0) & (year <= r1)
    N = len(C)
    n_i = np.bincount(top[feat].ravel(), minlength=V)
    n_rec = np.bincount(top[rec].ravel(), minlength=V)
    Pm, Pm_rec = pair_matrix(top, feat, V), pair_matrix(top, rec, V)
    c = counts_comb(top, feat, k, V, lookup, N)
    c_rec = counts_comb(top, rec, k, V, lookup, N)
    nf = feat.sum()
    subs = list(combinations(range(k), 2))
    f = {
        "POP": np.log1p(n_i[C]).sum(1),
        "m_atom": np.log1p(n_rec[C]).sum(1),
        "m_comb": np.log1p(c_rec).astype(float),
        "COH": np.mean([np.log((Pm[C[:, a], C[:, b]] + 1) * nf / ((n_i[C[:, a]] + 1) * (n_i[C[:, b]] + 1)))
                        for a, b in subs], axis=0),
        "RAR": -np.log1p(c).astype(float),
    }
    if k == 3:
        f["m_pair"] = np.sum([np.log1p(Pm_rec[C[:, a], C[:, b]]) for a, b in subs], axis=0)
    if outc is None:
        out_mask = (~past) & valid
    else:
        out_mask = past & valid & (year >= outc[0]) & (year <= outc[1])
    out_ids = np.concatenate([lookup[encode(s, V)] for s in paper_combos(top[out_mask], k)]) if k < 3 else \
        lookup[encode(top[out_mask], V)]
    # for pairs each paper contributes its 3 pairs; out_ids_paper keeps paper membership for the bootstrap
    n_out = int(out_mask.sum())
    paper_of = np.tile(np.arange(n_out), len(out_ids) // max(n_out, 1)) if n_out else np.zeros(0, int)
    return f, feat, out_ids, paper_of, n_out, c


def zs(a):
    s = a.std()
    return (a - a.mean()) / s if s > 0 else np.zeros_like(a, dtype=float)


MOM_KEYS = {3: ["m_atom", "m_pair", "m_comb"], 2: ["m_atom", "m_comb"]}


# ---------------------------------------------------------------- evaluation
def ranks_of(score: np.ndarray, perm: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """1..n ascending ranks (ties broken by a fixed permutation); 0 outside the subset."""
    r = np.zeros(len(score), np.int64)
    idx = np.arange(len(score)) if subset is None else np.where(subset)[0]
    order = idx[np.lexsort((perm[idx], score[idx]))]
    r[order] = np.arange(1, len(idx) + 1)
    return r


def metrics(pos: np.ndarray, rank_all: np.ndarray, rank_nov: np.ndarray, novel: np.ndarray, N: int, Nn: int) -> dict:
    K, Kn = round(TOPK_FRAC * N), round(TOPK_FRAC * Nn)
    pn = pos[novel[pos]]
    npos, nn = len(pos), len(pn)
    out = {}
    out["enr_all"] = (rank_all[pos] > N - K).sum() / (K * npos / N)
    out["auc_all"] = (rank_all[pos].sum() - npos * (npos + 1) / 2) / (npos * (N - npos))
    out["enr_novel_full"] = (rank_all[pn] > N - K).sum() / (K * nn / N) if nn else np.nan
    out["enr_novel_within"] = (rank_nov[pn] > Nn - Kn).sum() / (Kn * nn / Nn) if nn else np.nan
    out["auc_novel"] = (rank_nov[pn].sum() - nn * (nn + 1) / 2) / (nn * (Nn - nn)) if nn else np.nan
    return out


def analyse_arm(V, k, top, year, past, embs_hole, rng):
    t0 = time.time()
    C, lookup = candidates(V, k)
    N = len(C)
    W = {w: window_features(top, year, past, C, lookup, V, k, w) for w in WINDOWS}
    for w in WINDOWS:
        for e in EMBEDDERS:
            W[w][0][f"HOLE_{e}"] = embs_hole[(V, k, w, e)]
    fp, _, ids_p, _, _, _ = W["pseudo"]
    ft, _, ids_t, paper_t, n_out, c_test = W["test"]
    y_p = np.zeros(N, int)
    y_p[ids_p] = 1
    mk = MOM_KEYS[k]
    sets = {"FIT_M": mk, "FIT_MC": mk + ["COH", "RAR"]}
    for e in EMBEDDERS:
        sets[f"FIT_MCH_{e}"] = mk + ["COH", "RAR", f"HOLE_{e}"]
    scores = {
        "POP": ft["POP"], "M": sum(zs(ft[m]) for m in mk), "COH": ft["COH"],
        "NC_0.4": zs(ft["COH"]) + BETA * zs(ft["RAR"]), "RAR": ft["RAR"],
    }
    for e in EMBEDDERS:
        scores[f"HOLE_{e}"] = ft[f"HOLE_{e}"]
    coefs = {}
    for name, keys in sets.items():
        if y_p.min() == y_p.max():  # degenerate arm (every candidate positive or none): no fit possible
            scores[name] = np.zeros(N)
            coefs[name] = "skipped: one class in the pseudo window"
            continue
        Xp = np.column_stack([fp[q] for q in keys])
        sc = StandardScaler().fit(Xp)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xp), y_p)
        scores[name] = lr.decision_function(sc.transform(np.column_stack([ft[q] for q in keys])))
        coefs[name] = dict(zip(keys, lr.coef_[0].round(4).tolist()))
    novel = c_test == 0
    Nn = int(novel.sum())
    perm = np.random.default_rng(SEED).permutation(N)
    R = {s: (ranks_of(v, perm), ranks_of(v, perm, novel)) for s, v in scores.items()}
    pos = np.unique(ids_t)
    point = {s: metrics(pos, ra, rn, novel, N, Nn) for s, (ra, rn) in R.items()}
    boots = {s: {m: [] for m in point[s]} for s in scores}
    n_sub = len(ids_t) // max(n_out, 1)
    ids_mat = ids_t.reshape(n_sub, n_out) if n_out else ids_t.reshape(0, 0)
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_out, n_out)
        pb = np.unique(ids_mat[:, pick].ravel())
        for s, (ra, rn) in R.items():
            for m, v in metrics(pb, ra, rn, novel, N, Nn).items():
                boots[s][m].append(v)
    boots = {s: {m: np.array(v) for m, v in d.items()} for s, d in boots.items()}
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]

    def diff(s1, s2, m):
        d = boots[s1][m] - boots[s2][m]
        return {"diff": float(point[s1][m] - point[s2][m]), "ci": ci(d),
                "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}

    res = {
        "V": V, "k": k, "N_candidates": N, "N_novel_candidates": Nn, "K": round(TOPK_FRAC * N),
        "n_future_papers": n_out, "n_targets": int(len(pos)), "n_novel_targets": int(novel[pos].sum()),
        "n_pseudo_positives": int(y_p.sum()),
        "scores": {s: {m: {"value": float(v), "ci": ci(boots[s][m])} for m, v in point[s].items()} for s in scores},
        "coefs": coefs,
        "diffs": {
            "M_minus_NC0_auc_all": diff("M", "COH", "auc_all"),
            "M_minus_NC04_auc_all": diff("M", "NC_0.4", "auc_all"),
            "M_minus_NC0_enr_all": diff("M", "COH", "enr_all"),
            "M_minus_NC04_enr_all": diff("M", "NC_0.4", "enr_all"),
            "NC04_minus_NC0_enr_novel_full": diff("NC_0.4", "COH", "enr_novel_full"),
            "NC04_minus_M_enr_novel_full": diff("NC_0.4", "M", "enr_novel_full"),
            "M_minus_COH_auc_novel": diff("M", "COH", "auc_novel"),
            "FITMC_minus_FITM_auc_all": diff("FIT_MC", "FIT_M", "auc_all"),
            "FITMC_minus_FITM_auc_novel": diff("FIT_MC", "FIT_M", "auc_novel"),
            **{f"FITMCH_{e}_minus_FITMC_auc_all": diff(f"FIT_MCH_{e}", "FIT_MC", "auc_all") for e in EMBEDDERS},
        },
    }
    log(f"  arm V={V} k={k}: {time.time() - t0:.0f}s")
    return res


def decide(r: dict) -> dict:
    s, d = r["scores"], r["diffs"]
    v = lambda sc, m: s[sc][m]["value"]
    dec = {
        "H1": v("COH", "enr_all") > 5 and s["COH"]["enr_all"]["ci"][0] > 1,
        "H2": all(v("M", m) >= v(nc, m) for nc in ("COH", "NC_0.4") for m in ("enr_all", "auc_all")),
        "H3a": v("NC_0.4", "enr_novel_full") > v("COH", "enr_novel_full"),
        "H3b": v("M", "auc_novel") >= v("COH", "auc_novel"),
        "H4": d["FITMC_minus_FITM_auc_all"]["diff"] < 0.02 and d["FITMC_minus_FITM_auc_novel"]["diff"] < 0.02,
        "H5_qwen3e06": d["FITMCH_qwen3e06_minus_FITMC_auc_all"]["diff"] < 0.01,
        "H5_bgem3": d["FITMCH_bgem3_minus_FITMC_auc_all"]["diff"] < 0.01,
    }
    dec["H5_replicated_decision"] = dec["H5_qwen3e06"] == dec["H5_bgem3"]
    ps = {"H2": d["M_minus_NC04_auc_all"]["p"], "H3b": d["M_minus_COH_auc_novel"]["p"],
          "H4": d["FITMC_minus_FITM_auc_all"]["p"], "H5": d["FITMCH_qwen3e06_minus_FITMC_auc_all"]["p"]}
    keys = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for i, key in enumerate(keys):
        run = max(run, min(1.0, ps[key] * (len(keys) - i)))
        holm[key] = run
    return {"predictions_hold": dec, "p_raw": ps, "p_holm": holm}


def main() -> None:
    t_all = time.time()
    rng = np.random.default_rng(SEED)
    meta, year, past, idx = load()
    log(f"papers {len(meta)} (past {past.sum()}, future {(~past).sum()}), smoke={SMOKE}")
    cache: dict = {}
    tops, words = {}, {}
    for V in (V_PRIMARY,) if DROP_ROBUST else (V_PRIMARY, V_ROBUST):
        tops[V], words[V] = atom_sets(meta["text"], past, V, cache)
        log(f"  V={V}: papers without a triple {(tops[V][:, 0] < 0).sum()}")
    del cache
    arms = [(V_PRIMARY, 3), (V_PRIMARY, 2)] + ([] if DROP_ROBUST else [(V_ROBUST, 3), (V_ROBUST, 2)])
    hole = {}
    cands = {(V, k): candidates(V, k)[0] for V, k in arms}
    for e in EMBEDDERS:
        E = np.load(DATA / f"prepared/arxiv/emb_{e}.npy", mmap_mode="r")[idx].astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        for w, (last, _, _) in WINDOWS.items():
            feat = past & (year <= last)
            t0 = time.time()
            km, h = fit_cells(E[feat], np.random.default_rng(SEED))
            log(f"  cells {e}/{w}: {time.time() - t0:.0f}s")
            for V, k in arms:
                hole[(V, k, w, e)] = hole_feature(E, tops[V], feat, cands[(V, k)], km, h, V)
        del E
    results = {"smoke": SMOKE, "n_boot": N_BOOT, "reductions_applied": REDUCTIONS, "atoms_top_words": {
        str(V): words[V] for V in words}, "arms": {}}
    for V, k in arms:
        key = f"V{V}_k{k}"
        results["arms"][key] = analyse_arm(V, k, tops[V], year, past, hole, rng)
        results["arms"][key]["decisions"] = decide(results["arms"][key])
        OUT.write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    results["runtime_s"] = time.time() - t_all
    OUT.write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"done in {time.time() - t_all:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
