"""R1 of docs/prereg/review_far_decoding.md: does the far-point result depend on the uncentred cap?

Transformations T0-T4 (raw, centring, all-but-the-top D=10, BERT-whitening k=256, shrunk ZCA), fitted on the
non-adult PAST split of each corpus x embedder. W1 cap geometry, W2 targets rebuilt in each space (REAL', GAP', FAR'
with latent_gap.targets), W3 Table A2 transported into each space. T0 must reproduce Table A2 and the A.3 FAR
ceilings (docs/prereg/far_targets_vs_future.json) to 3 decimals, or the script stops. No generation, no GPU.

    uv run python scripts/review_far_whitening.py                     # full run -> docs/prereg/review_far_whitening.json
    uv run python scripts/review_far_whitening.py --smoke --out /tmp/x.json   # tiny subset, no reproduction stop

The per-item W3 table (read by scripts/review_far_units.py) goes to DATA/runs/review-far-decoding/w3_items.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, t as student_t

from latent_gap import targets as TG

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_far_whitening.json"
ITEMS_OUT = DATA / "runs/review-far-decoding/w3_items.csv"
SEED = 20261114
B = 10_000
DOMS = ("steam", "arxiv", "patents")
EMBS = (("qwen3", "emb_qwen3e06.npy"), ("bge", "emb_bgem3.npy"))
TRANSFORMS = ("T0", "T1", "T2", "T3", "T4")
D_ABTT = 10
K_BERT = 256
EPS_SHRINK = 0.01
N_TARGETS = 50
N_PAIRS = 100_000
N_SLAB_ITEMS = 2_000
N_HUB = 10_000
HUB_K = 10
N_REF = 5_000
VAR_KEEP = 0.90
W3_DOMS = ("steam", "arxiv")
W3_ARMS = ("REAL", "GAP", "FAR")
W3_GENS = ("luna", "sonnet")
W3_CONDS = ("true", "swap", "none")
KS = list(range(1, 21))
ALPHA_ORIG = 0.05 / 24
ALPHA_CORR = 0.05 / 96
TOL = 5e-4  # "to 3 decimals"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_split(dom: str, emb_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Non-adult past and future rows (corpus order), float32, L2-normalised."""
    meta = pd.read_parquet(DATA / f"prepared/{dom}/corpus.parquet", columns=["split", "adult"])
    raw = np.load(DATA / f"prepared/{dom}/{emb_file}", mmap_mode="r")
    keep = ~meta["adult"].to_numpy()
    past = keep & meta["split"].eq("past").to_numpy()
    fut = keep & ~meta["split"].eq("past").to_numpy()
    X = unit(np.asarray(raw[past], dtype=np.float32))
    F = unit(np.asarray(raw[fut], dtype=np.float32))
    return X, F


def second_moments(X: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """mu and the uncentred second moment X^T X / N, accumulated in float64 by chunks."""
    d = X.shape[1]
    S, s = np.zeros((d, d)), np.zeros(d)
    for a in range(0, len(X), chunk):
        c = X[a:a + chunk].astype(np.float64)
        S += c.T @ c
        s += c.sum(0)
    return s / len(X), S / len(X)


class Transform:
    """f(x) = norm((x - mu) M) (M = None means identity); T0 is f(x) = x. direction(t) = norm(t M), the linear part."""

    def __init__(self, name: str, X: np.ndarray):
        self.name = name
        self.mu = None
        self.M = None
        if name == "T0":
            self.dim = X.shape[1]
            return
        mu, M2 = second_moments(X)
        Sigma = M2 - np.outer(mu, mu)  # cov(X, bias=True)
        lam, U = np.linalg.eigh(Sigma)
        lam, U = lam[::-1], U[:, ::-1]
        self.mu = mu.astype(np.float32)
        d = X.shape[1]
        if name == "T1":
            self.M = None
        elif name == "T2":
            Ud = U[:, :D_ABTT]
            self.M = (np.eye(d) - Ud @ Ud.T).astype(np.float32)
        elif name == "T3":
            self.M = (U[:, :K_BERT] / np.sqrt(lam[:K_BERT])).astype(np.float32)
        elif name == "T4":
            eps = EPS_SHRINK * lam.mean()
            self.M = ((U / np.sqrt(lam + eps)) @ U.T).astype(np.float32)
        else:
            raise ValueError(name)
        self.dim = self.M.shape[1] if self.M is not None else d

    def __call__(self, X: np.ndarray, chunk: int = 16384) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if self.name == "T0":
            return unit(X).astype(np.float32)
        out = np.empty((len(X), self.dim), dtype=np.float32)
        for a in range(0, len(X), chunk):
            c = X[a:a + chunk] - self.mu
            if self.M is not None:
                c = c @ self.M
            out[a:a + chunk] = unit(c)
        return out

    def direction(self, T: np.ndarray) -> np.ndarray:
        T = np.atleast_2d(np.asarray(T, dtype=np.float32))
        if self.M is None:
            return unit(T).astype(np.float32)
        return unit(T @ self.M).astype(np.float32)


def far_with_null(P: np.ndarray, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Line-for-line latent_gap.targets.far, also returning the ceilings of all candidates (null scale, W2.5)."""
    C = TG._norm(rng.standard_normal((TG.N_CANDIDATES, P.shape[1])).astype(np.float32))
    m = TG.max_cos(P, C)
    return C[np.argsort(m)[:n]], m


# ---------------------------------------------------------------- W1

def w1_geometry(Y: np.ndarray, n_pairs: int, n_slab: int, n_hub: int) -> tuple[dict, np.ndarray]:
    N, d = Y.shape
    rng = np.random.default_rng(SEED + 1)
    a, b = rng.integers(0, N, (2, n_pairs))
    while (bad := np.flatnonzero(a == b)).size:
        b[bad] = rng.integers(0, N, bad.size)
    pair = np.einsum("nd,nd->n", Y[a], Y[b])
    slab_idx = rng.choice(N, n_slab, replace=False)  # same generator, after the pairs
    mu, M2 = second_moments(Y)
    w, U = np.linalg.eigh(M2)
    w, U = w[::-1], U[:, ::-1]
    k90 = int(np.searchsorted(np.cumsum(w) / w.sum(), VAR_KEEP) + 1)
    U90 = U[:, :k90].astype(np.float32)
    lc = np.clip(np.linalg.eigvalsh(M2 - np.outer(mu, mu)), 0, None)
    # hubness: 10-occurrence over a random subsample, cosine kNN inside it
    hr = np.random.default_rng(SEED + 2)
    H = Y[hr.choice(N, n_hub, replace=False)]
    occ = np.zeros(n_hub, dtype=np.int64)
    for s in range(0, n_hub, 1000):
        S = H[s:s + 1000] @ H.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        nn = np.argpartition(-S, HUB_K, axis=1)[:, :HUB_K]
        occ += np.bincount(nn.ravel(), minlength=n_hub)
    q25, q50, q75 = np.quantile(pair, [0.25, 0.5, 0.75])
    slab_items = np.linalg.norm(Y[slab_idx] @ U90, axis=1)
    out = {"d": int(d), "pair_cos_median": float(q50), "pair_cos_iqr": [float(q25), float(q75)],
           "norm_of_mean": float(np.linalg.norm(mu)), "k90": k90,
           "in_slab_random_items_median": float(np.median(slab_items)),
           "in_slab_random_direction": float(np.sqrt(k90 / d)),
           "participation_ratio": float(lc.sum() ** 2 / (lc ** 2).sum()),
           "hubness_skew_n10": float(skew(occ)), "n_pairs": int(n_pairs), "n_hub": int(n_hub)}
    return out, U90


# ---------------------------------------------------------------- W2

def boot_median_ci(x: np.ndarray, idx: np.ndarray) -> list[float]:
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def nn_reference(Y: np.ndarray, n_ref: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 3)
    idx = rng.choice(len(Y), n_ref, replace=False)
    out = np.empty(n_ref, dtype=np.float32)
    for s in range(0, n_ref, 500):
        S = Y[idx[s:s + 500]] @ Y.T
        S[S > TG.DUP_COS] = -np.inf  # itself (cos 1) and its duplicates
        out[s:s + 500] = S.max(1)
    return out


def w2_targets(Y: np.ndarray, Fy: np.ndarray, c: int, e: int, t: int, n_ref: int, n_boot: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED + 1000 * c + 100 * e + 10 * t)
    R, hidden, ridx = TG.real(Y, rng, N_TARGETS)
    G = TG.gap(Y, rng, N_TARGETS)
    Fr, null = far_with_null(Y, rng, N_TARGETS)
    ceil_real = np.empty(N_TARGETS, dtype=np.float32)
    for i in range(N_TARGETS):
        s = Y @ R[i]
        s[hidden[i]] = -np.inf
        ceil_real[i] = s.max()
    ceil_gap, ceil_far = TG.max_cos(Y, G), TG.max_cos(Y, Fr)
    ref = nn_reference(Y, n_ref)
    q01, q50 = float(np.quantile(ref, 0.01)), float(np.quantile(ref, 0.50))
    fut_far = (Fy @ Fr.T).max(0)
    fut_gap = (Fy @ G.T).max(0)
    br = np.random.default_rng(SEED + 4)
    meds = {"FAR": [], "GAP": [], "rho_FAR": [], "rho_GAP": []}
    step = min(1000, n_boot)
    for _ in range(0, n_boot, step):  # per chunk: FAR target idx, FAR ref idx, GAP target idx, GAP ref idx
        tf = br.integers(0, N_TARGETS, (step, N_TARGETS))
        rf = br.integers(0, len(ref), (step, len(ref)))
        tg = br.integers(0, N_TARGETS, (step, N_TARGETS))
        rg = br.integers(0, len(ref), (step, len(ref)))
        mf, mg = np.median(ceil_far[tf], 1), np.median(ceil_gap[tg], 1)
        meds["FAR"].append(mf)
        meds["GAP"].append(mg)
        meds["rho_FAR"].append(mf / np.median(ref[rf], 1))
        meds["rho_GAP"].append(mg / np.median(ref[rg], 1))
    meds = {k: np.concatenate(v) for k, v in meds.items()}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    med_far, max_far = float(np.median(ceil_far)), float(ceil_far.max())
    fut_any = bool((fut_far > q01).any())
    if max_far < q01 and not fut_any:
        verdict = "FAR' unreachable"
    elif med_far < q01:
        verdict = "mostly unreachable" if max_far >= q01 else "mostly unreachable (past unreachable, future reaches)"
    else:
        verdict = "reachable"
    res = {"q01": q01, "q50": q50, "n_ref": int(len(ref)),
           "REAL_ceiling_median": float(np.median(ceil_real)),
           "GAP_ceiling_median": float(np.median(ceil_gap)), "GAP_ceiling_ci": ci(meds["GAP"]),
           "GAP_ceiling_max": float(ceil_gap.max()), "rho_GAP": float(np.median(ceil_gap)) / q50,
           "rho_GAP_ci": ci(meds["rho_GAP"]),
           "FAR_ceiling_median": med_far, "FAR_ceiling_ci": ci(meds["FAR"]), "FAR_ceiling_max": max_far,
           "rho_FAR": med_far / q50, "rho_FAR_ci": ci(meds["rho_FAR"]),
           "FAR_future_max_median": float(np.median(fut_far)), "FAR_future_max_max": float(fut_far.max()),
           "FAR_share_future_above_q01": float((fut_far > q01).mean()),
           "GAP_future_max_median": float(np.median(fut_gap)),
           "FAR_null_median_ceiling": float(np.median(null)), "n_candidates": int(TG.N_CANDIDATES),
           "n_boot": int(n_boot), "verdict": verdict}
    return res, {"FAR": Fr}


# ---------------------------------------------------------------- W3

def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_anchoring(dom: str) -> tuple[dict, dict]:
    with np.load(DATA / f"runs/{dom}/pilot/targets.npz", allow_pickle=True) as z:
        tg = {a: np.asarray(z[a], dtype=np.float64) for a in W3_ARMS}
    rows = {}
    for gen in W3_GENS:
        for cond in W3_CONDS:
            for arm in W3_ARMS:
                for r in read_rows(DATA / f"runs/{dom}/anchoring-causal/{gen}/{cond}/{arm}/items.jsonl"):
                    if r["k"] in KS:
                        rows[gen, cond, arm, r["k"]] = r
    return tg, rows


def interval(v: np.ndarray, alpha: float) -> dict:
    v = np.asarray(v, dtype=float)
    m = float(v.mean())
    h = float(student_t.ppf(1 - alpha, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": m, "low": m - h, "high": m + h}


def w3_cells(dom: str, tname: str, f: Transform, Y: np.ndarray, tg: dict, rows: dict) -> tuple[list, list]:
    """Per-cell Table A2 quantities in the space of f, under the point map f(t) and the direction map g(t)."""
    items, cells = [], []
    for arm in W3_ARMS:
        for gen in W3_GENS:
            for mapname in ("point", "direction"):
                per = []
                for k in KS:
                    rt, rs, rn = (rows.get((gen, c, arm, k)) for c in W3_CONDS)
                    if not (rt and rs and rn and rt.get("best") and rs.get("best") and rn.get("best")):
                        continue
                    t = (f if mapname == "point" else f.direction)(tg[arm][k][None])[0].astype(np.float64)
                    E = {c: f(np.asarray(r["best"]["emb"])[None])[0].astype(np.float64)
                         for c, r in zip(W3_CONDS, (rt, rs, rn))}
                    Ct = unit(Y[rt["neighbours"]].astype(np.float64).mean(0))
                    Cd = unit(Y[rs["neighbours"]].astype(np.float64).mean(0))
                    lt, ls, ln = (float(E[c] @ t) for c in W3_CONDS)
                    it = {"domain": dom, "transform": tname, "map": mapname, "gen": gen, "arm": arm, "k": k,
                          "donor_k": rs["donor_k"], "landing_true": lt, "landing_swap": ls, "landing_none": ln,
                          "D": lt - ls, "D_none": lt - ln, "A": float((E["swap"] - E["true"]) @ (Cd - Ct))}
                    per.append(it)
                items += per
                if not per:
                    continue
                df = pd.DataFrame(per)
                cell = {"domain": dom, "transform": tname, "map": mapname, "gen": gen, "arm": arm, "n": len(df),
                        "landing": {c: float(df[f"landing_{c}"].mean()) for c in W3_CONDS}}
                for a_name, a in (("orig", ALPHA_ORIG), ("corr", ALPHA_CORR)):
                    cell[f"D_{a_name}"] = interval(df.D, a) if len(df) > 1 else None
                    cell[f"A_{a_name}"] = interval(df.A, a) if len(df) > 1 else None
                    cell[f"D_none_{a_name}"] = interval(df.D_none, a) if len(df) > 1 else None
                cells.append(cell)
    return items, cells


def check_table_a2(cells: list) -> list[str]:
    ref = json.loads((DATA / "runs/anchoring-causal-analysis/results.json").read_text())["cells"]
    bad = []
    for r in ref:
        mine = [c for c in cells if c["domain"] == r["domain"] and c["gen"] == r["gen"] and c["arm"] == r["arm"]
                and c["map"] == "point"]
        if not mine:
            continue
        c = mine[0]
        pairs = [(c["D_orig"][q], r["drop"][q]) for q in ("mean", "low", "high")]
        pairs += [(c["A_orig"][q], r["attraction"][q]) for q in ("mean", "low", "high")]
        pairs += [(c["landing"][cond], r["quantitative"][cond]["means"]["target"]) for cond in W3_CONDS]
        if c["n"] != 20 or any(abs(a - b) > TOL for a, b in pairs):
            bad.append(f"{r['domain']}/{r['gen']}/{r['arm']}: {[(round(a, 4), round(b, 4)) for a, b in pairs]}")
    return bad


def w3_verdict(cells: list, tname: str) -> dict:
    cs = [c for c in cells if c["transform"] == tname and c["map"] == "point"]
    n_pos = sum(c["A_corr"]["low"] > 0 for c in cs)
    any_neg = any(c["A_corr"]["high"] < 0 for c in cs)
    anch = "survives" if n_pos == 12 else ("weakens" if 9 <= n_pos <= 11 and not any_neg else "fails")
    far = [c for c in cs if c["arm"] == "FAR"]
    far_swap = sum(c["D_corr"]["low"] > 0 for c in far)
    far_none = sum(c["D_none_corr"]["low"] > 0 for c in far)
    return {"n_cells": len(cs), "A_low_corr_positive": n_pos, "any_A_entirely_negative": any_neg,
            "anchoring": anch if len(cs) == 12 else f"incomplete ({len(cs)} cells): {anch}",
            "FAR_D_swap_low_corr_positive": far_swap, "FAR_D_none_low_corr_positive": far_none,
            "FAR_landing_claim_survives_D_only": far_swap == 4 and len(far) == 4,
            "FAR_landing_claim_survives_D_and_none": far_swap == 4 and far_none == 4 and len(far) == 4}


# ---------------------------------------------------------------- driver

def check_a3(dom: str, Y: np.ndarray) -> dict:
    ref = json.loads((ROOT / "docs/prereg/far_targets_vs_future.json").read_text())[dom]["past_max_cos_to_target"]
    with np.load(DATA / f"runs/{dom}/grid-gpt-6-luna/targets.npz") as z:
        T = unit(np.asarray(z["FAR"], dtype=np.float32))
    c = (Y @ T.T).max(0)
    got = {"median": float(np.median(c)), "max": float(c.max())}
    return {"got": got, "ref": ref, "ok": all(abs(got[q] - ref[q]) <= TOL for q in ("median", "max"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--items-out", type=Path, default=ITEMS_OUT)
    ap.add_argument("--smoke", action="store_true", help="tiny subset (steam, qwen3); never stops on reproduction")
    ap.add_argument("--domains", default=",".join(DOMS))
    ap.add_argument("--embedders", default="qwen3,bge")
    args = ap.parse_args()
    smoke = args.smoke
    if smoke and args.out == OUT:
        raise SystemExit("--smoke must write to a temp --out, not the registered JSON")
    n_pairs, n_slab, n_hub, n_ref, n_boot = (N_PAIRS, N_SLAB_ITEMS, N_HUB, N_REF, B)
    if smoke:
        TG.N_CANDIDATES = 1000
        n_pairs, n_slab, n_hub, n_ref, n_boot = 5000, 500, 1000, 300, 200
        if args.items_out == ITEMS_OUT:
            args.items_out = args.out.with_suffix(".items.csv")
    doms = [d for d in args.domains.split(",") if d]
    embs = [e for e in EMBS if e[0] in args.embedders.split(",")]
    res = {"prereg": "docs/prereg/review_far_decoding.md#R1", "smoke": smoke, "w1": [], "w2": [], "w3_cells": [],
           "reproduction": {}, "constants": {"seed": SEED, "B": n_boot, "N_CANDIDATES": TG.N_CANDIDATES,
                                             "n_pairs": n_pairs, "n_hub": n_hub, "n_ref": n_ref}}
    all_items = []
    for dom in doms:
        c = DOMS.index(dom)
        for e, (ename, efile) in enumerate(EMBS):
            if (ename, efile) not in embs:
                continue
            log(f"load {dom}/{ename}")
            X, F = load_split(dom, efile)
            w3 = dom in W3_DOMS and ename == "qwen3"
            if w3:
                tg, rows = load_anchoring(dom)
            sub = None
            if smoke:  # W1/W2 on a 6,000-row subsample; transforms and W3 use the full past (neighbour indices)
                sub = np.random.default_rng(0).choice(len(X), 6000, replace=False)
                F = F[np.random.default_rng(1).choice(len(F), 2000, replace=False)]
            for ti, tname in enumerate(TRANSFORMS):
                t0 = time.time()
                f = Transform(tname, X)
                Y = f(X)
                Yw = Y[sub] if sub is not None else Y
                Fy = f(F)
                w2, keep = w2_targets(Yw, Fy, c, e, ti, n_ref, n_boot)
                del Fy
                w1, U90 = w1_geometry(Yw, n_pairs, n_slab, n_hub)
                w1["in_slab_FAR_targets_median"] = float(np.median(np.linalg.norm(keep["FAR"] @ U90, axis=1)))
                cell = {"domain": dom, "embedder": ename, "transform": tname}
                res["w1"].append(cell | w1)
                res["w2"].append(cell | w2)
                if tname == "T0" and ename == "qwen3" and not smoke:
                    a3 = check_a3(dom, Y)
                    res["reproduction"][f"A3_{dom}"] = a3
                    if not a3["ok"]:
                        raise SystemExit(f"T0 does not reproduce the A.3 FAR ceilings for {dom}: {a3}")
                if w3:
                    items, cells = w3_cells(dom, tname, f, Y, tg, rows)
                    all_items += items
                    res["w3_cells"] += cells
                    if tname == "T0":
                        bad = check_table_a2(cells)
                        res["reproduction"][f"TableA2_{dom}"] = {"ok": not bad, "mismatches": bad}
                        if bad and not smoke:
                            raise SystemExit(f"T0 does not reproduce Table A2 ({dom}): {bad}")
                        log(f"Table A2 reproduction {dom}: {'OK' if not bad else bad}")
                log(f"{dom}/{ename}/{tname}: pair med {w1['pair_cos_median']:.3f} |mu| {w1['norm_of_mean']:.3f} "
                    f"k90 {w1['k90']} FAR' med {w2['FAR_ceiling_median']:.4f} max {w2['FAR_ceiling_max']:.4f} "
                    f"q01 {w2['q01']:.4f} q50 {w2['q50']:.4f} fut>q01 {w2['FAR_share_future_above_q01']:.2f} "
                    f"-> {w2['verdict']} ({time.time() - t0:.0f}s)")
                del Y, Yw, f
            del X, F
    # verdicts
    v = {}
    for dom in doms:
        for tname in TRANSFORMS:
            ver = {r["embedder"]: r["verdict"] for r in res["w2"] if r["domain"] == dom and r["transform"] == tname}
            agree = len(set(ver.values())) == 1 and len(ver) == 2
            pc = {r["embedder"]: r["pair_cos_median"] for r in res["w1"] if r["domain"] == dom and r["transform"] == tname}
            v[f"{dom}/{tname}"] = {"by_embedder": ver, "counts": agree,
                                   "verdict": next(iter(ver.values())) if agree else "embedders disagree / missing",
                                   "explanation_cap_wrong": agree and ver and next(iter(ver.values())) == "FAR' unreachable"
                                   and all(p <= 0.05 for p in pc.values())}
    corr = [v[f"{d}/{t}"] for d in doms for t in TRANSFORMS[1:]]
    res["w2_verdicts"] = v
    res["w2_overall"] = ("supports current claim" if corr and all(x["counts"] and x["verdict"] == "FAR' unreachable" for x in corr)
                         else "counts against current claim" if any(x["counts"] and x["verdict"] == "reachable" for x in corr)
                         else "neither (mixed or embedders disagree)")
    if res["w3_cells"]:
        res["w3_verdicts"] = {t: w3_verdict(res["w3_cells"], t) for t in TRANSFORMS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    if all_items:
        args.items_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_items).to_csv(args.items_out, index=False)
    log(f"wrote {args.out}; overall W2: {res['w2_overall']}")
    if "w3_verdicts" in res:
        log("W3: " + json.dumps({t: (x["anchoring"], x["FAR_landing_claim_survives_D_only"]) for t, x in res["w3_verdicts"].items()}))


if __name__ == "__main__":
    main()
