"""Part A of docs/prereg/review_olmo_reliability.md (peer review M8 + question 8): how much the OLMo bound bounds.

  OMP_NUM_THREADS=4 uv run python scripts/review_olmo_reliability.py                  # full run -> JSON below
  OMP_NUM_THREADS=4 uv run python scripts/review_olmo_reliability.py --smoke 30 --boot 1000 --spaces qwen3e06 \
      --out /tmp/x.json                                                                 # smoke test (subset)

Registered follow-up to known results (no paid API, no GPU, no generation, no new labels). Loading, the pool, f, p, s
and the per-arm/round values come unchanged from scripts/olmo_direction_analysis.py (load_outputs, future_pool,
per_context, delta), imported. The extra per-context geometry needed here (consensus tangents t_j, the registered p,
u_j = P t_j) is recomputed with the same tang/unit and pool rule, and checked against per_context's s.
First check: the means of the three primary contrasts and of the four per-arm changes reproduce
strengthen_olmo_direction.json to 1e-9 on all 150 contexts (always on the full set, also in --smoke); else stop.
Output: docs/prereg/review_olmo_reliability.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import olmo_direction_analysis as O  # noqa: E402
from direction_tangent import tang, unit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_olmo_reliability.json"
REG = ROOT / "docs/prereg/strengthen_olmo_direction.json"
SEED, N_BOOT, BOUND, N_BLOCKS, MAX_PART = 20261106, 20000, 0.015, 20, 200
CRIT, ARMS = O.CRIT, O.ARMS
STRATA = (("k=2", lambda k: k == 2), ("k=3-4", lambda k: 3 <= k <= 4), ("k>=5", lambda k: k >= 5))


# ----------------------------------------------------------------------------------------------- geometry
def geometry(o: dict, rec: dict, citing: set, space: str, E, ids) -> dict:
    """t_j (consensus tangents), registered p (k nearest non-citing future-split papers), u_j = P t_j, s = unit(Σ u_j).
    Same pool rule and functions as O.per_context."""
    c, F = o[space]["C"], o[space]["F"]
    js = rec["consensus"]
    TC = tang(F, c)[js]
    pool = E[~np.isin(ids, list(citing | {rec["arxiv_id"]}))]
    order = np.argsort(-(pool @ c))
    p = unit(tang(pool[order[:len(js)]], c).mean(0))
    f = unit(TC.mean(0))
    s_reg = unit(f - (f @ p) * p)
    U = TC - np.outer(TC @ p, p)
    s = unit(U.sum(0))
    assert np.abs(s - s_reg).max() < 1e-5, "s != unit(f - (f.p)p)"
    # s reproduces per_context's projections of the NEUTRAL round-0 ideas
    idx = [n for n, x in enumerate(o["rows"]) if x["arm"] == "NEUTRAL" and x["round"] == 0]
    proj = float((tang(o[space]["V"][idx], c) @ s_reg).mean()) if idx else np.nan
    return {"T": TC, "U": U, "s": s, "proj_check": proj}


def loo_on_s(T, U) -> float:
    """L_c = mean_j t_j · unit(Σ_{i≠j} u_i)."""
    S = U.sum(0)
    return float(np.mean([T[j] @ unit(S - U[j]) for j in range(len(T))]))


def partitions(k: int) -> list[tuple]:
    """Distinct partitions into halves of sizes floor(k/2), ceil(k/2), as the floor-sized half (lexicographic).
    Even k: a half and its complement are the same partition, so only halves containing element 0 are kept."""
    h = k // 2
    out = list(itertools.combinations(range(k), h))
    if k % 2 == 0:
        out = [a for a in out if 0 in a]
    return out


def split_half(U, pos: int) -> tuple[float, int, int, int]:
    k = len(U)
    parts = partitions(k)
    n_all = len(parts)
    if n_all > MAX_PART:
        sel = np.random.default_rng(SEED + 7000 + pos).choice(n_all, MAX_PART, replace=False)
        parts = [parts[i] for i in sel]
    r = []
    for A in parts:
        B = [j for j in range(k) if j not in A]
        r.append(float(unit(U[list(A)].sum(0)) @ unit(U[B].sum(0))))
    return float(np.mean(r)), k // 2, k - k // 2, n_all


def pairwise(U) -> float:
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    G = Un @ Un.T
    iu = np.triu_indices(len(U), 1)
    return float(G[iu].mean())


# ------------------------------------------------------------------------------------------ measurement model
def R(n, lam):
    n, lam = np.asarray(n, float), np.asarray(lam, float)
    return n * lam / (n * lam + 1)


def a_of(n, lam):
    return np.sqrt(R(n, lam))


def solve_lambda(W, r, a, b):
    """λ̂ per row of the weight matrix W (n_rows × n_ctx, counts of each k>=4 context in the resample):
    mean_c sqrt(R(a_c)R(b_c)) = mean_c r_c, bisection on log λ in [1e-6, 1e6], 200 iterations; 0 if mean r <= 0."""
    W = np.atleast_2d(W).astype(float)
    tot = W.sum(1)
    target = (W @ r) / tot
    lo, hi = np.full(len(W), np.log(1e-6)), np.full(len(W), np.log(1e6))
    for _ in range(200):
        mid = (lo + hi) / 2
        lam = np.exp(mid)[:, None]
        lhs = (W * np.sqrt(R(a[None, :], lam) * R(b[None, :], lam))).sum(1) / tot
        up = lhs < target
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    lam = np.exp((lo + hi) / 2)
    return np.where(target <= 0, 0.0, lam)


# --------------------------------------------------------------------------------------------- statistics
def t_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, m = len(x), float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    q = stats.t.ppf(0.975, n - 1)
    return [m - q * se, m + q * se], se


def boot_means(x, IDX, mask=None):
    """Resampled means of x (NaN-aware) on the shared index matrix; mask restricts to a stratum of contexts."""
    x = np.asarray(x, float)
    xs = x[IDX]
    keep = ~np.isnan(xs)
    if mask is not None:
        keep &= mask[IDX]
    cnt = keep.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(keep, xs, 0.0).sum(1) / np.where(cnt > 0, cnt, np.nan)


def pct(b):
    b = np.asarray(b, float)
    b = b[~np.isnan(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mc_se(b):
    """SD of each percentile endpoint over N_BLOCKS consecutive blocks / sqrt(N_BLOCKS)."""
    blocks = np.array_split(np.asarray(b, float), N_BLOCKS)
    ends = np.array([pct(x) for x in blocks])
    return [float(ends[:, 0].std(ddof=1) / np.sqrt(N_BLOCKS)), float(ends[:, 1].std(ddof=1) / np.sqrt(N_BLOCKS))]


def margin(ci):
    return BOUND - max(abs(ci[0]), abs(ci[1]))


def describe(x, IDX, mask=None) -> dict:
    x = np.asarray(x, float)
    sub = x if mask is None else x[mask]
    sub = sub[~np.isnan(sub)]
    tci, se = t_ci(sub)
    b = boot_means(x, IDX, mask)
    bci = pct(b)
    return {"n": int(len(sub)), "mean": float(sub.mean()), "sd": float(sub.std(ddof=1)), "se": se,
            "t_ci95": tci, "boot_ci95": bci, "boot_mc_se": mc_se(b), "p_wilcoxon": O.wil(sub),
            "p_t": float(stats.ttest_1samp(sub, 0.0).pvalue),
            "margin_t": margin(tci), "margin_boot": margin(bci)}


def boot_lambda(IDX, r4, a4, b4, idx4):
    """λ̂ re-estimated in every resample from the resampled k>=4 contexts (their r_c fixed per context)."""
    W = np.zeros((len(IDX), len(idx4)))
    for j, c in enumerate(idx4):
        W[:, j] = (IDX == c).sum(1)
    lam = np.zeros(len(IDX))
    ok = W.sum(1) > 0
    lam[ok] = solve_lambda(W[ok], r4, a4, b4)
    return lam


def corrected_ci(d, kk, IDX, lam, mask=None):
    """Percentile CI of mean(d_c / a(k_c, λ̂_b)) on the shared resamples, λ̂_b from boot_lambda.
    λ̂_b = 0 -> ±inf (sign of that resample's raw mean); an end with > 2.5 % of resamples at that infinity is None."""
    A = a_of(kk[IDX], lam[:, None])                    # (B, n)
    xs = np.asarray(d, float)[IDX]
    keep = ~np.isnan(xs)
    if mask is not None:
        keep &= mask[IDX]
    cnt = keep.sum(1)
    raw = np.where(keep, xs, 0).sum(1) / np.maximum(cnt, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(keep, xs / A, 0).sum(1) / np.maximum(cnt, 1)
    zero = lam == 0
    corr = np.where(zero, np.sign(raw) * np.inf, corr)
    share_pos, share_neg = float(np.mean(corr == np.inf)), float(np.mean(corr == -np.inf))
    fin = np.where(np.isinf(corr), np.sign(corr) * 1e300, corr)
    lo, hi = float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))
    lo = None if share_neg > 0.025 or abs(lo) > 1e299 else lo
    hi = None if share_pos > 0.025 or abs(hi) > 1e299 else hi
    return [lo, hi], {"share_lambda0": float(zero.mean()), "share_+inf": share_pos, "share_-inf": share_neg}


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    diff = float(x.mean() - y.mean())
    q = stats.t.ppf(0.975, df)
    p = float(2 * stats.t.sf(abs(diff / se), df))
    return {"diff": diff, "ci95": [diff - q * se, diff + q * se], "df": float(df), "p": p}


def ols_hc3(y, g):
    X = np.column_stack([np.ones(len(y)), g])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = X.T @ (X * (e ** 2 / (1 - h) ** 2)[:, None])
    V = XtXi @ meat @ XtXi
    se = float(np.sqrt(V[1, 1]))
    return {"slope": float(beta[1]), "se_hc3": se, "z": float(beta[1] / se),
            "p_normal": float(2 * stats.norm.sf(abs(beta[1] / se)))}


# ------------------------------------------------------------------------------------------------- main
def load_space(space, outs, el, citing):
    E, ids = O.future_pool(space)
    recs = {r["pos"]: r for r in el["contexts"]}
    PC, G, pos = [], [], []
    for p in sorted(outs):
        if outs[p]["role"] != "eligible":
            continue
        PC.append(O.per_context(outs[p], recs[p], citing[recs[p]["arxiv_id"]], space, E, ids))
        G.append(geometry(outs[p], recs[p], citing[recs[p]["arxiv_id"]], space, E, ids))
        pos.append(p)
    del E
    return PC, G, pos


def reproduce(PC, G, space, reg) -> dict:
    cell = reg[f"olmo/{space}"]
    out = {}
    for a in CRIT:
        d = np.array([O.delta(pc, a, "s") - O.delta(pc, "NEUTRAL", "s") for pc in PC])
        m = float(d[~np.isnan(d)].mean())
        out[f"{a}-NEUTRAL"] = [m, cell["contrasts_along_s"][a]["mean"]]
    for a in ARMS:
        x = np.array([O.delta(pc, a, "s") for pc in PC])
        out[f"{a}:change"] = [float(x[~np.isnan(x)].mean()), cell["per_arm_change"][a]["s"]["mean"]]
    bad = {k: v for k, v in out.items() if abs(v[0] - v[1]) > 1e-9}
    pc_proj = [pc[("NEUTRAL", 0)]["s"] for pc in PC if ("NEUTRAL", 0) in pc]
    g_proj = [g["proj_check"] for g, pc in zip(G, PC) if ("NEUTRAL", 0) in pc]
    dev = float(np.max(np.abs(np.array(pc_proj) - np.array(g_proj))))
    if bad or dev > 1e-6:
        raise SystemExit(f"reproduction check failed in {space}: {bad}, projection dev {dev}")
    return {"max_abs_dev_means": max(abs(v[0] - v[1]) for v in out.values()), "s_projection_dev": dev,
            "n_contexts": len(PC)}


def analyse_space(PC, G, pos, n_boot, reg_cell) -> dict:
    n = len(PC)
    kk = np.array([len(g["T"]) for g in G])
    rng = np.random.default_rng(SEED)
    IDX = rng.integers(0, n, (n_boot, n))
    stat = {}
    for a in CRIT:
        stat[f"{a}-NEUTRAL"] = np.array([O.delta(pc, a, "s") - O.delta(pc, "NEUTRAL", "s") for pc in PC])
    for a in ARMS:
        stat[f"{a}:change"] = np.array([O.delta(pc, a, "s") for pc in PC])
    v0 = {a: np.array([pc[(a, 0)]["s"] if (a, 0) in pc else np.nan for pc in PC]) for a in ARMS}
    masks = {nm: np.array([f(k) for k in kk]) for nm, f in STRATA}
    cell = {"n_contexts": n, "n_boot": n_boot, "stratum_counts": {nm: int(m.sum()) for nm, m in masks.items()},
            "k_ge4": int((kk >= 4).sum())}

    # A1 ---------------------------------------------------------------------------------------------------
    A1 = {nm: describe(x, IDX) for nm, x in stat.items()}
    for fam, keys in (("contrasts", [f"{a}-NEUTRAL" for a in CRIT]), ("per_arm", [f"{a}:change" for a in ARMS])):
        ps = [A1[k]["p_wilcoxon"] if A1[k]["p_wilcoxon"] is not None else 1.0 for k in keys]
        for k, h in zip(keys, O.holm(ps)):
            A1[k]["p_holm_wilcoxon"] = h
            A1[k]["holm_family"] = fam
    for a in CRIT:
        e = A1[f"{a}-NEUTRAL"]
        both_in = all(-BOUND < x < BOUND for x in e["t_ci95"] + e["boot_ci95"])
        big = int(np.argmax(np.abs(e["boot_ci95"])))
        mse = e["boot_mc_se"][big]
        small = min(e["margin_t"], e["margin_boot"])
        if a == "ORACLE":
            excl = all(ci[0] > 0 or ci[1] < 0 for ci in (e["t_ci95"], e["boot_ci95"]))
            e["reading_R-A1"] = ("detected" if excl and e["p_holm_wilcoxon"] < 0.05
                                 else f"unchanged: {reg_cell['contrasts_along_s']['ORACLE']['reading']}")
        else:
            e["reading_R-A1"] = ("no feedback-specific turn of 0.015 or more" if both_in else "insufficient")
            e["state_margin_explicitly"] = bool(both_in and (small < 0.001 or small < 2 * mse))
            e["smaller_margin"] = small
    cell["A1_interval_robustness"] = A1

    # A2 ---------------------------------------------------------------------------------------------------
    L = np.array([loo_on_s(g["T"], g["U"]) for g in G])
    A2 = {}
    for nm, m in (("overall", None),) + tuple((k, masks[k]) for k, _ in STRATA):
        Lm = L if m is None else L[m]
        Lbar = float(Lm.mean())
        bL = boot_means(L, IDX, m)
        e = {"n": int(len(Lm)), "L_bar": Lbar, "L_bar_boot_ci95": pct(bL), "rho_bound": BOUND / Lbar,
             "rho_bound_boot_ci95": sorted(pct(BOUND / bL))}
        for a in CRIT:
            x = stat[f"{a}-NEUTRAL"] if m is None else stat[f"{a}-NEUTRAL"][m]
            tci, _ = t_ci(x)
            e[f"rho_CI_{a}"] = max(abs(tci[0]), abs(tci[1])) / Lbar
        for a in ARMS:
            vv = v0[a] if m is None else v0[a][m]
            room = Lbar - float(np.nanmean(vv))
            e[f"room_{a}"] = room
            e[f"bound_over_room_{a}"] = BOUND / room
        A2[nm] = e
    rb = A2["overall"]["rho_bound"]
    A2["reading_R-A2"] = ("under a quarter" if rb < 0.25 else "a quarter to a half" if rb < 0.5
                          else "not small: the test excludes only turns of at least half what a real continuation obtains")
    A2["k2_rho_bound"] = A2["k=2"]["rho_bound"]
    A2["note"] = ("L_c uses a direction from k-1 continuations, the ideas one from k: rho is conservative "
                  "(the bound looks larger), most of all at k = 2")
    cell["A2_continuation_loo_on_s"] = A2

    # A3 ---------------------------------------------------------------------------------------------------
    idx4 = [i for i in range(n) if kk[i] >= 4]
    sh = [split_half(G[i]["U"], pos[i]) for i in idx4]
    r4 = np.array([x[0] for x in sh])
    a4 = np.array([x[1] for x in sh], float)
    b4 = np.array([x[2] for x in sh], float)
    lam_hat = float(solve_lambda(np.ones((1, len(idx4))), r4, a4, b4)[0])
    lam_ref = lam_hat
    A3 = {"n_k_ge4": len(idx4), "r_bar": float(r4.mean()), "lambda_hat": lam_hat,
          "partitions": {"n_enumerated_total": int(sum(x[3] for x in sh)),
                         "n_contexts_sampled_200": int(sum(x[3] > MAX_PART for x in sh))},
          "a_implied": {str(m): float(a_of(m, lam_hat)) for m in (2, 3, 4, 5, 10)},
          "R_full_length_per_stratum": {nm: float(np.mean(R(kk[masks[nm]], lam_hat))) for nm, _ in STRATA}}
    a_k = a_of(kk, lam_hat)
    a_km1 = a_of(kk - 1, lam_hat)
    Lstar = L / a_km1
    A3["L_bar_star"] = float(Lstar.mean())
    A3["bound_over_L_bar_star"] = BOUND / float(Lstar.mean())
    corr = {}
    lam_boot = boot_lambda(IDX, r4, a4, b4, idx4)
    for nm, x in stat.items():
        dstar = x / a_k
        ci, sh_ = corrected_ci(x, kk, IDX, lam_boot)
        e = {"mean_raw": float(np.nanmean(x)), "mean_corrected": float(np.nanmean(dstar)), "boot_ci95_corrected": ci,
             **sh_}
        if nm.endswith("-NEUTRAL"):
            ends = [abs(v) if v is not None else np.inf for v in ci]
            e["rho_star"] = max(ends) / A3["L_bar_star"]
        if nm.startswith("ORACLE"):
            e["caveat"] = "ORACLE's feedback uses the saved follow-ups, so its ideas are not independent of the continuation noise"
        corr[nm] = e
    A3["corrected"] = corr
    A3["lambda_hat_boot_ci95"] = pct(lam_boot)
    # model checks
    r1 = np.array([pairwise(g["U"]) for g in G])
    chk = {}
    for nm, m in (("k>=4", kk >= 4),) + tuple((k, masks[k]) for k, _ in STRATA):
        rb1 = float(r1[m].mean())
        chk[nm] = {"r1_bar": rb1, "lambda_pair": rb1 / (1 - rb1) if rb1 > 0 else 0.0}
    lp4 = chk["k>=4"]["lambda_pair"]
    lci = A3["lambda_hat_boot_ci95"]
    chk["lambda_pair_k4_inside_splithalf_ci"] = bool(lci[0] <= lp4 <= lci[1])
    Lb = [A2[nm]["L_bar"] for nm, _ in STRATA]
    pa = [float(np.mean(a_of(kk[masks[nm]] - 1, lam_hat))) for nm, _ in STRATA]
    chk["L_bar_ratio_observed"] = [x / Lb[0] for x in Lb]
    chk["L_bar_ratio_model"] = [x / pa[0] for x in pa]
    chk["L_bar_ordering_opposite"] = bool(Lb[0] > Lb[1] > Lb[2])
    chk["L_bar_monotone_increasing"] = bool(Lb[0] < Lb[1] < Lb[2])
    lam_str = np.array([next(chk[nm]["lambda_pair"] for nm, f in STRATA if f(k)) for k in kk])
    a_str = a_of(kk, lam_str)
    chk["sensitivity_stratum_lambda_pair"] = {nm: float(np.nanmean(x / np.where(a_str > 0, a_str, np.nan)))
                                              for nm, x in stat.items()}
    A3["model_checks"] = chk
    fail = (not chk["lambda_pair_k4_inside_splithalf_ci"]) or chk["L_bar_ordering_opposite"]
    ends = []
    for a in ("GEO", "LLM"):
        ci = corr[f"{a}-NEUTRAL"]["boot_ci95_corrected"]
        ends.append(np.inf if None in ci else max(abs(ci[0]), abs(ci[1])))
    X = float(max(ends))
    if fail:
        out = "d: model checks fail; corrected interval model-dependent, only outcome (c) wording may be used"
    elif X < BOUND:
        out = "a: bound also holds on the latent continuation direction"
    elif X < 0.03:
        out = "b: excludes 0.015 on the estimated direction; on the latent one turns of about X or more"
    else:
        out = "c: reliability of s cannot support a bound on the latent direction"
    A3["X_larger_corrected_end_GEO_LLM"] = None if np.isinf(X) else X
    A3["reading_R-A3"] = out
    A3["lambda_hat_reference"] = lam_ref
    cell["A3_reliability"] = A3

    # A4 ---------------------------------------------------------------------------------------------------
    A4 = {"counts": cell["stratum_counts"]}
    for nm, _ in STRATA:
        A4[nm] = {k: describe(x, IDX, masks[nm]) for k, x in stat.items()}
    het = {}
    k2, k3 = kk == 2, kk >= 3
    for a in CRIT:
        x = stat[f"{a}-NEUTRAL"]
        het[a] = welch(x[k2 & ~np.isnan(x)], x[k3 & ~np.isnan(x)])
    for a, h in zip(CRIT, O.holm([het[a]["p"] for a in CRIT])):
        het[a]["p_holm3"] = h
    A4["heterogeneity_k2_vs_k3plus"] = het
    g = np.select([kk == 2, kk <= 4], [0, 1], 2).astype(float)
    A4["trend_descriptive"] = {a: ols_hc3(stat[f"{a}-NEUTRAL"][~np.isnan(stat[f"{a}-NEUTRAL"])],
                                          g[~np.isnan(stat[f"{a}-NEUTRAL"])]) for a in CRIT}
    A4["power_note"] = "SE of the k=2 vs k>=3 difference about 0.0115: differences below about 0.032 have under 80 % power"
    sig = [a for a in ("GEO", "LLM") if het[a]["p_holm3"] < 0.05]
    flag = [a for a in ("GEO", "LLM") if (lambda c: c[0] > 0 or c[1] < 0)(A4["k>=5"][f"{a}-NEUTRAL"]["t_ci95"])]
    A4["reading_R-A5"] = ("heterogeneous: pooled claim qualified, stratum estimates in the text" if sig
                          else "counts + stratum table in appendix with the power caveat")
    A4["k_ge5_flag"] = flag
    cell["A4_strata"] = A4

    # R-A4 -------------------------------------------------------------------------------------------------
    ch = {a: A1[f"{a}:change"] for a in ARMS}
    largest = max(ARMS, key=lambda a: ch[a]["mean"])
    cell["reading_R-A4"] = {"largest": largest, "oracle_largest": largest == "ORACLE",
                            "oracle_holm_sig": bool(ch["ORACLE"]["p_holm_wilcoxon"] < 0.05)}
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="analyse only every (150/N)-th context (reproduction on all)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--spaces", default=",".join(O.SPACES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.boot % N_BLOCKS:
        raise SystemExit("--boot must be a multiple of 20")
    el = json.loads((O.RUN / "eligible.json").read_text())
    citing = {json.loads(l)["arxiv_id"]: set(json.loads(l)["citing_arxiv_all"])
              for l in (O.RUN / "citing_all.jsonl").read_text().splitlines()}
    reg = json.loads(REG.read_text())
    outs = O.load_outputs()
    res = {"prereg": "docs/prereg/review_olmo_reliability.md", "part": "A",
           "status": "registered follow-up to known results (not an original registration)",
           "smoke": bool(args.smoke), "seed": SEED, "n_boot": args.boot, "deviations": DEVIATIONS}
    for space in args.spaces.split(","):
        PC, G, pos = load_space(space, outs, el, citing)
        chk = reproduce(PC, G, space, reg)
        print(space, "reproduction check ok", chk, flush=True)
        if args.smoke:
            step = max(1, len(PC) // args.smoke)
            sel = list(range(0, len(PC), step))[:args.smoke]
            PC, G, pos = [PC[i] for i in sel], [G[i] for i in sel], [pos[i] for i in sel]
        cell = analyse_space(PC, G, pos, args.boot, reg[f"olmo/{space}"])
        cell["reproduction_check"] = chk
        res[f"olmo/{space}"] = cell
        print(space, json.dumps(summary(cell), indent=1, default=str), flush=True)
    res["primary_cell"] = "olmo/qwen3e06"
    Path(args.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("wrote", args.out)


def summary(cell):
    A1, A2, A3 = cell["A1_interval_robustness"], cell["A2_continuation_loo_on_s"], cell["A3_reliability"]
    return {"n": cell["n_contexts"], "strata": cell["stratum_counts"],
            "A1": {k: [round(v["mean"], 4), [round(x, 4) for x in v["t_ci95"]], [round(x, 4) for x in v["boot_ci95"]],
                       v.get("reading_R-A1", "")] for k, v in A1.items()},
            "A2": {k: round(v["rho_bound"], 3) for k, v in A2.items() if isinstance(v, dict)},
            "A3": {"lambda": A3["lambda_hat"], "a2": A3["a_implied"]["2"], "R-A3": A3["reading_R-A3"]},
            "R-A5": cell["A4_strata"]["reading_R-A5"]}


DEVIATIONS = [
    "Bootstrap means are NaN-aware (a context with no value for a statistic is skipped within a resample); with no "
    "missing values, as in the registered 150, this is identical to the registered estimand.",
    "The per-context geometry (t_j, p, u_j) is recomputed in this script with the same tang/unit functions and pool "
    "rule as olmo_direction_analysis.per_context (which returns only scalars); it is checked against per_context's "
    "projections on s (max deviation reported) and s = unit(f - (f.p)p) is asserted.",
    "Even k: the two halves of a split are interchangeable, so a partition is counted once (halves containing the "
    "first continuation); odd k: the floor-sized half identifies the partition.",
    "Bisection: if mean r_c exceeds the value reachable at lambda = 1e6, lambda_hat is capped at 1e6 (not observed "
    "unless reported).",
    "A3 interval end 'unbounded' is applied per side: an end is None if more than 2.5 % of resamples sit at that "
    "infinity.",
    "R-A1 for ORACLE uses the Holm-3 Wilcoxon p (the registered test); the t interval and bootstrap interval must "
    "both exclude zero.",
]

if __name__ == "__main__":
    main()
