"""Shared measures for the NEUTRAL-arm studies (strengthen_modern_neutral.md, strengthen_critique_replication.md).

Per context i, arm a, round r (mean over the context's ideas with an abstract), exactly as revision_neutral.analyse:
step = cos(x, c); fit = max cos to the saved follow-ups; hit = share of ideas whose nearest neighbour among follow-ups
∪ distractors is a follow-up; align = tangent·t (t = unit mean tangent of the follow-ups); p = tangent·tp (tp = unit
mean tangent of the k nearest pool papers not among the saved follow-ups, k = number of follow-ups); s = tangent·ts
(ts = unit(t − (t·tp)tp)). Tangent of x = unit(x − (x·c)c).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

MEASURES = ("step", "fit", "hit", "align", "p", "s")


def tang(X, c):
    T = X - np.outer(X @ c, c)
    return T / np.linalg.norm(T, axis=-1, keepdims=True)


def unit(v):
    return v / np.linalg.norm(v)


def per_context(ctx, C, F, sources, arms, pool, pool_ids, distractors, pool_exclude=None):
    """sources: {arm: (rows, V)}; distractors(i, fids) -> matrix; pool_exclude(i) -> extra ids to drop from the pool.
    Returns M[a][m] = array (n_ctx, 3) and the follow-ups' leave-one-out alignment per context."""
    M = {a: {m: np.full((len(ctx), 3), np.nan) for m in MEASURES} for a in arms}
    index = {}
    for a in arms:
        rows, _ = sources[a]
        for n, x in enumerate(rows):
            if x["arm"] == a and x.get("abstract"):
                index.setdefault((a, x["ctx"], x["round"]), []).append(n)
    loo = np.full(len(ctx), np.nan)
    for i in range(len(ctx)):
        c, Fi = C[i], F[i]
        fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
        excl = fids | (pool_exclude(i) if pool_exclude else set())
        Dv = distractors(i, fids)
        TF = tang(Fi, c)
        t = unit(TF.mean(0))
        if len(TF) > 1:
            S = TF.sum(0)
            loo[i] = float(np.mean([TF[j] @ unit(S - TF[j]) for j in range(len(TF))]))
        P = pool[~np.isin(pool_ids, list(excl))]
        tp = unit(tang(P[np.argsort(-(P @ c))[:len(Fi)]], c).mean(0))
        ts = unit(t - (t @ tp) * tp)
        for a in arms:
            _, V = sources[a]
            for r in range(3):
                idx = index.get((a, i, r))
                if not idx:
                    continue
                X = V[idx]
                sf = X @ Fi.T
                T = tang(X, c)
                ok = np.isfinite(T).all(1)  # an idea identical to its context has no tangent
                M[a]["step"][i, r] = (X @ c).mean()
                M[a]["fit"][i, r] = sf.max(1).mean()
                M[a]["hit"][i, r] = (sf.max(1) > (X @ Dv.T).max(1)).mean()
                if ok.any():
                    M[a]["align"][i, r] = (T[ok] @ t).mean()
                    M[a]["p"][i, r] = (T[ok] @ tp).mean()
                    M[a]["s"][i, r] = (T[ok] @ ts).mean()
    return M, loo


def change(M, a, m):
    """step: r0 − r2 (positive = longer step); others r2 − r0."""
    return M[a][m][:, 0] - M[a][m][:, 2] if m == "step" else M[a][m][:, 2] - M[a][m][:, 0]


def boot(x, rng, n_boot=2000):
    return [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)], q)) for q in (2.5, 97.5)]


def holm(p: dict) -> dict:
    keys, out, run = sorted(p, key=p.get), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        out[k] = run
    return out


def tost(d, bound=0.015):
    """Paired-t TOST: p = max of the two one-sided p-values (each at 0.025 ⇔ 95 % t-CI inside ±bound)."""
    n, m, se = len(d), float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    p_lo = 1 - stats.t.cdf((m + bound) / se, n - 1)   # H0: mean <= −bound
    p_hi = stats.t.cdf((m - bound) / se, n - 1)       # H0: mean >= +bound
    tq = stats.t.ppf(0.975, n - 1)
    return {"p_tost": float(max(p_lo, p_hi)), "t_ci95": [m - tq * se, m + tq * se], "se": se}


def summary(M, arms, rng):
    out = {}
    for a in arms:
        e = {f"round{r}": {m: float(np.nanmean(M[a][m][:, r])) for m in MEASURES} for r in range(3)}
        for m in MEASURES:
            d = change(M, a, m)
            ok = ~np.isnan(d)
            e["step_lengthened" if m == "step" else f"change_{m}"] = {
                "mean": float(d[ok].mean()), "ci": boot(d[ok], rng), "p": float(stats.wilcoxon(d[ok]).pvalue), "n": int(ok.sum())}
        out[a] = e
    return out


def contrast(M, a, m, rng, alt="two-sided"):
    d = change(M, a, m) - change(M, "NEUTRAL", m)
    ok = ~np.isnan(d)
    d = d[ok]
    return d, {"diff_mean": float(d.mean()), "ci": boot(d, rng), "p": float(stats.wilcoxon(d, alternative=alt).pvalue),
               "alternative": alt, "n": int(ok.sum())}
