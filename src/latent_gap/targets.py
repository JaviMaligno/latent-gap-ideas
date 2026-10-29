"""Target points in embedding space, built from the past corpus only (see docs/design.md, Arms)."""
from __future__ import annotations

import numpy as np

N_CANDIDATES = 20_000
DUP_COS = 0.95


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def max_cos(P: np.ndarray, C: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Max cosine of each row of C against the corpus P (both L2-normalised)."""
    out = np.empty(len(C), dtype=np.float32)
    for s in range(0, len(C), chunk):
        out[s:s + chunk] = (C[s:s + chunk] @ P.T).max(axis=1)
    return out


def real(P: np.ndarray, rng: np.random.Generator, n: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Targets on held-out past items; `hidden` lists, per target, the corpus rows the decoder must not see."""
    idx = rng.choice(len(P), n, replace=False)
    hidden = [np.flatnonzero(P @ P[i] > DUP_COS) for i in idx]  # includes i itself
    return P[idx].copy(), hidden, idx


def distant_pairs(P: np.ndarray, rng: np.random.Generator, n: int, q: float = 0.10) -> np.ndarray:
    a, b = rng.integers(0, len(P), (2, 200_000))
    cos = np.einsum("ij,ij->i", P[a], P[b])
    thr = np.quantile(cos, q)
    ok = np.flatnonzero((cos < thr) & (a != b))
    pick = rng.choice(ok, n, replace=False)
    return np.stack([a[pick], b[pick]], axis=1)


def interp(P: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return _norm(P[pairs[:, 0]] + P[pairs[:, 1]])


def void(P: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    idx = rng.integers(0, len(P), (N_CANDIDATES, 3))
    w = rng.dirichlet([1.0, 1.0, 1.0], N_CANDIDATES).astype(np.float32)
    C = _norm(np.einsum("nk,nkd->nd", w, P[idx]))
    return C[np.argsort(max_cos(P, C))[:n]]


def far(P: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    C = _norm(rng.standard_normal((N_CANDIDATES, P.shape[1])).astype(np.float32))
    return C[np.argsort(max_cos(P, C))[:n]]


def gap(P: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Most isolated of N_CANDIDATES normalised draws from the corpus's own Gaussian N(mu, Sigma): points inside the
    cap where no item is (docs/design.md, Study 2)."""
    mu = P.mean(axis=0)
    w, U = np.linalg.eigh(np.cov(P.T, bias=True))
    Z = rng.standard_normal((N_CANDIDATES, P.shape[1])) * np.sqrt(np.clip(w, 0, None))
    C = _norm(mu + Z @ U.T).astype(np.float32)
    return C[np.argsort(max_cos(P, C))[:n]]


LADDER_LEVELS = (0.35, 0.45, 0.55, 0.70, 0.85)
K_CROWD = 20


def _isolation(P: np.ndarray, t: np.ndarray, hidden: np.ndarray) -> float:
    s = P @ t
    s[hidden] = -np.inf
    return 1.0 - float(s.max())


def _solve(path, P: np.ndarray, hidden: np.ndarray, level: float, lam_max: float, iters: int = 30):
    """Bisection on lambda so that isolation(path(lambda)) == level. Returns (target, lambda, isolation)."""
    lo, hi = 0.0, lam_max
    if _isolation(P, path(hi), hidden) < level:  # unreachable along this path: take the farthest point
        t = path(hi)
        return t, hi, _isolation(P, t, hidden)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _isolation(P, path(mid), hidden) < level:
            lo = mid
        else:
            hi = mid
    t = path(hi)
    return t, hi, _isolation(P, t, hidden)


def ladder(P: np.ndarray, rng: np.random.Generator, per_level: int, kind: str) -> dict:
    """Targets at fixed isolation levels, each built from a held-out real past item x.

    NOISE: t(l) = norm((1 - l) x + l r), r a random direction (leaves the data cone).
    CROWD: t(l) = norm(x + l (x - c)), c the centroid of x's K_CROWD nearest past items (pushes x away from its
    crowd, staying in the span of real items)."""
    out = {"t": [], "base": [], "hidden": [], "level": [], "lam": [], "iso": []}
    for level in LADDER_LEVELS:
        for _ in range(per_level):
            i = int(rng.integers(len(P)))
            x = P[i]
            sims = P @ x
            hidden = np.flatnonzero(sims > DUP_COS)
            if kind == "NOISE":
                r = _norm(rng.standard_normal(P.shape[1]).astype(np.float32))
                path = lambda l, x=x, r=r: _norm((1 - l) * x + l * r)
                lam_max = 1.0
            else:
                s = sims.copy()
                s[hidden] = -np.inf
                nn = np.argpartition(-s, K_CROWD)[:K_CROWD]
                d = x - P[nn].mean(axis=0)
                path = lambda l, x=x, d=d: _norm(x + l * d)
                lam_max = 50.0
            t, lam, iso = _solve(path, P, hidden, level, lam_max)
            for k, v in zip(out, (t, i, hidden, level, lam, iso)):
                out[k].append(v)
    return out


def sources(tg, arm: str, k: int) -> list[int]:
    """Past corpus rows a target was built from (empty for FAR and PLAIN)."""
    if arm == "REAL":
        return [int(tg["REAL_idx"][k])]
    if arm in ("INTERP", "PAIR"):
        return [int(i) for i in tg[f"{arm}_pairs"][k]]
    if arm in ("NOISE", "CROWD"):
        return [int(tg[f"{arm}_base"][k])]
    return []


def memory_probe(emb: np.ndarray, dates, P: np.ndarray, rng: np.random.Generator, per: int,
                 start: str = "2022-01-01", end: str = "2027-01-01") -> dict:
    """MEM: real items (past or future) sampled per half-year window, to be reconstructed from their visible past
    neighbours. A step in reconstruction quality at a generator's training cutoff measures memory, not skill."""
    import pandas as pd
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    edges = pd.date_range(start, end, freq="6MS")
    rows, wins = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        ok = np.flatnonzero(((d >= a) & (d < b)).to_numpy())
        if len(ok) < per:
            continue
        rows += rng.choice(ok, per, replace=False).tolist()
        wins += [f"{a.year}H{1 if a.month == 1 else 2}"] * per
    t = emb[rows].copy()
    hidden = [np.flatnonzero(P @ x > DUP_COS) for x in t]
    return {"MEM": t, "MEM_row": np.array(rows), "MEM_window": np.array(wins),
            "MEM_hidden": np.array(hidden + [None], dtype=object)[:-1]}
