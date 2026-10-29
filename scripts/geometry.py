"""Why decoding far points fails: the geometry of the embedding space against what the decoders produced.

Hypothesis (docs/design.md, E5): text embeddings fill a narrow cone of the sphere, and a decoder can only write texts
inside it, so it recovers the in-cone part of a target and loses the rest. Checks, per generator run:
  G1 corpus shape: mean direction, anisotropy, effective dimension of the past corpus.
  G2 permutation baseline: cos(decoded_i, target_j), j != i, i.e. what an unrelated text would score.
  G3 ladder law: landing ~= c_REAL * cos(t, x), where x is the real item the ladder target was built from.
  G4 in-cone fraction: m(t) = ||U_k^T t|| on the corpus' top-k second-moment directions predicts landing for all arms.
  G5 anchoring: the decoded item sits closer to the neighbours shown in the prompt than the target does.
  G6 efficiency law: landing ~= kappa * ceiling(t), ceiling = best past item's cos to t, kappa fitted on REAL only.
  G7 two-moment ceiling: if a target's scores over the visible corpus (rows not hidden for it) were Gaussian, the
     best of n would sit at mean + sd * z_n. Checked against the observed maximum, against the maximum of a held-out
     half predicted from the other half's moments, and against the scores' upper quantiles.
  G8 extrapolation: the corpus is split in halves A and B; FAR and GAP targets are selected against A, and ceiling(n)
     is measured on subsamples of B, with unselected random directions as control. The corpus size at which the
     fitted curve reaches a given ceiling is a conditional extrapolation, not a measurement; the uniform-sphere line is
     the exact one-expected-hit sample size 1/p_d(c), p_d(c) = I_{1-c^2}((d-1)/2, 1/2) / 2.
  G9 projection onto the cap: for x ~ N(mu, Sigma), E[x | x.t high] moves from mu along Sigma t, so the text that
     best answers a FAR target should sit at mu + lambda Sigma t. Alignment cos(e - mu, Sigma t) of decoded FAR items
     and of the best corpus item, against Sigma t of another FAR target.

    uv run python scripts/geometry.py --domain steam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betainc
from scipy.stats import norm

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUNS = {"pilot": "luna-gw", "grid-gpt-6-luna": "luna", "grid-gpt-6-1-sol": "sol61", "grid-claude-opus-5-5": "opus",
        "grid-claude-sonnet-5-5": "sonnet", "grid-claude-haiku-5-5": "haiku"}
DECODED_ARMS = ("REAL", "INTERP", "NOISE", "CROWD", "FAR", "GAP")
VAR_KEEP = 0.90
SUBSAMPLES = (1000, 2000, 4000, 8000, 16000, 32000)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def corpus_shape(P: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = P.mean(0)
    mu_hat = unit(mu)
    M = (P.T @ P) / len(P)  # uncentred second moment: the geometry cosines live in
    w, U = np.linalg.eigh(M)
    w, U = w[::-1], U[:, ::-1]
    k = int(np.searchsorted(np.cumsum(w) / w.sum(), VAR_KEEP) + 1)
    Pc = P - mu
    wc = np.linalg.eigvalsh((Pc.T @ Pc) / len(P))[::-1]
    a, b = rng.integers(0, len(P), (2, 20000))
    pair = np.einsum("nd,nd->n", P[a], P[b])
    stats = {"d": P.shape[1], "norm_of_mean": float(np.linalg.norm(mu)),
             "cos_to_mean_median": float(np.median(P @ mu_hat)),
             "pair_cos_median": float(np.median(pair)), "pair_cos_p05": float(np.quantile(pair, 0.05)),
             "pair_cos_p95": float(np.quantile(pair, 0.95)),
             "random_pair_cos_sd": float(1 / np.sqrt(P.shape[1])),
             f"k_{int(VAR_KEEP * 100)}": k,
             "participation_ratio_centred": float(wc.sum() ** 2 / (wc ** 2).sum()),
             "top1_share": float(w[0] / w.sum())}
    return mu_hat, U[:, :k], stats


def expected_max(n: int) -> float:
    """Asymptotic approximation to E[max of n iid standard normals]:
    sqrt(2 ln n) - (ln ln n + ln 4 pi) / (2 sqrt(2 ln n)) + gamma / sqrt(2 ln n), gamma the Euler-Mascheroni constant."""
    a = np.sqrt(2 * np.log(n))
    return float(a - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * a) + 0.5772 / a)


def visible_scores(P: np.ndarray, T: np.ndarray, hidden=None) -> list[np.ndarray]:
    """Scores of each target against the corpus rows not hidden for it."""
    S = P @ T.T
    out = []
    for j in range(S.shape[1]):
        s = S[:, j]
        if hidden is not None and len(hidden[j]):
            s = np.delete(s, np.asarray(hidden[j], dtype=np.int64))
        out.append(s)
    return out


def ceilings(P: np.ndarray, T: np.ndarray, hidden=None) -> np.ndarray:
    return np.array([s.max() for s in visible_scores(P, T, hidden)])


def two_moment(scores: list[np.ndarray], rng: np.random.Generator) -> dict:
    """Observed max vs mean + sd z_n on the same rows; held-out half's max from the other half's moments; and the
    standardised upper quantiles of the scores against the Gaussian's."""
    obs, pred, held, held_pred, tails = [], [], [], [], []
    for s in scores:
        obs.append(s.max())
        pred.append(s.mean() + s.std() * expected_max(len(s)))
        idx = rng.permutation(len(s))
        a, b = s[idx[: len(s) // 2]], s[idx[len(s) // 2:]]
        held.append(b.max())
        held_pred.append(a.mean() + a.std() * expected_max(len(b)))
        tails.append([(np.quantile(s, q) - s.mean()) / s.std() for q in (0.99, 0.999, 0.9999)])
    obs, pred, held, held_pred = map(np.array, (obs, pred, held, held_pred))
    return {"ceiling": np.median(obs), "pred": np.median(pred), "mae": np.abs(obs - pred).mean(),
            "r": np.corrcoef(obs, pred)[0, 1], "heldout_mae": np.abs(held - held_pred).mean(),
            "heldout_bias": (held - held_pred).mean(), "z_q": np.median(np.array(tails), axis=0)}


def cap_fraction(d: int, c: float) -> float:
    """Exact fraction of the unit sphere in R^d within cos >= c of a fixed direction."""
    return 0.5 * float(betainc((d - 1) / 2, 0.5, 1 - c ** 2))


def extrapolation(P: np.ndarray, rng: np.random.Generator, reps: int = 5) -> None:
    n, d = P.shape
    idx = rng.permutation(n)
    A, B = P[idx[: n // 2]], P[idx[n // 2:]]
    mu, Sig = A.mean(0), np.cov(A.T, bias=True)
    w, U = np.linalg.eigh(Sig)

    def most_isolated(C: np.ndarray, k: int = 50) -> np.ndarray:
        mc = np.concatenate([(A @ C[i:i + 2000].T).max(0) for i in range(0, len(C), 2000)])
        return C[np.argsort(mc)[:k]]
    R = unit(rng.standard_normal((20000, d))).astype(np.float32)
    G = unit(mu + (rng.standard_normal((20000, d)) * np.sqrt(np.clip(w, 0, None))) @ U.T).astype(np.float32)
    classes = {"RANDOM (unselected uniform directions)": R[:50], "FAR (most isolated of 20k vs half A)": most_isolated(R),
               "GAP (most isolated of 20k normalised N(mu, Sigma) draws vs half A)": most_isolated(G),
               "TYPICAL (normalised N(mu, Sigma) draws)": G[:50]}
    ns = np.array([m for m in SUBSAMPLES if m < len(B)] + [len(B)])
    print(f"\nG8 ceiling(n) on subsamples of half B (n_B = {len(B)}), median over 50 targets, {reps} replicates; "
          "fit ceiling = a + b sqrt(2 ln n) per replicate:")
    print(f"  random direction, Gaussian model: b = sqrt(tr Sigma / d) = {np.sqrt(np.trace(Sig) / d):.4f}")
    for name, T in classes.items():
        curves = np.array([[np.median((B[rng.choice(len(B), m, replace=False)] @ T.T).max(0)) if m < len(B)
                            else np.median((B @ T.T).max(0)) for m in ns] for _ in range(reps)])
        fits = [np.polyfit(np.sqrt(2 * np.log(ns)), c, 1) for c in curves]
        need = {c: [(((c - a) / b) ** 2 / 2) / np.log(10) if c > a else 0.0 for b, a in fits] for c in (0.3, 0.5)}
        b_all = [f[0] for f in fits]
        print(f"  {name}: " + " ".join(f"{v:.3f}" for v in curves.mean(0))
              + f" | b={np.mean(b_all):.4f} [{min(b_all):.4f},{max(b_all):.4f}] | fitted curve reaches "
              + ", ".join(f"{c} at n ~ 10^{np.median(v):.0f} [10^{min(v):.0f},10^{max(v):.0f}]" for c, v in need.items()))
    for c in (0.3, 0.5):
        print(f"  uniform sphere, d={d}: one expected text within cos >= {c} of a given direction needs "
              f"n ~ {1 / cap_fraction(d, c):.2e}")


def load_run(run: Path, arms: tuple[str, ...] = DECODED_ARMS) -> pd.DataFrame:
    rows = []
    for line in (run / "items.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arm"] in arms and r.get("best"):
            rows.append({"arm": r["arm"], "k": r["k"], "nn": r.get("neighbours") or [],
                         "e": unit(np.asarray(r["best"]["emb"], dtype=np.float32))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = {r: g for r, g in RUNS.items() if (DATA / "runs" / args.domain / r / "items.jsonl").exists()}

    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    P = unit(emb[keep][(meta["split"][keep] == "past").to_numpy()])  # stored norms deviate from 1 by < 5e-4
    mu_hat, Uk, stats = corpus_shape(P, rng)
    print("G1 corpus shape:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}))

    tg = np.load(DATA / "runs" / args.domain / "pilot" / "targets.npz", allow_pickle=True)  # the grid reuses them
    T = {arm: tg[arm] for arm in DECODED_ARMS}
    base = {"REAL": tg["REAL_idx"], "NOISE": tg["NOISE_base"], "CROWD": tg["CROWD_base"]}
    m_t = {arm: np.linalg.norm(T[arm] @ Uk, axis=1) for arm in DECODED_ARMS}
    print("\nin-cone fraction of targets m(t), median per arm:",
          {a: round(float(np.median(v)), 3) for a, v in m_t.items()},
          f"| corpus items: {np.median(np.linalg.norm(P[rng.integers(0, len(P), 5000)] @ Uk, axis=1)):.3f}",
          f"| random direction: {np.sqrt(Uk.shape[1] / P.shape[1]):.3f}")

    mu, Sig = P.mean(0), np.cov(P.T, bias=True)
    hid = {a: tg[f"{a}_hidden"] for a in DECODED_ARMS if f"{a}_hidden" in tg.files}
    ceil = {a: ceilings(P, T[a], hid.get(a)) for a in DECODED_ARMS}
    print(f"\nG7 two-moment ceiling on visible rows (z_n = {expected_max(len(P)):.3f} for n = {len(P)}); held-out: "
          "max of a random half predicted from the other half; z_q: standardised score quantiles q = 0.99, 0.999, "
          f"0.9999 (Gaussian {norm.ppf(0.99):.2f}, {norm.ppf(0.999):.2f}, {norm.ppf(0.9999):.2f}):")
    for a in DECODED_ARMS:
        r = two_moment(visible_scores(P, T[a], hid.get(a)), rng)
        print(f"  {a:6s} ceiling={r['ceiling']:.3f} pred={r['pred']:.3f} MAE={r['mae']:.3f} r={r['r']:+.2f} | "
              f"held-out MAE={r['heldout_mae']:.3f} bias={r['heldout_bias']:+.3f} | z_q=" +
              " ".join(f"{v:.2f}" for v in r["z_q"]))

    out, all_rows = [], []
    for run_name, gen in runs.items():
        df = load_run(DATA / "runs" / args.domain / run_name)
        t = np.stack([T[a][k] for a, k in zip(df.arm, df.k)])
        E = np.stack(df.e.to_list())
        df["landing"] = np.einsum("nd,nd->n", E, t)
        df["m_t"] = [m_t[a][k] for a, k in zip(df.arm, df.k)]
        df["m_e"] = np.linalg.norm(E @ Uk, axis=1)
        # Cauchy-Schwarz split into in-cone and out-of-cone parts: an upper bound on landing given both fractions.
        df["bound"] = df.m_t * df.m_e + np.sqrt(np.clip(1 - df.m_t ** 2, 0, 1) * np.clip(1 - df.m_e ** 2, 0, 1))
        df["land_in"] = np.einsum("nd,nd->n", E @ Uk, t @ Uk)  # landing = land_in + land_out, exactly
        df["land_out"] = df.landing - df.land_in
        df["t_mu"], df["e_mu"] = t @ mu_hat, E @ mu_hat
        df["cos_t_base"] = [float(T[a][k] @ P[base[a][k]]) if a in base else np.nan for a, k in zip(df.arm, df.k)]
        C = unit(np.stack([P[nn[:8]].mean(0) for nn in df.nn]))
        df["anchor_decoded"] = np.einsum("nd,nd->n", E, C)
        df["anchor_target"] = np.einsum("nd,nd->n", t, C)
        df["gen"] = gen
        df["ceiling"] = [float(ceil[a][k]) for a, k in zip(df.arm, df.k)]
        kappa = float((df[df.arm == "REAL"].landing / df[df.arm == "REAL"].ceiling).median())
        df["kappa"], df["eff_pred"] = kappa, kappa * df.ceiling
        # G2: score of the decoded item against the other targets of the same arm.
        for arm, g in df.groupby("arm"):
            idx = g.index.to_numpy()
            S = E[idx] @ t[idx].T
            off = S[~np.eye(len(idx), dtype=bool)]
            df.loc[idx, "perm_baseline"] = float(np.median(off))
        c_real = float(df[df.arm == "REAL"].landing.median())
        df["ladder_pred"] = c_real * df.cos_t_base
        all_rows.append(df.drop(columns=["e", "nn"]))

        for arm, g in df.groupby("arm", sort=False):
            rec = {"gen": gen, "arm": arm, "landing": g.landing.median(), "perm_baseline": g.perm_baseline.iloc[0],
                   "m_t": g.m_t.median(), "m_e": g.m_e.median(), "bound": g.bound.median(),
                   "anchor_decoded": g.anchor_decoded.median(), "anchor_target": g.anchor_target.median(),
                   "land_in": g.land_in.median(), "land_out": g.land_out.median(),
                   "t_mu": g.t_mu.median(), "e_mu": g.e_mu.median()}
            rec.update(kappa=g.kappa.iloc[0], ceiling=g.ceiling.median(), eff_mae=float(np.abs(g.eff_pred - g.landing).mean()),
                       eff_r=np.corrcoef(g.eff_pred, g.landing)[0, 1])
            if arm in ("NOISE", "CROWD"):
                rec["ladder_pred"] = g.ladder_pred.median()
                rec["ladder_r"] = np.corrcoef(g.ladder_pred, g.landing)[0, 1]
                rec["ladder_mae"] = float(np.abs(g.ladder_pred - g.landing).mean())
            out.append(rec)

    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("\nG2-G5 per generator and arm (medians):")
    print(res.round(3).to_string(index=False))

    allr = pd.concat(all_rows)
    allr.to_csv(DATA / "runs" / args.domain / "geometry_items.csv", index=False)
    print("\nG4 across all decoded items and generators:")
    for gen, g in allr.groupby("gen", sort=False):
        X = np.c_[g.m_t, np.ones(len(g))]
        beta, *_ = np.linalg.lstsq(X, g.landing, rcond=None)
        pred = X @ beta
        r2 = 1 - ((g.landing - pred) ** 2).sum() / ((g.landing - g.landing.mean()) ** 2).sum()
        print(f"  {gen:8s} landing ~ {beta[0]:.2f} m(t) {beta[1]:+.2f}  R2={r2:.2f}  "
              f"| landing <= bound in {(g.landing <= g.bound + 1e-6).mean():.0%}")
    print("\nG4 within arm (pooled generators): r(landing, m(t)) and r(landing, t.mu)")
    for arm, g in allr.groupby("arm", sort=False):
        print(f"  {arm:6s} r_m={np.corrcoef(g.m_t, g.landing)[0, 1]:+.2f}  r_mu={np.corrcoef(g.t_mu, g.landing)[0, 1]:+.2f}")
    lad = allr[allr.arm.isin(["NOISE", "CROWD"])]
    print("\nG3 ladder law (pooled NOISE+CROWD): r =", round(np.corrcoef(lad.ladder_pred, lad.landing)[0, 1], 3),
          " MAE =", round(float(np.abs(lad.ladder_pred - lad.landing).mean()), 3))
    print("G5 anchoring: decoded closer to the shown neighbours than the target is in",
          f"{(allr.anchor_decoded > allr.anchor_target).mean():.0%} of items;",
          "by arm:", allr.assign(a=allr.anchor_decoded > allr.anchor_target).groupby("arm").a.mean().round(2).to_dict())
    print("\nG6 efficiency law vs ladder law, mean over generators (MAE; r within arm):")
    print(res[res.arm.isin(["NOISE", "CROWD", "FAR"])].groupby("arm")[["eff_mae", "eff_r", "ladder_mae", "ladder_r"]]
          .mean().round(3).to_string())
    print("\nper-item spread (pooled generators): m(e) min/p05/median/p95/max, and out-of-slab part of landing")
    q = lambda x: " ".join(f"{v:.3f}" for v in np.quantile(x, [0, 0.05, 0.5, 0.95, 1]))
    print(f"  m(e), decoded arms: {q(allr.m_e)}")
    extra = pd.concat([load_run(DATA / "runs" / args.domain / r, ("PLAIN", "PAIR")).assign(gen=g) for r, g in runs.items()])
    extra["m_e"] = np.linalg.norm(np.stack(extra.e.to_list()) @ Uk, axis=1)
    print("  m(e) median, PLAIN/PAIR by generator:", extra.groupby(["arm", "gen"]).m_e.median().round(3).to_dict())
    print(f"  land_out, all decoded: {q(allr.land_out)}")
    far = allr[allr.arm == "FAR"]
    print(f"  FAR: land_out {q(far.land_out)}; |land_out| / (|land_in| + |land_out|) median "
          f"{np.median(far.land_out.abs() / (far.land_in.abs() + far.land_out.abs())):.2f}")
    print("\nG6 by ladder level (pooled generators): MAE and r within level, kappa*ceiling vs ladder law")
    for arm in ("NOISE", "CROWD"):
        lev = dict(zip(range(len(tg[f"{arm}_level"])), tg[f"{arm}_level"]))
        g = allr[allr.arm == arm].assign(level=lambda d: d.k.map(lev))
        for level, h in g.groupby("level"):
            print(f"  {arm:5s} l={level:.2f} n={len(h):3d} eff MAE={np.abs(h.eff_pred - h.landing).mean():.3f} "
                  f"r={np.corrcoef(h.eff_pred, h.landing)[0, 1]:+.2f} | ladder MAE={np.abs(h.ladder_pred - h.landing).mean():.3f} "
                  f"r={np.corrcoef(h.ladder_pred, h.landing)[0, 1]:+.2f}")
    St = T["FAR"] @ Sig
    perm = rng.permutation(len(St))

    def align(X: np.ndarray, S: np.ndarray) -> float:
        return float(np.median(np.einsum("nd,nd->n", unit(X - mu), unit(S))))
    best = P[(P @ T["FAR"].T).argmax(0)]
    print(f"\nG9 FAR projection, cos(x - mu, Sigma t): best corpus item {align(best, St):.3f} "
          f"(other target {align(best, St[perm]):.3f})")
    for run_name, gen in runs.items():
        df = load_run(DATA / "runs" / args.domain / run_name)
        df = df[df.arm == "FAR"]
        X, ks = np.stack(df.e.to_list()), df.k.to_numpy()
        print(f"  {gen:8s} decoded {align(X, St[ks]):.3f} (other target {align(X, St[perm][ks]):.3f})")
    extrapolation(P, rng)


if __name__ == "__main__":
    main()
