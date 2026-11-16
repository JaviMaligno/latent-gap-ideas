"""Item-level test of "new work lands in empty regions", no cells (docs/prereg/review_item_density.md).

Local CPU only, no LLM calls. `uv run python scripts/review_item_density.py [domain/embedder ...]`
writes docs/prereg/review_item_density.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_future_lands as W  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/prereg/review_item_density.json"
SEED = W.SEED
B = 2000
N_PROBES = 20_000
EXPECTED = {"steam": (61400, 6822, 40989), "arxiv": (64800, 7200, 41000), "patents": (64369, 7152, 40342)}
log = W.log


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = stats.rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(v) -> list[float]:
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def boot_p(v, null: float) -> float:
    v = np.asarray(v, float)
    return float(min(1.0, 2 * min(np.mean(v <= null), np.mean(v >= null))))


def load(domain: str, embedder: str):
    rng = np.random.default_rng(SEED)
    root = W.DATA / "prepared" / domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date"])
    emb = np.load(root / f"emb_{embedder}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    pm = meta[past].reset_index(drop=True)
    held = np.zeros(len(P), bool)
    held[rng.choice(len(P), len(P) // 10, replace=False)] = True
    R, H = P[~held], P[held]
    py = W.years(pm["date"], domain)
    sizes = (len(R), len(H), len(F))
    if sizes != EXPECTED[domain]:
        raise SystemExit(f"size mismatch {domain}/{embedder}: {sizes} != {EXPECTED[domain]}")
    return R, H, F, py[~held], py[held]


def auc_block(fs: dict, hs: dict, recent: np.ndarray, rng) -> dict:
    """fs/hs: score name -> array (higher = predicted future). Bootstrap F and H separately."""
    out = {}
    nf, nh = len(next(iter(fs.values()))), len(recent)
    idx = [(rng.integers(0, nf, nf), rng.integers(0, nh, nh)) for _ in range(B)]
    for name in fs:
        f, h = fs[name], hs[name]
        a_all, a_rec = auc(f, h), auc(f, h[recent])
        b_all, b_rec = [], []
        for fi, hi in idx:
            hb = hi
            b_all.append(auc(f[fi], h[hb]))
            b_rec.append(auc(f[fi], h[hb[recent[hb]]]))
        b_all, b_rec = np.array(b_all), np.array(b_rec)
        out[name] = {
            "H_all": {"auc": a_all, "ci": ci(b_all),
                      "p_one_sided": float(stats.mannwhitneyu(f, h, alternative="greater").pvalue),
                      "n_future": int(len(f)), "n_control": int(len(h))},
            "H_recent": {"auc": a_rec, "ci": ci(b_rec),
                         "p_one_sided": float(stats.mannwhitneyu(f, h[recent], alternative="greater").pvalue),
                         "n_future": int(len(f)), "n_control": int(recent.sum())},
            "delta_all_minus_recent": {"value": a_all - a_rec, "ci": ci(b_all - b_rec)},
        }
    return out


def analyse(domain: str, embedder: str) -> dict:
    t0 = time.time()
    R, H, F, ry, hy = load(domain, embedder)
    lo, hi = W.MOMENTUM_YEARS[domain]
    recent = (hy >= lo) & (hy <= hi)
    log(f"{domain}/{embedder}: R {len(R)} H {len(H)} (recent {recent.sum()}) F {len(F)}")

    f_max, f_dens = W.neighbours(F, R)
    h_max, h_dens = W.neighbours(H, R)
    log("  item measures done")

    # A1
    a1 = auc_block({"iso": 1 - f_max, "neg_dens": -f_dens}, {"iso": 1 - h_max, "neg_dens": -h_dens},
                   recent, np.random.default_rng(SEED + 11))
    a1["median"] = {"iso_future": float(np.median(1 - f_max)), "iso_heldout": float(np.median(1 - h_max)),
                    "iso_heldout_recent": float(np.median(1 - h_max[recent])),
                    "dens_future": float(np.median(f_dens)), "dens_heldout": float(np.median(h_dens)),
                    "dens_heldout_recent": float(np.median(h_dens[recent]))}
    # A1c backtest inside the past
    R_early = R[ry < lo]
    early = hy < lo
    pr_max, pr_dens = W.neighbours(H[recent], R_early)
    pe_max, pe_dens = W.neighbours(H[early], R_early)
    rng = np.random.default_rng(SEED + 12)
    a1c = {"n_reference_early": int(len(R_early)), "n_pseudo_future": int(recent.sum()), "n_pseudo_control": int(early.sum())}
    for name, f, h in (("iso", 1 - pr_max, 1 - pe_max), ("neg_dens", -pr_dens, -pe_dens)):
        bs = [auc(f[rng.integers(0, len(f), len(f))], h[rng.integers(0, len(h), len(h))]) for _ in range(B)]
        a1c[name] = {"auc": auc(f, h), "ci": ci(bs)}
    a1["A1c_backtest"] = a1c
    log(f"  A1 done: iso AUC all {a1['iso']['H_all']['auc']:.3f} recent {a1['iso']['H_recent']['auc']:.3f}")

    # A2 probes
    rng = np.random.default_rng(SEED + 200)
    mu = R.mean(0)
    w, U = np.linalg.eigh(np.cov(R.T, bias=True))
    Z = rng.standard_normal((N_PROBES, R.shape[1])).astype(np.float32) * np.sqrt(np.clip(w, 0, None)).astype(np.float32)
    C = mu + Z @ U.T.astype(np.float32)
    del Z
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    tau = W.kth_cos(C, R)
    _, c_dens = W.neighbours(C, R)
    e = 1 - tau
    rngw = np.random.default_rng(SEED + 22)
    WF = np.empty((len(F), B), np.float32)
    for j in range(0, B, 250):
        WF[:, j:j + 250] = rngw.poisson(1.0, (len(F), min(250, B - j)))
    WH = rngw.poisson(1.0, (len(H), B)).astype(np.float32)
    cF, cH = np.empty(N_PROBES, np.int64), np.empty(N_PROBES, np.int64)
    cFw, cHw = np.empty((N_PROBES, B), np.float32), np.empty((N_PROBES, B), np.float32)
    for i in range(0, N_PROBES, 500):
        t = tau[i:i + 500, None]
        mF = ((C[i:i + 500] @ F.T) > t).astype(np.float32)
        cF[i:i + 500], cFw[i:i + 500] = mF.sum(1), mF @ WF
        mH = ((C[i:i + 500] @ H.T) > t).astype(np.float32)
        cH[i:i + 500], cHw[i:i + 500] = mH.sum(1), mH @ WH
    sWF, sWH = WF.sum(0), WH.sum(0)
    del WF, WH, mF, mH
    log("  probe counts done")

    q30, q70, q90 = np.quantile(e, [0.3, 0.7, 0.9])
    emp, mid = e >= q90, (e >= q30) & (e <= q70)

    def pooled(cf, ch, nf, nh, m):
        sf, sh = cf[m].sum(), ch[m].sum()
        sh = sh if sh > 0 else 0.5
        return (sf / nf) / (sh / nh)

    rr = ((cF + 0.5) / len(F)) / ((cH + 0.5) / len(H))
    rr_emp, rr_mid = pooled(cF, cH, len(F), len(H), emp), pooled(cF, cH, len(F), len(H), mid)
    ror = rr_emp / rr_mid
    rho, p_rho = stats.spearmanr(e, np.log(rr))
    rngp = np.random.default_rng(SEED + 23)
    b_ror, b_rho = np.empty(B), np.empty(B)
    for b in range(B):
        pi = rngp.integers(0, N_PROBES, N_PROBES)
        cf, ch = cFw[pi, b], cHw[pi, b]
        b_ror[b] = pooled(cf, ch, sWF[b], sWH[b], emp[pi]) / pooled(cf, ch, sWF[b], sWH[b], mid[pi])
        b_rho[b] = stats.spearmanr(e[pi], np.log(((cf + 0.5) / sWF[b]) / ((ch + 0.5) / sWH[b])))[0]
    deciles = np.quantile(e, np.linspace(0, 1, 11))
    dec_id = np.clip(np.searchsorted(deciles, e, side="right") - 1, 0, 9)
    a2 = {"n_probes": N_PROBES,
          "tau_quantiles": {str(q): float(np.quantile(tau, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
          "heldout_tau_median": None,
          "median_count_future": float(np.median(cF)), "median_count_heldout": float(np.median(cH)),
          "share_probes_heldout_zero": float((cH == 0).mean()),
          "rr_emptiest_decile": float(rr_emp), "rr_middle_deciles": float(rr_mid),
          "ratio_of_ratios": float(ror), "ror_ci": ci(b_ror), "ror_p_boot": boot_p(np.log(b_ror), 0.0),
          "spearman_e_logrr": float(rho), "spearman_p": float(p_rho), "spearman_ci": ci(b_rho),
          "spearman_p_boot": boot_p(b_rho, 0.0),
          "rr_by_decile": [float(pooled(cF, cH, len(F), len(H), dec_id == d)) for d in range(10)],
          "decile_edges_emptiness": [float(x) for x in deciles]}

    # A2c density-matched held-out centres for the emptiest-decile probes
    h_tau = W.kth_cos(H, R)
    h_cnt = W.count_above(H, F, h_tau)
    a2["heldout_tau_median"] = float(np.median(h_tau))
    rngm = np.random.default_rng(SEED + 66)
    pct, n_exact = [], 0
    for d_, c_ in zip(c_dens[emp], cF[emp]):
        near = np.where(np.abs(h_dens - d_) <= 0.01)[0]
        if len(near) >= 200:
            n_exact += 1
            near = rngm.choice(near, 200, replace=False)
        else:
            near = np.argsort(np.abs(h_dens - d_))[:200]
        pct.append(stats.percentileofscore(h_cnt[near], c_, kind="mean"))
    a2["A2c"] = {"n_probes": int(emp.sum()), "matched_within_001": n_exact, "median_percentile": float(np.median(pct)),
                 "median_probe_dens": float(np.median(c_dens[emp])), "median_heldout_dens": float(np.median(h_dens)),
                 "min_heldout_dens": float(h_dens.min())}
    log(f"  A2 done: RoR {ror:.3f} rho {rho:.3f}; {time.time() - t0:.0f}s")
    return {"A1": a1, "A2": a2, "sizes": {"reference": len(R), "heldout": len(H), "future": len(F),
                                          "heldout_recent": int(recent.sum())},
            "runtime_s": round(time.time() - t0, 1)}


def a1_verdict(r: dict, p_holm: float) -> str:
    a, (lo, hi) = r["auc"], r["ci"]
    if a >= 0.60 and lo >= 0.55 and p_holm < 0.05:
        return "supports"
    if hi < 0.55:
        return "contradicts"
    if lo > 0.50:
        return "weak"
    return "inconclusive"


def a2_verdict(r: dict, p_holm: float) -> str:
    lo, hi = r["ror_ci"]
    if r["ratio_of_ratios"] >= 1.25 and lo > 1 and p_holm < 0.05 and r["spearman_ci"][0] > 0:
        return "supports"
    if hi < 1.10:
        return "contradicts"
    return "inconclusive"


def decisions(res: dict) -> dict:
    cells = [k for k in res if "/" in k]
    out = {}
    for score in ("iso", "neg_dens"):
        for comp in ("H_all", "H_recent"):
            fam = f"A1_{score}_{comp}"
            ph = W.holm({c: res[c]["A1"][score][comp]["p_one_sided"] for c in cells})
            out[fam] = {c: {"auc": res[c]["A1"][score][comp]["auc"], "ci": res[c]["A1"][score][comp]["ci"],
                            "p_holm": ph[c], "verdict": a1_verdict(res[c]["A1"][score][comp], ph[c])} for c in cells}
    ph = W.holm({c: res[c]["A2"]["ror_p_boot"] for c in cells})
    phr = W.holm({c: res[c]["A2"]["spearman_p_boot"] for c in cells})
    out["A2"] = {c: {"ror": res[c]["A2"]["ratio_of_ratios"], "ror_ci": res[c]["A2"]["ror_ci"], "p_holm": ph[c],
                     "rho": res[c]["A2"]["spearman_e_logrr"], "rho_ci": res[c]["A2"]["spearman_ci"],
                     "rho_p_holm": phr[c], "verdict": a2_verdict(res[c]["A2"], ph[c])} for c in cells}
    rep = {}
    for d in W.DOMAINS:
        for fam in out:
            v = [out[fam].get(f"{d}/{e}", {}).get("verdict") for e in W.EMBEDDERS]
            rep[f"{d}/{fam}"] = v[0] if v[0] == v[1] and v[0] is not None else f"mixed ({v[0]} / {v[1]})"
    out["replicated_by_domain"] = rep
    return out


def main() -> None:
    which = sys.argv[1:] or [f"{d}/{e}" for e in W.EMBEDDERS for d in W.DOMAINS]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key in which:
        d, e = key.split("/")
        res[key] = analyse(d, e)
        OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if all(f"{d}/{e}" in res for d in W.DOMAINS for e in W.EMBEDDERS):
        res["decisions"] = decisions(res)
        OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
