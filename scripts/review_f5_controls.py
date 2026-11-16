"""F5 with controls, Steam review-count strata and symmetric tails (docs/prereg/review_f5_controls.md).

Local CPU only, no LLM calls. `uv run python scripts/review_f5_controls.py` writes
docs/prereg/review_f5_controls.json.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_f5_controls.json"
REG = json.loads((ROOT / "docs/prereg/where_future_lands_result.json").read_text())
SEED = W.SEED
B = 2000
log = W.log


def ci(v) -> list[float]:
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def boot_p(v, null: float = 0.0) -> float:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(min(1.0, 2 * min(np.mean(v <= null), np.mean(v >= null))))


def dummies(cat: np.ndarray, ref: str) -> np.ndarray:
    levels = [c for c in sorted(set(cat)) if c != ref]
    return np.column_stack([(cat == c).astype(float) for c in levels])


def load(domain: str, embedder: str):
    rng = np.random.default_rng(SEED)
    root = W.DATA / "prepared" / domain
    cols = ["split", "adult", "date"] + (["positive", "negative", "owners", "genres"] if domain == "steam"
                                         else ["decision", "cpc"])
    meta = pd.read_parquet(root / "corpus.parquet", columns=cols)
    emb = np.load(root / f"emb_{embedder}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    past = (meta["split"] == "past").to_numpy()
    P, F = emb[past], emb[~past]
    fm = meta[~past].reset_index(drop=True)
    held = np.zeros(len(P), bool)
    held[rng.choice(len(P), len(P) // 10, replace=False)] = True
    f_max, _ = W.neighbours(F, P[~held])
    return fm, 1 - f_max


def sample(domain: str, fm: pd.DataFrame, iso: np.ndarray):
    if domain == "steam":
        n_rev = (fm["positive"] + fm["negative"]).to_numpy()
        m = n_rev >= 20
        q = fm["positive"].to_numpy()[m] / n_rev[m]
        s = fm[m].reset_index(drop=True)
        genre = s["genres"].fillna("").str.split(",").str[0].to_numpy()
        vc = pd.Series(genre).value_counts()
        genre = np.where(np.isin(genre, vc[vc >= 30].index) & (genre != ""), genre, "Other")
        blocks = {"log_reviews": np.log(n_rev[m])[:, None],
                  "year": pd.to_datetime(s["date"]).dt.year.to_numpy().astype(float)[:, None],
                  "owners": s["owners"].str.split(" - ").str[0].astype(float).to_numpy()[:, None],
                  "genre": dummies(genre, "Action")}
        extra = {"n_reviews": n_rev[m], "genre_levels": sorted(set(genre))}
    else:
        m = fm["decision"].isin(["ACCEPTED", "REJECTED"]).to_numpy()
        q = (fm["decision"].to_numpy()[m] == "ACCEPTED").astype(float)
        s = fm[m].reset_index(drop=True)
        sec = s["cpc"].fillna("").str[:1].replace("", "NONE").to_numpy()
        blocks = {"year": s["date"].astype(str).str[:4].astype(float).to_numpy()[:, None],
                  "cpc_section": dummies(sec, "H")}
        extra = {"cpc_levels": sorted(set(sec))}
    return iso[m], q, blocks, extra


def pr(y, x, X):
    return W.partial_spearman(y, x, X)[0]


def analyse(domain: str, embedder: str) -> dict:
    t0 = time.time()
    fm, iso_all = load(domain, embedder)
    x, q, blocks, extra = sample(domain, fm, iso_all)
    X = np.column_stack(list(blocks.values()))
    n = len(q)

    # reproduction check (registered F5)
    reg = REG[f"{domain}/{embedder}"]["F5"]
    r5, p5 = stats.spearmanr(x, q)
    cut = np.quantile(x, 0.99)
    rep = {"n": n, "rho": float(r5), "p": float(p5), "top1_mean_quality": float(q[x >= cut].mean()),
           "rest_mean_quality": float(q[x < cut].mean())}
    bad = {k: (rep[k], reg[k]) for k in rep if abs(rep[k] - reg[k]) > 1e-9 * max(1.0, abs(reg[k]))}
    if bad:
        raise SystemExit(f"reproduction failed {domain}/{embedder}: {bad}")
    log(f"{domain}/{embedder}: reproduction ok (n {n}, rho {r5:.4f})")

    # C1
    rng = np.random.default_rng(SEED + 33)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    rho_p, p_p = W.partial_spearman(q, x, X)
    b_raw = np.array([stats.spearmanr(x[i], q[i])[0] for i in idx])
    b_par = np.array([pr(q[i], x[i], X[i]) for i in idx])
    c1 = {"raw_rho": float(r5), "raw_ci": ci(b_raw), "partial_rho": float(rho_p), "partial_p": float(p_p),
          "partial_ci": ci(b_par), "attenuation": float(rho_p / r5),
          "partial_by_block": {k: float(pr(q, x, v)) for k, v in blocks.items()}}

    # C2 strata (Steam)
    c2 = None
    if domain == "steam":
        rng = np.random.default_rng(SEED + 44)
        c2 = {}
        nr = extra["n_reviews"]
        for name, lo, hi in (("20-99", 20, 99), ("100-999", 100, 999), (">=1000", 1000, np.inf)):
            m = (nr >= lo) & (nr <= hi)
            xs, qs, Xs = x[m], q[m], X[m]
            keepc = Xs.std(0) > 0  # drop controls constant within the stratum
            Xs = Xs[:, keepc]
            ii = [rng.integers(0, m.sum(), m.sum()) for _ in range(B)]
            c2[name] = {"n": int(m.sum()), "raw_rho": float(stats.spearmanr(xs, qs)[0]),
                        "raw_ci": ci([stats.spearmanr(xs[i], qs[i])[0] for i in ii]),
                        "partial_rho": float(pr(qs, xs, Xs)), "partial_ci": ci([pr(qs[i], xs[i], Xs[i]) for i in ii]),
                        "mean_quality": float(qs.mean())}

    # C3 tails
    rng = np.random.default_rng(SEED + 55)
    RX = np.column_stack([np.ones(n), X])
    q_res = q - RX @ np.linalg.lstsq(RX, q, rcond=None)[0]

    def tails(xx, qq, rr):
        out = {}
        for name, side, level in (("top1", "top", 0.99), ("bottom1", "bottom", 0.01),
                                  ("top10", "top", 0.90), ("bottom10", "bottom", 0.10)):
            c = np.quantile(xx, level)
            t = xx >= c if side == "top" else xx <= c
            out[name] = (qq[t].mean(), qq[~t].mean(), qq[t].mean() - qq[~t].mean(),
                         rr[t].mean() - rr[~t].mean(), int(t.sum()))
        return out

    point = tails(x, q, q_res)
    bt = [tails(x[i], q[i], q_res[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    c3 = {}
    for k, (mt, mr, d, dres, nt) in point.items():
        bd = np.array([b[k][2] for b in bt])
        bres = np.array([b[k][3] for b in bt])
        c3[k] = {"n_tail": nt, "mean_quality_tail": float(mt), "mean_quality_rest": float(mr), "diff": float(d),
                 "diff_ci": ci(bd), "p_boot": boot_p(bd), "diff_residualised": float(dres),
                 "diff_residualised_ci": ci(bres)}
    log(f"  done in {time.time() - t0:.0f}s: partial {rho_p:.4f} {c1['partial_ci']}")
    return {"reproduction": rep, "C1": c1, "C2": c2, "C3": c3, "controls": list(blocks), "levels": {
        k: v for k, v in extra.items() if k != "n_reviews"}, "runtime_s": round(time.time() - t0, 1)}


def decisions(res: dict) -> dict:
    cells = [f"{d}/{e}" for d in ("steam", "patents") for e in W.EMBEDDERS]
    ph = W.holm({c: res[c]["C1"]["partial_p"] for c in cells})
    out = {"C1": {}, "C1_holm": ph, "C2": {}, "C3": {}}
    for d in ("steam", "patents"):
        cs = [f"{d}/{e}" for e in W.EMBEDDERS]
        ok = all(np.sign(res[c]["C1"]["partial_rho"]) == np.sign(res[c]["C1"]["raw_rho"]) and
                 (res[c]["C1"]["partial_ci"][0] > 0 or res[c]["C1"]["partial_ci"][1] < 0) and ph[c] < 0.05 for c in cs)
        if ok and any(abs(res[c]["C1"]["partial_rho"]) < 0.05 for c in cs):
            out["C1"][d] = "survives controls but negligible in size"
        else:
            out["C1"][d] = "survives controls" if ok else "descriptive"
    sgn = {e: np.sign(res[f"steam/{e}"]["C1"]["raw_rho"]) for e in W.EMBEDDERS}
    strata = list(res["steam/qwen3e06"]["C2"])
    out["C2"]["consistent_across_review_volume"] = all(
        np.sign(res[f"steam/{e}"]["C2"][s]["raw_rho"]) == sgn[e] for e in W.EMBEDDERS for s in strata)
    out["C2"]["opposite_sign_ci_excl0_both"] = [s for s in strata if all(
        (res[f"steam/{e}"]["C2"][s]["raw_ci"][1] < 0) if sgn[e] > 0 else (res[f"steam/{e}"]["C2"][s]["raw_ci"][0] > 0)
        for e in W.EMBEDDERS)]
    pt = W.holm({f"{c}/{k}": res[c]["C3"][k]["p_boot"] for c in cells for k in res[c]["C3"]})
    out["C3_holm"] = pt
    for d in ("steam", "patents"):
        for k in res[f"{d}/qwen3e06"]["C3"]:
            cis = [res[f"{d}/{e}"]["C3"][k]["diff_ci"] for e in W.EMBEDDERS]
            diffs = [res[f"{d}/{e}"]["C3"][k]["diff"] for e in W.EMBEDDERS]
            same = all(c[0] > 0 for c in cis) or all(c[1] < 0 for c in cis)
            out["C3"][f"{d}/{k}"] = ("different from the rest" + (" (higher)" if diffs[0] > 0 else " (lower)")
                                     if same else "not distinguishable from the rest")
    return out


def main() -> None:
    res = {}
    for d in ("steam", "patents"):
        for e in W.EMBEDDERS:
            res[f"{d}/{e}"] = analyse(d, e)
            OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    res["decisions"] = decisions(res)
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
