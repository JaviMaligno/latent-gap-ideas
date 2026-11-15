"""R2 of docs/prereg/review_far_decoding.md: targets, not decoder cells, as the independent units of Table A2.

6 sets (Steam, arXiv x REAL, GAP, FAR) of 20 targets shared by the two decoders. Per set: decoder-pooled D_k and
A_k, percentile bootstrap over targets (B = 10,000, one generator seeded 20261114 + 20, sets in the order
steam/arxiv x REAL/GAP/FAR, one (B, 20) index draw per set shared by D and A), number of targets with A_k > 0 for both
decoders, Spearman rho of D_k^luna vs D_k^Sonnet, Holm over the 6 one-sided p = share(mean A* <= 0), floored at 1/B.

Primary: raw space, DATA/runs/anchoring-causal-analysis/items.csv. Sensitivity: each corrected space T1-T4 (point
map) from the W3 per-item table of scripts/review_far_whitening.py, each transformation its own Holm family.

    uv run python scripts/review_far_units.py              # -> docs/prereg/review_far_units.json
    uv run python scripts/review_far_units.py --out /tmp/u.json --boot 200 --w3-items /tmp/x.items.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_far_units.json"
ITEMS = DATA / "runs/anchoring-causal-analysis/items.csv"
W3_ITEMS = DATA / "runs/review-far-decoding/w3_items.csv"
SEED = 20261114 + 20
B = 10_000
DOMS, ARMS, GENS, KS = ("steam", "arxiv"), ("REAL", "GAP", "FAR"), ("luna", "sonnet"), list(range(1, 21))


def holm(p: dict, alpha: float = 0.05) -> dict:
    order = sorted(p, key=p.get)
    out, stop = {}, False
    for i, k in enumerate(order):
        adj = min(1.0, max(p[k] * (len(p) - i), max([out[o]["p_holm"] for o in order[:i]], default=0)))
        stop = stop or p[k] > alpha / (len(p) - i)
        out[k] = {"p": p[k], "p_holm": adj, "reject": not stop}
    return out


def per_target_raw() -> pd.DataFrame:
    it = pd.read_csv(ITEMS)
    w = it.pivot_table(index=["domain", "arm", "gen", "k"], columns="condition", values=["target", "attraction"])
    df = pd.DataFrame({"D": w["target"]["true"] - w["target"]["swap"],
                       "A": w["attraction"]["swap"] - w["attraction"]["true"]}).reset_index()
    return df


def per_target_w3(path: Path, tname: str) -> pd.DataFrame:
    it = pd.read_csv(path)
    it = it[(it["transform"] == tname) & (it["map"] == "point")]
    return it[["domain", "arm", "gen", "k", "D", "A"]].copy()


def analyse(df: pd.DataFrame, n_boot: int) -> dict:
    rng = np.random.default_rng(SEED)
    sets, pvals = {}, {}
    for dom in DOMS:
        for arm in ARMS:
            s = df[(df.domain == dom) & (df.arm == arm)]
            if s.empty:
                continue
            w = s.pivot_table(index="k", columns="gen", values=["D", "A"]).dropna()
            ks = w.index.to_numpy()
            Dp = ((w["D"]["luna"] + w["D"]["sonnet"]) / 2).to_numpy()
            Ap = ((w["A"]["luna"] + w["A"]["sonnet"]) / 2).to_numpy()
            idx = rng.integers(0, len(ks), (n_boot, len(ks)))
            bD, bA = Dp[idx].mean(1), Ap[idx].mean(1)
            ciD = [float(np.percentile(bD, 2.5)), float(np.percentile(bD, 97.5))]
            ciA = [float(np.percentile(bA, 2.5)), float(np.percentile(bA, 97.5))]
            p = max(float((bA <= 0).mean()), 1 / n_boot)
            rho = spearmanr(w["D"]["luna"], w["D"]["sonnet"]).statistic
            key = f"{dom}/{arm}"
            pvals[key] = p
            sets[key] = {"n_targets": int(len(ks)), "D_pooled_mean": float(Dp.mean()), "D_ci95": ciD,
                         "D_ci_halfwidth": (ciD[1] - ciD[0]) / 2, "A_pooled_mean": float(Ap.mean()), "A_ci95": ciA,
                         "n_targets_A_pos_both": int(((w["A"]["luna"] > 0) & (w["A"]["sonnet"] > 0)).sum()),
                         "spearman_D_luna_sonnet": float(rho), "decoders_same_confirmation": bool(rho >= 0.5),
                         "p_one_sided_A": p}
    h = holm(pvals)
    for k in sets:
        sets[k].update(h[k])
    n_rej = sum(v["reject"] for v in sets.values())
    return {"sets": sets, "n_reject_holm": n_rej, "n_sets": len(sets),
            "verdict": ("restated in independent units: 6 of 6 sets" if n_rej == 6 == len(sets)
                        else f"counts against '12 of 12': {n_rej} of {len(sets)} independent target sets")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--w3-items", type=Path, default=W3_ITEMS)
    args = ap.parse_args()
    if args.boot != B and args.out == OUT:
        raise SystemExit("a reduced --boot must not write the registered JSON")
    res = {"prereg": "docs/prereg/review_far_decoding.md#R2", "B": args.boot, "seed": SEED,
           "T0": analyse(per_target_raw(), args.boot)}
    if args.w3_items.exists():
        for t in ("T1", "T2", "T3", "T4"):
            d = per_target_w3(args.w3_items, t)
            if not d.empty:
                res[t] = analyse(d, args.boot)
    else:
        res["corrected_spaces"] = f"missing {args.w3_items}: run scripts/review_far_whitening.py first"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    for t in ("T0", "T1", "T2", "T3", "T4"):
        if t in res:
            print(t, res[t]["verdict"])
    for k, v in res["T0"]["sets"].items():
        print(f"  {k}: D {v['D_pooled_mean']:+.3f} {np.round(v['D_ci95'], 3)} A {v['A_pooled_mean']:.3f} "
              f"{np.round(v['A_ci95'], 3)} A>0 both {v['n_targets_A_pos_both']}/20 rho {v['spearman_D_luna_sonnet']:.2f} "
              f"p {v['p']:.4f} holm {v['reject']}")


if __name__ == "__main__":
    main()
