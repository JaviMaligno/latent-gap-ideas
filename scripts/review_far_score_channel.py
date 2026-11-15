"""R3 of docs/prereg/review_far_decoding.md: does the score feedback alone steer towards the OWN target?

In the `none` condition round 0 sees no target information and no feedback; rounds 1-3 add only score messages, and
every round keeps its best-of-3 by own-target cosine. Round 0 is a same-prompt, same-selection no-feedback control.

S1 = max(candidate_sims r3) - max(candidate_sims r0)          (raw space only: other candidates' embeddings are not stored)
m_r = e_r.t_k - mean_{j != k} e_r.t_j  (19 other targets, same domain and arm)
S2 = m_3 - m_0 (primary), S2' = mean(m_1..m_3) - m_0, selection-only margin m_0, final `best` margin.
Pooled over decoders per target (average of the decoders that have the target), percentile bootstrap over targets
(B = 10,000, one generator seeded 20261114 + 30, family order steam/arxiv x REAL/GAP/FAR, one (B, n) draw per set used
for every quantity of that set), Holm over the 6 one-sided S2 > 0 tests, p = share(S2* <= 0) floored at 1/B.
Raw Qwen3 is primary; T1 and T3 of R1 (point map for targets) are sensitivity; the `true` condition is descriptive.

    uv run python scripts/review_far_score_channel.py                # -> docs/prereg/review_far_score_channel.json
    uv run python scripts/review_far_score_channel.py --boot 200 --spaces T0 --out /tmp/s.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from review_far_whitening import Transform, load_split, unit
from review_far_units import holm

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_far_score_channel.json"
SEED = 20261114 + 30
B = 10_000
DOMS, ARMS, GENS, KS = ("steam", "arxiv"), ("REAL", "GAP", "FAR"), ("luna", "sonnet"), list(range(1, 21))
FAR_MIN = 0.02  # FAR minimum effect registered in anchoring_causal.md


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def per_target(dom: str, cond: str, f, T: dict) -> tuple[dict, dict]:
    """Per (arm, gen, k): S1, m_0..m_3 (None if the round is missing), m_best. Exclusion counts."""
    out, excluded = {}, {}
    for arm in ARMS:
        Tm = f(T[arm][1:21]).astype(np.float64)  # targets k = 1..20 in the space of f (point map)
        for gen in GENS:
            n_ex = 0
            for r in read_rows(DATA / f"runs/{dom}/anchoring-causal/{gen}/{cond}/{arm}/items.jsonl"):
                k = r["k"]
                if k not in KS:
                    continue
                h = {x["iter"]: x for x in r["history"] if x.get("sim") is not None}
                if 0 not in h or 3 not in h:
                    n_ex += 1
                    continue
                i = k - 1

                def margin(emb):
                    s = f(np.asarray(emb)[None])[0].astype(np.float64) @ Tm.T
                    return float(s[i] - np.delete(s, i).mean())

                m = {rr: margin(h[rr]["emb"]) for rr in range(4) if rr in h}
                out[arm, gen, k] = {"S1": max(h[3]["candidate_sims"]) - max(h[0]["candidate_sims"]),
                                    "m": m, "m_best": margin(r["best"]["emb"])}
            excluded[f"{arm}/{gen}"] = n_ex
    return out, excluded


def analyse(rows: dict, n_boot: int, raw: bool) -> dict:
    rng = np.random.default_rng(SEED)
    sets, pvals = {}, {}
    for dom in DOMS:
        for arm in ARMS:
            pooled = []
            for k in KS:
                g = [rows[dom].get((arm, gen, k)) for gen in GENS]
                g = [x for x in g if x]
                if not g:
                    continue
                q = {"S2": np.mean([x["m"][3] - x["m"][0] for x in g]),
                     "S2p": np.mean([np.mean([x["m"][r] for r in (1, 2, 3) if r in x["m"]]) - x["m"][0] for x in g]),
                     "m0": np.mean([x["m"][0] for x in g]), "m_best": np.mean([x["m_best"] for x in g]),
                     "S1": np.mean([x["S1"] for x in g]), "n_dec": len(g)}
                pooled.append(q)
            if not pooled:
                continue
            V = {key: np.array([p[key] for p in pooled]) for key in ("S2", "S2p", "m0", "m_best", "S1")}
            idx = rng.integers(0, len(pooled), (n_boot, len(pooled)))
            res = {"n_targets": len(pooled), "n_single_decoder": int(sum(p["n_dec"] == 1 for p in pooled))}
            for key, v in V.items():
                if key == "S1" and not raw:
                    continue
                b = v[idx].mean(1)
                res[key] = {"mean": float(v.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                            "upper95_one_sided": float(np.percentile(b, 95))}
                if key == "S2":
                    pvals[f"{dom}/{arm}"] = max(float((b <= 0).mean()), 1 / n_boot)
            sets[f"{dom}/{arm}"] = res
    h = holm(pvals)
    for k in sets:
        sets[k]["S2_test"] = h[k]
    far = [sets[f"{d}/FAR"] for d in DOMS if f"{d}/FAR" in sets]
    rej = any(s["S2_test"]["reject"] for s in far)
    ub = max(s["S2"]["ci95"][1] for s in far) if far else None
    verdict = ("counts against 'the score alone does not move text towards a far point'" if rej else
               "no detectable steering by the score alone" if ub is not None and ub < FAR_MIN else "inconclusive")
    return {"sets": sets, "FAR_verdict": verdict, "FAR_max_upper95": ub}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--spaces", default="T0,T1,T3")
    args = ap.parse_args()
    if (args.boot != B or args.spaces != "T0,T1,T3") and args.out == OUT:
        raise SystemExit("a reduced run must not write the registered JSON")
    T = {}
    for dom in DOMS:
        with np.load(DATA / f"runs/{dom}/pilot/targets.npz", allow_pickle=True) as z:
            T[dom] = {a: np.asarray(z[a], dtype=np.float32) for a in ARMS}
    res = {"prereg": "docs/prereg/review_far_decoding.md#R3", "B": args.boot, "seed": SEED, "spaces": {}}
    spaces = args.spaces.split(",")
    for space in spaces:
        per = {"none": {}, "true": {}}
        excl = {}
        for dom in DOMS:
            if space == "T0":
                f = Transform("T0", np.zeros((1, 1024), np.float32))
            else:
                X, _ = load_split(dom, "emb_qwen3e06.npy")
                f = Transform(space, X)
                del X
            for cond in per:
                per[cond][dom], excl[f"{dom}/{cond}"] = per_target(dom, cond, f, T[dom])
        res["spaces"][space] = {"excluded": excl, "none": analyse(per["none"], args.boot, space == "T0"),
                                "true_descriptive": analyse(per["true"], args.boot, space == "T0")}
        s = res["spaces"][space]["none"]
        print(f"{space} none: FAR verdict = {s['FAR_verdict']} (max upper95 {s['FAR_max_upper95']:.4f})")
        for k, v in s["sets"].items():
            print(f"  {k}: S2 {v['S2']['mean']:+.4f} {np.round(v['S2']['ci95'], 4)} m0 {v['m0']['mean']:+.4f} "
                  f"m_best {v['m_best']['mean']:+.4f} p {v['S2_test']['p']:.4f} holm {v['S2_test']['reject']}"
                  + (f" S1 {v['S1']['mean']:+.4f}" if "S1" in v else ""))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
