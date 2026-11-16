"""Step gap against citing papers first posted April-December 2023 (docs/prereg/review_olmo_2023_followups.md).

Read-only on saved vectors: contexts, 2024+ follow-ups and ideas from continuation_reanalysis.load_generator; the 2023
follow-ups from kaggle-output/review-olmo-2023-embed (kernel kaggle/review-olmo-2023-embed, same encoder recipe as the
follow-ups). No model is called.

Output: docs/prereg/review_olmo_2023_followups.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuation_reanalysis import load_generator  # noqa: E402
from step_gap_by_year import fe_slope  # noqa: E402

D = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/review_olmo_2023_followups.json"
B = 10_000
B_SLOPE = 2000
SEED = 20261115
GENS = ("olmo-7b-0424", "gpt-4o")
SPACES = ("qwen3e06", "bgem3")
YEARS = (2024, 2025, 2026)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def ci_of(m: np.ndarray) -> list[float]:
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summary(x: np.ndarray, idx: np.ndarray) -> dict:
    return {"n_contexts": int(len(x)), "mean": float(x.mean()), "ci95": ci_of(x[idx].mean(1))}


def v1_year(aid: str) -> int:
    return 2000 + int(aid[:2])  # new-style arXiv id: YYMM of the v1 submission


def main() -> None:
    rows = sorted((json.loads(l) for l in (D / "runs/review-olmo-2023/followups_2023.jsonl").read_text().splitlines()),
                  key=lambda r: r["ctx"])
    keys = pd.read_parquet(D / "kaggle-upload/review-olmo-2023-texts/texts.parquet")["key"].tolist()
    res: dict = {"n_boot": B, "seed": SEED, "window": ["2023-04-01", "2023-12-31"],
                 "fetch": {"contexts": len(rows), "with_ge1": sum(r["n_followups"] >= 1 for r in rows),
                           "with_ge5": sum(r["n_followups"] >= 5 for r in rows),
                           "total_followups_2023": sum(r["n_followups"] for r in rows),
                           "total_in_window_before_cap": sum(r["n_in_window"] for r in rows),
                           "contexts_capped_at_50": sum(r["n_in_window"] > 50 for r in rows)}}
    assert [r["ctx"] for r in rows] == list(range(150))
    for space in SPACES:
        V = norm(np.load(D / f"kaggle-output/review-olmo-2023-embed/emb_{space}.npy").astype(np.float32))
        assert len(V) == len(keys)
        F23 = {i: [] for i in range(150)}
        for k, v in zip(keys, V):
            F23[int(k.split("/")[0])].append(v)
        for gen in GENS:
            rng = np.random.default_rng(SEED)
            ctx, C, F, pb, PB, _it, _IT = load_generator(gen, space)
            assert [c["arxiv_id"] for c in ctx] == [r["arxiv_id"] for r in rows]
            g23, gY, gY_s2, slope_groups = [], {y: [] for y in YEARS}, {y: [] for y in YEARS}, []
            n_fu = {y: 0 for y in YEARS}
            for i, c in enumerate(ctx):
                idx = [n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == "CTX"]
                s_idea = float(np.median(PB[idx] @ C[i]))
                st = F[i] @ C[i]
                yr = np.array([v1_year(f["arxiv_id"]) for f in c["followups"]])
                yr_s2 = np.array([int(f["date"][:4]) for f in c["followups"]])
                st23 = np.array(F23[i]) @ C[i] if F23[i] else np.array([])
                g23.append(s_idea - float(np.median(st23)) if len(st23) else np.nan)
                for y in YEARS:
                    m = yr == y
                    n_fu[y] += int(m.sum())
                    gY[y].append(s_idea - float(np.median(st[m])) if m.any() else np.nan)
                    m2 = yr_s2 == y
                    gY_s2[y].append(s_idea - float(np.median(st[m2])) if m2.any() else np.nan)
                slope_groups.append((np.concatenate([np.full(len(st23), 2023.0), yr.astype(float)]),
                                     np.concatenate([st23, st])))
            g23 = np.array(g23)
            gY = {y: np.array(v) for y, v in gY.items()}
            gY_s2 = {y: np.array(v) for y, v in gY_s2.items()}
            ok23 = ~np.isnan(g23)
            out: dict = {}
            x = g23[ok23]
            out["gap_2023"] = summary(x, rng.integers(0, len(x), (B, len(x)))) | {"n_followups": int(sum(len(F23[i]) for i in range(150)))}
            out["gap_by_v1_year"] = {}
            for y in YEARS:
                ok = ~np.isnan(gY[y])
                xx = gY[y][ok]
                out["gap_by_v1_year"][str(y)] = summary(xx, rng.integers(0, len(xx), (B, len(xx)))) | {"n_followups": n_fu[y]}
            out["gap_by_s2_year_reproduction"] = {}
            for y in YEARS:
                ok = ~np.isnan(gY_s2[y])
                xx = gY_s2[y][ok]
                out["gap_by_s2_year_reproduction"][str(y)] = summary(xx, rng.integers(0, len(xx), (B, len(xx))))
            both = ok23 & ~np.isnan(gY[2024])
            a, b = g23[both], gY[2024][both]
            bi = rng.integers(0, len(a), (B, len(a)))
            ma, mb = a[bi].mean(1), b[bi].mean(1)
            out["paired_2023_vs_2024"] = {
                "n_contexts": int(both.sum()), "mean_gap_2023": float(a.mean()), "mean_gap_2024": float(b.mean()),
                "diff_2023_minus_2024": float(a.mean() - b.mean()), "diff_ci95": ci_of(ma - mb),
                "ratio_2023_over_2024": float(a.mean() / b.mean()), "ratio_ci95": ci_of(ma / mb),
                "mean_gap_2023_ci95": ci_of(ma), "mean_gap_2024_ci95": ci_of(mb)}
            sl = fe_slope(slope_groups)
            bs = [fe_slope([slope_groups[j] for j in rng.integers(0, 150, 150)]) for _ in range(B_SLOPE)]
            out["followup_step_slope_per_v1_year_2023_2026"] = {
                "slope": sl, "ci95": ci_of(np.array(bs)), "n_boot": B_SLOPE,
                "note": "positive = later follow-ups closer to the context"}
            if gen == "olmo-7b-0424":
                c1 = out["gap_2023"]["mean"] > 0 and out["gap_2023"]["ci95"][0] > 0
                c2 = out["paired_2023_vs_2024"]["mean_gap_2023"] >= 0.5 * out["paired_2023_vs_2024"]["mean_gap_2024"]
                out["decision"] = {"c1_positive_ci_excludes_0": bool(c1), "c2_not_smaller_than_half_2024": bool(c2),
                                   "underpowered": bool(ok23.sum() < 30), "passes": bool(c1 and c2 and ok23.sum() >= 30)}
            else:
                out["note"] = "comparison only: gpt-4o's October 2023 cutoff overlaps the 2023 window"
            res[f"{gen}/{space}"] = out
            print(gen, space, json.dumps({k: out[k] for k in ("gap_2023", "paired_2023_vs_2024")}, indent=1), flush=True)
    p = [res[f"olmo-7b-0424/{s}"]["decision"]["passes"] for s in SPACES]
    under = any(res[f"olmo-7b-0424/{s}"]["decision"]["underpowered"] for s in SPACES)
    res["verdict"] = ("underpowered" if under else "not a lag artefact" if all(p) else "mixed" if any(p)
                      else "a lag contribution cannot be ruled out")
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("verdict:", res["verdict"])


if __name__ == "__main__":
    main()
