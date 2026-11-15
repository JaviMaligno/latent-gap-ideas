"""Exploratory (not registered, 2026-11-15): does the short step depend on how much later the follow-up appeared?

Question raised in review: contexts are 2021-2022 papers and follow-ups are citing papers first posted from 2024 on,
so a field's drift over the intervening years could explain part of the step gap. Using the discovery sample of
Section 5.1 (gpt-4o one-shot CTX ideas, all follow-ups), both embedders, this script reports
  (a) the step gap against the follow-ups of each publication year (2024, 2025, 2026), per context where that year has
      at least one follow-up, with bootstrap CIs over contexts;
  (b) the within-context slope of follow-up step on publication year (context fixed effects, follow-ups as rows),
      with a cluster bootstrap over contexts;
  (c) the step gap by context year (2021 vs 2022).
Read-only on saved embeddings (scripts/continuation_reanalysis.load_generator); no model is called.

Output: docs/paper/step_gap_by_year.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuation_reanalysis import load_generator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/paper/step_gap_by_year.json"
B = 10_000
SEED = 20261115
YEARS = (2024, 2025, 2026)


def boot_mean(x: np.ndarray, rng) -> list[float]:
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def fe_slope(groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    xs, ys = [], []
    for yr, st in groups:
        if len(yr) < 2 or np.ptp(yr) == 0:
            continue
        xs.append(yr - yr.mean()); ys.append(st - st.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float((x * y).sum() / (x * x).sum())


def main() -> None:
    res = {"note": "exploratory, not registered; discovery sample, gpt-4o one-shot CTX ideas, all follow-ups",
           "n_boot": B, "seed": SEED}
    for space in ("qwen3e06", "bgem3"):
        rng = np.random.default_rng(SEED)
        ctx, C, F, pb, PB, it, IT = load_generator("gpt-4o", space)
        idea_step, fu_groups, ctx_year = [], [], []
        by_year = {y: [] for y in YEARS}
        all_gap = []
        n_fu = {y: 0 for y in YEARS}
        for i, c in enumerate(ctx):
            idx = [n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == "CTX"]
            if not idx:
                continue
            s_idea = float(np.median(PB[idx] @ C[i]))
            st = F[i] @ C[i]
            yr = np.array([int(f["date"][:4]) for f in c["followups"]])
            all_gap.append(s_idea - float(np.median(st)))
            ctx_year.append(int(c["date"][:4]))
            for y in YEARS:
                m = yr == y
                n_fu[y] += int(m.sum())
                by_year[y].append(s_idea - float(np.median(st[m])) if m.any() else np.nan)
            fu_groups.append((yr.astype(float), st))
        all_gap = np.array(all_gap); ctx_year = np.array(ctx_year)
        out = {"n_contexts": int(len(all_gap)), "step_gap_all": {"mean": float(all_gap.mean()),
                                                                 "ci95": boot_mean(all_gap, rng)},
               "by_followup_year": {}, "by_context_year": {}}
        for y in YEARS:
            g = np.array(by_year[y]); ok = ~np.isnan(g)
            out["by_followup_year"][str(y)] = {"n_contexts": int(ok.sum()), "n_followups": n_fu[y],
                                              "mean": float(g[ok].mean()) if ok.any() else None,
                                              "ci95": boot_mean(g[ok], rng) if ok.sum() > 1 else None}
        # paired: contexts with both 2024 and a later year
        g24, glate = np.array(by_year[2024]), np.array([np.nanmean([a, b]) if not (np.isnan(a) and np.isnan(b)) else np.nan
                                                         for a, b in zip(by_year[2025], by_year[2026])])
        ok = ~np.isnan(g24) & ~np.isnan(glate)
        d = glate[ok] - g24[ok]
        out["paired_later_minus_2024"] = {"n_contexts": int(ok.sum()), "mean": float(d.mean()), "ci95": boot_mean(d, rng)}
        slope = fe_slope(fu_groups)
        bs = []
        n = len(fu_groups)
        for _ in range(2000):
            pick = rng.integers(0, n, n)
            bs.append(fe_slope([fu_groups[j] for j in pick]))
        out["followup_step_slope_per_year_within_context"] = {
            "slope": slope, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_boot": 2000, "note": "positive slope = later follow-ups closer to the context (higher cosine)"}
        for y in (2021, 2022):
            g = all_gap[ctx_year == y]
            out["by_context_year"][str(y)] = {"n_contexts": int(len(g)), "mean": float(g.mean()),
                                             "ci95": boot_mean(g, rng)}
        res[space] = out
        print(space, json.dumps(out, indent=1))
    OUT.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
