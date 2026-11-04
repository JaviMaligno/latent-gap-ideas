"""Figures for docs/paper/draft.md, drawn only from saved results (no model calls, no re-analysis).

    uv run --with matplotlib python scripts/paper_figures.py
Output: docs/paper/figures/fig{1..5}_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "docs/prereg"
OUT = ROOT / "docs/paper/figures"
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, colour-blind safe
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7f7f7f", "black": "#000000"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.titlesize": 9, "legend.frameon": False})
EMB = {"qwen3e06": "Qwen3", "bgem3": "BGE-M3"}


def load(name: str) -> dict:
    return json.loads((PRE / name).read_text())


def fig1_anchoring() -> None:
    cells = load("anchoring_causal_result.json")["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    conds = [("true", "true neighbours", C["blue"]), ("swap", "swapped (donor's)", C["orange"]),
             ("none", "no neighbours", C["grey"])]
    for ax, arm in zip(axes, ["REAL", "GAP", "FAR"]):
        sub = [c for c in cells if c["arm"] == arm]
        x = np.arange(len(sub))
        for j, (k, lab, col) in enumerate(conds):
            ax.bar(x + (j - 1) * 0.26, [c["quantitative"][k]["means"]["target"] for c in sub], 0.26, color=col,
                   label=lab)
        gname = {"luna": "luna", "sonnet": "Sonnet"}
        dname = {"steam": "Steam", "arxiv": "arXiv"}
        ax.set_xticks(x, [f"{dname[c['domain']]}\n{gname[c['gen']]}" for c in sub], fontsize=7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{arm} targets")
    axes[0].set_ylabel("cosine of decoded text to target")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_anchoring.png")


def fig2_hole_vs_momentum() -> None:
    a = load("where_future_lands_controls.json")["auc_no_cv"]
    keys = [f"{d}/{e}" for e in EMB for d in ["steam", "arxiv", "patents"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(keys))[::-1]
    for off, sc, col, lab in [(0.15, "momentum", C["blue"], "momentum"), (-0.15, "hole", C["orange"], "hole score")]:
        m = [a[k][sc]["auc"] for k in keys]
        lo = [a[k][sc]["auc"] - a[k][sc]["ci"][0] for k in keys]
        hi = [a[k][sc]["ci"][1] - a[k][sc]["auc"] for k in keys]
        ax.errorbar(m, y + off, xerr=[lo, hi], fmt="o", color=col, ms=4, capsize=2, label=lab)
    ax.axvline(0.5, color="black", lw=0.6, ls="--")
    ax.set_yticks(y, [f"{k.split('/')[0]} / {EMB[k.split('/')[1]]}" for k in keys])
    ax.set_xlabel("AUC for top-decile growth cells (95 % CI)")
    ax.set_xlim(0.3, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    fig.savefig(OUT / "fig2_hole_vs_momentum.png")


def fig3_step() -> None:
    ic = load("continue_from_context_result.json")["iterative_critique"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cols = {"GEO": C["blue"], "LLM": C["green"], "ORACLE": C["red"]}
    labs = {"GEO": "GEO (rule)", "LLM": "LLM reviewer", "ORACLE": "ORACLE (score to follow-ups)"}
    for arm, col in cols.items():
        a1.plot([0, 1, 2], [ic[arm][f"round{r}"]["step"] for r in range(3)], "o-", color=col, label=labs[arm])
        a2.plot([0, 1, 2], [ic[arm][f"round{r}"]["fit"] for r in range(3)], "o-", color=col)
    a1.axhspan(0.574, 0.661, color=C["grey"], alpha=0.15, lw=0)
    a1.axhline(ic["followup_step_median"], color=C["grey"], lw=1, ls="--")
    a1.text(2.05, ic["followup_step_median"], "real follow-ups\n(median, IQR)", fontsize=7, va="center")
    a1.set_ylabel("step = cos(context, idea)\n(lower = longer step)")
    a2.set_ylabel("fit = max cos to follow-ups")
    for ax in (a1, a2):
        ax.set_xticks([0, 1, 2], ["round 0", "1", "2"])
        ax.set_xlim(-0.2, 2.2)
    a1.set_xlim(-0.2, 2.9)
    a1.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_step.png")


def fig4_tangent() -> None:
    t = load("direction_tangent.json")
    keys = ["gpt-4o/qwen3e06", "gpt-4o/bgem3", "olmo-7b-0424/qwen3e06", "olmo-7b-0424/bgem3"]
    cols = {"GEO": C["blue"], "LLM": C["green"], "ORACLE": C["red"]}
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for i, k in enumerate(keys):
        for j, (arm, col) in enumerate(cols.items()):
            x = i + (j - 1) * 0.22
            r0, r2 = t[k][arm]["alignment_round0"], t[k][arm]["alignment_round2"]
            ax.annotate("", xy=(x, r2), xytext=(x, r0), arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
            ax.plot(x, r0, "o", color=col, ms=3, label=arm if i == 0 else None)
        loo = t[k]["followups_loo_alignment"]
        ax.plot([i - 0.38, i + 0.38], [loo, loo], color=C["black"], lw=1.2,
                label="follow-ups among themselves" if i == 0 else None)
    g = {"gpt-4o": "gpt-4o", "olmo-7b-0424": "OLMo"}
    ax.set_xticks(range(4), [f"{g[k.split('/')[0]]} / {EMB[k.split('/')[1]]}" for k in keys])
    ax.set_ylabel("tangent alignment with\nfollow-ups' direction")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(OUT / "fig4_tangent.png")


def fig5_inspiration() -> None:
    res = load("inspiration_from_anything_result.json")
    order = ["steam_game", "board_game", "sports_rules", "movie", "paper", "patent", "news", "recipe", "wikipedia"]
    pal = [C["black"], C["blue"], C["sky"], C["purple"], C["green"], C["yellow"], C["grey"], C["red"], C["orange"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, e in zip(axes, EMB):
        d = pd.read_parquet(DATA / f"runs/inspiration/gpt-4o-0806/scored_{e}.parquet")
        d = d[d["arm"] != "NONE"].copy()
        # quintiles of distance re-ranked within source in this embedder, as in the analysis
        d["q"] = np.ceil(d.groupby("source")["dist"].rank(pct=True) * 5).clip(1, 5)
        for s, col in zip(order, pal):
            m = d[d["source"] == s].groupby("q")["margin"].mean()
            ax.plot(m.index, m.values, "o-", color=col, ms=3, lw=1.2, label=s.replace("_", " "))
        for b, ls in [("NONE", "--"), ("MOMENTUM", ":")]:
            ax.axhline(res[e]["baselines"][b]["margin"], color=C["black"], lw=1, ls=ls)
        ax.axhline(0, color=C["grey"], lw=0.4)
        ax.set_title(f"{EMB[e]} (both framings pooled)")
        ax.set_xticks(range(1, 6), ["Q1\nnear", "Q2", "Q3", "Q4", "Q5\nfar"])
    axes[0].set_ylabel("anticipation margin")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color="k", ls="--"), plt.Line2D([], [], color="k", ls=":")],
               lab + ["no material (NONE)", "real 2023 games (MOMENTUM)"], fontsize=7, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_inspiration.png")


if __name__ == "__main__":
    for f in (fig1_anchoring, fig2_hole_vs_momentum, fig3_step, fig4_tangent, fig5_inspiration):
        f()
        print("wrote", f.__name__, flush=True)
