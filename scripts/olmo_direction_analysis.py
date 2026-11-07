"""Analysis of docs/prereg/strengthen_olmo_direction.md (needs labels and the Kaggle outputs).

  OMP_NUM_THREADS=4 uv run python scripts/olmo_direction_analysis.py check   # stopping check after a wave: CI
                                                                              # widths, SDs, n*, next wave size only
  OMP_NUM_THREADS=4 uv run python scripts/olmo_direction_analysis.py final   # all estimands -> JSON

Inputs: runs/olmo-direction/{eligible.json, followups.jsonl, citing_all.jsonl}; kaggle-output/olmo-direction-w*p*/
(run_info.json, iter_olmo.jsonl, {ctx,fup,iter}_{space}.npy, fup_owner.npy); future-split corpus vectors
prepared/arxiv/emb_{space}.npy (Kaggle encoders). No encoder is loaded.
Output (final): docs/prereg/strengthen_olmo_direction.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_tangent import tang, unit  # noqa: E402

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/olmo-direction"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/strengthen_olmo_direction.json"
SEED, N_BOOT, BOUND, N_MAX, WAVE1 = 20261106, 2000, 0.015, 250, 100
SPACES = ("qwen3e06", "bgem3")
CRIT = ("GEO", "LLM", "ORACLE")
ARMS = CRIT + ("NEUTRAL",)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_outputs():
    """{pos: {"role", "wave", "part", "rows": [...], space: {"C", "F", "V"}}} over all Kaggle output parts."""
    out = {}
    for d in sorted((DATA / "kaggle-output").glob("olmo-direction-w*p*")):
        info = json.loads((d / "run_info.json").read_text())
        rows = [json.loads(l) for l in (d / "iter_olmo.jsonl").read_text().splitlines()]
        owner = np.load(d / "fup_owner.npy")
        ab = [r for r in rows if r["abstract"]]
        vec = {s: {k: norm(np.load(d / f"{k}_{s}.npy").astype(np.float32)) for k in ("ctx", "fup", "iter")}
               for s in SPACES}
        for s in SPACES:
            assert len(vec[s]["iter"]) == len(ab) and len(vec[s]["fup"]) == len(owner)
        for i, (p, role) in enumerate(zip(info["positions"], info["roles"])):
            assert p not in out, f"pos {p} in two output parts"
            sel = [n for n, r in enumerate(ab) if r["ctx"] == i]
            out[p] = {"role": role, "wave": info["wave"], "part": info["part"],
                      "rows": [ab[n] for n in sel],
                      "n_rows_all": sum(r["ctx"] == i for r in rows)}
            for s in SPACES:
                out[p][s] = {"C": vec[s]["ctx"][i], "F": vec[s]["fup"][owner == i], "V": vec[s]["iter"][sel]}
    return out


def future_pool(space: str):
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    E = norm(np.load(DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep][fut])
    return E, meta.loc[fut, "id"].to_numpy()


def per_context(o: dict, rec: dict, citing: set, space: str, E, ids) -> dict:
    c, F = o[space]["C"], o[space]["F"]
    js = rec["consensus"]
    TF = tang(F, c)
    TC = TF[js]
    f, fo = unit(TC.mean(0)), unit(TF.mean(0))
    pool = E[~np.isin(ids, list(citing | {rec["arxiv_id"]}))]
    order = np.argsort(-(pool @ c))

    def p_of(k):
        return unit(tang(pool[order[:k]], c).mean(0))

    def s_of(ff, pp):
        return unit(ff - (ff @ pp) * pp)

    p, p_rob = p_of(len(js)), p_of(len(F))
    S = {"s": s_of(f, p), "s_robust_kfollowups": s_of(f, p_rob), "s_all_followups": s_of(fo, p_rob)}
    res = {"k": len(js), "n_followups": len(F), "f_dot_p": float(f @ p),
           "loo": float(np.mean([TC[j] @ unit(TC.sum(0) - TC[j]) for j in range(len(TC))]))}
    V = o[space]["V"]
    for a in ARMS:
        for r in (0, 2):
            idx = [n for n, x in enumerate(o["rows"]) if x["arm"] == a and x["round"] == r]
            if not idx:
                continue
            T = tang(V[idx], c)
            res[(a, r)] = {**{n: float((T @ v).mean()) for n, v in S.items()},
                           "align_f": float((T @ f).mean()), "along_p": float((T @ p).mean()),
                           "step": float((V[idx] @ c).mean()), "n_ideas": len(idx)}
    return res


def boot_ci(x, rng):
    x = np.asarray(x)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def wil(x):
    x = np.asarray(x)
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 2 and np.any(x != 0) else None


def holm(ps):
    order = np.argsort(ps)
    out, run = [None] * len(ps), 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[k]))
        out[k] = run
    return out


def delta(pc, a, key):
    if (a, 0) in pc and (a, 2) in pc:
        d = pc[(a, 2)][key] - pc[(a, 0)][key]
        return -d if key == "step" else d  # step reported as lengthening (round 0 − round 2)
    return np.nan


def contrasts(PC: list[dict], key: str, rng, with_tests=True) -> dict:
    out = {}
    for a in CRIT:
        d = np.array([delta(pc, a, key) - delta(pc, "NEUTRAL", key) for pc in PC])
        d = d[~np.isnan(d)]
        ci = boot_ci(d, rng) if len(d) >= 2 else [np.nan, np.nan]
        e = {"n": int(len(d)), "sd": float(d.std(ddof=1)) if len(d) >= 2 else None, "ci": ci,
             "half_width": (ci[1] - ci[0]) / 2}
        if with_tests:
            e |= {"mean": float(d.mean()) if len(d) else None, "p": wil(d)}
        out[a] = e
    if with_tests:
        ps = [out[a]["p"] if out[a]["p"] is not None else 1.0 for a in CRIT]
        for a, h in zip(CRIT, holm(ps)):
            lo, hi = out[a]["ci"]
            inside = -BOUND < lo and hi < BOUND
            sig = h < 0.05
            out[a] |= {"p_holm": h, "significant_holm": sig, "ci_within_pm0.015": inside,
                       "reading": ("detected, below 0.015" if sig and inside else "feedback-specific turn detected" if sig
                                   else "no feedback-specific turn of 0.015 or more" if inside else "insufficient")}
    return out


def collect(space: str, outs: dict, el: dict, citing: dict, roles=("eligible",)) -> tuple[list, list]:
    E, ids = future_pool(space)
    recs = {r["pos"]: r for r in el["contexts"]}
    PC, pos = [], []
    for p in sorted(outs):
        if outs[p]["role"] not in roles:
            continue
        PC.append(per_context(outs[p], recs[p], citing[recs[p]["arxiv_id"]], space, E, ids))
        pos.append(p)
    return PC, pos


def check() -> None:
    el = json.loads((RUN / "eligible.json").read_text())
    citing = {json.loads(l)["arxiv_id"]: set(json.loads(l)["citing_arxiv_all"])
              for l in (RUN / "citing_all.jsonl").read_text().splitlines()}
    outs = load_outputs()
    PC, pos = collect("qwen3e06", outs, el, citing)
    c = contrasts(PC, "s", np.random.default_rng(SEED), with_tests=False)
    n = len(PC)
    sd = max(v["sd"] for v in c.values())
    nstar = math.ceil((1.96 * sd / BOUND) ** 2)
    stop = all(v["half_width"] <= BOUND for v in c.values()) or n >= N_MAX
    nxt = None if stop else (min(N_MAX, max(nstar, 150)) - WAVE1 if n <= WAVE1 else min(50, N_MAX - n))
    print(json.dumps({"n_eligible_generated": n, "waves": sorted({outs[p]["wave"] for p in pos}),
                      "half_widths": {a: round(v["half_width"], 4) for a, v in c.items()},
                      "sd": {a: round(v["sd"], 4) for a, v in c.items()}, "n_star": nstar, "stop": stop,
                      "next_wave_adds_eligible": nxt,
                      "note": "widths and SDs only; estimates are not read before the stop (prereg)"}, indent=1))


def final() -> None:
    el = json.loads((RUN / "eligible.json").read_text())
    citing = {json.loads(l)["arxiv_id"]: set(json.loads(l)["citing_arxiv_all"])
              for l in (RUN / "citing_all.jsonl").read_text().splitlines()}
    outs = load_outputs()
    res = {"prereg": "docs/prereg/strengthen_olmo_direction.md", "labels": {k: v for k, v in el.items()
                                                                             if k not in ("contexts", "eligible_pos")}}
    for space in SPACES:
        rng = np.random.default_rng(SEED)
        PC, pos = collect(space, outs, el, citing)
        cell = {"n_contexts": len(PC), "waves": sorted({outs[p]["wave"] for p in pos}),
                "contrasts_along_s": contrasts(PC, "s", rng)}
        for key in ("s_robust_kfollowups", "s_all_followups", "align_f", "along_p", "step"):
            cell[f"contrasts_{key}"] = contrasts(PC, key, rng)
        arm = {}
        for a in ARMS:
            arm[a] = {}
            for key in ("s", "s_all_followups", "align_f", "along_p", "step"):
                x = np.array([delta(pc, a, key) for pc in PC])
                x = x[~np.isnan(x)]
                arm[a][key] = {"n": int(len(x)), "mean": float(x.mean()), "ci": boot_ci(x, rng), "p": wil(x)}
            arm[a]["align_f_r0"] = float(np.nanmean([pc[(a, 0)]["align_f"] for pc in PC if (a, 0) in pc]))
            arm[a]["align_f_r2"] = float(np.nanmean([pc[(a, 2)]["align_f"] for pc in PC if (a, 2) in pc]))
        cell["per_arm_change"] = arm
        cell["positive_control_loo"] = float(np.mean([pc["loo"] for pc in PC]))
        cell["mean_f_dot_p"] = float(np.mean([pc["f_dot_p"] for pc in PC]))
        cell["k_continuations"] = {"mean": float(np.mean([pc["k"] for pc in PC])),
                                   "median": float(np.median([pc["k"] for pc in PC]))}
        cell["per_wave"] = {}
        for w in cell["waves"]:
            sub = [pc for pc, p in zip(PC, pos) if outs[p]["wave"] == w]
            cell["per_wave"][str(w)] = contrasts(sub, "s", rng)
        PCm, _ = collect(space, outs, el, citing, roles=("eligible", "missed"))
        cell["sensitivity_with_missed_audit"] = contrasts(PCm, "s", rng) | {"n_contexts": len(PCm)}
        res[f"olmo/{space}"] = cell
        print(space, json.dumps({a: {k: cell["contrasts_along_s"][a][k] for k in ("n", "mean", "ci", "p_holm", "reading")}
                                 for a in CRIT}, indent=1, default=str), flush=True)
    res["primary"] = res["olmo/qwen3e06"]["contrasts_along_s"]
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    {"check": check, "final": final}[sys.argv[1]]()
