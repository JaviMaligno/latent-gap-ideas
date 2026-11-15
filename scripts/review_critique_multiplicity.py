"""Part B of docs/prereg/review_olmo_reliability.md (peer review M9): the single "detected" feedback effect of the
gpt-4o replication, corrected for multiplicity and pooled with the discovery sample.

  OMP_NUM_THREADS=4 uv run python scripts/review_critique_multiplicity.py                 # full run -> JSON below
  OMP_NUM_THREADS=4 uv run python scripts/review_critique_multiplicity.py --smoke 40 --boot 1000 --out /tmp/x.json

Registered follow-up (post hoc Holm-12 family requested by the reviewer); no paid API, no GPU, no generation.
Per-context arrays are reproduced with exactly the calls of strengthen_critique_replication.analyse() (new 150) and
.pooled() (original 150), via neutral_contrasts.per_context / change. First check, always on all contexts (also in
--smoke): means and t_ci95 of E1-E6, the R1/R2 Wilcoxon p, the E TOST p and the pooled means reproduce
strengthen_critique_replication.json to 1e-9 in both embedders; otherwise stop. --smoke N computes the new
statistics on the first N contexts of each sample only.
Output: docs/prereg/review_critique_multiplicity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continuation_reanalysis as RA  # noqa: E402
import neutral_contrasts as N  # noqa: E402
import revision_neutral as RV  # noqa: E402
import strengthen_critique_replication as SC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_critique_multiplicity.json"
REG = ROOT / "docs/prereg/strengthen_critique_replication.json"
SEED, N_BOOT, BOUND = 20261106, 20000, 0.015
ARMS = SC.ARMS
ALL = ARMS + ("NEUTRAL",)
SPACES = ("qwen3e06", "bgem3")
DATA = SC.DATA


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def corpus(space, keep, fut, notpast):
    """Ft, Fh as in analyse()/pooled(): norm(emb.astype(f32)[keep]) sliced. Row-wise ops, so slicing the float16
    matrix first and normalising the slices gives the same values with less memory (checked by the reproduction)."""
    X = np.load(DATA / f"prepared/arxiv/emb_{space}.npy")[keep]
    Ft = norm(X[fut].astype(np.float32))
    Fh = Ft if np.array_equal(fut, notpast) else norm(X[notpast].astype(np.float32))
    return Ft, Fh


def per_context_both(space, meta_t):
    keep, fut, notpast, ids_t, ids_h, prim_h, by_id = meta_t
    Ft, Fh = corpus(space, keep, fut, notpast)
    # discovery 150, as pooled()
    ctx0, C0, F0, it0, IT0 = RV.load_gpt4o(RA, space)
    nrows0 = RV.neutral_rows()
    NV0 = norm(np.load(RV.RUN / f"neutral_emb_{space}.npy").astype(np.float32))

    def dis0(i, fids, _c=ctx0):
        return Fh[(prim_h == by_id.loc[_c[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, list(fids))]
    M0, _ = N.per_context(ctx0, C0, F0, {a: (it0, IT0) for a in ARMS} | {"NEUTRAL": (nrows0, NV0)},
                          ALL, Ft, ids_t, dis0)
    # replication 150, as analyse()
    ctx = SC.ctx_rows()
    nf = [len(c["followups"]) for c in ctx]
    starts = np.cumsum([0] + nf)
    it, nu = SC.idea_rows()
    if space == "qwen3e06":
        z = np.load(SC.RUN / "emb.npz")
        C, Fall, V = norm(z["C"]), norm(z["F"]), norm(np.load(SC.RUN / "ideas_emb_qwen3e06.npy"))
    else:
        z = np.load(SC.RUN / "emb_bgem3.npz")
        C, Fall, V = norm(z["ctx"]), norm(z["fup"]), norm(z["ideas"])
    assert len(V) == len(it) + len(nu) and len(Fall) == starts[-1]
    F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]

    def dis1(i, fids):
        return Fh[(prim_h == by_id.loc[ctx[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, list(fids))]
    M1, _ = N.per_context(ctx, C, F, {a: (it, V[:len(it)]) for a in ARMS} | {"NEUTRAL": (nu, V[len(it):])},
                          ALL, Ft, ids_t, dis1)
    return M0, M1


def arrays(M) -> dict:
    """Per-context values: E1-E3 Δs_A, E4-E6 Δs_A − Δs_NEUTRAL, NEUTRAL Δs, R1/R2 step contrasts (NaN kept)."""
    out = {}
    for j, a in enumerate(ARMS):
        out[f"E{j + 1}:{a}:s_change"] = N.change(M, a, "s")
        out[f"E{j + 4}:{a}-NEUTRAL:s"] = N.change(M, a, "s") - N.change(M, "NEUTRAL", "s")
    out["NEUTRAL:s_change"] = N.change(M, "NEUTRAL", "s")
    out["R1:GEO-NEUTRAL:step"] = N.change(M, "GEO", "step") - N.change(M, "NEUTRAL", "step")
    out["R2:LLM-NEUTRAL:step"] = N.change(M, "LLM", "step") - N.change(M, "NEUTRAL", "step")
    return out


def nn(x):
    x = np.asarray(x, float)
    return x[~np.isnan(x)]


def reproduce(A0, A1, reg_space) -> dict:
    conf, pool = reg_space["confirmatory"], reg_space["secondary"]["pooled_with_original_150"]
    dev = {}
    for k, v in conf.items():
        d = nn(A1[k])
        if k.startswith("E"):
            m = float(d.mean())
            ref = v["mean"] if "mean" in v else v["diff_mean"]
            t = N.tost(d)
            dev[k] = max(abs(m - ref), abs(t["t_ci95"][0] - v["t_ci95"][0]), abs(t["t_ci95"][1] - v["t_ci95"][1]),
                         abs(t["p_tost"] - v["p_tost"]))
        else:
            dev[k] = max(abs(float(d.mean()) - v["diff_mean"]),
                         abs(float(stats.wilcoxon(d, alternative="greater").pvalue) - v["p"]))
    for a in ARMS:
        j = ARMS.index(a)
        for key, src in ((f"{a}:s_change", f"E{j + 1}:{a}:s_change"), (f"{a}-NEUTRAL:s", f"E{j + 4}:{a}-NEUTRAL:s")):
            m = float(nn(np.concatenate([A0[src], A1[src]])).mean())
            dev[f"pooled {key}"] = abs(m - pool[key]["mean"])
    m = float(nn(np.concatenate([A0["NEUTRAL:s_change"], A1["NEUTRAL:s_change"]])).mean())
    dev["pooled NEUTRAL:s_change"] = abs(m - pool["NEUTRAL:s_change"]["mean"])
    bad = {k: v for k, v in dev.items() if v > 1e-9}
    if bad:
        raise SystemExit(f"reproduction check failed: {bad}")
    return {"max_abs_dev": max(dev.values()), "n_checked": len(dev)}


def t_ci(d):
    d = nn(d)
    n, m, se = len(d), float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    q = stats.t.ppf(0.975, n - 1)
    return {"n": n, "mean": m, "se": se, "t_ci95": [m - q * se, m + q * se],
            "p_t": float(stats.ttest_1samp(d, 0.0).pvalue)}


def boot_means(x, IDX):
    xs = np.asarray(x, float)[IDX]
    ok = ~np.isnan(xs)
    return np.where(ok, xs, 0).sum(1) / ok.sum(1)


def pct(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def inside(ci):
    return bool(-BOUND < ci[0] and ci[1] < BOUND)


def excludes0(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


def welch(x, y):
    x, y = nn(x), nn(y)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    diff = float(x.mean() - y.mean())
    q = stats.t.ppf(0.975, df)
    return {"diff_rep_minus_disc": diff, "ci95": [diff - q * se, diff + q * se], "df": float(df),
            "p": float(2 * stats.t.sf(abs(diff / se), df))}


E_KEYS = [f"E{j + 1}:{a}:s_change" for j, a in enumerate(ARMS)] + [f"E{j + 4}:{a}-NEUTRAL:s" for j, a in enumerate(ARMS)]
POOL_KEYS = [f"E{j + 4}:{a}-NEUTRAL:s" for j, a in enumerate(ARMS)]
ARM_KEYS = [f"E{j + 1}:{a}:s_change" for j, a in enumerate(ARMS)] + ["NEUTRAL:s_change"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="new statistics on the first N contexts of each sample")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--spaces", default=",".join(SPACES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    reg = json.loads(REG.read_text())
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "primary", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    notpast = (meta["split"] != "past").to_numpy()
    meta_t = (keep, fut, notpast, meta.loc[fut, "id"].to_numpy(), meta.loc[notpast, "id"].to_numpy(),
              meta.loc[notpast, "primary"].to_numpy(), meta.set_index("id"))
    spaces = args.spaces.split(",")
    res = {"prereg": "docs/prereg/review_olmo_reliability.md", "part": "B",
           "status": "registered follow-up; Holm-12 is a post hoc correction applied to a registered test",
           "smoke": bool(args.smoke), "seed": SEED, "n_boot": args.boot, "deviations": DEVIATIONS}
    A = {}
    for space in spaces:
        M0, M1 = per_context_both(space, meta_t)
        A0, A1 = arrays(M0), arrays(M1)
        chk = reproduce(A0, A1, reg[space])
        print(space, "reproduction check ok", chk, flush=True)
        if args.smoke:
            A0 = {k: v[:args.smoke] for k, v in A0.items()}
            A1 = {k: v[:args.smoke] for k, v in A1.items()}
        A[space] = (A0, A1)
        res.setdefault("reproduction_check", {})[space] = chk

    # B1 ---------------------------------------------------------------------------------------------------
    B1 = {}
    for space in spaces:
        A1 = A[space][1]
        for k in E_KEYS:
            d = nn(A1[k])
            B1[f"{space}|{k}"] = {**t_ci(d), "p_wilcoxon": float(stats.wilcoxon(d).pvalue),
                                  "p_tost": N.tost(d)["p_tost"],
                                  "registered_reading": reg[space]["confirmatory"][k]["reading"]}
    keys12 = list(B1)
    fam = {
        "holm12_t": N.holm({k: B1[k]["p_t"] for k in keys12}),
        "holm12_wilcoxon": N.holm({k: B1[k]["p_wilcoxon"] for k in keys12}),
        "holm6_feedback_specific_t": N.holm({k: B1[k]["p_t"] for k in keys12 if "-NEUTRAL" in k}),
    }
    for space in spaces:
        fam[f"holm6_within_{space}_t"] = N.holm({k: B1[k]["p_t"] for k in keys12 if k.startswith(space)})
    sens = {f"det|{k}": B1[k]["p_t"] for k in keys12}
    for space in spaces:
        A1 = A[space][1]
        for k in ("R1:GEO-NEUTRAL:step", "R2:LLM-NEUTRAL:step"):
            sens[f"reg|{space}|{k}"] = float(stats.wilcoxon(nn(A1[k]), alternative="greater").pvalue)
        for k in E_KEYS:
            sens[f"reg|{space}|{k}"] = min(1.0, 2 * N.tost(nn(A1[k]))["p_tost"])
    fam["sensitivity28"] = N.holm(sens)
    for k in keys12:
        e = B1[k]
        for f, h in fam.items():
            hk = f"det|{k}" if f == "sensitivity28" else k
            if hk in h:
                e[f"p_{f}"] = h[hk]
        lo, hi = e["t_ci95"]
        if e["p_holm12_t"] < 0.05:
            e["reading_R-B1"] = "turn detected"
        elif excludes0(e["t_ci95"]):
            e["reading_R-B1"] = f"estimated {e['mean']:+.3f} (95 % CI {lo:+.3f} to {hi:+.3f}), not significant after correction"
        else:
            e["reading_R-B1"] = "no turn detected"
        if e["registered_reading"] == "within ±0.015 (confirmed)":
            e["equivalence_reading_unchanged"] = "within ±0.015 (confirmed)"
    res["B1_holm_detection"] = {"tests": B1, "n_family": len(keys12), "n_sensitivity_family": len(sens)}

    # B2 ---------------------------------------------------------------------------------------------------
    B2 = {}
    for space in spaces:
        A0, A1 = A[space]
        n0, n1 = len(next(iter(A0.values()))), len(next(iter(A1.values())))
        rng = np.random.default_rng(SEED)
        IDX = rng.integers(0, n0 + n1, (args.boot, n0 + n1))
        rng1 = np.random.default_rng(SEED + 1)
        IDXs = np.concatenate([rng1.integers(0, n0, (args.boot, n0)), n0 + rng1.integers(0, n1, (args.boot, n1))], 1)
        cell = {"n_contexts": [n0, n1]}
        for k in POOL_KEYS + ARM_KEYS:
            x = np.concatenate([A0[k], A1[k]])
            disc, rep = t_ci(A0[k]), t_ci(A1[k])
            w0, w1 = 1 / disc["se"] ** 2, 1 / rep["se"] ** 2
            fe = (w0 * disc["mean"] + w1 * rep["mean"]) / (w0 + w1)
            fse = 1 / np.sqrt(w0 + w1)
            e = {**t_ci(x), "boot_ci95": pct(boot_means(x, IDX)), "boot_ci95_sample_stratified": pct(boot_means(x, IDXs)),
                 "discovery": disc, "replication": rep, "heterogeneity": welch(A1[k], A0[k]),
                 "fixed_effect": {"mean": float(fe), "se": float(fse), "ci95": [float(fe - 1.96 * fse), float(fe + 1.96 * fse)]},
                 "role": "pooled contrast (Holm-6 family)" if k in POOL_KEYS else "descriptive"}
            e["t_ci_within_pm0.015"] = inside(e["t_ci95"])
            e["boot_ci_within_pm0.015"] = inside(e["boot_ci95"])
            cell[k] = e
        B2[space] = cell
    h6 = N.holm({f"{s}|{k}": B2[s][k]["p_t"] for s in spaces for k in POOL_KEYS})
    for kk, h in h6.items():
        s, k = kk.split("|")
        B2[s][k]["p_holm6"] = h
    geo = "E4:GEO-NEUTRAL:s"
    sig = [s for s in spaces if B2[s][geo]["p_holm6"] < 0.05]
    het = {s: [k for k in POOL_KEYS if B2[s][k]["heterogeneity"]["p"] < 0.05] for s in spaces}

    # B3 ---------------------------------------------------------------------------------------------------
    B3 = {}
    if set(SPACES) <= set(spaces):
        for nm, f in (("replication", lambda s: A[s][1][geo]),
                      ("pooled", lambda s: np.concatenate([A[s][0][geo], A[s][1][geo]]))):
            B3[nm] = t_ci(f("bgem3") - f("qwen3e06")) | {"what": "per-context d_BGE - d_Qwen3 for E4, paired t"}
    b3_incl0 = (not excludes0(B3["replication"]["t_ci95"])) if B3 else None
    res["B3_embedder_difference"] = B3

    if len(sig) == len(SPACES):
        rb2 = "pooled over 300 contexts, distance feedback adds a turn beyond revision in both embedders"
    elif len(sig) == 1:
        rb2 = ("one embedder only: both estimates given; embedder-dependent claim allowed only if B3 excludes zero "
               f"(B3 replication excludes zero: {not b3_incl0 if b3_incl0 is not None else 'n/a'})")
    else:
        rb2 = "not shown"
    B2["reading_R-B2"] = {"GEO-NEUTRAL": rb2, "holm6_significant_in": sig,
                          "heterogeneous_contrasts_(not_headline)": het}
    res["B2_pooled"] = B2
    e4b = B1.get(f"bgem3|{geo}")
    res["reading_R-B3"] = ("no sentence may contrast the two embedders for E4" if b3_incl0
                           else "B3 excludes zero" if b3_incl0 is not None else "n/a (one embedder)")
    res["against_current_claim"] = {
        "E4_bgem3_fails_holm12": (e4b["p_holm12_t"] >= 0.05) if e4b else None,
        "B3_includes_zero": b3_incl0}
    Path(args.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"B1": {k: [round(v["mean"], 4), round(v["p_t"], 4), round(v["p_holm12_t"], 4), v["reading_R-B1"]]
                             for k, v in B1.items()},
                      "B2": {s: {k: [round(B2[s][k]["mean"], 4), [round(x, 4) for x in B2[s][k]["t_ci95"]],
                                     round(B2[s][k]["p_holm6"], 4), round(B2[s][k]["heterogeneity"]["p"], 4)]
                                 for k in POOL_KEYS} for s in spaces},
                      "B3": {k: [round(v["mean"], 4), [round(x, 4) for x in v["t_ci95"]]] for k, v in B3.items()},
                      "R-B2": B2["reading_R-B2"], "R-B3": res["reading_R-B3"]}, indent=1), flush=True)
    print("wrote", args.out)


DEVIATIONS = [
    "Corpus vectors: the float16 matrix is sliced (keep, future / not-past) before conversion to float32 and row "
    "normalisation, instead of converting the whole matrix first; row-wise operations, so the values are identical "
    "(the reproduction check to 1e-9 covers it) with about half the peak memory.",
    "Bootstrap means are NaN-aware (a context missing a value is skipped within a resample); with no missing values "
    "this equals the registered concatenate-and-drop estimand. Index matrices are drawn over all contexts of the "
    "sample(s).",
    "Sample-stratified bootstrap: one default_rng(20261107) draws the discovery (20,000 x n0) then the replication "
    "(20,000 x n1) index matrices.",
    "Sensitivity family of 28: R1/R2 enter with their registered one-sided Wilcoxon p, E1-E6 TOST as 2*p_TOST "
    "(capped at 1), both embedders, plus the 12 detection t-tests.",
]

if __name__ == "__main__":
    main()
