"""OLMo NEUTRAL arm vs OLMo's critique arms (docs/prereg/strengthen_olmo_neutral.md).

  kaggle kernels output javieraguilarmartin/latent-gap-olmo-neutral \
      -p ~/Documents/experiment-data/latent-gap-ideas/kaggle-output/olmo-neutral-v1
  OMP_NUM_THREADS=4 uv run python scripts/strengthen_olmo_neutral.py analyse   # -> docs/prereg/strengthen_olmo_neutral.json
  uv run python scripts/strengthen_olmo_neutral.py --check                     # synthetic + structural checks, no corpus

GEO / Reviewer (LLM) / ORACLE: idea, context and follow-up vectors of the critique run (llama-continuation-v1, via
continuation_reanalysis.load_generator). NEUTRAL: the new kernel's vectors for all its rounds (round 0 = GEO's
round-0 text), so its round 0 -> 2 change is within one encoder run. Measures and tests mirror revision_neutral.py
(O1-O6 = N1-N6, Holm over 14 per embedder); F4 mirrors strengthen_continuations_full_analysis.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
KOUT = DATA / "kaggle-output/olmo-neutral-v1"
V1 = DATA / "kaggle-output/llama-continuation-v1"
LABELS = ROOT / "docs/prereg/strengthen_continuations_full_labels.json"
OUT = ROOT / "docs/prereg/strengthen_olmo_neutral.json"
ARMS = ("GEO", "LLM", "ORACLE")
ALL = ARMS + ("NEUTRAL",)
SPACES = ("qwen3e06", "bgem3")
SEED, N_BOOT = 20261030, 2000
F4_CONTEXTS = [6, 8, 10, 23, 30, 37, 53, 58, 59, 64, 83, 114, 118, 133, 141, 143, 148]
MEASURES = ("step", "fit", "hit", "align", "p", "s")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def tang(X, c):
    """Unit tangent of each row of X at c (as direction_tangent.tang)."""
    return unit(X - (X @ c)[..., None] * c)


def holm(p: dict) -> dict:
    keys, out, run = sorted(p, key=p.get), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        out[k] = run
    return out


def neutral_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return [r for r in rows if r["abstract"]]  # file order = order of the kernel's vectors


def consensus_sets(kind: str) -> dict[int, list[int]]:
    S: dict[int, list[int]] = {}
    for p in json.loads(LABELS.read_text()):
        a, b = p["gpt61sol"], p["opus55"]
        if (a == b == "CONTINUES") if kind == "consensus" else ("CONTINUES" in (a, b)):
            S.setdefault(p["ctx_index"], []).append(p["followup_index"])
    return {i: sorted(v) for i, v in S.items() if len(v) >= 2}


# ---------------------------------------------------------------- core (pure arrays; exercised by --check)
def per_context(C, F, Dv, Ppool, src: dict, n_ctx: int):
    """src[arm] = (rows, V). Returns M[arm][measure] (n_ctx x 3), as revision_neutral.analyse.
    Dv[i]: distractor vectors for hit; Ppool[i]: future-split pool outside the saved follow-ups, sorted by
    similarity to the context (nearest first)."""
    M = {a: {m: np.full((n_ctx, 3), np.nan) for m in MEASURES} for a in src}
    for i in range(n_ctx):
        c, Fi = C[i], F[i]
        t = unit(tang(Fi, c).mean(0))
        tp = unit(tang(Ppool[i][:len(Fi)], c).mean(0))
        ts = unit(t - (t @ tp) * tp)
        for a, (rows, V) in src.items():
            for r in range(3):
                idx = [n for n, x in enumerate(rows) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                if not idx:
                    continue
                X = V[idx]
                sf = X @ Fi.T
                T = tang(X, c)
                M[a]["step"][i, r] = (X @ c).mean()
                M[a]["fit"][i, r] = sf.max(1).mean()
                M[a]["hit"][i, r] = (sf.max(1) > (X @ Dv[i].T).max(1)).mean()
                M[a]["align"][i, r] = (T @ t).mean()
                M[a]["p"][i, r] = (T @ tp).mean()
                M[a]["s"][i, r] = (T @ ts).mean()
    return M


def tests_from(M, rng):
    from scipy import stats

    def boot(x):
        return [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]

    def wil(x, alt="two-sided"):
        return float(stats.wilcoxon(x, alternative=alt).pvalue) if len(x) >= 2 and np.any(x != 0) else None

    def mde(sd, n):
        t = float(stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1))
        return {"n": n, "sd": sd, "t": t * sd / np.sqrt(n), "wilcoxon_approx": t * sd / np.sqrt(n) / np.sqrt(3 / np.pi)}

    cell, D = {}, {}
    for a in M:
        e = {f"round{r}": {m: float(np.nanmean(M[a][m][:, r])) for m in MEASURES} for r in range(3)}
        D[a] = {}
        for m in MEASURES:
            d = M[a][m][:, 0] - M[a][m][:, 2] if m == "step" else M[a][m][:, 2] - M[a][m][:, 0]
            D[a][m] = d
            ok = ~np.isnan(d)
            e["step_reduction" if m == "step" else f"change_{m}"] = {
                "mean": float(d[ok].mean()), "ci": boot(d[ok]), "p": wil(d[ok]), "n": int(ok.sum())}
        cell[a] = e
    for a in ARMS:  # share of each critique arm's change reproduced by NEUTRAL (descriptive)
        cell[a]["share_reproduced_by_NEUTRAL"] = {
            m: float(np.nanmean(D["NEUTRAL"][m]) / np.nanmean(D[a][m])) if np.nanmean(D[a][m]) != 0 else None
            for m in ("step", "s")}
    spec = [("O1", "GEO", "step", "greater"), ("O2", "LLM", "step", "greater")]
    for nm, m in (("O3", "fit"), ("O4", "hit"), ("O5", "align"), ("O6", "s")):
        spec += [(nm, a, m, "two-sided") for a in ARMS]
    explore = [("X_p", a, "p", "two-sided") for a in ARMS] + [("X_step", "ORACLE", "step", "two-sided")]
    tests, p = {}, {}
    for fam, rows_ in (("prereg", spec), ("exploratory", explore)):
        for nm, a, m, alt in rows_:
            d = D[a][m] - D["NEUTRAL"][m]
            ok = ~np.isnan(d)
            key = f"{nm}:{a}-NEUTRAL:{m}"
            tests[key] = {"family": fam, "diff_mean": float(d[ok].mean()), "ci": boot(d[ok]), "p": wil(d[ok], alt),
                          "alternative": alt, "n": int(ok.sum())}
            if nm == "O6":
                tests[key]["mde_80pct_two_sided_005"] = mde(float(d[ok].std(ddof=1)), int(ok.sum()))
            if fam == "prereg":
                p[key] = tests[key]["p"] if tests[key]["p"] is not None else 1.0
    assert len(p) == 14
    for k, v in holm(p).items():
        tests[k]["p_holm"] = v
    for k, t in tests.items():  # interpretation rules of the pre-registration
        if k.startswith("O6"):
            sig = t["p_holm"] < 0.05
            t["reading"] = ("feedback-specific turn" if sig and t["diff_mean"] > 0 else
                            "critique turns less than revision" if sig else
                            "consistent with no turn >= 0.015" if -0.015 < t["ci"][0] and t["ci"][1] < 0.015 else
                            "insufficient")
        elif k.startswith(("O1", "O2")):
            t["reading"] = "holds" if t["p_holm"] < 0.05 else "not detected"
        elif t["family"] == "prereg":
            t["reading"] = "differs from revision" if t["p_holm"] < 0.05 else "not detected"
    cell["tests"] = tests
    return cell


def f4_cell(C, F, Ppool_sorted, src: dict, sets: dict, pk: str, rng):
    from scipy import stats

    def boot(x):
        return [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]

    def wil(x):
        return float(stats.wilcoxon(x).pvalue) if len(x) >= 2 and np.any(x != 0) else None

    ch = {a: {"cont": [], "orig": []} for a in src}
    loo, fp = [], []
    for i, js in sets.items():
        c = C[i]
        TF = tang(F[i], c)
        TC = TF[js]
        S_ = TC.sum(0)
        loo.append(float(np.mean([TC[j] @ unit(S_ - TC[j]) for j in range(len(TC))])))
        fc, fo = unit(TC.mean(0)), unit(TF.mean(0))
        kk = len(js) if pk == "k_continuations" else len(F[i])
        tpc = unit(tang(Ppool_sorted[i][:kk], c).mean(0))
        tpo = unit(tang(Ppool_sorted[i][:len(F[i])], c).mean(0))
        sc_, so_ = unit(fc - (fc @ tpc) * tpc), unit(fo - (fo @ tpo) * tpo)
        fp.append(float(fc @ tpc))
        for a, (rows, V) in src.items():
            v = {}
            for r in (0, 2):
                idx = [n for n, x in enumerate(rows) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                if idx:
                    T = tang(V[idx], c)
                    v[r] = (float((T @ sc_).mean()), float((T @ so_).mean()))
            ok = 0 in v and 2 in v
            ch[a]["cont"].append(v[2][0] - v[0][0] if ok else np.nan)
            ch[a]["orig"].append(v[2][1] - v[0][1] if ok else np.nan)
    cell = {"n_contexts": len(sets), "contexts": sorted(sets), "continuations_loo_alignment": float(np.mean(loo)),
            "mean_f_dot_p": float(np.mean(fp))}
    for a in src:
        x, y = np.array(ch[a]["cont"]), np.array(ch[a]["orig"])
        ok = ~np.isnan(x)
        e = {"n": int(ok.sum())}
        if ok.sum() >= 2:
            e |= {"change_along_s_cont": float(x[ok].mean()), "ci": boot(x[ok]), "p": wil(x[ok]),
                  "change_along_s_original_same_contexts": float(y[ok].mean()), "ci_original": boot(y[ok]),
                  "p_original": wil(y[ok])}
        if a != "NEUTRAL":
            d = x - np.array(ch["NEUTRAL"]["cont"])
            ok2 = ~np.isnan(d)
            if ok2.sum() >= 2:
                ci = boot(d[ok2])
                e["minus_NEUTRAL"] = {"mean": float(d[ok2].mean()), "ci": ci, "p_uncorrected": wil(d[ok2]),
                                      "n": int(ok2.sum()), "ci_within_pm0.015": -0.015 < ci[0] and ci[1] < 0.015}
        cell[a] = e
    return cell


# ---------------------------------------------------------------- real analysis
def analyse() -> None:
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import continuation_reanalysis as R

    nrows = neutral_rows(KOUT / "iter_olmo_neutral.jsonl")
    allrows = [json.loads(l) for l in (KOUT / "iter_olmo_neutral.jsonl").read_text().splitlines()]
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "primary", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut_t, notpast = (meta["split"] == "future").to_numpy(), (meta["split"] != "past").to_numpy()
    ids_t, ids_h = meta.loc[fut_t, "id"].to_numpy(), meta.loc[notpast, "id"].to_numpy()
    prim_h = meta.loc[notpast, "primary"].to_numpy()
    by_id = meta.set_index("id")
    rng = np.random.default_rng(SEED)
    res = {"prereg": "docs/prereg/strengthen_olmo_neutral.md", "deviations": [],
           "n_neutral_rows_total": len(allrows), "n_neutral_rows_parsed": len(nrows),
           "n_parsed_by_round": {r: sum(x["round"] == r for x in nrows) for r in range(3)}}
    sets = {k: consensus_sets(k) for k in ("consensus", "lenient")}
    assert sorted(sets["consensus"]) == F4_CONTEXTS, "the 17 F4 contexts changed"
    for space in SPACES:
        NV = unit(np.load(KOUT / f"neutral_emb_{space}.npy").astype(np.float32))
        assert len(NV) == len(nrows)
        CN = unit(np.load(KOUT / f"ctx_emb_{space}.npy").astype(np.float32))
        ctx, C, F, _pb, _PB, it, IT = R.load_generator("olmo-7b-0424", space)
        n = len(ctx)
        geo0 = {x["ctx"]: IT[j] for j, x in enumerate(it) if x["arm"] == "GEO" and x["round"] == 0}
        a_r0 = [float(NV[j] @ geo0[r["ctx"]]) for j, r in enumerate(nrows) if r["round"] == 0]
        a_c = (CN * C).sum(1)
        agree = {"r0_neutral_vs_geo": {"min": float(np.min(a_r0)), "mean": float(np.mean(a_r0)), "n": len(a_r0)},
                 "contexts": {"min": float(a_c.min()), "mean": float(a_c.mean()), "n": int(len(a_c))}}
        if min(agree["r0_neutral_vs_geo"]["min"], agree["contexts"]["min"]) < 0.999:
            res["deviations"].append(f"{space}: new-vs-old encoder agreement below 0.999 ({agree})")
        emb = unit(np.load(DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep])
        Ft, Fh = emb[fut_t], emb[notpast]
        del emb
        Dv, Pp = [], []
        for i in range(n):
            fids = list({f["arxiv_id"] for f in ctx[i]["followups"]})
            Dv.append(Fh[(prim_h == by_id.loc[ctx[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, fids)])
            pool = Ft[~np.isin(ids_t, fids)]
            Pp.append(pool[np.argsort(-(pool @ C[i]))[:max(len(F[i]), 60)]])
        src = {a: (it, IT) for a in ARMS} | {"NEUTRAL": (nrows, NV)}
        cell = tests_from(per_context(C, F, Dv, Pp, src, n), rng)
        cell["encoder_agreement_new_vs_v1"] = agree
        cell["F4"] = {f"{kind}/{pk}": f4_cell(C, F, Pp, src, sets[kind], pk, rng)
                      for kind in ("consensus", "lenient") for pk in ("k_continuations", "k_followups")}
        cell["note"] = ("step: reduction r0 - r2 (positive = longer step); other measures r2 - r0. Diffs are critique "
                        "change minus NEUTRAL change, paired by context. Holm over the 14 pre-registered tests within "
                        "this embedder (Qwen3 primary; BGE-M3 robustness). F4 descriptive only.")
        res[space] = cell
        print(space, json.dumps({k: (round(v["diff_mean"], 4), v.get("p_holm"), v.get("reading"))
                                 for k, v in cell["tests"].items() if v["family"] == "prereg"}), flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT)


# ---------------------------------------------------------------- --check
def check() -> None:
    """Synthetic data with a planted effect (no corpus, no model) + structural checks on the existing files."""
    rng = np.random.default_rng(0)
    n, d, nf = 60, 32, 8
    C = unit(rng.normal(size=(n, d)))
    F = [unit(C[i] + 0.9 * unit(rng.normal(size=(nf, d)))) for i in range(n)]
    Dv = [unit(C[i] + 1.2 * unit(rng.normal(size=(40, d)))) for i in range(n)]
    Pp = [unit(C[i] + 1.0 * unit(rng.normal(size=(60, d)))) for i in range(n)]
    rows, V = [], []
    for a in ALL:
        for i in range(n):
            x0 = unit(C[i] + 0.4 * unit(rng.normal(size=d)))
            t = unit(tang(F[i], C[i]).mean(0))
            for r in range(3):
                # GEO: planted longer step and planted turn towards the follow-ups' tangent; NEUTRAL: neither
                push = (0.25 if a == "GEO" else 0.05) * r
                x = unit(x0 - (0.15 * r * C[i] if a == "GEO" else 0) + push * t + 0.02 * rng.normal(size=d))
                rows.append({"ctx": i, "arm": a, "k": 0, "round": r})
                V.append(x)
    V = np.array(V)
    src = {a: (rows, V) for a in ALL}
    cell = tests_from(per_context(C, F, Dv, Pp, src, n), np.random.default_rng(SEED))
    T = cell["tests"]
    assert T["O1:GEO-NEUTRAL:step"]["diff_mean"] > 0 and T["O1:GEO-NEUTRAL:step"]["p_holm"] < 0.05, T["O1:GEO-NEUTRAL:step"]
    assert T["O5:GEO-NEUTRAL:align"]["diff_mean"] > 0 and T["O5:GEO-NEUTRAL:align"]["p_holm"] < 0.05
    assert T["O6:LLM-NEUTRAL:s"]["p_holm"] > 0.05, T["O6:LLM-NEUTRAL:s"]
    assert sum(t["family"] == "prereg" for t in T.values()) == 14
    sets = {i: [0, 1, 2] for i in range(0, n, 4)}
    f4 = f4_cell(C, F, Pp, src, sets, "k_continuations", np.random.default_rng(SEED))
    assert f4["n_contexts"] == len(sets) and "minus_NEUTRAL" in f4["GEO"]
    print("synthetic ok:", {k: (round(v["diff_mean"], 4), round(v["p_holm"], 4), v["reading"])
                            for k, v in T.items() if v["family"] == "prereg"})

    # structural checks on real inputs (no corpus loaded)
    assert sorted(consensus_sets("consensus")) == F4_CONTEXTS
    print("F4 contexts:", len(F4_CONTEXTS), "consensus; lenient >= 2:", len(consensus_sets("lenient")))
    it = [json.loads(l) for l in (V1 / "iter_llama.jsonl").read_text().splitlines()]
    geo0 = {r["ctx"]: r for r in it if r["arm"] == "GEO" and r["round"] == 0}
    up = DATA / "kaggle-upload/olmo-neutral-inputs/geo_r0.jsonl"
    for r in map(json.loads, up.read_text().splitlines()):
        assert (r["title"], r["abstract"]) == (geo0[r["ctx"]]["title"], geo0[r["ctx"]]["abstract"])
    print("geo_r0.jsonl matches iter_llama.jsonl GEO round 0 (150 rows)")
    if (KOUT / "iter_olmo_neutral.jsonl").exists():
        nr = neutral_rows(KOUT / "iter_olmo_neutral.jsonl")
        for space in SPACES:
            p = KOUT / f"neutral_emb_{space}.npy"
            if p.exists():
                assert len(np.load(p)) == len(nr), space
        r0 = [r for r in nr if r["round"] == 0]
        assert all((r["title"], r["abstract"]) == (geo0[r["ctx"]]["title"], geo0[r["ctx"]]["abstract"]) for r in r0)
        print("kernel output present:", len(nr), "parsed rows; round 0 = GEO round 0")
    else:
        print("kernel output not downloaded yet:", KOUT)
    print("check ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"analyse": analyse, "--check": check}[arg]()
