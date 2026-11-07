"""Analysis of the full census (docs/prereg/strengthen_continuations_full.md). Run via
`OMP_NUM_THREADS=4 uv run python scripts/strengthen_continuations_full.py analyse`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402
import strengthen_continuations as S  # noqa: E402
import strengthen_continuations_full as F_  # noqa: E402

SEED, N_BOOT, LABELS, WORK = S.SEED, S.N_BOOT, S.LABELS, S.WORK
OLD = S.OUT  # first study's output, read only


def labels3(who: str) -> dict:
    out = {}
    for f in sorted(F_.D3.glob(f"labels_{who}_batch_*.json")):
        for r in json.loads(f.read_text())["rows"]:
            out[r["id"]] = r
    n = len(json.loads((F_.D3 / "pairs.json").read_text()))
    assert len(out) == n, ("stage3", who, len(out), n)
    return out


def analyse() -> None:
    from scipy import stats
    import pandas as pd
    import continuation_reanalysis as R
    from direction_tangent import tang, unit
    import revision_neutral as RN

    ctx = M.contexts()
    rng = np.random.default_rng(SEED)
    old = json.loads(OLD.read_text())
    res: dict = {"prereg": "docs/prereg/strengthen_continuations_full.md", "deviations": []}

    # ---- all pairs, all stages
    all_pairs = []
    for s in (1, 2, 3):
        P = json.loads((WORK / f"stage{s}/pairs.json").read_text())
        lc, ll = (S.labels(s, "codex"), S.labels(s, "claude")) if s < 3 else (labels3("codex"), labels3("claude"))
        for p in P:
            all_pairs.append({k: p[k] for k in ("id", "ctx_index", "followup_index", "context_id", "followup_id",
                                                 "source_title", "followup_title")}
                             | {"stage": s, "gpt61sol": lc[p["id"]]["label"], "gpt61sol_reason": lc[p["id"]]["reason"],
                                "opus55": ll[p["id"]]["label"], "opus55_reason": ll[p["id"]]["reason"]})
    keys = {(p["ctx_index"], p["followup_index"]) for p in all_pairs}
    N = sum(len(c["followups"]) for c in ctx)
    assert len(keys) == len(all_pairs) == N == 2883
    all_pairs.sort(key=lambda p: (p["ctx_index"], p["followup_index"]))
    F_.OUT_LABELS.write_text(json.dumps(all_pairs, indent=1) + "\n")
    res["labelling_attempts_stage3"] = {w: F_.attempt_counts(w) for w in ("codex", "claude")}
    res["labelling_attempts_stage3"]["labeller_names"] = {"codex": "gpt-6.1-sol", "claude": "claude-opus-5-5"}
    res["n_pairs_by_stage"] = {s: sum(p["stage"] == s for p in all_pairs) for s in (1, 2, 3)}

    A = np.array([p["gpt61sol"] for p in all_pairs]); B = np.array([p["opus55"] for p in all_pairs])
    ci = np.array([p["ctx_index"] for p in all_pairs]); st = np.array([p["stage"] for p in all_pairs])
    uctx = np.unique(ci)
    groups = {i: np.where(ci == i)[0] for i in uctx}
    boots = [np.concatenate([groups[i] for i in rng.choice(uctx, len(uctx))]) for _ in range(N_BOOT)]
    pct = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731

    # ---- F1 share (census)
    defs = {"gpt61sol_CONTINUES": A == "CONTINUES", "opus55_CONTINUES": B == "CONTINUES",
            "consensus_CONTINUES": (A == "CONTINUES") & (B == "CONTINUES"),
            "lenient_CONTINUES": (A == "CONTINUES") | (B == "CONTINUES"),
            "consensus_CONTINUES_or_USES": np.isin(A, ["CONTINUES", "USES"]) & np.isin(B, ["CONTINUES", "USES"]),
            "gpt61sol_CONTINUES_or_USES": np.isin(A, ["CONTINUES", "USES"]),
            "opus55_CONTINUES_or_USES": np.isin(B, ["CONTINUES", "USES"]),
            "both_BACKGROUND": (A == "BACKGROUND") & (B == "BACKGROUND")}
    F1 = {}
    for name, x in defs.items():
        xf = x.astype(float)
        per_ctx = np.array([xf[groups[i]].sum() for i in uctx])
        F1[name] = {"k": int(x.sum()), "n": N, "share_census": float(xf.mean()),
                    "superpopulation_cluster_ci": pct([xf[b].mean() for b in boots]),
                    "contexts_with_>=1": int((per_ctx >= 1).sum()), "contexts_with_>=2": int((per_ctx >= 2).sum()),
                    "contexts_with_>=3": int((per_ctx >= 3).sum()),
                    "share_by_stage": {int(s): float(xf[st == s].mean()) for s in (1, 2, 3)}}
    F1["stage1_weighted_estimate_from_first_study"] = {
        "consensus_CONTINUES": old["E1_share_stage1"]["consensus_CONTINUES"]["share_followup_weighted"],
        "ci": old["E1_share_stage1"]["consensus_CONTINUES"]["share_followup_weighted_ci"]}
    res["F1_share_census"] = F1
    res["label_counts_census"] = {"gpt61sol": {k: int((A == k).sum()) for k in LABELS},
                                  "opus55": {k: int((B == k).sum()) for k in LABELS}}
    res["confusion_census"] = {a: {b: int(((A == a) & (B == b)).sum()) for b in LABELS} for a in LABELS}

    # ---- F2 agreement
    def agree(m):
        bA, bB = A[m] == "CONTINUES", B[m] == "CONTINUES"
        bA2, bB2 = np.isin(A[m], ["CONTINUES", "USES"]), np.isin(B[m], ["CONTINUES", "USES"])
        return {"n": int(m.sum()),
                "binary_CONTINUES": {"kappa": S.kappa(bA, bB), "raw_agreement": float(np.mean(bA == bB))},
                "binary_CONTINUES_or_USES": {"kappa": S.kappa(bA2, bB2), "raw_agreement": float(np.mean(bA2 == bB2))},
                "four_labels": {"kappa": S.kappa(A[m], B[m]), "raw_agreement": float(np.mean(A[m] == B[m]))}}
    F2 = agree(np.ones(N, bool))
    bA, bB = A == "CONTINUES", B == "CONTINUES"
    bA2, bB2 = np.isin(A, ["CONTINUES", "USES"]), np.isin(B, ["CONTINUES", "USES"])
    F2["binary_CONTINUES"]["ci"] = pct([S.kappa(bA[b], bB[b]) for b in boots])
    F2["binary_CONTINUES_or_USES"]["ci"] = pct([S.kappa(bA2[b], bB2[b]) for b in boots])
    F2["four_labels"]["ci"] = pct([S.kappa(A[b], B[b]) for b in boots])
    F2["by_stage"] = {int(s): agree(st == s) for s in (1, 2, 3)}
    res["F2_agreement_census"] = F2

    # ---- F5 overlap with the S2 flags and the original 30-pair check
    oldchk = {(p["ctx_index"], p["followup_index"]): p["label"]
              for p in json.loads((S.ROOT / "docs/paper/followup_check_sample.json").read_text())["pairs"]}
    intents = json.loads((M.RUN / "followup_intents.json").read_text())

    def s2flag(i, j):
        v = intents.get(f"{i}/{j}", {})
        return bool(v) and bool(v.get("influential") or {"methodology", "result"} & set(v.get("intents", [])))
    flag = np.array([s2flag(p["ctx_index"], p["followup_index"]) for p in all_pairs])
    nflag = {i: int(flag[groups[i]].sum()) for i in uctx}
    kept = {i for i, n in nflag.items() if n >= 3}
    insub = flag & np.isin(ci, list(kept))
    cons = defs["consensus_CONTINUES"]
    res["F5_s2_overlap"] = {
        "s2_subset_retained_contexts": len(kept), "pairs_in_s2_subset_as_analysed": int(insub.sum()),
        "consensus_in_s2_subset": int((insub & cons).sum()),
        "share_consensus_in_s2_subset": float(cons[insub].mean()),
        "flagged_pairs_all_contexts": int(flag.sum()), "consensus_among_flagged": int((flag & cons).sum()),
        "share_consensus_among_flagged": float(cons[flag].mean()),
        "share_consensus_among_not_flagged": float(cons[~flag].mean()),
        "share_of_consensus_that_are_flagged": float(flag[cons].mean()) if cons.any() else None,
        "share_of_consensus_in_s2_subset": float(insub[cons].mean()) if cons.any() else None,
        "original_30_check": [{"ctx_followup": f"{p['ctx_index']}/{p['followup_index']}",
                               "original_claude": oldchk[(p["ctx_index"], p["followup_index"])],
                               "gpt61sol": p["gpt61sol"], "opus55": p["opus55"]}
                              for p in all_pairs if (p["ctx_index"], p["followup_index"]) in oldchk]}

    # ---- continuation sets
    def sets(kind):
        Sx = {i: [] for i in range(len(ctx))}
        for p in all_pairs:
            a, b = p["gpt61sol"], p["opus55"]
            ok = (a == b == "CONTINUES") if kind == "consensus" else ("CONTINUES" in (a, b)) if kind == "lenient" \
                else (a == b == "BACKGROUND")
            if ok:
                Sx[p["ctx_index"]].append(p["followup_index"])
        return {i: sorted(v) for i, v in Sx.items() if v}
    CS = {k: sets(k) for k in ("consensus", "lenient", "background")}
    res["n_contexts_with_continuations"] = {
        k: {">=1": len(v), ">=2": sum(len(x) >= 2 for x in v.values()), ">=3": sum(len(x) >= 3 for x in v.values())}
        for k, v in CS.items() if k != "background"}
    res["n_consensus_continuations_total"] = int(cons.sum())
    old18 = set()
    for p in json.loads(S.OUT_LABELS.read_text()):
        if p["gpt61sol"] == p["opus55"] == "CONTINUES":
            old18.add(p["ctx_index"])
    assert len(old18) == 18
    res["earlier_18_contexts"] = sorted(old18)
    res["new_consensus_contexts"] = sorted(set(CS["consensus"]) - old18)
    res["earlier_18_still_with_consensus"] = len(old18 & set(CS["consensus"]))

    def boot_ci(x):
        x = np.asarray(x, float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    def wil(x, alt="two-sided"):
        x = np.asarray(x)
        return float(stats.wilcoxon(x, alternative=alt).pvalue) if len(x) >= 2 and np.any(x != 0) else None

    def mde(sd, n):
        t = float(stats.t.ppf(0.95, n - 1) + stats.t.ppf(0.80, n - 1))
        z = float(stats.norm.ppf(0.95) + stats.norm.ppf(0.80))
        return {"n": n, "sd": sd, "t": t * sd / np.sqrt(n), "normal": z * sd / np.sqrt(n),
                "wilcoxon_approx": t * sd / np.sqrt(n) / np.sqrt(3 / np.pi)}

    # ---- F3 step gap
    F3 = {}
    for space in R.SPACES:
        for gen in ("gpt-4o", "olmo-7b-0424"):
            cctx, C, F, pb, PB, it, IT = R.load_generator(gen, space)
            for kind in ("consensus", "lenient"):
                g_c, g_all, rows = [], [], []
                for i, js in CS[kind].items():
                    idx = [n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == "CTX"]
                    if not idx:
                        continue
                    ideas = float(np.median(PB[idx] @ C[i]))
                    g_c.append(ideas - float(np.median(F[i][js] @ C[i])))
                    g_all.append(ideas - float(np.median(F[i] @ C[i])))
                    rows.append(i)
                g_c, g_all, rows = np.array(g_c), np.array(g_all), np.array(rows)

                def cell_of(m):
                    gc, ga = g_c[m], g_all[m]
                    ref = float(ga.mean()) / 2
                    c_ = {"n_contexts": int(m.sum()), "gap_vs_continuations": float(gc.mean()), "ci": boot_ci(gc),
                          "median_gap": float(np.median(gc)), "sd_gap": float(gc.std(ddof=1)),
                          "p_one_sided_gt0": wil(gc, "greater"), "share_contexts_gap_gt0": float(np.mean(gc > 0)),
                          "gap_vs_all_followups_same_contexts": float(ga.mean()), "ci_all": boot_ci(ga),
                          "difference_cont_minus_all": float((gc - ga).mean()), "ci_difference": boot_ci(gc - ga),
                          "p_difference": wil(gc - ga), "reference_effect_half_all_gap": ref}
                    c_["mde_80pct_one_sided_005"] = mde(c_["sd_gap"], c_["n_contexts"])
                    c_["mde_80pct_if_n18_same_sd"] = mde(c_["sd_gap"], 18)
                    p = c_["p_one_sided_gt0"]
                    c_["reading"] = ("holds" if p is not None and p < 0.05 else
                                     "contradicts gap >= reference" if c_["ci"][1] < ref else "insufficient")
                    return c_
                cell = cell_of(np.ones(len(rows), bool))
                cell["contexts"] = rows.tolist()
                cell["per_context_gap"] = [round(float(x), 5) for x in g_c]
                if kind == "consensus":
                    newm = ~np.isin(rows, list(old18))
                    cell["subset_new_contexts"] = cell_of(newm) if newm.sum() >= 2 else {"n_contexts": int(newm.sum())}
                    cell["subset_earlier_18"] = cell_of(~newm)
                    d = [float((F[i][js] @ C[i]).mean() - (F[i][CS["background"][i]] @ C[i]).mean())
                         for i, js in CS["consensus"].items() if i in CS["background"]]
                    cell["T2_continuation_minus_background_step"] = {
                        "n_contexts": len(d), "mean": float(np.mean(d)) if d else None,
                        "ci": boot_ci(d) if len(d) > 1 else None, "p": wil(d),
                        "share_contexts_closer": float(np.mean(np.array(d) > 0)) if d else None}
                F3[f"{gen}/{space}/{kind}"] = cell
                print("F3", gen, space, kind, json.dumps({k: v for k, v in cell.items()
                                                         if not isinstance(v, (list, dict))}), flush=True)
    res["F3_step_gap"] = F3
    res["F3_primary"] = {k: F3["gpt-4o/qwen3e06/consensus"][k] for k in (
        "n_contexts", "gap_vs_continuations", "ci", "p_one_sided_gt0", "reference_effect_half_all_gap", "reading",
        "mde_80pct_one_sided_005", "mde_80pct_if_n18_same_sd")}

    # ---- F4 direction from continuations
    n_ge2 = res["n_contexts_with_continuations"]["consensus"][">=2"]
    res["F4_confirmatory"] = n_ge2 >= 10
    meta = pd.read_parquet(M.DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    ids = meta.loc[fut, "id"].to_numpy()
    nrows = RN.neutral_rows()
    F4 = {}
    for space in R.SPACES:
        Fe = R.norm(np.load(M.DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep][fut])
        NV = R.norm(np.load(M.RUN / f"neutral_emb_{space}.npy").astype(np.float32))
        for gen in ("gpt-4o", "olmo-7b-0424"):
            cctx, C, F, pb, PB, it, IT = R.load_generator(gen, space)
            arms = R.I_ARMS + (("NEUTRAL",) if gen == "gpt-4o" else ())
            for kind in ("consensus", "lenient"):
                for pk in ("k_continuations", "k_followups"):
                    elig = {i: js for i, js in CS[kind].items() if len(js) >= 2}
                    ch = {a: {"cont": [], "orig": []} for a in arms}
                    loo, fp = [], []
                    for i, js in elig.items():
                        c = C[i]
                        TF = tang(F[i], c)
                        TC = TF[js]
                        S_ = TC.sum(0)
                        loo.append(float(np.mean([TC[j] @ unit(S_ - TC[j]) for j in range(len(TC))])))
                        fc, fo = unit(TC.mean(0)), unit(TF.mean(0))
                        fids = {f["arxiv_id"] for f in cctx[i]["followups"]}
                        pool = Fe[~np.isin(ids, list(fids))]
                        order = np.argsort(-(pool @ c))
                        kk = len(js) if pk == "k_continuations" else len(F[i])
                        tpc = unit(tang(pool[order[:kk]], c).mean(0))
                        tpo = unit(tang(pool[order[:len(F[i])]], c).mean(0))
                        sc_, so_ = unit(fc - (fc @ tpc) * tpc), unit(fo - (fo @ tpo) * tpo)
                        fp.append(float(fc @ tpc))
                        for a in arms:
                            src, V = (nrows, NV) if a == "NEUTRAL" else (it, IT)
                            v = {}
                            for r in (0, 2):
                                idx = [n for n, x in enumerate(src) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                                if idx:
                                    T = tang(V[idx], c)
                                    v[r] = (float((T @ sc_).mean()), float((T @ so_).mean()))
                            if 0 in v and 2 in v:
                                ch[a]["cont"].append(v[2][0] - v[0][0]); ch[a]["orig"].append(v[2][1] - v[0][1])
                            else:
                                ch[a]["cont"].append(np.nan); ch[a]["orig"].append(np.nan)
                    cell = {"n_contexts": len(elig), "continuations_loo_alignment": float(np.mean(loo)) if loo else None,
                            "mean_f_dot_p": float(np.mean(fp)) if fp else None}
                    for a in arms:
                        x, y = np.array(ch[a]["cont"]), np.array(ch[a]["orig"])
                        ok = ~np.isnan(x)
                        e = {"n": int(ok.sum())}
                        if ok.sum() >= 2:
                            e |= {"change_along_s_cont": float(x[ok].mean()), "ci": boot_ci(x[ok]), "p": wil(x[ok]),
                                  "change_along_s_original_same_contexts": float(y[ok].mean()),
                                  "ci_original": boot_ci(y[ok]), "p_original": wil(y[ok])}
                        if a != "NEUTRAL" and "NEUTRAL" in arms:
                            d = x - np.array(ch["NEUTRAL"]["cont"])
                            ok2 = ~np.isnan(d)
                            if ok2.sum() >= 2:
                                e["minus_NEUTRAL"] = {"mean": float(d[ok2].mean()), "ci": boot_ci(d[ok2]),
                                                      "p": wil(d[ok2]), "n": int(ok2.sum())}
                        cell[a] = e
                    F4[f"{gen}/{space}/{kind}/{pk}"] = cell
                    print("F4", gen, space, kind, pk, cell["n_contexts"],
                          {a: (round(cell[a].get("change_along_s_cont", np.nan), 4), cell[a].get("p")) for a in arms},
                          flush=True)
    res["F4_direction"] = F4
    if res["F4_confirmatory"]:
        prim = F4["gpt-4o/qwen3e06/consensus/k_continuations"]
        arms = [a for a in R.I_ARMS if "minus_NEUTRAL" in prim[a]]
        ps = [prim[a]["minus_NEUTRAL"]["p"] for a in arms]
        order = np.argsort(ps)
        holm, run = [None] * len(ps), 0.0
        for r, k in enumerate(order):
            run = max(run, min(1.0, (len(ps) - r) * ps[k]))
            holm[k] = run
        res["F4_confirmatory_contrasts"] = {
            a: prim[a]["minus_NEUTRAL"] | {"p_holm": holm[k], "significant_holm": holm[k] < 0.05,
                                           "ci_within_pm0.015": -0.015 < prim[a]["minus_NEUTRAL"]["ci"][0]
                                           and prim[a]["minus_NEUTRAL"]["ci"][1] < 0.015}
            for k, a in enumerate(arms)}
    F_.OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("F1_share_census", "n_contexts_with_continuations", "F3_primary",
                                          "F4_confirmatory")}, indent=1, default=str))
