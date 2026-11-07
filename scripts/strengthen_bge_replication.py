"""BGE-M3 replication of the Table 6 results marked "not run" (docs/prereg/strengthen_bge_replication.md).

  OMP_NUM_THREADS=2 uv run python scripts/strengthen_bge_replication.py [anchoring|modern|continuation|all]

Needs runs/strengthen-bge/emb_bgem3.npy (scripts/strengthen_bge_embed.py) for parts 1 and 2; part 3 uses only the
existing Kaggle BGE-M3 vectors. Never calls an LLM. Output: docs/prereg/strengthen_bge_replication.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import anchoring_causal as AC  # noqa: E402
import continue_from_context as M  # noqa: E402
import continuation_reanalysis as R  # noqa: E402

DATA = M.DATA
SB = DATA / "runs/strengthen-bge"
OUT = HERE.parent / "docs/prereg/strengthen_bge_replication.json"
SEED, N_BOOT = 20261030, 2000


def norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def save(key, value):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[key] = value
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def local():
    rows = json.loads((SB / "texts.json").read_text())
    V = norm(np.load(SB / "emb_bgem3.npy"))
    return {(r["group"], r["key"]): V[n] for n, r in enumerate(rows)}, {(r["group"], r["key"]): r["text"] for r in rows}


def holm(p):
    keys = sorted(p, key=p.get)
    out, run = {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        out[k] = run
    return out


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    return [float(np.nanpercentile([np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]


# ---------------------------------------------------------------- Part 1: causal anchoring
def anchoring():
    L, _ = local()
    qwen = json.loads((HERE.parent / "docs/prereg/anchoring_causal_result.json").read_text())
    qcell = {(c["domain"], c["gen"], c["arm"]): c for c in qwen["cells"]}
    cells = []
    for domain in AC.DOMAINS_ORDER:
        corpus, P, tg = AC.load_domain(domain)  # Qwen3, for the frozen neighbour sets, donors and REAL targets
        meta = pd.read_parquet(DATA / "prepared" / domain / "corpus.parquet", columns=["split", "adult"])
        keep = (~meta.adult & meta.split.eq("past")).to_numpy()
        B = norm(np.load(DATA / "prepared" / domain / "emb_bgem3.npy", mmap_mode="r")[keep].astype(np.float32))
        for arm in AC.ARMS:
            T, nn, _, _, donors = AC.geometry(corpus, P, tg, domain, arm)
            Cb = {k: norm(B[nn[k]].mean(0)) for k in nn}
            tb = {}
            if arm == "REAL":
                for k in AC.KS:
                    s = P @ T[k]
                    j = int(np.argmax(s))
                    assert s[j] >= 0.999, (domain, k, s[j])
                    tb[k] = B[j]
            for gen in AC.GENERATORS:
                E = {}
                for cond in AC.CONDITIONS:
                    for r in AC.read_rows(DATA / "runs" / domain / AC.run_name(gen, cond, arm) / "items.jsonl"):
                        k = r["k"]
                        exp = [] if cond == "none" else nn[donors[k] if cond == "swap" else k]
                        assert r["neighbours"] == exp, (domain, gen, cond, arm, k)
                        if r.get("best"):
                            E[cond, k] = L["anchoring", f"{domain}/{gen}/{cond}/{arm}/{k}"]
                complete = [k for k in AC.KS if all((c, k) in E for c in AC.CONDITIONS)]
                cell = {"domain": domain, "gen": gen, "arm": arm, "n_complete": len(complete)}
                A = [(E["swap", k] - E["true", k]) @ (Cb[donors[k]] - Cb[k]) for k in complete]
                cell["A"] = AC.interval(A)
                cell["A_replicates"] = bool(cell["A"]["low"] > 0)
                cell["A_ge_frozen_min_0.05"] = bool(cell["A"]["mean"] >= AC.ANCHOR_MIN)
                cell["centroid_contrast_sq_mean"] = float(np.mean([np.sum((Cb[donors[k]] - Cb[k]) ** 2) for k in complete]))
                q = qcell[(domain, gen, arm)]
                cell["qwen3"] = {"A": q["attraction"], "D": q["drop"], "verdict": q["verdict"]}
                cell["qwen3_centroid_contrast_sq_mean"] = float(np.mean([np.sum((AC.unit(P[nn[donors[k]]].mean(0)) - AC.unit(P[nn[k]].mean(0))) ** 2) for k in complete]))
                if arm == "REAL":
                    D = [E["true", k] @ tb[k] - E["swap", k] @ tb[k] for k in complete]
                    cell["D"] = AC.interval(D)
                    cell["D_replicates"] = bool(cell["D"]["low"] > 0 and cell["D"]["mean"] >= AC.DROP_MIN[arm])
                    cell["verdict"] = AC.verdict(cell["D"], cell["A"], arm)
                    cell["e_dot_t"] = {c: float(np.mean([E[c, k] @ tb[k] for k in complete])) for c in AC.CONDITIONS}
                    # own target vs another target of the arm (the swap donor's), none condition (descriptive)
                    cell["none_own_minus_donor_target"] = float(np.mean([E["none", k] @ tb[k] - E["none", k] @ tb[donors[k]] for k in complete]))
                cells.append(cell)
                print(domain, gen, arm, "A %.3f [%.3f, %.3f]" % (cell["A"]["mean"], cell["A"]["low"], cell["A"]["high"]),
                      ("D %.3f [%.3f]" % (cell["D"]["mean"], cell["D"]["low"])) if arm == "REAL" else "", flush=True)
        del corpus, P, tg, B
    n_rep = sum(c["A_replicates"] for c in cells)
    save("anchoring", {"cells": cells, "A_replicates_cells": n_rep, "of": len(cells),
                       "overall": "replicated" if n_rep == 12 else ("partial" if n_rep >= 10 else "not replicated"),
                       "REAL_D_replicates_cells": sum(c.get("D_replicates", False) for c in cells),
                       "REAL_supported_cells": sum(c.get("verdict") == "causal anchoring supported" for c in cells),
                       "interval_note": "t interval with two-sided level 1 - 2*0.05/24 (one-sided 0.05/24 bounds), as anchoring_causal.interval"})


# ---------------------------------------------------------------- Part 2: gpt-5.4-mini and ORACLE-H
def tang(X, c):
    T = X - np.outer(X @ c, c)
    n = np.linalg.norm(T, axis=-1, keepdims=True)
    return np.where(n > 1e-6, T / np.maximum(n, 1e-12), np.nan)


def unit(v):
    return v / np.linalg.norm(v)


def contexts(fups_dir):
    rows = [json.loads(l) for l in (DATA / "runs" / fups_dir / "followups.jsonl").read_text().splitlines()]
    return [r for r in rows if r["n_followups"] >= M.MIN_FUPS][:M.N_CTX]


def corpus_bge():
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract", "split", "adult", "date", "primary"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    emb = norm(np.load(DATA / "prepared/arxiv/emb_bgem3.npy").astype(np.float32)[keep]).astype(np.float32)
    return meta, emb


def oracle_like(ctx, C, F, rows, V, arms, meta, emb, rng):
    """oracle_history_analysis.main with step and fit recomputed from the vectors."""
    later = (meta["date"] >= min(f["date"] for c in ctx for f in c["followups"])).to_numpy()
    pool_all, pool_ids = emb[later], meta.loc[later, "id"].to_numpy()
    per = {a: {"step": [], "fit": [], "align": [], "spec": []} for a in arms}
    for i, c in enumerate(C):
        Fi = F[i]
        t = unit(tang(Fi, c).mean(0))
        fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
        pool = pool_all[~np.isin(pool_ids, list(fids))]
        tp = unit(tang(pool[np.argsort(-(pool @ c))[:len(Fi)]], c).mean(0))
        ts = unit(t - (t @ tp) * tp)
        for a in arms:
            v = {}
            for r in (0, 2):
                idx = [n for n, x in enumerate(rows) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                if idx:
                    X = V[idx]
                    T = tang(X, c)
                    v[r] = (float(np.mean(X @ c)), float(np.mean((X @ Fi.T).max(1))),
                            float(np.nanmean(T @ t)), float(np.nanmean(T @ ts)))
            if 0 in v and 2 in v:
                for j, k in enumerate(("step", "fit", "align", "spec")):
                    per[a][k].append(v[2][j] - v[0][j] if k != "step" else v[0][j] - v[2][j])
    res = {}
    for a in arms:
        d = {k: np.array(x) for k, x in per[a].items()}
        res[a] = {"n": int(len(d["fit"]))}
        for k, x in d.items():
            ci = boot_ci(x, rng)
            res[a][k] = {"mean": float(x.mean()), "ci": ci, "p": float(stats.wilcoxon(x).pvalue),
                         "ci_within_0.015": bool(-0.015 <= ci[0] and ci[1] <= 0.015)}
    n = min(len(per["ORACLE"]["fit"]), len(per["ORACLE-H"]["fit"]))
    dh = np.array(per["ORACLE-H"]["fit"][:n]) - np.array(per["ORACLE"]["fit"][:n])
    res["H1_fit_gain_H_minus_plain"] = {"mean": float(dh.mean()), "ci": boot_ci(dh, rng),
                                        "p_one_sided": float(stats.wilcoxon(dh, alternative="greater").pvalue)}
    return res


def modern():
    L, Ltext = local()
    meta, emb = corpus_bge()
    by_pos = {i: n for n, i in enumerate(meta["id"])}
    out = {}
    # gpt-5.4-mini on the 2025 contexts
    ctx = contexts("followups-2025")
    C = np.stack([L["modern/ctx", str(i)] for i in range(len(ctx))])
    F = [np.stack([L["modern/fup", f"{i}/{j}"] for j in range(len(c["followups"]))]) for i, c in enumerate(ctx)]
    agree = [float(C[i] @ emb[by_pos[c["arxiv_id"]]]) for i, c in enumerate(ctx)]
    out["agreement_modern_ctx_local_vs_kaggle_corpus"] = {"min": min(agree), "median": float(np.median(agree))}
    pb = [r for r in map(json.loads, (DATA / "runs/continue-modern/ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    PB = np.stack([L["modern/partB", f"{r['ctx']}/{r['arm']}/{r['k']}"] for r in pb])
    rng = np.random.default_rng(SEED)
    m1 = {}
    for arm in ("CTX", "CTX+MEM"):
        gap, div = [], []
        for i in range(len(C)):
            X = PB[[n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == arm]]
            gap.append(float(np.median(X @ C[i]) - np.median(F[i] @ C[i])))
            S = X @ X.T
            div.append(float(S[np.triu_indices(len(X), 1)].mean()))
        g = np.array(gap)
        m1[arm] = {"mean_step_gap": float(g.mean()), "ci": boot_ci(g, rng),
                   "p": float(stats.wilcoxon(g, alternative="greater").pvalue),
                   "followup_step_median": float(np.median([np.median(F[i] @ C[i]) for i in range(len(C))])),
                   "mean_div": float(np.mean(div))}
    it = [r for r in map(json.loads, (DATA / "runs/continue-modern/iter_ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    IT = np.stack([L["modern/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"] for r in it])
    rng = np.random.default_rng(SEED)
    mod = oracle_like(ctx, C, F, it, IT, ("GEO", "LLM", "ORACLE", "ORACLE-H"), meta, emb, rng)
    mod["M1"] = m1
    out["continue-modern"] = mod
    # gpt-4o ORACLE / ORACLE-H on the original contexts (Kaggle ctx/fup; both arms embedded locally)
    ctx4, C4, F4, *_ = R.load_generator("gpt-4o", "bgem3")
    it4 = [r for r in map(json.loads, (M.RUN / "iter_ideas.jsonl").read_text().splitlines())
           if r["abstract"] and r["arm"] in ("ORACLE", "ORACLE-H")]
    IT4 = np.stack([L["gpt4o/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"] for r in it4])
    rng = np.random.default_rng(SEED)
    out["continue-from-context"] = oracle_like(ctx4, C4, F4, it4, IT4, ("ORACLE", "ORACLE-H"), meta, emb, rng)
    # sensitivity: Kaggle vectors for ORACLE, local for ORACLE-H; and the local-vs-Kaggle agreement on ORACLE texts
    t = pd.read_parquet(DATA / "kaggle-upload/continuation-texts/texts.parquet")
    KV = norm(np.load(DATA / "kaggle-output/embed-continuation-v1/emb_bgem3.npy").astype(np.float32))
    pos = {k: n for n, k in enumerate(t["key"]) if t["kind"].iloc[n] == "iter"}
    ag, IT4k = [], IT4.copy()
    for n, r in enumerate(it4):
        if r["arm"] == "ORACLE":
            j = pos[f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"]
            assert t["text"].iloc[j] == Ltext["gpt4o/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"]
            ag.append(float(IT4[n] @ KV[j]))
            IT4k[n] = KV[j]
    out["agreement_gpt4o_ORACLE_local_vs_kaggle"] = {"min": min(ag), "median": float(np.median(ag)), "n": len(ag)}
    rng = np.random.default_rng(SEED)
    sens = oracle_like(ctx4, C4, F4, it4, IT4k, ("ORACLE", "ORACLE-H"), meta, emb, rng)
    out["sensitivity_gpt4o_ORACLE_kaggle_vectors"] = {"ORACLE": sens["ORACLE"], "H1": sens["H1_fit_gain_H_minus_plain"]}
    p = {"M1": m1["CTX"]["p"], "M2": mod["GEO"]["step"]["p"], "M3": mod["ORACLE"]["spec"]["p"],
         "H1": out["continue-from-context"]["H1_fit_gain_H_minus_plain"]["p_one_sided"]}
    h = holm(p)
    out["holm_M1_M2_M3_H1"] = h
    out["verdicts"] = {
        "M1_replicates": bool(m1["CTX"]["mean_step_gap"] > 0 and h["M1"] < 0.05),
        "M2_replicates": bool(mod["GEO"]["step"]["mean"] > 0 and h["M2"] < 0.05),
        "M3_replicates": bool(mod["ORACLE"]["spec"]["p"] >= 0.05),
        "H1_failure_replicates": bool(out["continue-from-context"]["H1_fit_gain_H_minus_plain"]["p_one_sided"] >= 0.05),
        "M4_failure_replicates": bool(mod["H1_fit_gain_H_minus_plain"]["p_one_sided"] >= 0.05),
        "H2_gpt4o_spec": out["continue-from-context"]["ORACLE-H"]["spec"],
        "H2_mini_spec": mod["ORACLE-H"]["spec"]}
    save("modern_oracle_history", out)
    print(json.dumps(out["verdicts"], indent=1), json.dumps(h), flush=True)
    for run in ("continue-from-context", "continue-modern"):
        for a, v in out[run].items():
            if isinstance(v, dict) and "step" in v:
                print(run, a, " ".join(f"{k} {v[k]['mean']:+.4f} (p {v[k]['p']:.2g})" for k in ("step", "fit", "align", "spec")), flush=True)


# ---------------------------------------------------------------- Part 3: continuation A, B2, B4, C2, C4
def continuation():
    meta, emb = corpus_bge()
    past = (meta["split"] == "past").to_numpy()
    P, pdate = emb[past], meta.loc[past, "date"].str[:4].to_numpy()
    fut = meta[~past].reset_index(drop=True)
    Fe = emb[~past]
    by_id = meta.set_index("id")
    mu_hat = M.norm(P.mean(0))
    ctx, C, F, pb, PB, it, IT = R.load_generator("gpt-4o", "bgem3")
    C = C.astype(np.float32)
    out = {}
    # Part A, as continue_from_context.part_a
    rng = np.random.default_rng(SEED)
    steps, proj = [], {k: [] for k in ("momentum", "hole", "centre")}
    for i, c in enumerate(C):
        _, _, dirs = M.directions(c, P, pdate, mu_hat, rng)
        Fi = F[i]
        steps.append(float(np.median(Fi @ c)))
        for k, v in dirs.items():
            proj[k].append(float(np.nanmean([M.cosv(f - c, v) for f in Fi])))
        if i % 25 == 0:
            print("part A", i, flush=True)
    steps, pm, ph, pc = (np.array(x) for x in (steps, proj["momentum"], proj["hole"], proj["centre"]))
    diff = pm - ph
    ok = ~np.isnan(diff)
    w = stats.wilcoxon(diff[ok], alternative="greater")
    bootA = lambda x: [float(np.nanpercentile([np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]
    out["A"] = {"A1_step_median": float(np.median(steps)), "A1_step_quartiles": [float(np.percentile(steps, q)) for q in (25, 75)],
                "A2_mean_proj_momentum": float(np.nanmean(pm)), "A2_mean_proj_hole": float(np.nanmean(ph)),
                "A2_mean_diff": float(np.nanmean(diff)), "A2_diff_ci": bootA(diff), "A2_wilcoxon_p": float(w.pvalue),
                "A2_failure_replicates": bool(np.nanmean(diff) < 0),
                "A3_mean_proj_centre": float(np.nanmean(pc)), "A3_ci": bootA(pc),
                "A3_wilcoxon_p_two_sided": float(stats.wilcoxon(pc[~np.isnan(pc)]).pvalue)}
    out["A"]["A3_replicates"] = bool(out["A"]["A3_mean_proj_centre"] > 0 and out["A"]["A3_wilcoxon_p_two_sided"] < 0.05)
    print(json.dumps(out["A"]), flush=True)
    # Part B, as continue_from_context.part_b
    ARMS = ("CTX", "CTX+NB", "CTX+MEM", "CTX+MOM")
    rng = np.random.default_rng(SEED)
    per = {a: {"step_gap": [], "hit": [], "fit": [], "div": [], "nb_vs_fup": []} for a in ARMS}
    dist = {}
    for i, c in enumerate(ctx):
        Fi = F[i]
        step_f = np.median(Fi @ C[i])
        prim = by_id.loc[c["arxiv_id"], "primary"]
        fup_ids = {f["arxiv_id"] for f in c["followups"]}
        Dv = Fe[((fut["primary"] == prim) & ~fut["id"].isin(fup_ids)).to_numpy()]
        dist[i] = Dv
        for a in ARMS:
            idx = [n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == a]
            if not idx:
                continue
            Ii = PB[idx]
            per[a]["step_gap"].append(float(np.median(Ii @ C[i]) - step_f))
            sf, sd = Ii @ Fi.T, Ii @ Dv.T
            per[a]["fit"].append(float(sf.max(1).mean()))
            per[a]["hit"].append(float((sf.max(1) > sd.max(1)).mean()))
            S = Ii @ Ii.T
            per[a]["div"].append(float(S[np.triu_indices(len(Ii), 1)].mean()))
            if a == "CTX+NB":
                cn = M.norm(P[pb[idx[0]]["shown"]].mean(0))
                cf = M.norm(Fi.mean(0))
                per[a]["nb_vs_fup"].append(float((Ii @ cn).mean() - (Ii @ cf).mean()))
    A = {a: {k: np.array(v) for k, v in d.items()} for a, d in per.items()}
    bootB = lambda x: [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]
    res = {a: {"n": int(len(A[a]["hit"])), **{f"mean_{k}": float(v.mean()) for k, v in A[a].items() if len(v)}} for a in ARMS}
    p = {}
    b1 = {a: stats.wilcoxon(A[a]["step_gap"], alternative="greater") for a in ("CTX", "CTX+NB")}
    p["B1"] = max(v.pvalue for v in b1.values())
    res["B1"] = {a: {"mean_step_gap": float(A[a]["step_gap"].mean()), "ci": bootB(A[a]["step_gap"]), "p": float(v.pvalue)} for a, v in b1.items()}
    w2 = stats.wilcoxon(A["CTX+NB"]["nb_vs_fup"], alternative="greater")
    p["B2"] = w2.pvalue
    res["B2"] = {"mean_diff": float(A["CTX+NB"]["nb_vs_fup"].mean()), "ci": bootB(A["CTX+NB"]["nb_vs_fup"]), "p": float(w2.pvalue)}
    w3 = stats.wilcoxon(A["CTX+MEM"]["div"], A["CTX"]["div"], alternative="less")
    p["B3"] = w3.pvalue
    res["B3"] = {"mean_pairwise_cos_MEM": float(A["CTX+MEM"]["div"].mean()), "mean_pairwise_cos_CTX": float(A["CTX"]["div"].mean()), "p": float(w3.pvalue)}
    d4 = A["CTX+MOM"]["hit"] - A["CTX"]["hit"]
    maj_m, maj_c = A["CTX+MOM"]["hit"] > 0.5, A["CTX"]["hit"] > 0.5
    b, cc = int((maj_m & ~maj_c).sum()), int((~maj_m & maj_c).sum())
    p_mc = float(stats.binomtest(b, b + cc, 0.5, alternative="greater").pvalue) if b + cc else 1.0
    p["B4"] = p_mc
    res["B4"] = {"hit_MOM": float(A["CTX+MOM"]["hit"].mean()), "hit_CTX": float(A["CTX"]["hit"].mean()),
                 "mean_diff": float(d4.mean()), "ci": bootB(d4), "mcnemar_b": b, "mcnemar_c": cc, "p": p_mc,
                 "fit_MOM": float(A["CTX+MOM"]["fit"].mean()), "fit_CTX": float(A["CTX"]["fit"].mean()),
                 "best_arm_by_hit": max(ARMS, key=lambda a: A[a]["hit"].mean())}
    p["A2"] = out["A"]["A2_wilcoxon_p"]
    res["holm"] = holm(p)
    res["B2_replicates_same_holm_decision"] = bool(res["holm"]["B2"] >= 0.05)
    res["B4_failure_replicates"] = bool(res["B4"]["mean_diff"] <= 0)
    res["B4_reversal_replicates"] = bool(res["B4"]["mean_diff"] <= 0 and res["B4"]["ci"][1] < 0)
    out["B"] = res
    print(json.dumps({k: res[k] for k in ("B2", "B4", "holm")}), flush=True)
    # Iterative critique, as iterative_critique.analyse (hit, step, fit), GEO / LLM / ORACLE
    IA = ("GEO", "LLM", "ORACLE")
    step_f = np.array([np.median(F[i] @ C[i]) for i in range(len(ctx))])
    pr = {(a, r): {"step": [], "fit": [], "hit": []} for a in IA for r in range(3)}
    for i in range(len(ctx)):
        Fi, Dv = F[i], dist[i]
        for a in IA:
            for r in range(3):
                idx = [n for n, x in enumerate(it) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                if idx:
                    V = IT[idx]
                    sf, sd = V @ Fi.T, V @ Dv.T
                    s, f, h = float((V @ C[i]).mean()), float(sf.max(1).mean()), float((sf.max(1) > sd.max(1)).mean())
                else:
                    s = f = h = np.nan
                pr[(a, r)]["step"].append(s); pr[(a, r)]["fit"].append(f); pr[(a, r)]["hit"].append(h)
    Pq = {k: {m: np.array(v) for m, v in d.items()} for k, d in pr.items()}
    rng = np.random.default_rng(SEED)
    ic = {"followup_step_median": float(np.median(step_f))}
    for a in IA:
        ic[a] = {f"round{r}": {m: float(np.nanmean(Pq[(a, r)][m])) for m in ("step", "fit", "hit")} for r in range(3)}
        ic[a]["step_gap_round0"] = float(np.nanmean(Pq[(a, 0)]["step"] - step_f))
        ic[a]["step_gap_round2"] = float(np.nanmean(Pq[(a, 2)]["step"] - step_f))
    p = {}
    d1 = Pq[("GEO", 0)]["step"] - Pq[("GEO", 2)]["step"]
    p["C1"] = stats.wilcoxon(d1, alternative="greater", nan_policy="omit").pvalue
    ic["C1"] = {"step_reduction": float(np.nanmean(d1)), "ci": boot_ci(d1, rng), "p": float(p["C1"]),
                "gap_closed": ic["GEO"]["step_gap_round0"] - ic["GEO"]["step_gap_round2"]}
    d2 = Pq[("GEO", 2)]["hit"] - Pq[("GEO", 0)]["hit"]
    p["C2"] = stats.wilcoxon(d2 + 0.03, alternative="greater", nan_policy="omit").pvalue
    ic["C2"] = {"hit_change": float(np.nanmean(d2)), "ci": boot_ci(d2, rng), "p_noninferiority": float(p["C2"]),
                "holds": bool(np.nanmean(d2) >= -0.03)}
    d3 = Pq[("LLM", 0)]["step"] - Pq[("LLM", 2)]["step"]
    p["C3"] = stats.wilcoxon(d3, alternative="greater", nan_policy="omit").pvalue
    ic["C3"] = {"step_reduction": float(np.nanmean(d3)), "ci": boot_ci(d3, rng), "p": float(p["C3"])}
    d4 = Pq[("LLM", 2)]["hit"] - Pq[("LLM", 0)]["hit"]
    ic["C4"] = {"hit_change": float(np.nanmean(d4)), "ci": boot_ci(d4, rng),
                "p_two_sided_exploratory": float(stats.wilcoxon(d4, nan_policy="omit").pvalue)}
    d5 = Pq[("ORACLE", 2)]["fit"] - Pq[("ORACLE", 0)]["fit"]
    p["C5"] = stats.wilcoxon(d5, alternative="greater", nan_policy="omit").pvalue
    ic["C5"] = {"fit_gain": float(np.nanmean(d5)), "ci": boot_ci(d5, rng), "p": float(p["C5"])}
    ic["holm"] = holm({k: float(v) for k, v in p.items()})
    ic["C2_failure_replicates"] = bool(ic["C2"]["hit_change"] < -0.03 and ic["holm"]["C2"] >= 0.05)
    ic["C4_replicates"] = bool(ic["C4"]["hit_change"] < 0 and ic["C4"]["ci"][1] < 0)
    out["iterative_critique"] = ic
    print(json.dumps({k: ic[k] for k in ("C1", "C2", "C4", "C5", "holm")}), flush=True)
    save("continuation", out)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("continuation", continuation), ("anchoring", anchoring), ("modern", modern)):
        if part in (name, "all"):
            fn()
