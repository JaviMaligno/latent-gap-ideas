"""Revision (review point 7): effect intervals, detectable magnitudes and equivalence bounds for the null claims of
Section 5 (continuation study). Reuses saved embeddings only; no embedding model is loaded.

For every quantity: per-context values -> mean, bootstrap 95 % CI (2,000 resamples, seed 20261030, same as the paper),
t-based SE, paired Wilcoxon p (as in the paper), minimum detectable effect at 80 % power (two-sided alpha 0.05:
2.802 SE; one-sided alpha 0.05 for H1: 2.486 SE), the 90 % CI and the smallest symmetric equivalence bound that a
TOST at alpha 0.05 would accept (delta* = max |90 % CI limit|), and, where an anchor is defined, all of this as a
fraction of the anchor gap:
  - step: the context's step gap (idea median step minus follow-up median step, CTX arm), as in Section 5.1;
  - alignment: follow-ups' own leave-one-out alignment minus the ideas' round-0 alignment (Section 5.3 control);
  - follow-up-specific: follow-ups' own leave-one-out projection on the follow-up-specific direction s (s rebuilt
    without the follow-up) minus the ideas' round-0 projection on s (new anchor, same construction);
  - fit: follow-ups' own leave-one-out fit (max cos to the other follow-ups) minus the ideas' round-0 fit (new anchor).

Parts:
  T3  Table 3 cells (direction_tangent.py logic, Kaggle vectors, pool = split "future")
  T4  Table 4 rows incl. H1 / M4 (oracle_history_analysis.py logic, cached local vectors, pool = date >= first follow-up)
  B   one-shot arms vs CTX (Section 5.1 / A.8, continue_from_context part B logic) and B2
  M1  gpt-5.4-mini step gap (modern_step_gap.py logic) and its difference from gpt-4o's
Output: docs/prereg/revision_null_effects.json
  OMP_NUM_THREADS=2 uv run python scripts/revision_null_effects.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/revision_null_effects.json"
SEED = 20261030
Z2, Z1, ZP = 1.959964, 1.644854, 0.841621
I_ARMS = ("GEO", "LLM", "ORACLE")


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def unit(v):
    return v / np.linalg.norm(v)


def tang(X, c):
    T = X - np.outer(X @ c, c)
    n = np.linalg.norm(T, axis=-1, keepdims=True)
    return np.where(n > 1e-6, T / np.maximum(n, 1e-12), np.nan)


def boot(x, rng):
    return [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], q)) for q in (2.5, 97.5)]


def summ(x, rng, anchor=None, one_sided=False, wilcox_alt="two-sided"):
    x = np.asarray(x, float)
    n = len(x)
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    tq = stats.t.ppf(0.95, n - 1)
    lo90, hi90 = m - tq * se, m + tq * se
    mde = (Z1 if one_sided else Z2) * se + ZP * se
    r = {"n": n, "mean": m, "ci95_boot": boot(x, rng), "se": se, "p_wilcoxon": float(stats.wilcoxon(x, alternative=wilcox_alt).pvalue),
         "mde80": float(mde), "ci90_t": [float(lo90), float(hi90)], "tost_delta_star": float(max(abs(lo90), abs(hi90)))}
    if anchor is not None:
        a = float(anchor)
        r["anchor_gap"] = a
        r["frac_of_anchor"] = {"mean": m / a, "ci95": [c / a for c in r["ci95_boot"]], "mde80": mde / a,
                               "tost_delta_star": r["tost_delta_star"] / a}
    return r


def loo_spec(TF, tp):
    """Follow-ups' own projection on the follow-up-specific direction, rebuilt without each follow-up."""
    S = np.nansum(TF, 0)
    out = []
    for j in range(len(TF)):
        t = unit(S - TF[j])
        s = unit(t - (t @ tp) * tp)
        out.append(TF[j] @ s)
    return float(np.mean(out))


def loo_align(TF):
    S = TF.sum(0)
    return float(np.mean([TF[j] @ unit(S - TF[j]) for j in range(len(TF))]))


def loo_fit(Fi):
    G = Fi @ Fi.T
    np.fill_diagonal(G, -np.inf)
    return float(G.max(1).mean())


def corpus_meta():
    meta = pd.read_parquet(D / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult", "date", "primary"])
    keep = ~meta["adult"].to_numpy()
    return meta[keep].reset_index(drop=True), keep


def contexts(fups):
    rows = [json.loads(l) for l in (D / "runs" / fups / "followups.jsonl").read_text().splitlines()]
    return [r for r in rows if r["n_followups"] >= 5][:150]


# ---------------------------------------------------------------- T3
def load_generator(gen, space, ctx):
    nf = [len(c["followups"]) for c in ctx]
    if gen == "gpt-4o":
        t = pd.read_parquet(D / "kaggle-upload/continuation-texts/texts.parquet")
        V = norm(np.load(D / f"kaggle-output/embed-continuation-v1/emb_{space}.npy").astype(np.float32))
        kinds, keys = t["kind"].to_numpy(), t["key"].to_numpy()
        C, Fall = V[kinds == "ctx"], V[kinds == "fup"]
        km = kinds == "iter"
        IT, ikeys = V[km], keys[km]
        it = [{"ctx": int(k.split("/")[0]), "arm": k.split("/")[1]} for k in ikeys]
        # positional match with iter_ideas.jsonl (rows in I_ARMS with an abstract), as direction_tangent.py; checked on ctx/arm
        rows = [json.loads(l) for l in (D / "runs/continue-from-context/iter_ideas.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["abstract"] and r["arm"] in I_ARMS]
        assert len(rows) == len(it) and all(r["ctx"] == x["ctx"] and r["arm"] == x["arm"] for r, x in zip(rows, it))
        it = rows
    else:
        K = D / "kaggle-output/llama-continuation-v1"
        V = norm(np.load(K / f"emb_{space}.npy").astype(np.float32))
        pb = [r for r in map(json.loads, (K / "ideas_llama.jsonl").read_text().splitlines()) if r["abstract"]]
        it = [r for r in map(json.loads, (K / "iter_llama.jsonl").read_text().splitlines()) if r["abstract"]]
        o = 0
        C = V[o:o + len(ctx)]; o += len(ctx)
        Fall = V[o:o + sum(nf)]; o += sum(nf)
        o += len(pb)
        IT = V[o:]
        assert len(IT) == len(it)
    starts = np.cumsum([0] + nf)
    F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]
    return C, F, it, IT


def part_t3(meta, keep, rng):
    ctx = contexts("followups")
    fut = (meta["split"] == "future").to_numpy()
    fut_ids = meta.loc[fut, "id"].to_numpy()
    res = {}
    for space in ("qwen3e06", "bgem3"):
        Fe = norm(np.load(D / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep][fut])
        for gen in ("gpt-4o", "olmo-7b-0424"):
            C, F, it, IT = load_generator(gen, space, ctx)
            byk = {}
            for n, x in enumerate(it):
                byk.setdefault((x["ctx"], x["arm"], x["round"]), []).append(n)
            per = {a: {k: [] for k in ("align", "p", "s", "a0", "s0")} for a in I_ARMS}
            loo_a, loo_s = [], []
            for i in range(len(ctx)):
                c = C[i]
                TF = tang(F[i], c)
                t = unit(TF.mean(0))
                fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
                pool = Fe[~np.isin(fut_ids, list(fids))]
                sc = pool @ c
                tp = unit(tang(pool[np.argsort(-sc)[:len(F[i])]], c).mean(0))
                ts = unit(t - (t @ tp) * tp)
                la, ls = loo_align(TF), loo_spec(TF, tp)
                for a in I_ARMS:
                    v = {}
                    for r in (0, 2):
                        idx = byk.get((i, a, r))
                        if idx:
                            T = tang(IT[idx], c)
                            v[r] = (np.nanmean(T @ t), np.nanmean(T @ tp), np.nanmean(T @ ts))
                    if 0 in v and 2 in v:
                        per[a]["align"].append(v[2][0] - v[0][0]); per[a]["p"].append(v[2][1] - v[0][1])
                        per[a]["s"].append(v[2][2] - v[0][2]); per[a]["a0"].append(la - v[0][0]); per[a]["s0"].append(ls - v[0][2])
                loo_a.append(la); loo_s.append(ls)
            cell = {"followups_loo_alignment": float(np.mean(loo_a)), "followups_loo_specific_projection": float(np.mean(loo_s))}
            for a in I_ARMS:
                d = {k: np.array(x) for k, x in per[a].items()}
                cell[a] = {"alignment_change": summ(d["align"], rng, anchor=d["a0"].mean()),
                           "along_p": summ(d["p"], rng),
                           "followup_specific": summ(d["s"], rng, anchor=d["s0"].mean())}
            res[f"{gen}/{space}"] = cell
            print("T3", gen, space, {a: (round(cell[a]["followup_specific"]["mean"], 4), round(cell[a]["followup_specific"]["p_wilcoxon"], 3),
                                        round(cell[a]["alignment_change"]["p_wilcoxon"], 6)) for a in I_ARMS}, flush=True)
    return res


# ---------------------------------------------------------------- T4
def part_t4(meta, keep, rng):
    emb = norm(np.load(D / "prepared/arxiv/emb_qwen3e06.npy").astype(np.float32)[keep])
    res = {}
    for run, fups, arms in (("continue-from-context", "followups", ("ORACLE", "ORACLE-H")),
                            ("continue-modern", "followups-2025", ("GEO", "LLM", "ORACLE", "ORACLE-H"))):
        RUN = D / "runs" / run
        ctx = contexts(fups)
        z = np.load(RUN / "emb.npz")
        C, F, owner = z["C"], z["F"], z["owner"]
        rows = [r for r in map(json.loads, (RUN / "iter_ideas.jsonl").read_text().splitlines()) if r["abstract"] and r["arm"] in arms]
        V = np.load(RUN / f"oracle_history_emb_{'_'.join(arms)}.npy")
        assert len(V) == len(rows)
        byk = {}
        for n, x in enumerate(rows):
            byk.setdefault((x["ctx"], x["arm"], x["round"]), []).append(n)
        later = (meta["date"] >= min(f["date"] for c in ctx for f in c["followups"])).to_numpy()
        pool_all, pool_ids = emb[later], meta.loc[later, "id"].to_numpy()
        per = {a: {k: [] for k in ("step", "fit", "align", "spec", "a0", "s0", "f0")} for a in arms}
        for i, c in enumerate(C):
            Fi = F[owner == i]
            TF = tang(Fi, c)
            t = unit(np.nanmean(TF, 0))
            fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
            pool = pool_all[~np.isin(pool_ids, list(fids))]
            tp = unit(tang(pool[np.argsort(-(pool @ c))[:len(Fi)]], c).mean(0))
            ts = unit(t - (t @ tp) * tp)
            la, ls, lf = loo_align(TF), loo_spec(TF, tp), loo_fit(Fi)
            for a in arms:
                v = {}
                for r in (0, 2):
                    idx = byk.get((i, a, r))
                    if idx:
                        T = tang(V[idx], c)
                        v[r] = (np.mean([rows[n]["step"] for n in idx]), np.mean([rows[n]["fit"] for n in idx]),
                                float(np.nanmean(T @ t)), float(np.nanmean(T @ ts)))
                if 0 in v and 2 in v:
                    per[a]["step"].append(v[0][0] - v[2][0]); per[a]["fit"].append(v[2][1] - v[0][1])
                    per[a]["align"].append(v[2][2] - v[0][2]); per[a]["spec"].append(v[2][3] - v[0][3])
                    per[a]["a0"].append(la - v[0][2]); per[a]["s0"].append(ls - v[0][3]); per[a]["f0"].append(lf - v[0][1])
        out = {}
        for a in arms:
            d = {k: np.array(x) for k, x in per[a].items()}
            out[a] = {"step_lengthened": summ(d["step"], rng), "fit_change": summ(d["fit"], rng, anchor=d["f0"].mean()),
                      "alignment_change": summ(d["align"], rng, anchor=d["a0"].mean()),
                      "followup_specific": summ(d["spec"], rng, anchor=d["s0"].mean())}
        dh = np.array(per["ORACLE-H"]["fit"]) - np.array(per["ORACLE"]["fit"])
        f0 = np.mean(per["ORACLE"]["f0"])
        out["H1_fit_gain_H_minus_plain"] = summ(dh, rng, anchor=f0, one_sided=True, wilcox_alt="greater")
        out["H1_fit_gain_H_minus_plain"]["anchor_note"] = "anchor = follow-ups' LOO fit minus ORACLE round-0 idea fit"
        out["H2_spec_H_minus_plain"] = summ(np.array(per["ORACLE-H"]["spec"]) - np.array(per["ORACLE"]["spec"]), rng,
                                            anchor=np.mean(per["ORACLE"]["s0"]))
        res[run] = out
        print("T4", run, {a: (round(out[a]["followup_specific"]["mean"], 4), round(out[a]["alignment_change"]["p_wilcoxon"], 6)) for a in arms},
              "H1", round(out["H1_fit_gain_H_minus_plain"]["mean"], 4), round(out["H1_fit_gain_H_minus_plain"]["p_wilcoxon"], 3), flush=True)
    return res


# ---------------------------------------------------------------- B and M1
def part_b(meta, keep, rng):
    RUN = D / "runs/continue-from-context"
    z = np.load(RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ideas = [r for r in map(json.loads, (RUN / "ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    I = np.load(RUN / "idea_emb.npy")
    assert len(I) == len(ideas)
    emb = norm(np.load(D / "prepared/arxiv/emb_qwen3e06.npy").astype(np.float32)[keep])
    P = emb[(meta["split"] == "past").to_numpy()]
    arms = ("CTX", "CTX+NB", "CTX+MEM", "CTX+MOM")
    gap = {a: [] for a in arms}
    nbf = []
    for i in range(len(C)):
        Fi = F[owner == i]
        step_f = np.median(Fi @ C[i])
        for a in arms:
            idx = [n for n, r in enumerate(ideas) if r["ctx"] == i and r["arm"] == a]
            X = I[idx]
            gap[a].append(float(np.median(X @ C[i]) - step_f))
            if a == "CTX+NB":
                cn = unit(P[ideas[idx[0]]["shown"]].mean(0)); cf = unit(Fi.mean(0))
                nbf.append(float((X @ cn).mean() - (X @ cf).mean()))
    g = {a: np.array(v) for a, v in gap.items()}
    anchor = g["CTX"].mean()
    res = {"step_gap": {a: summ(g[a], rng, anchor=anchor) for a in arms}}
    res["gap_closed_vs_CTX"] = {a: summ(g["CTX"] - g[a], rng, anchor=anchor) for a in arms[1:]}
    res["B2_nb_minus_fup_centroid"] = summ(np.array(nbf), rng)
    res["note"] = "anchor = mean CTX step gap (gpt-4o, Qwen3); gap_closed > 0 means the arm closes part of it"
    return res, g["CTX"]


def part_m1(rng, g4o):
    RUN = D / "runs/continue-modern"
    z = np.load(RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    rows = [r for r in map(json.loads, (RUN / "ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    V = np.load(RUN / "partB_emb.npy")
    assert len(V) == len(rows)
    gap = {a: [] for a in ("CTX", "CTX+MEM")}
    for i in range(len(C)):
        sf = np.median(F[owner == i] @ C[i])
        for a in gap:
            idx = [n for n, r in enumerate(rows) if r["ctx"] == i and r["arm"] == a]
            gap[a].append(float(np.median(V[idx] @ C[i]) - sf))
    g = {a: np.array(v) for a, v in gap.items()}
    res = {a: summ(g[a], rng, wilcox_alt="greater") for a in g}
    res["MEM_closes_vs_CTX"] = summ(g["CTX"] - g["CTX+MEM"], rng, anchor=g["CTX"].mean())
    # difference from gpt-4o (independent context sets): Welch and independent bootstrap
    a, b = g["CTX"], g4o
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(2000)]
    res["CTX_gap_minus_gpt4o_CTX_gap"] = {"mean": diff, "ci95_boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                          "se": se, "p_welch": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
                                          "p_mannwhitney": float(stats.mannwhitneyu(a, b).pvalue), "mde80": float((Z2 + ZP) * se),
                                          "note": "different contexts (2025 vs 2021-22) and follow-up windows; not a same-context comparison"}
    return res


def main():
    meta, keep = corpus_meta()
    rng = np.random.default_rng(SEED)
    res = {"method": __doc__.split("Parts:")[0].strip()}
    res["T3_direction_tangent"] = part_t3(meta, keep, rng)
    OUT.write_text(json.dumps(res, indent=1))
    res["T4_modern_and_oracle_history"] = part_t4(meta, keep, rng)
    OUT.write_text(json.dumps(res, indent=1))
    res["B_one_shot_arms_gpt4o_qwen3"], g4o = part_b(meta, keep, rng)
    res["M1_modern_step_gap"] = part_m1(rng, g4o)
    dt = json.loads((OUT.parent / "direction_tangent.json").read_text())
    oh = json.loads((OUT.parent / "oracle_history.json").read_text())
    res["ORACLE_alignment_p_gpt4o_qwen3"] = {
        "table3_direction_tangent_json": dt["gpt-4o/qwen3e06"]["ORACLE"]["alignment_change_p"],
        "table4_oracle_history_json": oh["continue-from-context"]["ORACLE"]["align"]["p"],
        "table3_recomputed": res["T3_direction_tangent"]["gpt-4o/qwen3e06"]["ORACLE"]["alignment_change"]["p_wilcoxon"],
        "table4_recomputed": res["T4_modern_and_oracle_history"]["continue-from-context"]["ORACLE"]["alignment_change"]["p_wilcoxon"]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["ORACLE_alignment_p_gpt4o_qwen3"], indent=1))


if __name__ == "__main__":
    main()
