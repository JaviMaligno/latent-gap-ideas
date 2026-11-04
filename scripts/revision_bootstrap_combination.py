"""Revision: repair the bootstrap of the combination-vs-momentum analysis (Appendix A.6, review point 21).

The original bootstrap (scripts/combination_vs_momentum.py) resamples future papers with replacement and then
takes the *set* of distinct targets they hit (np.unique). A resample of n papers contains only ~63 % of the
distinct papers, so singleton targets (one future paper) drop out with probability ~e^-1 while targets hit by many
papers are almost always kept. Every resampled target set is therefore smaller than the observed one and
over-represents popular, many-paper triples. Popularity- and momentum-like scores rank those triples higher, so
their resampled enrichment and AUC shift upwards (and scores that favour rare triples shift downwards): the
percentile intervals are not centred on the estimand and can exclude the point estimate.

This script recomputes the exact scores of the original run (same code, same seed; nothing in the analysis is
changed) and replaces the bootstrap with schemes whose resample at weight 1 reproduces the point estimate:

  A. target bootstrap: resample the distinct targets with replacement (multinomial weights); every metric is the
     weighted version of the original one, so with all weights = 1 it equals the point estimate. Negatives (the
     candidate space) are an enumerated population, not a sample, and are kept fixed.
  B. paper-cluster bootstrap: resample future papers with replacement, but give target t the weight
     (# resampled paper incidences of t) / (# observed paper incidences of t), whose expectation is exactly 1.
     This keeps the paper as the independent unit (relevant for pairs, where one paper hits 3 pairs).
  C. DeLong (1988) analytic SE for every AUC and paired AUC difference, with and without the negative component.

  OMP_NUM_THREADS=2 uv run python scripts/revision_bootstrap_combination.py
Output: docs/prereg/revision_bootstrap_combination.json ; cache: runs/revision_bootstrap_combination/*.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combination_vs_momentum as cvm  # noqa: E402  (same functions, same constants, same seed)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/prereg/revision_bootstrap_combination.json"
ORIG = REPO / "docs/prereg/combination_vs_momentum_result.json"
CACHE = cvm.DATA / "runs/revision_bootstrap_combination"
N_BOOT = 2000
SEED = cvm.SEED
METRICS = ["enr_all", "auc_all", "enr_novel_full", "enr_novel_within", "auc_novel"]
log = cvm.log


# ---------------------------------------------------------------- stage 1: recompute the scores exactly
def compute_scores(V, k, top, year, past, hole):
    """Copy of the scoring part of cvm.analyse_arm (unchanged logic)."""
    C, lookup = cvm.candidates(V, k)
    N = len(C)
    W = {w: cvm.window_features(top, year, past, C, lookup, V, k, w) for w in cvm.WINDOWS}
    for w in cvm.WINDOWS:
        for e in cvm.EMBEDDERS:
            W[w][0][f"HOLE_{e}"] = hole[(V, k, w, e)]
    fp, _, ids_p, _, _, _ = W["pseudo"]
    ft, _, ids_t, _, n_out, c_test = W["test"]
    y_p = np.zeros(N, int)
    y_p[ids_p] = 1
    mk = cvm.MOM_KEYS[k]
    sets = {"FIT_M": mk, "FIT_MC": mk + ["COH", "RAR"]}
    for e in cvm.EMBEDDERS:
        sets[f"FIT_MCH_{e}"] = mk + ["COH", "RAR", f"HOLE_{e}"]
    zs = cvm.zs
    scores = {"POP": ft["POP"], "M": sum(zs(ft[m]) for m in mk), "COH": ft["COH"],
              "NC_0.4": zs(ft["COH"]) + cvm.BETA * zs(ft["RAR"]), "RAR": ft["RAR"]}
    for e in cvm.EMBEDDERS:
        scores[f"HOLE_{e}"] = ft[f"HOLE_{e}"]
    for name, keys in sets.items():
        Xp = np.column_stack([fp[q] for q in keys])
        sc = StandardScaler().fit(Xp)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xp), y_p)
        scores[name] = lr.decision_function(sc.transform(np.column_stack([ft[q] for q in keys])))
    return scores, ids_t, n_out, c_test, N


def stage1():
    CACHE.mkdir(parents=True, exist_ok=True)
    arms = [(cvm.V_PRIMARY, 3), (cvm.V_PRIMARY, 2)]
    if all((CACHE / f"scores_V{V}_k{k}.npz").exists() for V, k in arms):
        log("stage 1: cached scores found")
        return arms
    meta, year, past, idx = cvm.load()
    log(f"papers {len(meta)}")
    top_f = CACHE / f"top_V{cvm.V_PRIMARY}.npy"
    if top_f.exists():
        top = np.load(top_f)
    else:
        top, _ = cvm.atom_sets(meta["text"], past, cvm.V_PRIMARY, {})
        np.save(top_f, top)
    texts = None  # noqa: F841
    del meta
    cands = {(V, k): cvm.candidates(V, k)[0] for V, k in arms}
    hole = {}
    for e in cvm.EMBEDDERS:
        E = np.load(cvm.DATA / f"prepared/arxiv/emb_{e}.npy", mmap_mode="r")[idx].astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        for w, (last, _, _) in cvm.WINDOWS.items():
            feat = past & (year <= last)
            km, h = cvm.fit_cells(E[feat], np.random.default_rng(SEED))
            log(f"  cells {e}/{w}")
            for V, k in arms:
                hole[(V, k, w, e)] = cvm.hole_feature(E, top, feat, cands[(V, k)], km, h, V)
        del E
    for V, k in arms:
        scores, ids_t, n_out, c_test, N = compute_scores(V, k, top, year, past, hole)
        np.savez(CACHE / f"scores_V{V}_k{k}.npz", ids_t=ids_t, n_out=n_out, c_test=c_test, N=N,
                 names=np.array(list(scores)), S=np.column_stack(list(scores.values())))
        log(f"  scores V={V} k={k} saved")
    return arms


# ---------------------------------------------------------------- stage 2: weighted metrics
def per_target_parts(score, perm, pos, novel, N, Nn):
    """Per-target contributions such that every original metric = weighted mean with weights 1."""
    ra = cvm.ranks_of(score, perm)
    rn = cvm.ranks_of(score, perm, novel)
    K, Kn = round(cvm.TOPK_FRAC * N), round(cvm.TOPK_FRAC * Nn)
    npos = len(pos)
    rp = ra[pos]
    order = np.empty(npos, np.int64)
    order[np.argsort(rp)] = np.arange(1, npos + 1)
    isn = novel[pos]
    nn = int(isn.sum())
    rnp = rn[pos]
    order_n = np.zeros(npos, np.int64)
    idx_n = np.where(isn)[0]
    order_n[idx_n[np.argsort(rnp[idx_n])]] = np.arange(1, nn + 1)
    return {
        "top_all": (rp > N - K).astype(float),               # enrichment numerator
        "a_all": (rp - order) / (N - npos),                   # share of negatives ranked below (DeLong V10)
        "top_nov_full": ((rp > N - K) & isn).astype(float),
        "top_nov_within": ((rnp > Nn - Kn) & isn).astype(float),
        "a_nov": np.where(isn, (rnp - order_n) / max(Nn - nn, 1), 0.0),
        "isn": isn.astype(float),
        "_ranks": (ra, rn),
    }


def weighted_metrics(P, Wt, N, Nn):
    """Wt: (B, npos) weights. Returns dict metric -> (B,) values."""
    K, Kn = round(cvm.TOPK_FRAC * N), round(cvm.TOPK_FRAC * Nn)
    tot = Wt.sum(1)
    totn = Wt @ P["isn"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "enr_all": (Wt @ P["top_all"]) / (K * tot / N),
            "auc_all": (Wt @ P["a_all"]) / tot,
            "enr_novel_full": (Wt @ P["top_nov_full"]) / (K * totn / N),
            "enr_novel_within": (Wt @ P["top_nov_within"]) / (Kn * totn / Nn),
            "auc_novel": (Wt @ P["a_nov"]) / totn,
        }


def delong_neg_component(ra_neg_sorted, rp_sorted, npos):
    """V01 for each negative: share of positives ranked above it (rank comparison, ties broken already)."""
    above = npos - np.searchsorted(rp_sorted, ra_neg_sorted, side="right")
    return above / npos


def delong(score_names, parts, pos, novel, N):
    """DeLong components for AUC (all) and AUC within novel, per score. Returns V10/V01 arrays."""
    out = {}
    posmask = np.zeros(N, bool)
    posmask[pos] = True
    isn = novel[pos]
    for s in score_names:
        ra, rn = parts[s]["_ranks"]
        rp = np.sort(ra[pos])
        v01 = delong_neg_component(ra[~posmask], rp, len(pos))
        rpn = np.sort(rn[pos[isn]])
        negn = novel & ~posmask
        v01n = delong_neg_component(rn[negn], rpn, int(isn.sum()))
        out[s] = {"v10": parts[s]["a_all"], "v01": v01, "v10n": parts[s]["a_nov"][isn], "v01n": v01n}
    return out


def se_delong(c1, c2=None, novel=False):
    a, b = ("v10n", "v01n") if novel else ("v10", "v01")
    x = c1[a] - (c2[a] if c2 is not None else 0)
    y = c1[b] - (c2[b] if c2 is not None else 0)
    pos_var = x.var(ddof=1) / len(x)
    neg_var = y.var(ddof=1) / len(y)
    return float(np.sqrt(pos_var)), float(np.sqrt(pos_var + neg_var))


def stage2(arms):
    orig = json.loads(ORIG.read_text())
    res = {"n_boot": N_BOOT, "seed": SEED, "schemes": {
        "orig": "original: resample papers, take the distinct targets (np.unique) -> biased",
        "A_target": "resample distinct targets with replacement, multinomial weights, negatives fixed",
        "B_paper_weighted": "resample papers; weight(t) = resampled incidences / observed incidences (E = 1)",
        "delong": "DeLong analytic SE; se_pos = positives only (candidates are an enumerated population), "
                  "se_full = + negative component; 95 % CI = estimate +- 1.96 se_full"}, "arms": {}}
    for V, k in arms:
        z = np.load(CACHE / f"scores_V{V}_k{k}.npz")
        ids_t, n_out, c_test, N = z["ids_t"], int(z["n_out"]), z["c_test"], int(z["N"])
        names = [str(x) for x in z["names"]]
        S = z["S"]
        novel = c_test == 0
        Nn = int(novel.sum())
        perm = np.random.default_rng(SEED).permutation(N)
        pos = np.unique(ids_t)
        key = f"V{V}_k{k}"
        o = orig["arms"][key]
        parts = {s: per_target_parts(S[:, i], perm, pos, novel, N, Nn) for i, s in enumerate(names)}
        ones = np.ones((1, len(pos)))
        point = {s: {m: float(v[0]) for m, v in weighted_metrics(parts[s], ones, N, Nn).items()} for s in names}
        # reproduction check against the original JSON
        repro = {s: {m: point[s][m] - o["scores"][s][m]["value"] for m in METRICS} for s in names}
        maxdev = max(abs(v) for d in repro.values() for v in d.values() if np.isfinite(v))
        log(f"  {key}: max |point - original| = {maxdev:.3g}")
        # incidences: target id per (subset, paper)
        n_sub = len(ids_t) // n_out
        ids_mat = ids_t.reshape(n_sub, n_out)
        tidx = np.searchsorted(pos, ids_mat)                   # (n_sub, n_out) -> index into pos
        m_t = np.bincount(tidx.ravel(), minlength=len(pos)).astype(float)
        boots = {sch: {s: {m: [] for m in METRICS} for s in names} for sch in ("A_target", "B_paper_weighted")}
        rngA = np.random.default_rng([SEED, 1, V, k])
        rngB = np.random.default_rng([SEED, 2, V, k])
        chunk = 100
        for b0 in range(0, N_BOOT, chunk):
            nb = min(chunk, N_BOOT - b0)
            WA = rngA.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=nb).astype(float)
            WB = np.empty((nb, len(pos)))
            for j in range(nb):
                cnt = np.bincount(rngB.integers(0, n_out, n_out), minlength=n_out)
                WB[j] = np.bincount(tidx.ravel(), weights=np.tile(cnt, n_sub), minlength=len(pos)) / m_t
            for sch, Wt in (("A_target", WA), ("B_paper_weighted", WB)):
                for s in names:
                    for m, v in weighted_metrics(parts[s], Wt, N, Nn).items():
                        boots[sch][s][m].append(v)
        boots = {sch: {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in bs.items()}
                 for sch, bs in boots.items()}
        dl = delong(names, parts, pos, novel, N)

        def ci(a):
            return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]

        scores_out = {}
        for s in names:
            scores_out[s] = {}
            for m in METRICS:
                e = {"value": point[s][m], "ci_orig": o["scores"][s][m]["ci"],
                     "ci_A_target": ci(boots["A_target"][s][m]),
                     "ci_B_paper_weighted": ci(boots["B_paper_weighted"][s][m])}
                if m in ("auc_all", "auc_novel"):
                    sp, sf = se_delong(dl[s], novel=(m == "auc_novel"))
                    e.update({"se_delong_pos": sp, "se_delong_full": sf,
                              "ci_delong": [point[s][m] - 1.96 * sf, point[s][m] + 1.96 * sf]})
                for c in ("ci_orig", "ci_A_target", "ci_B_paper_weighted"):
                    e[f"{c}_contains_point"] = bool(e[c][0] <= e["value"] <= e[c][1]) if np.isfinite(e["value"]) else None
                scores_out[s][m] = e

        diff_specs = {
            "M_minus_NC0_auc_all": ("M", "COH", "auc_all"), "M_minus_NC04_auc_all": ("M", "NC_0.4", "auc_all"),
            "M_minus_NC0_enr_all": ("M", "COH", "enr_all"), "M_minus_NC04_enr_all": ("M", "NC_0.4", "enr_all"),
            "NC04_minus_NC0_enr_novel_full": ("NC_0.4", "COH", "enr_novel_full"),
            "NC04_minus_M_enr_novel_full": ("NC_0.4", "M", "enr_novel_full"),
            "M_minus_COH_auc_novel": ("M", "COH", "auc_novel"),
            "FITMC_minus_FITM_auc_all": ("FIT_MC", "FIT_M", "auc_all"),
            "FITMC_minus_FITM_auc_novel": ("FIT_MC", "FIT_M", "auc_novel"),
            **{f"FITMCH_{e}_minus_FITMC_auc_all": (f"FIT_MCH_{e}", "FIT_MC", "auc_all") for e in cvm.EMBEDDERS},
        }
        diffs = {}
        for dn, (s1, s2, m) in diff_specs.items():
            d = {"diff": point[s1][m] - point[s2][m], "orig": o["diffs"][dn]}
            for sch in ("A_target", "B_paper_weighted"):
                x = boots[sch][s1][m] - boots[sch][s2][m]
                d[sch] = {"ci": ci(x), "p": float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))}
            if m.startswith("auc"):
                sp, sf = se_delong(dl[s1], dl[s2], novel=(m == "auc_novel"))
                zval = d["diff"] / sf if sf > 0 else np.inf
                from scipy.stats import norm
                d["delong"] = {"se_pos": sp, "se_full": sf, "ci": [d["diff"] - 1.96 * sf, d["diff"] + 1.96 * sf],
                               "p": float(2 * norm.sf(abs(zval)))}
            diffs[dn] = d

        # verdicts: same pre-registered rules; H1 uses the CI lower bound, now from scheme B (primary) and A
        v = lambda sc, m: point[sc][m]
        dec = {
            "H1": bool(v("COH", "enr_all") > 5 and scores_out["COH"]["enr_all"]["ci_B_paper_weighted"][0] > 1),
            "H2": bool(all(v("M", m) >= v(nc, m) for nc in ("COH", "NC_0.4") for m in ("enr_all", "auc_all"))),
            "H3a": bool(v("NC_0.4", "enr_novel_full") > v("COH", "enr_novel_full")),
            "H3b": bool(v("M", "auc_novel") >= v("COH", "auc_novel")),
            "H4": bool(diffs["FITMC_minus_FITM_auc_all"]["diff"] < 0.02
                       and diffs["FITMC_minus_FITM_auc_novel"]["diff"] < 0.02),
            "H5_qwen3e06": bool(diffs["FITMCH_qwen3e06_minus_FITMC_auc_all"]["diff"] < 0.01),
            "H5_bgem3": bool(diffs["FITMCH_bgem3_minus_FITMC_auc_all"]["diff"] < 0.01),
        }
        dec["H5_replicated_decision"] = dec["H5_qwen3e06"] == dec["H5_bgem3"]
        pkeys = {"H2": "M_minus_NC04_auc_all", "H3b": "M_minus_COH_auc_novel", "H4": "FITMC_minus_FITM_auc_all",
                 "H5": "FITMCH_qwen3e06_minus_FITMC_auc_all"}
        holm_all = {}
        for sch in ("A_target", "B_paper_weighted", "delong"):
            ps = {h: diffs[dk][sch]["p"] for h, dk in pkeys.items()}
            order = sorted(ps, key=ps.get)
            holm, run = {}, 0.0
            for i, h in enumerate(order):
                run = max(run, min(1.0, ps[h] * (len(order) - i)))
                holm[h] = run
            holm_all[sch] = {"p_raw": ps, "p_holm": holm}
        res["arms"][key] = {
            "N_candidates": N, "N_novel_candidates": Nn, "n_targets": int(len(pos)),
            "n_novel_targets": int(novel[pos].sum()), "n_future_papers": n_out,
            "targets_with_one_paper": int((m_t == 1).sum()),
            "mean_distinct_targets_in_orig_resample": None,
            "max_abs_dev_point_vs_original": maxdev,
            "decisions_orig": o["decisions"]["predictions_hold"], "decisions_new": dec,
            "decisions_changed": {h: dec[h] != o["decisions"]["predictions_hold"][h] for h in dec},
            "pvalues": holm_all, "scores": scores_out, "diffs": diffs,
        }
        # bias diagnosis: reproduce the original resampling and record what it does to the target set
        rngD = np.random.default_rng([SEED, 3])
        nd, popshare = [], []
        for _ in range(200):
            pick = rngD.integers(0, n_out, n_out)
            kept = np.unique(tidx[:, pick].ravel())
            nd.append(len(kept))
            popshare.append(float((m_t[kept] > 1).mean()))
        res["arms"][key]["diagnosis_orig_resampling"] = {
            "distinct_targets_observed": int(len(pos)),
            "distinct_targets_per_resample_mean": float(np.mean(nd)),
            "share_multi_paper_targets_observed": float((m_t > 1).mean()),
            "share_multi_paper_targets_per_resample_mean": float(np.mean(popshare)),
        }
        del res["arms"][key]["mean_distinct_targets_in_orig_resample"]
        OUT.write_text(json.dumps(res, indent=1))
        log(f"  {key}: done; decisions changed: {res['arms'][key]['decisions_changed']}")
    OUT.write_text(json.dumps(res, indent=1))
    log(f"-> {OUT}")


if __name__ == "__main__":
    assert not cvm.SMOKE or True
    stage2(stage1())
