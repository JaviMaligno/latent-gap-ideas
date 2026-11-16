"""Section 4 in leakage-free spaces (docs/prereg/review_leakfree_encoder.md).

Runs on Kaggle (kernels kaggle/review-leakfree-lsa and kaggle/review-leakfree-mpnet), reusing the registered code
unchanged: where_future_lands.analyse (F1-F5) and review_hole_validity.features/.tests (raw space, K = 256, H0-H3).

  python review_leakfree_encoder.py lsa                 # fit TF-IDF + SVD on past text, write emb_lsa256.npy
  python review_leakfree_encoder.py analyse <space> <domain> ...   # F-tests + gap-score validity
  uv run python scripts/review_leakfree_encoder.py collect <kernel_output_dir> ...   # local: JSON + decisions

Environment: LF_DATA = writable data root holding prepared/<domain>/{corpus.parquet, emb_<space>.npy};
LF_OUT = output directory. Patches are applied at import so spawned workers see them too.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import where_future_lands as W  # noqa: E402
import review_hole_validity as RHV  # noqa: E402

SEED = W.SEED
LF_DATA = Path(os.environ.get("LF_DATA", str(W.DATA)))
LF_OUT = Path(os.environ.get("LF_OUT", str(W.DATA / "runs/review-leakfree")))
DOMAINS = {"lsa256": ("steam", "arxiv", "patents"), "mpnet": ("steam", "arxiv")}

# ---------------------------------------------------------------- registered adaptations (see prereg "Analyses")
W.DATA = LF_DATA
RHV.D = LF_DATA
RHV.SPACES = ("raw",)
RHV.C.ks = (256,)
RHV.C.out = LF_OUT / "hole_validity"
if os.environ.get("LF_SMOKE"):  # tiny constants for a local wiring test only; never for registered outputs
    W.K_CELLS, W.N_DRAWS, W.N_BOOT = 32, 20_000, 50
    RHV.C.ks, RHV.C.kbase, RHV.C.B, RHV.C.n_draws, RHV.C.m_probe, RHV.C.n_planted, RHV.C.planted_min = \
        (32,), 32, 30, 20_000, 60, 20, 40
KB = f"raw_K{RHV.C.kbase}"
_orig_outcomes = RHV.outcomes
RHV.outcomes = lambda d, e: {k: v for k, v in _orig_outcomes(d, e).items() if k == KB}
RHV.configs = lambda d, e: [(KB, h) for h in RHV.DEFS]
RHV.reproduction = lambda *a, **k: {"passes": True, "note": "not applicable: new space, nothing registered to reproduce"}
_orig_boot = RHV.boot_idx


def _boot_idx(d, e, k, n=None, seed=None):
    if seed is not None:
        return _orig_boot(d, e, k, n, seed)
    return np.random.default_rng(SEED + k).integers(0, k, (RHV.C.B, k))


RHV.boot_idx = _boot_idx


def log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


def jdump(o, path: Path) -> None:
    path.write_text(json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


# ---------------------------------------------------------------- LSA
def fit_lsa(domains: list[str]) -> None:
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    for d in domains:
        out = LF_DATA / "prepared" / d / "emb_lsa256.npy"
        if out.exists():
            continue
        t0 = time.time()
        meta = pd.read_parquet(LF_DATA / "prepared" / d / "corpus.parquet", columns=["text", "split", "adult"])
        fit_rows = ((meta["split"] == "past") & ~meta["adult"]).to_numpy()
        vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=5, max_df=0.5, sublinear_tf=True,
                              dtype=np.float32)
        Xp = vec.fit_transform(meta["text"][fit_rows])
        svd = TruncatedSVD(n_components=256, algorithm="randomized", n_iter=7, random_state=SEED).fit(Xp)
        log(f"{d}: tfidf on {fit_rows.sum()} past rows, vocab {len(vec.vocabulary_)}, nnz {Xp.nnz}; "
            f"SVD explained variance {svd.explained_variance_ratio_.sum():.3f} ({time.time() - t0:.0f}s)")
        E = svd.transform(vec.transform(meta["text"])).astype(np.float32)
        nrm = np.linalg.norm(E, axis=1, keepdims=True)
        n_zero = int((nrm[:, 0] == 0).sum())
        E /= np.where(nrm == 0, 1, nrm)
        np.save(out, E)
        jdump({"domain": d, "n_fit_rows": int(fit_rows.sum()), "vocab": len(vec.vocabulary_), "nnz": int(Xp.nnz),
               "explained_variance": float(svd.explained_variance_ratio_.sum()), "n_zero_rows": n_zero,
               "seconds": time.time() - t0}, LF_OUT / f"lsa_{d}.json")
        log(f"{d}: wrote {out} {E.shape}, zero rows {n_zero}")


# ---------------------------------------------------------------- analysis
def f1_calibration(d: str, space: str) -> dict:
    """Held-out far share at the same 0.15 threshold (secondary F1 criterion), same split and RNG as W.analyse."""
    R, H, F, _, _, _ = RHV.load(d, space)
    f_max, _ = W.neighbours(F, R)
    h_max, _ = W.neighbours(H, R)
    out = {"far_share_future": float((f_max <= W.FAR_COS).mean()), "far_share_heldout": float((h_max <= W.FAR_COS).mean()),
           "max_cos_future_min": float(f_max.min()), "max_cos_heldout_min": float(h_max.min()),
           "max_cos_future_median": float(np.median(f_max)), "max_cos_heldout_median": float(np.median(h_max)),
           "max_cos_future_p01": float(np.percentile(f_max, 1)), "max_cos_heldout_p01": float(np.percentile(h_max, 1))}
    return out


def analyse(space: str, domains: list[str], workers: int) -> None:
    LF_OUT.mkdir(parents=True, exist_ok=True)
    for d in domains:
        t0 = time.time()
        p = LF_OUT / f"F_{d}_{space}.json"
        if not p.exists():
            res = W.analyse(d, space)
            res["F1_calibration"] = f1_calibration(d, space)
            res["seconds"] = time.time() - t0
            jdump(res, p)
            log(f"{d}/{space}: F-tests done in {time.time() - t0:.0f}s")
        t1 = time.time()
        RHV.features(d, space)
        log(f"{d}/{space}: hole features {time.time() - t1:.0f}s")
        t1 = time.time()
        RHV.tests(d, space, workers)
        log(f"{d}/{space}: hole tests {time.time() - t1:.0f}s; cell total {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- collect + decisions (local)
def collect(dirs: list[str]) -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "docs/prereg/review_leakfree_encoder.json"
    res: dict = {"cells": {}, "lsa_fit": {}}
    for dd in map(Path, dirs):
        for f in sorted(dd.glob("lsa_*.json")):
            res["lsa_fit"][f.stem[4:]] = json.loads(f.read_text())
        for space, doms in DOMAINS.items():
            for d in doms:
                fp = dd / f"F_{d}_{space}.json"
                td = dd / "hole_validity" / "tests" / f"{d}_{space}"
                if not fp.exists() or not (td / "V2.json").exists():
                    continue
                cell = {"F": json.loads(fp.read_text()), "V2": json.loads((td / "V2.json").read_text()), "defs": {}}
                with np.load(td / f"mom_{KB}.npz") as z:
                    cell["momentum"] = {"auc_direct": float(z["auc_direct"]), "auc_direct_ci": RHV.ci(z["direct"]),
                                        "cv_auc": float(z["cv"]), "cv_auc_cluster_ci": RHV.ci(z["arm"])}
                for h in RHV.DEFS:
                    t = json.loads((td / f"T_{KB}_{h}.json").read_text())
                    t["V1"] = json.loads((td / f"V1_{KB}_{h}.json").read_text())
                    cell["defs"][h] = t
                res["cells"][f"{d}/{space}"] = cell
    res["decisions"] = decide(res["cells"])
    jdump(res, out_path)
    print(json.dumps(res["decisions"], indent=1))


def decide(cells: dict) -> dict:
    out: dict = {}
    for space, doms in DOMAINS.items():
        keys = [f"{d}/{space}" for d in doms if f"{d}/{space}" in cells]
        if not keys:
            continue
        n = len(keys)
        need = 2  # >= 2 of 3 (LSA) / 2 of 2 (mpnet), as registered
        val = {}
        for h in RHV.DEFS:
            pl = sum(cells[k]["V2"][h]["auc"] >= 0.75 and cells[k]["V2"][h]["ci"][0] > 0.5 for k in keys)
            tr = dm = 0
            for k in keys:
                v = cells[k]["defs"][h]["V1"]
                a, p = v["a_density"], v["partial_a_given_bcd"]
                emp = a["rho"] <= -0.3 and a["ci"][1] < 0 and p["rho"] <= -0.2 and p["ci"][1] < 0
                tr += emp
                dm += v["r2_geom"]["value"] >= 0.5 and not emp
            sees, tracks, dom = pl >= need, tr >= need, dm >= need
            val[h] = {"n_planted": pl, "sees_planted_hole": sees, "n_tracks": tr, "tracks_emptiness": tracks,
                      "n_dominated": dm, "dominated_by_geometry": dom, "valid": sees or (tracks and not dom)}
        valid = [h for h in RHV.DEFS if val[h]["valid"]]
        fallback = not valid
        use = valid or ["H1", "H2", "H3"]
        per = {}
        for k in keys:
            c = cells[k]
            m = c["momentum"]["auc_direct"]
            r1a = all(m > c["defs"][h]["T1"]["auc_direct"] for h in set(use) | {"H0"})
            threat = [h for h in use if c["defs"][h]["T2"]["gain"] >= 0.03 and c["defs"][h]["T2"]["gain_cluster_ci"][0] > 0]
            inconcl = [h for h in use if c["defs"][h]["T2"]["gain"] >= 0.03 and h not in threat]
            above = [h for h in use if c["defs"][h]["T1"]["ci"][0] > 0.5]
            f1, cal = c["F"]["F1"], c["F"]["F1_calibration"]
            r3_lit = f1["far_share"] <= 0.001
            r3_cal = cal["far_share_future"] <= cal["far_share_heldout"] + 0.001
            per[k] = {"R1a_momentum_above_gap_scores": r1a, "R1b_C2_threatened_by": threat, "R1b_inconclusive": inconcl,
                      "R1": r1a and not threat, "R2_above_chance": above, "R2": not above,
                      "R3_literal": r3_lit, "R3_calibrated": r3_cal, "R3": r3_lit or r3_cal,
                      "R3_decided_by": "literal" if r3_lit else ("calibrated" if r3_cal else "fails")}
        fails = [f"{k}:{r}" for k, v in per.items() for r in ("R1", "R2", "R3") if not v[r]]
        out[space] = {"validity": val, "valid_definitions": valid, "fallback_to_paper_valid_defs": fallback,
                      "per_corpus": per, "verdict": "replicates Section 4" if not fails else "replicates except " + ", ".join(fails)}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "lsa":
        LF_OUT.mkdir(parents=True, exist_ok=True)
        fit_lsa(sys.argv[2:] or list(DOMAINS["lsa256"]))
    elif cmd == "analyse":
        analyse(sys.argv[2], sys.argv[3:], int(os.environ.get("LF_WORKERS", "4")))
    elif cmd == "collect":
        collect(sys.argv[2:])
