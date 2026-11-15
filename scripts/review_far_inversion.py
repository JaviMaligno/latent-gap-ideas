"""R4 of docs/prereg/review_far_decoding.md: a trained inverter (vec2text, gtr-t5-base) on REAL, GAP and FAR targets
rebuilt in the gtr space. No paid API, no gateway: Kaggle T4 x2 for the GPU stages, subscription CLIs for the gate.

Stages (DATA = ~/Documents/experiment-data/latent-gap-ideas):
  prepare   non-adult PAST texts of Steam and arXiv (corpus order) -> DATA/kaggle-upload/review-gtr-texts/
            then: kaggle datasets create -p <that dir>; kaggle kernels push -p kaggle/review-gtr-embed
  [Kaggle]  kaggle/review-gtr-embed -> emb_gtr_{steam,arxiv}.npy (float16 raw mean-pooled, row-aligned)
            kaggle kernels output javieraguilarmartin/latent-gap-review-gtr-embed -p DATA/kaggle-output/review-gtr-embed
  build     50 REAL / GAP / FAR targets per domain on the normalised gtr past matrix (latent_gap.targets, rng
            default_rng(20261114 + 500 + 10c)), hidden sets, ceilings, W1 geometry, calibration sample (seed 20261114+40)
            -> DATA/runs/review-far-decoding/gtr_targets_{dom}.npz + gtr_build.json, and the Kaggle upload
            DATA/kaggle-upload/review-gtr-targets/ (then: kaggle datasets create -p ...; kaggle kernels push -p kaggle/review-gtr-invert)
  [Kaggle]  kaggle/review-gtr-invert -> inversions_{dom}.jsonl, truncations_{dom}.jsonl (texts re-embedded on GPU)
            kaggle kernels output javieraguilarmartin/latent-gap-review-gtr-invert -p DATA/kaggle-output/review-gtr-invert
  score     h, ceilings, margins, discrimination, coherence gate (gibberish rule -> JUDGE1 Haiku 5.5 >= 7 -> JUDGE2
            gpt-6-luna via Codex >= 7, subscription CLIs), gates V-gate / V0, readings V1 / V2 / token soup
            -> docs/prereg/review_far_inversion.json

    uv run python scripts/review_far_inversion.py prepare
    uv run python scripts/review_far_inversion.py build [--smoke-invert]
    uv run python scripts/review_far_inversion.py score [--workers 4]
    uv run python scripts/review_far_inversion.py score --dry-run     # gate calls it would make, no CLI call
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from latent_gap import targets as TG

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/prereg/review_far_inversion.json"
RUNS = DATA / "runs/review-far-decoding"
UP_TEXTS = DATA / "kaggle-upload/review-gtr-texts"
UP_TARGETS = DATA / "kaggle-upload/review-gtr-targets"
EMB_OUT = DATA / "kaggle-output/review-gtr-embed"
INV_OUT = DATA / "kaggle-output/review-gtr-invert"
JUDGE_CACHE = RUNS / "gtr_judgements.jsonl"
KAGGLE_USER = "javieraguilarmartin"
SEED = 20261114
B = 10_000
DOMS = ("steam", "arxiv")
ARMS = ("REAL", "GAP", "FAR")
BUDGETS = ("C0", "C1", "C2")
N_TARGETS = 50
N_CALIB = 50
ESCAPE = 0.05
JUDGE1, JUDGE2, COH_PASS = "claude:claude-haiku-5-5", "codex:gpt-6-luna", 7  # scripts/seed_rewrite.py


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def past_texts(dom: str) -> pd.DataFrame:
    meta = pd.read_parquet(DATA / f"prepared/{dom}/corpus.parquet", columns=["text", "split", "adult"])
    keep = (~meta["adult"] & meta["split"].eq("past")).to_numpy()
    t = meta.loc[keep, ["text"]].reset_index(drop=True)
    t.insert(0, "row", np.arange(len(t)))
    return t


def dataset_meta(path: Path, slug: str, title: str) -> None:
    (path / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{KAGGLE_USER}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=1) + "\n")


# ---------------------------------------------------------------- prepare

def prepare() -> None:
    UP_TEXTS.mkdir(parents=True, exist_ok=True)
    for dom in DOMS:
        t = past_texts(dom)
        t.to_parquet(UP_TEXTS / f"texts_{dom}.parquet", index=False)
        print(f"{dom}: {len(t)} non-adult past texts")
    dataset_meta(UP_TEXTS, "latent-gap-review-gtr-texts", "latent-gap review gtr texts")
    print(f"kaggle datasets create -p {UP_TEXTS}\nkaggle kernels push -p {ROOT / 'kaggle/review-gtr-embed'}")


# ---------------------------------------------------------------- build

def geometry(P: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED + 1)
    a, b = rng.integers(0, len(P), (2, 100_000))
    while (bad := np.flatnonzero(a == b)).size:
        b[bad] = rng.integers(0, len(P), bad.size)
    pair = np.einsum("nd,nd->n", P[a], P[b])
    M = np.zeros((P.shape[1], P.shape[1]))
    for s in range(0, len(P), 8192):
        c = P[s:s + 8192].astype(np.float64)
        M += c.T @ c
    w = np.linalg.eigvalsh(M / len(P))[::-1]
    k90 = int(np.searchsorted(np.cumsum(w) / w.sum(), 0.90) + 1)
    return {"pair_cos_median": float(np.median(pair)), "norm_of_mean": float(np.linalg.norm(P.mean(0))),
            "k90": k90, "d": int(P.shape[1])}


def build(emb_dir: Path, smoke_invert: bool, fake: bool, out_runs: Path, out_up: Path) -> None:
    out_runs.mkdir(parents=True, exist_ok=True)
    out_up.mkdir(parents=True, exist_ok=True)
    summary = {}
    for c, dom in enumerate(DOMS):
        texts = past_texts(dom)
        if fake:  # smoke only: a stand-in matrix with the gtr shape, never a result
            raw = np.load(DATA / f"prepared/{dom}/emb_qwen3e06.npy", mmap_mode="r")
            meta = pd.read_parquet(DATA / f"prepared/{dom}/corpus.parquet", columns=["split", "adult"])
            keep = (~meta["adult"] & meta["split"].eq("past")).to_numpy()
            R = np.asarray(raw[keep][:6000, :768], dtype=np.float32) * 3.0
            texts = texts.iloc[:6000]
        else:
            R = np.load(emb_dir / f"emb_gtr_{dom}.npy").astype(np.float32)
        if len(R) != len(texts):
            raise SystemExit(f"{dom}: {len(R)} embeddings vs {len(texts)} past texts")
        norms = np.linalg.norm(R, axis=1)
        P = (R / norms[:, None]).astype(np.float32)
        rng = np.random.default_rng(SEED + 500 + 10 * c)
        Rt, hidden, ridx = TG.real(P, rng, N_TARGETS)
        G = TG.gap(P, rng, N_TARGETS)
        F = TG.far(P, rng, N_TARGETS)
        creal = np.empty(N_TARGETS, dtype=np.float32)
        for i, (t, h) in enumerate(zip(Rt, hidden)):
            s = P @ t
            s[h] = -np.inf  # the target's own row and its cos > 0.95 duplicates
            creal[i] = s.max()
        ceil = {"REAL": creal, "GAP": TG.max_cos(P, G), "FAR": TG.max_cos(P, F)}
        calib = np.random.default_rng(SEED + 40).choice(len(P), N_CALIB, replace=False)
        scale = float(np.median(norms))
        hid = np.array(hidden + [None], dtype=object)[:-1]
        np.savez(out_runs / f"gtr_targets_{dom}.npz", REAL=Rt, GAP=G, FAR=F, REAL_idx=ridx, REAL_hidden=hid,
                 ceil_REAL=ceil["REAL"], ceil_GAP=ceil["GAP"], ceil_FAR=ceil["FAR"], calib_idx=calib, scale=scale)
        np.savez(out_up / f"targets_{dom}.npz", REAL=Rt, GAP=G, FAR=F, REAL_raw=R[ridx], scale=np.float32(scale))
        tx = texts.set_index("row")["text"]
        (out_up / f"texts_{dom}.json").write_text(json.dumps(
            {"real": [{"k": k, "row": int(i), "text": tx[int(i)]} for k, i in enumerate(ridx)],
             "calib": [{"row": int(i), "text": tx[int(i)]} for i in calib]}))
        summary[dom] = {"n_past": int(len(P)), "median_raw_norm": scale, "geometry": geometry(P),
                        **{f"ceiling_median_{a}": float(np.median(v)) for a, v in ceil.items()},
                        **{f"ceiling_max_{a}": float(np.max(v)) for a, v in ceil.items()},
                        "REAL_hidden_sizes_median": float(np.median([len(h) for h in hidden]))}
        print(dom, json.dumps(summary[dom]), flush=True)
        del R, P
    (out_up / "run_info.json").write_text(json.dumps({"smoke": smoke_invert, "n_smoke": 25, "seed": SEED,
                                                      "budgets": {"C0": [0, 0], "C1": [20, 1], "C2": [50, 4]}}) + "\n")
    dataset_meta(out_up, "latent-gap-review-gtr-targets", "latent-gap review gtr targets")
    (out_runs / "gtr_build.json").write_text(json.dumps({"fake": fake, **summary}, indent=1) + "\n")
    print(f"kaggle datasets create -p {out_up}    # later: kaggle datasets version -p {out_up} -m full\n"
          f"kaggle kernels push -p {ROOT / 'kaggle/review-gtr-invert'}")


# ---------------------------------------------------------------- score

class Gate:
    """Gibberish rule, then JUDGE1 >= 7, then JUDGE2 >= 7 (A.4). Judgements cached by (judge, domain, sha256)."""

    def __init__(self, path: Path, dry: bool, no_judge: bool):
        self.path, self.dry, self.no_judge = path, dry, no_judge
        self.lock = threading.Lock()
        self.cache = {}
        self.calls = {JUDGE1: 0, JUDGE2: 0}
        if path.exists():
            for l in path.read_text().splitlines():
                r = json.loads(l)
                self.cache[r["judge"], r["domain"], r["sha256"]] = r["coherence"]

    def rate(self, spec: str, dom: str, text: str) -> int | None:
        from latent_gap.coherent import judge
        from latent_gap.domains import DOMAINS
        key = (spec, dom, hashlib.sha256(text.encode()).hexdigest())
        if key in self.cache:
            return self.cache[key]
        self.calls[spec] += 1
        if self.dry or self.no_judge:
            return None
        r = judge(spec, DOMAINS[dom].noun, text)
        with self.lock:
            self.cache[key] = r["coherence"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"judge": spec, "domain": dom, "sha256": key[2], "coherence": r["coherence"],
                                    "reason": r.get("reason")}) + "\n")
        return r["coherence"]

    def __call__(self, dom: str, text: str) -> dict:
        from latent_gap.coherent import gibberish_rule, gibberish_stats
        rule = gibberish_rule(gibberish_stats(text)) if text.strip() else "empty"
        if rule:
            return {"gibberish": rule, "j1": None, "j2": None, "pass": False}
        if self.no_judge:
            return {"gibberish": "", "j1": None, "j2": None, "pass": True}  # smoke only: gibberish rule alone
        j1 = self.rate(JUDGE1, dom, text)
        if self.dry and j1 is None:
            self.calls[JUDGE2] += 1  # upper bound: JUDGE2 runs only on JUDGE1 passes
            return {"gibberish": "", "j1": None, "j2": None, "pass": False}
        j2 = self.rate(JUDGE2, dom, text) if (j1 or 0) >= COH_PASS else None
        return {"gibberish": "", "j1": j1, "j2": j2, "pass": (j1 or 0) >= COH_PASS and (j2 or 0) >= COH_PASS}


def clopper(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def score(inv_dir: Path, runs: Path, out: Path, gate: Gate, workers: int, n_boot: int) -> None:
    res = {"prereg": "docs/prereg/review_far_decoding.md#R4", "B": n_boot,
           "build": json.loads((runs / "gtr_build.json").read_text()), "domains": {}}
    rng = np.random.default_rng(SEED + 41)
    for dom in DOMS:
        z = np.load(runs / f"gtr_targets_{dom}.npz", allow_pickle=True)
        T = {a: unit(z[a].astype(np.float64)) for a in ARMS}
        ceil = {a: z[f"ceil_{a}"].astype(float) for a in ARMS}
        inv = read_jsonl(inv_dir / f"inversions_{dom}.jsonl")
        tru = read_jsonl(inv_dir / f"truncations_{dom}.jsonl")
        texts = [r["text"] for r in inv] + [r["text"] for r in tru]
        with ThreadPoolExecutor(workers) as ex:
            gates = list(ex.map(lambda t: gate(dom, t), texts))
        for r, g in zip(inv + tru, gates):
            r["gate"] = g
        cal = [r for r in tru if r["kind"] == "calib"]
        realtx = [r for r in tru if r["kind"] == "real"]
        vgate_rate = float(np.mean([r["gate"]["pass"] for r in cal])) if cal else float("nan")
        D = {"calibration": {"random_truncations_pass": vgate_rate, "n": len(cal),
                             "REAL_own_truncations_pass": float(np.mean([r["gate"]["pass"] for r in realtx])) if realtx else None},
             "V_gate": bool(vgate_rate >= 0.80)}
        rt = [r for r in inv if r["budget"] == "RT"]
        if rt:
            D["round_trip"] = {"n": len(rt), "bleu_mean": float(np.mean([r["bleu"] for r in rt])),
                               "token_f1_mean": float(np.mean([r["token_f1"] for r in rt])),
                               "exact": int(sum(r["text"].strip() == r["ref_text"].strip() for r in rt))}
        best = {}  # (arm, k) -> best coherent h over budgets
        arms_out = {}
        for arm in ARMS:
            rows = [r for r in inv if r["arm"] == arm and r["budget"] in BUDGETS]
            for r in rows:
                e = unit(np.asarray(r["emb"], dtype=np.float64))
                s = T[arm] @ e
                k = r["k"]
                r["h"] = float(s[k])
                r["disc"] = float(s[k] - np.delete(s, k).mean())
                r["margin"] = r["h"] - ceil[arm][k]
                if r["gate"]["pass"]:
                    best[arm, k] = max(best.get((arm, k), -np.inf), r["h"])
            ks = sorted({r["k"] for r in rows})
            per_b = {}
            for b in BUDGETS:
                h = np.array([r["h"] for r in rows if r["budget"] == b])
                if not len(h):
                    continue
                bm = np.median(h[rng.integers(0, len(h), (n_boot, len(h)))], axis=1)
                per_b[b] = {"n": int(len(h)), "median_h": float(np.median(h)),
                            "median_h_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                            "median_disc": float(np.median([r["disc"] for r in rows if r["budget"] == b])),
                            "gate_pass_rate": float(np.mean([r["gate"]["pass"] for r in rows if r["budget"] == b]))}
            esc = sum(best.get((arm, k), -np.inf) >= ceil[arm][k] + ESCAPE for k in ks)
            beyond = sum(best.get((arm, k), -np.inf) > ceil[arm][k] for k in ks)
            coh = [r["h"] for r in rows if r["gate"]["pass"]]
            incoh = [r["h"] for r in rows if not r["gate"]["pass"]]
            coh_best = [best[arm, k] for k in ks if (arm, k) in best]
            arms_out[arm] = {"n_targets": len(ks), "median_ceiling": float(np.median(ceil[arm][ks])) if ks else None,
                             "budgets": per_b, "escape_k": int(esc), "escape_share": esc / len(ks) if ks else None,
                             "escape_ci95": clopper(esc, len(ks)) if ks else None,
                             "beyond_ceiling_share": beyond / len(ks) if ks else None,
                             "max_h_any": float(max(r["h"] for r in rows)) if rows else None,
                             "max_h_coherent": float(max(coh)) if coh else None,
                             "max_h_incoherent": float(max(incoh)) if incoh else None,
                             "median_best_coherent_h": float(np.median(coh_best)) if coh_best else None}
            a = arms_out[arm]
            a["token_soup"] = bool(a["max_h_incoherent"] is not None and
                                   a["max_h_incoherent"] - (a["max_h_coherent"] or -1.0) >= 0.1)
        D["arms"] = arms_out
        r_ = arms_out.get("REAL", {})
        c2 = r_.get("budgets", {}).get("C2")
        D["V0"] = bool(c2 and c2["median_h"] >= 0.90 and (r_["beyond_ceiling_share"] or 0) >= 0.70)
        D["informative"] = D["V_gate"] and D["V0"]
        if D["informative"]:
            g, f = arms_out.get("GAP", {}), arms_out.get("FAR", {})
            D["V1_GAP_escape"] = bool((g.get("escape_share") or 0) >= 0.50)
            D["V2_FAR_escape_ge_20pct"] = bool((f.get("escape_share") or 0) >= 0.20)
            D["V2_FAR_strong_against"] = bool(D["V2_FAR_escape_ge_20pct"] and (f.get("median_best_coherent_h") or 0) >= 0.30)
        res["domains"][dom] = D
        with (runs / f"gtr_scored_{dom}.jsonl").open("w") as fh:
            for r in inv + tru:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "emb"}) + "\n")
    info = [d for d in res["domains"].values() if d.get("informative")]
    if len(info) < len(DOMS):
        res["FAR_reading"] = "uninformative for: " + ",".join(k for k, d in res["domains"].items() if not d.get("informative"))
    if info:
        against = any(d["V2_FAR_escape_ge_20pct"] for d in info)
        res["FAR_verdict_informative_domains"] = ("counts strongly against" if any(d["V2_FAR_strong_against"] for d in info)
                                                  else "counts against" if against else "supports (< 20 % escape)")
    res["judge_calls_made_or_planned"] = gate.calls
    if gate.dry:
        print(json.dumps({"dry_run": True, "planned_calls_upper_bound": gate.calls}))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({d: {k: v for k, v in x.items() if k != "arms"} for d, x in res["domains"].items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "build", "score"])
    ap.add_argument("--smoke-invert", action="store_true", help="build: run_info.json asks the kernel for 25 REAL targets")
    ap.add_argument("--fake-emb", action="store_true", help="build smoke only: Qwen3 stand-in, never a result")
    ap.add_argument("--emb-dir", type=Path, default=EMB_OUT)
    ap.add_argument("--inv-dir", type=Path, default=INV_OUT)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--upload", type=Path, default=UP_TARGETS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--dry-run", action="store_true", help="score: count gate calls, make none")
    ap.add_argument("--no-judge", action="store_true", help="score smoke only: gibberish rule as the whole gate")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "build":
        if args.fake_emb and (args.runs == RUNS or args.upload == UP_TARGETS):
            raise SystemExit("--fake-emb must write to temp --runs and --upload")
        build(args.emb_dir, args.smoke_invert, args.fake_emb, args.runs, args.upload)
    else:
        if (args.no_judge or args.boot != B) and args.out == OUT:
            raise SystemExit("a smoke score must not write the registered JSON")
        score(args.inv_dir, args.runs, args.out, Gate(args.runs / JUDGE_CACHE.name, args.dry_run, args.no_judge),
              args.workers, args.boot)


if __name__ == "__main__":
    main()
