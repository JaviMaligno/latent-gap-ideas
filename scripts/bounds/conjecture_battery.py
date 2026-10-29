"""Falsification battery for the FAR-ceiling conjecture of docs/bounds.md §4, run in two committed steps.

    uv run python scripts/bounds/conjecture_battery.py --predict   # corpus statistics only -> predictions.csv
    git commit ...                                                  # predictions frozen before any measurement
    uv run python scripts/bounds/conjecture_battery.py --measure   # observed median ceilings -> battery.csv

Conjecture: the median corpus ceiling of the K most isolated of M uniform directions is
    (z_N sqrt(1 - r^2) + r Phi^{-1}(K / (2M))) / sqrt(d),   r = ||mean of the corpus||,
z_N the Gaussian expected-maximum approximation of geometry.py. Each configuration fixes domain, embedding space,
corpus size N (a random subsample of the past corpus, the same rows for prediction and measurement) and (K, M).
Selection and ceiling use the same subsample, as for the FAR arm. No model calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import DATA, expected_max, unit  # noqa: E402

OUT = DATA / "runs" / "bounds_battery"
SEED = 20261024
SIZES = (2_000, 5_000, 10_000, 20_000, None)             # None = full past corpus
SELECTIONS = ((50, 20_000), (10, 20_000), (500, 20_000), (50, 2_000))  # (K, M); non-default ones at full N only
SPACES = ("qwen3e06", "bgem3")
DOMAINS = ("steam", "arxiv", "patents")  # index seeds the subsample: append only
# Fixed before measuring: absolute error of the median ceiling.
STRICT, LENIENT = 0.005, 0.02


def corpus(domain: str, space: str) -> np.ndarray:
    root = DATA / "prepared" / domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
    emb = np.load(root / f"emb_{space}.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    return unit(emb[keep][(meta["split"][keep] == "past").to_numpy()]).astype(np.float32)


def configs(domains):
    for domain in domains:
        for space in SPACES:
            for n in SIZES:
                for k, m in SELECTIONS:
                    if n is None or (k, m) == SELECTIONS[0]:
                        yield domain, space, n, k, m


def subsample(P: np.ndarray, n, domain: str, space: str) -> np.ndarray:
    if n is None or n >= len(P):
        return P
    rng = np.random.default_rng([SEED, n, DOMAINS.index(domain)])  # same rows in both spaces and both steps
    return P[rng.choice(len(P), n, replace=False)]


def predict(Q: np.ndarray, k: int, m: int) -> float:
    n, d = Q.shape
    r = float(np.linalg.norm(Q.mean(0)))
    return (expected_max(n) * np.sqrt(1 - r * r) + r * norm.ppf(k / (2 * m))) / np.sqrt(d)


def measure(Q: np.ndarray, k: int, m: int, domain: str, space: str, n) -> float:
    rng = np.random.default_rng([SEED + 1, m, DOMAINS.index(domain), SPACES.index(space), n or 0])
    R = unit(rng.standard_normal((m, Q.shape[1]))).astype(np.float32)
    ceil = np.concatenate([(Q @ R[i:i + 2000].T).max(0) for i in range(0, m, 2000)])
    return float(np.median(np.sort(ceil)[:k]))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--predict", action="store_true")
    g.add_argument("--measure", action="store_true")
    ap.add_argument("--domains", default="steam,arxiv")
    args = ap.parse_args()
    domains = args.domains.split(",")
    tag = "" if domains == ["steam", "arxiv"] else "_" + "_".join(domains)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    rows = []
    if args.measure:
        pred = pd.read_csv(OUT / f"predictions{tag}.csv")
    for i, (domain, space, n, k, m) in enumerate(configs(domains), 1):
        if (domain, space) not in cache:
            cache = {(domain, space): corpus(domain, space)}
        Q = subsample(cache[(domain, space)], n, domain, space)
        row = {"domain": domain, "space": space, "N": len(Q), "K": k, "M": m}
        if args.predict:
            row["r"] = round(float(np.linalg.norm(Q.mean(0))), 5)
            row["predicted"] = round(predict(Q, k, m), 5)
        else:
            row["observed"] = round(measure(Q, k, m, domain, space, n), 5)
        rows.append(row)
        print(f"[{i}] {row}", flush=True)
    df = pd.DataFrame(rows)
    if args.predict:
        df.to_csv(OUT / f"predictions{tag}.csv", index=False)
        return
    df = pred.merge(df, on=["domain", "space", "N", "K", "M"])
    df["error"] = (df.observed - df.predicted).round(5)
    df.to_csv(OUT / f"battery{tag}.csv", index=False)
    print(df.to_string(index=False))
    e = df.error.abs()
    print(f"\nmedian |error| {e.median():.4f}, max {e.max():.4f}; within strict {STRICT}: {(e <= STRICT).mean():.0%}, "
          f"within lenient {LENIENT}: {(e <= LENIENT).mean():.0%}; mean signed error {df.error.mean():+.4f}")


if __name__ == "__main__":
    main()
