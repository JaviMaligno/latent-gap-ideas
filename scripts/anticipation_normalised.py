"""Style-free anticipation: every text in the comparison is rewritten by ONE normaliser before it is embedded.

For each idea (PLAIND and FAR of each generator) and each reference item (real past / real future, held out), take
its 10 nearest raw-embedding candidates in an equal-size past subsample and 10 in the future (from --future-from),
rewrite the idea and the 20 candidates with the judges' neutral-pitch prompt (domain rule included, so arXiv pitches
carry no results), re-embed, and score hit = best normalised future candidate beats best normalised past candidate.
Style differences between generators, and between LLM-era and older human text, are rewritten away; what is left is
content. Caveat: the candidate shortlist comes from the raw embedding.

    uv run python scripts/anticipation_normalised.py --domain arxiv
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from judge import NORMALISE, NORMALISER  # noqa: E402

from latent_gap.domains import DOMAINS  # noqa: E402
from latent_gap.embedder import LocalQwen  # noqa: E402
from latent_gap.gateway import Gateway  # noqa: E402

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
GENS = ("gpt-4o-0806", "gpt-6-luna", "claude-sonnet-5-5")
ARMS = ("PLAIND", "FAR")
SEED, K, N_REF = 20261027, 10, 50


def topk(E: np.ndarray, C: np.ndarray, k: int) -> np.ndarray:
    S = E @ C.T
    return np.argsort(-S, axis=1)[:, :k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="arxiv")
    ap.add_argument("--future-from", default="2024-01-01")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    domain = DOMAINS[args.domain]
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date", "text"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rng = np.random.default_rng(SEED)
    past_i = np.flatnonzero((meta["split"] == "past").to_numpy())
    fut = (meta["split"] == "future").to_numpy().copy()
    fut &= (pd.to_datetime(meta["date"], errors="coerce") >= args.future_from).to_numpy()
    fut_i = np.flatnonzero(fut)
    ref_p, ref_f = rng.choice(past_i, N_REF, replace=False), rng.choice(fut_i, N_REF, replace=False)
    pool_f = np.setdiff1d(fut_i, ref_f)
    pool_p = rng.choice(np.setdiff1d(past_i, ref_p), len(pool_f), replace=False)  # equal size: neither side wins by size

    # queries: (gen, arm, k, raw text, raw embedding)
    Q = [("real", "REF_PAST", int(i), meta.at[i, "text"], emb[i]) for i in ref_p]
    Q += [("real", "REF_FUTURE", int(i), meta.at[i, "text"], emb[i]) for i in ref_f]
    for g in GENS:
        for line in (DATA / "runs" / args.domain / f"grid-{g}" / "items.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["arm"] in ARMS and r.get("best"):
                e = np.array(r["best"]["emb"], np.float32)
                Q.append((g, r["arm"], r["k"], r["best"]["text"], e / np.linalg.norm(e)))
    E = np.stack([q[4] for q in Q])
    cand_p = pool_p[topk(E, emb[pool_p], K)]
    cand_f = pool_f[topk(E, emb[pool_f], K)]

    out = DATA / "runs" / args.domain / "style_free"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "pitch_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    need = {f"q:{g}:{a}:{k}": t for g, a, k, t, _ in Q}
    need.update({f"c:{i}": meta.at[int(i), "text"] for i in np.unique(np.concatenate([cand_p.ravel(), cand_f.ravel()]))})
    need = {k: v for k, v in need.items() if k not in cache}
    print(f"{len(Q)} queries, {len(need)} texts to normalise", flush=True)
    gw = Gateway(ledger=DATA / "ledger.jsonl")

    def norm(key: str, text: str):
        msg = NORMALISE.format(noun=domain.noun, does=domain.does, rule=domain.pitch_rule,
                               text=text[:2500].replace("\x00", ""))
        for budget in (2000, 8000):
            try:
                p = gw.chat([{"role": "user", "content": msg}], model=NORMALISER, max_tokens=budget,
                            tag=f"{args.domain}-stylefree").strip()
                if len(p) > 40:
                    return key, p
            except RuntimeError as e:  # content filter or transient: the text is dropped from the comparison
                print("FAILED", key, repr(e)[:120], flush=True)
                return key, None
        return key, None

    with ThreadPoolExecutor(args.workers) as ex:
        for n, fu in enumerate(as_completed([ex.submit(norm, k, t) for k, t in need.items()]), 1):
            k, p = fu.result()
            if p:
                cache[k] = p
            if n % 250 == 0:
                print(f"  {n}/{len(need)}", flush=True)
                cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))

    keys = sorted(cache)
    V = LocalQwen().embed([cache[k] for k in keys])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    vec = dict(zip(keys, V))
    rows = []
    for qi, (g, a, k, _, _) in enumerate(Q):
        qv = vec.get(f"q:{g}:{a}:{k}")
        cp = [vec[f"c:{i}"] for i in cand_p[qi] if f"c:{i}" in vec]
        cf = [vec[f"c:{i}"] for i in cand_f[qi] if f"c:{i}" in vec]
        if qv is None or not cp or not cf:
            continue
        raw_hit = float(E[qi] @ emb[cand_f[qi][0]]) > float(E[qi] @ emb[cand_p[qi][0]])
        rows.append({"gen": g, "arm": a, "k": k, "hit": float(max(np.array(cf) @ qv) > max(np.array(cp) @ qv)),
                     "raw_hit": float(raw_hit)})
    df = pd.DataFrame(rows)
    t = df.groupby(["arm", "gen"])[["hit", "raw_hit"]].mean().round(2)
    t["n"] = df.groupby(["arm", "gen"]).size()
    print(t.to_string())
    df.to_csv(out / "anticipation_normalised.csv", index=False)


if __name__ == "__main__":
    main()
