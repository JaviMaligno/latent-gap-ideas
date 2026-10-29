"""Embed a prepared corpus (texts.parquet: id, text) through the gateway, in resumable shards.

    uv run python scripts/embed_corpus.py --domain steam
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.gateway import Gateway, ledger_cost

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
BATCH = 128
SHARD = 4096
MAX_CHARS = 6000  # text-embedding-3-small accepts 8191 tokens; keep a wide margin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    root = DATA / "prepared" / args.domain
    texts = pd.read_parquet(root / "texts.parquet")["text"].str.slice(0, MAX_CHARS).tolist()
    shard_dir = root / "emb_shards"
    shard_dir.mkdir(exist_ok=True)
    ledger = DATA / "ledger.jsonl"
    gw = Gateway(ledger=ledger)

    def do_shard(s: int) -> int:
        out = shard_dir / f"{s:05d}.npy"
        if out.exists():
            return s
        chunk = texts[s * SHARD:(s + 1) * SHARD]
        vecs = [gw.embed(chunk[i:i + BATCH], tag=f"corpus-{args.domain}") for i in range(0, len(chunk), BATCH)]
        np.save(out, np.concatenate(vecs).astype(np.float16))
        return s

    n_shards = (len(texts) + SHARD - 1) // SHARD
    with ThreadPoolExecutor(args.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(do_shard, s) for s in range(n_shards)]), 1):
            f.result()
            print(f"shard {i}/{n_shards} done", flush=True)

    emb = np.concatenate([np.load(shard_dir / f"{s:05d}.npy") for s in range(n_shards)])
    assert len(emb) == len(texts)
    np.save(root / "emb_te3s.npy", emb)
    print("saved", emb.shape, "cost so far", ledger_cost(ledger))


if __name__ == "__main__":
    main()
