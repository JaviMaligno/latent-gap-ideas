"""arXiv cs.* domain: prepare the corpus from Cornell-University/arxiv (title + abstract, first-version date, past
sampled to Steam's size) and embed it with Qwen3-Embedding-0.6B and bge-m3 on both T4s. Embedding code is the Steam
kernel's (kaggle/embed-steam/embed.py); progress printed per block, no tqdm.
"""

import glob
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.0"], check=True)

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# --- corpus preparation (arXiv cs.*), mirrors docs/design.md Domains ---------------------------------------------
import json
import re

CUTOFF = "2023-01-01"
N_PAST, N_FUTURE = 72_000, 41_000  # Steam's sizes, so ceilings are comparable across domains at equal n
SEED = 20261024
ABSTRACT_CHARS = 1200  # same as src/latent_gap/arxiv.py


def item_text(title: str, abstract: str) -> str:
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def prepare() -> pd.DataFrame:
    src = glob.glob("/kaggle/input/**/arxiv-metadata-oai-snapshot.json", recursive=True)[0]
    rows = []
    with open(src) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            cats = r["categories"].split()
            if not any(c.startswith("cs.") for c in cats):
                continue
            rows.append((r["id"], _clean(r["title"]), _clean(r["abstract"]), r["versions"][0]["created"],
                         cats[0], " ".join(cats)))
            if i % 500_000 == 0:
                print(f"scanned {i} lines, {len(rows)} cs.*", flush=True)
    df = pd.DataFrame(rows, columns=["id", "name", "abstract", "date", "primary", "categories"])
    df["date"] = pd.to_datetime(df["date"], format="%a, %d %b %Y %H:%M:%S %Z", utc=True).dt.strftime("%Y-%m-%d")
    df = df[df["abstract"].str.len() >= 200].drop_duplicates("abstract")
    df["split"] = (df["date"] >= CUTOFF).map({False: "past", True: "future"})
    print(df["split"].value_counts().to_dict(), flush=True)
    past = df[df.split == "past"].sample(N_PAST, random_state=SEED)
    fut = df[df.split == "future"].sample(N_FUTURE, random_state=SEED)
    df = pd.concat([past, fut]).reset_index(drop=True)
    df["adult"] = False  # pipeline column (Steam filter); no such filter for arXiv
    df["text"] = [item_text(t, a) for t, a in zip(df["name"], df["abstract"])]
    return df

MODELS = [("qwen3e06", "Qwen/Qwen3-Embedding-0.6B"), ("bgem3", "BAAI/bge-m3")]
BLOCK = 8192
BATCH = 64
MAX_TOKENS = 512
OUT = "/kaggle/working"


def main() -> None:
    path = f"{OUT}/corpus.parquet"
    if not os.path.exists(path):
        prepare().to_parquet(path)
    models = MODELS
    texts = pd.read_parquet(path)["text"].tolist()
    order = np.argsort([len(t) for t in texts])[::-1]  # longest first: similar lengths per batch, less padding
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"{len(texts)} texts from {path}; devices {devices} ({torch.cuda.get_device_name(0)})", flush=True)

    for name, mid in models:
        final = f"{OUT}/emb_{name}.npy"
        if os.path.exists(final):
            continue
        t0 = time.time()
        m = SentenceTransformer(mid, device="cpu", model_kwargs={"torch_dtype": torch.float16})
        m.max_seq_length = MAX_TOKENS
        pool = m.start_multi_process_pool(devices)
        blocks = []
        n_blocks = (len(texts) + BLOCK - 1) // BLOCK
        for b in range(n_blocks):
            shard = f"{OUT}/{name}_block{b:03d}.npy"
            idx = order[b * BLOCK:(b + 1) * BLOCK]
            if os.path.exists(shard):
                blocks.append(np.load(shard))
                continue
            e = m.encode([texts[i] for i in idx], pool=pool, batch_size=BATCH, normalize_embeddings=True,
                         convert_to_numpy=True, chunk_size=512)
            e = e.astype(np.float16)
            np.save(shard, e)
            blocks.append(e)
            done = min((b + 1) * BLOCK, len(texts))
            el = time.time() - t0
            print(f"{name} {done}/{len(texts)} ({done / el:.0f}/s, {el / 60:.1f} min, "
                  f"eta {(len(texts) - done) / (done / el) / 60:.1f} min)", flush=True)
        m.stop_multi_process_pool(pool)
        e = np.concatenate(blocks)
        out = np.empty_like(e)
        out[order] = e
        np.save(final, out)
        for f in glob.glob(f"{OUT}/{name}_block*.npy"):
            os.remove(f)
        print(f"{name} done {out.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
