"""Embed a prepared corpus (any texts.parquet in the kernel inputs) with Qwen3-Embedding-0.6B (primary space) and
bge-m3 (robustness space), on both T4s, printing one progress line per block and saving each block as it lands.

Kernel conventions for this repo: always use every GPU (multi-process pool) and always print progress lines
(plain prints, not tqdm: Kaggle's log does not render carriage returns).
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

MODELS = [("qwen3e06", "Qwen/Qwen3-Embedding-0.6B"), ("bgem3", "BAAI/bge-m3")]
BLOCK = 8192
BATCH = 64
MAX_TOKENS = 512
OUT = "/kaggle/working"


def main() -> None:
    path = glob.glob("/kaggle/input/**/texts.parquet", recursive=True)[0]
    sel = os.path.join(os.path.dirname(path), "models.txt")  # optional: subset of MODELS names, one per line
    models = MODELS
    if os.path.exists(sel):
        keep = open(sel).read().split()
        models = [m for m in MODELS if m[0] in keep]
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
