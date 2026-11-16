"""review_leakfree_encoder (docs/prereg/review_leakfree_encoder.md), mpnet space: embed Steam and arXiv with
sentence-transformers/all-mpnet-base-v2 (2021) on both T4s (multi-process pool, one progress line per block, no tqdm),
then Section 4 and the gap-score validity (raw, K = 256) on CPU. Patents are not embedded: the model's training data
postdates the 2016 patents split.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.0"], check=True)

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

W = "/kaggle/working"
MID = "sentence-transformers/all-mpnet-base-v2"
BLOCK, BATCH = 8192, 128
src = os.path.dirname(glob.glob("/kaggle/input/**/review_leakfree_encoder.py", recursive=True)[0])
os.makedirs(f"{W}/repo/scripts", exist_ok=True)
os.makedirs(f"{W}/repo/docs/prereg", exist_ok=True)
for f in ("where_future_lands.py", "review_hole_validity.py", "revision_future_cells.py", "review_leakfree_encoder.py"):
    shutil.copy(f"{src}/{f}", f"{W}/repo/scripts/{f}")
shutil.copy(f"{src}/where_future_lands_result.json", f"{W}/repo/docs/prereg/")


def embed(d: str) -> None:
    out = f"{W}/data/prepared/{d}/emb_mpnet.npy"
    if os.path.exists(out):
        return
    texts = pd.read_parquet(f"{W}/data/prepared/{d}/corpus.parquet", columns=["text"])["text"].tolist()
    order = np.argsort([len(t) for t in texts])[::-1]
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    gpu = bool(devices)  # CPU fallback (fp32, single process) when no GPU slot is available
    m = SentenceTransformer(MID, device="cpu", model_kwargs={"torch_dtype": torch.float16} if gpu else {})
    print(f"{d}: {len(texts)} texts; devices {devices or ['cpu']}; max_seq_length {m.max_seq_length}", flush=True)
    pool = m.start_multi_process_pool(devices) if gpu else None
    t0, blocks = time.time(), []
    for b in range(0, len(texts), BLOCK):
        idx = order[b:b + BLOCK]
        kw = dict(pool=pool, chunk_size=512) if gpu else {}
        e = m.encode([texts[i] for i in idx], batch_size=BATCH if gpu else 32, normalize_embeddings=True,
                     convert_to_numpy=True, **kw).astype(np.float16)
        blocks.append(e)
        done = min(b + BLOCK, len(texts))
        el = time.time() - t0
        print(f"{d} {done}/{len(texts)} ({done / el:.0f}/s, {el / 60:.1f} min)", flush=True)
    if gpu:
        m.stop_multi_process_pool(pool)
    e = np.concatenate(blocks)
    res = np.empty_like(e)
    res[order] = e
    np.save(out, res)
    print(f"{d}: saved {res.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)
    del m
    torch.cuda.empty_cache()



def main() -> None:
    for d in ("steam", "arxiv"):
        os.makedirs(f"{W}/data/prepared/{d}", exist_ok=True)
        if not os.path.exists(f"{W}/data/prepared/{d}/corpus.parquet"):
            os.symlink(f"{src}/{d}_corpus.parquet", f"{W}/data/prepared/{d}/corpus.parquet")
        embed(d)

    env = dict(os.environ, LF_DATA=f"{W}/data", LF_OUT=f"{W}/out", LF_WORKERS=str(os.cpu_count() or 4),
               OMP_NUM_THREADS=str(os.cpu_count() or 4), PYTHONUNBUFFERED="1")
    t0 = time.time()
    args = ["analyse", "mpnet", "steam", "arxiv"]
    print(time.strftime("%H:%M:%S"), "running", args, f"cpus {os.cpu_count()}", flush=True)
    subprocess.run([sys.executable, f"{W}/repo/scripts/review_leakfree_encoder.py", *args], env=env, check=True)
    print(time.strftime("%H:%M:%S"), "finished", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":  # the spawn pool re-imports this module
    main()
