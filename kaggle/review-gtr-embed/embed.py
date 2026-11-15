"""R4 stage 1 of docs/prereg/review_far_decoding.md: embed the non-adult PAST texts of Steam and arXiv with
gtr-t5-base exactly as vec2text does (AutoModel("sentence-transformers/gtr-t5-base").encoder, mean pooling of the last
hidden state over the attention mask, max_length=32, truncation=True, padding="max_length"), fp32 on both T4s (one
process per GPU, half the rows each). Saves float16 RAW (unnormalised) mean-pooled vectors per block of 4,096 texts
and prints one plain progress line per block (no tqdm). Output emb_gtr_{steam,arxiv}.npy, row-aligned with
texts_{dom}.parquet (= the non-adult past matrix in corpus order). Inputs from scripts/review_far_inversion.py prepare.
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np

OUT = "/kaggle/working"
BLOCK = 4096
BATCH = 512
DOMS = ("steam", "arxiv")

WORKER = r'''
import sys, time, os
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer
path, dom, gpu, ngpu, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
BLOCK, BATCH = 4096, 512
texts = pd.read_parquet(path)["text"].tolist()
blocks = list(range((len(texts) + BLOCK - 1) // BLOCK))[gpu::ngpu]
tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda").eval()

def mean_pool(h, m):  # vec2text.models.model_utils.mean_pool
    return (h * m[..., None]).sum(dim=1) / m.sum(dim=1)[:, None]

def embed(batch):
    x = tok(batch, return_tensors="pt", max_length=32, truncation=True, padding="max_length").to("cuda")
    with torch.no_grad():
        h = enc(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
    return mean_pool(h, x["attention_mask"]).float().cpu().numpy()

def run(batch, bs):
    out_ = []
    i = 0
    while i < len(batch):
        try:
            out_.append(embed(batch[i:i + bs]))
            i += bs
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs //= 2
            print(f"[{dom} gpu{gpu}] OOM, batch -> {bs}", flush=True)
    return np.concatenate(out_), bs

t0, bs = time.time(), BATCH
for n, b in enumerate(blocks):
    shard = f"{out}/{dom}_block{b:03d}.npy"
    if os.path.exists(shard):
        continue
    e, bs = run(texts[b * BLOCK:(b + 1) * BLOCK], bs)
    if not np.isfinite(e).all():
        raise SystemExit(f"non-finite embeddings in {dom} block {b}")
    np.save(shard, e.astype(np.float16))
    el = time.time() - t0
    print(f"[{dom} gpu{gpu}] block {n + 1}/{len(blocks)} (rows {b * BLOCK}-{min((b + 1) * BLOCK, len(texts))}) "
          f"{el / 60:.1f} min, eta {el / (n + 1) * (len(blocks) - n - 1) / 60:.1f} min", flush=True)
'''


def main() -> None:
    import torch
    ngpu = max(1, torch.cuda.device_count())
    print(f"gpus {ngpu}: {[torch.cuda.get_device_name(i) for i in range(ngpu)]}", flush=True)
    wpath = f"{OUT}/worker.py"
    open(wpath, "w").write(WORKER)
    t0 = time.time()
    for dom in DOMS:
        final = f"{OUT}/emb_gtr_{dom}.npy"
        if os.path.exists(final):
            continue
        path = glob.glob(f"/kaggle/input/**/texts_{dom}.parquet", recursive=True)[0]
        procs = [subprocess.Popen([sys.executable, wpath, path, dom, str(g), str(ngpu), OUT],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g in range(ngpu)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"{dom}: worker exit codes {codes}")
        import pandas as pd
        n = len(pd.read_parquet(path, columns=["row"]))
        shards = sorted(glob.glob(f"{OUT}/{dom}_block*.npy"))
        e = np.concatenate([np.load(s) for s in shards])
        assert len(e) == n, (len(e), n)
        np.save(final, e)
        for s in shards:
            os.remove(s)
        nr = np.linalg.norm(e.astype(np.float32), axis=1)
        print(f"{dom} done {e.shape}, raw norm median {np.median(nr):.4f} [{nr.min():.4f}, {nr.max():.4f}], "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    os.remove(wpath)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
