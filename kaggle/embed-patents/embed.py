"""Patents domain: prepare the corpus from HUPD (Harvard USPTO patent applications, title + abstract, filing date;
past and future sampled to Steam's sizes) and embed it with Qwen3-Embedding-0.6B and bge-m3 on both T4s. Embedding code is the Steam
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


# --- corpus preparation (HUPD patent applications), mirrors docs/design.md Domains -------------------------------
import json
import re
import tarfile

import requests

HF = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{year}.tar.gz"
# HUPD covers filings 2004-2018 (2018 partial). Past = filed 2012-2015, future = 2016-2017; uniform within each year.
PER_YEAR = {2012: 18_000, 2013: 18_000, 2014: 18_000, 2015: 18_000, 2016: 20_500, 2017: 20_500}
SEED = 20261024
ABSTRACT_CHARS = 1200  # same as src/latent_gap/patents.py


def item_text(title: str, abstract: str) -> str:
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _year(year: int, k: int) -> list:
    """Reservoir sample of k applications filed in `year` with an abstract of >= 200 characters, streamed (no disk)."""
    rng = np.random.default_rng([SEED, year])
    keep, seen, t0 = [], 0, time.time()
    with requests.get(HF.format(year=year), stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for i, m in enumerate(tar):
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                j = json.loads(tar.extractfile(m).read())
                abstract, title = _clean(j.get("abstract")), _clean(j.get("title"))
                if len(abstract) < 200 or not str(j.get("filing_date", "")).startswith(str(year)):
                    continue
                row = (j["application_number"], title.capitalize(), abstract, j["filing_date"], j.get("decision", ""),
                       j.get("main_cpc_label", ""))
                if seen < k:
                    keep.append(row)
                else:
                    s = int(rng.integers(seen + 1))
                    if s < k:
                        keep[s] = row
                seen += 1
                if seen % 50_000 == 0:
                    print(f"  {year}: {seen} eligible ({time.time() - t0:.0f}s)", flush=True)
    print(f"{year}: kept {len(keep)} of {seen} eligible in {(time.time() - t0) / 60:.1f} min", flush=True)
    return keep


def prepare() -> pd.DataFrame:
    rows = [r for y, k in PER_YEAR.items() for r in _year(y, k)]
    df = pd.DataFrame(rows, columns=["id", "name", "abstract", "date", "decision", "cpc"]).drop_duplicates("abstract")
    df["split"] = (df["date"].str[:4].astype(int) >= 2016).map({False: "past", True: "future"})
    print(df["split"].value_counts().to_dict(), df["decision"].value_counts().to_dict(), flush=True)
    df["adult"] = False  # pipeline column (Steam filter); no such filter for patents
    df["text"] = [item_text(t, a) for t, a in zip(df["name"], df["abstract"])]
    return df.reset_index(drop=True)

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
