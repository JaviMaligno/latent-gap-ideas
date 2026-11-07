"""Texts and local BGE-M3 vectors for docs/prereg/strengthen_bge_replication.md. Never calls an LLM.

  uv run python scripts/strengthen_bge_embed.py texts        # project env: writes runs/strengthen-bge/texts.json
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 uv run --no-project --with sentence-transformers==3.0.1 \
      --with transformers==4.44.2 --with "numpy<2" python scripts/strengthen_bge_embed.py embed   # -> emb_bgem3.npy

Same settings as kaggle/embed-continuation/embed.py (sentence-transformers, max_seq_length 512, normalised); here
float32 on Apple MPS (torch 2.2 in this env; transformers 4.44 because the cached weights are a .bin file).
Groups, in order: anchoring (720 decoded texts), modern/ctx, modern/fup, modern/partB, modern/iter, gpt4o/iter
(ORACLE and ORACLE-H rows of runs/continue-from-context/iter_ideas.jsonl).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = DATA / "runs/strengthen-bge"
ABSTRACT_CHARS = None  # taken from latent_gap.arxiv in the texts step


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def texts() -> None:
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from latent_gap.arxiv import item_text
    rows = []  # (group, key, text)
    for domain in ("steam", "arxiv"):
        for gen in ("luna", "sonnet"):
            for cond in ("true", "swap", "none"):
                for arm in ("REAL", "GAP", "FAR"):
                    for r in read(DATA / "runs" / domain / "anchoring-causal" / gen / cond / arm / "items.jsonl"):
                        if r.get("best"):
                            rows.append(("anchoring", f"{domain}/{gen}/{cond}/{arm}/{r['k']}", r["best"]["text"]))
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract", "adult"])
    meta = meta[~meta["adult"]].set_index("id")
    fups = [r for r in read(DATA / "runs/followups-2025/followups.jsonl") if r["n_followups"] >= 5][:150]
    assert len(fups) == 150
    for i, c in enumerate(fups):
        rows.append(("modern/ctx", str(i), item_text(meta.loc[c["arxiv_id"], "name"], meta.loc[c["arxiv_id"], "abstract"])))
    for i, c in enumerate(fups):
        for j, f in enumerate(c["followups"]):
            rows.append(("modern/fup", f"{i}/{j}", item_text(f["title"], f["abstract"])))
    for r in read(DATA / "runs/continue-modern/ideas.jsonl"):
        if r["abstract"]:
            rows.append(("modern/partB", f"{r['ctx']}/{r['arm']}/{r['k']}", item_text(r["title"], r["abstract"])))
    for r in read(DATA / "runs/continue-modern/iter_ideas.jsonl"):
        if r["abstract"]:
            rows.append(("modern/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}", item_text(r["title"], r["abstract"])))
    for r in read(DATA / "runs/continue-from-context/iter_ideas.jsonl"):
        if r["abstract"] and r["arm"] in ("ORACLE", "ORACLE-H"):
            rows.append(("gpt4o/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}", item_text(r["title"], r["abstract"])))
    keys = [(g, k) for g, k, _ in rows]
    assert len(set(keys)) == len(keys), "duplicate keys"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "texts.json").write_text(json.dumps([{"group": g, "key": k, "text": t} for g, k, t in rows]))
    from collections import Counter
    print(Counter(g for g, _, _ in rows), len(rows), flush=True)


def embed() -> None:
    import os
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rows = json.loads((OUT / "texts.json").read_text())
    T = [r["text"] for r in rows]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer("BAAI/bge-m3", device=device)
    m.max_seq_length = 512
    order = sorted(range(len(T)), key=lambda i: -len(T[i]))  # similar lengths per batch
    V = np.zeros((len(T), 1024), dtype=np.float32)
    t0, B = time.time(), 32
    for b in range(0, len(order), B):
        idx = order[b:b + B]
        V[idx] = m.encode([T[i] for i in idx], batch_size=4, normalize_embeddings=True, convert_to_numpy=True)
        if device == "mps":
            torch.mps.empty_cache()  # 16 GB Mac: batch 16 at 512 tokens ran out of MPS memory
        if (b // B) % 20 == 0 or b + B >= len(order):
            print(f"bge {min(b + B, len(T))}/{len(T)} {time.time() - t0:.0f}s ({device})", flush=True)
    np.save(OUT / "emb_bgem3.npy", V)
    sha = hashlib.sha256((OUT / "texts.json").read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps({"texts_sha256": sha, "n": len(T), "device": device,
                                                   "model": "BAAI/bge-m3", "max_seq_length": 512,
                                                   "sentence_transformers": __import__("sentence_transformers").__version__,
                                                   "torch": torch.__version__}, indent=1))
    print("done", V.shape, flush=True)


if __name__ == "__main__":
    {"texts": texts, "embed": embed}[sys.argv[1]]()
