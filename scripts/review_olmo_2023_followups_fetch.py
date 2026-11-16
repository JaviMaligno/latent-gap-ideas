"""Fetch the 2023 citing papers of the 150 discovery contexts (docs/prereg/review_olmo_2023_followups.md).

For each context: all citing papers with an arXiv id (Semantic Scholar, scripts/followups_fetch.citing_arxiv with no
date filter); candidates whose new-style id starts with 2303..2401 go to the arXiv API, which gives title, abstract and
`published` (v1 date); kept if published is in [2023-04-01, 2023-12-31]; at most 50 per context (seeded as in
followups_fetch.py). Resumes from what is written. Also writes kaggle-upload/review-olmo-2023-texts/texts.parquet.

Usage: uv run python scripts/review_olmo_2023_followups_fetch.py [fetch|texts]
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import continue_from_context as M  # noqa: E402
from followups_fetch import ARXIV, ATOM, MAX_FUPS, citing_arxiv, fetch  # noqa: E402
from latent_gap.arxiv import item_text  # noqa: E402

DATA = M.DATA
RUN = DATA / "runs/review-olmo-2023"
UP = DATA / "kaggle-upload/review-olmo-2023-texts"
LO, HI = "2023-04-01", "2023-12-31"
SEED = 20261030


def arxiv_meta(ids: list[str]) -> dict[str, tuple[str, str, str]]:
    res = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        root = ET.fromstring(fetch(ARXIV.format(",".join(chunk), len(chunk)), 3.1))
        for e in root.findall(f"{ATOM}entry"):
            aid = re.sub(r"v\d+$", "", e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1])
            title = " ".join(e.findtext(f"{ATOM}title", "").split())
            summ = " ".join(e.findtext(f"{ATOM}summary", "").split())
            pub = e.findtext(f"{ATOM}published", "")[:10]
            if summ:
                res[aid] = (title, summ, pub)
    return res


def do_fetch() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "followups_2023.jsonl"
    done = {json.loads(l)["arxiv_id"] for l in path.read_text().splitlines()} if path.exists() else set()
    ctx = M.contexts()
    t0 = time.time()
    with path.open("a") as fh:
        for j, c in enumerate(ctx):
            if c["arxiv_id"] in done:
                continue
            allc = sorted(set(citing_arxiv(c["arxiv_id"], "")))
            cand = sorted({x for x, _ in allc if re.match(r"^(230[3-9]|231[0-2]|2401)\.\d{4,5}$", x)})
            meta = arxiv_meta(cand) if cand else {}
            inwin = [x for x in cand if x in meta and LO <= meta[x][2] <= HI]
            n_win = len(inwin)
            if len(inwin) > MAX_FUPS:
                sub = np.random.default_rng(SEED + j).choice(len(inwin), MAX_FUPS, replace=False)
                inwin = [inwin[k] for k in sorted(sub)]
            fups = [{"arxiv_id": x, "published": meta[x][2], "title": meta[x][0], "abstract": meta[x][1]} for x in inwin]
            fh.write(json.dumps({"ctx": j, "arxiv_id": c["arxiv_id"], "n_citing_arxiv_any": len(allc),
                                 "n_candidates": len(cand), "n_in_window": n_win, "n_followups": len(fups),
                                 "followups": fups}) + "\n")
            fh.flush()
            print(f"{time.strftime('%H:%M:%S')} ctx {j + 1}/150 citing {len(allc)} cand {len(cand)} "
                  f"in-window {n_win} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    n = np.array([r["n_followups"] for r in rows])
    print(json.dumps({"contexts": len(rows), "with_ge1": int((n >= 1).sum()), "with_ge5": int((n >= 5).sum()),
                      "total": int(n.sum())}))


def do_texts() -> None:
    rows = sorted((json.loads(l) for l in (RUN / "followups_2023.jsonl").read_text().splitlines()), key=lambda r: r["ctx"])
    recs = [{"kind": "fup2023", "key": f"{r['ctx']}/{k}", "text": item_text(f["title"], f["abstract"])}
            for r in rows for k, f in enumerate(r["followups"])]
    UP.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(UP / "texts.parquet", index=False)
    (UP / "dataset-metadata.json").write_text(json.dumps(
        {"title": "latent-gap review-olmo-2023 texts", "id": "javieraguilarmartin/latent-gap-review-olmo-2023-texts",
         "licenses": [{"name": "other"}]}, indent=1))
    print(len(recs), "texts")


if __name__ == "__main__":
    {"fetch": do_fetch, "texts": do_texts}[sys.argv[1] if len(sys.argv) > 1 else "fetch"]()
