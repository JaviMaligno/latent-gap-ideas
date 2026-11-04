"""Fetch real follow-ups (citing arXiv papers) of past arXiv papers. Free APIs, no keys.

Citing papers come from Semantic Scholar (graph API, citations of arXiv:<id>), kept if they have an arXiv id and
were published on or after 2024-01-01 (after the gpt-4o-0806 training cutoff); their titles and abstracts come
from the arXiv API. At most MAX_FUPS follow-ups per context (seeded sample). Writes one JSON line per context and
resumes from what is already written.

Usage: uv run python scripts/followups_fetch.py <n_contexts> <out_dir> [--seed S]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
S2 = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/citations?fields=externalIds,publicationDate&limit=1000&offset={}"
ARXIV = "https://export.arxiv.org/api/query?id_list={}&max_results={}"
SINCE = "2024-01-01"
MAX_FUPS = 50
ATOM = "{http://www.w3.org/2005/Atom}"


def fetch(url: str, delay: float) -> bytes:
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            time.sleep(delay)
            return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            time.sleep(min(60, 3 * 2 ** attempt))
        except Exception:
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"giving up on {url}")


def citing_arxiv(aid: str, since: str = SINCE) -> list[tuple[str, str]]:
    out, off = [], 0
    while True:
        body = fetch(S2.format(aid, off), 1.1)
        if not body:
            return out
        d = json.loads(body)
        for c in d.get("data", []):
            p = c["citingPaper"]
            ax = (p.get("externalIds") or {}).get("ArXiv")
            if ax and (p.get("publicationDate") or "") >= since:
                out.append((ax, p["publicationDate"]))
        if "next" not in d:
            return out
        off = d["next"]


def abstracts(ids: list[str]) -> dict[str, tuple[str, str]]:
    res = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        root = ET.fromstring(fetch(ARXIV.format(",".join(chunk), len(chunk)), 3.1))
        for e in root.findall(f"{ATOM}entry"):
            aid = re.sub(r"v\d+$", "", e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1])
            title = " ".join(e.findtext(f"{ATOM}title", "").split())
            summ = " ".join(e.findtext(f"{ATOM}summary", "").split())
            if summ:
                res[aid] = (title, summ)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--from-date", default="2021-01-01", help="contexts dated on or after (YYYY-MM-DD)")
    ap.add_argument("--to-date", default="2022-12-31", help="contexts dated on or before")
    ap.add_argument("--since", default=SINCE, help="follow-ups published on or after")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "followups.jsonl"
    done = {json.loads(l)["arxiv_id"] for l in path.read_text().splitlines()} if path.exists() else set()
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "date", "split", "adult"])
    pool = meta[~meta.adult & (meta.date >= a.from_date) & (meta.date <= a.to_date)].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    ctx = pool.iloc[rng.choice(len(pool), a.n, replace=False)]
    with path.open("a") as fh:
        for j, (_, c) in enumerate(ctx.iterrows()):
            if c["id"] in done:
                continue
            cites = sorted(set(citing_arxiv(c["id"], a.since)))
            if len(cites) > MAX_FUPS:
                sub = np.random.default_rng(a.seed + j).choice(len(cites), MAX_FUPS, replace=False)
                cites = [cites[k] for k in sorted(sub)]
            ab = abstracts([x for x, _ in cites]) if cites else {}
            fups = [{"arxiv_id": x, "date": d, "title": ab[x][0], "abstract": ab[x][1]} for x, d in cites if x in ab]
            fh.write(json.dumps({"arxiv_id": c["id"], "title": c["name"], "date": c["date"],
                                 "n_citing_arxiv": len(cites), "n_followups": len(fups), "followups": fups}) + "\n")
            fh.flush()
            if j % 25 == 0:
                print(time.strftime("%H:%M:%S"), j, "contexts", flush=True)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    n = np.array([r["n_followups"] for r in rows])
    print(json.dumps({"contexts": len(rows), "with_ge5": int((n >= 5).sum()), "with_ge10": int((n >= 10).sum()),
                      "median_followups": float(np.median(n)), "total_followups": int(n.sum())}))


if __name__ == "__main__":
    main()
