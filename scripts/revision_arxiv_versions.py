"""Revision audit (point 3): first arXiv appearance (v1 'published') and version count of every context and
follow-up id of the continuation studies (Section 5 / 5.1 and Section 5.4).

Queries the public arXiv API (export.arxiv.org/api/query, id_list batches of <= 100, >= 3 s between calls) and caches
every raw Atom response under <run dir>/arxiv_audit/. Rerunning reuses the cache. Ids missing from a batch response
are retried one by one. Writes <run dir>/arxiv_audit/versions.json: {id: {"published": ..., "updated": ...,
"n_versions": k}} (n_versions = version number of the latest entry the API returns).

Usage: uv run python scripts/revision_arxiv_versions.py [followups followups-2025]
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
API = "https://export.arxiv.org/api/query?id_list={}&max_results={}"
ATOM = "{http://www.w3.org/2005/Atom}"
N_CTX, MIN_FUPS, BATCH, DELAY = 150, 5, 100, 3.2
_last = [0.0]


def used(run: str) -> tuple[list[str], list[str]]:
    """Context and follow-up ids exactly as continue_from_context.contexts() selects them (file order, >= 5)."""
    rows = [json.loads(l) for l in (DATA / "runs" / run / "followups.jsonl").read_text().splitlines()]
    sel = [r for r in rows if r["n_followups"] >= MIN_FUPS][:N_CTX]
    assert len(sel) == N_CTX
    return [r["arxiv_id"] for r in sel], [f["arxiv_id"] for r in sel for f in r["followups"]]


def get(url: str) -> bytes:
    for attempt in range(8):
        wait = DELAY - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                body = r.read()
            _last[0] = time.time()
            return body
        except Exception as e:  # noqa: BLE001
            _last[0] = time.time()
            print("retry", attempt, type(e).__name__, flush=True)
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"giving up on {url[:120]}")


def parse(body: bytes) -> dict[str, dict]:
    out = {}
    for e in ET.fromstring(body).findall(f"{ATOM}entry"):
        full = e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1]
        m = re.match(r"(.+?)v(\d+)$", full)
        if not m or not e.findtext(f"{ATOM}published"):
            continue  # error entries have no version / published
        out[m.group(1)] = {"published": e.findtext(f"{ATOM}published"), "updated": e.findtext(f"{ATOM}updated"),
                           "n_versions": int(m.group(2))}
    return out


def fetch_batch(ids: list[str], cache: Path) -> dict[str, dict]:
    if not cache.exists():
        cache.write_bytes(get(API.format(",".join(ids), len(ids))))
    return parse(cache.read_bytes())


def audit(run: str) -> None:
    ctx, fup = used(run)
    ids = sorted(set(ctx) | set(fup))
    d = DATA / "runs" / run / "arxiv_audit"
    d.mkdir(exist_ok=True)
    res: dict[str, dict] = {}
    for b in range(0, len(ids), BATCH):
        res.update(fetch_batch(ids[b:b + BATCH], d / f"batch_{b // BATCH:03d}.xml"))
        print(time.strftime("%H:%M:%S"), run, f"batch {b // BATCH}", len(res), "/", len(ids), flush=True)
    missing = [i for i in ids if i not in res]
    for i in missing:
        res.update(fetch_batch([i], d / f"single_{i.replace('/', '_')}.xml"))
    still = [i for i in ids if i not in res]
    (d / "versions.json").write_text(json.dumps({"ids": res, "missing": still}, indent=0))
    print(run, "ids", len(ids), "resolved", len(res), "missing", still, flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:] or ["followups", "followups-2025"]:
        audit(r)
