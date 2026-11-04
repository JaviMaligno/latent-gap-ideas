"""Revision (point 14): complete citing sets of the 150 continuation contexts, no cap and no date filter.

followups_fetch.py kept at most 50 citing papers per context (seeded sample) and only those dated >= 2024-01-01,
so the 'nearest later non-citing papers' pool of direction_tangent.py could still contain citing papers. This script
fetches every citing paper with an arXiv id from the Semantic Scholar public graph API (no key, ~1 request/s) and
caches one JSON per context under runs/followups/complete_citations/.

Usage: uv run python scripts/revision_citing_fetch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402
import followups_fetch as FF  # noqa: E402

CACHE = M.DATA / "runs/followups/complete_citations"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for j, c in enumerate(M.contexts()):
        p = CACHE / f"{c['arxiv_id'].replace('/', '_')}.json"
        if p.exists():
            continue
        cites = sorted(set(FF.citing_arxiv(c["arxiv_id"], since="")))
        p.write_text(json.dumps({"arxiv_id": c["arxiv_id"], "n_citing_arxiv_all_dates": len(cites),
                                 "citing": [{"arxiv_id": a, "date": d} for a, d in cites]}))
        print(j, c["arxiv_id"], len(cites), flush=True)


if __name__ == "__main__":
    main()
