"""Post-hoc descriptive correction (2026-11-15): how many seed-and-rewrite targets exceed their corpus ceiling.

The registered summary of the seed-and-rewrite test run (docs/prereg/fluent_search.md, Rev 3 result; Table A5 of the
paper) counts targets whose best judge-accepted rewrite h_coh is >= the corpus q99 (the 99th percentile of corpus
cosines to the target). The paper had described that count as "closer than any existing item", which is the corpus
ceiling (the best visible corpus item). This script recomputes, read-only from the saved checkpoints of the test run,
the counts against the ceiling, exactly as scripts/seed_rewrite.py builds its table (max over the two rewriters per
target). No model is called and nothing under the run directory is written.

Output: docs/paper/seed_rewrite_ceiling_count.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/seed-rewrite/test"
REFS = DATA / "runs/fluent-search/refs.csv"
OUT = ROOT / "docs/paper/seed_rewrite_ceiling_count.json"


def main() -> None:
    rows = []
    for ck in sorted(RUN.glob("*/*/*.json")):
        rec = json.loads(ck.read_text())
        g = rec.get("gate", {})
        rows.append({"domain": rec["domain"], "writer": rec["config"]["writer"].split(":")[1], "arm": rec["arm"],
                     "k": rec["k"], "h_coh": g.get("h_coh")})
    D = pd.DataFrame(rows)
    D["h_coh"] = D.h_coh.astype(float)
    refs = pd.read_csv(REFS)
    refs["ceiling_star"] = np.maximum(refs.ceiling, refs.heldout_ceiling)
    comb = D.groupby(["domain", "arm", "k"], as_index=False).agg(h_coh=("h_coh", "max"))
    comb = comb.merge(refs, on=["domain", "arm", "k"])
    out = {"note": "post-hoc descriptive correction; read-only recount of runs/seed-rewrite/test (max over rewriters)",
           "source_run": str(RUN.relative_to(DATA)), "cells": {}}
    for (d, arm), h in comb.groupby(["domain", "arm"]):
        cov = h.h_coh.notna()
        diff = (h.h_coh - h.ceiling)
        out["cells"][f"{d}/{arm}"] = {
            "n_targets": int(len(h)),
            "coverage": int(cov.sum()),
            "median_ceiling": float(h.ceiling.median()),
            "median_corpus_q99": float(h.corpus_q99.median()),
            "median_h_coh": float(h.h_coh.median()),
            "n_ge_q99": int((h.h_coh >= h.corpus_q99).sum()),
            "n_gt_ceiling": int((h.h_coh > h.ceiling).sum()),
            "n_ge_ceiling_minus_0.02": int((h.h_coh >= h.ceiling - 0.02).sum()),
            "n_ge_ceiling_star_plus_0.05": int((h.h_coh >= h.ceiling_star + 0.05).sum()),
            "n_ge_ceiling_plus_0.05": int((h.h_coh >= h.ceiling + 0.05).sum()),
            "n_le_ceiling_star_plus_0.03": int((h.h_coh <= h.ceiling_star + 0.03).sum()),
            "n_le_ceiling_plus_0.03": int((h.h_coh <= h.ceiling + 0.03).sum()),
            "median_ceiling_star": float(h.ceiling_star.median()),
            "median_h_coh_minus_ceiling_star": float((h.h_coh - h.ceiling_star).median()),
            "max_h_coh": float(h.h_coh.max()),
            "median_h_coh_minus_ceiling": float(diff.median()),
            "min_h_coh_minus_ceiling": float(diff.min()),
            "max_h_coh_minus_ceiling": float(diff.max()),
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out["cells"].items():
        print(k, v)


if __name__ == "__main__":
    main()
