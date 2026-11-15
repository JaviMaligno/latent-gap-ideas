"""Review follow-up: overlap between the gpt-4o replication and the OLMo-direction contexts
(docs/prereg/review_sample_overlap.md).

  OMP_NUM_THREADS=4 uv run python scripts/review_sample_overlap.py            # overlap + reruns -> JSON
  OMP_NUM_THREADS=4 uv run python scripts/review_sample_overlap.py --overlap-only --out /tmp/x.json

1. Overlap, computed on arXiv ids: the 150 replication contexts (runs/critique-replication/followups.jsonl), the
   eligible contexts the OLMo run actually generated (kaggle-output/olmo-direction-w*p*/run_info.json positions with
   role "eligible", mapped to arXiv ids through eligible.json["contexts"]), and the original discovery sample
   (continue_from_context.contexts()).
2. If replication ∩ OLMo is non-empty: the registered confirmatory contrasts of both studies on the non-overlapping
   contexts only, with the original code and seeds:
   - strengthen_critique_replication.analyse and .pooled (E1–E6, R1/R2, pooled secondary), both embedders. The
     overlapping contexts are removed by masking their rows of the per-context matrix M (and the follow-ups' LOO
     value) to NaN right after neutral_contrasts.per_context; every downstream estimator in those functions drops
     NaN rows, so this equals running on the subset. The original output path is redirected to a temp file.
   - olmo_direction_analysis.final (three primary contrasts along s, plus everything final() writes), both
     embedders, with collect() restricted to the non-overlapping positions; output redirected to a temp file.
   Published values are read from the committed JSONs and reported next to the subset values.
Output: docs/prereg/review_sample_overlap.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = ROOT / "docs/prereg/review_sample_overlap.json"
SPACES = ("qwen3e06", "bgem3")


def ids_replication() -> list[str]:
    return [json.loads(l)["arxiv_id"] for l in (DATA / "runs/critique-replication/followups.jsonl").read_text().splitlines()]


def ids_olmo() -> list[str]:
    el = json.loads((DATA / "runs/olmo-direction/eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    out = []
    for d in sorted((DATA / "kaggle-output").glob("olmo-direction-w*p*")):
        info = json.loads((d / "run_info.json").read_text())
        out += [recs[p]["arxiv_id"] for p, role in zip(info["positions"], info["roles"]) if role == "eligible"]
    return out


def ids_discovery() -> list[str]:
    import continue_from_context as M
    return [c["arxiv_id"] for c in M.contexts()]


def overlap() -> dict:
    rep, olmo, disc = ids_replication(), ids_olmo(), ids_discovery()
    R, O, D = set(rep), set(olmo), set(disc)
    assert len(R) == len(rep) and len(O) == len(olmo) and len(D) == len(disc)
    return {"n": {"replication": len(rep), "olmo_eligible_generated": len(olmo), "discovery": len(disc)},
            "replication_and_olmo": {"count": len(R & O), "ids": sorted(R & O)},
            "replication_and_discovery": {"count": len(R & D), "ids": sorted(R & D)},
            "olmo_and_discovery": {"count": len(O & D), "ids": sorted(O & D)},
            "recorded_in_strengthen_critique_replication_json": json.loads(
                (ROOT / "docs/prereg/strengthen_critique_replication.json").read_text()).get(
                "overlap_with_olmo_direction_eligible"),
            "note": "the recorded value was computed as set(json.loads(eligible.json)), i.e. over the dict's keys"}


def rerun_replication(drop: set[str], tmp: Path) -> dict:
    import neutral_contrasts as N
    import strengthen_critique_replication as SCR

    orig = N.per_context

    def masked(ctx, *a, **k):
        M, loo = orig(ctx, *a, **k)
        bad = np.array([c["arxiv_id"] in drop for c in ctx])
        if bad.any():
            for arm in M:
                for m in M[arm]:
                    M[arm][m][bad] = np.nan
            loo = loo.copy()
            loo[bad] = np.nan
        return M, loo

    N.per_context = masked
    SCR.OUT = tmp
    try:
        SCR.analyse()
        SCR.pooled()
    finally:
        N.per_context = orig
    return json.loads(tmp.read_text())


def rerun_olmo(drop: set[str], tmp: Path) -> dict:
    import olmo_direction_analysis as OD

    el = json.loads((OD.RUN / "eligible.json").read_text())
    aid = {r["pos"]: r["arxiv_id"] for r in el["contexts"]}
    orig_collect, orig_load = OD.collect, OD.load_outputs

    def load_filtered():
        return {p: v for p, v in orig_load().items() if aid[p] not in drop}

    OD.load_outputs = load_filtered
    OD.OUT = tmp
    try:
        OD.final()
    finally:
        OD.load_outputs, OD.collect = orig_load, orig_collect
    return json.loads(tmp.read_text())


def rep_rows(res: dict) -> dict:
    out = {}
    for s in SPACES:
        conf = res[s]["confirmatory"]
        out[s] = {}
        for k, v in conf.items():
            e = {"mean": v.get("mean", v.get("diff_mean")), "ci_boot": v["ci"], "n": v["n"], "p_holm": v["p_holm"]}
            if k.startswith("E"):
                e |= {"t_ci95": v["t_ci95"], "p_tost": v["p_tost"], "reading": v["reading"]}
            else:
                e |= {"p": v["p"], "reading": "detected (Holm)" if v["p_holm"] < 0.05 else "not detected"}
            out[s][k] = e
        pw = res[s]["secondary"].get("pooled_with_original_150")
        if pw:
            out[s]["pooled_with_original_150"] = {k: {"mean": v["mean"], "ci": v["ci"], "p": v["p"], "n": v["n"],
                                                      "p_tost": v["p_tost"]}
                                                  for k, v in pw.items() if isinstance(v, dict)}
    return out


def olmo_rows(res: dict) -> dict:
    return {s: {a: {k: res[f"olmo/{s}"]["contrasts_along_s"][a][k] for k in ("n", "mean", "ci", "p", "p_holm", "reading")}
                for a in ("GEO", "LLM", "ORACLE")} for s in SPACES}


def compare(pub: dict, sub: dict) -> list[dict]:
    ch = []
    for s in pub:
        for k in pub[s]:
            if "reading" in pub[s][k] and pub[s][k]["reading"] != sub[s][k]["reading"]:
                ch.append({"space": s, "contrast": k, "published": pub[s][k]["reading"], "non_overlapping": sub[s][k]["reading"]})
    return ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    res = {"prereg": "docs/prereg/review_sample_overlap.md", "overlap": overlap()}
    print(json.dumps({k: v["count"] for k, v in res["overlap"].items() if isinstance(v, dict) and "count" in v}), flush=True)
    drop = set(res["overlap"]["replication_and_olmo"]["ids"])
    if drop and not a.overlap_only:
        with tempfile.TemporaryDirectory() as td:
            rep = rerun_replication(drop, Path(td) / "rep.json")
            olmo = rerun_olmo(drop, Path(td) / "olmo.json")
        pub_rep = rep_rows(json.loads((ROOT / "docs/prereg/strengthen_critique_replication.json").read_text()))
        pub_olmo = olmo_rows(json.loads((ROOT / "docs/prereg/strengthen_olmo_direction.json").read_text()))
        sub_rep, sub_olmo = rep_rows(rep), olmo_rows(olmo)
        res["replication"] = {"n_contexts_non_overlapping": len(ids_replication()) - len(drop),
                              "published": pub_rep, "non_overlapping": sub_rep}
        res["olmo"] = {"n_contexts_non_overlapping": olmo["olmo/qwen3e06"]["n_contexts"],
                       "published": pub_olmo, "non_overlapping": sub_olmo}
        res["reading_changes"] = {"replication": compare(pub_rep, sub_rep), "olmo": compare(pub_olmo, sub_olmo)}
        res["any_reading_changes"] = bool(res["reading_changes"]["replication"] or res["reading_changes"]["olmo"])
        res["method_note"] = ("replication: overlapping contexts masked to NaN in neutral_contrasts.per_context output "
                              "(all downstream estimators drop NaN rows); OLMo: overlapping positions removed from "
                              "load_outputs before final(). Same seeds; RNG draws differ from the published run only "
                              "through the smaller samples.")
        print(json.dumps(res["reading_changes"], indent=1), flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
