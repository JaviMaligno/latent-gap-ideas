"""Stage 3 inputs of docs/prereg/strengthen_olmo_direction.md: the Kaggle dataset for one wave part.

  uv run python scripts/olmo_direction_kaggle.py build --wave 1 --part 1 --first 0 --last 100
      -> kaggle-upload/olmo-direction-inputs/{contexts.json, templates.json, run_info.json, dataset-metadata.json}
  then (printed by build): kaggle datasets create|version, kaggle kernels push -p kaggle/olmo-direction, status, output.

--first/--last index the eligible contexts in pos order (runs/olmo-direction/eligible.json); missed audit contexts
whose pos lies inside that range are appended with role "missed" (sensitivity analysis only). At most 140 contexts per
part (about 220 s per context on T4 x2, inside Kaggle's 12 h session). The prompt functions and templates of
kaggle/olmo-direction/run.py are checked against scripts/continue_from_context.py and scripts/iterative_critique.py.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402
import iterative_critique as IC  # noqa: E402
from latent_gap.arxiv import item_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "kaggle/olmo-direction"
RUN = M.DATA / "runs/olmo-direction"
UP = M.DATA / "kaggle-upload/olmo-direction-inputs"
DATASET = "javieraguilarmartin/latent-gap-olmo-direction-inputs"
KERNEL = "javieraguilarmartin/latent-gap-olmo-direction"
SEED, MAX_PART = 20261106, 140


def check_verbatim() -> None:
    src = (KDIR / "run.py").read_text()
    for fn, repl in ((M.prompt, {}), (IC.geo_feedback, {}), (IC.revise_prompt, {"M.prompt(": "prompt("})):
        s = inspect.getsource(fn)
        for a, b in repl.items():
            s = s.replace(a, b)
        assert s in src, f"{fn.__name__} in run.py is not a verbatim copy"
    assert f'NEUTRAL_FB = "{IC.NEUTRAL_FB}"' in src
    assert f"BAND = {IC.BAND}" in src


def build(wave: int, part: int, first: int, last: int) -> None:
    check_verbatim()
    el = json.loads((RUN / "eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    pos = el["eligible_pos"][first:last]
    assert pos and len(pos) == last - first, f"only {len(el['eligible_pos'])} eligible contexts labelled"
    lo, hi = min(pos), max(pos)
    missed = [p for p in sorted(recs) if recs[p]["missed"] and lo <= p <= hi]
    sel = [(p, "eligible") for p in pos] + [(p, "missed") for p in missed]
    assert len(sel) <= MAX_PART, f"{len(sel)} contexts > {MAX_PART} per part: split the wave"
    rows = {}
    for l in (RUN / "followups.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["n_followups"] >= 5:
            rows[len(rows)] = r
    import pandas as pd
    by_id = pd.read_parquet(M.DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"]).set_index("id")
    ctx = []
    for i, (p, role) in enumerate(sel):
        r = rows[p]
        assert r["arxiv_id"] == recs[p]["arxiv_id"]
        name, abstract = by_id.loc[r["arxiv_id"], "name"], by_id.loc[r["arxiv_id"], "abstract"]
        ctx.append({"ctx": i, "pos": p, "arxiv_id": r["arxiv_id"], "role": role,
                    "ctx_block": M.paper_block(name, abstract), "ctx_text": item_text(name, abstract),
                    "fups": [item_text(f["title"], f["abstract"]) for f in r["followups"]]})
    UP.mkdir(parents=True, exist_ok=True)
    (UP / "contexts.json").write_text(json.dumps(ctx))
    (UP / "templates.json").write_text(json.dumps({"system": M.SYSTEM, "review": IC.REVIEW, "band": list(IC.BAND)}))
    info = {"wave": wave, "part": part, "seed": SEED + 10 * wave + part, "eligible_first": first,
            "eligible_last": last, "positions": [p for p, _ in sel], "roles": [r for _, r in sel]}
    (UP / "run_info.json").write_text(json.dumps(info))
    (UP / "dataset-metadata.json").write_text(json.dumps({"title": "latent-gap olmo direction inputs", "id": DATASET,
                                                          "licenses": [{"name": "other"}]}))
    n_f = sum(len(c["fups"]) for c in ctx)
    print(f"wave {wave} part {part}: {len(pos)} eligible + {len(missed)} missed audit contexts, {n_f} follow-ups; "
          f"{len(ctx) * 39} OLMo calls (33 generations + 6 reviews each), ~{len(ctx) * 220 / 3600:.1f} h on T4 x2")
    out = M.DATA / f"kaggle-output/olmo-direction-w{wave}p{part}"
    print("next:\n"
          f"  kaggle datasets create -p {UP}      # first time; later: kaggle datasets version -p {UP} -m 'w{wave}p{part}'\n"
          f"  kaggle kernels push -p {KDIR}\n"
          f"  kaggle kernels status {KERNEL}        # check it is running (not queued/error) a few minutes after the push\n"
          f"  kaggle kernels output {KERNEL} -p {out}   # when complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "check":
        check_verbatim()
        print("run.py prompt functions are verbatim copies")
    else:
        build(a.wave, a.part, a.first, a.last)
