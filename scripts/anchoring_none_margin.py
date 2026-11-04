"""Derived from the causal anchoring run: with NO neighbours shown (score feedback only), how much closer does the
decoded text end up to its own target than to the donor target of the same arm? (means of e·t − e·u, condition none)
Input: experiment-data runs/anchoring-causal-analysis/results.json. Output: docs/prereg/anchoring_none_margin.json
"""
import json
from pathlib import Path

SRC = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/anchoring-causal-analysis/results.json"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/anchoring_none_margin.json"
cells = json.loads(SRC.read_text())["cells"]
res = {f"{c['domain']}/{c['gen']}/{c['arm']}": {k: c["quantitative"]["none"]["means"][k] for k in ("target", "donor")} for c in cells}
for v in res.values():
    v["own_minus_other"] = v["target"] - v["donor"]
m = [v["own_minus_other"] for v in res.values()]
res["range_all"] = [min(m), max(m)]
res["range_REAL_GAP"] = [min(v["own_minus_other"] for k, v in res.items() if k.endswith(("REAL", "GAP"))),
                         max(v["own_minus_other"] for k, v in res.items() if k.endswith(("REAL", "GAP")))]
OUT.write_text(json.dumps(res, indent=1))
print(res["range_all"], res["range_REAL_GAP"])
