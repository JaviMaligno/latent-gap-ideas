"""Compare the 30-pair abstract labels (docs/paper/followup_check_sample.json) with Semantic Scholar's flags.

"S2 continuing" is the definition of scripts/continuation_reanalysis.py: the citation is marked influential, or its
intents include methodology or result. Output: docs/paper/followup_check_s2.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper/followup_check_s2.json"


def s2_continuing(v: dict) -> bool:
    return bool(v) and bool(v.get("influential") or {"methodology", "result"} & set(v.get("intents", [])))


def main():
    sample = json.loads((ROOT / "docs/paper/followup_check_sample.json").read_text())["pairs"]
    intents = json.loads((M.RUN / "followup_intents.json").read_text())
    rows = []
    for p in sample:
        v = intents.get(f"{p['ctx_index']}/{p['followup_index']}", {})
        rows.append({"n": p["n"], "label": p["label"], "s2_continuing": s2_continuing(v),
                     "s2_intents": v.get("intents", []), "s2_influential": v.get("influential")})
    cont = {"CONTINUES", "USES"}
    out = {
        "n_pairs": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "s2_continuing": sum(r["s2_continuing"] for r in rows),
        "label_continuing_and_s2_continuing": sum(r["s2_continuing"] and r["label"] in cont for r in rows),
        "label_continuing": sum(r["label"] in cont for r in rows),
        "s2_continuing_by_label": dict(Counter(r["label"] for r in rows if r["s2_continuing"])),
        "s2_all_followups_continuing_share": sum(s2_continuing(v) for v in intents.values()) / len(intents),
        "pairs": rows,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
