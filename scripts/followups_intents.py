"""Citation intents and influence for the follow-ups of the 150 study contexts (Semantic Scholar, free API).

For each context, fetch its citations with fields intents and isInfluential and attach them to the follow-ups we
already have (matched by arXiv id). Output: <run>/followup_intents.json  {"<ctx>/<j>": {"intents": [...], "influential": bool}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402
from followups_fetch import fetch  # noqa: E402

URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/citations?fields=externalIds,intents,isInfluential&limit=1000&offset={}"


def main() -> None:
    out = M.RUN / "followup_intents.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for i, c in enumerate(M.contexts()):
        if any(k.startswith(f"{i}/") for k in res):
            continue
        info, off = {}, 0
        while True:
            body = fetch(URL.format(c["arxiv_id"], off), 1.1)
            if not body:
                break
            d = json.loads(body)
            for x in d.get("data", []):
                ax = (x["citingPaper"].get("externalIds") or {}).get("ArXiv")
                if ax:
                    info[ax] = {"intents": x.get("intents") or [], "influential": bool(x.get("isInfluential"))}
            if "next" not in d:
                break
            off = d["next"]
        for j, f in enumerate(c["followups"]):
            res[f"{i}/{j}"] = info.get(f["arxiv_id"], {"intents": [], "influential": False, "missing": True})
        out.write_text(json.dumps(res))
        if i % 25 == 0:
            M.log(f"{i} contexts")
    vals = list(res.values())
    print(json.dumps({"followups": len(vals), "influential": sum(v["influential"] for v in vals),
                      "methodology": sum("methodology" in v["intents"] for v in vals),
                      "result": sum("result" in v["intents"] for v in vals),
                      "background_only": sum(v["intents"] == ["background"] for v in vals),
                      "missing": sum(bool(v.get("missing")) for v in vals)}))


if __name__ == "__main__":
    main()
