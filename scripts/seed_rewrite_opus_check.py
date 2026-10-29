"""Sensitivity: re-judge every accepted FAR/GAP test text with Opus (neither writer nor gate judge)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from latent_gap.coherent import judge
from latent_gap.domains import DOMAINS
T = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs/seed-rewrite/test"
out = T / "opus_check.jsonl"
done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
jobs = []
for f in sorted(T.glob("*/*/*.json")):
    g = json.loads(f.read_text())
    if g["arm"] in ("FAR", "GAP") and g["gate"].get("coh_text"):
        key = f"{g['domain']}/{f.parent.name}/{g['arm']}-{g['k']}"
        if key not in done:
            jobs.append((key, g["domain"], g["gate"]["coh_text"]))
print(len(jobs), "to judge", flush=True)
def run(j):
    key, d, text = j
    try:
        r = judge("claude:claude-opus-5-5", DOMAINS[d].noun, text)
    except Exception as e:
        r = {"error": repr(e)[:200]}
    return {"key": key, **r}
with ThreadPoolExecutor(3) as ex, out.open("a") as fh:
    for n, r in enumerate(ex.map(run, jobs), 1):
        fh.write(json.dumps(r) + "\n"); fh.flush()
        if n % 10 == 0: print(n, flush=True)
