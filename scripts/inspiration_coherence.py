"""Coherence judge for docs/prereg/inspiration_from_anything.md (secondary measure).

Stratified sample (seed 20261104): 10 ideas per source x quintile x framing (900) + 20 NONE = 920. The judge sees
only title, description, genres and tags, shuffled, never the arm, source or stimulus. 40 ideas per call.
    uv run python scripts/inspiration_coherence.py codex           # all 920 with Codex (subscription)
    uv run python scripts/inspiration_coherence.py claude          # the first 100 of the shuffled sample (agreement)
Output: runs/inspiration/gpt-4o-0806/coherence_<judge>.jsonl  ({"row": scored-row index, "score": 1-5})
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/inspiration/gpt-4o-0806"
SEED, BATCH = 20261104, 40
PROMPT = ("You are reviewing short video game concepts. For each concept below, rate on a 1-5 scale whether it is a "
          "coherent, playable game concept: 1 = incoherent or not a game, 3 = a game but vague or self-contradictory "
          "in places, 5 = clear, coherent and playable as described. Judge coherence only, not originality or "
          "appeal. Do not use any tools. Answer with only a JSON list of objects {\"id\": <id>, \"score\": <1-5>}, "
          "one per concept.\n\n")


def sample() -> pd.DataFrame:
    d = pd.read_parquet(RUN / "scored_qwen3e06.parquet")
    rng = np.random.default_rng(SEED)
    parts = [g.iloc[rng.choice(len(g), 10, replace=False)] for _, g in d[d["arm"] != "NONE"].groupby(["source", "quintile", "arm"])]
    none = d[d["arm"] == "NONE"]
    parts.append(none.iloc[rng.choice(len(none), 20, replace=False)])
    s = pd.concat(parts)
    return s.iloc[rng.permutation(len(s))]


def ask(judge: str, text: str) -> str:
    """One non-interactive call, run from an empty directory so no project file or instruction reaches the model."""
    cwd = tempfile.mkdtemp(prefix="insp-")
    if judge == "codex":
        with tempfile.NamedTemporaryFile("r", suffix=".txt") as f:
            subprocess.run(["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-o", f.name, text],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=900, check=True, cwd=cwd)
            return Path(f.name).read_text()
    out = subprocess.run(["claude", "-p", text, "--output-format", "text", "--disallowed-tools",
                          "Bash,Edit,Write,Read,WebSearch,WebFetch,Glob,Grep,Agent"],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900, check=True, cwd=cwd)
    return out.stdout


def main() -> None:
    judge = sys.argv[1]
    s = sample()
    if judge == "claude":
        s = s.iloc[:100]
    out = RUN / f"coherence_{judge}.jsonl"
    done = {json.loads(l)["row"] for l in out.read_text().splitlines()} if out.exists() else set()
    s = s[~s.index.isin(done)]
    for b in range(0, len(s), BATCH):
        g = s.iloc[b:b + BATCH]
        body = "\n\n".join(f"id {i}: {r.title}\n{r.short_description}\nGenres: {r.genres}. Tags: {r.tags}"
                           for i, r in zip(range(len(g)), g.itertuples()))
        for _ in range(3):
            try:
                raw = ask(judge, PROMPT + body)
                got = {int(x["id"]): int(x["score"]) for x in json.loads(re.search(r"\[.*\]", raw, re.S).group(0))}
                if set(got) == set(range(len(g))):
                    break
            except Exception as e:  # noqa: BLE001
                print("retry", b, repr(e)[:120], flush=True)
        else:
            continue
        with out.open("a") as fh:
            for i, row in enumerate(g.index):
                fh.write(json.dumps({"row": int(row), "score": got[i]}) + "\n")
        print(f"{judge} {min(b + BATCH, len(s))}/{len(s)}", flush=True)


if __name__ == "__main__":
    main()
