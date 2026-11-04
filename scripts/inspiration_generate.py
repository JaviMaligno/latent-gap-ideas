"""Generation for docs/prereg/inspiration_from_anything.md (gpt-4o arms INCIDENTAL, INSTRUCTED, NONE).

One idea per call, JSON {title, short_description, genres, tags}; resumable JSONL.
    ALIAS=gpt-4o-0806 uv run python scripts/inspiration_generate.py          # full run
    LG_LIMIT=6 ALIAS=gpt-4o-0806 uv run python scripts/inspiration_generate.py  # smoke test (first 6 jobs)
Output: runs/inspiration/<alias>/ideas.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
INS = DATA / "prepared/inspiration"
N_NONE = 200
SYSTEM = "You are a creative video game designer. Answer with a single JSON object and nothing else."
TASK = ("Propose an original video game that could be released on Steam. Return a JSON object with the keys "
        "\"title\", \"short_description\" (at most 60 words, like a Steam store blurb), \"genres\" (comma-separated "
        "Steam genres) and \"tags\" (5-10 comma-separated Steam user tags).")


def stimulus_block(title: str, text: str) -> str:
    return f"{title}\n{text}"


def prompt(arm: str, stim: str | None) -> str:
    if arm == "NONE":
        return TASK
    p = "Here is something you happened to read today:\n\n" + stim + "\n\n" + TASK
    return p + (" Use what you read as inspiration for the game." if arm == "INSTRUCTED" else "")


def jobs(st: pd.DataFrame) -> list[tuple[int, str, int]]:
    out = [(i, a, 0) for i in range(len(st)) for a in ("INCIDENTAL", "INSTRUCTED")]
    return out + [(-1, "NONE", k) for k in range(N_NONE)]


def main() -> None:
    from latent_gap.gateway import Gateway
    alias = os.environ["ALIAS"]
    gw = Gateway(ledger=DATA / "ledger.jsonl", model=alias)
    st = pd.read_parquet(INS / "stimuli.parquet")
    out = DATA / "runs/inspiration" / alias / "ideas.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["stim"], r["arm"], r["k"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [j for j in jobs(st) if j not in done]
    if os.environ.get("LG_LIMIT"):
        n = int(os.environ["LG_LIMIT"])
        todo = todo[:n // 2] + [j for j in todo if j[1] == "NONE"][:n - n // 2]
    lock = threading.Lock()

    def call(text: str) -> dict:
        for _ in range(3):
            raw = gw.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], max_tokens=400,
                          json_mode=True, tag="inspiration", extra={"temperature": 1.0})
            try:
                d = json.loads(raw)
                if d.get("title") and d.get("short_description"):
                    return {k: str(d.get(k, "")) for k in ("title", "short_description", "genres", "tags")}
            except json.JSONDecodeError:
                pass
        return {"title": "", "short_description": "", "genres": "", "tags": ""}

    def job(j: tuple[int, str, int]) -> None:
        i, arm, k = j
        s = None if i < 0 else stimulus_block(st.loc[i, "title"], st.loc[i, "text"])
        rec = {"stim": i, "arm": arm, "k": k, **call(prompt(arm, s))}
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"{len(todo)} calls to {alias}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(job, todo), 1):
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
