"""Stress test of docs/prereg/inspiration_from_anything.md with Claude Code and Codex as generators (subscription,
no API spend; exploratory, contamination-tolerant: both were trained after the future games appeared, so they are read
only against their own NONE arm).

10 stimuli per source x quintile (the first 10 of each cell of stimuli.parquet, which is already a random draw),
INSTRUCTED only, 25 stimuli per call returned as a JSON list; NONE = 4 calls of 25 ideas with no material. No tools.
    uv run python scripts/inspiration_cli_generate.py codex
    uv run python scripts/inspiration_cli_generate.py claude
Output: runs/inspiration/<judge>-cli/ideas.jsonl, same fields as scripts/inspiration_generate.py
"""
from __future__ import annotations

import json
import re
import sys

import inspiration_coherence as C
import inspiration_generate as G
import pandas as pd

PER_CELL, BATCH, N_NONE_CALLS = 10, 25, 4
FIELDS = ("title", "short_description", "genres", "tags")
FORMAT = ("Answer with only a JSON list of objects, one per item, each with the keys \"id\", \"title\", "
          "\"short_description\" (at most 60 words, like a Steam store blurb), \"genres\" (comma-separated Steam "
          "genres) and \"tags\" (5-10 comma-separated Steam user tags). Do not use any tools.")


def main() -> None:
    gen = sys.argv[1]
    st = pd.read_parquet(G.INS / "stimuli.parquet")
    picks = [i for _, g in st.groupby(["source", "quintile"]) for i in g.index[:PER_CELL]]
    out = G.DATA / "runs/inspiration" / f"{gen}-cli" / "ideas.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    done_s = {r["stim"] for r in prior if r["arm"] == "INSTRUCTED"}
    done_n = sum(r["arm"] == "NONE" for r in prior) // BATCH
    todo = [i for i in picks if i not in done_s]
    jobs = [("INSTRUCTED", todo[b:b + BATCH]) for b in range(0, len(todo), BATCH)]
    jobs += [("NONE", list(range(BATCH)))] * max(0, N_NONE_CALLS - done_n)
    for n, (arm, ids) in enumerate(jobs):
        if arm == "INSTRUCTED":
            body = "\n\n".join(f"Item {k}. Here is something you happened to read today:\n\n"
                               f"{G.stimulus_block(st.loc[i, 'title'], st.loc[i, 'text'])}"
                               for k, i in enumerate(ids))
            text = ("For each item below, propose an original video game that could be released on Steam, using what "
                    "you read in that item as inspiration for the game. Treat each item independently.\n\n"
                    f"{body}\n\n{FORMAT}")
        else:
            text = f"Propose {BATCH} original video games that could be released on Steam, ids 0-{BATCH - 1}. {FORMAT}"
        for _ in range(3):
            try:
                raw = C.ask(gen, text)
                got = {int(x["id"]): x for x in json.loads(re.search(r"\[.*\]", raw, re.S).group(0))}
                if set(got) >= set(range(len(ids))):
                    break
            except Exception as e:  # noqa: BLE001
                print("retry", n, repr(e)[:120], flush=True)
        else:
            continue
        with out.open("a") as fh:
            for k, i in enumerate(ids):
                rec = {"stim": int(i) if arm == "INSTRUCTED" else -1, "arm": arm,
                       "k": k if arm == "NONE" else 0, **{f: str(got[k].get(f, "")) for f in FIELDS}}
                fh.write(json.dumps(rec) + "\n")
        print(f"{gen} {n + 1}/{len(jobs)} {arm}", flush=True)


if __name__ == "__main__":
    main()
