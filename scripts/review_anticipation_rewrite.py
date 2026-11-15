"""A4 of docs/prereg/review_anticipation_control.md: gpt-4o rewrites of the held-out games in the idea format. PAID.

Rewrites the 200 A1 held-out future games (H0) and the 200 A1 past-2023 games, read from
runs/review-anticipation/heldout_ids.json (written by `review_anticipation_control.py free`), in one random order
(default_rng(SEED + 3)). One call per item, resumable JSONL, ledger tag `review-rewrite`.
    uv run python scripts/review_anticipation_rewrite.py --dry-run                     # no call: prompts + call count
    LG_LIMIT=6 ALIAS=gpt-4o-0806 uv run python scripts/review_anticipation_rewrite.py   # format check (kept, disclosed)
    ALIAS=gpt-4o-0806 uv run python scripts/review_anticipation_rewrite.py              # the 400 calls (after approval)
Output: runs/review-anticipation/rewrites.jsonl (id, group fut/past23, title, short_description, genres, tags)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspiration_generate import SYSTEM, TASK  # noqa: E402  (the Section 6 prompts, verbatim)

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
WORK = DATA / "runs/review-anticipation"
SEED = 20261114
ALIAS = "gpt-4o-0806"
# Output specification = the Section 6 TASK from "Return a JSON object" on, verbatim.
SPEC = TASK[TASK.index("Return a JSON object"):]
assert SPEC.startswith("Return a JSON object with the keys \"title\", \"short_description\" (at most 60 words")


def prompt(text: str) -> str:
    return ("Here is the Steam store entry of a video game:\n\n" + text + "\n\nRewrite it as a proposal for a video game "
            "that could be released on Steam, keeping its concept, setting and mechanics. Do not reuse its title. " + SPEC)


def jobs(work: Path) -> list[tuple[str, str]]:
    ids = json.loads((work / "heldout_ids.json").read_text())
    assert ids["seed"] == SEED and len(ids["H0"]) == 200 and len(ids["past23"]) == 200
    items = [(i, "fut") for i in ids["H0"]] + [(i, "past23") for i in ids["past23"]]
    order = np.random.default_rng(SEED + 3).permutation(len(items))
    return [items[k] for k in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build and print prompts and the call count; no call")
    ap.add_argument("--work", default=str(WORK))
    args = ap.parse_args()
    work = Path(args.work)
    meta = pd.read_parquet(DATA / "prepared/steam/corpus.parquet", columns=["id", "name", "text", "adult"])
    meta = meta[~meta["adult"]].set_index("id")
    out = work / "rewrites.jsonl"
    done = {(r["id"], r["group"]) for r in map(json.loads, out.read_text().splitlines()) if r} if out.exists() else set()
    todo = [j for j in jobs(work) if j not in done]
    if os.environ.get("LG_LIMIT"):
        todo = todo[:int(os.environ["LG_LIMIT"])]
    # the name field is not in the prompt; count prompts whose store text still mentions the name (disclosure only)
    n_name = sum(len(str(meta.loc[i, "name"]).strip()) >= 4 and str(meta.loc[i, "name"]).strip() in meta.loc[i, "text"]
                 for i, _ in todo)
    alias = os.environ.get("ALIAS", ALIAS)
    if args.dry_run:
        n_in = [len(prompt(meta.loc[i, "text"])) + len(SYSTEM) for i, _ in todo]
        print(f"DRY RUN: {len(todo)} calls would go to {alias} ({len(done)} already done); "
              f"groups {pd.Series([g for _, g in todo]).value_counts().to_dict() if todo else {}}; "
              f"prompt chars median {int(np.median(n_in)) if n_in else 0}, ~{sum(n_in) / 4 / 1e6:.3f}M input tokens; "
              f"max 3 attempts each (worst case {3 * len(todo)} calls); {n_name} store texts mention the game's name")
        for i, g in todo[:2]:
            print(f"\n--- {g} {i} ---\n[system] {SYSTEM}\n[user] {prompt(meta.loc[i, 'text'])}")
        return
    from latent_gap.gateway import Gateway
    gw = Gateway(ledger=DATA / "ledger.jsonl", model=alias)
    work.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def call(text: str) -> dict:
        for _ in range(3):
            try:
                raw = gw.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], max_tokens=400,
                              json_mode=True, tag="review-rewrite", extra={"temperature": 1.0})
            except RuntimeError as e:  # Azure content filter: record as a failed item (counted), do not retry
                if "ContentPolicy" in str(e) or "content management" in str(e):
                    return {"title": "", "short_description": "", "genres": "", "tags": "", "filtered": True}
                raise
            try:
                d = json.loads(raw)
                if d.get("title") and d.get("short_description"):
                    return {k: str(d.get(k, "")) for k in ("title", "short_description", "genres", "tags")}
            except json.JSONDecodeError:
                pass
        return {"title": "", "short_description": "", "genres": "", "tags": ""}

    def job(j: tuple[str, str]) -> None:
        i, g = j
        rec = {"id": i, "group": g, **call(prompt(meta.loc[i, "text"]))}
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"{len(todo)} calls to {alias}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(job, todo), 1):
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
