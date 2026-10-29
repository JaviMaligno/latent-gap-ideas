"""Calibration of the coherence gate of the coherent no-neighbour search, before any non-smoke run.

Per domain: 25 visible corpus texts, 25 GPU token-search witnesses (kaggle-out/witness_gpu*.jsonl) and 50 constructed
fluent-but-degenerate texts (10 each: keyword-list body, shuffled body, one sentence repeated, off-domain body, body
that drifts into the other domain halfway). Every text is rated blind by the three judges the search can use; the
non-LLM gibberish flag and the length bounds are also applied (flag rates on 500 corpus texts and all 200 witnesses).

Usage: uv run python scripts/coherent_calibration.py [--workers 8] [--report-only]
Output: runs/coherent_calibration/{texts.jsonl,judgements.jsonl,report.md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from latent_gap.coherent import bounds_violation, gibberish_rule, gibberish_stats, judge
from latent_gap.domains import DOMAINS

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = DATA / "runs/coherent_calibration"
JUDGES = ("claude:claude-sonnet-5-5", "codex:gpt-6-luna", "claude:claude-haiku-5-5")
BODY = {"steam": "about", "arxiv": "abstract"}


def corpus_items(d: str, n: int, seed: int) -> list[dict]:
    m = pd.read_parquet(DATA / f"prepared/{d}/corpus.parquet")
    m = m[~m.adult & (m.split == "past")].sample(n, random_state=seed)
    if d == "steam":
        return [{"name": r.name, "genres": [g for g in str(r.genres).split(",") if g],
                 "tags": [t for t in str(r.tags).split(",") if t], "about": r.text.split("About: ", 1)[1]}
                for r in m.itertuples()]
    return [{"title": r.name, "abstract": r.abstract} for r in m.itertuples()]


def degenerate(d: str, base: list[dict], other: list[dict], rng: random.Random) -> list[tuple[str, dict]]:
    body, obody = BODY[d], BODY["arxiv" if d == "steam" else "steam"]
    vocab = sorted({w for it in base for w in re.findall(r"[A-Za-z]{4,}", it[body])})
    out = []
    for i, it in enumerate(base[:50]):
        kind = ("keywords", "shuffled", "repeated", "off_domain", "drift")[i // 10]
        new = dict(it)
        words = it[body].split()
        if kind == "keywords":
            new[body] = ", ".join(rng.sample(vocab, 60))
        elif kind == "shuffled":
            new[body] = " ".join(rng.sample(words, len(words)))
        elif kind == "repeated":
            first = re.split(r"(?<=[.!?])\s+", it[body].strip())[0]
            new[body] = " ".join([first] * max(2, 150 // max(1, len(first.split()))))
        elif kind == "off_domain":
            new[body] = other[i][obody]
        else:
            half = " ".join(words[: len(words) // 2])
            ow = other[i][obody].split()
            new[body] = half + " " + " ".join(ow[len(ow) // 2:])
        out.append((kind, new))
    return out


def build() -> list[dict]:
    rng = random.Random(0)
    wit = [json.loads(line) for f in sorted((DATA / "runs/support-search/kaggle-out").glob("witness_gpu*.jsonl"))
           for line in f.read_text().splitlines()]
    texts = []
    for d in ("steam", "arxiv"):
        dom = DOMAINS[d]
        for it in corpus_items(d, 50, 1)[:25]:
            texts.append({"domain": d, "set": "corpus", "kind": "corpus", "item": it, "text": dom.render(it)})
        mine = [w for w in wit if w["domain"] == d]
        for w in rng.sample(mine, 50)[:25]:
            texts.append({"domain": d, "set": "witness", "kind": w["arm"], "item": None, "text": w["text"]})
        other = corpus_items("arxiv" if d == "steam" else "steam", 50, 3)
        for kind, it in degenerate(d, corpus_items(d, 50, 2), other, rng):
            texts.append({"domain": d, "set": "degenerate", "kind": kind, "item": it, "text": dom.render(it)})
    for t in texts:
        t["sha256"] = hashlib.sha256(t["text"].encode()).hexdigest()
    return texts


def flag_rates() -> pd.DataFrame:
    """Gibberish-flag rate per rule on 500 visible corpus texts per domain and all 200 GPU witnesses."""
    rows = []
    for d in ("steam", "arxiv"):
        dom = DOMAINS[d]
        for it in corpus_items(d, 500, 4):
            rows.append({"domain": d, "set": "corpus (500)", "rule": gibberish_rule(gibberish_stats(dom.render(it)))})
    for f in sorted((DATA / "runs/support-search/kaggle-out").glob("witness_gpu*.jsonl")):
        for line in f.read_text().splitlines():
            w = json.loads(line)
            rows.append({"domain": w["domain"], "set": "witness (all 100)", "rule": gibberish_rule(gibberish_stats(w["text"]))})
    df = pd.DataFrame(rows)
    tab = df.assign(flagged=df.rule != "").groupby(["domain", "set"]).agg(n=("rule", "size"), flagged=("flagged", "mean"))
    for r in ("script", "non_ascii", "dict_rate", "dict_window"):
        tab[r] = df.groupby(["domain", "set"]).rule.apply(lambda s, r=r: (s == r).mean())
    return tab.round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "models/qwen3e06-onnx/tokenizer.json"))
    texts = build()
    for t in texts:
        st = gibberish_stats(t["text"])
        t["gib_rule"] = gibberish_rule(st)
        t["bounds"] = bounds_violation(t["domain"], t["item"], len(tok.encode(t["text"]).ids)) if t["item"] else "-"
    (OUT / "texts.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts))
    jpath = OUT / "judgements.jsonl"
    done = {(r["judge"], r["sha256"]) for r in map(json.loads, jpath.read_text().splitlines())} if jpath.exists() else set()
    order = {"degenerate": 0, "corpus": 1, "witness": 2}  # the false-accept test first
    todo = [(j, t) for t in sorted(texts, key=lambda t: order[t["set"]]) for j in JUDGES
            if (j, t["sha256"]) not in done]
    lock = threading.Lock()
    if not args.report_only and todo:
        print(f"{len(todo)} judgements to do", flush=True)

        def one(job):
            j, t = job
            r = judge(j, DOMAINS[t["domain"]].noun, t["text"])
            with lock, jpath.open("a") as f:
                f.write(json.dumps({"judge": j, "sha256": t["sha256"], **r}) + "\n")

        with ThreadPoolExecutor(args.workers) as ex:
            for i, _ in enumerate(ex.map(one, todo)):
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    J = pd.DataFrame(map(json.loads, jpath.read_text().splitlines())).drop_duplicates(["judge", "sha256"])
    T = pd.DataFrame(texts)[["domain", "set", "kind", "sha256", "gib_rule", "bounds"]]
    W = J.pivot(index="sha256", columns="judge", values="coherence")
    W.columns = [c.split(":")[1].replace("claude-", "").replace("-5-5", "") for c in W.columns]
    df = T.merge(W, left_on="sha256", right_index=True, how="left")
    rows = []
    for (d, s), g in df.groupby(["domain", "set"]):
        row = {"domain": d, "set": s, "n": len(g), "flagged": f"{(g.gib_rule != '').mean():.2f}"}
        for c in ("sonnet", "gpt-6-luna", "haiku"):
            row[f"{c} >=7"] = f"{(g[c] >= 7).mean():.2f}"
            row[f"{c} >=8"] = f"{(g[c] >= 8).mean():.2f}"
            row[f"{c} errors"] = int(g[c].isna().sum())
        for a, b in (("sonnet", "gpt-6-luna"), ("sonnet", "haiku")):
            for th in (7, 8):
                row[f"gate {a}+{b} >={th}"] = f"{((g[a] >= th) & (g[b] >= th) & (g.gib_rule == '')).mean():.2f}"
        rows.append(row)
    rep = pd.DataFrame(rows)
    kinds = df[df.set == "degenerate"].groupby(["domain", "kind"]).agg(
        n=("sha256", "size"), flagged=("gib_rule", lambda s: (s != "").mean()),
        sonnet7=("sonnet", lambda s: (s >= 7).mean()), luna7=("gpt-6-luna", lambda s: (s >= 7).mean()),
        haiku7=("haiku", lambda s: (s >= 7).mean())).round(2)
    out = ["## Flag rates per rule", flag_rates().reset_index().to_string(index=False),
           "## Accept rates (share of texts)", rep.T.to_string(header=False),
           "## Degenerate texts by kind (accept >= 7)", kinds.reset_index().to_string(index=False)]
    (OUT / "report.md").write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
