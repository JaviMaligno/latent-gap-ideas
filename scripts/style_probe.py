"""Style confound in the anticipation test: does rewriting a real PAST item (content fixed, wording by an LLM) move
its nearest neighbour to the future? If so, anticipation partly measures LLM-era style, not ideas.

For 200 real past items per domain (scored against the past minus those 200), each generator paraphrases the item into the same JSON schema the
generated ideas use. Outcomes: anticipation hit rate (scripts/anticipation.py rules: future from --future-from, past
subsampled to the future's size, the 200 items themselves removed from the past), and
cos(paraphrase, original) as the content check.

    uv run python scripts/style_probe.py --domain arxiv --gens gateway:gpt-4o-0806,gateway:gpt-6-luna,claude:claude-sonnet-5-5
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.cli_judges import CliChat
from latent_gap.decode import SYSTEM, _parse
from latent_gap.domains import DOMAINS
from latent_gap.embedder import LocalQwen
from latent_gap.gateway import Gateway

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
SEED, B = 20261026, 5
PROMPT = ("Rewrite this {noun} in your own words as the JSON object. Keep exactly the same content: the same subject, "
          "the same {does}, the same claims and details. Add nothing, remove nothing, invent nothing; only the wording "
          "changes.\n\n{text}")


def nearest(E, C, chunk=4096):
    out = np.full(len(E), -1.0, np.float32)
    for i in range(0, len(C), chunk):
        out = np.maximum(out, (E @ C[i:i + chunk].T).max(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="arxiv")
    ap.add_argument("--gens", required=True)
    ap.add_argument("--future-from", default="2024-01-01")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    domain = DOMAINS[args.domain]
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult", "date", "text"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    past_i = np.flatnonzero((meta["split"] == "past").to_numpy())
    fut = (meta["split"] == "future").to_numpy().copy()
    fut &= (pd.to_datetime(meta["date"], errors="coerce") >= args.future_from).to_numpy()
    F = emb[np.flatnonzero(fut)]
    rng = np.random.default_rng(SEED)
    N = args.n
    pick = rng.choice(past_i, N, replace=False)
    out_dir = DATA / "runs" / args.domain / ("style" if N == 200 else f"style-smoke{N}")
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = LocalQwen()

    def score(E: np.ndarray) -> tuple[float, float]:
        """Hit rate vs the past minus the picked items and their near-duplicates, equal-size subsamples."""
        f = nearest(E, F)
        hits = []
        for b in range(B):
            sub = np.random.default_rng(SEED + b).choice(np.setdiff1d(past_i, pick), len(F), replace=False)
            Psub = emb[sub]
            p = nearest(E, Psub)
            hits.append((f > p).mean())
        return float(np.mean(hits)), float(np.median(f))

    rows = [{"gen": "original", "n": N, "hit": score(emb[pick])[0], "content_cos": 1.0}]
    for spec in args.gens.split(","):
        path = out_dir / f"{spec.replace(':', '_')}.jsonl"
        done = {json.loads(l)["i"]: json.loads(l) for l in path.read_text().splitlines()} if path.exists() else {}
        gw = Gateway(ledger=DATA / "ledger.jsonl", model=spec.split(":", 1)[1]) if spec.startswith("gateway") \
            else CliChat(spec, domain.schema)
        sys = SYSTEM.format(noun=domain.noun, spec=domain.json_spec)

        def one(i: int):
            msg = PROMPT.format(noun=domain.noun, does=domain.does, text=meta.at[i, "text"][:2500])
            for _ in range(3):
                try:
                    item = _parse(gw.chat([{"role": "system", "content": sys}, {"role": "user", "content": msg}],
                                          json_mode=True, tag=f"style-{args.domain}"))
                except Exception as e:  # noqa: BLE001
                    print("FAILED", spec, i, repr(e)[:160], flush=True)
                    continue
                if item:
                    return {"i": int(i), "item": item}
            return None

        todo = [int(i) for i in pick if int(i) not in done]
        with ThreadPoolExecutor(args.workers) as ex, path.open("a") as f:
            for rec in ex.map(one, todo):
                if rec:
                    f.write(json.dumps(rec) + "\n")
                    done[rec["i"]] = rec
        ids = [int(i) for i in pick if int(i) in done]
        E = embedder.embed([domain.render(done[i]["item"]) for i in ids])
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        rows.append({"gen": spec, "n": len(ids), "hit": score(E)[0],
                     "content_cos": float(np.median(np.einsum("ij,ij->i", E, emb[ids])))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    df.to_csv(out_dir / "style_probe.csv", index=False)


if __name__ == "__main__":
    main()
