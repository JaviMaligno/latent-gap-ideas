"""SUPERSEDED by judge_multi.py (it did not reject empty pitches: 33/392 items were scored on an empty pitch in the
Steam pilot; results kept in runs/<domain>/<run>/legacy/). `load_items` and the prompts are still used from here.

E3: blind judged quality. Every item (generated, plus real future items as calibration) is normalised into
the same short pitch, then scored by a judge model different from the generator.

    uv run python scripts/judge.py --domain steam --run pilot
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.domains import DOMAINS
from latent_gap.gateway import Gateway, ledger_cost

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
JUDGE = "gpt-5.6-sol"
NORMALISER = "gpt-6-luna"
N_CALIBRATION = 50
SEED = 7

NORMALISE = ("Rewrite this {noun} as a neutral pitch of about 120 words: what it is, {does}, what "
             "makes it distinct. Keep every concrete idea; add nothing; no marketing adjectives, no name, no "
             "superlatives. {rule}Plain prose, one paragraph.\n\n{text}")

RUBRIC = ("You are reviewing pitches for {noun}s. Score the pitch from 1 to 10 on each criterion and answer with "
          "JSON only: {{\"coherence\": int, \"feasibility\": int, \"appeal\": int, \"novelty\": int, "
          "\"rationale\": str}}.\n"
          "- coherence: the idea makes sense as a whole; its parts fit together.\n"
          "- feasibility: {feasible}.\n"
          "- appeal: {appeal}.\n"
          "- novelty: how different it is from the existing {noun}s listed below (10 = nothing like them).\n\n"
          "Existing {noun}s closest to it:\n{neighbours}\n\nPITCH:\n{pitch}")


def load_items(domain_name: str, run_name: str, n_cal: int) -> tuple[list[dict], np.ndarray, list[str]]:
    """Generated items of a run plus `n_cal` real future items (calibration), and the past corpus they are judged
    against."""
    root = DATA / "prepared" / domain_name
    run = DATA / "runs" / domain_name / run_name
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "text", "adult"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    past = (meta["split"] == "past").to_numpy()
    P, ptexts = emb[past], meta.loc[past, "text"].tolist()
    ftexts, F = meta.loc[~past, "text"].tolist(), emb[~past]

    items = []
    for line in (run / "items.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("best"):
            items.append({"arm": r["arm"], "k": r["k"], "text": r["best"]["text"],
                          "emb": np.array(r["best"]["emb"], dtype=np.float32)})
    rng = np.random.default_rng(SEED)
    for k, i in enumerate(rng.choice(len(ftexts), n_cal, replace=False)):
        items.append({"arm": "FUTURE_REAL", "k": k, "text": ftexts[i], "emb": F[i]})

    return items, P, ptexts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-cal", type=int, default=N_CALIBRATION)
    args = ap.parse_args()

    domain = DOMAINS[args.domain]
    run = DATA / "runs" / args.domain / args.run
    items, P, ptexts = load_items(args.domain, args.run, args.n_cal)

    out_path = run / "judged.jsonl"
    done = {(json.loads(l)["arm"], json.loads(l)["k"]) for l in out_path.read_text().splitlines()} \
        if out_path.exists() else set()
    gw = Gateway(ledger=DATA / "ledger.jsonl")

    def pitch(text: str, tag: str) -> str:
        return gw.chat([{"role": "user", "content": NORMALISE.format(noun=domain.noun, does=domain.does, rule=domain.pitch_rule, text=text[:2500])}],
                       model=NORMALISER, max_tokens=2000, tag=tag).strip()

    def work(it: dict) -> dict:
        tag = f"{args.domain}-{args.run}-judge"
        s = P @ it["emb"]
        nn = np.argsort(-s)[:6]
        nn = [i for i in nn if s[i] < 0.999][:5]  # a real past item must not be its own neighbour
        neighbours = "\n".join(f"- {pitch(ptexts[i], tag)}" for i in nn)
        own = pitch(it["text"], tag)
        raw = gw.chat([{"role": "user", "content": RUBRIC.format(noun=domain.noun, feasible=domain.feasible, appeal=domain.appeal, neighbours=neighbours, pitch=own)}],
                      model=JUDGE, max_tokens=3000, json_mode=True, tag=tag)
        score = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return {"arm": it["arm"], "k": it["k"], "pitch": own, **score}

    todo = [it for it in items if (it["arm"], it["k"]) not in done]
    with ThreadPoolExecutor(args.workers) as ex, out_path.open("a") as f:
        futs = {ex.submit(work, it): it for it in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                rec = fu.result()
            except Exception as e:
                print("FAILED", futs[fu]["arm"], futs[fu]["k"], repr(e)[:200], flush=True)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {rec['arm']} {rec['k']} coh={rec.get('coherence')} nov={rec.get('novelty')}",
                  flush=True)

    df = pd.read_json(out_path, lines=True)
    print(df.groupby("arm")[["coherence", "feasibility", "appeal", "novelty"]].median().to_string())
    print("cost", ledger_cost(DATA / "ledger.jsonl"))


if __name__ == "__main__":
    main()
