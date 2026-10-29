"""Blind items for the human-judge page: exactly what the judges saw (pitch + 5 nearest pitches) plus their scores.

    uv run python scripts/human_items.py --run grid-gpt-6-luna --arms GAP:6,INTERP:5,PLAIN:5 --out items.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
JUDGES = ("gpt61sol", "opus", "sonnet", "haiku")
SEED = 20261024
JUDGE_SEED, N_CAL = 7, 50  # must match scripts/judge.py
NOUN = {"steam": "juego", "arxiv": "paper", "patents": "patente"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", required=True, help="ARM:n,ARM:n")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", default="B2", help="id prefix")
    ap.add_argument("--exclude", nargs="*", default=[], help="items.json of earlier batches: never reuse their (arm, k)")
    args = ap.parse_args()
    run = DATA / "runs" / args.domain / args.run
    inputs = {(x["arm"], x["k"]): x for x in map(json.loads, (run / "judge_inputs.jsonl").read_text().splitlines())}
    scores: dict = {}
    for j in JUDGES:
        for x in map(json.loads, (run / f"judged_{j}.jsonl").read_text().splitlines()):
            scores.setdefault((x["arm"], x["k"]), {})[j] = {c: x[c] for c in ("coherence", "feasibility", "appeal", "novelty")}
    items = {(r["arm"], r["k"]): r for r in map(json.loads, (run / "items.jsonl").read_text().splitlines())}
    tg = np.load(run / "targets.npz", allow_pickle=True)
    meta = pd.read_parquet(DATA / "prepared" / args.domain / "corpus.parquet", columns=["name", "split", "adult"])
    names = meta[(~meta.adult) & (meta.split == "past")]["name"].reset_index(drop=True)
    fnames = meta[(~meta.adult) & (meta.split != "past")]["name"].reset_index(drop=True)
    # FUTURE_REAL calibration items are drawn in scripts/judge.py: same seed, same order
    fut_pick = np.random.default_rng(JUDGE_SEED).choice(len(fnames), N_CAL, replace=False)

    noun = NOUN[args.domain]
    seen = {(x["arm"], x["k"]) for f in args.exclude for x in json.loads(Path(f).read_text())}
    rng = np.random.default_rng(SEED)
    out = []
    for spec in args.arms.split(","):
        arm, n = spec.split(":")
        ks = sorted(k for (a, k) in inputs if a == arm and len(scores.get((a, k), {})) == len(JUDGES) and (a, k) not in seen)
        for k in rng.choice(ks, int(n), replace=False).tolist():
            prompt = inputs[(arm, k)]["prompt"]
            body = prompt.split("closest to it:\n", 1)[1]
            nn, pitch = body.split("\n\nPITCH:\n", 1)
            rec = items.get((arm, k), {})
            best = rec.get("best") or {}
            if arm == "INTERP":
                a, b = tg["INTERP_pairs"][k]
                origin = f"Pareja: {names[int(a)]} + {names[int(b)]}"
            elif arm == "GAP":
                origin = f"Hueco dentro del catálogo: punto típico del espacio sin ningún {noun} cerca"
            elif arm == "PLAIND":
                origin = f"Sin espacio: «inventa un {noun} que no exista y distinto de todos los que ya propusiste»"
            elif arm == "FUTURE_REAL":
                origin = f"No lo generó nadie: es un {noun} real publicado después del corte (calibración)"
            elif arm == "PLAIN":
                origin = f"Sin espacio: «inventa un {noun} que no exista»"
            else:
                origin = f"Punto del espacio del brazo {arm}"
            out.append({"id": f"{args.batch}-{arm}-{k}", "arm": arm, "k": k, "pitch": pitch.strip(),
                        "neighbours": [l[2:] for l in nn.strip().splitlines() if l.startswith("- ")],
                        "origin": origin, "name": fnames[int(fut_pick[k])] if arm == "FUTURE_REAL" else
                        (best.get("item") or {}).get("name") or (best.get("item") or {}).get("title"),
                        "landing": best.get("sim"), "judges": scores[(arm, k)]})
    rng.shuffle(out)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(len(out), [i["id"] for i in out])


if __name__ == "__main__":
    main()
