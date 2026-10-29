"""Generate the six arms of docs/design.md for one domain. Resumable: finished ids are skipped.

    uv run python scripts/run_pilot.py --domain steam --n 2 --out smoke     # smoke test
    uv run python scripts/run_pilot.py --domain steam --n 50 --out pilot
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap import targets as T
from latent_gap.decode import Corpus, decode, pair, plain, plain_diverse, summary_line
from latent_gap.domains import DOMAINS
from latent_gap.embedder import LocalQwen
from latent_gap.cli_judges import CliChat
from latent_gap.gateway import CHAT_MODEL, Gateway, ledger_cost

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
SEED = 20261022


MEM_PER = 20  # memory-probe items per half-year window


def swap_map(ks: list[int], seed: int) -> dict[int, int]:
    """Seeded single-cycle derangement, independent of outcomes and generator."""
    if len(ks) < 2:
        raise ValueError("swap requires at least two target indices")
    order = np.random.default_rng(seed).permutation(ks).tolist()
    return dict(zip(order, order[1:] + order[:1]))


def build_targets(P: np.ndarray, n: int, path: Path) -> dict:
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rng = np.random.default_rng(SEED)
    real_t, hidden, real_idx = T.real(P, rng, n)
    pairs_interp = T.distant_pairs(P, rng, n)
    pairs_pair = T.distant_pairs(P, rng, n)
    out = {
        "REAL": real_t, "REAL_idx": real_idx, "REAL_hidden": np.array(hidden + [None], dtype=object)[:-1],
        "INTERP": T.interp(P, pairs_interp), "INTERP_pairs": pairs_interp,
        "FAR": T.far(P, rng, n), "PAIR_pairs": pairs_pair,
    }
    for kind in ("NOISE", "CROWD"):
        lad = T.ladder(P, rng, n // len(T.LADDER_LEVELS), kind)
        out[kind] = np.stack(lad["t"])
        out[f"{kind}_hidden"] = np.array(lad["hidden"] + [None], dtype=object)[:-1]
        for key in ("base", "level", "lam", "iso"):
            out[f"{kind}_{key}"] = np.array(lad[key])
    np.savez(path, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="pilot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--feedback", default="score", choices=["residual", "score", "none"])
    ap.add_argument("--neighbours", default="true", choices=["true", "swap", "none"])
    ap.add_argument("--start-k", type=int, default=0, help="first saved target index; --n is the count")
    ap.add_argument("--neighbour-seed", type=int, default=20261028, help="same-arm donor cycle seed")
    ap.add_argument("--candidate-workers", type=int, default=None, help="parallel CLI candidates per target (default 3)")
    ap.add_argument("--targets-from", default=None, help="reuse targets.npz of another run (same corpus)")
    ap.add_argument("--arms", default="REAL,INTERP,FAR,NOISE,CROWD,PLAIN,PAIR")
    ap.add_argument("--generator", default="gateway",
                    help='"gateway" (gpt-6-luna), "gateway:<alias>" or a CLI harness: "claude:<model>", "codex:<model>"')
    args = ap.parse_args()
    intervention = args.neighbours != "true" or args.start_k != 0
    ks = list(range(args.start_k, args.start_k + args.n))
    if args.start_k < 0 or args.n < 1 or (args.candidate_workers is not None and args.candidate_workers < 1):
        ap.error("target indices/count and candidate workers must be positive (start-k may be zero)")
    if intervention and (not args.targets_from or args.feedback != "score"
                         or set(args.arms.split(",")) - {"REAL", "GAP", "FAR"}):
        ap.error("interventions require --targets-from, --feedback score and REAL/GAP/FAR arms")
    if intervention and args.generator not in ("claude:claude-sonnet-5-5", "codex:gpt-6-luna"):
        ap.error("interventions use only the two pre-registered CLI generators")

    domain = DOMAINS[args.domain]
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["id", "name", "split", "text", "adult", "date"])
    emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    past = (meta["split"] == "past").to_numpy()
    P = emb[past]
    pm = meta[past].reset_index(drop=True)
    corpus = Corpus(P, pm["name"].tolist(), pm["text"].tolist(), domain)

    run = DATA / "runs" / args.domain / args.out
    run.mkdir(parents=True, exist_ok=True)
    if intervention:
        import hashlib
        source = DATA / "runs" / args.domain / args.targets_from / "targets.npz"
        config = {k: v for k, v in vars(args).items() if k != "workers"}
        config["targets_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        config_path = run / "intervention.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError(f"resume configuration mismatch: {config_path}")
        elif (run / "items.jsonl").exists():
            raise ValueError("refusing to mix an intervention with an unlabelled existing run")
        else:
            config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.targets_from and not (run / "targets.npz").exists():
        import shutil
        shutil.copy(DATA / "runs" / args.domain / args.targets_from / "targets.npz", run / "targets.npz")
    tg = build_targets(P, args.n, run / "targets.npz")
    if intervention:
        if hashlib.sha256((run / "targets.npz").read_bytes()).hexdigest() != config["targets_sha256"]:
            raise ValueError("run targets differ from frozen source")
        if any(a not in tg or ks[-1] >= len(tg[a]) for a in args.arms.split(",")):
            raise ValueError("requested targets missing; interventions never generate targets")
    if "GAP" in args.arms.split(",") and "GAP" not in tg:  # added after the pilot: same seed scheme, own stream
        tg["GAP"] = T.gap(P, np.random.default_rng(SEED + 1), args.n)
        np.savez(run / "targets.npz", **tg)
    if "MEM" in args.arms.split(",") and "MEM" not in tg:  # memory probe: same seed for every generator
        tg.update(T.memory_probe(emb, meta["date"], P, np.random.default_rng(SEED + 2), MEM_PER))
        np.savez(run / "targets.npz", **tg)
    if args.generator.startswith("gateway"):  # "gateway" or "gateway:<alias>"
        gw = Gateway(ledger=DATA / "ledger.jsonl", model=args.generator.partition(":")[2] or CHAT_MODEL)
    else:
        gw = CliChat(args.generator, domain.schema)
    embedder = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4"))) if intervention else LocalQwen()
    donors = swap_map(ks, args.neighbour_seed) if args.neighbours == "swap" else {}

    jobs = []
    for arm in args.arms.split(","):
        for k in (range(len(tg["MEM"])) if arm == "MEM" else ks):
            jobs.append((arm, k))
    done = {(json.loads(l)["arm"], json.loads(l)["k"]) for l in (run / "items.jsonl").read_text().splitlines()} \
        if (run / "items.jsonl").exists() else set()
    jobs = [j for j in jobs if j not in done]

    # PLAIND (PLAIN with diversity control) is sequential: item k sees the ideas 0..k-1 of this run.
    if any(a == "PLAIND" for a, _ in jobs):
        prev = [summary_line(json.loads(l)["best"]["item"]) for l in (run / "items.jsonl").read_text().splitlines()
                if json.loads(l)["arm"] == "PLAIND" and json.loads(l).get("best")] if (run / "items.jsonl").exists() else []
        with (run / "items.jsonl").open("a") as f:
            for a, k in sorted(j for j in jobs if j[0] == "PLAIND"):
                try:
                    res = plain_diverse(gw, embedder, domain, prev, tag=f"{args.domain}-{args.out}-PLAIND")
                except Exception as e:
                    print("FAILED", (a, k), repr(e)[:300], flush=True)
                    continue
                f.write(json.dumps({"arm": a, "k": k, **res}) + "\n")
                f.flush()
                if res.get("best"):
                    prev.append(summary_line(res["best"]["item"]))
                print(f"[{k + 1}/{args.n}] PLAIND {k}: {prev[-1][:90] if prev else ''}", flush=True)
        jobs = [j for j in jobs if j[0] != "PLAIND"]

    def work(arm: str, k: int) -> dict:
        tag = f"{args.domain}-{args.out}-{arm}"
        if arm in ("REAL", "INTERP", "FAR", "NOISE", "CROWD", "GAP"):
            hidden = np.asarray(tg[f"{arm}_hidden"][k], dtype=np.int64) if f"{arm}_hidden" in tg else None
            donor = donors.get(k)
            donor_hidden = np.asarray(tg[f"{arm}_hidden"][donor], dtype=np.int64) \
                if donor is not None and f"{arm}_hidden" in tg else None
            res = decode(gw, embedder, domain, corpus, tg[arm][k], hidden=hidden, tag=tag,
                         feedback_mode=args.feedback, neighbours=args.neighbours,
                         neighbour_target=tg[arm][donor] if donor is not None else None,
                         neighbour_hidden=donor_hidden, candidate_workers=args.candidate_workers)
            if intervention:
                res.update(neighbour_condition=args.neighbours, donor_k=donor, generator=args.generator)
        elif arm == "MEM":  # one round of best-of-3: memory shows on the first try
            res = decode(gw, embedder, domain, corpus, tg["MEM"][k], hidden=np.asarray(tg["MEM_hidden"][k], dtype=np.int64),
                         tag=tag, feedback_mode="score", max_iters=1)
        elif arm == "PLAIN":
            res = plain(gw, embedder, domain, k, tag=tag)
        else:
            a, b = tg["PAIR_pairs"][k]
            res = pair(gw, embedder, domain, corpus, int(a), int(b), tag=tag)
        return {"arm": arm, "k": k, **res}

    failures = 0
    with ThreadPoolExecutor(args.workers) as ex, (run / "items.jsonl").open("a") as f:
        futs = {ex.submit(work, a, k): (a, k) for a, k in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rec = fu.result()
            except Exception as e:  # keep going; the job stays undone and is retried on the next invocation
                failures += 1
                print("FAILED", futs[fu], repr(e)[:300], flush=True)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            best = rec.get("best") or {}
            print(f"[{i}/{len(jobs)}] {rec['arm']} {rec['k']} sim={best.get('sim')}", flush=True)
    print("cost", ledger_cost(DATA / "ledger.jsonl"))
    if intervention and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
