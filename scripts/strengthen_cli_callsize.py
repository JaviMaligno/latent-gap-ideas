"""Call size within one run: 5 vs 25 ideas per call (pre-registered in docs/prereg/strengthen_cli_callsize.md).

Same generators, CLI flags and prompt as scripts/strengthen_cli_powered.py; NONE and steam_game arms at 5 and at 25
ideas per call, interleaved in one random call order.
    uv run python scripts/strengthen_cli_callsize.py gen codex      # subscription CLI, no API spend
    uv run python scripts/strengthen_cli_callsize.py gen claude
    OMP_NUM_THREADS=2 uv run python scripts/strengthen_cli_callsize.py analyze codex claude   # local Qwen3 ONNX
Output: runs/inspiration/<gen>-clisize/{ideas,attempts}.jsonl, docs/prereg/strengthen_cli_callsize.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402
import inspiration_generate as G  # noqa: E402
import strengthen_cli_powered as P  # noqa: E402

ORDER_SEED, N_NONE_IDEAS = 20261106, 400
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/strengthen_cli_callsize.json"


def sg_ids(st: pd.DataFrame) -> dict[int, list[int]]:
    s = st[st["source"] == "steam_game"]
    return {q: [int(i) for i in g.index] for q, g in s.groupby("quintile")}


def jobs(st: pd.DataFrame) -> list[tuple[int, str, int, list[int]]]:
    byq = sg_ids(st)
    assert sorted(len(v) for v in byq.values()) == [40] * 5
    js = [("NONE", 5, list(range(5)))] * (N_NONE_IDEAS // 5) + [("NONE", 25, list(range(25)))] * (N_NONE_IDEAS // 25)
    js += [("INSTRUCTED", 5, v[b:b + 5]) for v in byq.values() for b in range(0, 40, 5)]
    js += [("INSTRUCTED", 25, [i for v in byq.values() for i in v[b:b + 5]]) for b in range(0, 40, 5)]
    order = np.random.default_rng(ORDER_SEED).permutation(len(js))
    return [(int(c), *js[c]) for c in order]  # call id = index in the unshuffled list


def prompt(st, arm, n, ids):
    if arm == "NONE":
        return f"Propose {n} original video games that could be released on Steam, ids 0-{n - 1}. {P.FORMAT}"
    body = "\n\n".join(f"Item {k}. Here is something you happened to read today:\n\n"
                       f"{G.stimulus_block(st.loc[i, 'title'], st.loc[i, 'text'])}" for k, i in enumerate(ids))
    return ("For each item below, propose an original video game that could be released on Steam, using what "
            "you read in that item as inspiration for the game. Treat each item independently.\n\n"
            f"{body}\n\n{P.FORMAT}")


def gen(g: str) -> None:
    st = pd.read_parquet(G.INS / "stimuli.parquet")
    run = G.DATA / "runs/inspiration" / f"{g}-clisize"
    run.mkdir(parents=True, exist_ok=True)
    out, log = run / "ideas.jsonl", run / "attempts.jsonl"
    js = jobs(st)
    for pas in range(2):
        done = {json.loads(l)["call"] for l in out.read_text().splitlines()} if out.exists() else set()
        todo = [j for j in js if j[0] not in done]
        print(f"{g} pass {pas}: {len(todo)} calls to do", flush=True)
        for n, (call, arm, size, ids) in enumerate(todo):
            got = None
            for att in range(3):
                t0, status = time.time(), "ok"
                try:
                    raw = P.ask(g, prompt(st, arm, size, ids))
                    x = {int(r["id"]): r for r in json.loads(re.search(r"\[.*\]", raw, re.S).group(0))}
                    if set(x) >= set(range(len(ids))):
                        got = x
                    else:
                        status = f"missing_ids:{sorted(set(range(len(ids))) - set(x))}"
                except Exception as e:  # noqa: BLE001
                    status = "error:" + repr(e)[:120]
                with log.open("a") as fh:
                    fh.write(json.dumps({"call": call, "arm": arm, "size": size, "pass": pas, "attempt": att,
                                         "status": status, "seconds": round(time.time() - t0),
                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                if got is not None:
                    break
                print("retry", call, status[:120], flush=True)
                time.sleep(5)
            if got is None:
                print(f"{g} call {call} FAILED", flush=True)
                continue
            with out.open("a") as fh:
                for k, i in enumerate(ids):
                    rec = {"call": call, "size": size, "stim": int(i) if arm == "INSTRUCTED" else -1, "arm": arm,
                           "k": k, "model": P.MODELS[g], **{f: str(got[k].get(f, "")) for f in P.FIELDS}}
                    fh.write(json.dumps(rec) + "\n")
            print(f"{g} {time.strftime('%H:%M:%S')} {n + 1}/{len(todo)} call {call} {arm} x{size}", flush=True)
            time.sleep(2)


# ---------------------------------------------------------------- analysis

def ols_cr1(y, X, cl, j, rng, nboot=9999):
    """CR1 t test of coefficient j and a wild cluster bootstrap-t with the null imposed."""
    b, V, G_ = P.cr1(y, X, cl)
    se = float(np.sqrt(V[j, j]))
    t = b[j] / se
    X0 = np.delete(X, j, axis=1)
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    f0, r0 = X0 @ b0, y - X0 @ b0
    uc = np.unique(cl)
    idx = np.searchsorted(uc, cl)
    tb = []
    for _ in range(nboot):
        ys = f0 + rng.choice([-1.0, 1.0], len(uc))[idx] * r0
        bb, vv, _ = P.cr1(ys, X, cl)
        tb.append(abs(bb[j] / np.sqrt(vv[j, j])))
    q = stats.t.ppf(0.975, G_ - 1)
    return {"coef": float(b[j]), "se_cr1": se, "n_clusters": int(G_), "ci95_cr1": [float(b[j] - q * se),
            float(b[j] + q * se)], "p_two_cr1": float(2 * stats.t.sf(abs(t), G_ - 1)),
            "p_two_wild_cluster_boot": float((np.array(tb) >= abs(t)).mean())}


def boot_mean(d, rng, n=4000):
    gs = [g["margin"].to_numpy() for _, g in d.groupby("call")]
    bs = [np.concatenate([gs[i] for i in rng.integers(0, len(gs), len(gs))]).mean() for _ in range(n)]
    return np.array(bs)


def cell(d, rng):
    return {"mean": float(d["margin"].mean()), "n_ideas": int(len(d)), "n_calls": int(d["call"].nunique()),
            "boot": boot_mean(d, rng)}


def score(g, P_, Fc, enc_box):
    from latent_gap.embedder import LocalQwen
    from latent_gap.steam import item_text
    st = pd.read_parquet(A.INS / "stimuli.parquet")
    rng = np.random.default_rng(A.SEED)
    run = A.DATA / "runs/inspiration" / f"{g}-clisize"
    d = pd.DataFrame([json.loads(l) for l in (run / "ideas.jsonl").read_text().splitlines()])
    n_calls_written = int(d["call"].nunique())
    d = d[d["title"] != ""].reset_index(drop=True)
    cache = run / "ideas_emb_qwen3e06.npy"
    if cache.exists() and len(np.load(cache)) == len(d):
        V = A.norm(np.load(cache))
    else:
        if enc_box[0] is None:
            enc_box[0] = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "2")))
        V = A.norm(enc_box[0].embed([item_text(a, b, c) for a, b, c in
                                     zip(d["genres"], d["tags"], d["short_description"])]))
        np.save(cache, V)
    fut = A.maxcos(V, Fc)
    d["margin"] = np.mean([fut - A.maxcos(V, P_[rng.choice(len(P_), len(Fc), replace=False)])
                           for _ in range(A.B)], 0)
    has = (d["stim"] >= 0).to_numpy()
    d["source"] = "NONE"
    d.loc[has, "source"] = st.loc[d.loc[has, "stim"].to_numpy(), "source"].to_numpy()
    assert set(d.loc[has, "source"]) == {"steam_game"}
    d.to_parquet(run / "scored_qwen3e06.parquet")
    return d, n_calls_written


def analyze(gens: list[str]) -> None:
    meta = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    E = A.norm(np.load(A.DATA / "prepared/steam/emb_qwen3e06.npy")[keep])
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    P_, Fc = E[past], E[(~past) & (appid > A.APPID_CUT)]
    del E
    enc_box = [None]
    res = {"prereg": "docs/prereg/strengthen_cli_callsize.md", "models": {g: P.MODELS[g] for g in gens}}
    frames = {}
    for g in gens:
        d, ncw = score(g, P_, Fc, enc_box)
        d["call_u"] = g + "-" + d["call"].astype(str)
        frames[g] = d
        rng = np.random.default_rng(A.SEED + 1)
        cells = {f"{a}-{s}": cell(d[(d["source"] == a) & (d["size"] == s)], rng)
                 for a in ("NONE", "steam_game") for s in (5, 25)}
        n = d[d["source"] == "NONE"]
        c1 = ols_cr1(n["margin"].to_numpy(), np.column_stack([np.ones(len(n)), (n["size"] == 25).to_numpy(float)]),
                     n["call"].to_numpy(), 1, rng)
        sg, s25 = (d["source"] == "steam_game").to_numpy(float), (d["size"] == 25).to_numpy(float)
        c2 = ols_cr1(d["margin"].to_numpy(), np.column_stack([np.ones(len(d)), sg, s25, sg * s25]),
                     d["call"].to_numpy(), 3, rng)

        def ci(x):
            return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        c1["effect_ci95_cluster_boot"] = ci(cells["NONE-25"]["boot"] - cells["NONE-5"]["boot"])
        gaps = {s: cells[f"steam_game-{s}"]["boot"] - cells[f"NONE-{s}"]["boot"] for s in (5, 25)}
        c2["effect_ci95_cluster_boot"] = ci(gaps[25] - gaps[5])
        att = pd.DataFrame([json.loads(l) for l in (A.DATA / "runs/inspiration" / f"{g}-clisize" /
                                                    "attempts.jsonl").read_text().splitlines()])
        res[g] = {"n_ideas": int(len(d)), "n_calls_written": ncw, "n_calls_missing": 144 - ncw,
                  "cells": {k: {kk: vv for kk, vv in v.items() if kk != "boot"} | {"ci95": ci(v["boot"])}
                            for k, v in cells.items()},
                  "C1_none_25_minus_5": c1, "C2_gap_interaction": c2,
                  "gap_sg_minus_none": {str(s): {"diff": cells[f"steam_game-{s}"]["mean"] - cells[f"NONE-{s}"]["mean"],
                                                 "ci95": ci(gaps[s])} for s in (5, 25)},
                  "attempts": {"total": int(len(att)), "non_ok": int((att["status"] != "ok").sum()),
                               "non_ok_by_arm_size": att[att["status"] != "ok"].groupby(["arm", "size"]).size()
                               .rename(lambda x: f"{x[0]}-{x[1]}").to_dict()}}
        print(g, json.dumps({k: res[g][k] for k in ("cells", "C1_none_25_minus_5", "C2_gap_interaction")},
                            indent=1, default=float), flush=True)
    if len(gens) == 2:
        res["holm2_C1"] = A.holm({g: res[g]["C1_none_25_minus_5"]["p_two_cr1"] for g in gens})
        res["holm2_C2"] = A.holm({g: res[g]["C2_gap_interaction"]["p_two_cr1"] for g in gens})
        d = pd.concat([frames[g].assign(gen=g) for g in gens], ignore_index=True)
        rng = np.random.default_rng(A.SEED + 2)
        gfe = (d["gen"] == gens[1]).to_numpy(float)
        n = d[d["source"] == "NONE"]
        res["pooled_C1"] = ols_cr1(n["margin"].to_numpy(), np.column_stack(
            [np.ones(len(n)), (n["size"] == 25).to_numpy(float), (n["gen"] == gens[1]).to_numpy(float)]),
            n["call_u"].to_numpy(), 1, rng)
        sg, s25 = (d["source"] == "steam_game").to_numpy(float), (d["size"] == 25).to_numpy(float)
        res["pooled_C2"] = ols_cr1(d["margin"].to_numpy(), np.column_stack(
            [np.ones(len(d)), sg, s25, sg * s25, gfe]), d["call_u"].to_numpy(), 3, rng)
    pw = json.loads((Path(__file__).resolve().parents[1] / "docs/prereg/strengthen_cli_powered.json").read_text())
    orig = json.loads((Path(__file__).resolve().parents[1] / "docs/prereg/revision_inspiration.json").read_text())
    res["earlier_runs"] = {g: {"x5_none": pw[g]["none_margin"],
                               "x5_steam_game": pw[g]["by_source_mean_margin"]["steam_game"],
                               "x25_steam_game_diff": orig["cli_clustered"][g]["comparisons"]["steam_game"]["diff"]}
                           for g in gens}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2])
    else:
        analyze(sys.argv[2:])
