"""Call size and the NONE baseline across 8 sessions (pre-registered in docs/prereg/strengthen_cli_sessions.md).
    uv run python scripts/strengthen_cli_sessions.py gen codex      # 8 sessions, 2 h apart, subscription CLI
    uv run python scripts/strengthen_cli_sessions.py gen claude
    OMP_NUM_THREADS=2 uv run python scripts/strengthen_cli_sessions.py analyze codex claude
Output: runs/inspiration/<gen>-clisessions/{ideas,attempts}.jsonl, docs/prereg/strengthen_cli_sessions.json
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
import strengthen_cli_callsize as CS  # noqa: E402
import strengthen_cli_powered as P  # noqa: E402

N_SESSIONS, GAP_S, SEED0 = 8, 2 * 3600, 20261110
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/strengthen_cli_sessions.json"


def session_jobs(k: int) -> list[tuple[int, int]]:
    js = [5] * 16 + [25] * 4
    order = np.random.default_rng(SEED0 + k).permutation(len(js))
    return [(int(c), js[c]) for c in order]


def run_dir(g):
    return A.DATA / "runs/inspiration" / f"{g}-clisessions"


def done_calls(out):
    return {(r["session"], r["call"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()


def gen(g: str) -> None:
    run = run_dir(g)
    run.mkdir(parents=True, exist_ok=True)
    out, log = run / "ideas.jsonl", run / "attempts.jsonl"
    for k in range(N_SESSIONS):
        js = session_jobs(k)
        if all((k, c) in done_calls(out) for c, _ in js):
            continue
        if k > 0:
            last = max(json.loads(l)["time_end"] for l in log.read_text().splitlines() if json.loads(l)["session"] == k - 1)
            wait = time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%S")) + GAP_S - time.time()
            if wait > 0:
                print(f"{g} session {k}: waiting {wait / 60:.0f} min", flush=True)
                time.sleep(wait)
        for pas in range(2):
            todo = [(c, s) for c, s in js if (k, c) not in done_calls(out)]
            for c, size in todo:
                got = None
                for att in range(3):
                    t0, status = time.time(), "ok"
                    try:
                        raw = P.ask(g, CS.prompt(None, "NONE", size, list(range(size))))
                        x = {int(r["id"]): r for r in json.loads(re.search(r"\[.*\]", raw, re.S).group(0))}
                        if set(x) >= set(range(size)):
                            got = x
                        else:
                            status = f"missing_ids:{sorted(set(range(size)) - set(x))}"
                    except Exception as e:  # noqa: BLE001
                        status = "error:" + repr(e)[:120]
                    with log.open("a") as fh:
                        fh.write(json.dumps({"session": k, "call": c, "size": size, "pass": pas, "attempt": att,
                                             "status": status, "seconds": round(time.time() - t0),
                                             "time_end": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                    if got is not None:
                        break
                    time.sleep(5)
                if got is None:
                    print(f"{g} s{k} call {c} FAILED", flush=True)
                    continue
                with out.open("a") as fh:
                    for i in range(size):
                        fh.write(json.dumps({"session": k, "call": c, "size": size, "k": i, "model": P.MODELS[g],
                                             **{f: str(got[i].get(f, "")) for f in P.FIELDS}}) + "\n")
                print(f"{g} {time.strftime('%H:%M:%S')} s{k} call {c} x{size}", flush=True)
                time.sleep(2)
        print(f"{g} session {k} done", flush=True)


# ---------------------------------------------------------------- analysis

def wald(b, V, idx, G):
    R = np.zeros((len(idx), len(b)))
    R[np.arange(len(idx)), idx] = 1
    rb = R @ b
    W = float(rb @ np.linalg.pinv(R @ V @ R.T) @ rb)
    q = len(idx)
    F = W / q
    return {"F": F, "df": [q, G - 1], "p": float(stats.f.sf(F, q, G - 1))}


def analyze(gens):
    meta = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    E = A.norm(np.load(A.DATA / "prepared/steam/emb_qwen3e06.npy")[keep])
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    P_, Fc = E[past], E[(~past) & (appid > A.APPID_CUT)]
    del E
    from latent_gap.embedder import LocalQwen
    from latent_gap.steam import item_text
    enc = None
    res = {"prereg": "docs/prereg/strengthen_cli_sessions.md", "models": {g: P.MODELS[g] for g in gens}}
    frames = {}
    for g in gens:
        run = run_dir(g)
        d = pd.DataFrame([json.loads(l) for l in (run / "ideas.jsonl").read_text().splitlines()])
        n_written = int(d.groupby(["session", "call"]).ngroups)
        d = d[d["title"] != ""].reset_index(drop=True)
        cache = run / "ideas_emb_qwen3e06.npy"
        if cache.exists() and len(np.load(cache)) == len(d):
            V = A.norm(np.load(cache))
        else:
            if enc is None:
                enc = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "2")))
            V = A.norm(enc.embed([item_text(a, b, c) for a, b, c in
                                  zip(d["genres"], d["tags"], d["short_description"])]))
            np.save(cache, V)
        rng = np.random.default_rng(A.SEED)
        fut = A.maxcos(V, Fc)
        d["margin"] = np.mean([fut - A.maxcos(V, P_[rng.choice(len(P_), len(Fc), replace=False)])
                               for _ in range(A.B)], 0)
        d["cl"] = d["session"].astype(str) + "-" + d["call"].astype(str)
        d.to_parquet(run / "scored_qwen3e06.parquet")
        frames[g] = d
        srng = np.random.default_rng(A.SEED + 1)
        y = d["margin"].to_numpy()
        s25 = (d["size"] == 25).to_numpy(float)
        S = pd.get_dummies(d["session"], prefix="s", drop_first=True).to_numpy(float)
        X = np.column_stack([np.ones(len(d)), s25, S])
        cl = d["cl"].to_numpy()
        s1 = CS.ols_cr1(y, X, cl, 1, srng)
        b, Vc, G_ = P.cr1(y, X, cl)
        s2 = wald(b, Vc, list(range(2, 2 + S.shape[1])), G_)
        XI = np.column_stack([X, S * s25[:, None]])
        bi, Vi, Gi = P.cr1(y, XI, cl)
        inter = wald(bi, Vi, list(range(X.shape[1], XI.shape[1])), Gi)
        # cluster bootstrap within session x size
        groups = {key: [x["margin"].to_numpy() for _, x in gg.groupby("cl")] for key, gg in d.groupby(["session", "size"])}
        effs, sds = [], []
        for _ in range(4000):
            m = {key: np.concatenate([v[i] for i in srng.integers(0, len(v), len(v))]).mean() for key, v in groups.items()}
            effs.append(np.mean([m[(k, 25)] - m[(k, 5)] for k in range(N_SESSIONS) if (k, 25) in m and (k, 5) in m]))
            sds.append(np.std([m[(k, 5)] for k in range(N_SESSIONS) if (k, 5) in m], ddof=1))
        sess = d[d["size"] == 5].groupby("session")["margin"].mean()
        att = pd.DataFrame([json.loads(l) for l in (run / "attempts.jsonl").read_text().splitlines()])
        res[g] = {"n_ideas": int(len(d)), "n_calls_written": n_written, "n_calls_missing": N_SESSIONS * 20 - n_written,
                  "S1_size25_effect": s1 | {"effect_ci95_cluster_boot": np.percentile(effs, [2.5, 97.5]).tolist()},
                  "S2_session_effects": s2 | {"sd_session_means_5": float(sess.std(ddof=1)),
                                              "sd_ci95_cluster_boot": np.percentile(sds, [2.5, 97.5]).tolist()},
                  "size_x_session": inter,
                  "session_means_5": sess.round(5).to_dict(),
                  "size_effect_by_session": {int(k): float(x[x["size"] == 25]["margin"].mean()
                                                           - x[x["size"] == 5]["margin"].mean())
                                             for k, x in d.groupby("session")},
                  "attempts": {"total": int(len(att)), "non_ok": int((att["status"] != "ok").sum())}}
        print(g, json.dumps({k: res[g][k] for k in ("S1_size25_effect", "S2_session_effects", "size_x_session")},
                            indent=1, default=float), flush=True)
    if len(gens) == 2:
        res["holm2_S1"] = A.holm({g: res[g]["S1_size25_effect"]["p_two_cr1"] for g in gens})
        res["holm2_S2"] = A.holm({g: res[g]["S2_session_effects"]["p"] for g in gens})
        d = pd.concat([frames[g].assign(gen=g) for g in gens], ignore_index=True)
        d["gs"] = d["gen"] + "-" + d["session"].astype(str)
        X = np.column_stack([np.ones(len(d)), (d["size"] == 25).to_numpy(float),
                             pd.get_dummies(d["gs"], drop_first=True).to_numpy(float)])
        res["pooled_S1"] = CS.ols_cr1(d["margin"].to_numpy(), X, (d["gen"] + "-" + d["cl"]).to_numpy(), 1,
                                      np.random.default_rng(A.SEED + 2))
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2])
    else:
        analyze(sys.argv[2:])
