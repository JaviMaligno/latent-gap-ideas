"""Powered re-run of the Section 6 CLI stress test (pre-registered in docs/prereg/strengthen_cli_powered.md).

Same stimuli and prompt as scripts/inspiration_cli_generate.py, but 5 ideas per call (90 stimulus calls + 20 NONE
calls per generator) in a fixed random call order, with the model pinned on the command line.
    uv run python scripts/strengthen_cli_powered.py gen codex      # subscription CLI, no API spend
    uv run python scripts/strengthen_cli_powered.py gen claude
    OMP_NUM_THREADS=2 uv run python scripts/strengthen_cli_powered.py analyze codex claude   # local Qwen3 ONNX
Output: runs/inspiration/<gen>-cli5/ideas.jsonl, docs/prereg/strengthen_cli_powered.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspiration_analysis as A  # noqa: E402
import inspiration_generate as G  # noqa: E402

PER_CELL, BATCH, N_NONE_CALLS, ORDER_SEED = 10, 5, 20, 20261105
FIELDS = ("title", "short_description", "genres", "tags")
FORMAT = ("Answer with only a JSON list of objects, one per item, each with the keys \"id\", \"title\", "
          "\"short_description\" (at most 60 words, like a Steam store blurb), \"genres\" (comma-separated Steam "
          "genres) and \"tags\" (5-10 comma-separated Steam user tags). Do not use any tools.")
MODELS = {"codex": "gpt-6.1-sol", "claude": "claude-opus-5-5"}
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/strengthen_cli_powered.json"


def ask(gen: str, text: str) -> str:
    cwd = tempfile.mkdtemp(prefix="insp5-")
    if gen == "codex":
        with tempfile.NamedTemporaryFile("r", suffix=".txt") as f:
            subprocess.run(["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
                            "-m", MODELS[gen], "-o", f.name, text],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=900, check=True, cwd=cwd)
            return Path(f.name).read_text()
    out = subprocess.run(["claude", "-p", text, "--model", MODELS[gen], "--output-format", "text",
                          "--disallowed-tools", "Bash,Edit,Write,Read,WebSearch,WebFetch,Glob,Grep,Agent"],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900, check=True, cwd=cwd)
    return out.stdout


def jobs(st: pd.DataFrame) -> list[tuple[int, str, list[int]]]:
    picks = [i for _, g in st.groupby(["source", "quintile"]) for i in g.index[:PER_CELL]]
    js = [("INSTRUCTED", [int(x) for x in picks[b:b + BATCH]]) for b in range(0, len(picks), BATCH)]
    js += [("NONE", list(range(BATCH)))] * N_NONE_CALLS
    order = np.random.default_rng(ORDER_SEED).permutation(len(js))
    return [(int(c), *js[c]) for c in order]  # call id = index in the unshuffled list


def prompt(st, arm, ids):
    if arm == "NONE":
        return f"Propose {BATCH} original video games that could be released on Steam, ids 0-{BATCH - 1}. {FORMAT}"
    body = "\n\n".join(f"Item {k}. Here is something you happened to read today:\n\n"
                       f"{G.stimulus_block(st.loc[i, 'title'], st.loc[i, 'text'])}" for k, i in enumerate(ids))
    return ("For each item below, propose an original video game that could be released on Steam, using what "
            "you read in that item as inspiration for the game. Treat each item independently.\n\n"
            f"{body}\n\n{FORMAT}")


def gen(g: str) -> None:
    st = pd.read_parquet(G.INS / "stimuli.parquet")
    out = G.DATA / "runs/inspiration" / f"{g}-cli5" / "ideas.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    js = jobs(st)
    for pas in range(2):
        done = {json.loads(l)["call"] for l in out.read_text().splitlines()} if out.exists() else set()
        todo = [j for j in js if j[0] not in done]
        print(f"{g} pass {pas}: {len(todo)} calls to do", flush=True)
        for n, (call, arm, ids) in enumerate(todo):
            got = None
            for _ in range(3):
                try:
                    raw = ask(g, prompt(st, arm, ids))
                    x = {int(r["id"]): r for r in json.loads(re.search(r"\[.*\]", raw, re.S).group(0))}
                    if set(x) >= set(range(len(ids))):
                        got = x
                        break
                except Exception as e:  # noqa: BLE001
                    print("retry", call, repr(e)[:120], flush=True)
                time.sleep(5)
            if got is None:
                print(f"{g} call {call} FAILED", flush=True)
                continue
            with out.open("a") as fh:
                for k, i in enumerate(ids):
                    rec = {"call": call, "stim": int(i) if arm == "INSTRUCTED" else -1, "arm": arm,
                           "k": k, "model": MODELS[g], **{f: str(got[k].get(f, "")) for f in FIELDS}}
                    fh.write(json.dumps(rec) + "\n")
            print(f"{g} {time.strftime('%H:%M:%S')} {n + 1}/{len(todo)} call {call} {arm}", flush=True)
            time.sleep(2)


# ---------------------------------------------------------------- analysis

def cr1(y, X, cl):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    uc = np.unique(cl)
    meat = sum(np.outer(X[cl == c].T @ r[cl == c], X[cl == c].T @ r[cl == c]) for c in uc)
    G_, n, k = len(uc), len(y), np.linalg.matrix_rank(X)
    V = XtXi @ meat @ XtXi * G_ / (G_ - 1) * (n - 1) / (n - k)
    return b, V, G_


def slope_test(y, x, src, cl, rng, nboot=9999):
    X = np.column_stack([x, A.dummies(src, A.SOURCES)])
    b, V, G_ = cr1(y, X, cl)
    se = float(np.sqrt(V[0, 0]))
    t = b[0] / se
    X0 = X[:, 1:]
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    f0, r0 = X0 @ b0, y - X0 @ b0
    uc = np.unique(cl)
    idx = np.searchsorted(uc, cl)
    tb = []
    for _ in range(nboot):
        ys = f0 + rng.choice([-1.0, 1.0], len(uc))[idx] * r0
        bb, vv, _ = cr1(ys, X, cl)
        tb.append(abs(bb[0] / np.sqrt(vv[0, 0])))
    q = stats.t.ppf(0.975, G_ - 1)
    return {"slope": float(b[0]), "se_cr1": se, "n_clusters": int(G_), "ci95_cr1": [float(b[0] - q * se),
            float(b[0] + q * se)], "p_two_cr1": float(2 * stats.t.sf(abs(t), G_ - 1)),
            "p_two_wild_cluster_boot": float((np.array(tb) >= abs(t)).mean())}


def call_compare(d, s, rng):
    cm = d.groupby("call").agg(margin=("margin", "mean"), source=("source", "first"))
    hc, nc = cm[cm["source"] == s]["margin"], cm[cm["source"] == "NONE"]["margin"]
    h, none = d[d["source"] == s], d[d["source"] == "NONE"]
    obs = h["margin"].mean() - none["margin"].mean()
    gs = {c: g["margin"].to_numpy() for c, g in d.groupby("call")}
    sc, ncl = list(hc.index), list(nc.index)
    boots = [np.concatenate([gs[c] for c in rng.choice(sc, len(sc))]).mean()
             - np.concatenate([gs[c] for c in rng.choice(ncl, len(ncl))]).mean() for _ in range(4000)]
    allc = np.array(sc + ncl)
    perm = []
    for _ in range(20000):
        p = rng.permutation(allc)
        perm.append(np.concatenate([gs[c] for c in p[:len(sc)]]).mean()
                    - np.concatenate([gs[c] for c in p[len(sc):]]).mean())
    return {"diff": float(obs), "n_calls": len(sc), "n_none_calls": len(ncl), "n_ideas": int(len(h)),
            "call_mean_diff": float(hc.mean() - nc.mean()),
            "p_callmeans_mw_exact": float(stats.mannwhitneyu(hc, nc, method="exact").pvalue),
            "p_idea_mw": float(stats.mannwhitneyu(h["margin"], none["margin"]).pvalue),
            "p_perm_calls_two_sided": float((np.abs(perm) >= abs(obs) - 1e-12).mean()),
            "boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "call_means": [float(v) for v in hc]}


def analyze(gens: list[str]) -> None:
    st = pd.read_parquet(A.INS / "stimuli.parquet")
    meta = pd.read_parquet(A.DATA / "prepared/steam/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    E = A.norm(np.load(A.DATA / "prepared/steam/emb_qwen3e06.npy")[keep])
    past = (meta["split"] == "past").to_numpy()
    appid = pd.to_numeric(meta["id"], errors="coerce").to_numpy()
    P, Fc = E[past], E[(~past) & (appid > A.APPID_CUT)]
    del E
    from latent_gap.embedder import LocalQwen
    from latent_gap.steam import item_text
    enc = None
    res = {"prereg": "docs/prereg/strengthen_cli_powered.md", "models": {g: MODELS[g] for g in gens}}
    frames = {}
    for g in gens:
        rng = np.random.default_rng(A.SEED)
        run = A.DATA / "runs/inspiration" / f"{g}-cli5"
        d = pd.DataFrame([json.loads(l) for l in (run / "ideas.jsonl").read_text().splitlines()])
        n_calls_written = int(d["call"].nunique())
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
        fut = A.maxcos(V, Fc)
        d["margin"] = np.mean([fut - A.maxcos(V, P[rng.choice(len(P), len(Fc), replace=False)])
                               for _ in range(A.B)], 0)
        d["iso"] = 1 - A.maxcos(V, P)
        has = (d["stim"] >= 0).to_numpy()
        si = d.loc[has, "stim"].to_numpy()
        d["source"] = "NONE"
        d.loc[has, "source"] = st.loc[si, "source"].to_numpy()
        d.loc[has, "rank"] = st.loc[si, "rank"].to_numpy()
        d.to_parquet(run / "scored_qwen3e06.parquet")
        frames[g] = d
        srng = np.random.default_rng(A.SEED + 1)
        comps = {s: call_compare(d, s, srng) for s in A.SOURCES}
        gi = d[has]
        out = {"n_ideas": int(len(d)), "n_calls_written": n_calls_written,
               "n_calls_missing": 110 - n_calls_written,
               "none_margin": float(d.loc[~has, "margin"].mean()),
               "by_source_mean_margin": d.groupby("source")["margin"].mean().round(5).to_dict(),
               "H1_steam_game_vs_none": comps["steam_game"],
               "H2_margin_slope": slope_test(gi["margin"].to_numpy(), gi["rank"].to_numpy(float),
                                             gi["source"].to_numpy(), gi["call"].to_numpy(), srng),
               "iso_slope_P3": slope_test(gi["iso"].to_numpy(), gi["rank"].to_numpy(float),
                                          gi["source"].to_numpy(), gi["call"].to_numpy(), srng),
               "sources_vs_none": comps,
               "holm9_p_callmeans_mw_exact": A.holm({s: comps[s]["p_callmeans_mw_exact"] for s in A.SOURCES})}
        res[g] = out
        print(g, json.dumps({k: out[k] for k in ("n_ideas", "H1_steam_game_vs_none", "H2_margin_slope",
                                                  "iso_slope_P3")}, indent=1), flush=True)
    if len(gens) == 2:
        res["holm2_H1_p_callmeans_mw_exact"] = A.holm({g: res[g]["H1_steam_game_vs_none"]["p_callmeans_mw_exact"]
                                                      for g in gens})
        res["holm2_H2_p_two_cr1"] = A.holm({g: res[g]["H2_margin_slope"]["p_two_cr1"] for g in gens})
        # pooled, generator-stratified: average of within-generator gaps; permute call labels within generator
        prng = np.random.default_rng(A.SEED + 2)
        prep = {}
        for g in gens:
            d = frames[g][frames[g]["source"].isin(["steam_game", "NONE"])]
            prep[g] = ({c: x["margin"].to_numpy() for c, x in d.groupby("call")},
                       d.groupby("call")["source"].first())
        def gap(lab):
            return np.mean([np.concatenate([prep[g][0][c] for c in lab[g].index[lab[g] == "steam_game"]]).mean()
                            - np.concatenate([prep[g][0][c] for c in lab[g].index[lab[g] == "NONE"]]).mean()
                            for g in gens])
        obs = gap({g: prep[g][1] for g in gens})
        perm = [gap({g: pd.Series(prng.permutation(prep[g][1].to_numpy()), index=prep[g][1].index)
                     for g in gens}) for _ in range(20000)]
        res["pooled_steam_game_vs_none"] = {"diff": float(obs),
                                            "p_perm_calls_within_gen": float((np.abs(perm) >= abs(obs) - 1e-12).mean())}
    # replication check against the original 25-per-call run
    orig = json.loads((Path(__file__).resolve().parents[1] / "docs/prereg/revision_inspiration.json").read_text())
    res["original_25_per_call"] = {g: {"steam_game_diff": orig["cli_clustered"][g]["comparisons"]["steam_game"]["diff"],
                                       "margin_slope": orig["cli_clustered"][g]["margin_slope"]["slope"]}
                                   for g in gens}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2])
    else:
        analyze(sys.argv[2:])
