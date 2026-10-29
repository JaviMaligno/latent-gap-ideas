"""Patents pre-registration checks (3)-(6), docs/design.md. Run from the repo with uv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from geometry import DATA, load_run, unit  # noqa: E402

D = DATA / "runs" / "patents"
root = DATA / "prepared" / "patents"
meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
emb = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
keep = ~meta["adult"].to_numpy()
P = unit(emb[keep][(meta["split"][keep] == "past").to_numpy()])
tg = np.load(D / "pilot" / "targets.npz", allow_pickle=True)
rng = np.random.default_rng(0)
GUIDED = ("REAL", "INTERP", "NOISE", "CROWD", "FAR", "GAP")


def boot(a, b, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), np.quantile(d, 0.025), np.quantile(d, 0.975)


def medpair(E):
    S = E @ E.T
    return float(np.median(S[np.triu_indices(len(E), 1)]))


for run in ("grid-gpt-6-luna", "grid-claude-sonnet-5-5"):
    print("=====", run)
    df = load_run(D / run)
    # (3) anchoring: decoded closer to the shown neighbours' centroid than to the target
    for a in ("FAR", "GAP"):
        g = df[df.arm == a]
        t = np.stack([tg[a][k] for k in g.k])
        E = np.stack(g.e.to_list())
        C = unit(np.stack([P[nn[:8]].mean(0) for nn in g.nn]))
        frac = float(((E * C).sum(1) > (E * t).sum(1)).mean())
        print(f"(3) {a}: closer to centroid than target in {frac:.0%} of {len(g)}")
    # (5) diversity: median pairwise cosine of decoded items per arm
    items = [json.loads(l) for l in (D / run / "items.jsonl").read_text().splitlines()]
    div = {}
    for a in GUIDED + ("PLAIN", "PLAIND", "PAIR"):
        E = unit(np.array([r["best"]["emb"] for r in items if r["arm"] == a and r.get("best")], np.float32))
        div[a] = medpair(E)
    print("(5) median pairwise cos:", {k: round(v, 3) for k, v in div.items()})
    # (4) and (6): judged novelty, mean over the 4 judges per item
    J = pd.concat([pd.read_json(f, lines=True) for f in (D / run).glob("judged_*.jsonl")])
    nov = J.groupby(["arm", "k"]).novelty.mean().reset_index()
    by = {a: nov[nov.arm == a].novelty.to_numpy() for a in nov.arm.unique()}
    d, lo, hi = boot(by["GAP"], by["INTERP"])
    print(f"(4) GAP - INTERP novelty: {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    for a in GUIDED:
        d, lo, hi = boot(by["PLAIND"], by[a])
        print(f"(6) PLAIND - {a:6s}: {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
