"""Continuing from a given paper: the geometry of the next step (docs/prereg/continue_from_context.md).

  uv run python scripts/continue_from_context.py embed      # local Qwen embeddings of contexts and follow-ups
  uv run python scripts/continue_from_context.py part-a     # free, local
  uv run python scripts/continue_from_context.py generate   # paid: gpt-4o-0806 through the gateway (ALIAS env)
  uv run python scripts/continue_from_context.py part-b
Results: docs/prereg/continue_from_context_result.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latent_gap.arxiv import item_text  # noqa: E402

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
# LG_RUN / LG_FUPS select another context set (e.g. the 2025 contexts of the modern-generator test)
RUN = DATA / "runs" / os.environ.get("LG_RUN", "continue-from-context")
FUPS = DATA / "runs" / os.environ.get("LG_FUPS", "followups") / "followups.jsonl"
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/continue_from_context_result.json"
SEED = 20261030
N_CTX, MIN_FUPS, K_NB, N_IDEAS, N_DRAWS, N_BOOT = 150, 5, 50, 5, 2000, 2000
RECENT = ("2021", "2022")
ARMS = tuple(os.environ.get("LG_ARMS", "CTX,CTX+NB,CTX+MEM,CTX+MOM").split(","))


def log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def corpus():
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet",
                           columns=["id", "name", "abstract", "date", "split", "primary", "adult"])
    emb = np.load(DATA / "prepared/arxiv/emb_qwen3e06.npy").astype(np.float32)
    keep = ~meta["adult"].to_numpy()
    meta, emb = meta[keep].reset_index(drop=True), norm(emb[keep])
    return meta, emb


def contexts() -> list[dict]:
    rows = [json.loads(l) for l in FUPS.read_text().splitlines()]
    sel = [r for r in rows if r["n_followups"] >= MIN_FUPS][:N_CTX]  # file order = seeded sampling order
    if len(sel) < N_CTX:
        raise SystemExit(f"only {len(sel)} contexts with >= {MIN_FUPS} follow-ups")
    return sel


def embed() -> None:
    from latent_gap.embedder import LocalQwen
    meta, _ = corpus()
    by_id = meta.set_index("id")
    ctx = contexts()
    q = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6")))
    RUN.mkdir(parents=True, exist_ok=True)
    C = q.embed([item_text(by_id.loc[c["arxiv_id"], "name"], by_id.loc[c["arxiv_id"], "abstract"]) for c in ctx])
    F, owner = [], []
    for i, c in enumerate(ctx):
        F.append(q.embed([item_text(f["title"], f["abstract"]) for f in c["followups"]]))
        owner += [i] * len(c["followups"])
        if i % 10 == 0:
            log(f"embedded follow-ups of {i + 1}/{len(ctx)}")
    np.savez(RUN / "emb.npz", C=C, F=np.concatenate(F), owner=np.array(owner))
    log("embed done")


def neighbourhood(c: np.ndarray, P: np.ndarray, pdate: np.ndarray):
    s = P @ c
    nb = np.argpartition(-s, K_NB + 1)[:K_NB + 1]
    nb = nb[np.argsort(-s[nb])]
    return nb[s[nb] < 0.995][:K_NB]  # the context itself is in the past corpus


def directions(c, P, pdate, mu_hat, rng):
    nb = neighbourhood(c, P, pdate)
    X = P[nb]
    recent = np.isin(pdate[nb], RECENT)
    mom = X[recent].mean(0) - X[~recent].mean(0) if recent.any() and (~recent).any() else np.zeros_like(c)
    m = X.mean(0)
    Z = rng.standard_normal((N_DRAWS, len(X))) / np.sqrt(len(X) - 1)
    D = norm(m + Z @ (X - m))
    iso = np.max(D @ P.T, axis=1)
    hole = D[np.argmin(iso)] - c
    centre = mu_hat - c
    return nb, recent, {"momentum": mom, "hole": hole, "centre": centre}


def cosv(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def save(key: str, value) -> None:
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[key] = value
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def part_a() -> None:
    meta, emb = corpus()
    past = (meta["split"] == "past").to_numpy()
    P, pdate = emb[past], meta.loc[past, "date"].str[:4].to_numpy()
    mu_hat = norm(P.mean(0))
    z = np.load(RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    rng = np.random.default_rng(SEED)
    steps, proj = [], {k: [] for k in ("momentum", "hole", "centre")}
    for i, c in enumerate(C):
        _, _, dirs = directions(c, P, pdate, mu_hat, rng)
        Fi = F[owner == i]
        steps.append(float(np.median(Fi @ c)))
        for k, v in dirs.items():
            proj[k].append(float(np.nanmean([cosv(f - c, v) for f in Fi])))
    steps, pm, ph, pc = (np.array(x) for x in (steps, proj["momentum"], proj["hole"], proj["centre"]))
    diff = pm - ph
    ok = ~np.isnan(diff)
    w = stats.wilcoxon(diff[ok], alternative="greater")
    boot = lambda x: [float(np.nanpercentile([np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]
    save("A", {
        "n_contexts": int(len(C)), "n_followups": int(len(F)),
        "A1_step_median": float(np.median(steps)), "A1_step_quartiles": [float(np.percentile(steps, q)) for q in (25, 75)],
        "A2_mean_proj_momentum": float(np.nanmean(pm)), "A2_mean_proj_hole": float(np.nanmean(ph)),
        "A2_mean_diff": float(np.nanmean(diff)), "A2_diff_ci": boot(diff), "A2_wilcoxon_p": float(w.pvalue),
        "A2_holds": bool(np.nanmean(diff) > 0 and w.pvalue < 0.05),
        "A3_mean_proj_centre": float(np.nanmean(pc)), "A3_ci": boot(pc),
        "A3_wilcoxon_p_two_sided": float(stats.wilcoxon(pc[~np.isnan(pc)]).pvalue),
    })
    log("part A done")


SYSTEM = "You are a researcher proposing the next paper. Answer with JSON {\"title\": ..., \"abstract\": ...}."


def paper_block(title: str, abstract: str) -> str:
    return f"Title: {title}\nAbstract: {abstract[:1200]}"


def prompt(arm: str, ctx_block: str, extra: list[str], previous: list[str]) -> str:
    p = ("Here is a paper:\n\n" + ctx_block + "\n\nPropose the paper that should come next, building on this one: "
         "an original follow-up a research group could write. Give its title and an arXiv-style abstract of "
         "120-180 words.")
    if arm == "CTX+NB":
        p += "\n\nThe closest existing papers to it are:\n\n" + "\n\n".join(extra)
    if arm == "CTX+MOM":
        p += "\n\nThe most recent papers in the same area, showing where it is moving, are:\n\n" + "\n\n".join(extra)
    if arm == "CTX+MEM" and previous:
        p += "\n\nYou already proposed these; propose something different from all of them:\n" + "\n".join(f"- {t}" for t in previous)
    return p


def generate() -> None:
    from latent_gap.gateway import Gateway
    alias = os.environ["ALIAS"]
    gw = Gateway(ledger=DATA / "ledger.jsonl", model=alias)
    meta, emb = corpus()
    by_id = meta.set_index("id")
    past = (meta["split"] == "past").to_numpy()
    P, pm = emb[past], meta[past].reset_index(drop=True)
    pdate = pm["date"].str[:4].to_numpy()
    Cm = np.load(RUN / "emb.npz")["C"]  # loaded once: NpzFile is not safe to read from several threads
    ctx = contexts()
    out = RUN / "ideas.jsonl"
    prior = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    done = {(r["ctx"], r["arm"], r["k"]) for r in prior}
    titles = {}
    for r in sorted(prior, key=lambda r: r["k"]):
        titles.setdefault((r["ctx"], r["arm"]), []).append(r["title"])
    lock = __import__("threading").Lock()

    def call(text: str) -> dict:
        for _ in range(3):
            raw = gw.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], max_tokens=600,
                          json_mode=True, tag=os.environ.get("LG_TAG", "continue-ctx"),
                          extra={"temperature": 1.0, **json.loads(os.environ.get("LG_EXTRA", "{}"))})
            try:
                d = json.loads(raw)
                if d.get("title") and d.get("abstract"):
                    return {"title": d["title"], "abstract": d["abstract"]}
            except json.JSONDecodeError:
                pass
        return {"title": "", "abstract": ""}

    def write(rec: dict) -> None:
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def job(i: int, arm: str) -> None:
        c = ctx[i]
        cb = paper_block(by_id.loc[c["arxiv_id"], "name"], by_id.loc[c["arxiv_id"], "abstract"])
        nb = neighbourhood(Cm[i], P, pdate)
        nb = [j for j in nb if pm.loc[j, "id"] != c["arxiv_id"]]
        if arm == "CTX+NB":
            shown = nb[:8]
        elif arm == "CTX+MOM":
            shown = sorted(nb, key=lambda j: pm.loc[j, "date"], reverse=True)[:8]
        else:
            shown = []
        extra = [paper_block(pm.loc[j, "name"], pm.loc[j, "abstract"]) for j in shown]
        previous = list(titles.get((i, arm), []))  # resume: the MEM arm sees what it already proposed
        for k in range(N_IDEAS):
            if (i, arm, k) in done:
                continue
            idea = call(prompt(arm, cb, extra, previous if arm == "CTX+MEM" else []))
            previous.append(idea["title"])
            write({"ctx": i, "arm": arm, "k": k, "shown": [int(j) for j in shown], **idea})
        log(f"ctx {i} {arm} done")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda a: job(*a), [(i, arm) for i in range(len(ctx)) for arm in ARMS]))
    log("generation done")


def part_b() -> None:
    from latent_gap.embedder import LocalQwen
    meta, emb = corpus()
    past = (meta["split"] == "past").to_numpy()
    P = emb[past]
    fut = meta[~past].reset_index(drop=True)
    Fe = emb[~past]
    by_id = meta.set_index("id")
    z = np.load(RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ctx = contexts()
    ideas = [json.loads(l) for l in (RUN / "ideas.jsonl").read_text().splitlines()]
    ideas = [r for r in ideas if r["abstract"]]
    cache = RUN / "idea_emb.npy"
    if cache.exists() and len(np.load(cache)) == len(ideas):
        I = np.load(cache)
    else:
        q = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6")))
        I = q.embed([item_text(r["title"], r["abstract"]) for r in ideas])
        np.save(cache, I)
    rng = np.random.default_rng(SEED)
    per = {a: {"step_gap": [], "hit": [], "fit": [], "div": [], "nb_vs_fup": []} for a in ARMS}
    for i, c in enumerate(ctx):
        Fi = F[owner == i]
        step_f = np.median(Fi @ C[i])
        prim = by_id.loc[c["arxiv_id"], "primary"]
        fup_ids = {f["arxiv_id"] for f in c["followups"]}
        dm = ((fut["primary"] == prim) & ~fut["id"].isin(fup_ids)).to_numpy()
        Dv = Fe[dm]
        for a in ARMS:
            idx = [n for n, r in enumerate(ideas) if r["ctx"] == i and r["arm"] == a]
            if not idx:
                continue
            Ii = I[idx]
            per[a]["step_gap"].append(float(np.median(Ii @ C[i]) - step_f))
            sf, sd = Ii @ Fi.T, Ii @ Dv.T
            per[a]["fit"].append(float(sf.max(1).mean()))
            per[a]["hit"].append(float((sf.max(1) > sd.max(1)).mean()))
            S = Ii @ Ii.T
            per[a]["div"].append(float(S[np.triu_indices(len(Ii), 1)].mean()))
            if a == "CTX+NB":
                shown = ideas[idx[0]]["shown"]
                cn = norm(P[shown].mean(0))
                cf = norm(Fi.mean(0))
                per[a]["nb_vs_fup"].append(float((Ii @ cn).mean() - (Ii @ cf).mean()))
    A = {a: {k: np.array(v) for k, v in d.items()} for a, d in per.items()}
    boot = lambda x: [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]
    res = {a: {"n": int(len(A[a]["hit"])), **{f"mean_{k}": float(v.mean()) for k, v in A[a].items() if len(v)}} for a in ARMS}
    p = {}
    b1 = {a: stats.wilcoxon(A[a]["step_gap"], alternative="greater") for a in ("CTX", "CTX+NB")}
    p["B1"] = max(v.pvalue for v in b1.values())
    res["B1"] = {a: {"mean_step_gap": float(A[a]["step_gap"].mean()), "ci": boot(A[a]["step_gap"]), "p": float(v.pvalue)} for a, v in b1.items()}
    w2 = stats.wilcoxon(A["CTX+NB"]["nb_vs_fup"], alternative="greater")
    p["B2"] = w2.pvalue
    res["B2"] = {"mean_diff": float(A["CTX+NB"]["nb_vs_fup"].mean()), "ci": boot(A["CTX+NB"]["nb_vs_fup"]), "p": float(w2.pvalue)}
    n = min(len(A["CTX"]["div"]), len(A["CTX+MEM"]["div"]))
    w3 = stats.wilcoxon(A["CTX+MEM"]["div"][:n], A["CTX"]["div"][:n], alternative="less")
    p["B3"] = w3.pvalue
    res["B3"] = {"mean_pairwise_cos_MEM": float(A["CTX+MEM"]["div"].mean()), "mean_pairwise_cos_CTX": float(A["CTX"]["div"].mean()), "p": float(w3.pvalue)}
    d4 = A["CTX+MOM"]["hit"] - A["CTX"]["hit"]
    maj_m, maj_c = A["CTX+MOM"]["hit"] > 0.5, A["CTX"]["hit"] > 0.5
    b, cc = int((maj_m & ~maj_c).sum()), int((~maj_m & maj_c).sum())
    p_mc = float(stats.binomtest(b, b + cc, 0.5, alternative="greater").pvalue) if b + cc else 1.0
    p["B4"] = p_mc
    res["B4"] = {"hit_MOM": float(A["CTX+MOM"]["hit"].mean()), "hit_CTX": float(A["CTX"]["hit"].mean()),
                 "mean_diff": float(d4.mean()), "ci": boot(d4), "mcnemar_b": b, "mcnemar_c": cc, "p": p_mc,
                 "fit_MOM": float(A["CTX+MOM"]["fit"].mean()), "fit_CTX": float(A["CTX"]["fit"].mean()),
                 "best_arm_by_hit": max(ARMS, key=lambda a: A[a]["hit"].mean())}
    a_res = json.loads(OUT.read_text()).get("A", {}) if OUT.exists() else {}
    if a_res:
        p["A2"] = a_res["A2_wilcoxon_p"]
    keys = sorted(p, key=p.get)
    holm, run = {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        holm[k] = run
    res["holm"] = holm
    save("B", res)
    log("part B done")


if __name__ == "__main__":
    {"embed": embed, "part-a": part_a, "generate": generate, "part-b": part_b}[sys.argv[1]]()
