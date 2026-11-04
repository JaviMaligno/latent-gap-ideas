"""Steering by iterative critique (docs/prereg/iterative_critique.md).

  ALIAS=gpt-4o-0806 uv run python scripts/iterative_critique.py generate [--limit N]
  uv run python scripts/iterative_critique.py analyse
Ideas: <run>/iter_ideas.jsonl (one line per idea and round). Results: key "iterative_critique" in
docs/prereg/continue_from_context_result.json
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import continue_from_context as M  # noqa: E402
from latent_gap.arxiv import item_text  # noqa: E402

OUT = M.RUN / os.environ.get("LG_ITER_OUT", "iter_ideas.jsonl")
ARMS = tuple(os.environ.get("LG_ITER_ARMS", "GEO,LLM,ORACLE").split(","))
N_IDEAS, N_ROUNDS = 3, 3
BAND = (0.574, 0.661)
TAG = os.environ.get("LG_TAG", "iter-critique")


def geo_feedback(s: float) -> str:
    head = f"Your proposal's similarity to the original paper is {s:.2f} (embedding cosine). Real follow-up papers typically sit between {BAND[0]:.2f} and {BAND[1]:.2f}. "
    if s > BAND[1]:
        return head + "You are too close: build on the paper but take a bigger step - a new question, method or setting, not a variation of the same one."
    if s < BAND[0]:
        return head + "You are too far: keep a clearer link to the original paper."
    return head + "You are within the range: keep the step, sharpen the contribution."


REVIEW = ("Here is a paper and a proposed follow-up to it.\n\nPAPER:\n{ctx}\n\nPROPOSED FOLLOW-UP:\nTitle: {title}\nAbstract: {abstract}\n\n"
          "Is this a genuine follow-up that moves the research forward, or mostly a rephrasing or minor variation of the "
          "original paper? In at most 80 words, say what would make it a stronger next paper. Answer with JSON {{\"review\": ...}}.")


def revise_prompt(ctx_block: str, prev: dict, feedback: str) -> str:
    return (M.prompt("CTX", ctx_block, [], []) + f"\n\nYour previous proposal was:\nTitle: {prev['title']}\nAbstract: {prev['abstract']}\n\n"
            f"Feedback: {feedback}\n\nWrite a revised proposal: a full new title and abstract.")


def generate(limit: int | None) -> None:
    from latent_gap.embedder import LocalQwen
    from latent_gap.gateway import Gateway
    gw = Gateway(ledger=M.DATA / "ledger.jsonl", model=os.environ["ALIAS"])
    q = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4")))
    meta, _ = M.corpus()
    by_id = meta.set_index("id")
    z = np.load(M.RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]  # loaded once, before threading
    ctx = M.contexts()[: limit or None]
    prior = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    done = {(r["ctx"], r["arm"], r["k"], r["round"]): r for r in prior}
    lock = threading.Lock()

    def chat(text: str, key: str) -> dict:
        for _ in range(3):
            raw = gw.chat([{"role": "system", "content": M.SYSTEM if key == "title" else "Answer with JSON."},
                           {"role": "user", "content": text}], max_tokens=600, json_mode=True, tag=TAG,
                          extra={"temperature": 1.0, **json.loads(os.environ.get("LG_EXTRA", "{}"))})
            try:
                d = json.loads(raw)
                if key == "title" and d.get("title") and d.get("abstract"):
                    return {"title": d["title"], "abstract": d["abstract"]}
                if key == "review" and d.get("review"):
                    return {"review": d["review"]}
            except json.JSONDecodeError:
                pass
        return {"title": "", "abstract": ""} if key == "title" else {"review": ""}

    def job(i: int, arm: str, k: int) -> None:
        c = ctx[i]
        cb = M.paper_block(by_id.loc[c["arxiv_id"], "name"], by_id.loc[c["arxiv_id"], "abstract"])
        Fi = F[owner == i]
        prev, feedback, history = None, "", []
        for r in range(N_ROUNDS):
            if (i, arm, k, r) in done:
                rec = done[(i, arm, k, r)]
            else:
                idea = chat(M.prompt("CTX", cb, [], []) if r == 0 else revise_prompt(cb, prev, feedback), "title")
                e = q.embed_one(item_text(idea["title"], idea["abstract"])) if idea["abstract"] else None
                step = float(e @ C[i]) if e is not None else float("nan")
                fit = float((Fi @ e).max()) if e is not None else float("nan")
                rec = {"ctx": i, "arm": arm, "k": k, "round": r, **idea, "feedback_in": feedback, "step": step, "fit": fit}
                with lock, OUT.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            prev = rec
            if rec["abstract"]:
                history.append(rec)
            if arm == "ORACLE-H" and history:
                prev = max(history, key=lambda x: x["fit"])  # revise from the best proposal so far
            if r < N_ROUNDS - 1 and rec["abstract"]:
                nxt = (i, arm, k, r + 1)
                if nxt in done:
                    feedback = done[nxt]["feedback_in"]
                elif arm == "GEO":
                    feedback = geo_feedback(rec["step"])
                elif arm == "ORACLE-H":
                    best = max(range(len(history)), key=lambda j: history[j]["fit"])
                    lines = "\n".join(f"{j + 1}. \"{h['title']}\": {h['fit']:.2f}" for j, h in enumerate(history))
                    feedback = ("Your proposals so far and their similarity (embedding cosine) to the closest paper actually "
                                f"published as a follow-up:\n{lines}\nThe best so far is proposal {best + 1}; the proposal "
                                "above is that one. Try to get closer.")
                elif arm == "ORACLE":
                    feedback = (f"Your proposal's similarity to the closest paper actually published as a follow-up is "
                                f"{rec['fit']:.2f} (embedding cosine). Try to get closer.")
                else:
                    feedback = chat(REVIEW.format(ctx=cb, title=rec["title"], abstract=rec["abstract"]), "review")["review"]
        M.log(f"ctx {i} {arm} idea {k} done")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda a: job(*a), [(i, a, k) for i in range(len(ctx)) for a in ARMS for k in range(N_IDEAS)]))
    M.log("generation done")


def analyse() -> None:
    from latent_gap.embedder import LocalQwen
    meta, emb = M.corpus()
    past = (meta["split"] == "past").to_numpy()
    fut = meta[~past].reset_index(drop=True)
    Fe = emb[~past]
    by_id = meta.set_index("id")
    z = np.load(M.RUN / "emb.npz")
    C, F, owner = z["C"], z["F"], z["owner"]
    ctx = M.contexts()
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    rows = [r for r in rows if r["abstract"]]
    q = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6")))
    I = q.embed([item_text(r["title"], r["abstract"]) for r in rows])  # re-embedded for the hit metric
    # Part B CTX ideas k = 0..2 as the one-shot reference
    pb = [json.loads(l) for l in (M.RUN / "ideas.jsonl").read_text().splitlines()]
    pb_idx = [n for n, r in enumerate([r for r in pb if r["abstract"]]) if r["arm"] == "CTX" and r["k"] < N_IDEAS]
    pbI = np.load(M.RUN / "idea_emb.npy")
    pbr = [r for r in pb if r["abstract"]]
    step_f = np.array([np.median(F[owner == i] @ C[i]) for i in range(len(ctx))])
    per = {(a, r): {"step": [], "fit": [], "hit": []} for a in ARMS for r in range(N_ROUNDS)}
    ref = {"step": [], "fit": [], "hit": []}
    for i, c in enumerate(ctx):
        Fi = F[owner == i]
        fids = {f["arxiv_id"] for f in c["followups"]}
        Dv = Fe[((fut["primary"] == by_id.loc[c["arxiv_id"], "primary"]) & ~fut["id"].isin(fids)).to_numpy()]

        def metrics(V):
            sf, sd = V @ Fi.T, V @ Dv.T
            return float((V @ C[i]).mean()), float(sf.max(1).mean()), float((sf.max(1) > sd.max(1)).mean())
        for a in ARMS:
            for r in range(N_ROUNDS):
                idx = [n for n, x in enumerate(rows) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                s, f, h = metrics(I[idx]) if idx else (np.nan,) * 3
                per[(a, r)]["step"].append(s); per[(a, r)]["fit"].append(f); per[(a, r)]["hit"].append(h)
        idx = [n for n in pb_idx if pbr[n]["ctx"] == i]
        s, f, h = metrics(pbI[idx])
        ref["step"].append(s); ref["fit"].append(f); ref["hit"].append(h)
    P = {k: {m: np.array(v) for m, v in d.items()} for k, d in per.items()}
    rng = np.random.default_rng(M.SEED)
    boot = lambda x: [float(np.nanpercentile([np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(M.N_BOOT)], q)) for q in (2.5, 97.5)]
    res = {"followup_step_median": float(np.median(step_f)), "partB_CTX_k0_2": {m: float(np.nanmean(v)) for m, v in ref.items()}}
    for a in ARMS:
        res[a] = {f"round{r}": {m: float(np.nanmean(P[(a, r)][m])) for m in ("step", "fit", "hit")} for r in range(N_ROUNDS)}
        res[a]["step_gap_round0"] = float(np.nanmean(P[(a, 0)]["step"] - step_f))
        res[a]["step_gap_round2"] = float(np.nanmean(P[(a, 2)]["step"] - step_f))
    p = {}
    d1 = P[("GEO", 0)]["step"] - P[("GEO", 2)]["step"]
    p["C1"] = stats.wilcoxon(d1, alternative="greater", nan_policy="omit").pvalue
    res["C1"] = {"step_reduction": float(np.nanmean(d1)), "ci": boot(d1), "p": float(p["C1"]),
                 "gap_closed": res["GEO"]["step_gap_round0"] - res["GEO"]["step_gap_round2"],
                 "holds": bool(np.nanmean(d1) > 0 and (res["GEO"]["step_gap_round0"] - res["GEO"]["step_gap_round2"]) >= 0.07)}
    d2 = P[("GEO", 2)]["hit"] - P[("GEO", 0)]["hit"]
    p["C2"] = stats.wilcoxon(d2 + 0.03, alternative="greater", nan_policy="omit").pvalue
    res["C2"] = {"hit_change": float(np.nanmean(d2)), "ci": boot(d2), "p_noninferiority": float(p["C2"]), "holds": bool(np.nanmean(d2) >= -0.03)}
    d3 = P[("LLM", 0)]["step"] - P[("LLM", 2)]["step"]
    p["C3"] = stats.wilcoxon(d3, alternative="greater", nan_policy="omit").pvalue
    res["C3"] = {"step_reduction": float(np.nanmean(d3)), "ci": boot(d3), "p": float(p["C3"])}
    d4 = P[("LLM", 2)]["hit"] - P[("LLM", 0)]["hit"]
    res["C4"] = {"hit_change": float(np.nanmean(d4)), "ci": boot(d4)}
    d5 = P[("ORACLE", 2)]["fit"] - P[("ORACLE", 0)]["fit"]
    p["C5"] = stats.wilcoxon(d5, alternative="greater", nan_policy="omit").pvalue
    res["C5"] = {"fit_gain": float(np.nanmean(d5)), "ci": boot(d5), "p": float(p["C5"]), "below_007": bool(np.nanmean(d5) < 0.07)}
    keys = sorted(p, key=p.get)
    holm, run = {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        holm[k] = run
    res["holm"] = holm
    M.save("iterative_critique", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        generate(lim)
    else:
        analyse()
