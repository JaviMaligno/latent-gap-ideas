"""Robustness re-analysis of continue_from_context (Part B) and iterative_critique (exploratory, not pre-registered).

Grid: generator {gpt-4o, llama-3.1-8b} x space {qwen3e06, bgem3} x follow-ups {all, continuing}.
- Every vector comes from the Kaggle sentence-transformers encoders, the same settings as the corpus (no local ONNX).
- "continuing" follow-ups: Semantic Scholar marks the citation as influential, or its intents include methodology or
  result (runs/continue-from-context/followup_intents.json). Contexts with fewer than MIN_CONT such follow-ups are
  dropped from that cell; the cell reports its n.
- Measures as before: step (cos to context), fit (max cos to follow-ups), diversity (mean pairwise cos within a
  context), and the distance-matched excess fit against the nearest-step later paper of the corpus (with replacement).
The gpt-4o / qwen3e06 / all cell must reproduce the pre-registered numbers up to the encoder difference.
Output: docs/prereg/continuation_reanalysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402

D = M.DATA
OUT = Path(__file__).resolve().parents[1] / "docs/prereg/continuation_reanalysis.json"
MIN_CONT = 3
SPACES = ("qwen3e06", "bgem3")
B_ARMS = ("CTX", "CTX+NB", "CTX+MEM", "CTX+MOM")
I_ARMS = ("GEO", "LLM", "ORACLE")


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_generator(gen: str, space: str):
    """Context vectors, follow-up vectors per context, Part B ideas and iterative ideas with their vectors."""
    ctx = M.contexts()
    nf = [len(c["followups"]) for c in ctx]
    if gen == "gpt-4o":
        t = pd.read_parquet(D / "kaggle-upload/continuation-texts/texts.parquet")
        V = norm(np.load(D / f"kaggle-output/embed-continuation-v1/emb_{space}.npy").astype(np.float32))
        kinds = t["kind"].to_numpy()
        pb = [json.loads(l) for l in (M.RUN / "ideas.jsonl").read_text().splitlines()]
        it = [json.loads(l) for l in (M.RUN / "iter_ideas.jsonl").read_text().splitlines()]
        C, Fall = V[kinds == "ctx"], V[kinds == "fup"]
        PB, IT = V[kinds == "partB"], V[kinds == "iter"]
    else:
        K = D / "kaggle-output/llama-continuation-v1"
        V = norm(np.load(K / f"emb_{space}.npy").astype(np.float32))
        pb = [json.loads(l) for l in (K / "ideas_llama.jsonl").read_text().splitlines()]
        it = [json.loads(l) for l in (K / "iter_llama.jsonl").read_text().splitlines()]
        n_pb = sum(1 for r in pb if r["abstract"])
        o = 0
        C = V[o:o + len(ctx)]; o += len(ctx)
        Fall = V[o:o + sum(nf)]; o += sum(nf)
        PB = V[o:o + n_pb]; o += n_pb
        IT = V[o:]
    pb = [r for r in pb if r["abstract"]]
    it = [r for r in it if r["abstract"]]
    if gen == "gpt-4o":  # oracle_history.py later appended ORACLE-H rows, embedded separately
        it = [r for r in it if r.get("arm") != "ORACLE-H"]
    assert len(pb) == len(PB) and len(it) == len(IT), (gen, len(pb), len(PB), len(it), len(IT))
    starts = np.cumsum([0] + nf)
    F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]
    return ctx, C, F, pb, PB, it, IT


def main() -> None:
    meta = pd.read_parquet(D / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut_mask = (meta["split"] == "future").to_numpy()
    fut_ids = meta.loc[fut_mask, "id"].to_numpy()
    intents = json.loads((M.RUN / "followup_intents.json").read_text())
    res = {}
    for space in SPACES:
        Fe = norm(np.load(D / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep][fut_mask])
        for gen in ("gpt-4o", "olmo-7b-0424"):
            try:
                ctx, C, F, pb, PB, it, IT = load_generator(gen, space)
            except FileNotFoundError as e:
                print("skip", gen, space, e, flush=True)
                continue
            for fset in ("all", "continuing"):
                rows = {"B": {a: {"step_gap": [], "fit": [], "div": []} for a in B_ARMS},
                        "I": {(a, r): {"step": [], "fit": [], "excess": []} for a in I_ARMS for r in range(3)}}
                n_used = 0
                for i, c in enumerate(ctx):
                    sel = np.arange(len(c["followups"]))
                    if fset == "continuing":
                        sel = np.array([j for j in sel if (v := intents.get(f"{i}/{j}", {})) and
                                        (v.get("influential") or {"methodology", "result"} & set(v.get("intents", [])))], dtype=int)
                        if len(sel) < MIN_CONT:
                            continue
                    n_used += 1
                    Fi = F[i][sel]
                    step_f = float(np.median(Fi @ C[i]))
                    for a in B_ARMS:
                        idx = [n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == a]
                        if not idx:
                            continue
                        X = PB[idx]
                        rows["B"][a]["step_gap"].append(float(np.median(X @ C[i]) - step_f))
                        rows["B"][a]["fit"].append(float((X @ Fi.T).max(1).mean()))
                        S = X @ X.T
                        rows["B"][a]["div"].append(float(S[np.triu_indices(len(X), 1)].mean()))
                    fids = {f["arxiv_id"] for f in c["followups"]}
                    pool = Fe[~np.isin(fut_ids, list(fids))]
                    s_all = pool @ C[i]
                    order = np.argsort(s_all)
                    ss = s_all[order]
                    for a in I_ARMS:
                        for r in range(3):
                            idx = [n for n, x in enumerate(it) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                            if not idx:
                                continue
                            X = IT[idx]
                            st = X @ C[i]
                            ft = (X @ Fi.T).max(1)
                            j = np.clip(np.searchsorted(ss, st), 1, len(ss) - 1)
                            jj = np.where(np.abs(ss[j - 1] - st) < np.abs(ss[j] - st), j - 1, j)
                            mf = (pool[order[jj]] @ Fi.T).max(1)
                            rows["I"][(a, r)]["step"].append(float(st.mean()))
                            rows["I"][(a, r)]["fit"].append(float(ft.mean()))
                            rows["I"][(a, r)]["excess"].append(float((ft - mf).mean()))
                cell = {"n_contexts": n_used}
                B = {a: {k: np.array(v) for k, v in d.items()} for a, d in rows["B"].items()}
                cell["B"] = {a: {f"mean_{k}": float(v.mean()) for k, v in d.items() if len(v)} for a, d in B.items()}
                cell["B1_step_gap_CTX_p"] = float(stats.wilcoxon(B["CTX"]["step_gap"]).pvalue) if len(B["CTX"]["step_gap"]) else None
                n = min(len(B["CTX"]["div"]), len(B["CTX+MEM"]["div"]))
                cell["B3_div_MEM_vs_CTX_p"] = float(stats.wilcoxon(B["CTX+MEM"]["div"][:n], B["CTX"]["div"][:n]).pvalue) if n else None
                I = {k: {m: np.array(v) for m, v in d.items()} for k, d in rows["I"].items()}
                cell["I"] = {}
                for a in I_ARMS:
                    e = {f"round{r}": {m: float(I[(a, r)][m].mean()) for m in ("step", "fit", "excess")} for r in range(3) if len(I[(a, r)]["step"])}
                    if len(I[(a, 0)]["step"]) and len(I[(a, 2)]["step"]):
                        d_step = I[(a, 0)]["step"] - I[(a, 2)]["step"]
                        d_fit = I[(a, 2)]["fit"] - I[(a, 0)]["fit"]
                        d_ex = I[(a, 2)]["excess"] - I[(a, 0)]["excess"]
                        e["step_reduction"] = float(d_step.mean()); e["step_reduction_p"] = float(stats.wilcoxon(d_step).pvalue)
                        e["fit_change"] = float(d_fit.mean()); e["fit_change_p"] = float(stats.wilcoxon(d_fit).pvalue)
                        e["excess_change"] = float(d_ex.mean()); e["excess_change_p"] = float(stats.wilcoxon(d_ex).pvalue)
                        e["excess_round0_p"] = float(stats.wilcoxon(I[(a, 0)]["excess"]).pvalue)
                    cell["I"][a] = e
                key = f"{gen}/{space}/{fset}"
                res[key] = cell
                OUT.write_text(json.dumps(res, indent=1))
                print(key, "n", n_used, "B1 gap %.3f" % cell["B"]["CTX"]["mean_step_gap"],
                      "GEO step -%.3f fit %+.3f excess r0 %+.3f change %+.3f" % (
                          cell["I"]["GEO"].get("step_reduction", np.nan), cell["I"]["GEO"].get("fit_change", np.nan),
                          cell["I"]["GEO"].get("round0", {}).get("excess", np.nan), cell["I"]["GEO"].get("excess_change", np.nan)), flush=True)


if __name__ == "__main__":
    main()
