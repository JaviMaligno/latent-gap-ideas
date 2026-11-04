"""NEUTRAL revision arm vs the critique arms (docs/prereg/neutral_revision.md; review point 16).

  OMP_NUM_THREADS=2 uv run python scripts/revision_neutral.py embed-qwen      # local ONNX Qwen3 -> neutral_emb_qwen3e06.npy
  uv run --no-project --with sentence-transformers python scripts/revision_neutral.py embed-bge   # -> neutral_emb_bgem3.npy
  uv run python scripts/revision_neutral.py analyse

Existing arms (GEO, LLM, ORACLE) use their Kaggle vectors and the Kaggle context / follow-up vectors, exactly as
direction_tangent.py and continuation_reanalysis.py; NEUTRAL uses local vectors of the same encoder for all its
rounds (incl. its round 0, which is GEO's round-0 text), so its round-0 -> round-2 change is within one encoder.
Encoder agreement local vs Kaggle is measured on those shared round-0 texts and reported.
Measures: step, fit, hit (Table 2 / iterative_critique.analyse) and tangent alignment, change along the nearest
non-citing later papers' direction p and along the follow-up-specific direction s (Table 3 / direction_tangent.py).
Output: docs/prereg/revision_neutral.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/continue-from-context"
NEU = RUN / "iter_ideas_neutral.jsonl"
OUT = ROOT / "docs/prereg/revision_neutral.json"
ARMS = ("GEO", "LLM", "ORACLE")
SEED, N_BOOT = 20261030, 2000


def neutral_rows() -> list[dict]:
    rows = [json.loads(l) for l in NEU.read_text().splitlines()]
    rows = [r for r in rows if r["abstract"]]
    return sorted(rows, key=lambda r: (r["ctx"], r["k"], r["round"]))


def texts(rows):
    sys.path.insert(0, str(ROOT / "src"))
    from latent_gap.arxiv import item_text
    return [item_text(r["title"], r["abstract"]) for r in rows]


def embed_qwen() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from latent_gap.embedder import LocalQwen
    rows = neutral_rows()
    V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "2"))).embed(texts(rows))
    np.save(RUN / "neutral_emb_qwen3e06.npy", V.astype(np.float32))
    print("qwen", V.shape, flush=True)


def embed_bge() -> None:
    """Same settings as kaggle/embed-continuation/embed.py (sentence-transformers, max_seq_length 512, normalised);
    CPU float32 here instead of T4 float16."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rows = neutral_rows()
    m = SentenceTransformer("BAAI/bge-m3", device="cpu")
    m.max_seq_length = 512
    T = texts(rows)
    out = []
    for b in range(0, len(T), 64):
        out.append(m.encode(T[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        print(f"bge {min(b + 64, len(T))}/{len(T)}", flush=True)
    np.save(RUN / "neutral_emb_bgem3.npy", np.concatenate(out).astype(np.float32))


def load_gpt4o(R, space):
    """Kaggle vectors of contexts, follow-ups and the GEO/LLM/ORACLE ideas, matched by the texts.parquet key
    "ctx/arm/k/round" (iter_ideas.jsonl now also holds ORACLE-H rows, which the Kaggle run did not embed, so
    continuation_reanalysis.load_generator's positional assert no longer applies). Text equality is checked."""
    import pandas as pd
    ctx = R.M.contexts()
    t = pd.read_parquet(DATA / "kaggle-upload/continuation-texts/texts.parquet")
    V = R.norm(np.load(DATA / f"kaggle-output/embed-continuation-v1/emb_{space}.npy").astype(np.float32))
    kinds = t["kind"].to_numpy()
    C, Fall = V[kinds == "ctx"], V[kinds == "fup"]
    pos = {k: n for n, k in enumerate(t["key"]) if kinds[n] == "iter"}
    rows = [json.loads(l) for l in (RUN / "iter_ideas.jsonl").read_text().splitlines()]
    it = [r for r in rows if r["abstract"] and r["arm"] in ARMS]
    T = texts(it)
    idx = [pos[f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"] for r in it]
    assert all(t["text"].iloc[j] == T[n] for n, j in enumerate(idx)), "text mismatch"
    nf = [len(c["followups"]) for c in ctx]
    starts = np.cumsum([0] + nf)
    F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]
    return ctx, C, F, it, V[idx]


def analyse() -> None:
    import pandas as pd
    from scipy import stats
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import continuation_reanalysis as R
    from direction_tangent import tang, unit

    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "primary", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut_t = (meta["split"] == "future").to_numpy()      # pool of direction_tangent.py
    notpast = (meta["split"] != "past").to_numpy()      # distractor set of iterative_critique.analyse
    ids_t, ids_h = meta.loc[fut_t, "id"].to_numpy(), meta.loc[notpast, "id"].to_numpy()
    prim_h = meta.loc[notpast, "primary"].to_numpy()
    by_id = meta.set_index("id")
    nrows = neutral_rows()
    rng = np.random.default_rng(SEED)

    def boot(x):
        return [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)], q)) for q in (2.5, 97.5)]

    res = {"n_neutral_ideas": len(nrows), "n_neutral_rows_total": sum(1 for _ in NEU.read_text().splitlines())}
    for space in R.SPACES:
        vp = RUN / f"neutral_emb_{space}.npy"
        if not vp.exists():
            res[space] = {"skipped": f"{vp.name} missing"}
            continue
        NV = R.norm(np.load(vp).astype(np.float32))
        assert len(NV) == len(nrows)
        emb = R.norm(np.load(DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep])
        Ft, Fh = emb[fut_t], emb[notpast]
        ctx, C, F, it, IT = load_gpt4o(R, space)
        # encoder agreement on the shared round-0 texts (NEUTRAL r0 = GEO r0)
        geo0 = {(x["ctx"], x["k"]): IT[n] for n, x in enumerate(it) if x["arm"] == "GEO" and x["round"] == 0}
        agree = [float(NV[n] @ geo0[(r["ctx"], r["k"])]) for n, r in enumerate(nrows) if r["round"] == 0 and (r["ctx"], r["k"]) in geo0]
        allarms = ARMS + ("NEUTRAL",)
        M = {a: {m: np.full((len(ctx), 3), np.nan) for m in ("step", "fit", "hit", "align", "p", "s")} for a in allarms}
        for i in range(len(ctx)):
            c, Fi = C[i], F[i]
            fids = {f["arxiv_id"] for f in ctx[i]["followups"]}
            Dv = Fh[(prim_h == by_id.loc[ctx[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, list(fids))]
            t = unit(tang(Fi, c).mean(0))
            pool = Ft[~np.isin(ids_t, list(fids))]
            tp = unit(tang(pool[np.argsort(-(pool @ c))[:len(Fi)]], c).mean(0))
            ts = unit(t - (t @ tp) * tp)
            for a in allarms:
                src, V = (nrows, NV) if a == "NEUTRAL" else (it, IT)
                for r in range(3):
                    idx = [n for n, x in enumerate(src) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                    if not idx:
                        continue
                    X = V[idx]
                    sf = X @ Fi.T
                    T = tang(X, c)
                    M[a]["step"][i, r] = (X @ c).mean()
                    M[a]["fit"][i, r] = sf.max(1).mean()
                    M[a]["hit"][i, r] = (sf.max(1) > (X @ Dv.T).max(1)).mean()
                    M[a]["align"][i, r] = (T @ t).mean()
                    M[a]["p"][i, r] = (T @ tp).mean()
                    M[a]["s"][i, r] = (T @ ts).mean()
        cell = {"encoder_agreement_r0_local_vs_kaggle": {"min": float(np.min(agree)), "mean": float(np.mean(agree)), "n": len(agree)}}
        D = {}
        for a in allarms:
            e = {f"round{r}": {m: float(np.nanmean(M[a][m][:, r])) for m in M[a]} for r in range(3)}
            D[a] = {}
            for m in M[a]:
                d = M[a][m][:, 0] - M[a][m][:, 2] if m == "step" else M[a][m][:, 2] - M[a][m][:, 0]
                D[a][m] = d
                ok = ~np.isnan(d)
                e[f"change_{m}" if m != "step" else "step_reduction"] = {
                    "mean": float(d[ok].mean()), "ci": boot(d[ok]), "p": float(stats.wilcoxon(d[ok]).pvalue), "n": int(ok.sum())}
            cell[a] = e
        tests, p = {}, {}
        spec = [("N1", "GEO", "step", "greater"), ("N2", "LLM", "step", "greater")]
        for nm, m in (("N3", "fit"), ("N4", "hit"), ("N5", "align"), ("N6", "s")):
            spec += [(nm, a, m, "two-sided") for a in ARMS]
        spec_explore = [("X_p", a, "p", "two-sided") for a in ARMS] + [("X_step", "ORACLE", "step", "two-sided"), ("X_step_LLM2", "LLM", "step", "two-sided")]
        for fam, rows_ in (("prereg", spec), ("exploratory", spec_explore)):
            for nm, a, m, alt in rows_:
                d = D[a][m] - D["NEUTRAL"][m]
                ok = ~np.isnan(d)
                w = stats.wilcoxon(d[ok], alternative=alt)
                key = f"{nm}:{a}-NEUTRAL:{m}"
                tests[key] = {"family": fam, "diff_mean": float(d[ok].mean()), "ci": boot(d[ok]), "p": float(w.pvalue),
                              "alternative": alt, "n": int(ok.sum())}
                if fam == "prereg":
                    p[key] = float(w.pvalue)
        keys = sorted(p, key=p.get)
        run = 0.0
        for j, k in enumerate(keys):
            run = max(run, min(1.0, p[k] * (len(keys) - j)))
            tests[k]["p_holm"] = run
        cell["tests"] = tests
        cell["note"] = ("step: reduction r0 - r2 (positive = longer step); other measures r2 - r0. Diffs are critique "
                        "change minus NEUTRAL change, paired by context. Holm over the 14 pre-registered tests within "
                        "this embedder (Qwen3 primary; BGE-M3 robustness).")
        res[space] = cell
        print(space, json.dumps({a: {k: (v["mean"] if isinstance(v, dict) and "mean" in v else None) for k, v in cell[a].items() if not k.startswith("round")} for a in allarms}), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"embed-qwen": embed_qwen, "embed-bge": embed_bge, "analyse": analyse}[sys.argv[1]]()
