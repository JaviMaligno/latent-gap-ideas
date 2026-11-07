"""Confirmatory replication of the critique study with gpt-4o (docs/prereg/strengthen_critique_replication.md).

  uv run python scripts/strengthen_critique_replication.py build     # first 150 frame contexts with >= 5 follow-ups
  LG_RUN=critique-replication LG_FUPS=critique-replication uv run python scripts/continue_from_context.py embed
  (generation: scripts/iterative_critique.py, commands in the pre-registration)
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow python scripts/strengthen_critique_replication.py embed-bge
  uv run python scripts/strengthen_critique_replication.py analyse
  uv run python scripts/strengthen_critique_replication.py pooled    # secondary: pooled with the original 150
Output: docs/prereg/strengthen_critique_replication.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
FRAME = DATA / "runs/olmo-direction/followups.jsonl"
RUN = DATA / "runs/critique-replication"
OUT = ROOT / "docs/prereg/strengthen_critique_replication.json"
N_CTX, MIN_FUPS = 150, 5


def build() -> None:
    rows = sorted((json.loads(l) for l in FRAME.read_text().splitlines()), key=lambda r: r["frame_pos"])
    pos = [r["frame_pos"] for r in rows]
    assert pos == list(range(len(pos))), "frame has gaps"
    sel = [r for r in rows if r.get("n_followups", 0) >= MIN_FUPS][:N_CTX]
    assert len(sel) == N_CTX, f"only {len(sel)} contexts so far"
    RUN.mkdir(parents=True, exist_ok=True)
    out = RUN / "followups.jsonl"
    assert not out.exists(), "context set already built"
    with out.open("w") as fh:
        for r in sel:
            fh.write(json.dumps({"arxiv_id": r["arxiv_id"], "title": r["title"], "date": r["date"],
                                 "frame_pos": r["frame_pos"], "n_followups": r["n_followups"],
                                 "followups": r["followups"]}) + "\n")
    print("contexts", len(sel), "frame_pos up to", sel[-1]["frame_pos"], "follow-ups", sum(r["n_followups"] for r in sel))


ARMS = ("GEO", "LLM", "ORACLE")
SEED = 20261106


def ctx_rows():
    return [json.loads(l) for l in (RUN / "followups.jsonl").read_text().splitlines()]


def idea_rows():
    """Critique arms in file order, then NEUTRAL sorted (ctx, k, round); rows with an abstract only."""
    it = [json.loads(l) for l in (RUN / "iter_ideas.jsonl").read_text().splitlines()]
    nu = [json.loads(l) for l in (RUN / "iter_ideas_neutral.jsonl").read_text().splitlines()]
    it = [r for r in it if r["abstract"] and r["arm"] in ARMS]
    nu = sorted((r for r in nu if r["abstract"]), key=lambda r: (r["ctx"], r["k"], r["round"]))
    return it, nu


def all_texts():
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd
    from latent_gap.arxiv import item_text
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"]).set_index("id")
    ctx = ctx_rows()
    it, nu = idea_rows()
    T = {"ctx": [item_text(meta.loc[c["arxiv_id"], "name"], meta.loc[c["arxiv_id"], "abstract"]) for c in ctx],
         "fup": [item_text(f["title"], f["abstract"]) for c in ctx for f in c["followups"]],
         "ideas": [item_text(r["title"], r["abstract"]) for r in it + nu]}
    return T


def embed_qwen() -> None:
    """Ideas only (contexts and follow-ups are in emb.npz from continue_from_context.embed)."""
    sys.path.insert(0, str(ROOT / "src"))
    from latent_gap.embedder import LocalQwen
    V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4"))).embed(all_texts()["ideas"])
    np.save(RUN / "ideas_emb_qwen3e06.npy", V.astype(np.float32))
    print("qwen ideas", V.shape, flush=True)


def embed_bge() -> None:
    """Settings of strengthen_bge_embed.py (sentence-transformers, max_seq_length 512, normalised)."""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = os.environ.get("LG_BGE_DEVICE", "cpu")  # MPS ran out of memory next to a running generation (2026-11-06)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    m = SentenceTransformer("BAAI/bge-m3", device=dev)
    m.max_seq_length = 512
    T = all_texts()
    out = {}
    for k, xs in T.items():
        parts = []
        for b in range(0, len(xs), 64):
            parts.append(m.encode(xs[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
            print(f"bge {k} {min(b + 64, len(xs))}/{len(xs)}", flush=True)
        out[k] = np.concatenate(parts).astype(np.float32)
    np.savez(RUN / "emb_bgem3.npz", **out)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def analyse() -> None:
    import pandas as pd
    from scipy import stats
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import neutral_contrasts as N

    ctx = ctx_rows()
    nf = [len(c["followups"]) for c in ctx]
    starts = np.cumsum([0] + nf)
    it, nu = idea_rows()
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "primary", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()        # direction pool, as direction_tangent.py
    notpast = (meta["split"] != "past").to_numpy()      # hit distractors, as iterative_critique.analyse
    ids_t, ids_h = meta.loc[fut, "id"].to_numpy(), meta.loc[notpast, "id"].to_numpy()
    prim_h = meta.loc[notpast, "primary"].to_numpy()
    by_id = meta.set_index("id")
    citing = {}
    for l in (DATA / "runs/olmo-direction/citing_all.jsonl").read_text().splitlines():
        r = json.loads(l)
        citing.setdefault(r["arxiv_id"], set()).update(r.get("citing_arxiv_all") or [])
    res = {"n_contexts": len(ctx), "n_followups": int(sum(nf)),
           "n_rows": {a: sum(1 for r in it if r["arm"] == a) for a in ARMS} | {"NEUTRAL": len(nu)},
           "n_empty": {"critique": sum(1 for l in (RUN / "iter_ideas.jsonl").read_text().splitlines() if not json.loads(l)["abstract"]),
                       "neutral": sum(1 for l in (RUN / "iter_ideas_neutral.jsonl").read_text().splitlines() if not json.loads(l)["abstract"])}}
    allarms = ARMS + ("NEUTRAL",)
    for space in ("qwen3e06", "bgem3"):
        if space == "qwen3e06":
            vp = RUN / "ideas_emb_qwen3e06.npy"
            if not vp.exists():
                res[space] = {"skipped": "ideas_emb_qwen3e06.npy missing"}
                continue
            z = np.load(RUN / "emb.npz")
            C, Fall, V = norm(z["C"]), norm(z["F"]), norm(np.load(vp))
        else:
            vp = RUN / "emb_bgem3.npz"
            if not vp.exists():
                res[space] = {"skipped": "emb_bgem3.npz missing"}
                continue
            z = np.load(vp)
            C, Fall, V = norm(z["ctx"]), norm(z["fup"]), norm(z["ideas"])
        assert len(V) == len(it) + len(nu) and len(Fall) == starts[-1]
        F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]
        IT, NV = V[:len(it)], V[len(it):]
        emb = norm(np.load(DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep])
        Ft, Fh = emb[fut], emb[notpast]
        sources = {a: (it, IT) for a in ARMS} | {"NEUTRAL": (nu, NV)}

        def distractors(i, fids):
            return Fh[(prim_h == by_id.loc[ctx[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, list(fids))]
        rng = np.random.default_rng(SEED)
        M, loo = N.per_context(ctx, C, F, sources, allarms, Ft, ids_t, distractors)
        cell = {"followups_loo_alignment": float(np.nanmean(loo))}
        cell.update(N.summary(M, allarms, rng))
        conf, p = {}, {}
        for nm, a in (("R1", "GEO"), ("R2", "LLM")):
            _, t = N.contrast(M, a, "step", rng, "greater")
            conf[f"{nm}:{a}-NEUTRAL:step"] = t
            p[f"{nm}:{a}-NEUTRAL:step"] = t["p"]
        for j, a in enumerate(ARMS):
            d = N.change(M, a, "s")
            d = d[~np.isnan(d)]
            k = f"E{j + 1}:{a}:s_change"
            conf[k] = {"mean": float(d.mean()), "ci": N.boot(d, rng), "n": int(len(d)), "p_wilcoxon": float(stats.wilcoxon(d).pvalue), **N.tost(d)}
            p[k] = min(1.0, 2 * conf[k]["p_tost"])  # each one-sided test at 0.025 <=> 2·p_TOST at 0.05 in the Holm family
        for j, a in enumerate(ARMS):
            d, t = N.contrast(M, a, "s", rng)
            k = f"E{j + 4}:{a}-NEUTRAL:s"
            conf[k] = {**t, **N.tost(d)}
            p[k] = min(1.0, 2 * conf[k]["p_tost"])
        for k, v in N.holm(p).items():
            conf[k]["p_holm"] = v
        for k in conf:
            if k.startswith("E"):
                v = conf[k]
                lo, hi = v["t_ci95"]
                v["reading"] = ("within ±0.015 (confirmed)" if v["p_holm"] < 0.05 else
                                "turn detected" if (lo > 0 or hi < 0) else "insufficient")
        cell["confirmatory"] = conf
        sec = {}
        for nm, m in (("N3", "fit"), ("N4", "hit"), ("N5", "align"), ("X_p", "p")):
            for a in ARMS:
                _, t = N.contrast(M, a, m, rng)
                sec[f"{nm}:{a}-NEUTRAL:{m}"] = t
        g0, g2 = M["GEO"]["step"][:, 0], M["GEO"]["step"][:, 2]
        sf = np.array([np.median(F[i] @ C[i]) for i in range(len(ctx))])
        sec["C1_GEO_step_reduction"] = {"mean": float(np.nanmean(g0 - g2)), "gap_round0": float(np.nanmean(g0 - sf)),
                                        "gap_round2": float(np.nanmean(g2 - sf))}
        sec["C2_GEO_hit_change"] = float(np.nanmean(N.change(M, "GEO", "hit")))
        sec["C3_LLM_step_reduction"] = float(np.nanmean(N.change(M, "LLM", "step")))
        sec["C4_LLM_hit_change"] = float(np.nanmean(N.change(M, "LLM", "hit")))
        sec["C5_ORACLE_fit_gain"] = float(np.nanmean(N.change(M, "ORACLE", "fit")))
        if citing:
            M2, _ = N.per_context(ctx, C, F, sources, allarms, Ft, ids_t, distractors,
                                  pool_exclude=lambda i: citing.get(ctx[i]["arxiv_id"], set()))
            sec["variant_pool_excluding_all_citing"] = {
                a: {"s_change": float(np.nanmean(N.change(M2, a, "s")))} for a in allarms} | {
                f"{a}-NEUTRAL:s": N.contrast(M2, a, "s", rng)[1] for a in ARMS}
        cell["secondary"] = sec
        res[space] = cell
        print(space, json.dumps({k: (round(v.get("mean", v.get("diff_mean", 0)), 4), round(v["p_holm"], 4), v.get("reading", ""))
                                 for k, v in conf.items()}), flush=True)
    elig = DATA / "runs/olmo-direction/eligible.json"
    if elig.exists():
        e = set(json.loads(elig.read_text()))
        res["overlap_with_olmo_direction_eligible"] = sum(c["arxiv_id"] in e for c in ctx)
    OUT.write_text(json.dumps(res, indent=1))


def pooled() -> None:
    """Secondary (registered): pooled estimate over the original 150 contexts (revision_neutral.py, Kaggle vectors)
    and the 150 new ones (local vectors), per embedder: each arm's change along s and its contrast with NEUTRAL,
    per-context values concatenated (300 contexts), bootstrap CI and Wilcoxon. Added to the output's 'secondary'."""
    import pandas as pd
    from scipy import stats
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import continuation_reanalysis as R
    import neutral_contrasts as N
    import revision_neutral as RV

    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "primary", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    notpast = (meta["split"] != "past").to_numpy()
    ids_t, ids_h = meta.loc[fut, "id"].to_numpy(), meta.loc[notpast, "id"].to_numpy()
    prim_h = meta.loc[notpast, "primary"].to_numpy()
    by_id = meta.set_index("id")
    allarms = ARMS + ("NEUTRAL",)
    res = json.loads(OUT.read_text())
    ctx = ctx_rows()
    nf = [len(c["followups"]) for c in ctx]
    starts = np.cumsum([0] + nf)
    it, nu = idea_rows()
    for space in ("qwen3e06", "bgem3"):
        emb = norm(np.load(DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep])
        Ft, Fh = emb[fut], emb[notpast]
        Ms = []
        # original 150 (as revision_neutral.analyse)
        ctx0, C0, F0, it0, IT0 = RV.load_gpt4o(R, space)
        nrows0 = RV.neutral_rows()
        NV0 = norm(np.load(RV.RUN / f"neutral_emb_{space}.npy").astype(np.float32))
        def dis0(i, fids, _c=ctx0):
            return Fh[(prim_h == by_id.loc[_c[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, list(fids))]
        Ms.append(N.per_context(ctx0, C0, F0, {a: (it0, IT0) for a in ARMS} | {"NEUTRAL": (nrows0, NV0)},
                                allarms, Ft, ids_t, dis0)[0])
        # new 150 (as analyse)
        if space == "qwen3e06":
            z = np.load(RUN / "emb.npz")
            C, Fall, V = norm(z["C"]), norm(z["F"]), norm(np.load(RUN / "ideas_emb_qwen3e06.npy"))
        else:
            z = np.load(RUN / "emb_bgem3.npz")
            C, Fall, V = norm(z["ctx"]), norm(z["fup"]), norm(z["ideas"])
        F = [Fall[starts[i]:starts[i + 1]] for i in range(len(ctx))]
        def dis1(i, fids):
            return Fh[(prim_h == by_id.loc[ctx[i]["arxiv_id"], "primary"]) & ~np.isin(ids_h, list(fids))]
        Ms.append(N.per_context(ctx, C, F, {a: (it, V[:len(it)]) for a in ARMS} | {"NEUTRAL": (nu, V[len(it):])},
                                allarms, Ft, ids_t, dis1)[0])
        rng = np.random.default_rng(SEED)
        out = {"n_contexts": [len(ctx0), len(ctx)],
               "note": "original 150 (Kaggle vectors, Semantic Scholar-dated follow-ups) + new 150 (local vectors, arXiv v1-dated)"}
        def est(d):
            d = d[~np.isnan(d)]
            return {"mean": float(d.mean()), "ci": N.boot(d, rng), "p": float(stats.wilcoxon(d).pvalue), "n": int(len(d)), **N.tost(d)}
        for a in allarms:
            out[f"{a}:s_change"] = est(np.concatenate([N.change(M, a, "s") for M in Ms]))
        for a in ARMS:
            out[f"{a}-NEUTRAL:s"] = est(np.concatenate([N.change(M, a, "s") - N.change(M, "NEUTRAL", "s") for M in Ms]))
        res[space]["secondary"]["pooled_with_original_150"] = out
        print(space, json.dumps({k: [round(v["mean"], 4), [round(x, 4) for x in v["ci"]]] for k, v in out.items() if isinstance(v, dict)}), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"build": build, "embed-qwen": embed_qwen, "embed-bge": embed_bge, "analyse": analyse, "pooled": pooled}[sys.argv[1]]()
