"""NEUTRAL arm for gpt-5.4-mini (docs/prereg/strengthen_modern_neutral.md).

  OMP_NUM_THREADS=4 uv run python scripts/strengthen_modern_neutral.py embed-qwen
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 uv run --no-project --with sentence-transformers==3.0.1 --with transformers==4.44.2 \
      --with "numpy<2" --with pandas --with pyarrow python scripts/strengthen_modern_neutral.py embed-bge
  uv run python scripts/strengthen_modern_neutral.py analyse

Existing arms: local Qwen3 vectors runs/continue-modern/oracle_history_emb_GEO_LLM_ORACLE_ORACLE-H.npy and BGE-M3
vectors of runs/strengthen-bge (strengthen_bge_replication.local). NEUTRAL: local vectors of the same encoders, all its
rounds (incl. round 0 = GEO's round-0 text; agreement with GEO's stored vector reported).
Direction pool and hit distractors as revision_null_effects.part_t4 / strengthen_bge_replication.oracle_like for the
2025 contexts: corpus papers dated on or after the earliest saved follow-up, not among the context's saved follow-ups;
hit distractors = that pool restricted to the context's primary category (no hit was defined for these contexts
before; this is the transfer of Section 5.2's rule to the 2025 date window).
Output: docs/prereg/strengthen_modern_neutral.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/continue-modern"
NEU = RUN / "iter_ideas_neutral.jsonl"
OUT = ROOT / "docs/prereg/strengthen_modern_neutral.json"
ARMS = ("GEO", "LLM", "ORACLE", "ORACLE-H")
SEED = 20261030


def rows_of(path, arms=None):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return [r for r in rows if r["abstract"] and (arms is None or r["arm"] in arms)]


def neutral_rows():
    return sorted(rows_of(NEU), key=lambda r: (r["ctx"], r["k"], r["round"]))


def texts(rows):
    sys.path.insert(0, str(ROOT / "src"))
    from latent_gap.arxiv import item_text
    return [item_text(r["title"], r["abstract"]) for r in rows]


def embed_qwen():
    sys.path.insert(0, str(ROOT / "src"))
    from latent_gap.embedder import LocalQwen
    V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4"))).embed(texts(neutral_rows()))
    np.save(RUN / "neutral_emb_qwen3e06.npy", V.astype(np.float32))
    print("qwen", V.shape, flush=True)


def embed_bge():
    """Settings of strengthen_bge_embed.py (sentence-transformers, max_seq_length 512, normalised)."""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = os.environ.get("LG_BGE_DEVICE", "cpu")  # MPS ran out of memory next to a running generation (2026-11-06)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    m = SentenceTransformer("BAAI/bge-m3", device=dev)
    m.max_seq_length = 512
    T, out = texts(neutral_rows()), []
    for b in range(0, len(T), 64):
        out.append(m.encode(T[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        print(f"bge {min(b + 64, len(T))}/{len(T)}", flush=True)
    np.save(RUN / "neutral_emb_bgem3.npy", np.concatenate(out).astype(np.float32))


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def analyse():
    import pandas as pd
    import neutral_contrasts as N
    import revision_null_effects as RN
    import strengthen_bge_replication as SB

    ctx = RN.contexts("followups-2025")
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "date", "primary", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    later = (meta["date"] >= min(f["date"] for c in ctx for f in c["followups"])).to_numpy()
    ids, prim = meta.loc[later, "id"].to_numpy(), meta.loc[later, "primary"].to_numpy()
    by_id = meta.set_index("id")
    nrows = neutral_rows()
    res = {"n_neutral_ideas": len(nrows), "n_neutral_rows_total": sum(1 for _ in NEU.read_text().splitlines()),
           "n_neutral_new_rows_empty": sum(1 for l in NEU.read_text().splitlines() if not json.loads(l)["abstract"])}
    for space in ("qwen3e06", "bgem3"):
        vp = RUN / f"neutral_emb_{space}.npy"
        if not vp.exists():
            res[space] = {"skipped": f"{vp.name} missing"}
            continue
        NV = norm(np.load(vp).astype(np.float32))
        assert len(NV) == len(nrows)
        it = rows_of(RUN / "iter_ideas.jsonl", ARMS)
        if space == "qwen3e06":
            z = np.load(RUN / "emb.npz")
            C, owner = z["C"], z["owner"]
            F = [z["F"][owner == i] for i in range(len(ctx))]
            IT = norm(np.load(RUN / "oracle_history_emb_GEO_LLM_ORACLE_ORACLE-H.npy").astype(np.float32))
        else:
            L, _ = SB.local()
            C = np.stack([L["modern/ctx", str(i)] for i in range(len(ctx))])
            F = [np.stack([L["modern/fup", f"{i}/{j}"] for j in range(len(c["followups"]))]) for i, c in enumerate(ctx)]
            IT = np.stack([L["modern/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"] for r in it])
        assert len(IT) == len(it)
        emb = norm(np.load(DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep])
        pool = emb[later]
        geo0 = {(x["ctx"], x["k"]): IT[n] for n, x in enumerate(it) if x["arm"] == "GEO" and x["round"] == 0}
        agree = [float(NV[n] @ geo0[(r["ctx"], r["k"])]) for n, r in enumerate(nrows) if r["round"] == 0 and (r["ctx"], r["k"]) in geo0]
        allarms = ARMS + ("NEUTRAL",)
        sources = {a: (it, IT) for a in ARMS} | {"NEUTRAL": (nrows, NV)}

        def distractors(i, fids, _p=pool):
            sel = (prim == by_id.loc[ctx[i]["arxiv_id"], "primary"]) & ~np.isin(ids, list(fids))
            return _p[sel]
        M, loo = N.per_context(ctx, C, F, sources, allarms, pool, ids, distractors)
        rng = np.random.default_rng(SEED)
        cell = {"encoder_agreement_r0_neutral_vs_geo": {"min": float(np.min(agree)), "mean": float(np.mean(agree)), "n": len(agree)},
                "followups_loo_alignment": float(np.nanmean(loo))}
        cell.update(N.summary(M, allarms, rng))
        tests, p = {}, {}
        spec = [("N1", "GEO", "step", "greater"), ("N2", "LLM", "step", "greater")]
        for nm, m in (("N3", "fit"), ("N4", "hit"), ("N5", "align"), ("N6", "s")):
            spec += [(nm, a, m, "two-sided") for a in ("GEO", "LLM", "ORACLE")]
        explore = [("X_p", a, "p", "two-sided") for a in ("GEO", "LLM", "ORACLE")] + \
                  [("X_H", "ORACLE-H", m, "two-sided") for m in ("step", "fit", "s")] + [("X_step", "ORACLE", "step", "two-sided")]
        for fam, rows_ in (("prereg", spec), ("exploratory", explore)):
            for nm, a, m, alt in rows_:
                d, t = N.contrast(M, a, m, rng, alt)
                key = f"{nm}:{a}-NEUTRAL:{m}"
                tests[key] = {"family": fam, **t}
                if nm == "N6" or (fam == "exploratory" and m == "s"):
                    tests[key]["tost_015"] = N.tost(d)
                if fam == "prereg":
                    p[key] = t["p"]
        for k, v in N.holm(p).items():
            tests[k]["p_holm"] = v
        cell["tests"] = tests
        cell["note"] = ("step: lengthening r0 - r2; other measures r2 - r0. Diffs = critique change minus NEUTRAL change, "
                        "paired by context. Holm over the 14 pre-registered tests within this embedder.")
        res[space] = cell
        print(space, json.dumps({k: (round(v["diff_mean"], 4), round(v.get("p_holm", v["p"]), 4)) for k, v in tests.items()}), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"embed-qwen": embed_qwen, "embed-bge": embed_bge, "analyse": analyse}[sys.argv[1]]()
