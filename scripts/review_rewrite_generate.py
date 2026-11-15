"""Part B of docs/prereg/review_tangent_confounds.md (Section 11): rewrite and paraphrase items, generation and
embedding inputs. PAID for DISC and REP (gpt-4o through an API gateway); OLMO-C runs on Kaggle (free).

  uv run python scripts/review_rewrite_generate.py dry-run --study disc        # no call: 2 prompts per set + call count
  uv run python scripts/review_rewrite_generate.py prepare --study disc|rep|olmo   # step 3: nb_ids.json (+ OLMo sample)
  uv run python scripts/review_rewrite_generate.py smoke --study disc --confirm-paid     # step 4: 20 calls, parse rate
  uv run python scripts/review_rewrite_generate.py generate --study disc --confirm-paid  # step 5 (resumable)
  uv run python scripts/review_rewrite_generate.py kaggle-disc-inputs   # texts for kaggle/review-rewrite-embed (DISC)
  OMP_NUM_THREADS=4 uv run python scripts/review_rewrite_generate.py embed-rep-qwen      # REP, local ONNX Qwen3
  uv run --no-project --with sentence-transformers --with pandas --with pyarrow \
      python scripts/review_rewrite_generate.py embed-rep-bge                            # REP, BGE-M3 on CPU
  uv run python scripts/review_rewrite_generate.py kaggle-olmo-inputs   # inputs of kaggle/review-rewrite (OLMo)

Items (fixed by the registration): RW-F = every saved follow-up rewritten as a proposal (prompt R, context shown;
OLMO-C: the consensus continuations plus a seeded sample of up to 5 non-continuation follow-ups per context);
NA = 3 paraphrases of the context (prompt PA = style_probe.PROMPT); NB = 3 nearby later papers per context drawn from
ranks k+1..k+200 of the all-citing-removed future pool (prompt R). System prompt = continue_from_context.SYSTEM,
temperature 1.0, max_tokens 600, JSON mode, the ideas' parser, one retry on a parse failure, then dropped (counted).
Raw outputs: runs/review-rewrite/{disc,rep}/rewrites.jsonl (one row per item; empty title/abstract = dropped).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import continue_from_context as CFC  # noqa: E402
import style_probe as SP  # noqa: E402

DATA = CFC.DATA
OUTDIR = DATA / "runs/review-rewrite"
SEED, N_NA, N_NB, NB_BAND, OLMO_NONCONT = 20261114, 3, 3, 200, 5
ALIAS = "gpt-4o-0806"
MAX_TOKENS, TEMPERATURE, ATTEMPTS = 600, 1.0, 2  # one retry
TAGS = {"disc": "review-rewrite-disc", "rep": "review-rewrite-rep"}
NOUN, DOES = "arXiv computer-science paper", "what the method or study does"  # DOMAINS["arxiv"]
# per-call cost assumption of the registration (Section 11.5): ~720 in / ~230 out tokens at $2.50 / $10 per M
IN_TOK, OUT_TOK, PRICE_IN, PRICE_OUT, AUD = 720, 230, 2.50, 10.0, 1.52

PROMPT_R = ("Here is a paper:\n\n{ctx}\n\nBelow is a later paper. Rewrite it as the proposal for the paper that should "
            "come next, building on the paper above: give its title and an arXiv-style abstract of 120-180 words. Keep "
            "exactly its content: the same subject, method, claims and details. Add nothing, remove nothing, invent "
            "nothing; only the wording and the framing as a proposal change.\n\n{later}")


def prompt_pa(ctx_block: str) -> str:
    return SP.PROMPT.format(noun=NOUN, does=DOES, text=ctx_block)


def log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ------------------------------------------------------------------------------------------------ contexts per study
def study_contexts(study: str) -> list[dict]:
    """Context rows in the order of the Part A cells: arxiv_id, followups, k (pool size that builds p),
    citing (all citing ids, for the NB pool), consensus (continuation indices, OLMO-C and DISC labels)."""
    if study == "disc":
        ctx = CFC.contexts()
        cache = DATA / "runs/followups/complete_citations"
        out = []
        lab = {}
        for p in json.loads((ROOT / "docs/prereg/strengthen_continuations_full_labels.json").read_text()):
            if p["gpt61sol"] == p["opus55"] == "CONTINUES":
                lab.setdefault(p["ctx_index"], []).append(p["followup_index"])
        for i, c in enumerate(ctx):
            d = json.loads((cache / f"{c['arxiv_id'].replace('/', '_')}.json").read_text())
            out.append({"arxiv_id": c["arxiv_id"], "followups": c["followups"], "k": len(c["followups"]),
                        "citing": {x["arxiv_id"] for x in d["citing"]} | {f["arxiv_id"] for f in c["followups"]},
                        "consensus": sorted(lab.get(i, []))})
        return out
    citing = {}
    for l in (DATA / "runs/olmo-direction/citing_all.jsonl").read_text().splitlines():
        r = json.loads(l)
        citing.setdefault(r["arxiv_id"], set()).update(r.get("citing_arxiv_all") or [])
    if study == "rep":
        import strengthen_critique_replication as SCR
        return [{"arxiv_id": c["arxiv_id"], "followups": c["followups"], "k": len(c["followups"]),
                 "citing": citing.get(c["arxiv_id"], set()) | {f["arxiv_id"] for f in c["followups"]}, "consensus": []}
                for c in SCR.ctx_rows()]
    import review_tangent_confounds as RTC
    el = json.loads((DATA / "runs/olmo-direction/eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    frame = RTC.olmo_frame_rows()
    out = []
    for p in sorted(el["eligible_pos"]):
        r, fr = recs[p], frame[p]
        out.append({"arxiv_id": r["arxiv_id"], "followups": fr["followups"], "k": len(r["consensus"]), "pos": p,
                    "citing": citing.get(r["arxiv_id"], set()) | {f["arxiv_id"] for f in fr["followups"]},
                    "consensus": list(r["consensus"])})
    return out


def qwen_context_vectors(study: str) -> np.ndarray:
    """The cell's Qwen3 context vectors (the same rows the Part A cells use)."""
    if study == "disc":
        import revision_direction_tangent as RDT
        return RDT.load_generator("gpt-4o", "qwen3e06")[1]
    if study == "rep":
        import strengthen_critique_replication as SCR
        z = np.load(SCR.RUN / "emb.npz")
        return z["C"] / np.linalg.norm(z["C"], axis=1, keepdims=True)
    import review_tangent_confounds as RTC
    import olmo_direction_analysis as OD
    outs = OD.load_outputs()
    return np.stack([outs[p]["qwen3e06"]["C"] for p in sorted(q for q in outs if outs[q]["role"] == "eligible")])


def nb_selection(study: str, ctx: list[dict], n_only: int | None = None) -> list[dict]:
    """NB: future-split pool minus every citing paper and the context, ranked by Qwen3 cosine to the context; skip the
    first k, draw 3 uniformly from ranks k+1..k+200 with seed 20261114 + context index."""
    import pandas as pd
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    raw = np.load(DATA / "prepared/arxiv/emb_qwen3e06.npy", mmap_mode="r")
    E = np.asarray(raw[np.flatnonzero(keep)[fut]]).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    ids = meta.loc[fut, "id"].to_numpy()
    row = {x: n for n, x in enumerate(ids)}
    C = qwen_context_vectors(study)
    out = []
    for i, c in enumerate(ctx[:n_only] if n_only else ctx):
        sc = E @ C[i]
        ex = [row[x] for x in c["citing"] | {c["arxiv_id"]} if x in row]
        sc[ex] = -np.inf
        order = np.argsort(-sc)
        band = order[c["k"]:c["k"] + NB_BAND]
        pick = np.random.default_rng(SEED + i).choice(NB_BAND, N_NB, replace=False)
        out.append({"ctx": i, "arxiv_id": c["arxiv_id"], "k": c["k"],
                    "nb": [{"arxiv_id": str(ids[band[r]]), "rank": int(c["k"] + r + 1), "cos": float(sc[band[r]])} for r in pick]})
    return out


def olmo_sample(ctx: list[dict]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for i, c in enumerate(ctx):
        nc = [j for j in range(len(c["followups"])) if j not in set(c["consensus"])]
        pick = sorted(int(x) for x in rng.choice(nc, min(OLMO_NONCONT, len(nc)), replace=False)) if nc else []
        out.append({"ctx": i, "arxiv_id": c["arxiv_id"], "noncontinuations": pick})
    return out


def prepare(study: str) -> None:
    d = OUTDIR / study
    d.mkdir(parents=True, exist_ok=True)
    ctx = study_contexts(study)
    p = d / "nb_ids.json"
    assert not p.exists(), f"{p} exists: the NB selection is written once (registered order of work, step 3)"
    p.write_text(json.dumps(nb_selection(study, ctx), indent=1))
    log(f"wrote {p}")
    if study == "olmo":
        q = d / "olmo_followup_sample.json"
        assert not q.exists()
        q.write_text(json.dumps(olmo_sample(ctx), indent=1))
        log(f"wrote {q}")


# ------------------------------------------------------------------------------------------------ items
def build_items(study: str, nb: list[dict] | None = None, sample: list[dict] | None = None,
                n_ctx: int | None = None) -> list[dict]:
    import pandas as pd
    ctx = study_contexts(study)
    if n_ctx:
        ctx = ctx[:n_ctx]
    d = OUTDIR / study
    nb = nb if nb is not None else json.loads((d / "nb_ids.json").read_text())
    if study == "olmo":
        sample = sample if sample is not None else json.loads((d / "olmo_followup_sample.json").read_text())
    need = {c["arxiv_id"] for c in ctx} | {x["arxiv_id"] for r in nb for x in r["nb"]}
    t = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"])
    t = t[t["id"].isin(need)].set_index("id")
    items = []
    for i, c in enumerate(ctx):
        cb = CFC.paper_block(t.loc[c["arxiv_id"], "name"], t.loc[c["arxiv_id"], "abstract"])
        if study == "olmo":
            js = sorted(set(c["consensus"]) | set(sample[i]["noncontinuations"]))
        else:
            js = range(len(c["followups"]))
        for j in js:
            f = c["followups"][j]
            items.append({"id": f"rwf/{i}/{j}", "set": "RW-F", "ctx": i, "j": int(j), "orig_id": f["arxiv_id"],
                          "continuation": int(j) in set(c["consensus"]),
                          "prompt": PROMPT_R.format(ctx=cb, later=CFC.paper_block(f["title"], f["abstract"]))})
        for r in range(N_NA):
            items.append({"id": f"na/{i}/{r}", "set": "NA", "ctx": i, "r": r, "orig_id": c["arxiv_id"], "prompt": prompt_pa(cb)})
        assert nb[i]["arxiv_id"] == c["arxiv_id"]
        for r, x in enumerate(nb[i]["nb"]):
            items.append({"id": f"nb/{i}/{r}", "set": "NB", "ctx": i, "r": r, "orig_id": x["arxiv_id"], "rank": x["rank"],
                          "prompt": PROMPT_R.format(ctx=cb, later=CFC.paper_block(t.loc[x["arxiv_id"], "name"],
                                                                                   t.loc[x["arxiv_id"], "abstract"]))})
    return items


def counts(study: str) -> dict:
    """Exact call counts without building prompts (one call per item; retries come on top)."""
    ctx = study_contexts(study)
    if study == "olmo":
        rwf = sum(len(c["consensus"]) + min(OLMO_NONCONT, len(c["followups"]) - len(c["consensus"])) for c in ctx)
    else:
        rwf = sum(len(c["followups"]) for c in ctx)
    n = {"RW-F": rwf, "NA": N_NA * len(ctx), "NB": N_NB * len(ctx)}
    n["total"] = sum(n.values())
    return n


def dry_run(study: str, show: int) -> None:
    n = counts(study)
    ctx = study_contexts(study)
    d = OUTDIR / study
    nb = json.loads((d / "nb_ids.json").read_text()) if (d / "nb_ids.json").exists() else nb_selection(study, ctx, n_only=2)
    sample = None
    if study == "olmo":
        sample = json.loads((d / "olmo_followup_sample.json").read_text()) if (d / "olmo_followup_sample.json").exists() \
            else olmo_sample(ctx)
    items = build_items(study, nb=nb, sample=sample, n_ctx=2)
    print(f"=== DRY RUN {study}: no gateway client is created, no call is made ===")
    print(f"system: {CFC.SYSTEM}\nmodel: {ALIAS if study != 'olmo' else 'OLMo-7B-0424-Instruct (Kaggle)'}  temperature "
          f"{TEMPERATURE}  max_tokens {MAX_TOKENS if study != 'olmo' else 450}  json_mode  attempts {ATTEMPTS}")
    for s in ("RW-F", "NA", "NB"):
        for it in [x for x in items if x["set"] == s][:show]:
            print(f"\n--- {it['id']} ({s}) ---\n{it['prompt']}")
    print(f"\ncalls (one per item, before retries): {json.dumps(n)}")
    if study != "olmo":
        usd = n["total"] * (IN_TOK * PRICE_IN + OUT_TOK * PRICE_OUT) / 1e6
        print(f"estimated cost: ${usd:.2f} USD (~{usd * AUD:.0f} AUD) at {IN_TOK}/{OUT_TOK} tokens per call; "
              f"retries add at most the parse-failure share; ledger tag {TAGS[study]}")
    print(f"nb_ids.json {'present' if (d / 'nb_ids.json').exists() else 'NOT written (computed in memory for 2 contexts)'}")


# ------------------------------------------------------------------------------------------------ paid generation
def parse(raw: str) -> dict | None:
    """The ideas' parser (continue_from_context.generate.call): JSON with a title and an abstract."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and d.get("title") and d.get("abstract"):
        return {"title": d["title"], "abstract": d["abstract"]}
    return None


def run_calls(study: str, items: list[dict], out: Path, workers: int) -> dict:
    from latent_gap.gateway import Gateway
    gw = Gateway(ledger=DATA / "ledger.jsonl", model=ALIAS)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    lock = threading.Lock()
    stats_ = {"calls": 0, "parse_fail": 0, "dropped": 0}

    def one(it: dict) -> None:
        rec = {"title": "", "abstract": ""}
        att = 0
        for att in range(1, ATTEMPTS + 1):
            raw = gw.chat([{"role": "system", "content": CFC.SYSTEM}, {"role": "user", "content": it["prompt"]}],
                          max_tokens=MAX_TOKENS, json_mode=True, tag=TAGS[study], extra={"temperature": TEMPERATURE})
            p = parse(raw)
            with lock:
                stats_["calls"] += 1
                stats_["parse_fail"] += p is None
            if p:
                rec = p
                break
        with lock, out.open("a") as fh:
            stats_["dropped"] += not rec["abstract"]
            fh.write(json.dumps({k: v for k, v in it.items() if k != "prompt"} | rec | {"attempts": att}) + "\n")

    log(f"{study}: {len(todo)} items to call ({len(done)} already done) -> {out}")
    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            if (n + 1) % 100 == 0:
                log(f"  {n + 1}/{len(todo)} items, {stats_}")
    log(f"{study}: done {stats_}")
    return stats_


def smoke(study: str, n: int, workers: int) -> None:
    items = build_items(study)
    rng = np.random.default_rng(SEED)
    by = {s: [x for x in items if x["set"] == s] for s in ("RW-F", "NA", "NB")}
    k = {"RW-F": n - 2 * (n // 7), "NA": n // 7, "NB": n // 7}
    pick = [by[s][j] for s in by for j in rng.choice(len(by[s]), k[s], replace=False)]
    out = OUTDIR / study / "smoke.jsonl"
    st = run_calls(study, pick, out, workers)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    ok = sum(bool(r["abstract"]) for r in rows)
    first = sum(r.get("attempts") == 1 and bool(r["abstract"]) for r in rows)
    print(json.dumps({"items": len(rows), "parsed": ok, "parse_rate": ok / len(rows), "parsed_first_attempt": first,
                      "registered_threshold": 0.95, **st}))


def generate(study: str, workers: int) -> None:
    run_calls(study, build_items(study), OUTDIR / study / "rewrites.jsonl", workers)


# ------------------------------------------------------------------------------------------------ embedding inputs
def parsed_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return sorted((r for r in rows if r["abstract"]), key=lambda r: r["id"])  # fixed order = order of the vectors


def texts_of(rows):
    from latent_gap.arxiv import item_text
    return [item_text(r["title"], r["abstract"]) for r in rows]


def kaggle_disc_inputs() -> None:
    """texts.parquet for kaggle/review-rewrite-embed (verbatim copy of kaggle/embed-continuation/embed.py)."""
    import pandas as pd
    rows = parsed_rows(OUTDIR / "disc/rewrites.jsonl")
    up = DATA / "kaggle-upload/review-rewrite-disc-texts"
    up.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"kind": [r["set"] for r in rows], "key": [r["id"] for r in rows], "text": texts_of(rows)}).to_parquet(up / "texts.parquet")
    (up / "dataset-metadata.json").write_text(json.dumps({"title": "latent-gap review rewrite disc texts",
                                                          "id": "javieraguilarmartin/latent-gap-review-rewrite-disc-texts",
                                                          "licenses": [{"name": "other"}]}))
    print(f"{len(rows)} texts -> {up}\n  kaggle datasets create -p {up}\n  kaggle kernels push -p {ROOT / 'kaggle/review-rewrite-embed'}\n"
          f"  kaggle kernels output javieraguilarmartin/latent-gap-review-rewrite-embed -p {DATA / 'kaggle-output/review-rewrite-disc-embed'}")


def embed_rep_qwen() -> None:
    from latent_gap.embedder import LocalQwen
    rows = parsed_rows(OUTDIR / "rep/rewrites.jsonl")
    V = LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "4"))).embed(texts_of(rows))
    np.save(OUTDIR / "rep/rewrites_emb_qwen3e06.npy", V.astype(np.float32))
    log(f"qwen {V.shape}")


def embed_rep_bge() -> None:
    """Settings of revision_neutral.embed_bge (sentence-transformers, max_seq_length 512, normalised, CPU)."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rows = parsed_rows(OUTDIR / "rep/rewrites.jsonl")
    m = SentenceTransformer("BAAI/bge-m3", device="cpu")
    m.max_seq_length = 512
    T, out = texts_of(rows), []
    for b in range(0, len(T), 64):
        out.append(m.encode(T[b:b + 64], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        log(f"bge {min(b + 64, len(T))}/{len(T)}")
    np.save(OUTDIR / "rep/rewrites_emb_bgem3.npy", np.concatenate(out).astype(np.float32))


def kaggle_olmo_inputs() -> None:
    items = build_items("olmo")
    up = DATA / "kaggle-upload/review-rewrite-olmo-inputs"
    up.mkdir(parents=True, exist_ok=True)
    (up / "items.json").write_text(json.dumps(items))
    (up / "templates.json").write_text(json.dumps({"system": CFC.SYSTEM}))
    (up / "run_info.json").write_text(json.dumps({"seed": SEED, "n_items": len(items), "attempts": ATTEMPTS}))
    (up / "dataset-metadata.json").write_text(json.dumps({"title": "latent-gap review rewrite olmo inputs",
                                                          "id": "javieraguilarmartin/latent-gap-review-rewrite-olmo-inputs",
                                                          "licenses": [{"name": "other"}]}))
    print(f"{len(items)} OLMo items -> {up}\n  kaggle datasets create -p {up}\n  kaggle kernels push -p {ROOT / 'kaggle/review-rewrite'}"
          f"\n  kaggle kernels output javieraguilarmartin/latent-gap-review-rewrite -p {DATA / 'kaggle-output/review-rewrite-olmo'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry-run", "prepare", "smoke", "generate", "kaggle-disc-inputs", "embed-rep-qwen",
                                    "embed-rep-bge", "kaggle-olmo-inputs", "counts"])
    ap.add_argument("--study", choices=["disc", "rep", "olmo"], default="disc")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--confirm-paid", action="store_true", help="required for smoke/generate (paid spend)")
    a = ap.parse_args()
    if a.cmd in ("smoke", "generate"):
        if a.study == "olmo":
            raise SystemExit("OLMo items are generated on Kaggle: kaggle-olmo-inputs + kaggle/review-rewrite")
        if not a.confirm_paid:
            raise SystemExit("paid step: pass --confirm-paid only after Javier's go-ahead with the registered cost")
    {"dry-run": lambda: dry_run(a.study, a.show), "prepare": lambda: prepare(a.study),
     "smoke": lambda: smoke(a.study, a.n, a.workers), "generate": lambda: generate(a.study, a.workers),
     "kaggle-disc-inputs": kaggle_disc_inputs, "embed-rep-qwen": embed_rep_qwen, "embed-rep-bge": embed_rep_bge,
     "kaggle-olmo-inputs": kaggle_olmo_inputs, "counts": lambda: print(json.dumps(counts(a.study)))}[a.cmd]()


if __name__ == "__main__":
    main()
