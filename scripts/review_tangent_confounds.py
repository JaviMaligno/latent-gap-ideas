"""Part A of docs/prereg/review_tangent_confounds.md: tangent confounds, the gap yardstick, prompt echo and the
context sample (T1-T6), on existing vectors and texts. No generation, no GPU, no paid call, no encoder loaded.

  OMP_NUM_THREADS=4 uv run python scripts/review_tangent_confounds.py all        # registered run -> JSON
  OMP_NUM_THREADS=4 uv run python scripts/review_tangent_confounds.py gate       # reproduction gate only
  uv run python scripts/review_tangent_confounds.py t1 ... t6                    # one analysis (development only)
  # smoke: --cells DISC-4o --spaces qwen3e06 --n-ctx 12 --n-boot 200 --out /tmp/x.json --no-gate-stop

The registered numbers are those of `all` (one numpy generator per cell x embedder, seed 20261114, drawn in the order
cells -> analyses T1..T6 as listed in the pre-registration; a single subcommand starts a fresh generator and so gives
different bootstrap draws). Loaders of the original cell scripts are reused unchanged; every quantity is recomputed by
one engine (`engine`) that reproduces their definitions, optionally after removing a genre direction g.
Output: docs/prereg/review_tangent_confounds.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = ROOT / "docs/prereg/review_tangent_confounds.json"
LABELS = ROOT / "docs/prereg/strengthen_continuations_full_labels.json"
SEED, N_BOOT, BOUND, GATE_TOL = 20261114, 10000, 0.015, 5e-4
SPACES = ("qwen3e06", "bgem3")
CELLS = ("DISC-4o", "DISC-OLMo", "REP-4o", "OLMO-C", "MOD")
STUDY = {"DISC-4o": "DISC", "DISC-OLMo": "DISC", "REP-4o": "REP", "OLMO-C": "OLMO-C", "MOD": "MOD"}
CRIT = ("GEO", "LLM", "ORACLE")
MEASURES = ("step", "fit", "hit", "align", "p", "s")
CFG = {"n_boot": N_BOOT, "n_ctx": None}
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def tang(X, c):
    """Unit tangent of each row of X at c; NaN rows for an item identical to its context."""
    T = X - np.outer(X @ c, c)
    n = np.linalg.norm(T, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-6, T / n, np.nan)


def proj_out(X, g):
    """v -> unit(v - (v.g)g) row-wise (genre removal)."""
    return unit(X - np.outer(X @ g, g)) if X.ndim == 2 else unit(X - (X @ g) * g)


# =====================================================================================================  corpus
class Corpus:
    """Future-split corpus vectors of one embedder (non-adult rows; the direction pool and hit distractors of every
    cell are subsets of it: notpast == future in this corpus)."""

    def __init__(self, space: str):
        meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "primary", "adult", "date"])
        keep = ~meta["adult"].to_numpy()
        meta = meta[keep].reset_index(drop=True)
        fut = (meta["split"] == "future").to_numpy()
        assert ((meta["split"] != "past").to_numpy() == fut).all()
        raw = np.load(DATA / f"prepared/arxiv/emb_{space}.npy", mmap_mode="r")
        idx = np.flatnonzero(keep)[fut]
        self.E = unit(np.asarray(raw[idx]).astype(np.float32))
        m = meta[fut].reset_index(drop=True)
        self.ids = m["id"].to_numpy()
        self.primary = m["primary"].to_numpy()
        self.date = m["date"].to_numpy()
        self.row = {x: n for n, x in enumerate(self.ids)}
        self.cat = {p: np.flatnonzero(self.primary == p) for p in np.unique(self.primary)}
        self.Ecat = {p: self.E[r] for p, r in self.cat.items()}
        self.prim_of = dict(zip(meta["id"], meta["primary"]))
        self.space = space


def corpus_texts(ids) -> dict:
    t = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"])
    t = t[t["id"].isin(set(ids))]
    return {r.id: (r.name, r.abstract) for r in t.itertuples()}


# =====================================================================================================  cells
class Cell:
    """name, space, ctx (list of dicts: arxiv_id, followups), C (n,d), F (list), fset (list of index arrays that build
    f), excl (list of id sets removed from the pool), later (bool mask over corpus rows or None), rows + V (all ideas
    with an abstract: arm, ctx, round, title, abstract), pb (rows, V) one-shot CTX ideas or None, labels {i: {cont, bg}}."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.n = len(self.ctx)
        self.index = {}
        for n, r in enumerate(self.rows):
            self.index.setdefault((r["arm"], r["ctx"], r["round"]), []).append(n)
        self.arms = sorted({r["arm"] for r in self.rows}, key=lambda a: (a not in CRIT + ("NEUTRAL",), a))
        self.pb_index = {}
        if self.pb is not None:
            for n, r in enumerate(self.pb[0]):
                self.pb_index.setdefault(r["ctx"], []).append(n)


def _subset(cell_kw: dict) -> dict:
    n = CFG["n_ctx"]
    if not n or n >= len(cell_kw["ctx"]):
        return cell_kw
    kw = dict(cell_kw)
    for k in ("ctx", "F", "fset", "excl"):
        kw[k] = cell_kw[k][:n]
    kw["C"] = cell_kw["C"][:n]
    keep = [j for j, r in enumerate(cell_kw["rows"]) if r["ctx"] < n]
    kw["rows"] = [cell_kw["rows"][j] for j in keep]
    kw["V"] = cell_kw["V"][keep]
    if cell_kw.get("pb") is not None:
        rows, V = cell_kw["pb"]
        keep = [j for j, r in enumerate(rows) if r["ctx"] < n]
        kw["pb"] = ([rows[j] for j in keep], V[keep])
    if cell_kw.get("labels"):
        kw["labels"] = {i: v for i, v in cell_kw["labels"].items() if i < n}
    return kw


def disc_labels() -> dict:
    out = {}
    for p in json.loads(LABELS.read_text()):
        d = out.setdefault(p["ctx_index"], {"cont": [], "bg": []})
        if p["gpt61sol"] == p["opus55"] == "CONTINUES":
            d["cont"].append(p["followup_index"])
        if p["gpt61sol"] == p["opus55"] == "BACKGROUND":
            d["bg"].append(p["followup_index"])
    return out


def load_disc(gen: str, space: str, corp: Corpus) -> Cell:
    import revision_direction_tangent as RDT
    ctx, C, F, pb, PB, it, IT = RDT.load_generator(gen, space)
    rows, V = [dict(r) for r in it], [IT]
    if gen == "gpt-4o":
        import revision_neutral as RV
        nrows = RV.neutral_rows()
        NV = unit(np.load(RV.RUN / f"neutral_emb_{space}.npy").astype(np.float32))
        assert len(NV) == len(nrows)
        rows += nrows
        V.append(NV)
        hp = RV.RUN / "oracle_history_emb_ORACLE_ORACLE-H.npy"
        if space == "qwen3e06" and hp.exists():  # ORACLE-H (Table 4): local Qwen3 vectors only, with the local ORACLE
            allr = [json.loads(l) for l in (RV.RUN / "iter_ideas.jsonl").read_text().splitlines()]
            hr = [r for r in allr if r["abstract"] and r["arm"] in ("ORACLE", "ORACLE-H")]
            HV = unit(np.load(hp).astype(np.float32))
            assert len(HV) == len(hr)
            rows += [{**r, "arm": "ORACLE@local" if r["arm"] == "ORACLE" else "ORACLE-H"} for r in hr]
            V.append(HV)
        tag = "DISC-4o"
    else:
        import strengthen_olmo_neutral as SON
        nrows = SON.neutral_rows(SON.KOUT / "iter_olmo_neutral.jsonl")
        NV = unit(np.load(SON.KOUT / f"neutral_emb_{space}.npy").astype(np.float32))
        assert len(NV) == len(nrows)
        rows += nrows
        V.append(NV)
        tag = "DISC-OLMo"
    keep = [n for n, r in enumerate(pb) if r["arm"] == "CTX"]
    kw = dict(name=tag, space=space, ctx=ctx, C=C, F=F, fset=[np.arange(len(f)) for f in F],
              excl=[{f["arxiv_id"] for f in c["followups"]} for c in ctx], later=None,
              rows=rows, V=np.concatenate(V).astype(np.float32), pb=([pb[n] for n in keep], PB[keep]),
              labels=disc_labels(), primary=[corp.prim_of[c["arxiv_id"]] for c in ctx])
    return Cell(**_subset(kw))


def load_rep(space: str, corp: Corpus) -> Cell:
    import strengthen_critique_replication as SCR
    ctx = SCR.ctx_rows()
    nf = [len(c["followups"]) for c in ctx]
    starts = np.cumsum([0] + nf)
    it, nu = SCR.idea_rows()
    if space == "qwen3e06":
        z = np.load(SCR.RUN / "emb.npz")
        C, Fall, V = unit(z["C"]), unit(z["F"]), unit(np.load(SCR.RUN / "ideas_emb_qwen3e06.npy"))
    else:
        z = np.load(SCR.RUN / "emb_bgem3.npz")
        C, Fall, V = unit(z["ctx"]), unit(z["fup"]), unit(z["ideas"])
    assert len(V) == len(it) + len(nu) and len(Fall) == starts[-1]
    F = [Fall[starts[i]:starts[i + 1]].astype(np.float32) for i in range(len(ctx))]
    kw = dict(name="REP-4o", space=space, ctx=ctx, C=C.astype(np.float32), F=F, fset=[np.arange(n) for n in nf],
              excl=[{f["arxiv_id"] for f in c["followups"]} for c in ctx], later=None,
              rows=[dict(r) for r in it] + [dict(r) for r in nu], V=V.astype(np.float32), pb=None, labels=None,
              primary=[corp.prim_of[c["arxiv_id"]] for c in ctx])
    return Cell(**_subset(kw))


_OLMO_OUTS = {}


def olmo_frame_rows() -> list[dict]:
    rows = []
    for l in (DATA / "runs/olmo-direction/followups.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["n_followups"] >= 5:
            rows.append(r)
    return rows  # index = pos (as olmo_direction_kaggle.build)


def load_olmoc(space: str, corp: Corpus) -> Cell:
    import olmo_direction_analysis as OD
    if "outs" not in _OLMO_OUTS:
        _OLMO_OUTS["outs"] = OD.load_outputs()
    outs = _OLMO_OUTS["outs"]
    el = json.loads((OD.RUN / "eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    citing = {json.loads(l)["arxiv_id"]: set(json.loads(l)["citing_arxiv_all"])
              for l in (OD.RUN / "citing_all.jsonl").read_text().splitlines()}
    frame = olmo_frame_rows()
    ctx, C, F, fset, excl, rows, V, labels = [], [], [], [], [], [], [], {}
    for i, p in enumerate(sorted(q for q in outs if outs[q]["role"] == "eligible")):
        rec, fr = recs[p], frame[p]
        assert fr["arxiv_id"] == rec["arxiv_id"] and len(fr["followups"]) == len(outs[p][space]["F"])
        ctx.append({"arxiv_id": rec["arxiv_id"], "title": fr["title"], "followups": fr["followups"], "pos": p,
                    "frame_pos": fr["frame_pos"]})
        C.append(outs[p][space]["C"])
        F.append(outs[p][space]["F"])
        fset.append(np.array(rec["consensus"], dtype=int))
        excl.append(citing[rec["arxiv_id"]] | {rec["arxiv_id"]})
        rows += [{**r, "ctx": i} for r in outs[p]["rows"]]
        V.append(outs[p][space]["V"])
        labels[i] = {"cont": list(rec["consensus"]), "bg": list(rec.get("both_background", []))}
    kw = dict(name="OLMO-C", space=space, ctx=ctx, C=np.stack(C), F=F, fset=fset, excl=excl, later=None, rows=rows,
              V=np.concatenate(V), pb=None, labels=labels, primary=[corp.prim_of[c["arxiv_id"]] for c in ctx])
    return Cell(**_subset(kw))


def load_mod(space: str, corp: Corpus) -> Cell:
    import revision_null_effects as RN
    import strengthen_modern_neutral as SMN
    ctx = RN.contexts("followups-2025")
    it = SMN.rows_of(SMN.RUN / "iter_ideas.jsonl", SMN.ARMS)
    nrows = SMN.neutral_rows()
    NV = unit(np.load(SMN.RUN / f"neutral_emb_{space}.npy").astype(np.float32))
    pb = [r for r in map(json.loads, (SMN.RUN / "ideas.jsonl").read_text().splitlines()) if r["abstract"]]
    if space == "qwen3e06":
        z = np.load(SMN.RUN / "emb.npz")
        C, owner = unit(z["C"]), z["owner"]
        F = [unit(z["F"][owner == i]) for i in range(len(ctx))]
        IT = unit(np.load(SMN.RUN / "oracle_history_emb_GEO_LLM_ORACLE_ORACLE-H.npy").astype(np.float32))
        PB = unit(np.load(SMN.RUN / "partB_emb.npy").astype(np.float32))
    else:
        import strengthen_bge_replication as SB
        L, _ = SB.local()
        C = np.stack([L["modern/ctx", str(i)] for i in range(len(ctx))])
        F = [np.stack([L["modern/fup", f"{i}/{j}"] for j in range(len(c["followups"]))]) for i, c in enumerate(ctx)]
        IT = np.stack([L["modern/iter", f"{r['ctx']}/{r['arm']}/{r['k']}/{r['round']}"] for r in it])
        PB = np.stack([L["modern/partB", f"{r['ctx']}/{r['arm']}/{r['k']}"] for r in pb])
    assert len(IT) == len(it) and len(PB) == len(pb)
    later = corp.date >= min(f["date"] for c in ctx for f in c["followups"])
    keep = [n for n, r in enumerate(pb) if r["arm"] == "CTX"]
    kw = dict(name="MOD", space=space, ctx=ctx, C=np.asarray(C, np.float32), F=[np.asarray(f, np.float32) for f in F],
              fset=[np.arange(len(f)) for f in F], excl=[{f["arxiv_id"] for f in c["followups"]} for c in ctx],
              later=later, rows=[dict(r) for r in it] + nrows, V=np.concatenate([IT, NV]).astype(np.float32),
              pb=([pb[n] for n in keep], np.asarray(PB, np.float32)[keep]), labels=None,
              primary=[corp.prim_of[c["arxiv_id"]] for c in ctx])
    return Cell(**_subset(kw))


def load_cell(name: str, space: str, corp: Corpus) -> Cell:
    if name == "DISC-4o":
        return load_disc("gpt-4o", space, corp)
    if name == "DISC-OLMo":
        return load_disc("olmo-7b-0424", space, corp)
    if name == "REP-4o":
        return load_rep(space, corp)
    if name == "OLMO-C":
        return load_olmoc(space, corp)
    return load_mod(space, corp)


# =====================================================================================================  engine
def engine(cell: Cell, corp: Corpus, g=None, fset=None, keep_geom=False) -> dict:
    """All per-context quantities of the cell scripts (Table 2/3 measures per arm and round; f, p, s), the follow-ups'
    leave-one-out projections (T1, T2 iii) and the round-0 idea items, optionally after removing a genre direction g
    ((d,) shared, or (n, d) one per context) and with the f-set replaced (fset: list of index arrays; k = its size)."""
    n, d = cell.n, cell.C.shape[1]
    fset = fset if fset is not None else cell.fset
    M = {a: {m: np.full((n, 3), np.nan) for m in MEASURES} for a in cell.arms}
    out = {k: np.full(n, np.nan) for k in ("fu_s", "fu_align", "loo_align", "cont_s", "bg_s", "noncont_s", "I0_s",
                                           "I0_align", "f_dot_p", "gap_b1", "gap_r0", "k")}
    items_r0, items_fu, geom = [], [], []
    E = corp.E
    shared_g = g is not None and np.ndim(g) == 1
    if shared_g:
        Eg_shared = E @ g
    pool_ok = cell.later if cell.later is not None else None
    for i in range(n):
        gi = None if g is None else (g if shared_g else g[i])
        c0, F0 = cell.C[i], cell.F[i]
        c = c0 if gi is None else proj_out(c0, gi)
        Fi = F0 if gi is None else proj_out(F0, gi)
        js = np.asarray(fset[i], dtype=int)
        TF = tang(Fi, c)
        if len(js) == 0:
            geom.append(None)
            continue
        TC = TF[js]
        f = unit(TC.mean(0))
        # direction pool: k nearest (after removal, by the closed form) outside the excluded set
        sc = E @ c0
        if gi is not None:
            Eg = Eg_shared if shared_g else E @ gi
            cg = float(c0 @ gi)
            sc = (sc - Eg * cg) / (np.sqrt(np.clip(1 - Eg ** 2, 1e-12, None)) * np.sqrt(max(1 - cg ** 2, 1e-12)))
        sc = sc.copy()
        ex = [corp.row[x] for x in cell.excl[i] if x in corp.row]
        sc[ex] = -np.inf
        if pool_ok is not None:
            sc[~pool_ok] = -np.inf
        k = len(js)
        top = np.argsort(-sc)[:k]
        P = E[top] if gi is None else proj_out(E[top], gi)
        p = unit(tang(P, c).mean(0))
        s = unit(f - (f @ p) * p)
        out["f_dot_p"][i], out["k"][i] = float(f @ p), k
        # follow-ups' leave-one-out projections on s_-j (p unchanged) and on f_-j
        fsteps = Fi @ c
        if len(js) > 1:
            Fm = unit(TC.sum(0) - TC)
            Sm = unit(Fm - np.outer(Fm @ p, p))
            ps, pa = np.einsum("ij,ij->i", TC, Sm), np.einsum("ij,ij->i", TC, Fm)
            out["fu_s"][i], out["fu_align"][i] = ps.mean(), pa.mean()
            out["loo_align"][i] = pa.mean()
            items_fu.append(np.column_stack([np.full(len(js), i), fsteps[js], ps, TC @ p, pa]))
            loo = dict(zip(js.tolist(), ps.tolist()))
        else:
            loo = {}
        lab = (cell.labels or {}).get(i)
        if lab:
            if set(js.tolist()) == set(range(len(Fi))):  # DISC: f from all follow-ups -> s_-j is the LOO direction
                cv = [loo[j] for j in lab["cont"] if j in loo]
                bv = [loo[j] for j in lab["bg"] if j in loo]
                out["cont_s"][i] = np.mean(cv) if cv else np.nan
                out["bg_s"][i] = np.mean(bv) if bv else np.nan
        if cell.name == "OLMO-C":
            nc = np.setdiff1d(np.arange(len(Fi)), js)
            if len(nc):
                out["noncont_s"][i] = float(np.nanmean(TF[nc] @ s))
        # ideas
        Dv_rows = None
        r0_steps, r0_s, r0_a = [], [], []
        for a in cell.arms:
            for r in range(3):
                idx = cell.index.get((a, i, r))
                if not idx:
                    continue
                X0 = cell.V[idx]
                X = X0 if gi is None else proj_out(X0, gi)
                st = X @ c
                sf = X @ Fi.T
                if Dv_rows is None:
                    cat = corp.cat.get(cell.primary[i], np.zeros(0, int))
                    m = np.ones(len(cat), bool)
                    exf = {corp.row[f["arxiv_id"]] for f in cell.ctx[i]["followups"] if f["arxiv_id"] in corp.row}
                    if exf:
                        m &= ~np.isin(cat, list(exf))
                    if pool_ok is not None:
                        m &= pool_ok[cat]
                    Dv_rows = m
                    Dcat = corp.Ecat.get(cell.primary[i], np.zeros((0, d), np.float32))
                    Dg = None if gi is None else Dcat @ gi
                sd = X0 @ Dcat.T
                if gi is not None:
                    xg = X0 @ gi
                    sd = (sd - np.outer(xg, Dg)) / (np.sqrt(np.clip(1 - xg ** 2, 1e-12, None))[:, None]
                                                    * np.sqrt(np.clip(1 - Dg ** 2, 1e-12, None))[None])
                sd = np.where(Dv_rows[None], sd, -np.inf).max(1) if sd.shape[1] else np.full(len(X0), -np.inf)
                T = tang(X, c)
                ok = np.isfinite(T).all(1)
                M[a]["step"][i, r] = st.mean()
                M[a]["fit"][i, r] = sf.max(1).mean()
                M[a]["hit"][i, r] = (sf.max(1) > sd).mean()
                if ok.any():
                    M[a]["align"][i, r] = (T[ok] @ f).mean()
                    M[a]["p"][i, r] = (T[ok] @ p).mean()
                    M[a]["s"][i, r] = (T[ok] @ s).mean()
                if r == 0 and a in CRIT:
                    r0_steps += st.tolist()
                    r0_s += (T[ok] @ s).tolist()
                    r0_a += (T[ok] @ f).tolist()
                    items_r0.append(np.column_stack([np.full(ok.sum(), i), st[ok], T[ok] @ s, T[ok] @ p, T[ok] @ f]))
        out["I0_s"][i] = np.mean(r0_s) if r0_s else np.nan
        out["I0_align"][i] = np.mean(r0_a) if r0_a else np.nan
        med_f = float(np.median(fsteps[js])) if fset is not cell.fset else float(np.median(fsteps))
        if cell.name == "OLMO-C":
            med_f = float(np.median(fsteps))
        out["gap_r0"][i] = (np.median(r0_steps) - med_f) if r0_steps else np.nan
        if cell.pb is not None and i in cell.pb_index:
            Xb = cell.pb[1][cell.pb_index[i]]
            Xb = Xb if gi is None else proj_out(Xb, gi)
            out["gap_b1"][i] = float(np.median(Xb @ c) - med_f)
        if keep_geom:
            geom.append({"c": c, "f": f, "p": p, "s": s, "js": js, "TF": TF, "gi": gi})
    out["M"] = M
    out["items_r0"] = np.concatenate(items_r0) if items_r0 else np.zeros((0, 5))
    out["items_fu"] = np.concatenate(items_fu) if items_fu else np.zeros((0, 5))
    if keep_geom:
        out["geom"] = geom
    return out


def change(M, a, m):
    """step: r0 - r2 (positive = longer step); others r2 - r0."""
    if a not in M:
        return None
    return M[a][m][:, 0] - M[a][m][:, 2] if m == "step" else M[a][m][:, 2] - M[a][m][:, 0]


# =====================================================================================================  statistics
def draws(n, rng, B=None):
    return rng.integers(0, n, size=(B or CFG["n_boot"], n))


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]


def boot_mean(x, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [None, None]
    return ci(x[draws(len(x), rng)].mean(1))


def wil(x, alt="two-sided"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(stats.wilcoxon(x, alternative=alt).pvalue) if len(x) >= 2 and np.any(x != 0) else None


def holm(p: dict) -> dict:
    keys, out, run = sorted(p, key=lambda k: p[k]), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, p[k] * (len(keys) - j)))
        out[k] = run
    return out


def summ(x, rng) -> dict:
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    return {"mean": float(x[ok].mean()) if ok.any() else None, "ci": boot_mean(x, rng), "n": int(ok.sum())}


def tost(d, bound=BOUND):
    n, m, se = len(d), float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return tost_from(m, se, n - 1, bound)


def tost_from(m, se, df, bound=BOUND):
    p_lo = 1 - stats.t.cdf((m + bound) / se, df)
    p_hi = stats.t.cdf((m - bound) / se, df)
    tq = stats.t.ppf(0.975, df)
    return {"p_tost": float(max(p_lo, p_hi)), "t_ci95": [m - tq * se, m + tq * se], "se": se}


def ratio_boot(num, den, rng):
    """mean(num)/mean(den) over contexts where both exist, resampling numerator and denominator together."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = ~np.isnan(num) & ~np.isnan(den)
    num, den = num[ok], den[ok]
    if len(num) < 2:
        return {"value": None, "ci": [None, None], "n": int(len(num))}
    D = draws(len(num), rng)
    return {"value": float(num.mean() / den.mean()), "ci": ci(num[D].mean(1) / den[D].mean(1)), "n": int(len(num)),
            "n_dropped": int((~ok).sum())}


def within(a, inv, G):
    s = np.zeros((G,) + a.shape[1:])
    np.add.at(s, inv, a)
    cnt = np.bincount(inv, minlength=G).astype(float)
    return a - (s / cnt.reshape((-1,) + (1,) * (a.ndim - 1)))[inv]


def fe_ols(y, X, groups, rng=None, W=None, L=None, B=None):
    """OLS of y on X with group fixed effects (within transform), cluster-robust (CR1, by group) variance and, when rng
    or W is given, a cluster bootstrap over groups (per-cluster X'X and X'y reweighted by the draw counts).
    L: dict name -> contrast vector over the columns of X. Returns beta, se, t-CI, p, bootstrap CI per contrast."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, groups = y[ok], X[ok], np.asarray(groups)[ok]
    uniq, inv = np.unique(groups, return_inverse=True)
    G, N, K = len(uniq), len(y), X.shape[1]
    yd, Xd = within(y, inv, G), within(X, inv, G)
    A_c = np.zeros((G, K, K))
    np.add.at(A_c, inv, Xd[:, :, None] * Xd[:, None, :])
    b_c = np.zeros((G, K))
    np.add.at(b_c, inv, Xd * yd[:, None])
    A = A_c.sum(0)
    beta = np.linalg.solve(A, b_c.sum(0))
    e = yd - Xd @ beta
    sc = np.zeros((G, K))
    np.add.at(sc, inv, Xd * e[:, None])
    Ai = np.linalg.inv(A)
    Vc = Ai @ (sc.T @ sc) @ Ai * G / (G - 1) * (N - 1) / max(N - K, 1)
    L = L or {f"b{j}": np.eye(K)[j] for j in range(K)}
    res = {"n_obs": int(N), "n_clusters": int(G), "beta": beta.tolist()}
    if W is None and rng is not None:
        D = draws(G, rng, B)
        W = np.zeros((D.shape[0], G))
        np.add.at(W, (np.repeat(np.arange(D.shape[0]), G), D.ravel()), 1)
    bb = None
    if W is not None:
        Ab = np.einsum("bg,gij->bij", W, A_c)
        rhs = W @ b_c
        good = np.abs(np.linalg.det(Ab)) > 1e-12
        bb = np.full((W.shape[0], K), np.nan)
        bb[good] = np.linalg.solve(Ab[good], rhs[good][..., None])[..., 0]
    tq = stats.t.ppf(0.975, G - 1)
    for name, l in L.items():
        l = np.asarray(l, float)
        est, se = float(l @ beta), float(np.sqrt(l @ Vc @ l))
        tval = est / se if se > 0 else np.nan
        e_ = {"est": est, "se_cluster": se, "t_ci95": [est - tq * se, est + tq * se],
              "p_cluster_t": float(2 * stats.t.sf(abs(tval), G - 1)) if np.isfinite(tval) else None}
        if bb is not None:
            e_["boot_ci"] = ci(bb @ l)
        res[name] = e_
    res["_draws"] = bb
    res["_W"] = W
    return res


def strip(d):
    """Drop private arrays from fe_ols results before JSON."""
    if isinstance(d, dict):
        return {str(k): strip(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (bool, np.bool_)):
        return bool(d)
    return d


def ols_hc3(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    S = (X * (e / (1 - h))[:, None])
    V = XtXi @ (S.T @ S) @ XtXi
    se = np.sqrt(np.diag(V))
    df = len(x) - 2
    tq = stats.t.ppf(0.975, df)
    t = beta[1] / se[1]
    return {"a": float(beta[0]), "b": float(beta[1]), "se_b_hc3": float(se[1]),
            "b_t_ci95": [float(beta[1] - tq * se[1]), float(beta[1] + tq * se[1])],
            "p_b_hc3": float(2 * stats.t.sf(abs(t), df)), "a_t_ci95": [float(beta[0] - tq * se[0]), float(beta[0] + tq * se[0])]}


# =====================================================================================================  tables
def n_family(M, rng, crit=CRIT, tag="N"):
    """The 14 pre-registered NEUTRAL contrasts (revision_neutral / strengthen_olmo_neutral / strengthen_modern_neutral)."""
    spec = [(f"{tag}1", "GEO", "step", "greater"), (f"{tag}2", "LLM", "step", "greater")]
    for nm, m in ((f"{tag}3", "fit"), (f"{tag}4", "hit"), (f"{tag}5", "align"), (f"{tag}6", "s")):
        spec += [(nm, a, m, "two-sided") for a in crit]
    tests, p = {}, {}
    for nm, a, m, alt in spec:
        d = change(M, a, m) - change(M, "NEUTRAL", m)
        d = d[~np.isnan(d)]
        key = f"{nm}:{a}-NEUTRAL:{m}"
        pv = wil(d, alt)
        tests[key] = {"diff_mean": float(d.mean()), "ci": boot_mean(d, rng), "p": pv, "alternative": alt, "n": int(len(d))}
        p[key] = pv if pv is not None else 1.0
    for k, v in holm(p).items():
        tests[k]["p_holm"] = v
        tests[k]["holm_significant"] = v < 0.05
    return tests


def rep_family(M, rng):
    """R1, R2, E1-E6 with the registered rule of strengthen_critique_replication (Holm over 8, 2*p_TOST)."""
    conf, p = {}, {}
    for nm, a in (("R1", "GEO"), ("R2", "LLM")):
        d = change(M, a, "step") - change(M, "NEUTRAL", "step")
        d = d[~np.isnan(d)]
        conf[f"{nm}:{a}-NEUTRAL:step"] = {"diff_mean": float(d.mean()), "ci": boot_mean(d, rng), "p": wil(d, "greater"), "n": int(len(d))}
        p[f"{nm}:{a}-NEUTRAL:step"] = conf[f"{nm}:{a}-NEUTRAL:step"]["p"]
    for j, a in enumerate(CRIT):
        d = change(M, a, "s")
        d = d[~np.isnan(d)]
        k = f"E{j + 1}:{a}:s_change"
        conf[k] = {"mean": float(d.mean()), "ci": boot_mean(d, rng), "n": int(len(d)), "p_wilcoxon": wil(d), **tost(d)}
        p[k] = min(1.0, 2 * conf[k]["p_tost"])
    for j, a in enumerate(CRIT):
        d = change(M, a, "s") - change(M, "NEUTRAL", "s")
        d = d[~np.isnan(d)]
        k = f"E{j + 4}:{a}-NEUTRAL:s"
        conf[k] = {"diff_mean": float(d.mean()), "ci": boot_mean(d, rng), "n": int(len(d)), "p_wilcoxon": wil(d), **tost(d)}
        p[k] = min(1.0, 2 * conf[k]["p_tost"])
    return rep_readings(conf, p)


def rep_readings(conf, p):
    for k, v in holm(p).items():
        conf[k]["p_holm"] = v
    for k, v in conf.items():
        if k.startswith("E"):
            lo, hi = v["t_ci95"]
            v["reading"] = ("within ±0.015 (confirmed)" if v["p_holm"] < 0.05 else
                            "turn detected" if (lo > 0 or hi < 0) else "insufficient")
    return conf


def olmo_reading(sig, lo, hi):
    inside = -BOUND < lo and hi < BOUND
    return ("detected, below 0.015" if sig and inside else "feedback-specific turn detected" if sig
            else "no feedback-specific turn of 0.015 or more" if inside else "insufficient")


def olmo_family(M, rng, key="s"):
    out, ps = {}, {}
    for a in CRIT:
        d = change(M, a, key) - change(M, "NEUTRAL", key)
        d = d[~np.isnan(d)]
        out[a] = {"n": int(len(d)), "mean": float(d.mean()), "ci": boot_mean(d, rng), "p": wil(d)}
        ps[a] = out[a]["p"] if out[a]["p"] is not None else 1.0
    for a, h in holm(ps).items():
        lo, hi = out[a]["ci"]
        out[a] |= {"p_holm": h, "significant_holm": h < 0.05, "ci_within_pm0.015": -BOUND < lo and hi < BOUND,
                   "reading": olmo_reading(h < 0.05, lo, hi)}
    return out


def table3_cell(eng, rng=None):
    """Per arm: alignment change, change along p and along s (Wilcoxon p), weighted contributions a*dp, b*ds and the
    share carried by p (revision_direction_tangent.run_mode)."""
    M = eng["M"]
    a_ = eng["f_dot_p"]
    b_ = np.sqrt(np.clip(1 - a_ ** 2, 0, None))
    res = {}
    for arm in CRIT + ("NEUTRAL",):
        if arm not in M:
            continue
        dal, dp, ds = change(M, arm, "align"), change(M, arm, "p"), change(M, arm, "s")
        ok = ~np.isnan(dal) & ~np.isnan(dp) & ~np.isnan(ds)
        cp, cs = a_[ok] * dp[ok], b_[ok] * ds[ok]
        res[arm] = {"n": int(ok.sum()), "alignment_change": float(dal[ok].mean()), "alignment_change_p": wil(dal[ok]),
                    "along_p": float(dp[ok].mean()), "along_p_p": wil(dp[ok]),
                    "along_s": float(ds[ok].mean()), "along_s_p": wil(ds[ok]),
                    "contrib_p_weighted": float(cp.mean()), "contrib_s_weighted": float(cs.mean()),
                    "share_p": float(cp.mean() / dal[ok].mean()) if dal[ok].mean() != 0 else None,
                    "alignment_r0": float(np.nanmean(M[arm]["align"][:, 0])),
                    "alignment_r2": float(np.nanmean(M[arm]["align"][:, 2]))}
    return res


def table3_holm(cells: dict) -> dict:
    """cells: {"gpt-4o/qwen3e06": table3_cell, ...}; Holm over the 36 critique-arm tests."""
    rows, p = [], {}
    for cname, t in cells.items():
        for arm in CRIT:
            for k in ("alignment_change", "along_p", "along_s"):
                key = f"{cname} {arm} {k}"
                p[key] = t[arm][k + "_p"] if t[arm][k + "_p"] is not None else 1.0
    h = holm(p)
    out = {"n_tests": len(p), "p_holm": h}
    out["alignment_rises_holm"] = [k for k in h if k.endswith("alignment_change") and h[k] < 0.05
                                   and cells[k.split(" ")[0]][k.split(" ")[1]]["alignment_change"] > 0]
    out["share_p_above_half"] = [f"{c} {a}" for c, t in cells.items() for a in CRIT
                                 if t[a]["share_p"] is not None and t[a]["share_p"] > 0.5]
    out["along_s_holm_significant"] = {k: cells[k.split(" ")[0]][k.split(" ")[1]]["along_s"] for k in h
                                       if k.endswith("along_s") and h[k] < 0.05}
    return out


# =====================================================================================================  gate
def gate(engs: dict) -> dict:
    """engs[(cell, space)] = engine output without g. Compares with the published JSONs."""
    P = ROOT / "docs/prereg"
    rows = []

    def add(name, got, want, gating=True):
        rows.append({"quantity": name, "recomputed": got, "published": want, "gating": gating,
                     "abs_diff": None if got is None or want is None else abs(got - want)})
    rdt = json.loads((P / "revision_direction_tangent.json").read_text())["decomposition"]["saved"]
    for (cell, space), eng in engs.items():
        M = eng["M"]
        if cell in ("DISC-4o", "DISC-OLMo"):
            gen = "gpt-4o" if cell == "DISC-4o" else "olmo-7b-0424"
            for a in CRIT:
                add(f"Table3 {gen}/{space} {a} along_s", float(np.nanmean(change(M, a, "s"))),
                    rdt[f"{gen}/{space}"][a]["along_s_unweighted"])
        if cell == "DISC-4o":
            rn = json.loads((P / "revision_neutral.json").read_text())[space]["tests"]
            for nm, m in (("N5", "align"), ("N6", "s")):
                for a in CRIT:
                    d = change(M, a, m) - change(M, "NEUTRAL", m)
                    add(f"DISC-4o {space} {nm}:{a}-NEUTRAL:{m}", float(np.nanmean(d)), rn[f"{nm}:{a}-NEUTRAL:{m}"]["diff_mean"])
            if space == "qwen3e06":
                add("DISC-4o qwen3e06 B1 step gap", float(np.nanmean(eng["gap_b1"])),
                    json.loads((P / "continuation_reanalysis.json").read_text())["gpt-4o/qwen3e06/all"]["B"]["CTX"]["mean_step_gap"])
        if cell == "REP-4o":
            sc = json.loads((P / "strengthen_critique_replication.json").read_text())[space]["confirmatory"]
            for j, a in enumerate(CRIT):
                add(f"REP {space} E{j + 1}:{a}:s_change", float(np.nanmean(change(M, a, "s"))), sc[f"E{j + 1}:{a}:s_change"]["mean"])
                d = change(M, a, "s") - change(M, "NEUTRAL", "s")
                add(f"REP {space} E{j + 4}:{a}-NEUTRAL:s", float(np.nanmean(d)), sc[f"E{j + 4}:{a}-NEUTRAL:s"]["diff_mean"])
        if cell == "OLMO-C":
            od = json.loads((P / "strengthen_olmo_direction.json").read_text())[f"olmo/{space}"]["contrasts_along_s"]
            for a in CRIT:
                d = change(M, a, "s") - change(M, "NEUTRAL", "s")
                add(f"OLMO-C {space} {a}-NEUTRAL along s", float(np.nanmean(d)), od[a]["mean"])
        if cell == "MOD":  # not in the registered gate list: informational check of the MOD loader
            mn = json.loads((P / "strengthen_modern_neutral.json").read_text())[space]["tests"]
            for nm, m in (("N5", "align"), ("N6", "s")):
                for a in CRIT:
                    d = change(M, a, m) - change(M, "NEUTRAL", m)
                    add(f"MOD {space} {nm}:{a}-NEUTRAL:{m}", float(np.nanmean(d)), mn[f"{nm}:{a}-NEUTRAL:{m}"]["diff_mean"], gating=False)
    worst = max((r["abs_diff"] for r in rows if r["abs_diff"] is not None and r["gating"]), default=0.0)
    return {"tolerance": GATE_TOL, "rows": rows, "max_abs_diff": worst, "passed": bool(worst <= GATE_TOL),
            "subset_n_ctx": CFG["n_ctx"]}


# =====================================================================================================  T1
def t1(cell: Cell, eng: dict, rng) -> dict:
    M = eng["M"]
    FU, I0 = eng["fu_s"], eng["I0_s"]
    Gi = FU - I0
    res = {"n_contexts": cell.n, "n_dropped_single_followup": int(np.isnan(FU).sum()),
           "FU": summ(FU, rng), "I0": summ(I0, rng), "G": summ(Gi, rng)}
    if np.isfinite(eng["cont_s"]).any():
        res["CONT"] = summ(eng["cont_s"], rng)
        res["BG"] = summ(eng["bg_s"], rng)
        res["G_cont"] = summ(eng["cont_s"] - I0, rng)
        res["CONT_below_I0"] = bool(np.nanmean(eng["cont_s"]) < np.nanmean(I0[np.isfinite(eng["cont_s"])]))
    if np.isfinite(eng["noncont_s"]).any():
        res["noncontinuations_on_full_s"] = summ(eng["noncont_s"], rng)
    res["bound_0.015_over_G"] = ratio_boot(np.full(cell.n, BOUND), Gi, rng)
    dN = change(M, "NEUTRAL", "s")
    phi, psi = {}, {}
    for a in CRIT + ("NEUTRAL",):
        phi[a] = ratio_boot(change(M, a, "s"), Gi, rng)
    for a in CRIT:
        psi[a] = ratio_boot(change(M, a, "s") - dN, Gi, rng)
    if "ORACLE-H" in M:
        ref = "ORACLE@local" if "ORACLE@local" in M else "ORACLE"
        psi[f"ORACLE-H-{ref}"] = ratio_boot(change(M, "ORACLE-H", "s") - change(M, ref, "s"), Gi, rng)
        phi["ORACLE-H"] = ratio_boot(change(M, "ORACLE-H", "s"), Gi, rng)
    res["phi_s"], res["psi_s"] = phi, psi
    Ga = eng["fu_align"] - eng["I0_align"]
    res["FU_align_loo"] = summ(eng["fu_align"], rng)
    res["G_align"] = summ(Ga, rng)
    dNa = change(M, "NEUTRAL", "align")
    res["phi_align"] = {a: ratio_boot(change(M, a, "align"), Ga, rng) for a in CRIT + ("NEUTRAL",)}
    res["psi_align"] = {a: ratio_boot(change(M, a, "align") - dNa, Ga, rng) for a in CRIT}
    return res


# =====================================================================================================  T2
def slope_boot(x, y, rng):
    D = draws(len(x), rng)
    xb, yb = x[D], y[D]
    xm, ym = xb.mean(1, keepdims=True), yb.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = ((xb - xm) * (yb - ym)).sum(1) / ((xb - xm) ** 2).sum(1)
        pi = b * xm[:, 0] / ym[:, 0]
    return b, pi, ym[:, 0]


def t2(cell: Cell, eng: dict, rng) -> dict:
    M = eng["M"]
    arms = [a for a in CRIT + ("NEUTRAL",) if a in M]
    res = {"dose_response": {}, "adjusted": {}, "cross_section": {}}
    pv = {}
    for a in arms:
        ds = change(M, a, "step")
        e = {}
        for y in ("s", "p", "align"):
            dy = change(M, a, y)
            ok = ~np.isnan(ds) & ~np.isnan(dy)
            x, z = ds[ok], dy[ok]
            o = ols_hc3(x, z)
            bb, pib, ymb = slope_boot(x, z, rng)
            pi = o["b"] * x.mean() / z.mean() if abs(z.mean()) >= 0.002 else None
            e[y] = {"n": int(ok.sum()), "pearson_r": float(stats.pearsonr(x, z)[0]), "spearman_rho": float(stats.spearmanr(x, z)[0]),
                    **o, "b_boot_ci": ci(bb), "mean_dstep": float(x.mean()), "mean_dy": float(z.mean()),
                    "pi_share_predicted_by_step": pi,
                    "pi_boot_ci": ci(pib[np.abs(ymb) >= 0.002]) if pi is not None else "not defined (|mean dy| < 0.002)"}
            pv[f"{a}:{y}"] = o["p_b_hc3"]
        res["dose_response"][a] = e
    for k, v in holm(pv).items():
        a, y = k.split(":")
        res["dose_response"][a][y]["p_holm12"] = v
    # (ii) stacked context x arm, arm effects + context FE + common slope
    if "NEUTRAL" in M:
        rows = [(i, a) for i in range(cell.n) for a in arms]
        ctxid = np.array([i for i, _ in rows])
        dstep = np.array([change(M, a, "step")[i] for i, a in rows])
        Dm = np.array([[a == b for b in CRIT] for _, a in rows], float)
        for y in ("s", "p", "align"):
            dy = np.array([change(M, a, y)[i] for i, a in rows])
            X = np.column_stack([Dm, dstep])
            L = {f"{a}-NEUTRAL": np.eye(4)[j] for j, a in enumerate(CRIT)} | {"slope_b": np.eye(4)[3]}
            fit = fe_ols(dy, X, ctxid, rng=rng, L=L)
            sec = {}
            # arm-specific slopes, contrasts evaluated at the NEUTRAL mean step change
            mN = float(np.nanmean(change(M, "NEUTRAL", "step")))
            Sl = np.column_stack([dstep * (np.array([a for _, a in rows]) == b) for b in arms])
            X2 = np.column_stack([Dm, Sl])
            L2 = {}
            for j, a in enumerate(CRIT):
                l = np.zeros(X2.shape[1])
                l[j] = 1
                l[3 + arms.index(a)] += mN
                l[3 + arms.index("NEUTRAL")] -= mN
                L2[f"{a}-NEUTRAL_at_NEUTRAL_mean_dstep"] = l
            sec["arm_specific_slopes"] = strip(fe_ols(dy, X2, ctxid, rng=rng, L=L2))
            # paired-difference regression
            pdr = {}
            for a in CRIT:
                dd = change(M, a, y) - change(M, "NEUTRAL", y)
                xx = change(M, a, "step") - change(M, "NEUTRAL", "step")
                ok = ~np.isnan(dd) & ~np.isnan(xx)
                o = ols_hc3(xx[ok], dd[ok])
                D = draws(int(ok.sum()), rng)
                xb, yb = xx[ok][D], dd[ok][D]
                xm, ym = xb.mean(1, keepdims=True), yb.mean(1, keepdims=True)
                bb = ((xb - xm) * (yb - ym)).sum(1) / ((xb - xm) ** 2).sum(1)
                pdr[a] = {"alpha": o["a"], "alpha_t_ci95": o["a_t_ci95"], "alpha_boot_ci": ci(ym[:, 0] - bb * xm[:, 0]),
                          "beta": o["b"], "n": int(ok.sum())}
            sec["paired_difference"] = pdr
            res["adjusted"][y] = {"common_slope": strip(fit), "secondary": sec,
                                  "mean_dstep_stacked": float(np.nanmean(dstep[~np.isnan(dy)]))}
    # (iii) cross-section without revision
    I = eng["items_r0"]
    Fu = eng["items_fu"]
    for nm, col in (("s", 2), ("p", 3), ("align", 4)):
        e = {}
        if len(I):
            e["round0_ideas"] = strip(fe_ols(I[:, col], 1 - I[:, 1], I[:, 0].astype(int), rng=rng, L={"slope": [1.0]}))
        if len(Fu):
            e["followups_loo"] = strip(fe_ols(Fu[:, col], 1 - Fu[:, 1], Fu[:, 0].astype(int), rng=rng, L={"slope": [1.0]}))
        res["cross_section"][nm] = e
    return res


def t2_classify(cell_name: str, res2: dict, t3orig: dict) -> dict:
    """Adjusted feedback-specific turns classified with the cell's original rule."""
    out = {}
    adj = res2["adjusted"]
    if not adj:
        return out
    if cell_name == "DISC-4o":
        p = {f"{a}:{y}": adj[y]["common_slope"][f"{a}-NEUTRAL"]["p_cluster_t"] for y in ("s", "align") for a in CRIT}
        h = holm({k: (v if v is not None else 1.0) for k, v in p.items()})
        orig = t3orig["n_family"]
        for k, v in h.items():
            a, y = k.split(":")
            nm = "N6" if y == "s" else "N5"
            o = orig[f"{nm}:{a}-NEUTRAL:{y}"]["holm_significant"]
            out[k] = {"adjusted_p_holm6": v, "adjusted_holm_significant": v < 0.05, "unadjusted_holm_significant_N14": o,
                      "changed": bool((v < 0.05) != o)}
    elif cell_name == "REP-4o":
        orig = t3orig["rep_family"]
        p = {k: min(1.0, 2 * v["p_tost"]) for k, v in orig.items() if k.startswith("E")} | \
            {k: v["p"] for k, v in orig.items() if k.startswith("R")}
        conf = {k: dict(v) for k, v in orig.items()}
        for j, a in enumerate(CRIT):
            e = adj["s"]["common_slope"][f"{a}-NEUTRAL"]
            k = f"E{j + 4}:{a}-NEUTRAL:s"
            df = adj["s"]["common_slope"]["n_clusters"] - 1
            conf[k] = {"adjusted_mean": e["est"], **tost_from(e["est"], e["se_cluster"], df)}
            p[k] = min(1.0, 2 * conf[k]["p_tost"])
        conf = rep_readings(conf, p)
        for j, a in enumerate(CRIT):
            k = f"E{j + 4}:{a}-NEUTRAL:s"
            out[k] = {"adjusted": conf[k], "unadjusted_reading": orig[k]["reading"],
                      "changed": conf[k]["reading"] != orig[k]["reading"]}
    elif cell_name == "OLMO-C":
        orig = t3orig["olmo_family"]
        p = {a: adj["s"]["common_slope"][f"{a}-NEUTRAL"]["p_cluster_t"] or 1.0 for a in CRIT}
        h = holm(p)
        for a in CRIT:
            e = adj["s"]["common_slope"][f"{a}-NEUTRAL"]
            lo, hi = e["boot_ci"]
            r = olmo_reading(h[a] < 0.05, lo, hi)
            out[a] = {"adjusted_mean": e["est"], "boot_ci": e["boot_ci"], "p_holm3": h[a], "reading": r,
                      "unadjusted_reading": orig[a]["reading"], "changed": r != orig[a]["reading"]}
    return out


# =====================================================================================================  T3
def genre_dirs(cell: Cell) -> dict:
    crit = [n for n, r in enumerate(cell.rows) if r["arm"] in CRIT]
    I = cell.V[crit]
    Ctx = cell.C
    Fu = np.concatenate(cell.F)
    R = np.concatenate([Ctx, Fu])
    mI = I.mean(0)
    out = {"primary": unit(mI - R.mean(0)), "ctx": unit(mI - Ctx.mean(0)), "fu": unit(mI - Fu.mean(0)),
           "_norm_primary": float(np.linalg.norm(mI - R.mean(0)))}
    # leave-one-context-out primary
    ictx = np.array([cell.rows[n]["ctx"] for n in crit])
    sI, nI = I.sum(0), len(I)
    sR, nR = R.sum(0), len(R)
    G = np.zeros((cell.n, Ctx.shape[1]), np.float32)
    for i in range(cell.n):
        Ii = I[ictx == i]
        Ri = np.concatenate([Ctx[i:i + 1], cell.F[i]])
        G[i] = unit((sI - Ii.sum(0)) / (nI - len(Ii)) - (sR - Ri.sum(0)) / (nR - len(Ri)))
    out["loco"] = G
    if cell.name == "DISC-4o" and cell.pb is not None:
        out["ctx_ideas"] = unit(cell.pb[1].mean(0) - R.mean(0))
    return out


def genre_diag(cell: Cell, g) -> dict:
    a0, a2, af = [], [], []
    for i in range(cell.n):
        c = cell.C[i]
        af += (tang(cell.F[i], c) @ g).tolist()
        for a in CRIT:
            for r, acc in ((0, a0), (2, a2)):
                idx = cell.index.get((a, i, r))
                if idx:
                    acc += (tang(cell.V[idx], c) @ g).tolist()
    return {"cos_g_idea_tangent_r0": float(np.nanmean(a0)), "cos_g_idea_tangent_r2": float(np.nanmean(a2)),
            "cos_g_followup_tangent": float(np.nanmean(af))}


def recompute_tables(cell: Cell, eng: dict, rng) -> dict:
    M = eng["M"]
    out = {"followups_loo_alignment": float(np.nanmean(eng["loo_align"])), "G": float(np.nanmean(eng["fu_s"] - eng["I0_s"])),
           "step_gap_b1": float(np.nanmean(eng["gap_b1"])) if np.isfinite(eng["gap_b1"]).any() else None,
           "step_gap_r0": float(np.nanmean(eng["gap_r0"]))}
    if cell.name in ("DISC-4o", "DISC-OLMo"):
        out["table3"] = table3_cell(eng)
        out["n_family"] = n_family(M, rng, tag="N" if cell.name == "DISC-4o" else "O")
    if cell.name == "MOD":
        out["n_family"] = n_family(M, rng)
    if cell.name == "REP-4o":
        out["rep_family"] = rep_family(M, rng)
    if cell.name == "OLMO-C":
        out["olmo_family"] = olmo_family(M, rng)
    return out


def t3(cell: Cell, corp: Corpus, eng0: dict, rng) -> dict:
    gd = genre_dirs(cell)
    res = {"variants": {}, "diagnostics": {"norm_meanI_minus_meanR": gd["_norm_primary"]}}
    res["variants"]["none"] = recompute_tables(cell, eng0, rng)
    for v in ("primary", "ctx", "fu", "loco", "ctx_ideas"):
        if v not in gd:
            continue
        eng = engine(cell, corp, g=gd[v])
        res["variants"][v] = recompute_tables(cell, eng, rng)
        if v != "loco":
            res["diagnostics"][v] = genre_diag(cell, gd[v])
        log(f"  T3 {cell.name}/{cell.space} variant {v} done")
    return res


def t3_survival(t3s: dict) -> dict:
    """t3s[(cell, space)] -> t3 result. Applies the fixed 'survives' definitions per variant."""
    out = {}
    variants = sorted({v for r in t3s.values() for v in r["variants"]})
    for v in variants:
        e = {}
        cells3 = {}
        for (cell, space), r in t3s.items():
            if cell in ("DISC-4o", "DISC-OLMo") and v in r["variants"]:
                gen = "gpt-4o" if cell == "DISC-4o" else "olmo-7b-0424"
                cells3[f"{gen}/{space}"] = r["variants"][v]["table3"]
        if cells3:
            e["table3_holm36"] = table3_holm(cells3)
        cats = {}
        for (cell, space), r in t3s.items():
            if v not in r["variants"]:
                continue
            x = r["variants"][v]
            if "n_family" in x:
                for k, t in x["n_family"].items():
                    if k[1] in "56":
                        cats[f"{cell}/{space} {k}"] = t["holm_significant"]
            if "rep_family" in x:
                for k, t in x["rep_family"].items():
                    if k.startswith("E"):
                        cats[f"{cell}/{space} {k}"] = t["reading"]
            if "olmo_family" in x:
                for k, t in x["olmo_family"].items():
                    cats[f"{cell}/{space} {k}"] = t["reading"]
            cats[f"{cell}/{space} followups_loo_alignment"] = x["followups_loo_alignment"]
            cats[f"{cell}/{space} step_gap_b1"] = x["step_gap_b1"]
            cats[f"{cell}/{space} step_gap_r0"] = x["step_gap_r0"]
            cats[f"{cell}/{space} G"] = x["G"]
        e["categories"] = cats
        out[v] = e
    if "none" in out:
        base = out["none"]["categories"]
        for v, e in out.items():
            if v == "none":
                continue
            e["changed_vs_none"] = {k: [base[k], val] for k, val in e["categories"].items()
                                    if k in base and isinstance(val, (str, bool)) and val != base[k]}
            e["loo_alignment_ratio_vs_none"] = {k: (val / base[k] if base[k] else None) for k, val in e["categories"].items()
                                                if k.endswith("followups_loo_alignment") and k in base}
            e["step_gap_b1_ratio_vs_none"] = {k: (val / base[k] if base.get(k) else None) for k, val in e["categories"].items()
                                              if k.endswith("step_gap_b1") and val is not None and k in base}
            if "table3_holm36" in e and "table3_holm36" in out["none"]:
                b, x = out["none"]["table3_holm36"], e["table3_holm36"]
                e["table3_reading"] = {
                    "alignment_rises_holm_cells": len(x["alignment_rises_holm"]),
                    "alignment_rises_holm_cells_none": len(b["alignment_rises_holm"]),
                    "share_p_above_half_cells": len(x["share_p_above_half"]),
                    "along_s_significant_kept": {k: (k in x["along_s_holm_significant"] and
                                                     np.sign(x["along_s_holm_significant"][k]) == np.sign(val))
                                                 for k, val in b["along_s_holm_significant"].items()},
                    "new_along_s_significant": sorted(set(x["along_s_holm_significant"]) - set(b["along_s_holm_significant"]))}
    return out


# =====================================================================================================  T4
TOK = re.compile(r"[a-z0-9]+")


def toks(s: str) -> list[str]:
    return TOK.findall(unicodedata.normalize("NFKC", s).lower())


def ngrams(t, n):
    return {tuple(t[j:j + n]) for j in range(len(t) - n + 1)}


def containment(x, ctxg, n):
    gx = ngrams(x, n)
    return len(gx & ctxg[n]) / len(gx) if gx else np.nan


def lcs_len(a: list, b: list) -> int:
    """Bit-parallel LCS length (Allison-Dix / Hyyro)."""
    if not a or not b:
        return 0
    pos = {}
    for j, t in enumerate(b):
        pos[t] = pos.get(t, 0) | (1 << j)
    m = len(b)
    mask = (1 << m) - 1
    S = mask
    for t in a:
        U = S & pos.get(t, 0)
        S = ((S + U) | (S - U)) & mask
    return m - bin(S).count("1")


def overlap_feats(text: str, ctx_tok: list, ctxg: dict) -> list[float]:
    t = toks(text)
    return [containment(t, ctxg, n) for n in (1, 2, 3, 4)] + [lcs_len(t, ctx_tok) / len(t) if t else np.nan]


def t4_texts(cell: Cell) -> dict:
    """Overlap features (C1..C4, ROUGE-L recall vs the context) for the cell's items; text-only (embedder-free)."""
    import continue_from_context as CFC
    ids = [c["arxiv_id"] for c in cell.ctx]
    ct = corpus_texts(ids)
    feats = {"ideas": [], "ideas_r2": [], "fu": []}
    for i, c in enumerate(cell.ctx):
        name, ab = ct[c["arxiv_id"]]
        ctx_text = name + " " + ab[:1200]  # what the generator saw (paper_block: title + abstract[:1200])
        assert CFC.paper_block(name, ab).endswith(ab[:1200])
        tk = toks(ctx_text)
        g = {n: ngrams(tk, n) for n in (1, 2, 3, 4)}
        for j, f in enumerate(c["followups"]):
            feats["fu"].append([i, j] + overlap_feats(f["title"] + " " + f["abstract"], tk, g))
        if cell.pb is not None:
            for n in cell.pb_index.get(i, []):
                r = cell.pb[0][n]
                feats["ideas"].append([i, n] + overlap_feats(r["title"] + " " + r["abstract"], tk, g))
        else:
            for a in CRIT:
                for n in cell.index.get((a, i, 0), []):
                    r = cell.rows[n]
                    feats["ideas"].append([i, n] + overlap_feats(r["title"] + " " + r["abstract"], tk, g))
        if cell.name == "DISC-4o":
            for a in CRIT:
                for rr in (0, 2):
                    for n in cell.index.get((a, i, rr), []):
                        r = cell.rows[n]
                        feats["ideas_r2"].append([i, n, rr, CRIT.index(a)] + overlap_feats(r["title"] + " " + r["abstract"], tk, g))
    return {k: np.array(v, float) for k, v in feats.items()}


def t4_overlap(cell: Cell, F4: dict, rng) -> dict:
    """Per context: median C3 over ideas minus median over follow-ups / continuations / background."""
    I, Fu = F4["ideas"], F4["fu"]
    lab = cell.labels or {}
    res = {"idea_set": "one-shot CTX" if cell.pb is not None else "round-0 critique ideas (GEO, LLM, ORACLE)",
           "means": {}}
    names = ("C1", "C2", "C3", "C4", "ROUGE_L_recall")
    for k, nm in enumerate(names):
        res["means"][nm] = {"ideas": float(np.nanmean(I[:, 2 + k])), "followups": float(np.nanmean(Fu[:, 2 + k]))}
        if lab:
            cm = np.array([int(r[1]) in lab.get(int(r[0]), {}).get("cont", []) for r in Fu])
            res["means"][nm]["continuations"] = float(np.nanmean(Fu[cm, 2 + k])) if cm.any() else None
    for ref in ("followups", "continuations", "background"):
        if ref != "followups" and not lab:
            continue
        diffs = []
        for i in range(cell.n):
            xi = I[I[:, 0] == i][:, 4]
            fi = Fu[Fu[:, 0] == i]
            if ref == "followups":
                yi = fi[:, 4]
            else:
                sel = lab.get(i, {}).get("cont" if ref == "continuations" else "bg", [])
                yi = fi[np.isin(fi[:, 1], sel)][:, 4]
            if len(xi) and len(yi):
                diffs.append(np.nanmedian(xi) - np.nanmedian(yi))
        d = np.array(diffs)
        res[f"C3_ideas_minus_{ref}"] = {"mean": float(np.mean(d)), "ci": boot_mean(d, rng), "p_wilcoxon": wil(d), "n_contexts": int(len(d))}
    return res


def t4_echo(cell: Cell, eng_items: dict, F4: dict, rng) -> dict:
    """Echo-adjusted step gap: step = u_i + gamma*[idea] + beta*C3 (+ beta2*C1), within context."""
    I, Fu = F4["ideas"], F4["fu"]
    lab = cell.labels or {}
    step_i = eng_items["idea_step"]
    step_f = np.array([float(cell.F[int(r[0])][int(r[1])] @ cell.C[int(r[0])]) for r in Fu])
    out = {}
    for ref in ("followups", "continuations"):
        if ref == "continuations" and not lab:
            continue
        fm = np.ones(len(Fu), bool) if ref == "followups" else \
            np.array([int(r[1]) in lab.get(int(r[0]), {}).get("cont", []) for r in Fu])
        ctxs = set(Fu[fm, 0].astype(int))
        im = np.isin(I[:, 0].astype(int), list(ctxs))
        y = np.concatenate([step_i[im], step_f[fm]])
        isidea = np.concatenate([np.ones(im.sum()), np.zeros(fm.sum())])
        c3 = np.concatenate([I[im, 4], Fu[fm, 4]])
        c1 = np.concatenate([I[im, 2], Fu[fm, 2]])
        gid = np.concatenate([I[im, 0], Fu[fm, 0]]).astype(int)
        ok = np.isfinite(c3) & np.isfinite(c1)
        y, isidea, c3, c1, gid = y[ok], isidea[ok], c3[ok], c1[ok], gid[ok]
        f0 = fe_ols(y, isidea, gid, rng=rng, L={"gamma0": [1.0]})
        W = f0["_W"]
        f1 = fe_ols(y, np.column_stack([isidea, c3]), gid, W=W, L={"gamma": [1, 0], "beta_C3": [0, 1]})
        f2 = fe_ols(y, np.column_stack([isidea, c3, c1]), gid, W=W, L={"gamma": [1, 0, 0], "beta_C3": [0, 1, 0], "beta_C1": [0, 0, 1]})
        sh = 1 - f1["_draws"][:, 0] / f0["_draws"][:, 0]
        sh2 = 1 - f2["_draws"][:, 0] / f0["_draws"][:, 0]
        out[ref] = {"gamma0": strip(f0)["gamma0"], "with_C3": strip(f1), "with_C3_C1": strip(f2),
                    "echo_share": float(1 - f1["gamma"]["est"] / f0["gamma0"]["est"]), "echo_share_ci": ci(sh),
                    "echo_share_C3_C1": float(1 - f2["gamma"]["est"] / f0["gamma0"]["est"]), "echo_share_C3_C1_ci": ci(sh2),
                    "n_contexts": len(ctxs)}
    return out


def idea_steps(cell: Cell, F4: dict) -> np.ndarray:
    I = F4["ideas"]
    if cell.pb is not None:
        return np.array([float(cell.pb[1][int(r[1])] @ cell.C[int(r[0])]) for r in I])
    return np.array([float(cell.V[int(r[1])] @ cell.C[int(r[0])]) for r in I])


def t4_critique(cell: Cell, F4: dict, eng: dict) -> dict:
    X = F4["ideas_r2"]
    if not len(X):
        return {}
    out = {}
    for j, a in enumerate(CRIT):
        dc3 = np.full(cell.n, np.nan)
        for i in range(cell.n):
            r0 = X[(X[:, 0] == i) & (X[:, 3] == j) & (X[:, 2] == 0)][:, 6]
            r2 = X[(X[:, 0] == i) & (X[:, 3] == j) & (X[:, 2] == 2)][:, 6]
            if len(r0) and len(r2):
                dc3[i] = np.nanmean(r2) - np.nanmean(r0)
        dst = change(eng["M"], a, "step")
        ok = ~np.isnan(dc3) & ~np.isnan(dst)
        out[a] = {"mean_change_C3_r0_to_r2": float(np.nanmean(dc3)), "spearman_with_dstep": float(stats.spearmanr(dc3[ok], dst[ok])[0]),
                  "n": int(ok.sum())}
    return out


# =====================================================================================================  T5
def study_contexts() -> dict:
    import continue_from_context as CFC
    import strengthen_critique_replication as SCR
    import revision_null_effects as RN
    el = json.loads((DATA / "runs/olmo-direction/eligible.json").read_text())
    frame = olmo_frame_rows()
    olmo = [frame[p] for p in sorted(el["eligible_pos"])]
    return {"DISC": CFC.contexts(), "REP": SCR.ctx_rows(), "OLMO-C": olmo, "MOD": RN.contexts("followups-2025")}


def components(ctx: list) -> dict:
    owner = {}
    for i, c in enumerate(ctx):
        for f in c["followups"]:
            owner.setdefault(f["arxiv_id"], set()).add(i)
    parent = list(range(len(ctx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s in owner.values():
        s = sorted(s)
        for x in s[1:]:
            ra, rb = find(s[0]), find(x)
            if ra != rb:
                parent[rb] = ra
    comp = np.array([find(i) for i in range(len(ctx))])
    _, comp = np.unique(comp, return_inverse=True)
    shared = {x for x, s in owner.items() if len(s) >= 2}
    sizes = np.bincount(comp)
    return {"comp": comp, "shared": shared, "summary": {
        "n_contexts": len(ctx), "distinct_followups": len(owner), "followups_under_2plus_contexts": len(shared),
        "contexts_sharing_at_least_one": int(sum(any(f["arxiv_id"] in shared for f in c["followups"]) for c in ctx)),
        "n_components": int(len(sizes)), "component_size_distribution": {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}}}


def comp_boot(x, comp, rng):
    """Cluster bootstrap of a mean over contexts with components as resampling units."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x, cc = x[ok], comp[ok]
    _, cc = np.unique(cc, return_inverse=True)
    G = cc.max() + 1
    S, N = np.bincount(cc, weights=x, minlength=G), np.bincount(cc, minlength=G).astype(float)
    D = draws(G, rng)
    return ci(S[D].sum(1) / N[D].sum(1))


def t5_cell(cell: Cell, corp: Corpus, eng: dict, comp_info: dict, rng) -> dict:
    comp = comp_info["comp"][:cell.n]
    M = eng["M"]
    res = {"component_bootstrap": {}}
    cb = res["component_bootstrap"]
    if cell.name == "DISC-4o":
        cb["B1_step_gap"] = {"mean": float(np.nanmean(eng["gap_b1"])), "ci_components": comp_boot(eng["gap_b1"], comp, rng)}
        for a in CRIT:
            cb[f"table3_along_s_{a}"] = {"mean": float(np.nanmean(change(M, a, "s"))), "ci_components": comp_boot(change(M, a, "s"), comp, rng)}
            d = change(M, a, "s") - change(M, "NEUTRAL", "s")
            cb[f"N6_{a}"] = {"mean": float(np.nanmean(d)), "ci_components": comp_boot(d, comp, rng)}
    if cell.name == "REP-4o":
        for j, a in enumerate(CRIT):
            d1 = change(M, a, "s")
            d2 = d1 - change(M, "NEUTRAL", "s")
            for k, d in ((f"E{j + 1}", d1), (f"E{j + 4}", d2)):
                lo, hi = comp_boot(d, comp, rng)
                cb[k + f"_{a}"] = {"mean": float(np.nanmean(d)), "ci_components": [lo, hi],
                                   "ci_reading": "inside ±0.015" if -BOUND < lo and hi < BOUND else
                                   "excludes 0" if lo > 0 or hi < 0 else "includes 0, not inside ±0.015"}
    if cell.name == "OLMO-C":
        orig = olmo_family(M, rng)
        for a in CRIT:
            d = change(M, a, "s") - change(M, "NEUTRAL", "s")
            lo, hi = comp_boot(d, comp, rng)
            r = olmo_reading(orig[a]["significant_holm"], lo, hi)
            cb[f"{a}-NEUTRAL_s"] = {"mean": float(np.nanmean(d)), "ci_components": [lo, hi], "reading_with_component_ci": r,
                                    "reading_context_bootstrap": orig[a]["reading"], "changed": r != orig[a]["reading"]}
    # secondary: drop every shared follow-up from f and from the step gap
    shared = comp_info["shared"]
    fs = [np.array([j for j in cell.fset[i] if cell.ctx[i]["followups"][j]["arxiv_id"] not in shared], dtype=int)
          for i in range(cell.n)]
    if cell.name in ("DISC-4o", "REP-4o", "OLMO-C"):
        e2 = engine(cell, corp, fset=fs)
        sec = {"n_followups_dropped_from_f": int(sum(len(a) - len(b) for a, b in zip(cell.fset, fs))),
               "contexts_with_empty_f": int(sum(len(b) == 0 for b in fs))}
        if cell.name == "DISC-4o":
            sec["B1_step_gap"] = float(np.nanmean(e2["gap_b1"]))
            sec["table3_along_s"] = {a: float(np.nanmean(change(e2["M"], a, "s"))) for a in CRIT}
            sec["n_family"] = n_family(e2["M"], rng)
        if cell.name == "REP-4o":
            sec["rep_family"] = rep_family(e2["M"], rng)
        if cell.name == "OLMO-C":
            sec["olmo_family"] = olmo_family(e2["M"], rng)
        res["drop_shared_followups"] = sec
    return res


# =====================================================================================================  T6
def q(x):
    x = np.asarray(x, float)
    return {f"q{p}": float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)} | {"share_ge5": float((x >= 5).mean()), "n": int(len(x))}


def t6_frames() -> dict:
    import strengthen_critique_replication as SCR
    disc = [json.loads(l) for l in (DATA / "runs/followups/followups.jsonl").read_text().splitlines()]
    elig = [n for n, r in enumerate(disc) if r["n_followups"] >= 5][:150]
    last = elig[-1]
    out = {"DISC": {"n_citing_arxiv": {"frame_all_1000": q([r["n_citing_arxiv"] for r in disc]),
                                       "frame_traversed": q([r["n_citing_arxiv"] for r in disc[:last + 1]]),
                                       "sample_150": q([disc[n]["n_citing_arxiv"] for n in elig])},
                    "n_followups": {"frame_all_1000": q([r["n_followups"] for r in disc]),
                                    "frame_traversed": q([r["n_followups"] for r in disc[:last + 1]]),
                                    "sample_150": q([disc[n]["n_followups"] for n in elig])},
                    "frame_rows_traversed": last + 1}}
    fr = sorted((json.loads(l) for l in (DATA / "runs/olmo-direction/followups.jsonl").read_text().splitlines()),
                key=lambda r: r["frame_pos"])
    byfp = {r["frame_pos"]: r for r in fr}
    rep = [byfp[c["frame_pos"]] for c in SCR.ctx_rows()]
    el = json.loads((DATA / "runs/olmo-direction/eligible.json").read_text())
    recs = {r["pos"]: r for r in el["contexts"]}
    olmo = [byfp[recs[p]["frame_pos"]] for p in sorted(el["eligible_pos"])]
    out["REP_OLMO_frame"] = {}
    for fld in ("n_citing_all", "n_citing_arxiv_all", "n_followups_uncapped"):
        vals = lambda rows: [r.get(fld) for r in rows if r.get(fld) is not None]  # noqa: E731
        out["REP_OLMO_frame"][fld] = {"frame": q(vals(fr)), "REP_150": q(vals(rep)), "OLMO-C_150": q(vals(olmo))}
    out["_cites"] = {"DISC": [disc[n]["n_citing_arxiv"] for n in elig],
                     "REP": [r.get("n_citing_arxiv_all") for r in rep], "OLMO-C": [r.get("n_citing_arxiv_all") for r in olmo],
                     "REP_all": [r.get("n_citing_all") for r in rep], "OLMO-C_all": [r.get("n_citing_all") for r in olmo]}
    return out


def spearman_boot(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    rho = float(stats.spearmanr(x, y)[0])
    D = draws(len(x), rng)
    rx = stats.rankdata(x[D], axis=1)
    ry = stats.rankdata(y[D], axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    return {"rho": rho, "ci": ci(rb), "n": int(len(x))}


def t6_cell(cell: Cell, eng: dict, cites: list, rng) -> dict:
    lc = np.log1p(np.asarray(cites[:cell.n], float))
    M = eng["M"]
    gap = eng["gap_b1"] if cell.name == "DISC-4o" else eng["gap_r0"]
    res = {"step_gap_definition": "B1 (one-shot CTX ideas)" if cell.name == "DISC-4o" else "round-0 critique ideas"}
    res["rho_step_gap"] = spearman_boot(lc, gap, rng)
    res["rho_GEO_ds"] = spearman_boot(lc, change(M, "GEO", "s"), rng)
    if "NEUTRAL" in M:
        res["rho_GEO_minus_NEUTRAL_ds"] = spearman_boot(lc, change(M, "GEO", "s") - change(M, "NEUTRAL", "s"), rng)
    t1_, t2_ = np.nanpercentile(lc, [100 / 3, 200 / 3])
    res["step_gap_bottom_tercile"] = float(np.nanmean(gap[lc <= t1_]))
    res["step_gap_top_tercile"] = float(np.nanmean(gap[lc > t2_]))
    return res


# =====================================================================================================  readings
def readings(res: dict) -> dict:
    """Mechanical evaluation of the fixed reading rules (the paper text follows from these flags)."""
    out = {}
    t1r = res.get("T1", {})
    conf = [k for k in ("REP-4o/qwen3e06", "OLMO-C/qwen3e06") if k in t1r]
    if conf:
        e = {}
        for a in CRIT:
            ub = [t1r[k]["psi_s"][a]["ci"][1] for k in conf]
            pts = {k: (t1r[k]["psi_s"][a]["value"], t1r[k]["psi_s"][a]["ci"]) for k in conf}
            e[a] = {"barely_kept": all(u is not None and u < 0.25 for u in ub),
                    "feedback_closed_X": [k for k, (v, c) in pts.items() if v is not None and v >= 0.10 and c[0] > 0],
                    "psi": pts}
        e["phi_against_claim"] = {k: [a for a in CRIT if (t1r[k]["phi_s"][a]["value"] or 0) >= 0.33] for k in conf}
        e["bound_over_G"] = {k: v["bound_0.015_over_G"]["value"] for k, v in t1r.items()}
        e["CONT_below_I0"] = {k: v.get("CONT_below_I0") for k, v in t1r.items() if k.startswith("DISC")}
        out["T1"] = e
    t2r = res.get("T2", {})
    prim = [k for k in ("DISC-4o/qwen3e06", "REP-4o/qwen3e06", "OLMO-C/qwen3e06") if k in t2r]
    if prim:
        e = {}
        for k in prim:
            adj = t2r[k]["adjusted"].get("align")
            if not adj:
                continue
            b = adj["common_slope"]["slope_b"]
            pis = [t2r[k]["dose_response"][a]["align"]["pi_share_predicted_by_step"] for a in CRIT]
            e[k] = {"b_align": b["est"], "b_align_boot_ci": b.get("boot_ci"),
                    "b_positive_ci_excl_0": bool(b.get("boot_ci") and b["boot_ci"][0] > 0),
                    "pi_align_by_arm": dict(zip(CRIT, pis)),
                    "majority_pi_ge_half": sum(p is not None and p >= 0.5 for p in pis) >= 2}
            e[k]["against_claim"] = e[k]["b_positive_ci_excl_0"] and e[k]["majority_pi_ge_half"]
            sb = t2r[k]["adjusted"]["s"]["common_slope"]["slope_b"]
            md = t2r[k]["adjusted"]["s"]["mean_dstep_stacked"]
            e[k]["s_b_times_mean_dstep"] = sb["est"] * md
            e[k]["s_not_explained_by_step"] = bool(sb.get("boot_ci") and sb["boot_ci"][0] <= 0 <= sb["boot_ci"][1]
                                                   and abs(sb["est"] * md) < 0.005)
            cs = t2r[k]["cross_section"]["align"]
            if "round0_ideas" in cs and "followups_loo" in cs:
                ri, rf = cs["round0_ideas"]["slope"], cs["followups_loo"]["slope"]
                e[k]["cross_section_ideas_slope_pos_and_above_followups"] = bool(ri["boot_ci"][0] > 0 and ri["est"] > rf["est"])
        e["any_against_claim"] = any(v.get("against_claim") for v in e.values() if isinstance(v, dict))
        e["turns_not_explained_by_step_all_primary"] = all(v.get("s_not_explained_by_step") for v in e.values()
                                                           if isinstance(v, dict) and "s_not_explained_by_step" in v)
        e["classification_changes"] = {k: t2r[k].get("classification") for k in prim}
        out["T2"] = e
    if "T3_survival" in res:
        s = res["T3_survival"].get("primary", {})
        out["T3"] = {"table3": s.get("table3_reading"), "changed_categories": s.get("changed_vs_none"),
                     "loo_alignment_ratio": s.get("loo_alignment_ratio_vs_none"),
                     "step_gap_b1_ratio": s.get("step_gap_b1_ratio_vs_none"),
                     "secondary_changes": {v: e.get("changed_vs_none") for v, e in res["T3_survival"].items()
                                           if v not in ("none", "primary")}}
    if "T4" in res:
        e = res["T4"].get("DISC-4o/qwen3e06", {}).get("echo", {})
        out["T4"] = {ref: {"echo_share": v["echo_share"], "ci": v["echo_share_ci"],
                           "band": "against (>=0.5)" if v["echo_share"] >= 0.5 else "qualify [0.2,0.5)" if v["echo_share"] >= 0.2 else "<0.2"}
                     for ref, v in e.items()}
    return out


# =====================================================================================================  driver
def save(res: dict, path: Path) -> None:
    path.write_text(json.dumps(strip(res), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "t1", "t2", "t3", "t4", "t5", "t6", "all"])
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--spaces", default=",".join(SPACES))
    ap.add_argument("--n-ctx", type=int, default=None, help="smoke only: first N contexts per cell")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="smoke only")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-gate-stop", action="store_true", help="smoke only: report the gate but do not stop")
    a = ap.parse_args()
    CFG["n_boot"], CFG["n_ctx"] = a.n_boot, a.n_ctx
    cells, spaces = a.cells.split(","), a.spaces.split(",")
    want = {"t1", "t2", "t3", "t4", "t5", "t6"} if a.cmd == "all" else ({a.cmd} if a.cmd != "gate" else set())
    out = Path(a.out)
    res = {"prereg": "docs/prereg/review_tangent_confounds.md", "seed": SEED, "n_boot": CFG["n_boot"],
           "n_ctx_subset": CFG["n_ctx"], "cells": cells, "spaces": spaces, "deviations": DEVIATIONS}
    engs = {}
    rngs = {}
    comp_info = None
    if want & {"t5"}:
        sc = study_contexts()
        comp_info = {s: components(c) for s, c in sc.items()}
        ids = {s: {f["arxiv_id"] for c in v for f in c["followups"]} for s, v in sc.items()}
        res["T5_counts"] = {s: v["summary"] for s, v in comp_info.items()} | {"cross_study": {
            f"{x}&{y}": len(ids[x] & ids[y]) for x, y in (("DISC", "REP"), ("DISC", "OLMO-C"), ("REP", "OLMO-C"),
                                                          ("DISC", "MOD"), ("REP", "MOD"), ("OLMO-C", "MOD"))}}
    frames = t6_frames() if "t6" in want else None
    if frames:
        res["T6_frames"] = {k: v for k, v in frames.items() if not k.startswith("_")}
    F4cache = {}
    for space in spaces:
        log(f"loading corpus {space}")
        corp = Corpus(space)
        for cname in cells:
            cell = load_cell(cname, space, corp)
            key = f"{cname}/{space}"
            rng = rngs.setdefault(key, np.random.default_rng(SEED))
            log(f"{key}: {cell.n} contexts, {len(cell.rows)} ideas, arms {cell.arms}")
            eng = engine(cell, corp)
            engs[(cname, space)] = eng
            g = gate({(cname, space): eng})
            res.setdefault("gate", {"tolerance": GATE_TOL, "rows": [], "passed": True})
            res["gate"]["rows"] += g["rows"]
            res["gate"]["max_abs_diff"] = max(res["gate"].get("max_abs_diff", 0.0), g["max_abs_diff"])
            res["gate"]["passed"] = bool(res["gate"]["passed"] and g["passed"])
            for r in g["rows"]:
                log(f"  gate {r['quantity']}: {r['recomputed']:.5f} vs {r['published']:.5f}")
            if not g["passed"] and not a.no_gate_stop:
                save(res, out)
                raise SystemExit(f"reproduction gate failed for {key} (max diff {g['max_abs_diff']:.2e}); fix before proceeding")
            if "t1" in want:
                res.setdefault("T1", {})[key] = t1(cell, eng, rng)
            if "t2" in want:
                r2 = t2(cell, eng, rng)
                res.setdefault("T2", {})[key] = r2
            if "t3" in want:
                r3 = t3(cell, corp, eng, rng)
                res.setdefault("T3", {})[key] = r3
                if "t2" in want:
                    res["T2"][key]["classification"] = t2_classify(cname, res["T2"][key], r3["variants"]["none"])
            if "t4" in want:
                if cname not in F4cache:
                    F4cache[cname] = t4_texts(cell)
                    res.setdefault("T4_overlap", {})[cname] = t4_overlap(cell, F4cache[cname], rng)
                F4 = F4cache[cname]
                e4 = {"echo": t4_echo(cell, {"idea_step": idea_steps(cell, F4)}, F4, rng)}
                if cname == "DISC-4o":
                    e4["critique_arms"] = t4_critique(cell, F4, eng)
                res.setdefault("T4", {})[key] = e4
            if "t5" in want:
                res.setdefault("T5", {})[key] = t5_cell(cell, corp, eng, comp_info[STUDY[cname]], rng)
            if "t6" in want:
                st = STUDY[cname]
                if st in ("DISC", "REP", "OLMO-C"):
                    res.setdefault("T6", {})[key] = t6_cell(cell, eng, frames["_cites"][st], rng)
            save(res, out)
            log(f"{key} done")
        del corp
    if "t3" in want:
        res["T3_survival"] = t3_survival({tuple(k.split("/")): v for k, v in res["T3"].items()})
    if "t4" in want and "T4_overlap" in res:
        p = {}
        for nm, ref in (("DISC-4o", "followups"), ("DISC-4o", "continuations"), ("OLMO-C", "continuations")):
            x = res["T4_overlap"].get(nm, {}).get(f"C3_ideas_minus_{ref}")
            if x:
                p[f"{nm} vs {ref}"] = x["p_wilcoxon"] if x["p_wilcoxon"] is not None else 1.0
        res["T4_primary_holm3"] = holm(p) if p else {}
    res["readings"] = readings(res)
    save(res, out)
    log(f"wrote {out}")


DEVIATIONS = [
    "Seeds: one numpy generator per cell x embedder (seed 20261114), drawn cells -> analyses (T1..T6) in the `all` run; "
    "the text-only part of T4 (n-gram overlap) uses the cell's generator of the first embedder processed (qwen3e06).",
    "DISC-4o ORACLE-H exists only as local Qwen3 vectors (with the local ORACLE vectors of the same file): T1's "
    "ORACLE-H - ORACLE fraction uses that pair (ORACLE@local) with the Kaggle context/follow-up vectors; no BGE-M3 value.",
    "T1 I0 pools the round-0 ideas of GEO, LLM and ORACLE (idea-level mean), not the mean of arm means.",
    "T2(ii)/T2 classification: Holm p-values of the step-adjusted contrasts come from the cluster-robust t-test "
    "(the unadjusted families used Wilcoxon); REP-4o's adjusted E4-E6 replace E4-E6 inside the original family of 8; "
    "the TOST of an adjusted contrast uses its cluster-robust SE with G-1 df; OLMO-C uses the cluster-bootstrap CI.",
    "T5 secondary (drop shared follow-ups): k for p is the number of follow-ups left in the f-set (k = |f-set| as in "
    "every cell script); OLMO-C's step gap keeps all follow-ups (its f-set is the consensus continuations).",
    "T6 within-sample gradient: log(1 + n_citing_arxiv) for DISC, log(1 + n_citing_arxiv_all) for REP and OLMO-C "
    "(arXiv citing counts in both frames); the all-venue n_citing_all is reported in the frame quantiles.",
    "Step gap of the cells without a one-shot arm (REP-4o, OLMO-C, and the 'r0' gap of every cell): median step of "
    "the context's round-0 critique ideas minus the median step of its follow-ups.",
]


if __name__ == "__main__":
    main()
