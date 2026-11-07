"""Do the Section 5 results hold for true continuations? (docs/prereg/strengthen_continuations.md)

  uv run python scripts/strengthen_continuations.py sample 1            # stage-1 pairs + batches
  uv run python scripts/strengthen_continuations.py label 1 codex 3     # gpt-6.1-sol via codex exec, 3 in parallel
  uv run python scripts/strengthen_continuations.py label 1 claude 3    # Claude Opus 5.5 via claude -p
  uv run python scripts/strengthen_continuations.py sample 2            # stage-2 enrichment (needs both stage-1 labels)
  OMP_NUM_THREADS=2 uv run python scripts/strengthen_continuations.py analyse

Raw batches and CLI outputs: ~/Documents/experiment-data/latent-gap-ideas/runs/strengthen-continuations/.
Outputs: docs/prereg/strengthen_continuations.json, docs/prereg/strengthen_continuations_labels.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = M.DATA / "runs/strengthen-continuations"
OUT = ROOT / "docs/prereg/strengthen_continuations.json"
OUT_LABELS = ROOT / "docs/prereg/strengthen_continuations_labels.json"
SEED, BATCH, N_BOOT, STAGE2_CAP = 20261105, 20, 2000, 1000
LABELS = ("CONTINUES", "USES", "BACKGROUND", "UNCLEAR")

PROMPT = """You are labelling how a later paper (FOLLOW-UP) relates to an earlier paper it cites (SOURCE), using only the two titles and abstracts given below. Do not use any tools, files or web search; answer from the text.

Labels:
- CONTINUES: builds directly on the source: extends, applies, improves or tests its method or result.
- USES: uses the source as a tool, dataset or baseline without building on its idea.
- BACKGROUND: cites it in passing / related work only.
- UNCLEAR: the abstracts do not allow a call.

Where the abstracts do not say how the source is used, lean to BACKGROUND or UNCLEAR rather than guessing a closer relation.

Return ONLY a JSON array with one object per pair, in input order, and nothing else:
[{"id": <id>, "label": "CONTINUES|USES|BACKGROUND|UNCLEAR", "reason": "<at most 25 words>"}, ...]

PAIRS:
"""


def corpus_text() -> dict:
    import pandas as pd
    m = pd.read_parquet(M.DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "abstract"])
    return {r.id: (r.name, r.abstract) for r in m.itertuples()}


def pair_record(ctx, by_id, i, j, pid):
    c, f = ctx[i], ctx[i]["followups"][j]
    name, abstract = by_id[c["arxiv_id"]]
    return {"id": pid, "ctx_index": i, "followup_index": j, "context_id": c["arxiv_id"], "followup_id": f["arxiv_id"],
            "source_title": name, "source_abstract": abstract, "followup_title": f["title"],
            "followup_abstract": f["abstract"]}


def write_batches(stage: int, pairs: list[dict], rng) -> None:
    d = WORK / f"stage{stage}"
    d.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(pairs))
    pairs = [pairs[k] for k in order]
    (d / "pairs.json").write_text(json.dumps(pairs, indent=1))
    for b in range(0, len(pairs), BATCH):
        (d / f"batch_{b // BATCH:03d}.json").write_text(json.dumps(pairs[b:b + BATCH], indent=1))
    print(f"stage {stage}: {len(pairs)} pairs, {-(-len(pairs) // BATCH)} batches, "
          f"{len({p['ctx_index'] for p in pairs})} contexts", flush=True)


def sample(stage: int) -> None:
    ctx = M.contexts()
    by_id = corpus_text()
    if stage == 1:
        rng = np.random.default_rng(SEED)
        three = set(rng.permutation(len(ctx))[:100].tolist())
        pairs = []
        for i, c in enumerate(ctx):
            k = 3 if i in three else 2
            for j in rng.choice(len(c["followups"]), k, replace=False):
                pairs.append(pair_record(ctx, by_id, i, int(j), f"s1-{i}-{int(j)}"))
        assert len(pairs) == 400
        write_batches(1, pairs, rng)
    else:
        lab = {L: labels(1, L) for L in ("codex", "claude")}
        p1 = json.loads((WORK / "stage1/pairs.json").read_text())
        hit = sorted({p["ctx_index"] for p in p1 if any(lab[L][p["id"]]["label"] == "CONTINUES" for L in lab)})
        rng = np.random.default_rng(SEED + 2)
        hit = [hit[k] for k in rng.permutation(len(hit))]
        done = {(p["ctx_index"], p["followup_index"]) for p in p1}
        pairs, used = [], []
        for i in hit:
            rest = [j for j in range(len(ctx[i]["followups"])) if (i, j) not in done]
            if len(pairs) + len(rest) > STAGE2_CAP:
                break
            used.append(i)
            pairs += [pair_record(ctx, by_id, i, j, f"s2-{i}-{j}") for j in rest]
        (WORK / "stage2_contexts.json").write_text(json.dumps({"eligible": sorted(hit), "used": sorted(used)}))
        write_batches(2, pairs, rng)


def prompt_for(batch: list[dict]) -> str:
    body = []
    for p in batch:
        body.append(f"### id {p['id']}\nSOURCE: {p['source_title']}\nSOURCE ABSTRACT: {p['source_abstract']}\n"
                    f"FOLLOW-UP: {p['followup_title']}\nFOLLOW-UP ABSTRACT: {p['followup_abstract']}\n")
    return PROMPT + "\n".join(body)


def parse(text: str, batch: list[dict]) -> list[dict] | None:
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        return None
    try:
        rows = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    by = {str(r.get("id")): r for r in rows if isinstance(r, dict)}
    out = []
    for p in batch:
        r = by.get(p["id"])
        if not r or r.get("label") not in LABELS:
            return None
        out.append({"id": p["id"], "label": r["label"], "reason": str(r.get("reason", ""))})
    return out


def run_cli(who: str, prompt: str, tmp: Path) -> str:
    scratch = WORK / "cwd"
    scratch.mkdir(exist_ok=True)
    if who == "codex":
        cmd = ["codex", "exec", "-m", "gpt-6.1-sol", "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
               "-C", str(scratch), "-o", str(tmp), prompt]
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=1500, check=False)
        return tmp.read_text() if tmp.exists() else ""
    cmd = ["claude", "-p", "--model", "claude-opus-5-5", "--output-format", "text",
           "--disallowedTools=Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task", prompt]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1500, check=False,
                       cwd=scratch)
    tmp.write_text(r.stdout)
    return r.stdout


def label_batch(stage: int, who: str, path: Path) -> str:
    out = WORK / f"stage{stage}" / f"labels_{who}_{path.stem}.json"
    if out.exists():
        return f"{path.stem} cached"
    batch = json.loads(path.read_text())
    prompt = prompt_for(batch)
    for attempt in range(3):
        t0 = time.time()
        raw = WORK / f"stage{stage}" / f"raw_{who}_{path.stem}_try{attempt}.txt"
        text = run_cli(who, prompt, raw)
        rows = parse(text, batch)
        if rows:
            out.write_text(json.dumps({"labeller": who, "attempt": attempt, "seconds": round(time.time() - t0),
                                       "rows": rows}, indent=1))
            return f"{path.stem} ok try{attempt} {time.time() - t0:.0f}s"
    return f"{path.stem} FAILED"


def label(stage: int, who: str, par: int) -> None:
    files = sorted((WORK / f"stage{stage}").glob("batch_*.json"))
    with ThreadPoolExecutor(par) as ex:
        for msg in ex.map(lambda f: label_batch(stage, who, f), files):
            print(who, msg, flush=True)


def labels(stage: int, who: str) -> dict:
    out = {}
    for f in sorted((WORK / f"stage{stage}").glob(f"labels_{who}_batch_*.json")):
        for r in json.loads(f.read_text())["rows"]:
            out[r["id"]] = r
    n = len(json.loads((WORK / f"stage{stage}/pairs.json").read_text()))
    assert len(out) == n, (stage, who, len(out), n)
    return out


# ----------------------------------------------------------------------------------------------------------- analysis

def kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == k) * np.mean(b == k) for k in cats))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def analyse() -> None:
    from scipy import stats
    import pandas as pd
    import continuation_reanalysis as R
    from direction_tangent import tang, unit
    import revision_neutral as RN

    ctx = M.contexts()
    nf = np.array([len(c["followups"]) for c in ctx])
    rng = np.random.default_rng(SEED)
    res: dict = {"prereg": "docs/prereg/strengthen_continuations.md",
                 "deviations": ["stage 2 batch_012 (Claude Opus 5.5): 3 runs omitted pair s2-37-28 (19 of 20 labels); "
                                "that pair was labelled in a single-pair call with the same prompt and merged "
                                "(labels_claude_batch_012.json, attempt field)",
                                "E4 has 8 consensus contexts with >= 2 continuations (< 10): descriptive only, as "
                                "pre-registered; the lenient set (15 contexts) is reported alongside"]}
    all_pairs = []
    stages = [s for s in (1, 2) if (WORK / f"stage{s}/pairs.json").exists()]
    for s in stages:
        P = json.loads((WORK / f"stage{s}/pairs.json").read_text())
        lc, ll = labels(s, "codex"), labels(s, "claude")
        for p in P:
            all_pairs.append({k: p[k] for k in ("id", "ctx_index", "followup_index", "context_id", "followup_id",
                                                 "source_title", "followup_title")}
                             | {"stage": s, "gpt61sol": lc[p["id"]]["label"], "gpt61sol_reason": lc[p["id"]]["reason"],
                                "opus55": ll[p["id"]]["label"], "opus55_reason": ll[p["id"]]["reason"]})
    OUT_LABELS.write_text(json.dumps(all_pairs, indent=1) + "\n")

    # ---- E1, E2 on stage 1
    s1 = [p for p in all_pairs if p["stage"] == 1]
    A = np.array([p["gpt61sol"] for p in s1]); B = np.array([p["opus55"] for p in s1])
    ci = np.array([p["ctx_index"] for p in s1])
    kc = np.bincount(ci, minlength=len(ctx))
    w = nf[ci] / kc[ci]
    defs = {"gpt61sol_CONTINUES": A == "CONTINUES", "opus55_CONTINUES": B == "CONTINUES",
            "consensus_CONTINUES": (A == "CONTINUES") & (B == "CONTINUES"),
            "lenient_CONTINUES": (A == "CONTINUES") | (B == "CONTINUES"),
            "consensus_CONTINUES_or_USES": np.isin(A, ["CONTINUES", "USES"]) & np.isin(B, ["CONTINUES", "USES"]),
            "gpt61sol_CONTINUES_or_USES": np.isin(A, ["CONTINUES", "USES"]),
            "opus55_CONTINUES_or_USES": np.isin(B, ["CONTINUES", "USES"])}
    uctx = np.unique(ci)
    groups = {i: np.where(ci == i)[0] for i in uctx}
    boots = [np.concatenate([groups[i] for i in rng.choice(uctx, len(uctx))]) for _ in range(N_BOOT)]
    E1 = {}
    for name, x in defs.items():
        x = x.astype(float)
        bw = [float((x[b] * w[b]).sum() / w[b].sum()) for b in boots]
        bu = [float(x[b].mean()) for b in boots]
        E1[name] = {"k": int(x.sum()), "n": len(x), "share_pairs": float(x.mean()),
                    "share_pairs_cluster_ci": [float(np.percentile(bu, 2.5)), float(np.percentile(bu, 97.5))],
                    "share_pairs_wilson": wilson(int(x.sum()), len(x)),
                    "share_followup_weighted": float((x * w).sum() / w.sum()),
                    "share_followup_weighted_ci": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
                    "contexts_with_one": int(len({int(i) for i in ci[x > 0]}))}
    res["E1_share_stage1"] = E1
    res["label_counts_stage1"] = {"gpt61sol": {k: int((A == k).sum()) for k in LABELS},
                                  "opus55": {k: int((B == k).sum()) for k in LABELS}}
    res["confusion_stage1"] = {a: {b: int(((A == a) & (B == b)).sum()) for b in LABELS} for a in LABELS}
    bA, bB = A == "CONTINUES", B == "CONTINUES"
    kb = [kappa(bA[b], bB[b]) for b in boots]
    k4 = [kappa(A[b], B[b]) for b in boots]
    bA2, bB2 = np.isin(A, ["CONTINUES", "USES"]), np.isin(B, ["CONTINUES", "USES"])
    res["E2_agreement_stage1"] = {
        "binary_CONTINUES": {"kappa": kappa(bA, bB), "ci": [float(np.nanpercentile(kb, 2.5)), float(np.nanpercentile(kb, 97.5))],
                             "raw_agreement": float(np.mean(bA == bB))},
        "binary_CONTINUES_or_USES": {"kappa": kappa(bA2, bB2), "raw_agreement": float(np.mean(bA2 == bB2))},
        "four_labels": {"kappa": kappa(A, B), "ci": [float(np.nanpercentile(k4, 2.5)), float(np.nanpercentile(k4, 97.5))],
                        "raw_agreement": float(np.mean(A == B))}}
    # overlap with the original 30-pair check and with Semantic Scholar's flags
    old = {(p["ctx_index"], p["followup_index"]): p["label"]
           for p in json.loads((ROOT / "docs/paper/followup_check_sample.json").read_text())["pairs"]}
    intents = json.loads((M.RUN / "followup_intents.json").read_text())

    def s2flag(i, j):
        v = intents.get(f"{i}/{j}", {})
        return bool(v) and bool(v.get("influential") or {"methodology", "result"} & set(v.get("intents", [])))
    cons_all = [p for p in all_pairs if p["gpt61sol"] == "CONTINUES" and p["opus55"] == "CONTINUES"]
    res["overlap"] = {
        "pairs_also_in_original_30": [{"id": p["id"], "original_claude": old[(p["ctx_index"], p["followup_index"])],
                                       "gpt61sol": p["gpt61sol"], "opus55": p["opus55"]}
                                      for p in all_pairs if (p["ctx_index"], p["followup_index"]) in old],
        "stage1_s2_flagged": int(sum(s2flag(p["ctx_index"], p["followup_index"]) for p in s1)),
        "stage1_consensus_and_s2_flagged": int(sum(s2flag(p["ctx_index"], p["followup_index"]) for p, x in
                                                   zip(s1, defs["consensus_CONTINUES"]) if x)),
        "stage1_s2_flagged_share_among_consensus": None}
    nc = int(defs["consensus_CONTINUES"].sum())
    if nc:
        res["overlap"]["stage1_s2_flagged_share_among_consensus"] = res["overlap"]["stage1_consensus_and_s2_flagged"] / nc
    # the S2-flagged robustness subset as analysed: flagged follow-ups of contexts with >= MIN_CONT (3) flagged ones
    n_flag_ctx = {i: sum(s2flag(i, j) for j in range(len(c["followups"]))) for i, c in enumerate(ctx)}
    kept = {i for i, n in n_flag_ctx.items() if n >= 3}
    in_sub = [s2flag(p["ctx_index"], p["followup_index"]) and p["ctx_index"] in kept for p in s1]
    res["overlap"]["s2_subset_retained_contexts"] = len(kept)
    res["overlap"]["stage1_in_s2_subset"] = int(sum(in_sub))
    res["overlap"]["stage1_consensus_in_s2_subset"] = int(sum(f and x for f, x in zip(in_sub, defs["consensus_CONTINUES"])))
    nn = [p for p, x in zip(s1, defs["consensus_CONTINUES"]) if not x]
    res["overlap"]["stage1_s2_flagged_share_among_not_consensus"] = float(np.mean([s2flag(p["ctx_index"], p["followup_index"]) for p in nn]))
    if 2 in stages:
        res["stage2"] = json.loads((WORK / "stage2_contexts.json").read_text()) | {
            "n_pairs": sum(p["stage"] == 2 for p in all_pairs)}

    # sets of continuation follow-up indices per context (stages 1 + 2)
    def sets(kind):
        S = {i: [] for i in range(len(ctx))}
        for p in all_pairs:
            a, b = p["gpt61sol"], p["opus55"]
            ok = (a == b == "CONTINUES") if kind == "consensus" else ("CONTINUES" in (a, b)) if kind == "lenient" \
                else (a == b == "BACKGROUND")
            if ok:
                S[p["ctx_index"]].append(p["followup_index"])
        return {i: sorted(v) for i, v in S.items() if v}
    CS = {k: sets(k) for k in ("consensus", "lenient", "background")}
    res["n_contexts_with_continuations"] = {
        k: {">=1": len(v), ">=2": sum(len(x) >= 2 for x in v.values()), ">=3": sum(len(x) >= 3 for x in v.values())}
        for k, v in CS.items() if k != "background"}
    res["n_consensus_continuations_total"] = len(cons_all)

    def boot_ci(x):
        x = np.asarray(x)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    def wil(x, alt="two-sided"):
        x = np.asarray(x)
        return float(stats.wilcoxon(x, alternative=alt).pvalue) if len(x) >= 2 and np.any(x != 0) else None

    # ---- E3 step gap
    E3 = {}
    for space in R.SPACES:
        for gen in ("gpt-4o", "olmo-7b-0424"):
            cctx, C, F, pb, PB, it, IT = R.load_generator(gen, space)
            for kind in ("consensus", "lenient"):
                g_c, g_all, rows = [], [], []
                for i, js in CS[kind].items():
                    idx = [n for n, r in enumerate(pb) if r["ctx"] == i and r["arm"] == "CTX"]
                    if not idx:
                        continue
                    ideas = float(np.median(PB[idx] @ C[i]))
                    sc = F[i][js] @ C[i]
                    g_c.append(ideas - float(np.median(sc)))
                    g_all.append(ideas - float(np.median(F[i] @ C[i])))
                    rows.append(i)
                g_c, g_all = np.array(g_c), np.array(g_all)
                cell = {"n_contexts": len(g_c), "gap_vs_continuations": float(g_c.mean()), "ci": boot_ci(g_c),
                        "p_one_sided_gt0": wil(g_c, "greater"), "share_contexts_gap_gt0": float(np.mean(g_c > 0)),
                        "gap_vs_all_followups_same_contexts": float(g_all.mean()), "ci_all": boot_ci(g_all),
                        "difference_cont_minus_all": float((g_c - g_all).mean()), "ci_difference": boot_ci(g_c - g_all),
                        "p_difference": wil(g_c - g_all)}
                if kind == "consensus":
                    d = []
                    for i, js in CS["consensus"].items():
                        if i in CS["background"]:
                            d.append(float((F[i][js] @ C[i]).mean() - (F[i][CS["background"][i]] @ C[i]).mean()))
                    cell["T2_continuation_minus_background_step"] = {"n_contexts": len(d), "mean": float(np.mean(d)) if d else None,
                                                                      "ci": boot_ci(d) if len(d) > 1 else None, "p": wil(d)}
                E3[f"{gen}/{space}/{kind}"] = cell
                print("E3", gen, space, kind, json.dumps({k: v for k, v in cell.items() if not isinstance(v, (list, dict))}), flush=True)
    res["E3_step_gap"] = E3

    # ---- E4 direction from continuations
    meta = pd.read_parquet(M.DATA / "prepared/arxiv/corpus.parquet", columns=["id", "split", "adult"])
    keep = ~meta["adult"].to_numpy()
    meta = meta[keep].reset_index(drop=True)
    fut = (meta["split"] == "future").to_numpy()
    ids = meta.loc[fut, "id"].to_numpy()
    nrows = RN.neutral_rows()
    E4 = {}
    for space in R.SPACES:
        Fe = R.norm(np.load(M.DATA / f"prepared/arxiv/emb_{space}.npy").astype(np.float32)[keep][fut])
        NV = R.norm(np.load(M.RUN / f"neutral_emb_{space}.npy").astype(np.float32))
        for gen in ("gpt-4o", "olmo-7b-0424"):
            cctx, C, F, pb, PB, it, IT = R.load_generator(gen, space)
            arms = R.I_ARMS + (("NEUTRAL",) if gen == "gpt-4o" else ())
            for kind in ("consensus", "lenient"):
                for pk in ("k_continuations", "k_followups"):
                    elig = {i: js for i, js in CS[kind].items() if len(js) >= 2}
                    ch = {a: {"cont": [], "orig": []} for a in arms}
                    loo, fp = [], []
                    for i, js in elig.items():
                        c = C[i]
                        TF = tang(F[i], c)
                        TC = TF[js]
                        S_ = TC.sum(0)
                        loo.append(float(np.mean([TC[j] @ unit(S_ - TC[j]) for j in range(len(TC))])))
                        fc, fo = unit(TC.mean(0)), unit(TF.mean(0))
                        fids = {f["arxiv_id"] for f in cctx[i]["followups"]}
                        pool = Fe[~np.isin(ids, list(fids))]
                        order = np.argsort(-(pool @ c))
                        kk = len(js) if pk == "k_continuations" else len(F[i])
                        tpc = unit(tang(pool[order[:kk]], c).mean(0))
                        tpo = unit(tang(pool[order[:len(F[i])]], c).mean(0))
                        sc_, so_ = unit(fc - (fc @ tpc) * tpc), unit(fo - (fo @ tpo) * tpo)
                        fp.append(float(fc @ tpc))
                        for a in arms:
                            src, V = (nrows, NV) if a == "NEUTRAL" else (it, IT)
                            v = {}
                            for r in (0, 2):
                                idx = [n for n, x in enumerate(src) if x["ctx"] == i and x["arm"] == a and x["round"] == r]
                                if idx:
                                    T = tang(V[idx], c)
                                    v[r] = (float((T @ sc_).mean()), float((T @ so_).mean()))
                            if 0 in v and 2 in v:
                                ch[a]["cont"].append(v[2][0] - v[0][0]); ch[a]["orig"].append(v[2][1] - v[0][1])
                            else:
                                ch[a]["cont"].append(np.nan); ch[a]["orig"].append(np.nan)
                    cell = {"n_contexts": len(elig), "continuations_loo_alignment": float(np.mean(loo)) if loo else None,
                            "mean_f_dot_p": float(np.mean(fp)) if fp else None}
                    for a in arms:
                        x, y = np.array(ch[a]["cont"]), np.array(ch[a]["orig"])
                        ok = ~np.isnan(x)
                        e = {"n": int(ok.sum())}
                        if ok.sum() >= 2:
                            e |= {"change_along_s_cont": float(x[ok].mean()), "ci": boot_ci(x[ok]), "p": wil(x[ok]),
                                  "change_along_s_original_same_contexts": float(y[ok].mean()), "ci_original": boot_ci(y[ok]),
                                  "p_original": wil(y[ok])}
                        if a != "NEUTRAL" and "NEUTRAL" in arms:
                            d = x - np.array(ch["NEUTRAL"]["cont"])
                            ok2 = ~np.isnan(d)
                            if ok2.sum() >= 2:
                                e["minus_NEUTRAL"] = {"mean": float(d[ok2].mean()), "ci": boot_ci(d[ok2]), "p": wil(d[ok2]),
                                                      "n": int(ok2.sum())}
                        cell[a] = e
                    E4[f"{gen}/{space}/{kind}/{pk}"] = cell
                    print("E4", gen, space, kind, pk, cell["n_contexts"],
                          {a: (round(cell[a].get("change_along_s_cont", np.nan), 4), cell[a].get("p")) for a in arms}, flush=True)
    res["E4_direction"] = E4
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("E1_share_stage1", "E2_agreement_stage1", "n_contexts_with_continuations")}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        sample(int(sys.argv[2]))
    elif cmd == "label":
        label(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    elif cmd == "analyse":
        analyse()
