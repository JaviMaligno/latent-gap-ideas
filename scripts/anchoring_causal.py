"""Round 5B: frozen design, offline prompt audit and paired causal analysis. Never calls an LLM.

uv run python scripts/anchoring_causal.py design     # reproduce pre-run numbers (stdout only)
uv run python scripts/anchoring_causal.py dry-check --out /tmp/anchoring-prompts
uv run python scripts/anchoring_causal.py analyze   # requires the frozen preregistration
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nct, norm, t as student_t

from latent_gap.cli_judges import render
from latent_gap.decode import Corpus, K_NEIGHBOURS, MAX_ITERS, N_CANDIDATES, SYSTEM, _listing, decode
from latent_gap.domains import DOMAINS
from run_pilot import DATA, swap_map

DOMAINS_ORDER = ("steam", "arxiv")
ARMS = ("REAL", "GAP", "FAR")
GENERATORS = {"luna": "codex:gpt-6-luna", "sonnet": "claude:claude-sonnet-5-5"}
CONDITIONS = ("true", "swap", "none")
KS = list(range(1, 21))
CALIBRATION = list(range(21, 50))
PREREG = Path(__file__).resolve().parents[1] / "docs/prereg/anchoring_causal.md"
ALPHA = 0.05 / 24  # 12 cells x 2 directional primary endpoints
ANCHOR_MIN = 0.05
DROP_MIN = {"REAL": 0.04, "GAP": 0.04, "FAR": 0.02}
MAE_MAX = {"REAL": 0.08, "GAP": 0.05, "FAR": 0.025}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed(domain, arm):
    return 20261028 + 100 * DOMAINS_ORDER.index(domain) + ARMS.index(arm)


def run_name(gen, condition, arm):
    return f"anchoring-causal/{gen}/{condition}/{arm}"


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_domain(domain):
    root = DATA / "prepared" / domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["name", "text", "split", "adult"])
    raw = np.load(root / "emb_qwen3e06.npy", mmap_mode="r")
    keep = (~meta.adult & meta.split.eq("past")).to_numpy()
    P = np.asarray(raw[keep], dtype=np.float32)
    pm = meta[keep]
    corpus = Corpus(P, pm.name.tolist(), pm.text.tolist(), DOMAINS[domain])
    with np.load(DATA / "runs" / domain / "pilot/targets.npz", allow_pickle=True) as z:
        tg = {k: z[k] for k in z.files}
    return corpus, unit(P), tg


def hidden(tg, arm, k):
    return np.asarray(tg[f"{arm}_hidden"][k], dtype=np.int64) if f"{arm}_hidden" in tg else np.array([], dtype=np.int64)


def geometry(corpus, P, tg, domain, arm):
    donors = swap_map(KS, seed(domain, arm))
    T = unit(tg[arm])
    nn, centroids, ceilings = {}, {}, {}
    for k in range(1, 50):
        nn[k] = corpus.top(tg[arm][k], K_NEIGHBOURS, hidden(tg, arm, k))
        centroids[k] = unit(P[nn[k]].mean(0))
        scores = P @ T[k]
        scores[hidden(tg, arm, k)] = -np.inf
        ceilings[k] = float(scores.max())
    for k, j in donors.items():
        # A donor must really supply its original eight; never expose recipient hidden REAL rows.
        if set(nn[j]) & set(hidden(tg, arm, k)):
            raise ValueError(f"frozen donor leaks hidden rows: {domain}/{arm}/{k}->{j}")
    return T, nn, centroids, ceilings, donors


def historical(domain, gen):
    model = GENERATORS[gen].split(":", 1)[1]
    rows = read_rows(DATA / "runs" / domain / f"grid-{model}" / "items.jsonl")
    by_id = {}
    for r in rows:
        key = (r["arm"], r["k"])
        if key in by_id:
            raise ValueError(f"duplicate historical row: {domain}/{gen}/{key}")
        by_id[key] = r
    return by_id


def predictions(T, C, ceiling, donors, alpha, kappa, prior):
    rows = []
    for k in KS:
        j = donors[k]
        for condition in CONDITIONS:
            c = C[j] if condition == "swap" else C[k]
            pred = prior if condition == "none" else alpha * c
            rows.append({"k": k, "donor_k": j, "condition": condition,
                         "target": float(pred @ T[k]), "donor": float(pred @ T[j]),
                         "shown": None if condition == "none" else alpha,
                         "kappa": kappa * ceiling[k],
                         "attraction": float(pred @ (C[j] - C[k]))})
    return rows


def power(effect, sd, minimum, n=20):
    """Noncentral-t significance power; normal plug-in joint effect-size/significance approximation."""
    se = sd / np.sqrt(n)
    crit = student_t.ppf(1 - ALPHA, n - 1)
    return {"sd_pair": sd, "effect": effect,
            "significance_power": float(nct.sf(crit, n - 1, effect / se)),
            "joint_power_approx": float(norm.sf((max(minimum, crit * se) - effect) / se)),
            "mde80_approx": float((crit + norm.ppf(0.8)) * se)}


def design():
    manifest = {"version": 1, "ks": KS, "calibration_ks": CALIBRATION,
                "generators": GENERATORS, "conditions": list(CONDITIONS),
                "alpha_per_endpoint": ALPHA, "anchor_min": ANCHOR_MIN,
                "drop_min": DROP_MIN, "mae_max": MAE_MAX,
                "sources": {}, "donors": {}, "calibration": {}}
    tables, powers = [], []
    for domain in DOMAINS_ORDER:
        sources = [DATA / "prepared" / domain / f for f in ("corpus.parquet", "emb_qwen3e06.npy")]
        sources += [DATA / "runs" / domain / "pilot/targets.npz"]
        sources += [DATA / "runs" / domain / f"grid-{g.split(':')[1]}/items.jsonl" for g in GENERATORS.values()]
        for p in sources:
            manifest["sources"][str(p.relative_to(DATA))] = digest(p)
        corpus, P, tg = load_domain(domain)
        geo = {a: geometry(corpus, P, tg, domain, a) for a in ARMS}
        for arm in ARMS:
            manifest["donors"][f"{domain}/{arm}"] = geo[arm][4]
        for gen in GENERATORS:
            old = historical(domain, gen)
            Treal, _, _, creal, _ = geo["REAL"]
            kappa = float(np.median([unit(old["REAL", k]["best"]["emb"]) @ Treal[k] / creal[k]
                                     for k in CALIBRATION]))
            prior = np.mean([unit(old["PLAIN", k]["best"]["emb"]) for k in CALIBRATION], axis=0)
            for arm in ARMS:
                T, nn, C, ceiling, donors = geo[arm]
                for k in CALIBRATION + KS:
                    if old[arm, k]["neighbours"] != nn[k]:
                        raise ValueError(f"historical neighbourhood mismatch: {domain}/{gen}/{arm}/{k}")
                alpha = float(np.mean([unit(old[arm, k]["best"]["emb"]) @ C[k] for k in CALIBRATION]))
                manifest["calibration"][f"{domain}/{gen}/{arm}"] = {"alpha": alpha, "kappa": kappa}
                pred = pd.DataFrame(predictions(T, C, ceiling, donors, alpha, kappa, prior))
                for cond, group in pred.groupby("condition", sort=False):
                    tables.append({"domain": domain, "gen": gen, "arm": arm, "condition": cond,
                                   **{key: float(group[key].mean()) for key in ("target", "shown", "donor", "kappa")}})
                # Observed historical per-item spread, not invented swapped outcomes; rho=0 working assumption.
                E = np.stack([unit(old[arm, k]["best"]["emb"]) for k in KS])
                landing = np.array([e @ T[k] for e, k in zip(E, KS)])
                contrast = np.array([e @ (C[donors[k]] - C[k]) for e, k in zip(E, KS)])
                pt = pred[pred.condition == "true"].set_index("k")
                ps = pred[pred.condition == "swap"].set_index("k")
                for endpoint, effect, sd, minimum in (
                    ("drop", float((pt.target - ps.target).mean()), float(landing.std(ddof=1)), DROP_MIN[arm]),
                    ("attraction", float((ps.attraction - pt.attraction).mean()), float(contrast.std(ddof=1)), ANCHOR_MIN),
                ):
                    powers.append({"domain": domain, "gen": gen, "arm": arm, "endpoint": endpoint,
                                   "sd_old": sd, **power(effect, np.sqrt(2) * sd, minimum),
                                   "power_sd2": power(effect, 2 * sd, minimum)["joint_power_approx"]})
        del corpus, P, tg
    return manifest, tables, powers


def frozen():
    text = PREREG.read_text()
    return json.loads(text.split("<!-- frozen-manifest -->\n```json\n", 1)[1].split("\n```", 1)[0])


def check_plan():
    m = frozen()
    if (K_NEIGHBOURS, MAX_ITERS, N_CANDIDATES) != (8, 4, 3):
        raise ValueError("decoder budget differs from frozen protocol")
    expected = {"version": 1, "ks": KS, "calibration_ks": CALIBRATION, "generators": GENERATORS,
                "conditions": list(CONDITIONS), "alpha_per_endpoint": ALPHA,
                "anchor_min": ANCHOR_MIN, "drop_min": DROP_MIN, "mae_max": MAE_MAX}
    for k, v in expected.items():
        if m[k] != v:
            raise ValueError(f"code differs from frozen registration: {k}")
    for path, sha in m["sources"].items():
        if digest(DATA / path) != sha:
            raise ValueError(f"frozen input changed: {path}")
    for d in DOMAINS_ORDER:
        for a in ARMS:
            if m["donors"][f"{d}/{a}"] != {str(k): v for k, v in swap_map(KS, seed(d, a)).items()}:
                raise ValueError("donor assignment changed")
    return m


def dry_check(out):
    """Exercise the actual four-round decoder with spies; neither CLI nor ONNX is constructed."""
    check_plan()
    out.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS_ORDER:
        corpus, P, tg = load_domain(domain)
        arm, k = "REAL", 1
        j = swap_map(KS, seed(domain, arm))[k]
        e = corpus.P[corpus.top(tg[arm][k], 8, hidden(tg, arm, k))[0]]
        item = {field: ["OFFLINE FIXTURE"] if spec["type"] == "array" else "OFFLINE FIXTURE"
                for field, spec in DOMAINS[domain].schema["properties"].items()}
        raw = json.dumps(item)

        class Spy:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append(copy.deepcopy(messages))
                return raw

        class FakeEmbedder:
            def embed(self, texts):
                return np.stack([e for _ in texts])

        captures = {}
        for condition in ("default", *CONDITIONS):
            spy = Spy()
            options = {} if condition == "default" else {"neighbours": condition}
            if condition == "swap":
                options.update(neighbour_target=tg[arm][j], neighbour_hidden=hidden(tg, arm, j))
            result = decode(spy, FakeEmbedder(), DOMAINS[domain], corpus, tg[arm][k],
                            hidden=hidden(tg, arm, k), feedback_mode="score", candidate_workers=1, **options)
            assert len(spy.calls) == MAX_ITERS * N_CANDIDATES == 12
            assert all(h["sim"] == float(e @ tg[arm][k]) for h in result["history"])
            captures[condition] = spy.calls
            (out / f"{domain}-{condition}.txt").write_text(render(spy.calls[-1]) + "\n")
        assert captures["default"] == captures["true"]
        # Reconstruct the legacy protocol literally, including every score feedback message.
        nn = corpus.top(tg[arm][k], 8, hidden(tg, arm, k))
        noun = DOMAINS[domain].noun
        legacy_first = (f"There is a location in the space of all {noun}s that we want to fill with a new one. "
                        f"These existing {noun}s are the closest to that location (similarity to it in brackets; "
                        f"1.0 would be exactly on it):\n{_listing(corpus, nn, tg[arm][k])}\n\n"
                        f"Write a NEW {noun} that sits exactly at that location: not a copy of any of them, but the "
                        f"item whose description would be closer to the location than all of them.")
        score = float(e @ tg[arm][k])
        legacy_feedback = (f"Your item reached similarity {score:.3f} to the location (best so far {score:.3f}). "
                           f"The existing {noun}s listed above are still the closest ones. Write a different NEW "
                           f"{noun} that gets closer to the location.")
        expected = [{"role": "system", "content": SYSTEM.format(noun=noun, spec=DOMAINS[domain].json_spec)},
                    {"role": "user", "content": legacy_first}]
        for round_ in range(4):
            assert captures["default"][round_ * 3] == expected
            expected += [{"role": "assistant", "content": raw}, {"role": "user", "content": legacy_feedback}]
        donor_nn = corpus.top(tg[arm][j], 8, hidden(tg, arm, j))
        a, b = captures["true"][-1], captures["swap"][-1]
        assert a[0] == b[0] and a[2:] == b[2:]
        assert a[1]["content"].replace(_listing(corpus, nn, tg[arm][k]), "<LIST>") == \
            b[1]["content"].replace(_listing(corpus, donor_nn, tg[arm][k]), "<LIST>")
        none = captures["none"][-1]
        assert none[0] == a[0]
        assert none[1]["content"] == (f"There is a location in the space of all {noun}s that we want to fill with a new one. "
                                      f"Write a NEW {noun} that sits exactly at that location.")
        for i in (3, 5, 7):
            assert none[i]["content"] == a[i]["content"].replace(f"The existing {noun}s listed above are still the closest ones. ", "")
            assert none[i - 1] == a[i - 1]
        for condition in ("swap", "none"):
            diff = "".join(difflib.unified_diff(render(a).splitlines(True), render(captures[condition][-1]).splitlines(True),
                                               fromfile="true", tofile=condition))
            (out / f"{domain}-{condition}.diff").write_text(diff)
        print(f"PASS {domain} REAL-{k} donor={j}: legacy=default=true byte-identical; swap=list only; "
              "none=example block/references only; 12 mock calls/condition; scores use real target; 0 LLM calls")


def interval(values):
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    half = float(student_t.ppf(1 - ALPHA, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": mean, "low": mean - half, "high": mean + half}


def verdict(drop, attraction, arm):
    if drop["mean"] >= DROP_MIN[arm] and drop["low"] > 0 and attraction["mean"] >= ANCHOR_MIN and attraction["low"] > 0:
        return "causal anchoring supported"
    if max(abs(drop["low"]), abs(drop["high"])) < DROP_MIN[arm] and \
            max(abs(attraction["low"]), abs(attraction["high"])) < ANCHOR_MIN:
        return "equivalent within frozen margins; anchoring weakened"
    if drop["high"] < 0 or attraction["high"] < 0:
        return "opposite-direction effect"
    return "inconclusive/mixed"


def analyze(out):
    m = check_plan()
    all_items, cells = [], []
    for domain in DOMAINS_ORDER:
        corpus, P, tg = load_domain(domain)
        for arm in ARMS:
            T, nn, C, ceiling, donors = geometry(corpus, P, tg, domain, arm)
            for gen in GENERATORS:
                cal = m["calibration"][f"{domain}/{gen}/{arm}"]
                old = historical(domain, gen)
                prior = np.mean([unit(old["PLAIN", k]["best"]["emb"]) for k in CALIBRATION], axis=0)
                pred = {(r["condition"], r["k"]): r for r in predictions(T, C, ceiling, donors, **cal, prior=prior)}
                outputs, counts = {}, {}
                for cond in CONDITIONS:
                    run = DATA / "runs" / domain / run_name(gen, cond, arm)
                    rows = read_rows(run / "items.jsonl")
                    if rows:
                        config = json.loads((run / "intervention.json").read_text())
                        required = {"domain": domain, "n": 20, "start_k": 1, "arms": arm, "feedback": "score",
                                    "neighbours": cond, "neighbour_seed": seed(domain, arm), "candidate_workers": 1,
                                    "generator": GENERATORS[gen], "targets_from": "pilot", "out": run_name(gen, cond, arm),
                                    "targets_sha256": m["sources"][f"runs/{domain}/pilot/targets.npz"]}
                        if config != required or digest(run / "targets.npz") != required["targets_sha256"]:
                            raise ValueError(f"run configuration/targets mismatch: {run}")
                    seen = set()
                    for r in rows:
                        k = r["k"]
                        if r["arm"] != arm or k not in KS or k in seen:
                            raise ValueError(f"unexpected/duplicate item: {run}/{k}")
                        seen.add(k)
                        donor = donors[k] if cond == "swap" else None
                        expected_nn = [] if cond == "none" else nn[donors[k] if cond == "swap" else k]
                        if (r.get("neighbour_condition"), r.get("donor_k"), r.get("generator"), r["neighbours"]) != \
                                (cond, donor, GENERATORS[gen], expected_nn):
                            raise ValueError(f"intervention provenance mismatch: {run}/{k}")
                        if len(r["history"]) != 4 or [h["iter"] for h in r["history"]] != list(range(4)):
                            raise ValueError(f"search budget mismatch: {run}/{k}")
                        if not r.get("best"):
                            if any(h.get("sim") is not None for h in r["history"]):
                                raise ValueError(f"valid candidate discarded: {run}/{k}")
                            continue
                        Eraw = np.asarray(r["best"]["emb"])
                        if Eraw.shape != T[k].shape or not np.isfinite(Eraw).all() or np.linalg.norm(Eraw) == 0:
                            raise ValueError(f"invalid embedding: {run}/{k}")
                        for h in r["history"]:
                            if h.get("sim") is not None and not np.isclose(np.asarray(h["emb"]) @ tg[arm][k], h["sim"], atol=2e-6):
                                raise ValueError(f"feedback was not scored against real target: {run}/{k}")
                            if h.get("sim") is not None and (not 1 <= len(h["candidate_sims"]) <= 3 or h["sim"] != max(h["candidate_sims"])):
                                raise ValueError(f"candidate budget/selection mismatch: {run}/{k}")
                        if r["best"] not in r["history"] or r["best"]["sim"] != max(h["sim"] for h in r["history"] if h["sim"] is not None):
                            raise ValueError(f"search budget/best mismatch: {run}/{k}")
                        E = unit(Eraw)
                        c = C[donors[k]] if cond == "swap" else C[k]
                        p = pred[cond, k]
                        item = {"domain": domain, "gen": gen, "arm": arm, "condition": cond, "k": k,
                                "donor_k": donors[k], "target": float(E @ T[k]), "donor": float(E @ T[donors[k]]),
                                "shown": None if cond == "none" else float(E @ c),
                                "true_centroid": float(E @ C[k]), "donor_centroid": float(E @ C[donors[k]]),
                                "attraction": float(E @ (C[donors[k]] - C[k])),
                                **{f"pred_{key}": p[key] for key in ("target", "donor", "shown", "kappa")}}
                        item["remainder"] = None if cond == "none" else item["target"] - item["shown"] * float(c @ T[k])
                        outputs[cond, k] = item
                        all_items.append(item)
                    counts[cond] = {"attempted": len(seen), "valid": sum((cond, k) in outputs for k in KS)}
                complete = [k for k in KS if all((c, k) in outputs for c in CONDITIONS)]
                cell = {"domain": domain, "gen": gen, "arm": arm, "counts": counts, "n_complete": len(complete)}
                # Require 20/20, so selective refusal/loss cannot produce a confirmatory result.
                if len(complete) != 20:
                    cell["verdict"] = "incomplete: no confirmatory verdict"
                else:
                    drop = interval([outputs["true", k]["target"] - outputs["swap", k]["target"] for k in complete])
                    attraction = interval([outputs["swap", k]["attraction"] - outputs["true", k]["attraction"] for k in complete])
                    cell.update(drop=drop, attraction=attraction, verdict=verdict(drop, attraction, arm))
                    cell["no_neighbours_drop_descriptive"] = float(np.mean([outputs["true", k]["target"] - outputs["none", k]["target"] for k in complete]))
                    cell["quantitative"] = {}
                    for cond in CONDITIONS:
                        group = [outputs[cond, k] for k in complete]
                        errors = {key: float(np.mean([abs(r[key] - r[f"pred_{key}"]) for r in group]))
                                  for key in ("target", "donor")}
                        errors["kappa_mae"] = float(np.mean([abs(r["target"] - r["pred_kappa"]) for r in group]))
                        errors["means"] = {key: float(np.mean([r[key] for r in group]))
                                           for key in ("target", "donor", "true_centroid", "donor_centroid")}
                        if cond != "none":
                            errors["means"]["shown"] = float(np.mean([r["shown"] for r in group]))
                            errors["mean_remainder"] = float(np.mean([r["remainder"] for r in group]))
                            errors["mae_remainder"] = float(np.mean([abs(r["remainder"]) for r in group]))
                            errors["shown_bias"] = float(np.mean([r["shown"] - r["pred_shown"] for r in group]))
                            errors["transport_pass"] = max(errors["target"], errors["donor"]) <= MAE_MAX[arm] and abs(errors["shown_bias"]) <= 0.10
                        cell["quantitative"][cond] = errors
                cells.append(cell)
        del corpus, P, tg
    support = {a: sum(c["arm"] == a and c["verdict"] == "causal anchoring supported" for c in cells) for a in ARMS}
    complete = all(c["n_complete"] == 20 for c in cells)
    overall = "incomplete" if not complete else ("cross-domain/model anchoring supported" if support["FAR"] == 4 and support["GAP"] >= 3 and support["REAL"] >= 3 else "no global confirmation; inspect frozen cell verdicts")
    report = {"overall": overall, "supported_cells": support, "cells": cells}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_items).to_csv(out / "items.csv", index=False)
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return complete


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["design", "check-plan", "dry-check", "analyze"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "design":
        manifest, tables, powers = design()
        print(json.dumps({"manifest": manifest, "predictions": tables, "power": powers}, indent=2, allow_nan=True))
    elif args.command == "check-plan":
        check_plan()
        print("PASS frozen inputs, parameters and donor assignments")
    elif args.command == "dry-check":
        dry_check(args.out or Path("/tmp/anchoring-causal-prompts"))
    else:
        if not analyze(args.out or DATA / "runs/anchoring-causal-analysis"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
