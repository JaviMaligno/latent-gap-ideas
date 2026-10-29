"""Result tables and pre-registered verdicts of the coherent no-neighbour search (docs/prereg/coherent_search.md,
rev 2): main search against the corpus ceiling, the matched held-out null, GPU token search (h_any) and the anchored
decoder; the REAL positive control; the random (no feedback) and free (no coherence) controls; pooled P3 rates.

Usage: uv run python scripts/coherent_summary.py [--domains steam,arxiv]
       uv run python scripts/coherent_summary.py --runs <dir>,... --no-check   (smoke runs; no verdict is binding)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
FAR_MARGIN, GAP_MARGIN, COVERAGE, ESCAPE_SHARE = 0.03, 0.02, 0.90, 0.20  # pre-registered
COH_PASS = 7  # pre-registered after the judge calibration (prereg §Judge calibration)
# Pre-registered run directories (runs/<d>/coherent_search/<name>); anything else, smoke runs included, is ignored.
MAIN_RUNS = {"claude-sonnet-5-5": "search", "gateway-gpt-6-luna": "search", "claude-haiku-5-5": "search",
             "claude-sonnet-5-5-random": "random", "claude-sonnet-5-5-free": "free"}
EXPECTED = {"T": 10, "K": 4, "judge": "claude:claude-sonnet-5-5", "coh_pass": COH_PASS, "judge_cap": 15,
            "judge_sample": 4, "prompt": "v3"}
DECODER = "grid-claude-sonnet-5-5"  # anchored decoder compared in P4, same model family as the primary writer


def md(df: pd.DataFrame) -> str:
    df = df.fillna("-")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(df.columns), *rows])


def passes(c: dict, cfg: dict) -> bool:
    th = cfg["coh_pass"]
    return bool(c.get("valid") and not c["gibberish"] and (c.get("coherence") or 0) >= th
                and (cfg["judge2"] == "none" or (c.get("coherence2") or 0) >= th))


def load_run(domain: str, run: Path, check: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cands = [], []
    for line in (run / "items.jsonl").read_text().splitlines():
        r = json.loads(line)
        cfg = r["config"]
        if check:
            want = EXPECTED | {"mode": MAIN_RUNS[run.name],
                               "judge2": "claude:claude-haiku-5-5" if r["backend"] == "gateway" else "codex:gpt-6-luna"}
            if cfg != want:
                raise SystemExit(f"{run}: {r['arm']} {r['k']} has config {cfg}, pre-registered {want}")
        ok = [c for c in r["candidates"] if c["valid"]]
        passing = [c for c in ok if passes(c, cfg)]
        best = max(ok, key=lambda c: c["score"]) if ok else None
        bc = max(passing, key=lambda c: c["score"]) if passing else None
        rows.append({"domain": domain, "run": run.name, "mode": cfg["mode"], "arm": r["arm"], "k": r["k"],
                     "ceiling": r["ceiling"], "heldout_ceiling": r["heldout_ceiling"], "null": r["null_matched"],
                     "q99": r["corpus_q99"], "h_all": best["score"] if best else np.nan,
                     "h_coh": bc["score"] if bc else np.nan, "corpus_max_coh": bc["corpus_max"] if bc else np.nan,
                     "seed_div": r["seed_div"], "valid": len(ok) / len(r["candidates"]),
                     "judge_errors": r["n_judge_errors"]})
        seen = set()
        for c in ok:  # distinct candidates for the pooled P3 rates
            if c["sha256"] not in seen:
                seen.add(c["sha256"])
                cands.append({"domain": domain, "run": run.name, "arm": r["arm"], "gibberish": c["gibberish"],
                              "in_sample": c.get("in_sample", False), "coherence": c.get("coherence")})
    return pd.DataFrame(rows), pd.DataFrame(cands)


def decoder_landing(domain: str) -> pd.DataFrame:
    """Per-target landing of the pre-registered anchored decoder, only if its targets equal the pilot's."""
    m = DATA / f"runs/{domain}/{DECODER}"
    if not (m / "metrics.csv").exists():
        return pd.DataFrame(columns=["arm", "k", "landing"])
    pilot = np.load(DATA / f"runs/{domain}/pilot/targets.npz", allow_pickle=True)
    if (m / "targets.npz").exists():
        z = np.load(m / "targets.npz", allow_pickle=True)
        if not all(a in z.files and np.allclose(z[a], pilot[a]) for a in ("FAR", "GAP", "REAL")):
            raise SystemExit(f"{m}: targets differ from the pilot's")
    return pd.read_csv(m / "metrics.csv")[["arm", "k", "landing"]]


def boot_median(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.median(x)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="steam,arxiv")
    ap.add_argument("--runs", default="", help="explicit run dir names (e.g. smoke runs); default: MAIN_RUNS")
    ap.add_argument("--no-check", action="store_true", help="skip the pre-registered config check (smoke only)")
    args = ap.parse_args()
    check = not args.no_check
    names = args.runs.split(",") if args.runs else list(MAIN_RUNS)
    if check and any(n not in MAIN_RUNS for n in names):
        raise SystemExit("only pre-registered runs can be summarised with the config check; use --no-check")
    bounds = pd.read_csv(DATA / "runs/support-search/support_bounds.csv")[["domain", "arm", "k", "h_any_lower"]]
    frames, cfr = [], []
    for d in args.domains.split(","):
        for n in names:
            run = DATA / f"runs/{d}/coherent_search/{n}"
            if (run / "items.jsonl").exists():
                f, c = load_run(d, run, check)
                frames.append(f)
                cfr.append(c)
    if not frames:
        raise SystemExit("no runs found")
    df, C = pd.concat(frames), pd.concat(cfr)
    main_df = df[df["mode"] == "search"]
    if main_df.run.nunique() > 1:  # lower bounds combine by max: the best coherent witness of any writer counts
        agg = main_df.groupby(["domain", "arm", "k"], as_index=False).agg(
            ceiling=("ceiling", "first"), heldout_ceiling=("heldout_ceiling", "first"), null=("null", "first"),
            q99=("q99", "first"), h_all=("h_all", "max"), h_coh=("h_coh", "max"))
        main_df = pd.concat([main_df, agg.assign(run="ALL (max over writers)", mode="search")])
    main_df = main_df.merge(bounds, on=["domain", "arm", "k"], how="left")
    dec = pd.concat([decoder_landing(d).assign(domain=d) for d in df.domain.unique()])
    main_df = main_df.merge(dec, on=["domain", "arm", "k"], how="left")
    main_df["escape"] = main_df.h_coh > np.maximum(main_df.ceiling, main_df.heldout_ceiling) + FAR_MARGIN
    main_df["reached"] = main_df.h_coh >= main_df.ceiling - GAP_MARGIN  # NaN (unmeasured) -> False

    def med(s):
        return f"{s.median():.3f}" if s.notna().any() else "-"

    rows, verdicts = [], []
    for (d, run), g0 in main_df.groupby(["domain", "run"], sort=False):
        real = g0[g0.arm == "REAL"]
        cov_real = real.h_coh.notna().mean() if len(real) else np.nan
        powered = bool(len(real) and cov_real >= COVERAGE and (real.h_coh - real.q99).median() >= 0)
        c_txt = ("unmeasured (no REAL targets)" if not len(real) else
                 f"{'POWERED' if powered else 'UNDERPOWERED'}: median h_coh - corpus q99 = "
                 f"{(real.h_coh - real.q99).median():+.3f}, coverage {cov_real:.2f}")
        verdicts.append(f"- {d} {run} C (REAL control): {c_txt}")
        for arm, g in g0.groupby("arm"):
            cov = g.h_coh.notna().mean()
            rows.append({"domain": d, "writer": run, "arm": arm, "n": len(g), "ceiling": med(g.ceiling),
                         "matched null": med(g.null), "corpus q99": med(g.q99), "h_all": med(g.h_all),
                         "h_coh": med(g.h_coh), "h_coh - ceiling": med(g.h_coh - g.ceiling),
                         "corpus_max(coh)": med(g.corpus_max_coh), "h_any (GPU)": med(g.h_any_lower),
                         "anchored landing": med(g.landing), "measured": f"{g.h_coh.notna().sum()}/{len(g)}",
                         "escape +0.03": f"{g.escape.mean():.2f}" if arm == "FAR" else "-",
                         "reached -0.02": f"{g.reached.mean():.2f}" if arm == "GAP" else "-"})
            if arm == "FAR":
                if cov < COVERAGE:
                    v = f"insufficient coverage ({cov:.2f} < {COVERAGE})"
                elif g.escape.mean() > ESCAPE_SHARE:
                    v = f"objection REFUTED: coherent text above max(ceiling, held-out ceiling) + 0.03 on {g.escape.mean():.0%}"
                elif powered and g.h_coh.median() >= g.null.median():
                    v = "no coherent escape exhibited within 44 candidates, by a powered optimiser (supports the objection)"
                else:
                    v = ("uninformative: optimiser underpowered (REAL control failed or FAR h_coh below the matched "
                         "held-out null)")
                verdicts.append(f"- {d} {run} P1 FAR: {v}")
            elif arm == "GAP":
                if cov < COVERAGE:
                    v = f"insufficient coverage ({cov:.2f} < {COVERAGE})"
                elif g.reached.mean() > 0.5:
                    v = f"gaps reachable: coherent text within 0.02 of the ceiling on {g.reached.mean():.0%}"
                elif powered:
                    v = "not reached by a powered blind optimiser within 44 candidates (evidence against P2, not proof)"
                else:
                    v = "inconclusive: optimiser underpowered"
                verdicts.append(f"- {d} {run} P2 GAP: {v}")
            if arm in ("FAR", "GAP"):
                p = g.dropna(subset=["h_coh", "landing"])
                if len(p) >= 5:
                    m, lo, hi = boot_median((p.h_coh - p.landing).to_numpy())
                    v = ("removing anchors helps" if lo > 0 else "anchored decoder lands closer" if hi < 0
                         else "no difference shown")
                    verdicts.append(f"- {d} {run} P4 {arm} vs {DECODER}: median paired h_coh - landing {m:+.3f} "
                                    f"[{lo:+.3f}, {hi:+.3f}], n={len(p)}: {v}")
    print("Main search (medians over targets; h_coh = best gate-passing candidate)\n")
    print(md(pd.DataFrame(rows)))

    ctrl = df[df["mode"] != "search"]
    if len(ctrl):
        rows = []
        for (d, run, arm), g in ctrl.groupby(["domain", "run", "arm"]):
            base = run.replace(f"-{g['mode'].iloc[0]}", "", 1)  # the same writer's main search run
            ref = main_df[main_df.run == base][["domain", "arm", "k", "h_coh", "h_all"]]
            s = g.merge(ref, on=["domain", "arm", "k"], suffixes=("", "_search"))
            val = g.h_all if g["mode"].iloc[0] == "free" else g.h_coh
            rows.append({"domain": d, "control": run, "arm": arm, "n": len(g), "ceiling": med(g.ceiling),
                         "control value": med(val),
                         "control - ceiling": med(val - g.ceiling),
                         "search h_coh (same targets)": med(s.h_coh_search),
                         "search h_coh - ceiling": med(s.h_coh_search - s.ceiling)})
        print("\nControls (free: h_any_blackbox = best of all valid; random: h_coh without feedback)\n")
        print(md(pd.DataFrame(rows)))

    rows = []
    for (d, run), g in C.groupby(["domain", "run"]):
        samp = g[g.in_sample & g.coherence.notna()]
        gib_ok = 1 - g.gibberish.mean()
        coh_ok = (samp.coherence >= COH_PASS).mean() if len(samp) else np.nan
        v = "holds" if gib_ok >= 0.95 and coh_ok >= 0.90 else "unmeasured" if np.isnan(coh_ok) else "REFUTED"
        rows.append({"domain": d, "writer": run, "distinct valid": len(g), "pass gibberish flag": f"{gib_ok:.3f}",
                     "random sample judged": len(samp), "judge 1 >= 7 (sample)": f"{coh_ok:.3f}", "P3": v})
    print("\nP3 coherence (pooled over distinct candidates; the judge rate on the random sample only)\n")
    print(md(pd.DataFrame(rows)))
    print("\nVerdicts" + (" (pre-registered)" if check else " (NOT binding: config check off)") + ":")
    print("\n".join(verdicts))
    out = DATA / "runs/coherent_search_targets.csv" if check else DATA / "runs/coherent_search_targets_unchecked.csv"
    main_df.to_csv(out, index=False)
    print(f"\nper-target table: {out}")


if __name__ == "__main__":
    main()
