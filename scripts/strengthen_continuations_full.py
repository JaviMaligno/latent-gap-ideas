"""Full census of the follow-up pairs (docs/prereg/strengthen_continuations_full.md)

  uv run python scripts/strengthen_continuations_full.py sample            # stage-3 pairs + batches
  uv run python scripts/strengthen_continuations_full.py label codex       # sequential, all attempts logged
  uv run python scripts/strengthen_continuations_full.py label claude
  uv run python scripts/strengthen_continuations_full.py status
  OMP_NUM_THREADS=4 uv run python scripts/strengthen_continuations_full.py analyse

Reuses prompt_for / run_cli / parse / pair_record / write_batches of strengthen_continuations.py unchanged.
Raw: ~/Documents/experiment-data/latent-gap-ideas/runs/strengthen-continuations/stage3/.
Outputs: docs/prereg/strengthen_continuations_full.json, docs/prereg/strengthen_continuations_full_labels.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_from_context as M  # noqa: E402
import strengthen_continuations as S  # noqa: E402

ROOT = S.ROOT
WORK = S.WORK
D3 = WORK / "stage3"
OUT = ROOT / "docs/prereg/strengthen_continuations_full.json"
OUT_LABELS = ROOT / "docs/prereg/strengthen_continuations_full_labels.json"
SEED, N_BOOT, LABELS = S.SEED, S.N_BOOT, S.LABELS


def sample() -> None:
    ctx = M.contexts()
    by_id = S.corpus_text()
    done = set()
    for s in (1, 2):
        done |= {(p["ctx_index"], p["followup_index"]) for p in json.loads((WORK / f"stage{s}/pairs.json").read_text())}
    pairs = [S.pair_record(ctx, by_id, i, j, f"s3-{i}-{j}") for i, c in enumerate(ctx)
             for j in range(len(c["followups"])) if (i, j) not in done]
    assert len(pairs) + len(done) == sum(len(c["followups"]) for c in ctx)
    S.write_batches(3, pairs, np.random.default_rng(SEED + 3))


# ------------------------------------------------------------------------------------------------- logged labelling

def diagnose(text: str, batch: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Status of one CLI output, the valid rows it contains and the ids it misses (diagnostic twin of S.parse)."""
    if not text.strip():
        return "empty", [], [p["id"] for p in batch]
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        return "no_json", [], [p["id"] for p in batch]
    try:
        rows = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "invalid_json", [], [p["id"] for p in batch]
    by = {str(r.get("id")): r for r in rows if isinstance(r, dict)}
    ok, miss, bad = [], [], False
    for p in batch:
        r = by.get(p["id"])
        if not r:
            miss.append(p["id"])
        elif r.get("label") not in LABELS:
            miss.append(p["id"]); bad = True
        else:
            ok.append({"id": p["id"], "label": r["label"], "reason": str(r.get("reason", ""))})
    if not miss:
        return "ok", ok, []
    return ("invalid_label" if bad else "incomplete"), ok, miss


def log_attempt(who: str, rec: dict) -> None:
    with open(D3 / f"attempts_{who}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def n_prior_attempts(who: str, stem: str) -> int:
    f = D3 / f"attempts_{who}.jsonl"
    if not f.exists():
        return 0
    return sum(1 for line in f.read_text().splitlines() if json.loads(line)["batch"] == stem)


def call(who: str, batch: list[dict], raw: Path) -> tuple[str, list[dict], list[str], float]:
    t0 = time.time()
    try:
        text = S.run_cli(who, S.prompt_for(batch), raw)
    except Exception as e:  # timeout etc.
        return f"error:{type(e).__name__}", [], [p["id"] for p in batch], time.time() - t0
    st, rows, miss = diagnose(text, batch)
    if st == "ok":
        assert S.parse(text, batch) == rows
    return st, rows, miss, time.time() - t0


def label_batch(who: str, path: Path, max_total: int = 6) -> str:
    out = D3 / f"labels_{who}_{path.stem}.json"
    if out.exists():
        return f"{path.stem} cached"
    batch = json.loads(path.read_text())
    prior = n_prior_attempts(who, path.stem)
    for attempt in range(prior, min(prior + 3, max_total)):
        raw = D3 / f"raw_{who}_{path.stem}_try{attempt}.txt"
        st, rows, miss, sec = call(who, batch, raw)
        log_attempt(who, {"batch": path.stem, "attempt": attempt, "kind": "batch", "seconds": round(sec), "status": st,
                          "n_valid": len(rows), "missing": miss, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if st == "ok":
            out.write_text(json.dumps({"labeller": who, "attempt": attempt, "seconds": round(sec), "rows": rows},
                                      indent=1))
            return f"{path.stem} ok try{attempt} {sec:.0f}s"
    if prior + 3 < max_total:
        return f"{path.stem} FAILED (resume pass pending)"
    return single_pair_fallback(who, path, batch)


def single_pair_fallback(who: str, path: Path, batch: list[dict]) -> str:
    """After 6 failed batch attempts: keep the most complete attempt's valid rows, label the rest one by one."""
    best: list[dict] = []
    for raw in sorted(D3.glob(f"raw_{who}_{path.stem}_try*.txt")):
        _, rows, _ = diagnose(raw.read_text(), batch)
        if len(rows) > len(best):
            best = rows
    have = {r["id"] for r in best}
    rows = list(best)
    for p in batch:
        if p["id"] in have:
            continue
        for a in range(3):
            raw = D3 / f"raw_{who}_{path.stem}_single_{p['id']}_try{a}.txt"
            st, r1, miss, sec = call(who, [p], raw)
            log_attempt(who, {"batch": path.stem, "attempt": a, "kind": "single", "pair": p["id"], "seconds": round(sec),
                              "status": st, "n_valid": len(r1), "missing": miss,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if st == "ok":
                rows += r1
                break
    order = {p["id"]: k for k, p in enumerate(batch)}
    rows.sort(key=lambda r: order[r["id"]])
    if len(rows) < len(batch):
        return f"{path.stem} FAILED after single-pair fallback ({len(rows)}/{len(batch)})"
    (D3 / f"labels_{who}_{path.stem}.json").write_text(json.dumps(
        {"labeller": who, "attempt": "fallback_single_pair", "n_from_batch": len(best), "rows": rows}, indent=1))
    return f"{path.stem} ok via single-pair fallback ({len(best)} from batch)"


def label(who: str) -> None:
    for f in sorted(D3.glob("batch_*.json")):
        print(who, label_batch(who, f), flush=True)


def status() -> None:
    files = sorted(D3.glob("batch_*.json"))
    for who in ("codex", "claude"):
        done = len(list(D3.glob(f"labels_{who}_batch_*.json")))
        print(who, f"{done}/{len(files)}", attempt_counts(who))


def attempt_counts(who: str) -> dict:
    f = D3 / f"attempts_{who}.jsonl"
    recs = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    out: dict = {"attempts": len(recs), "batch_attempts": sum(r["kind"] == "batch" for r in recs),
                 "single_pair_attempts": sum(r["kind"] == "single" for r in recs), "by_status": {}}
    for r in recs:
        out["by_status"][r["status"]] = out["by_status"].get(r["status"], 0) + 1
    out["batches_needing_retry"] = sorted({r["batch"] for r in recs if r["status"] != "ok"})
    out["retries"] = sum(r["status"] != "ok" for r in recs)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        sample()
    elif cmd == "label":
        label(sys.argv[2])
    elif cmd == "status":
        status()
    elif cmd == "analyse":
        import strengthen_continuations_full_analysis as A  # noqa: E402
        A.analyse()
