"""Stage 2 of docs/prereg/strengthen_olmo_direction.md: continuation labels with the S2-flag hybrid.

  uv run python scripts/olmo_direction_label.py --dry-run                 # batch counts from what is fetched so far
  nohup uv run python scripts/olmo_direction_label.py run --until-eligible 100 \
        > ~/Documents/experiment-data/latent-gap-ideas/runs/olmo-direction/label.log 2>&1 < /dev/null &
  uv run python scripts/olmo_direction_label.py status
  uv run python scripts/olmo_direction_label.py eligible                  # writes runs/olmo-direction/eligible.json
  uv run python scripts/olmo_direction_label.py census                    # switch to the named fallback (prereg)

Prompt, CLI calls, parsing, attempt logging, retries and the single-pair fallback are those of
strengthen_continuations.py / strengthen_continuations_full.py, called unchanged: each labeller runs in its own
process (label-stage) with strengthen_continuations_full.D3 pointed at the stage directory.

Blocks of 200 contexts (>= 5 follow-ups, frame order). Stage A: flagged pairs + all pairs of the block's audit
contexts (seeded 10 %). Stage B (after both labellers finish A): remaining pairs of triggered contexts (a flagged
follow-up both call CONTINUES). Census mode: stage A = every pair; stage C completes hybrid blocks to a census.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strengthen_continuations as S  # noqa: E402
import strengthen_continuations_full as SF  # noqa: E402

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/olmo-direction"
LAB = RUN / "labels"
SEED, BLOCK, BATCH, MIN_FUPS, AUDIT_SHARE = 20261106, 200, 20, 5, 0.10
WHO = ("codex", "claude")
SEC_PER_BATCH = 30  # census stage 3 median (codex 30 s, claude 25 s)
CENSUS_YIELD, HYBRID_YIELD = 17 / 150, 14 / 150  # census of the 150 contexts
TRIGGER_SHARE = 525 / 2398  # census: share of unflagged pairs that sit in triggered contexts


def log(m: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


def flagged(f: dict) -> bool:
    return bool(f.get("influential")) or bool({"methodology", "result"} & set(f.get("intents") or []))


def contexts() -> list[dict]:
    p = RUN / "followups.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    out = [r for r in rows if r["n_followups"] >= MIN_FUPS]
    for pos, r in enumerate(out):
        r["pos"] = pos
    return out


def fetch_done() -> bool:
    """True when the fetch log says it stopped (target reached or frame exhausted)."""
    lg = RUN / "fetch.log"
    return lg.exists() and " stop: " in lg.read_text()[-4000:]


def blocks(ctx: list[dict], allow_partial: bool) -> list[list[dict]]:
    out = [ctx[b:b + BLOCK] for b in range(0, len(ctx), BLOCK)]
    if out and len(out[-1]) < BLOCK and not allow_partial:
        out = out[:-1]
    return out


def audit_set(b: int, blk: list[dict]) -> set[int]:
    n = min(len(blk), round(AUDIT_SHARE * len(blk)))
    pick = np.random.default_rng(SEED + 100 + b).choice(len(blk), n, replace=False)
    return {blk[k]["pos"] for k in pick}


def mode() -> str:
    f = LAB / "mode.json"
    return json.loads(f.read_text())["mode"] if f.exists() else "hybrid"


_TEXT: dict | None = None


def source_text(aid: str) -> tuple[str, str]:
    global _TEXT
    if _TEXT is None:
        _TEXT = S.corpus_text()
    return _TEXT[aid]


def pair(c: dict, j: int) -> dict:
    f = c["followups"][j]
    name, abstract = source_text(c["arxiv_id"])
    return {"id": f"od-{c['pos']}-{j}", "pos": c["pos"], "followup_index": j, "context_id": c["arxiv_id"],
            "followup_id": f["arxiv_id"], "flagged": flagged(f), "source_title": name, "source_abstract": abstract,
            "followup_title": f["title"], "followup_abstract": f["abstract"]}


def stage_dir(b: int, st: str) -> Path:
    return LAB / f"block{b:02d}{st}"


def write_stage(b: int, st: str, pairs: list[dict]) -> None:
    d = stage_dir(b, st)
    tmp = d.with_name(d.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(SEED + 10 * b + {"A": 1, "B": 2, "C": 3}[st]).permutation(len(pairs))
    pairs = [pairs[k] for k in order]
    (tmp / "pairs.json").write_text(json.dumps(pairs, indent=1))
    for k in range(0, len(pairs), BATCH):
        (tmp / f"batch_{k // BATCH:03d}.json").write_text(json.dumps(pairs[k:k + BATCH], indent=1))
    tmp.rename(d)
    log(f"planned block {b} stage {st}: {len(pairs)} pairs, {-(-len(pairs) // BATCH)} batches, "
        f"{len({p['pos'] for p in pairs})} contexts")


def stage_labels(b: int, st: str, who: str) -> dict | None:
    """All labels of a stage for one labeller, or None if any batch is still missing."""
    d = stage_dir(b, st)
    if not d.exists():
        return None
    files = sorted(d.glob("batch_*.json"))
    out = {}
    for f in files:
        lf = d / f"labels_{who}_{f.stem}.json"
        if not lf.exists():
            return None
        out |= {r["id"]: r["label"] for r in json.loads(lf.read_text())["rows"]}
    return out


def labelled(b: int) -> dict:
    """{pair id: (codex, claude)} over the finished stages of block b."""
    out = {}
    for st in ("A", "B", "C"):
        L = {w: stage_labels(b, st, w) for w in WHO}
        if all(v is not None for v in L.values()):
            out |= {i: (L["codex"][i], L["claude"][i]) for i in L["codex"]}
    return out


def plan_next(b: int, blk: list[dict]) -> str | None:
    """Plan the next stage of block b if possible. Returns the stage to label, 'done' or None (waiting)."""
    m = mode()
    aud = audit_set(b, blk)
    have = labelled(b)
    if not stage_dir(b, "A").exists():
        if b == 0 and m == "hybrid":
            fl = [f for c in blk for f in c["followups"]]
            miss = sum(1 for f in fl if f.get("intents") is None and f.get("influential") is None) / len(fl)
            (LAB / "intents_check.json").write_text(json.dumps({"block0_intents_missing_share": miss}))
            if miss > 0.20:
                (LAB / "mode.json").write_text(json.dumps({"mode": "census", "why": f"fallback (a): {miss:.1%} missing"}))
                log(f"fallback (a): intents missing for {miss:.1%} of block-0 follow-ups -> census")
                m = "census"
        pairs = [pair(c, j) for c in blk for j in range(len(c["followups"]))
                 if m == "census" or c["pos"] in aud or flagged(c["followups"][j])]
        write_stage(b, "A", pairs)
        return "A"
    if any(stage_labels(b, "A", w) is None for w in WHO):
        return "A"
    if m == "hybrid":
        if not stage_dir(b, "B").exists():
            trig = triggered(blk, have)
            pairs = [pair(c, j) for c in blk if c["pos"] in trig for j in range(len(c["followups"]))
                     if f"od-{c['pos']}-{j}" not in have]
            if not pairs:
                (LAB / f"block{b:02d}B.none").write_text("no triggered context with unlabelled pairs\n")
                return "done"
            write_stage(b, "B", pairs)
            return "B"
        return "B" if any(stage_labels(b, "B", w) is None for w in WHO) else "done"
    # census mode: complete a block started in hybrid mode
    rest = [pair(c, j) for c in blk for j in range(len(c["followups"])) if f"od-{c['pos']}-{j}" not in have]
    if not stage_dir(b, "C").exists():
        if any(stage_dir(b, "B").exists() and stage_labels(b, "B", w) is None for w in WHO):
            return "B"
        if not rest:
            return "done"
        write_stage(b, "C", rest)
        return "C"
    return "C" if any(stage_labels(b, "C", w) is None for w in WHO) else "done"


def triggered(blk: list[dict], have: dict) -> set[int]:
    out = set()
    for c in blk:
        for j, f in enumerate(c["followups"]):
            if flagged(f) and have.get(f"od-{c['pos']}-{j}") == ("CONTINUES", "CONTINUES"):
                out.add(c["pos"])
    return out


def label_stage(d: Path, who: str) -> None:
    """One labeller over one stage: pass 1 (3 attempts per batch), resume pass (3 more + single-pair fallback)."""
    SF.D3 = d
    files = sorted(d.glob("batch_*.json"))
    for p in (1, 2):
        for f in files:
            print(who, f"pass{p}", SF.label_batch(who, f), flush=True)


def run(until: int, allow_partial: bool) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    idle_since = None
    while True:
        ctx = contexts()
        bl = blocks(ctx, allow_partial or fetch_done())
        n_el, b_next = 0, None
        for b, blk in enumerate(bl):
            st = plan_next(b, blk)
            if st != "done":
                b_next = (b, st)
                break
            n_el += sum(r["eligible"] for r in block_records(b, blk))
        if n_el >= until:
            log(f"stop: {n_el} eligible contexts in completed blocks (target {until})")
            break
        if b_next is None:
            if fetch_done() and len(bl) * BLOCK >= len(ctx):
                log(f"stop: fetch finished and every block labelled; {n_el} eligible (target {until})")
                break
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 6 * 3600:
                log("stop: no new block for 6 h (fetch stalled?)")
                break
            log(f"{n_el} eligible so far; waiting for the fetch (block {len(bl)} needs {BLOCK * (len(bl) + 1)} "
                f"contexts, have {len(ctx)})")
            time.sleep(900)
            continue
        idle_since = None
        b, st = b_next
        d = stage_dir(b, st)
        log(f"labelling block {b} stage {st} ({len(list(d.glob('batch_*.json')))} batches), {n_el} eligible so far")
        procs = [subprocess.Popen([sys.executable, __file__, "label-stage", str(d), w], stdin=subprocess.DEVNULL,
                                  stdout=open(d / f"run_{w}.log", "a"), stderr=subprocess.STDOUT) for w in WHO]
        for p in procs:
            p.wait()
        missing = [w for w in WHO if stage_labels(b, st, w) is None]
        if missing:
            log(f"stop: block {b} stage {st} still has unlabelled batches for {missing} after the resume pass and "
                f"the single-pair fallback; see {d}/attempts_*.jsonl")
            break
    eligible()


def block_records(b: int, blk: list[dict]) -> list[dict]:
    have, aud, m = labelled(b), audit_set(b, blk), mode()
    trig = triggered(blk, have)
    out = []
    for c in blk:
        lab = {j: have.get(f"od-{c['pos']}-{j}") for j in range(len(c["followups"]))}
        full = all(v is not None for v in lab.values())
        cons = [j for j, v in lab.items() if v == ("CONTINUES", "CONTINUES")]
        bg = [j for j, v in lab.items() if v == ("BACKGROUND", "BACKGROUND")]
        trg = c["pos"] in trig
        el = (len(cons) >= 2 and full) if m == "census" else (trg and len(cons) >= 2)
        out.append({"pos": c["pos"], "frame_pos": c["frame_pos"], "arxiv_id": c["arxiv_id"], "block": b,
                    "n_followups": len(c["followups"]), "n_labelled": sum(v is not None for v in lab.values()),
                    "n_flagged": sum(flagged(f) for f in c["followups"]), "audit": c["pos"] in aud,
                    "triggered": trg, "consensus": cons, "both_background": bg, "eligible": bool(el),
                    "missed": bool(m == "hybrid" and c["pos"] in aud and not trg and len(cons) >= 2)})
    return out


def eligible() -> dict:
    ctx = contexts()
    recs = []
    for b, blk in enumerate(blocks(ctx, True)):
        if plan_ready(b, blk):
            recs += block_records(b, blk)
    el = [r for r in recs if r["eligible"]]
    aud = [r for r in recs if r["audit"]]
    found = sum(r["eligible"] for r in aud)
    missed = sum(r["missed"] for r in aud)
    res = {"mode": mode(), "contexts_ge5_labelled": len(recs), "eligible": len(el),
           "yield": len(el) / len(recs) if recs else None, "audit_contexts": len(aud), "audit_found": found,
           "audit_missed": missed, "audit_loss": missed / (missed + found) if missed + found else None,
           "fallback_b_triggered": bool(missed >= 4 and missed / max(1, missed + found) > 0.40),
           "eligible_pos": [r["pos"] for r in el], "contexts": recs}
    (RUN / "eligible.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("contexts", "eligible_pos")}, indent=1))
    return res


def plan_ready(b: int, blk: list[dict]) -> bool:
    """Block finished in the current mode (no planning side effects)."""
    if any(stage_labels(b, "A", w) is None for w in WHO):
        return False
    if mode() == "hybrid":
        return (LAB / f"block{b:02d}B.none").exists() or all(stage_labels(b, "B", w) is not None for w in WHO)
    have = labelled(b)
    return all(f"od-{c['pos']}-{j}" in have for c in blk for j in range(len(c["followups"])))


def status() -> None:
    for d in sorted(LAB.glob("block*")):
        if not d.is_dir():
            continue
        nb = len(list(d.glob("batch_*.json")))
        print(d.name, {w: f"{len(list(d.glob(f'labels_{w}_batch_*.json')))}/{nb}" for w in WHO})
    if (RUN / "eligible.json").exists():
        e = json.loads((RUN / "eligible.json").read_text())
        print({k: v for k, v in e.items() if k not in ("contexts", "eligible_pos")})


def dry_run() -> None:
    ctx = contexts()
    fetch = ("finished" if fetch_done() else "running") if (RUN / "fetch.log").exists() else "not started"
    print(f"fetched: {len(ctx)} contexts with >= {MIN_FUPS} follow-ups (fetch {fetch}); mode {mode()}")
    tot = {"A": 0, "B_proj": 0.0, "census": 0}
    for b, blk in enumerate(blocks(ctx, True)):
        aud = audit_set(b, blk)
        n_all = sum(len(c["followups"]) for c in blk)
        a = sum(len(c["followups"]) if c["pos"] in aud else sum(map(flagged, c["followups"])) for c in blk)
        rest = sum(len(c["followups"]) - sum(map(flagged, c["followups"])) for c in blk if c["pos"] not in aud)
        bp = TRIGGER_SHARE * rest
        tot["A"] += a; tot["B_proj"] += bp; tot["census"] += n_all
        part = "" if len(blk) == BLOCK else f" (partial: {len(blk)} contexts)"
        print(f"block {b}{part}: stage A {a} pairs = {-(-a // BATCH)} batches per labeller (exact); stage B ~{bp:.0f} "
              f"pairs = ~{bp / BATCH:.0f} batches (projected: {TRIGGER_SHARE:.0%} of unflagged pairs, as in the census); "
              f"census would be {n_all} pairs = {-(-n_all // BATCH)} batches")
    hyb = tot["A"] + tot["B_proj"]
    print(f"total hybrid: ~{hyb:.0f} pairs, ~{hyb / BATCH:.0f} batches per labeller, ~{hyb / BATCH * SEC_PER_BATCH / 3600:.1f} h "
          f"per labeller (two in parallel, {SEC_PER_BATCH} s/batch); census: {tot['census']} pairs, "
          f"{tot['census'] / BATCH:.0f} batches, {tot['census'] / BATCH * SEC_PER_BATCH / 3600:.1f} h")
    print(f"expected eligible: hybrid ~{HYBRID_YIELD * len(ctx):.0f} ({HYBRID_YIELD:.1%}), census ~{CENSUS_YIELD * len(ctx):.0f}")
    print("launch: nohup uv run python scripts/olmo_direction_label.py run --until-eligible 100 "
          "> ~/Documents/experiment-data/latent-gap-ideas/runs/olmo-direction/label.log 2>&1 < /dev/null &")


def to_census() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    (LAB / "mode.json").write_text(json.dumps({"mode": "census", "why": "fallback (b) or manual switch",
                                               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    log("mode set to census: the next run completes every block (stage C) and labels new blocks in full")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=["run", "status", "eligible", "census", "label-stage"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--until-eligible", type=int, default=100)
    ap.add_argument("--allow-partial", action="store_true", help="label a last block of fewer than 200 contexts")
    a = ap.parse_args()
    if a.dry_run:
        dry_run()
    elif a.cmd == "run":
        run(a.until_eligible, a.allow_partial)
    elif a.cmd == "label-stage":
        label_stage(Path(a.args[0]), a.args[1])
    elif a.cmd == "eligible":
        eligible()
    elif a.cmd == "census":
        to_census()
    else:
        status()
