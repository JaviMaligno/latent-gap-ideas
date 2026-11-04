"""E3 across judge families. One set of judge inputs (normalised pitch of the item + pitches of its 5 nearest
past items, rubric), shared by every judge, so judges differ only in the model.

    uv run python scripts/judge_multi.py --run pilot --prepare        # pitches (luna), cached and validated
    uv run python scripts/judge_multi.py --run pilot --judge opus      # one process per judge

Each result row stores the hash of the prompt it answered; a row whose prompt changed is judged again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from judge import DATA, JUDGE, NORMALISE, NORMALISER, RUBRIC, load_items  # noqa: E402

from latent_gap.cli_judges import judge_call  # noqa: E402
from latent_gap.domains import DOMAINS  # noqa: E402
from latent_gap.gateway import Gateway  # noqa: E402

JUDGES = {
    "sol": ("gateway", JUDGE),
    "opus": ("claude", "claude-opus-5-5"),
    "sonnet": ("claude", "claude-sonnet-5-5"),
    "haiku": ("claude", "claude-haiku-5-5"),
    "gpt61sol": ("codex", "gpt-6.1-sol"),
}
MIN_PITCH_CHARS = 40  # a reasoning model can spend the whole budget thinking and return empty content
PITCH_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pitch"],
                "properties": {"pitch": {"type": "string"}}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["coherence", "feasibility", "appeal", "novelty", "rationale"],
          "properties": {**{c: {"type": "integer", "minimum": 1, "maximum": 10}
                            for c in ("coherence", "feasibility", "appeal", "novelty")},
                         "rationale": {"type": "string"}}}


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _valid(p: str | None) -> bool:
    return p is not None and len(p.strip()) >= MIN_PITCH_CHARS


def prepare(args, domain) -> None:
    run = DATA / "runs" / args.domain / args.run
    items, P, ptexts = load_items(args.domain, args.run, args.n_cal)
    gw = Gateway(ledger=DATA / "ledger.jsonl")
    suffix = "_noresults" if domain.pitch_rule else ""  # new pitch prompt: never mix with pitches of the old one
    cache_path = run / f"pitch_cache{suffix}.json"  # key: past row index (neighbours) or "item:<arm>:<k>"
    cache = {k: v for k, v in (json.loads(cache_path.read_text()) if cache_path.exists() else {}).items() if _valid(v)}
    fb_path = run / f"pitch_fallback{suffix}.json"  # keys normalised through Codex instead of the gateway
    fallback = set(json.loads(fb_path.read_text())) if fb_path.exists() else set()

    need: dict[str, str] = {}
    for it in items:
        s = P @ it["emb"]
        it["nn"] = [int(i) for i in np.argsort(-s)[:6] if s[i] < 0.999][:5]
        it["key"] = f"item:{it['arm']}:{it['k']}"
        need.update({str(i): ptexts[i] for i in it["nn"] if str(i) not in cache})
        if it["key"] not in cache:
            need[it["key"]] = it["text"]

    def pitch(key: str, text: str) -> tuple[str, str]:
        msg = NORMALISE.format(noun=domain.noun, does=domain.does, rule=domain.pitch_rule, text=text[:2500].replace("\x00", ""))
        for budget in () if args.pitch_codex else (2000, 8000, 8000):
            try:
                p = gw.chat([{"role": "user", "content": msg}], model=NORMALISER, max_tokens=budget,
                            tag=f"{args.domain}-{args.run}-pitch").strip()
                if _valid(p):
                    return key, p
            except RuntimeError as e:  # Azure's content filter is not deterministic: retry before falling back
                if "content management policy" not in str(e):
                    raise
        # Same model through Codex (subscription): no Azure content filter, no output budget to run out of.
        fallback.add(key)
        return key, judge_call("codex", "gpt-6-luna", msg, PITCH_SCHEMA)["pitch"].strip()

    print(f"{len(need)} pitches to write", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for n, fu in enumerate(as_completed([ex.submit(pitch, k, t) for k, t in need.items()]), 1):
            k, p = fu.result()
            if _valid(p):
                cache[k] = p
            if n % 100 == 0:
                print(f"  {n}/{len(need)}", flush=True)
                cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    fb_path.write_text(json.dumps(sorted(fallback)))

    missing = 0
    with (run / "judge_inputs.jsonl").open("w") as f:
        for it in items:
            keys = [it["key"], *map(str, it["nn"])]
            if not all(k in cache for k in keys):
                missing += 1
                continue
            neighbours = "\n".join(f"- {cache[str(i)]}" for i in it["nn"])
            prompt = RUBRIC.format(noun=domain.noun, feasible=domain.feasible, appeal=domain.appeal, neighbours=neighbours, pitch=cache[it["key"]]).replace("\x00", "")
            f.write(json.dumps({"arm": it["arm"], "k": it["k"], "hash": _hash(prompt), "prompt": prompt}) + "\n")
    print(f"inputs written: {len(items) - missing}; missing a pitch: {missing}; via Codex: {len(fallback)}", flush=True)


def _gateway_call(gw: Gateway, model: str, prompt: str) -> dict:
    raw = gw.chat([{"role": "user", "content": prompt}], model=model, max_tokens=8000, json_mode=True,
                  tag="judge-multi-sol")
    return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])


def judge(args) -> None:
    run = DATA / "runs" / args.domain / args.run
    cli, model = JUDGES[args.judge]
    inputs = [json.loads(l) for l in (run / "judge_inputs.jsonl").read_text().splitlines()]
    out_path = run / f"judged_{args.judge}.jsonl"
    want = {(x["arm"], x["k"]): x["hash"] for x in inputs}
    rows = [json.loads(l) for l in out_path.read_text().splitlines()] if out_path.exists() else []
    keep = [r for r in rows if want.get((r["arm"], r["k"])) == r.get("hash")]
    if len(keep) < len(rows):  # drop answers to prompts that have since changed (e.g. an empty pitch, fixed)
        out_path.write_text("".join(json.dumps(r) + "\n" for r in keep))
    done = {(r["arm"], r["k"]) for r in keep}
    todo = [x for x in inputs if (x["arm"], x["k"]) not in done]
    print(f"{args.judge} ({cli} {model}): {len(todo)} to judge, {len(keep)} kept", flush=True)
    gw = Gateway(ledger=DATA / "ledger.jsonl") if cli == "gateway" else None

    def call(x: dict) -> dict:
        score = _gateway_call(gw, model, x["prompt"]) if gw else judge_call(cli, model, x["prompt"], SCHEMA)
        if not all(isinstance(score.get(c), int) for c in ("coherence", "feasibility", "appeal", "novelty")):
            raise ValueError(f"incomplete score: {str(score)[:200]}")
        return score

    with ThreadPoolExecutor(args.workers) as ex, out_path.open("a") as f:
        futs = {ex.submit(call, x): x for x in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            x = futs[fu]
            try:
                score = fu.result()
            except Exception as e:
                print("FAILED", x["arm"], x["k"], repr(e)[:200], flush=True)
                continue
            f.write(json.dumps({"arm": x["arm"], "k": x["k"], "judge": args.judge, "hash": x["hash"], **score}) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {x['arm']} {x['k']} coh={score['coherence']} nov={score['novelty']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--pitch-codex", action="store_true",
                    help="write new pitches with the same model through Codex (no gateway); recorded as fallback")
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--n-cal", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--judge", choices=sorted(JUDGES))
    args = ap.parse_args()
    if args.prepare:
        prepare(args, DOMAINS[args.domain])
    else:
        judge(args)


if __name__ == "__main__":
    main()
