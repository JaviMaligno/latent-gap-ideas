"""Coherent no-neighbour support search: a lower bound on h_coh(t) by black-box LLM optimisation.

The writer sees NO corpus item, neighbour or direction listing: only the domain's JSON spec, its own earlier
candidates (text) and their cosine scores to a hidden target. Pre-registration: docs/prereg/coherent_search.md (rev 2).

Modes (same writer, same budget of K x (T+1) candidates per target):
  search  K diversified seeds, then T rounds of K variants with score feedback; coherence required (the main arm).
  random  K x (T+1) independent diversified seed calls, no feedback (separates feedback from sampling).
  free    the search loop with the coherence requirement removed from the prompt and no gate (h_any_blackbox).
Arms: FAR and GAP (the test), REAL (positive control: hidden real items, the hidden item masked from the corpus).

Usage:
  uv run python scripts/coherent_search.py --domain steam --backend claude:claude-sonnet-5-5 --workers 4
  uv run python scripts/coherent_search.py --domain steam --mode random --backend claude:claude-sonnet-5-5
  --backend gateway uses gpt-6-luna through a LiteLLM gateway (PAID; --max-usd caps the total spend of
  that run label over every invocation and both domains, checked against the ledger before each call).
Output: runs/<domain>/coherent_search/<label>/items.jsonl (one record per target), cands/<arm>-<k>.json checkpoints
(a judging failure re-runs only the judging) and summary.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from latent_gap.cli_judges import CliChat, RateLimitedError, RefusedError
from latent_gap.coherent import bounds_violation, gibberish_rule, gibberish_stats, judge
from latent_gap.domains import DOMAINS, Domain
from latent_gap.embedder import LocalQwen
from latent_gap.gateway import Gateway, price
from latent_gap.targets import DUP_COS

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
GATEWAY_MODEL = "gpt-6-luna"
PROMPT_VERSION = "v3"   # rev 2 of the prereg: diversified seeds, contrastive history, free/random modes
SHOW_BEST, SHOW_LOW = 4, 2
NULL_DRAWS = 200
DEFAULT_KS = {"search": {"FAR": range(1, 21), "GAP": range(1, 21), "REAL": range(1, 11)},
              "random": {"FAR": range(1, 6), "GAP": range(1, 6)},
              "free": {"FAR": range(1, 6), "GAP": range(1, 6)}}

# Seed constraints, fixed before any run and never tuned on scores: each seed call gets one (genre x setting or
# subfield x kind) pair, drawn without replacement with a per-target RNG.
SEED_AXES = {
    "steam": (["turn-based strategy", "racing", "sports management", "rhythm", "fighting", "visual novel",
               "city builder", "roguelike deckbuilder", "flight simulator", "horror", "platformer", "puzzle",
               "farming sim", "tactical shooter", "MMORPG", "point-and-click adventure", "tower defense",
               "dating sim", "survival crafting", "educational", "trading card", "open-world RPG", "party",
               "idle clicker"],
              ["in a contemporary city", "in medieval Europe", "in outer space", "underwater", "in a fantasy kingdom",
               "in the Wild West", "in a high school", "in a post-apocalyptic wasteland", "in ancient Egypt",
               "in a hospital", "on a farm", "in a cyberpunk megacity", "during World War II",
               "in an abstract geometric world", "in a restaurant kitchen", "in a sports stadium",
               "in a haunted house", "in prehistoric times"]),
    "arxiv": (["computer vision", "natural language processing", "cryptography and security",
               "distributed computing", "databases", "computational complexity", "robotics",
               "human-computer interaction", "information retrieval", "programming languages", "software engineering",
               "computer networks", "learning theory", "computer graphics", "formal verification", "quantum computing",
               "computational geometry", "multi-agent systems", "signal processing", "bioinformatics",
               "operating systems", "algorithmic game theory", "data structures and algorithms",
               "computers and society"],
              ["proposes a new algorithm", "presents the design of a system", "proves a theoretical result",
               "introduces a dataset or benchmark", "reports an empirical study", "surveys the area",
               "proposes a formal model", "identifies a limitation of existing methods", "describes a tool",
               "studies a security or privacy risk"]),
}
SEED_FORM = {"steam": "a {0} game set {1}", "arxiv": "a paper in {0} that {1}"}

SYSTEM = ("You write {noun} descriptions. You always answer with JSON and nothing else. Each {noun} is one JSON "
          "object with exactly these fields: {spec}")
SEED = ("Write a new {noun}: an original one, written as a real, sensible description. Make it {constraint}. "
        "Answer with the JSON object only.")
ITER = ("We are searching for a {noun} description that lands at a hidden location in an embedding space of "
        "{noun} descriptions. The score of a description is its cosine similarity to that hidden location "
        "(1.0 would be exactly on it). Nothing else is known about the location. Below are earlier descriptions "
        "with their scores: the best so far, all of the latest round (good or bad) and the lowest so far, so you "
        "can see which moves helped and which did not.\n\n{shown}\n\nBest score so far: {best:.4f} (search round "
        "{it} of {T}). Write {K} NEW {noun} descriptions that you expect to score HIGHER: variants of the best ones, "
        "or different moves suggested by how the scores differ. At least one of the {K} must be a bold exploration: "
        "a substantially different {noun} (another kind of subject, setting and form) to probe other regions, since "
        "the best so far may be a local optimum. {rule} Answer with JSON only: {{\"candidates\": [<{K} {noun} "
        "objects>]}}.")
RULE = {"search": ("Each must still be a real, sensible {noun} description in normal language (no lists of random "
                   "words, no keyword stuffing, no other languages)."),
        "free": "The fields may contain any text at all: it does not need to read as a real description."}


class BudgetExceeded(RuntimeError):
    pass


class TargetFailed(RuntimeError):
    pass


def family(spec: str) -> str:
    """Model family, for the writer/judge self-preference check: 'claude' or 'gpt' (gateway = gpt-6-luna)."""
    return "gpt" if spec == "gateway" or spec.startswith("codex:") else spec.split(":")[0]


# ---------------------------------------------------------------- writer backends

class SpendGuard:
    """Ledger spend of every call tagged `cohsearch-<label>-` (all invocations, both domains), checked per call."""

    def __init__(self, label: str, cap: float):
        self.prefix, self.cap, self.lock = f'"tag": "cohsearch-{label}-', cap, threading.Lock()

    def spent(self) -> float:
        path = DATA / "ledger.jsonl"
        if not path.exists():
            return 0.0
        tot = 0.0
        for line in path.read_text().splitlines():
            if self.prefix in line:
                r = json.loads(line)
                pin, pout = price(r["model"])
                tot += (r["in"] * pin + r["out"] * pout) / 1e6
        return tot

    def check(self) -> None:
        with self.lock:
            s = self.spent()
        if s >= self.cap:
            raise BudgetExceeded(f"gateway spend {s:.4f} USD reached --max-usd {self.cap:.2f}")


class Writer:
    """One stateless call -> JSON string. `gateway` (PAID, one shared Gateway) or a subscription CLI 'claude:<m>'.
    Refusals, timeouts and per-call errors return None (a failed candidate); 3 errors in one target fail the target;
    usage limits that outlast the CLI waits and the spend cap propagate."""

    MAX_ERRORS = 3

    def __init__(self, spec: str, domain: Domain, K: int, tag: str, gw: Gateway | None, guard: SpendGuard | None):
        self.spec, self.tag, self.gw, self.guard, self.errors = spec, tag, gw, guard, 0
        self.lock = threading.Lock()
        if spec != "gateway":
            many = {"type": "object", "additionalProperties": False, "required": ["candidates"],
                    "properties": {"candidates": {"type": "array", "items": domain.schema,
                                                  "minItems": K, "maxItems": K}}}
            self.one, self.many = CliChat(spec, domain.schema), CliChat(spec, many)

    def call(self, messages: list[dict], many: bool, log) -> str | None:
        try:
            if self.spec == "gateway":
                self.guard.check()
                return self.gw.chat(messages, json_mode=True, max_tokens=16000, tag=self.tag)
            return (self.many if many else self.one).chat(messages)
        except (RefusedError, BudgetExceeded, RateLimitedError) as e:
            if isinstance(e, RefusedError):
                return None
            raise
        except Exception as e:
            if "content management policy" in str(e) or "ContentPolicyViolation" in str(e):
                return None
            with self.lock:
                self.errors += 1
                n = self.errors
            log(f"writer error {n}/{self.MAX_ERRORS}: {type(e).__name__}: {str(e)[:200]}")
            if n >= self.MAX_ERRORS:
                raise TargetFailed(f"{n} writer errors") from e
            return None


def label(spec: str) -> str:
    return f"gateway-{GATEWAY_MODEL}" if spec == "gateway" else spec.split(":", 1)[1]


def _objects(raw: str | None, domain: Domain, many: bool) -> list[dict | None]:
    """Parse the answer into candidate objects; invalid ones become None (kept as failed candidates). Lenient on the
    container (gateway json_object mode does not enforce the schema): {"candidates": [...]}, a bare list, or a bare
    candidate object are all accepted."""
    if raw is None:
        return [None]
    try:
        s = raw.strip()
        d = json.loads(s if s[:1] in "[{" else s[s.index("{"): s.rindex("}") + 1])
    except ValueError:
        return [None]
    if isinstance(d, dict) and isinstance(d.get("candidates"), list):
        objs = d["candidates"]
    elif isinstance(d, list):
        objs = d
    else:
        objs = [d]
    if not many:
        objs = objs[:1]
    out = []
    for o in objs:
        ok = isinstance(o, dict) and all(f in o for f in domain.schema["required"])
        if ok:
            for f, sch in domain.schema["properties"].items():
                v = o[f]
                ok &= isinstance(v, str) if sch["type"] == "string" else (
                    isinstance(v, list) and all(isinstance(x, str) for x in v))
        out.append(o if ok else None)
    return out


# ---------------------------------------------------------------- one target

def seed_prompts(domain: Domain, rng: random.Random, n: int) -> list[str]:
    a, b = SEED_AXES[domain.name]
    pairs = list(itertools.product(a, b))
    return [SEED_FORM[domain.name].format(*p) for p in rng.sample(pairs, n)]


def search_target(domain: Domain, writer: Writer, emb: LocalQwen, P: np.ndarray, hidden_pos: np.ndarray,
                  t: np.ndarray, T: int, K: int, mode: str, rng: random.Random, log) -> list[dict]:
    sysmsg = {"role": "system", "content": SYSTEM.format(noun=domain.noun, spec=domain.json_spec)}
    cands: list[dict] = []

    def score(objs: list[dict | None], it: int, constraints: list[str | None]) -> None:
        for slot, (o, con) in enumerate(zip(objs, constraints)):
            rec = {"iter": it, "slot": slot, "valid": False, "status": "invalid", "seed_constraint": con}
            if o is not None:
                text = domain.render(o)
                n_tok = len(emb.tok.encode(text).ids)
                bad = bounds_violation(domain.name, o, n_tok)
                e = emb.embed_one(text)
                sims = P @ e
                sims[hidden_pos] = -np.inf
                st = gibberish_stats(text)
                rec |= {"item": o, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "n_tokens": n_tok, "score": float(e @ t), "corpus_max": float(sims.max()), **st,
                        "gib_rule": gibberish_rule(st), "valid": not bad, "status": bad or "ok"}
                rec["gibberish"] = rec["gib_rule"] != ""
            cands.append(rec)

    def seed_round(it: int, n: int) -> None:
        cons = seed_prompts(domain, rng, n)
        msgs = [[sysmsg, {"role": "user", "content": SEED.format(noun=domain.noun, constraint=c)}] for c in cons]
        with ThreadPoolExecutor(K) as ex:
            raws = list(ex.map(lambda m: writer.call(m, False, log), msgs))
        score([_objects(r, domain, many=False)[0] for r in raws], it, cons)

    if mode == "random":  # K x (T+1) independent seeds, no feedback
        for it in range(T + 1):
            seed_round(it, K)
        return cands
    seed_round(0, K)
    for it in range(1, T + 1):
        ok = sorted((c for c in cands if c["valid"]), key=lambda c: -c["score"])
        if not ok:  # nothing valid yet: re-seed
            seed_round(it, K)
            continue
        groups, seen = [], set()
        for group, members in (("best so far", ok[:SHOW_BEST]),
                               ("latest round", [c for c in cands if c["iter"] == it - 1 and c["valid"]]),
                               ("lowest so far", ok[-SHOW_LOW:][::-1])):
            for c in members:
                if c["sha256"] not in seen:
                    seen.add(c["sha256"])
                    groups.append((group, c))
        listing = "\n\n".join(f"[{g}] Score {c['score']:.4f}:\n{json.dumps(c['item'], ensure_ascii=False)}"
                              for g, c in groups)
        msg = {"role": "user", "content": ITER.format(noun=domain.noun, shown=listing, best=ok[0]["score"], it=it,
                                                      T=T, K=K, rule=RULE[mode].format(noun=domain.noun))}
        objs = _objects(writer.call([sysmsg, msg], True, log), domain, many=True)
        objs = (objs + [None] * K)[:K]
        if not any(objs):
            log(f"  iter {it}: WARNING 0 parseable candidates")
        score(objs, it, [None] * K)
        log(f"  iter {it}: best {max(c['score'] for c in cands if c['valid']):.4f}")
    return cands


def distinct_valid(cands: list[dict]) -> list[dict]:
    out, seen = [], set()
    for c in sorted((c for c in cands if c["valid"]), key=lambda c: -c["score"]):
        if c["sha256"] not in seen:
            seen.add(c["sha256"])
            out.append(c)
    return out


def judge_target(domain: Domain, cands: list[dict], cfg: dict, rng: random.Random, cache: dict,
                 lock: threading.Lock) -> int:
    """Gate path: distinct valid non-gibberish candidates in descending score order, judge 1 then (if it accepts)
    judge 2, until one passes both, at most judge_cap candidates. Plus a random sample of distinct valid candidates,
    drawn before any judging and rated by judge 1 only (the unbiased P3 coherence rate). free mode: judge 1 on the
    best candidate and the sample only. Ratings are copied to duplicates. Returns the number of judge errors."""
    th, j1, j2 = cfg["coh_pass"], cfg["judge"], cfg["judge2"]
    distinct = distinct_valid(cands)
    sample = rng.sample(distinct, min(cfg["judge_sample"], len(distinct)))
    rated: dict[tuple[str, str], dict] = {}

    def rate(spec: str, c: dict) -> int | None:
        key = (spec, c["sha256"])
        with lock:
            hit = cache.get(key)
        if hit is None:
            hit = judge(spec, domain.noun, c["text"])
            if hit.get("coherence") is not None:
                with lock:
                    cache[key] = hit
        rated[key] = hit
        return hit.get("coherence")

    if cfg["mode"] == "free":
        if distinct:
            rate(j1, distinct[0])
    else:
        for c in [c for c in distinct if not c["gibberish"]][: cfg["judge_cap"]]:
            if (rate(j1, c) or 0) >= th and (j2 == "none" or (rate(j2, c) or 0) >= th):
                break
    for c in sample:
        rate(j1, c)
    sample_sha = {c["sha256"] for c in sample}
    for c in cands:
        if not c.get("valid"):
            continue
        c["in_sample"] = c["sha256"] in sample_sha
        for spec, f in ((j1, "coherence"), (j2, "coherence2")):
            if (spec, c["sha256"]) in rated:
                c[f], c[f + "_reason"] = rated[spec, c["sha256"]].get("coherence"), rated[spec, c["sha256"]].get("reason")
    return sum(h.get("coherence") is None for h in rated.values())


def passes(c: dict, cfg: dict) -> bool:
    th = cfg["coh_pass"]
    return bool(c.get("valid") and not c["gibberish"] and (c.get("coherence") or 0) >= th
                and (cfg["judge2"] == "none" or (c.get("coherence2") or 0) >= th))


def summarise(rec: dict) -> dict:
    cfg = rec["config"]
    ok = [c for c in rec["candidates"] if c["valid"]]
    best = max(ok, key=lambda c: c["score"]) if ok else None
    passing = [c for c in ok if passes(c, cfg)]
    bc = max(passing, key=lambda c: c["score"]) if passing else None
    return {"domain": rec["domain"], "arm": rec["arm"], "k": rec["k"], "mode": cfg["mode"],
            "n_cand": len(rec["candidates"]), "n_valid": len(ok),
            "n_bounds": sum(c["status"].startswith(("genres", "tags", "about", "title", "abstract", "tokens"))
                            for c in rec["candidates"]),
            "n_gibberish": sum(c["gibberish"] for c in ok), "n_judge_errors": rec["n_judge_errors"],
            "ceiling": rec["ceiling"], "null_matched": rec["null_matched"], "h_all": best["score"] if best else None,
            "h_coh_lower": bc["score"] if bc else None, "h_coh_corpus_max": bc["corpus_max"] if bc else None,
            "seed_best": max((c["score"] for c in ok if c["iter"] == 0), default=None),
            "seed_div": rec["seed_div"], "in_tokens": rec.get("in_tokens"), "out_tokens": rec.get("out_tokens"),
            "writer_seconds": rec["writer_seconds"], "judge_seconds": rec["judge_seconds"]}


def _ledger_tokens(tag: str) -> tuple[int, int]:
    path = DATA / "ledger.jsonl"
    if not path.exists():
        return 0, 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if f'"{tag}"' in line]
    return sum(r["in"] for r in rows), sum(r["out"] for r in rows)


def parse_ks(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=sorted(SEED_AXES))
    ap.add_argument("--mode", default="search", choices=["search", "random", "free"])
    ap.add_argument("--arms", default="", help="default: FAR,GAP,REAL for search, FAR,GAP for the controls")
    ap.add_argument("--ks", default="", help="target indices, e.g. '0' or '1-20' (default: the prereg set per arm)")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--backend", default="claude:claude-sonnet-5-5", help="'gateway' (PAID) or 'claude:<model>'")
    ap.add_argument("--judge", default="claude:claude-sonnet-5-5", help="subscription CLI spec, or 'none'")
    ap.add_argument("--judge2", default="auto", help="second blind judge; 'auto' = codex:gpt-6-luna for claude "
                    "writers, claude:claude-haiku-5-5 for the gateway (gpt-6-luna) writer; or 'none'")
    ap.add_argument("--coh-pass", type=int, default=7)
    ap.add_argument("--judge-cap", type=int, default=15)
    ap.add_argument("--judge-sample", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4, help="targets in parallel")
    ap.add_argument("--tag", default="", help="suffix for the output dir (e.g. smoke)")
    ap.add_argument("--max-usd", type=float, default=0.10,
                    help="gateway only: cap on the total spend of this run label (owner's OK above 0.50)")
    args = ap.parse_args()

    judge2 = args.judge2 if args.judge2 != "auto" else (
        "claude:claude-haiku-5-5" if args.backend == "gateway" else "codex:gpt-6-luna")
    cfg = {"mode": args.mode, "T": args.T, "K": args.K, "judge": args.judge, "judge2": judge2,
           "coh_pass": args.coh_pass, "judge_cap": args.judge_cap, "judge_sample": args.judge_sample,
           "prompt": PROMPT_VERSION}
    domain = DOMAINS[args.domain]
    root = DATA / "prepared" / args.domain
    meta = pd.read_parquet(root / "corpus.parquet", columns=["split", "adult"])
    E = np.load(root / "emb_qwen3e06.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    keep_idx = np.flatnonzero((~meta.adult & (meta.split == "past")).to_numpy())
    held_idx = np.flatnonzero((~meta.adult & (meta.split != "past")).to_numpy())
    P, H = E[keep_idx], E[held_idx]
    del E
    arms = args.arms.split(",") if args.arms else list(DEFAULT_KS[args.mode])
    with np.load(DATA / f"runs/{args.domain}/pilot/targets.npz", allow_pickle=True) as z:
        tg = {a: z[a] for a in arms}  # NpzFile lazy reads are not thread-safe
        hidden = {k: np.asarray(v, dtype=int) for k, v in enumerate(z["REAL_hidden"])} if "REAL" in arms else {}
    jobs = [(arm, k) for arm in arms for k in (parse_ks(args.ks) if args.ks else DEFAULT_KS[args.mode][arm])]

    name = label(args.backend) + ("" if args.mode == "search" else f"-{args.mode}") + (f"-{args.tag}" if args.tag else "")
    out = DATA / f"runs/{args.domain}/coherent_search/{name}"
    (out / "cands").mkdir(parents=True, exist_ok=True)
    items = out / "items.jsonl"
    done = {}
    if items.exists():
        for line in items.read_text().splitlines():
            r = json.loads(line)
            done[r["arm"], r["k"]] = r
    for r in done.values():
        if r.get("config") != cfg:
            sys.exit(f"{items} was made with {r.get('config')}; this call has {cfg}. Use another --tag.")
    todo = [j for j in jobs if j not in done]

    gw = guard = None
    if args.backend == "gateway":
        guard = SpendGuard(name, args.max_usd)
        spent = guard.spent()
        # Per target: K seed calls + T rounds. Smoke rev 2 (2026-10-26, Steam, K=2): see the prereg; projected with
        # margin as seed 150 in / 500 out, round (500 + 450 x shown) in / 500 x K out, shown = SHOW_BEST+K+SHOW_LOW.
        pin, pout = price(GATEWAY_MODEL)
        shown = SHOW_BEST + args.K + SHOW_LOW
        calls_seed = args.K * (args.T + 1) if args.mode == "random" else args.K
        rounds = 0 if args.mode == "random" else args.T
        proj = len(todo) * (calls_seed * (150 * pin + 500 * pout)
                            + rounds * ((500 + 450 * shown) * pin + 500 * args.K * pout)) / 1e6
        print(f"gateway: already spent ${spent:.4f} under this label; projected ${proj:.3f} for {len(todo)} targets "
              f"(cap ${args.max_usd:.2f}, total)", flush=True)
        if spent + proj > args.max_usd:
            sys.exit("refusing: spent + projected cost above --max-usd (paid company spend needs the owner's OK)")
        gw = Gateway(ledger=DATA / "ledger.jsonl", model=GATEWAY_MODEL)

    print(f"{args.domain} {args.mode}: {len(todo)} targets to do, {len(done)} done -> {out}", flush=True)
    emb = LocalQwen()
    run_id = time.strftime("%Y%m%d%H%M%S")
    cache: dict = {}
    lock = threading.Lock()
    stop = threading.Event()

    def one(job):
        arm, k = job
        log = lambda m: print(f"[{arm} {k}] {m}", flush=True)
        if stop.is_set():
            return None
        try:
            t = tg[arm][k].astype(np.float32)
            t /= np.linalg.norm(t)
            sims = P @ t
            hs = H @ t
            if arm == "REAL":  # REAL_hidden = positions in P (the item and its near-duplicates, cos > DUP_COS)
                hidden_pos = np.union1d(hidden[k], np.flatnonzero(sims > DUP_COS)).astype(int)
                hs = hs[hs <= DUP_COS]
            else:
                hidden_pos = np.array([], dtype=int)
            sims[hidden_pos] = -np.inf
            nrng = np.random.default_rng(int(hashlib.sha256(f"{args.domain}-{arm}-{k}".encode()).hexdigest()[:8], 16))
            refs = {"ceiling": float(sims.max()), "corpus_q99": float(np.quantile(sims[np.isfinite(sims)], 0.99)),
                    "heldout_ceiling": float(hs.max()),
                    "null_matched": float(np.median([nrng.choice(hs, args.K * (args.T + 1), replace=False).max()
                                               for _ in range(NULL_DRAWS)]))}
            ck = out / "cands" / f"{arm}-{k}.json"
            if ck.exists() and json.loads(ck.read_text())["config"] == cfg:
                c0 = json.loads(ck.read_text())
                cands, wsec, tok = c0["candidates"], c0["writer_seconds"], c0.get("tokens")
                log("search checkpoint found: judging only")
            else:
                tag = f"cohsearch-{name}-{args.domain}-{arm}-{k}-{run_id}"
                writer = Writer(args.backend, domain, args.K, tag, gw, guard)
                t0 = time.time()
                cands = search_target(domain, writer, emb, P, hidden_pos, t, args.T, args.K, args.mode,
                                      random.Random(f"{args.domain}-{arm}-{k}-seeds"), log)
                wsec = round(time.time() - t0, 1)
                tok = _ledger_tokens(tag) if gw else None
                ck.write_text(json.dumps({"config": cfg, "candidates": cands, "writer_seconds": wsec, "tokens": tok},
                                         ensure_ascii=False))
            t1 = time.time()
            nerr = 0
            if args.judge != "none":
                nerr = judge_target(domain, cands, cfg, random.Random(f"{args.domain}-{arm}-{k}"), cache, lock)
            seeds = [c for c in cands if c["iter"] == 0 and c["valid"]]
            if len(seeds) > 1:
                S = np.stack([emb.embed_one(c["text"]) for c in seeds])
                G = S @ S.T
                seed_div = float(G[np.triu_indices(len(seeds), 1)].mean())
            else:
                seed_div = None
            rec = {"domain": args.domain, "arm": arm, "k": k, "backend": args.backend, "config": cfg, **refs,
                   "families": {"writer": family(args.backend), "judge": family(args.judge),
                                "judge2": family(judge2) if judge2 != "none" else None},
                   "seed_div": seed_div, "candidates": cands, "n_judge_errors": nerr,
                   "writer_seconds": wsec, "judge_seconds": round(time.time() - t1, 1)}
            if tok:
                rec["in_tokens"], rec["out_tokens"] = tok
            s = summarise(rec)
            with lock, items.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"done: ceiling {s['ceiling']:.4f} null {s['null_matched']:.4f} h_all {s['h_all']} h_coh "
                f"{s['h_coh_lower']} ({s['writer_seconds']}s write, {s['judge_seconds']}s judge, {nerr} judge errors)")
            return rec
        except BudgetExceeded as e:
            stop.set()
            log(f"STOP: {e}")
        except Exception as e:  # noqa: BLE001  (left undone; a rerun resumes it)
            log(f"FAILED, left undone: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        return None

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, todo))
    failed = sum(r is None for r in res)
    if not items.exists():
        sys.exit(f"no target finished ({failed} failed)")
    recs = [json.loads(line) for line in items.read_text().splitlines()]
    df = pd.DataFrame([summarise(r) for r in recs]).sort_values(["arm", "k"])
    df.to_csv(out / "summary.csv", index=False)
    print(df.to_string(index=False))
    if failed:
        print(f"{failed} targets failed or were stopped; rerun the same command to resume them", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
