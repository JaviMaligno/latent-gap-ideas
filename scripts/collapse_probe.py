"""Why does gpt-6-luna's PLAIN collapse more via Codex (0.811) than via the gateway (0.574)? One factor at a time.

Exploratory (not pre-registered). PLAIN prompt, Steam, 50 independent calls per condition. Outcome: median pairwise
cosine among the 50 items (lower = more diverse) with a bootstrap 95% CI over items, plus the most repeated title word.
Gateway conditions reproduce one Codex difference each; what remains between G-all and Codex is the harness's own
system prompt, which cannot be removed. Temperature is not a factor: luna accepts only the default.

    uv run python scripts/collapse_probe.py --conds G0,G0b,Geffort,Gschema,Gflat,Gall,Clow,Chigh
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from latent_gap.cli_judges import CliChat, render
from latent_gap.decode import SYSTEM, _parse
from latent_gap.domains import DOMAINS
from latent_gap.embedder import LocalQwen
from latent_gap.gateway import Gateway

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = DATA / "runs" / "steam" / "collapse"
N = 50
SCHEMA_FMT = lambda s: {"type": "json_schema", "json_schema": {"name": "item", "schema": s, "strict": True}}

# name -> (backend, effort, structured schema, flattened prompt)
CONDS = {
    "G0": ("gateway", None, False, False),        # as in the grid (luna-gateway PLAIN)
    "G0b": ("gateway", None, False, False),       # identical repeat: run-to-run noise
    "Geffort": ("gateway", "medium", False, False),
    "Gschema": ("gateway", None, True, False),
    "Gflat": ("gateway", None, False, True),
    "Gall": ("gateway", "medium", True, True),    # all three Codex differences at once
    "Cmed": ("codex", "medium", True, True),      # as in the grid (luna-Codex PLAIN)
    "Clow": ("codex", "low", True, True),
    "Chigh": ("codex", "high", True, True),
}


def messages(domain, k: int) -> list[dict]:
    user = (f"Invent a brand-new {domain.noun}, unlike any {domain.noun} that exists. "
            f"(This is request number {k}; other requests are answered independently.)")
    return [{"role": "system", "content": SYSTEM.format(noun=domain.noun, spec=domain.json_spec)},
            {"role": "user", "content": user}]


def call(cond: str, domain, gw, k: int) -> dict | None:
    backend, effort, schema, flat = CONDS[cond]
    msgs = messages(domain, k)
    if flat:
        msgs = [{"role": "user", "content": render(msgs)}]
    for _ in range(3):
        try:
            if backend == "gateway":
                extra = {**({"reasoning_effort": effort} if effort else {}),
                         **({"response_format": SCHEMA_FMT(domain.schema)} if schema else {})}
                raw = gw.chat(msgs, json_mode=True, tag=f"collapse-{cond}", extra=extra)
            else:
                from latent_gap import cli_judges
                raw = json.dumps(cli_judges._codex(render(msgs), "gpt-6-luna", domain.schema, 600, effort=effort))
        except Exception as e:  # noqa: BLE001
            print("FAILED", cond, k, repr(e)[:200], flush=True)
            continue
        item = _parse(raw)
        if item is not None:
            return item
    return None


def stats(E: np.ndarray, names: list[str], rng) -> dict:
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    med = float(np.median(S[iu]))
    bs = []
    for _ in range(2000):
        i = rng.choice(len(E), len(E), replace=True)
        Sb = S[np.ix_(i, i)]
        m = ~np.eye(len(i), dtype=bool) & (i[:, None] != i[None, :])
        bs.append(np.median(Sb[m]))
    word, count = collections.Counter(n.split()[0].strip(":,").lower() if n else "" for n in names).most_common(1)[0]
    return {"n": len(E), "median_cos": round(med, 3), "ci": [round(float(np.quantile(bs, .025)), 3),
            round(float(np.quantile(bs, .975)), 3)], "top_word": f"{word} {count}/{len(E)}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    domain = DOMAINS["steam"]
    OUT.mkdir(parents=True, exist_ok=True)
    gw = Gateway(ledger=DATA / "ledger.jsonl")
    emb = LocalQwen()
    rng = np.random.default_rng(20261025)
    for cond in args.conds.split(","):
        path = OUT / f"{cond}.jsonl"
        done = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        todo = [k for k in range(N) if k not in {d["k"] for d in done}]
        with ThreadPoolExecutor(args.workers if CONDS[cond][0] == "gateway" else 4) as ex, path.open("a") as f:
            for k, item in zip(todo, ex.map(lambda k: call(cond, domain, gw, k), todo)):
                if item is not None:
                    rec = {"k": k, "item": item}
                    f.write(json.dumps(rec) + "\n")
                    done.append(rec)
        E = emb.embed([domain.render(d["item"]) for d in done])
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        print(cond, CONDS[cond], stats(E, [d["item"].get("name", "") for d in done], rng), flush=True)


if __name__ == "__main__":
    main()
