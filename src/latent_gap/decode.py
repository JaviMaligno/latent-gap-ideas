"""Closed-loop LLM decoder (target vector -> text) and the two space-free baselines."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from latent_gap.cli_judges import RefusedError
from latent_gap.domains import Domain
from latent_gap.gateway import Gateway

K_NEIGHBOURS = 8
K_DIRECTIONS = 5
MAX_ITERS = 4
N_CANDIDATES = 3  # best-of-n per iteration; the best one is what the conversation continues from

SYSTEM = ("You design new {noun}s. You always answer with one JSON object and nothing else, "
          "with exactly these fields: {spec}")


class Corpus:
    """Past corpus: normalised embeddings plus display strings, with optional rows hidden per call."""

    def __init__(self, emb: np.ndarray, names: list[str], texts: list[str], domain: Domain):
        self.P = emb
        self.show = [domain.show(n, t) for n, t in zip(names, texts)]

    def top(self, v: np.ndarray, k: int, hidden: np.ndarray | None = None, sign: float = 1.0) -> list[int]:
        s = sign * (self.P @ v)
        if hidden is not None and len(hidden):
            s[hidden] = -np.inf
        idx = np.argpartition(-s, k)[:k]
        return idx[np.argsort(-s[idx])].tolist()


def _chat_or_none(gw: Gateway, messages: list[dict], tag: str) -> str | None:
    """A content-filter refusal from the provider counts as a failed candidate, not a crash."""
    try:
        return gw.chat(messages, json_mode=True, tag=tag)
    except RefusedError:
        return None
    except RuntimeError as e:
        if "content management policy" in str(e) or "ContentPolicyViolation" in str(e):
            return None
        raise


def _parse(raw: str) -> dict | None:
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except ValueError:
        return None


def _listing(corpus: Corpus, idx: list[int], v: np.ndarray | None = None) -> str:
    lines = []
    for i in idx:
        sim = f" (similarity {float(corpus.P[i] @ v):.3f})" if v is not None else ""
        lines.append(f"- {corpus.show[i]}{sim}")
    return "\n".join(lines)


def decode(gw: Gateway, embedder, domain: Domain, corpus: Corpus, target: np.ndarray,
           hidden: np.ndarray | None = None, tag: str = "", feedback_mode: str = "residual",
           max_iters: int = MAX_ITERS, neighbours: str = "true",
           neighbour_target: np.ndarray | None = None, neighbour_hidden: np.ndarray | None = None,
           candidate_workers: int | None = None) -> dict:
    """Iteratively write an item whose embedding approaches `target`. Returns every iteration and the best one."""
    sys = SYSTEM.format(noun=domain.noun, spec=domain.json_spec)
    if neighbours not in ("true", "swap", "none"):
        raise ValueError(f"unknown neighbour condition: {neighbours}")
    if neighbours != "true" and feedback_mode != "score":
        raise ValueError("neighbour interventions require score-only feedback")
    if neighbours == "swap" and neighbour_target is None:
        raise ValueError("swap requires a donor target")
    if neighbours == "swap":
        excluded = np.union1d([] if hidden is None else hidden,
                              [] if neighbour_hidden is None else neighbour_hidden).astype(np.int64)
        nn = corpus.top(neighbour_target, K_NEIGHBOURS, excluded)
    else:
        nn = corpus.top(target, K_NEIGHBOURS, hidden) if neighbours == "true" else []
    first = (f"There is a location in the space of all {domain.noun}s that we want to fill with a new one. "
             f"These existing {domain.noun}s are the closest to that location (similarity to it in brackets; "
             f"1.0 would be exactly on it):\n{_listing(corpus, nn, target)}\n\n"
             f"Write a NEW {domain.noun} that sits exactly at that location: not a copy of any of them, but the "
             f"item whose description would be closer to the location than all of them.")
    if neighbours == "none":
        first = (f"There is a location in the space of all {domain.noun}s that we want to fill with a new one. "
                 f"Write a NEW {domain.noun} that sits exactly at that location.")
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": first}]
    history = []
    iters, n_cand = (1, max_iters * N_CANDIDATES) if feedback_mode == "none" else (max_iters, N_CANDIDATES)
    for it in range(iters):
        with ThreadPoolExecutor(n_cand if candidate_workers is None else candidate_workers) as ex:
            raws = list(ex.map(lambda _: _chat_or_none(gw, messages, tag), range(n_cand)))
        raws = [r for r in raws if r is not None]
        cands = []
        for raw in raws:
            item = _parse(raw)
            if item is None:
                continue
            text = domain.render(item)
            e = embedder.embed([text])[0]
            cands.append((float(e @ target), raw, item, text, e))
        if not cands:
            history.append({"iter": it, "raw": raws[0] if raws else None, "sim": None})
            if not raws:
                continue
            messages += [{"role": "assistant", "content": raws[0]},
                         {"role": "user", "content": "That was not valid JSON. Answer again with the JSON object only."}]
            continue
        sim, raw, item, text, e = max(cands, key=lambda c: c[0])
        history.append({"iter": it, "item": item, "text": text, "sim": sim, "emb": e.tolist(),
                        "candidate_sims": [c[0] for c in cands]})
        if it == iters - 1:
            break
        best_sim = max(h["sim"] for h in history if h["sim"] is not None)
        if feedback_mode == "score":
            feedback = (f"Your item reached similarity {sim:.3f} to the location (best so far {best_sim:.3f}). "
                        f"The existing {domain.noun}s listed above are still the closest ones. Write a different NEW "
                        f"{domain.noun} that gets closer to the location.")
            if neighbours == "none":
                feedback = (f"Your item reached similarity {sim:.3f} to the location (best so far {best_sim:.3f}). "
                            f"Write a different NEW {domain.noun} that gets closer to the location.")
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
            continue
        r = target - e
        r /= np.linalg.norm(r)
        towards = corpus.top(r, K_DIRECTIONS, hidden)
        away = corpus.top(r, K_DIRECTIONS, hidden, sign=-1.0)
        feedback = (f"Your item reached similarity {sim:.3f} to the location (best so far "
                    f"{max(h['sim'] for h in history if h['sim'] is not None):.3f}). To get closer, move your item "
                    f"TOWARDS what these {domain.noun}s have in common:\n{_listing(corpus, towards)}\n\n"
                    f"and AWAY from what these have in common:\n{_listing(corpus, away)}\n\n"
                    f"Write a revised NEW {domain.noun}.")
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
    ok = [h for h in history if h["sim"] is not None]
    best = max(ok, key=lambda h: h["sim"]) if ok else None
    return {"neighbours": nn, "history": history, "best": best}


def plain(gw: Gateway, embedder, domain: Domain, k: int, tag: str = "") -> dict:
    sys = SYSTEM.format(noun=domain.noun, spec=domain.json_spec)
    user = (f"Invent a brand-new {domain.noun}, unlike any {domain.noun} that exists. "
            f"(This is request number {k}; other requests are answered independently.)")
    return _single(gw, embedder, domain, sys, user, tag)


def plain_diverse(gw: Gateway, embedder, domain: Domain, previous: list[str], tag: str = "") -> dict:
    """PLAIN with diversity control: the generator sees one line per idea it already proposed in this run and must
    invent something clearly different from all of them (as a person asking for many ideas would)."""
    sys = SYSTEM.format(noun=domain.noun, spec=domain.json_spec)
    user = f"Invent a brand-new {domain.noun}, unlike any {domain.noun} that exists."
    if previous:
        listing = "\n".join(f"- {p}" for p in previous)
        user += (f" It must also be clearly different from every idea you have already proposed in this series "
                 f"(different premise, setting and core mechanic, not a variation):\n{listing}")
    return _single(gw, embedder, domain, sys, user, tag)


def summary_line(item: dict) -> str:
    """One line per earlier idea for plain_diverse: its title and the first sentence of its description."""
    title = str(item.get("name") or item.get("title") or "").strip()
    body = str(item.get("about") or item.get("abstract") or "").strip()
    first = body.split(". ")[0][:160]
    return f"{title}: {first}" if title else first


def pair(gw: Gateway, embedder, domain: Domain, corpus: Corpus, a: int, b: int, tag: str = "") -> dict:
    sys = SYSTEM.format(noun=domain.noun, spec=domain.json_spec)
    user = (f"Here are two very different existing {domain.noun}s:\n{_listing(corpus, [a, b])}\n\n"
            f"Invent one NEW {domain.noun} that combines them into something coherent that does not exist yet.")
    return _single(gw, embedder, domain, sys, user, tag)


def _single(gw: Gateway, embedder, domain: Domain, sys: str, user: str, tag: str) -> dict:
    for _ in range(3):
        raw = _chat_or_none(gw, [{"role": "system", "content": sys}, {"role": "user", "content": user}], tag)
        item = _parse(raw) if raw is not None else None
        if item is not None:
            text = domain.render(item)
            e = embedder.embed([text])[0]
            return {"best": {"item": item, "text": text, "emb": e.tolist(), "sim": None}}
    return {"best": None, "raw": raw}
