"""Minimal client for a LiteLLM gateway (set LG_BASE_URL and LG_KEY_FILE). Logs token usage to a JSONL ledger."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import httpx
import numpy as np

# Any OpenAI-compatible endpoint; e.g. OpenRouter: LG_BASE_URL=https://openrouter.ai/api/v1 LG_KEY_FILE=~/.openrouter-key
# and model ids like ALIAS=openai/gpt-4o-2024-08-06.
BASE_URL = os.environ.get("LG_BASE_URL", "https://openrouter.ai/api/v1")
KEY_FILE = Path(os.environ.get("LG_KEY_FILE", Path.home() / ".litellm-key")).expanduser()
CHAT_MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"

# USD per million tokens (gpt-6-luna: developers.openai.com pricing as recorded in agent-telephone/config/prices.yaml;
# text-embedding-3-small: list price).
# gpt-5.6-sol: conservative ESTIMATE pending the Azure price list (not list-verified).
PRICES = {CHAT_MODEL: (0.10, 0.50), EMBED_MODEL: (0.02, 0.0), "gpt-5.6-sol": (2.50, 20.0)}
# gpt-4o 2024-08-06 (Azure Global list price), matched by alias prefix since the gateway alias is not fixed here.
PREFIX_PRICES = {"gpt-4o": (2.50, 10.0)}


def price(model: str) -> tuple[float, float]:
    model = model.split("/")[-1]  # OpenRouter ids carry a provider prefix (openai/gpt-4o-...)
    if model in PRICES:
        return PRICES[model]
    return next((v for k, v in PREFIX_PRICES.items() if model.startswith(k)), (0.0, 0.0))


class Gateway:
    def __init__(self, ledger: Path | None = None, timeout: float = 180.0, model: str = CHAT_MODEL):
        self.model = model
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {KEY_FILE.read_text().strip()}"},
            timeout=timeout,
        )
        self._ledger = ledger
        self._lock = threading.Lock()

    def _post(self, path: str, payload: dict, retries: int = 6) -> dict:
        for attempt in range(retries):
            try:
                r = self._client.post(path, json=payload)
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == retries - 1:
                    raise
            time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError(f"gateway {path} failed after {retries} attempts")

    def _log(self, model: str, usage: dict, tag: str) -> None:
        if not self._ledger:
            return
        rec = {"t": time.time(), "model": model, "tag": tag,
               "in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0) or 0}
        with self._lock, self._ledger.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def chat(self, messages: list[dict], model: str | None = None, max_tokens: int = 4000,
             json_mode: bool = False, tag: str = "", extra: dict | None = None) -> str:
        model = model or self.model
        payload = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, **(extra or {})}
        if json_mode and "response_format" not in payload:
            payload["response_format"] = {"type": "json_object"}
        d = self._post("/chat/completions", payload)
        self._log(model, d.get("usage", {}), tag)
        return d["choices"][0]["message"]["content"] or ""

    def embed(self, texts: list[str], model: str = EMBED_MODEL, tag: str = "") -> np.ndarray:
        d = self._post("/embeddings", {"model": model, "input": texts})
        self._log(model, d.get("usage", {}), tag)
        vecs = np.array([e["embedding"] for e in sorted(d["data"], key=lambda e: e["index"])], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def ledger_cost(ledger: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not ledger.exists():
        return out
    for line in ledger.read_text().splitlines():
        r = json.loads(line)
        pin, pout = price(r["model"])
        out[r["model"]] = out.get(r["model"], 0.0) + (r["in"] * pin + r["out"] * pout) / 1e6
    return out
