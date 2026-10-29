"""Shared pieces of the coherent no-neighbour search (scripts/coherent_search.py) and its judge calibration
(scripts/coherent_calibration.py): the non-LLM gibberish flag, the length bounds and the blind coherence judge.
Pre-registration: docs/prereg/coherent_search.md."""
from __future__ import annotations

import re
import subprocess
import unicodedata
from functools import lru_cache
from pathlib import Path

from latent_gap import arxiv, steam
from latent_gap.cli_judges import RateLimitedError, RefusedError, judge_call

# Gibberish flag (calibration in docs/prereg/coherent_search.md). Flagged if any letter that is not Latin, Greek or a
# mathematical alphanumeric (Greek and maths symbols are legitimate in arXiv abstracts), a non-ASCII share > 0.02, a
# dictionary-word rate (alphabetic tokens of length >= 3, web2 list with suffix stripping) < 0.70, or a worst window
# of 12 consecutive such tokens with a dictionary rate < 0.40.
NONASCII_MAX, DICT_MIN, WINDOW, WINDOW_MIN = 0.02, 0.70, 12, 0.40
ALLOWED_SCRIPTS = ("LATIN", "GREEK", "MATHEMATICAL")

# Validity bounds: the corpus 99th percentile of each field (5000 visible items per domain, 2026-10-26) and the
# encoder's 512-token window. The body (about / abstract) is measured as rendered, i.e. cut at 1200 characters exactly
# like the corpus texts. A candidate outside them is invalid (status "<field>>cap"), never a witness.
BOUNDS = {"steam": {"genres": 7, "tags": 20, "about": 225}, "arxiv": {"title": 18, "abstract": 197}}
BODY_CHARS = {"about": steam.ABOUT_CHARS, "abstract": arxiv.ABSTRACT_CHARS}
MAX_TOKENS = 511

JUDGE = ("Below is a {noun} description, exactly as written. Rate its COHERENCE from 1 to 10: does it read as a "
         "real, sensible {noun} description, not word salad? 10 = fluent, self-consistent description a real "
         "{noun} could have; 7 = sensible overall with minor oddities; 4 = partly intelligible, with stuffed keywords "
         "or contradictions; 1 = word salad or not a {noun} description at all. Do not judge novelty, quality or "
         "factual accuracy.\n\n<description>\n{text}\n</description>\n\n"
         "Answer with JSON only: {{\"coherence\": int, \"reason\": str  // one sentence}}")
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["coherence", "reason"],
                "properties": {"coherence": {"type": "integer"}, "reason": {"type": "string"}}}


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    return frozenset(w.strip().lower() for w in Path("/usr/share/dict/words").read_text().splitlines())


def _in_dict(w: str, words: frozenset[str]) -> bool:
    if w in words:
        return True
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                     ("ly", ""), ("er", ""), ("ers", ""), ("ation", "e"), ("al", ""), ("ness", ""), ("ment", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and w[: len(w) - len(suf)] + rep in words:
            return True
    return False


def gibberish_stats(text: str) -> dict:
    non_ascii = sum(ord(c) > 127 for c in text) / max(1, len(text))
    other = sum(c.isalpha() and ord(c) > 127 and not any(s in unicodedata.name(c, "") for s in ALLOWED_SCRIPTS)
                for c in text)
    words = _words()
    hits = [_in_dict(w.lower(), words) for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 3] or [True]
    worst = min(sum(hits[i:i + WINDOW]) / len(hits[i:i + WINDOW]) for i in range(max(1, len(hits) - WINDOW + 1)))
    return {"non_ascii": round(non_ascii, 4), "non_latin": other, "dict_rate": round(sum(hits) / len(hits), 4),
            "dict_window_min": round(worst, 4)}


def gibberish_rule(stats: dict) -> str:
    """The first rule that fires ('' = not gibberish)."""
    if stats["non_latin"] > 0:
        return "script"
    if stats["non_ascii"] > NONASCII_MAX:
        return "non_ascii"
    if stats["dict_rate"] < DICT_MIN:
        return "dict_rate"
    if stats["dict_window_min"] < WINDOW_MIN:
        return "dict_window"
    return ""


def bounds_violation(domain: str, item: dict, n_tokens: int) -> str:
    """'' if the item is inside the corpus p99 bounds and the encoder window, else the field that is not."""
    for field, cap in BOUNDS[domain].items():
        v = item[field]
        n = len(v) if isinstance(v, list) else len(v[: BODY_CHARS.get(field)].split())
        if n > cap:
            return f"{field}>{cap}"
    return f"tokens>{MAX_TOKENS}" if n_tokens > MAX_TOKENS else ""


def judge(spec: str, noun: str, text: str) -> dict:
    """One blind coherence rating. Refusals and per-call failures become coherence None with the reason recorded;
    usage limits that outlast the CLI wrapper's waits propagate (the target is then left undone)."""
    cli, model = spec.split(":", 1)
    try:
        return judge_call(cli, model, JUDGE.format(noun=noun, text=text), JUDGE_SCHEMA)
    except RateLimitedError:
        raise
    except RefusedError:
        return {"coherence": None, "reason": "judge refused"}
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
        return {"coherence": None, "reason": f"judge error: {type(e).__name__}: {str(e)[:200]}"}
