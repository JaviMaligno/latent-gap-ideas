"""Per-domain rendering and prompt wording. Everything else in the pipeline is domain-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from latent_gap import arxiv, patents, steam


@dataclass(frozen=True)
class Domain:
    name: str
    noun: str                      # "Steam game"
    json_spec: str                 # fields the generator must return
    render: Callable[[dict], str]  # generated JSON -> canonical text (same function as the corpus)
    show: Callable[[str, str], str]  # (name, corpus text) -> short display for prompts
    schema: dict                   # JSON schema of json_spec, for generators that take one (CLI harnesses)
    # Judge wording (scripts/judge.py). Defaults are Steam's, so Steam prompts and their cached judgements are unchanged.
    does: str = "what the player does"
    feasible: str = "a small studio could actually build it"
    appeal: str = "a meaningful audience would want to play it"
    # Extra pitch rule, empty for Steam (prompt unchanged). Papers and patents: real abstracts report measured results
    # and generated ones can only invent them, so results would both unblind the judge and reward fabrication.
    pitch_rule: str = ""


def _steam_render(d: dict) -> str:
    def j(v):
        return ",".join(v) if isinstance(v, list) else str(v or "")
    return steam.item_text(j(d.get("genres")), j(d.get("tags")), str(d.get("about", "")))


STEAM = Domain(
    name="steam",
    noun="Steam game",
    json_spec=('{"name": str, "genres": [str], "tags": [str], '
               '"about": str  // 120-220 words, the store page "About the game" text}'),
    render=_steam_render,
    show=lambda name, text: f"{name} — {text[:450]}",
    schema={"type": "object", "additionalProperties": False, "required": ["name", "genres", "tags", "about"],
            "properties": {"name": {"type": "string"}, "genres": {"type": "array", "items": {"type": "string"}},
                           "tags": {"type": "array", "items": {"type": "string"}}, "about": {"type": "string"}}},
)

ARXIV = Domain(
    name="arxiv",
    noun="arXiv computer-science paper",
    json_spec='{"title": str, "abstract": str  // 150-250 words, the arXiv abstract}',
    render=lambda d: arxiv.item_text(str(d.get("title", "")), str(d.get("abstract", ""))),
    show=lambda name, text: f"{name} — {text[:450]}",
    schema={"type": "object", "additionalProperties": False, "required": ["title", "abstract"],
            "properties": {"title": {"type": "string"}, "abstract": {"type": "string"}}},
    does="what the method or study does",
    feasible="a small research group could actually carry it out with current methods",
    pitch_rule=("Describe only the problem, the approach and what makes it distinct: leave out every result, number, "
                "benchmark, performance or evaluation claim, and any statement of what was shown or achieved. "),
    appeal="a meaningful research community would want to read and build on it",
)

PATENTS = Domain(
    name="patents",
    noun="US patent application",
    json_spec='{"title": str, "abstract": str  // 100-150 words, the patent application abstract}',
    render=lambda d: patents.item_text(str(d.get("title", "")), str(d.get("abstract", ""))),
    show=lambda name, text: f"{name} — {text[:450]}",
    schema={"type": "object", "additionalProperties": False, "required": ["title", "abstract"],
            "properties": {"title": {"type": "string"}, "abstract": {"type": "string"}}},
    does="what the invention does and how",
    feasible="a small engineering team could actually build and patent it with current technology",
    pitch_rule=("Describe only the problem, the approach and what makes it distinct: leave out every result, number, "
                "benchmark, performance or evaluation claim, and any statement of what was shown or achieved. "),
    appeal="a meaningful market or industry would want to use or license it",
)

DOMAINS = {"steam": STEAM, "arxiv": ARXIV, "patents": PATENTS}
