"""Patents corpus (HUPD, Harvard USPTO patent applications). The corpus itself is prepared inside the embedding kernel
(kaggle/embed-patents/embed.py, which holds a copy of `item_text`); this module renders generated items identically."""
from __future__ import annotations

ABSTRACT_CHARS = 1200


def item_text(title: str, abstract: str) -> str:
    """Canonical text that gets embedded. Generated items are rendered with the same function."""
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"
