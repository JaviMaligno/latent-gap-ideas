"""Steam corpus preparation (Kaggle fronkongames/steam-games-dataset, games.csv)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2024-01-01")
ABOUT_CHARS = 1200
MIN_ABOUT_CHARS = 200
# Items whose prompts Azure's content filter rejects; excluded from the space (kept in the files, flagged).
ADULT = re.compile(r"Sexual Content|Nudity|NSFW|Hentai|Mature|Adult|Erotic", re.I)
# ADULT misses explicit games whose tags are clean (~3% of the past corpus match this on name + text). Only a flag,
# applied per item for a sensitivity check; such items stay in the analysis.
EXPLICIT = re.compile(r"\bsex(?:ual|y)?\b|\bxxx\b|hentai|porn|\bnud(?:e|ity)\b|erotic|nsfw|"
                      r"\badult (?:game|content|only|visual)", re.I)


def load_raw(path: Path) -> pd.DataFrame:
    # The CSV header merges "Discount" and "DLC count" into one name, shifting every later column by one.
    cols = list(pd.read_csv(path, nrows=0).columns)
    i = cols.index("DiscountDLC count")
    cols = cols[:i] + ["Discount", "DLC count"] + cols[i + 1:]
    return pd.read_csv(path, header=None, skiprows=1, names=cols, low_memory=False)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _ascii_ratio(s: str) -> float:
    return sum(c.isascii() for c in s) / max(len(s), 1)


def item_text(genres: str, tags: str, about: str) -> str:
    """Canonical text that gets embedded. Generated items are rendered with the same function."""
    return f"Genres: {genres}. Tags: {tags}. About: {about[:ABOUT_CHARS]}"


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "id": raw["AppID"].astype(str),
        "name": raw["Name"].fillna("").map(_clean),
        "date": pd.to_datetime(raw["Release date"], errors="coerce", format="mixed"),
        "genres": raw["Genres"].fillna("").map(_clean),
        "tags": raw["Tags"].fillna("").map(_clean),
        "about": raw["About the game"].fillna("").map(_clean),
        "positive": pd.to_numeric(raw["Positive"], errors="coerce").fillna(0).astype(int),
        "negative": pd.to_numeric(raw["Negative"], errors="coerce").fillna(0).astype(int),
        "owners": raw["Estimated owners"].fillna(""),
    })
    df = df[df["date"].notna() & (df["about"].str.len() >= MIN_ABOUT_CHARS)]
    df = df[df["about"].map(_ascii_ratio) >= 0.95]
    df = df.drop_duplicates("about")
    df["split"] = (df["date"] >= CUTOFF).map({False: "past", True: "future"})
    df["adult"] = (df["tags"] + "," + df["genres"]).str.contains(ADULT)
    df["text"] = [item_text(g, t, a) for g, t, a in zip(df["genres"], df["tags"], df["about"])]
    return df.reset_index(drop=True)
