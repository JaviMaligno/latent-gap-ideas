"""Build the inspiration pool for the "inspiration from anything" experiment.

Nine sources, all dated before gpt-4o's October 2023 cutoff, sampled with seed 20261102, N per source, written as
"Title: ... Text: ..." with the text cut at TEXT_CHARS:
  steam_game (past Steam games of our corpus), board_game (BoardGameGeek, 2022 dump), movie (Wikipedia plots, 2018),
  paper (past arXiv cs of our corpus), patent (past patents of our corpus), recipe (Food.com, 2019),
  sports_rules (Wikipedia 2022-03-01 articles describing a sport or its rules, by a lead-sentence pattern),
  news (AG News, 2004-05), wikipedia (other Wikipedia 2022-03-01 articles, at random).
Raw data: experiment-data/latent-gap-ideas/raw/inspiration (pickle files there are never loaded).
Output: experiment-data/latent-gap-ideas/prepared/inspiration/pool.parquet (source, id, title, text)
"""
from __future__ import annotations

import gc
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RAW = DATA / "raw/inspiration"
OUT = DATA / "prepared/inspiration/pool.parquet"
SEED, N, TEXT_CHARS = 20261102, 6000, 700
# sports_rules: the lead defines a sport or game, is not about a person, video game, company or media work, and the body
# uses the vocabulary of rules (pattern checked by reading samples, 2026-11-02)
DEFN = r"\bis an? (?:\w+[ -]){0,4}(?:sport|game|pastime|martial art)\b"
EXCL = (r"\(born|was born|footballer|video game|computer game|mobile game|esports?|arcade|game show|television|film|novel|"
        r"album|song|developed by|published by|PlayStation|Nintendo|Xbox|Windows|Sega|MS-DOS|Amiga|game company|"
        r"corporation|website|software")
RULE_WORDS = ("rule", "player", "point", "turn", "score", "wins", "opponent", "ball", "card", "board", "move")


def clean(s) -> str:
    s = html.unescape(str(s).replace("#39;", "'").replace("#36;", "$"))
    return " ".join(re.sub(r"<[^>]+>", " ", s).split())


def sample(df: pd.DataFrame, rng: np.random.Generator, n: int = N) -> pd.DataFrame:
    return df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)] if len(df) else df


def main() -> None:
    rng = np.random.default_rng(SEED)
    parts = []

    def add(source, ids, titles, texts):
        parts.append(pd.DataFrame({"source": source, "id": [str(x) for x in ids],
                                   "title": [clean(t) for t in titles], "text": [clean(x)[:TEXT_CHARS] for x in texts]}))

    for dom, src in (("steam", "steam_game"), ("arxiv", "paper"), ("patents", "patent")):
        c = pd.read_parquet(DATA / f"prepared/{dom}/corpus.parquet")
        c = sample(c[(c["split"] == "past") & ~c["adult"]], rng)
        body = c["abstract"] if "abstract" in c else c["text"]
        add(src, c["id"], c["name"], body)
    # raw descriptions (the threnjen dump's are stop-word-stripped); games published up to 2022
    b = pd.read_csv(RAW / "bgg2/games_detailed_info.csv", usecols=["id", "primary", "description", "yearpublished"])
    b = sample(b[b["description"].notna() & (b["yearpublished"] <= 2022)], rng)
    add("board_game", b["id"], b["primary"], b["description"])
    m = pd.read_csv(RAW / "movieplots/wiki_movie_plots_deduped.csv", usecols=["Title", "Plot", "Release Year"])
    m = sample(m[m["Plot"].notna()], rng)
    add("movie", m.index, m["Title"], m["Plot"])
    r = pd.read_csv(RAW / "foodcom/RAW_recipes.csv", usecols=["id", "name", "description", "steps", "ingredients"])
    r = sample(r[r["name"].notna()], rng)
    add("recipe", r["id"], r["name"],
        [f"{clean(d) if isinstance(d, str) else ''} Ingredients: {i}. Steps: {s}" for d, i, s in zip(r["description"], r["ingredients"], r["steps"])])
    a = pd.read_parquet(RAW / "agnews/train.parquet")
    a = sample(a, rng)
    add("news", a.index, [t.split(" - ")[0][:80] for t in a["text"]], a["text"])
    # Wikipedia shard by shard (all ten at once exhausted the 16 GB machine): keep every sports_rules match and a
    # per-shard random draw of other articles, then sample from the union
    files = sorted((RAW / "wikipedia").glob("*.parquet"))
    sp, ot = [], []
    for f in files:
        w = pq.read_table(f, columns=["id", "title", "text"]).to_pandas()
        lead, body = w["text"].str[:400], w["text"].str[:6000].str.lower()
        sport = (lead.str.contains(DEFN, case=False) & ~lead.str.contains(EXCL, case=False)
                 & (sum(body.str.count(k) for k in RULE_WORDS) >= 12) & body.str.contains("rule"))
        sp.append(w[sport].copy())
        other = w[~sport & (w["text"].str.len() > 300)]
        ot.append(sample(other, rng, -(-2 * N // len(files))).copy())
        del w, lead, body, sport, other
        gc.collect()
    s = sample(pd.concat(sp, ignore_index=True), rng)  # one draw per source, so titles and texts stay aligned
    add("sports_rules", s["id"], s["title"], s["text"])
    o = sample(pd.concat(ot, ignore_index=True), rng)
    add("wikipedia", o["id"], o["title"], o["text"])
    pool = pd.concat(parts, ignore_index=True)
    pool = pool[pool["text"].str.len() > 80].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT)
    print(pool["source"].value_counts().to_dict(), "sports candidates", sum(len(x) for x in sp))


if __name__ == "__main__":
    main()
