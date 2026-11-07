"""Stage 1 of docs/prereg/strengthen_olmo_direction.md: contexts and follow-ups, free APIs, no keys.

Frame (written once, frame.json): the 196 unused contexts of runs/followups/followups.jsonl (rows after the 150th
context with >= 5 follow-ups that have >= 5 follow-ups, file order), then the followups_fetch.py pool minus every
context id of any runs/*/followups.jsonl, in default_rng(20261106).permutation order.

Per context: one paginated Semantic Scholar citations call with externalIds,publicationDate,intents,isInfluential;
every citing arXiv paper of any date saved uncapped (citing_all.jsonl); candidates = S2 date >= 2024-01-01; arXiv API
for title, abstract and v1 date ('published'); follow-up = v1 >= 2024-01-01 with abstract; at most 50 (seeded).
Rows go to followups.jsonl in frame order; resumable; stops when --target contexts with >= 5 follow-ups exist.

  nohup uv run python scripts/olmo_direction_fetch.py --target 1200 > <run>/fetch.log 2>&1 < /dev/null &
  uv run python scripts/olmo_direction_fetch.py --status
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
RUN = DATA / "runs/olmo-direction"
SEED = 20261106
SINCE = "2024-01-01"
MAX_FUPS, MIN_FUPS = 50, 5
S2 = ("https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/citations"
      "?fields=externalIds,publicationDate,intents,isInfluential&limit=1000&offset={}")
ARXIV = "https://export.arxiv.org/api/query?id_list={}&max_results={}"
ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "latent-gap-ideas research fetch (olmo-direction)"}


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def get(url: str, delay: float) -> tuple[int, bytes]:
    """GET with backoff on 429 / 5xx / network errors that never gives up; 4xx other than 429 returned as is."""
    wait = 5.0
    for attempt in range(10_000):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                body = r.read()
            time.sleep(delay)
            return 200, body
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                time.sleep(delay)
                return e.code, b""
            ra = e.headers.get("Retry-After") if e.headers else None
            sl = float(ra) if ra and ra.isdigit() else wait
            if attempt % 5 == 0 or sl >= 120:
                log(f"HTTP {e.code}, retry {attempt + 1} in {sl:.0f}s: {url[:110]}")
        except Exception as e:  # noqa: BLE001 (timeouts, resets)
            sl = wait
            if attempt % 5 == 0:
                log(f"{type(e).__name__}, retry {attempt + 1} in {sl:.0f}s: {url[:110]}")
        time.sleep(sl)
        wait = min(300.0, wait * 2)
    raise RuntimeError(url)


def citations(aid: str) -> dict:
    out, off, status = [], 0, "ok"
    while True:
        code, body = get(S2.format(aid, off), 1.1)
        if code == 404:
            status = "s2_missing" if off == 0 else status
            break
        if code != 200:
            status = f"s2_http_{code}_at_offset_{off}"
            break
        d = json.loads(body)
        for x in d.get("data", []):
            p = x.get("citingPaper") or {}
            out.append({"arxiv_id": (p.get("externalIds") or {}).get("ArXiv"), "s2_date": p.get("publicationDate"),
                        "intents": x.get("intents"), "influential": x.get("isInfluential")})
        if "next" not in d:
            break
        off = d["next"]
    return {"status": status, "citing": out}


def arxiv_meta(ids: list[str]) -> dict:
    res = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        for _ in range(5):
            code, body = get(ARXIV.format(",".join(chunk), len(chunk)), 3.1)
            try:
                root = ET.fromstring(body)
                break
            except ET.ParseError:
                log(f"arXiv parse error (HTTP {code}), retrying chunk of {len(chunk)}")
                time.sleep(30)
        else:
            raise RuntimeError(f"arXiv chunk failed: {chunk[:3]}")
        for e in root.findall(f"{ATOM}entry"):
            aid = re.sub(r"v\d+$", "", e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1])
            summ = " ".join(e.findtext(f"{ATOM}summary", "").split())
            if aid and summ:
                res[aid] = {"title": " ".join(e.findtext(f"{ATOM}title", "").split()), "abstract": summ,
                            "v1": e.findtext(f"{ATOM}published", "")[:10]}
    return res


def frame() -> list[dict]:
    path = RUN / "frame.json"
    if path.exists():
        return json.loads(path.read_text())
    import pandas as pd
    rows = [json.loads(l) for l in (DATA / "runs/followups/followups.jsonl").read_text().splitlines()]
    ge5 = [k for k, r in enumerate(rows) if r["n_followups"] >= MIN_FUPS]
    last_used = ge5[149]
    first = [rows[k] for k in ge5[150:]]
    assert len(first) == 196 and all(k > last_used for k in ge5[150:])
    used = set()
    for f in sorted((DATA / "runs").glob("*/followups.jsonl")):
        used |= {json.loads(l)["arxiv_id"] for l in f.read_text().splitlines()}
    meta = pd.read_parquet(DATA / "prepared/arxiv/corpus.parquet", columns=["id", "name", "date", "split", "adult"])
    pool = meta[~meta.adult & (meta.split == "past") & (meta.date >= "2021-01-01")
                & (meta.date <= "2022-12-31")].reset_index(drop=True)
    pool = pool[~pool.id.isin(used)].reset_index(drop=True)
    order = np.random.default_rng(SEED).permutation(len(pool))
    fr = [{"arxiv_id": r["arxiv_id"], "title": r["title"], "date": r["date"], "source": "followups_rows_416_999"}
          for r in first]
    fr += [{"arxiv_id": pool.id[k], "title": pool.name[k], "date": str(pool.date[k]), "source": "new_draw"}
           for k in order]
    RUN.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fr))
    log(f"frame written: {len(fr)} contexts ({len(first)} earlier + {len(fr) - len(first)} new draws; "
        f"{len(used)} ids excluded as used)")
    return fr


def status() -> dict:
    p = RUN / "followups.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    n = np.array([r["n_followups"] for r in rows]) if rows else np.zeros(0)
    fl = [f for r in rows if r["n_followups"] >= MIN_FUPS for f in r["followups"]]
    miss = sum(1 for f in fl if f["intents"] is None and f["influential"] is None)
    flagged = sum(1 for f in fl if f["influential"] or {"methodology", "result"} & set(f["intents"] or []))
    return {"drawn": len(rows), "contexts_ge5": int((n >= MIN_FUPS).sum()),
            "share_ge5": float((n >= MIN_FUPS).mean()) if rows else None,
            "saved_followups_ge5": len(fl), "mean_followups_ge5": len(fl) / max(1, int((n >= MIN_FUPS).sum())),
            "flagged_share": flagged / max(1, len(fl)), "intents_missing_share": miss / max(1, len(fl)),
            "s2_status": {s: sum(r["s2_status"] == s for r in rows) for s in {r["s2_status"] for r in rows}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1200, help="stop at this many contexts with >= 5 follow-ups")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-draws", type=int, default=None, help="stop after this many frame positions (smoke tests)")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(status(), indent=1))
        return
    fr = frame()
    out, cit = RUN / "followups.jsonl", RUN / "citing_all.jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    if cit.exists():  # a crash between the two writes leaves one extra citing row: drop it
        cl = cit.read_text().splitlines()
        if len(cl) > len(rows):
            cit.write_text("".join(l + "\n" for l in cl[:len(rows)]))
    for k, r in enumerate(rows):
        assert r["frame_pos"] == k and r["arxiv_id"] == fr[k]["arxiv_id"], "followups.jsonl out of frame order"
    n5 = sum(r["n_followups"] >= MIN_FUPS for r in rows)
    log(f"resume at frame position {len(rows)} ({n5} contexts with >= {MIN_FUPS} follow-ups; target {a.target})")
    t0, start = time.time(), len(rows)
    with out.open("a") as fo, cit.open("a") as fc:
        for k in range(len(rows), len(fr)):
            if n5 >= a.target or (a.max_draws is not None and k - start >= a.max_draws):
                break
            c = fr[k]
            s2 = citations(c["arxiv_id"])
            ax = [x for x in s2["citing"] if x["arxiv_id"]]
            info = {}
            for x in ax:  # one row per citing paper; duplicates (rare) keep the first
                info.setdefault(x["arxiv_id"], x)
            cand = sorted(i for i, x in info.items() if (x["s2_date"] or "") >= SINCE)
            meta = arxiv_meta(cand) if cand else {}
            elig = [i for i in cand if i in meta and meta[i]["v1"] >= SINCE]
            uncapped = len(elig)
            if len(elig) > MAX_FUPS:
                sub = np.random.default_rng(SEED + k).choice(len(elig), MAX_FUPS, replace=False)
                elig = [elig[j] for j in sorted(sub)]
            fups = [{"arxiv_id": i, "date": info[i]["s2_date"], "v1": meta[i]["v1"], "title": meta[i]["title"],
                     "abstract": meta[i]["abstract"], "intents": info[i]["intents"],
                     "influential": info[i]["influential"]} for i in elig]
            fc.write(json.dumps({"frame_pos": k, "arxiv_id": c["arxiv_id"], "citing_arxiv_all": sorted(info),
                                 "n_citing_all": len(s2["citing"])}) + "\n")
            fc.flush()
            fo.write(json.dumps({"frame_pos": k, "arxiv_id": c["arxiv_id"], "title": c["title"], "date": c["date"],
                                 "source": c["source"], "s2_status": s2["status"], "n_citing_all": len(s2["citing"]),
                                 "n_citing_arxiv_all": len(info), "n_candidates": len(cand),
                                 "n_v1_dropped": sum(1 for i in cand if i in meta and meta[i]["v1"] < SINCE),
                                 "n_no_abstract": sum(1 for i in cand if i not in meta),
                                 "n_followups_uncapped": uncapped, "n_followups": len(fups), "followups": fups}) + "\n")
            fo.flush()
            n5 += len(fups) >= MIN_FUPS
            done = k + 1 - start
            if done % 25 == 0:
                rate = done / (time.time() - t0)
                left = (a.target - n5) / max(1e-9, n5 / (k + 1)) / rate / 3600 if n5 else float("nan")
                log(f"frame {k + 1}: {n5}/{a.target} contexts with >= {MIN_FUPS} ({n5 / (k + 1):.1%} of drawn), "
                    f"{rate * 3600:.0f} contexts/h, ~{left:.1f} h left")
    log("stop: " + json.dumps(status()))


if __name__ == "__main__":
    main()
