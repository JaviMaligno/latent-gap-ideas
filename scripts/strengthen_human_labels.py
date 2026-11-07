"""Blind author check of the LLM continuation labels (pre-registered in docs/prereg/strengthen_human_labels.md).
    uv run python scripts/strengthen_human_labels.py sample     # draws the 40 pairs, writes blind and key files
    uv run python scripts/strengthen_human_labels.py analyse <labels.json>   # {"finished": true, "labels": [{hid, label, note}]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "docs/prereg/strengthen_continuations_full_labels.json"
WORK = Path.home() / "Documents/experiment-data/latent-gap-ideas/runs"
OUTD = WORK / "human-labels"
SEED, SIZES = 20261107, {"S-cons": 25, "S-dis": 8, "S-bg": 7}


def stratum(p):
    a, b = p["gpt61sol"], p["opus55"]
    if a == b == "CONTINUES":
        return "S-cons"
    if (a == "CONTINUES") != (b == "CONTINUES"):
        return "S-dis"
    if a == b == "BACKGROUND":
        return "S-bg"
    return None


def texts():
    out = {}
    for s in (1, 2, 3):
        for p in json.loads((WORK / f"strengthen-continuations/stage{s}/pairs.json").read_text()):
            out[(p["ctx_index"], p["followup_index"])] = p
    return out


def sample():
    census = json.loads(CENSUS.read_text())
    rng = np.random.default_rng(SEED)
    picked = []
    for s, n in SIZES.items():
        pool = [p for p in census if stratum(p) == s]
        picked += [(s, pool[i]) for i in rng.choice(len(pool), n, replace=False)]
        print(s, "pool", len(pool), "drawn", n)
    order = rng.permutation(len(picked))
    tx = texts()
    blind, key = [], []
    for k, o in enumerate(order):
        s, p = picked[o]
        t = tx[(p["ctx_index"], p["followup_index"])]
        hid = f"h{k + 1:02d}"
        blind.append({"hid": hid, "source_title": t["source_title"], "source_abstract": t["source_abstract"],
                      "followup_title": t["followup_title"], "followup_abstract": t["followup_abstract"]})
        key.append({"hid": hid, "stratum": s, "pair_id": p["id"], "ctx_index": p["ctx_index"],
                    "followup_index": p["followup_index"], "gpt61sol": p["gpt61sol"], "opus55": p["opus55"]})
    (OUTD / "blind_pairs.json").write_text(json.dumps(blind, indent=1, ensure_ascii=False))
    (OUTD / "key.json").write_text(json.dumps(key, indent=1))
    print("wrote", len(blind))


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def analyse(path):
    key = {k["hid"]: k for k in json.loads((OUTD / "key.json").read_text())}
    raw = json.loads(Path(path).read_text())
    # pre-registration: nothing is analysed before the author marks the set finished
    assert raw.get("finished") is True, "set not marked finished"
    rows_in = raw["labels"]
    hids = [r["hid"] for r in rows_in]
    assert len(hids) == len(set(hids)), "duplicate ids"
    hum = {r["hid"]: r for r in rows_in}
    assert set(hum) == set(key), "labels incomplete"
    allowed = {"CONTINUES", "USES", "BACKGROUND", "UNCLEAR"}
    assert all(r["label"] in allowed for r in rows_in), "invalid label"
    res = {"prereg": "docs/prereg/strengthen_human_labels.md", "n": len(hum)}
    rows = [key[h] | {"human": hum[h]["label"], "note": hum[h].get("note", "")} for h in sorted(key)]
    cons = [r for r in rows if r["stratum"] == "S-cons"]
    k = sum(r["human"] == "CONTINUES" for r in cons)
    k2 = sum(r["human"] in ("CONTINUES", "USES") for r in cons)
    res["P1_precision_consensus"] = {"k": k, "n": len(cons), "share": k / len(cons), "wilson95": wilson(k, len(cons)),
                                     "continues_or_uses": {"k": k2, "share": k2 / len(cons),
                                                           "wilson95": wilson(k2, len(cons))}}
    dis = [r for r in rows if r["stratum"] == "S-dis"]
    res["S_dis"] = {w: sum((r[w] == "CONTINUES") == (r["human"] == "CONTINUES") for r in dis)
                    for w in ("gpt61sol", "opus55")} | {"n": len(dis)}
    bg = [r for r in rows if r["stratum"] == "S-bg"]
    res["S_bg_human_continues"] = {"k": sum(r["human"] == "CONTINUES" for r in bg), "n": len(bg)}
    h = np.array([r["human"] == "CONTINUES" for r in rows])
    res["agreement_stratified_sample"] = {w: {"raw_4label": float(np.mean([r[w] == r["human"] for r in rows])),
                                              "kappa_continues": kappa(np.array([r[w] == "CONTINUES" for r in rows]), h)}
                                          for w in ("gpt61sol", "opus55")}
    res["human_label_counts"] = {l: sum(r["human"] == l for r in rows) for l in ("CONTINUES", "USES", "BACKGROUND", "UNCLEAR")}
    res["rows"] = rows
    (ROOT / "docs/prereg/strengthen_human_labels.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "rows"}, indent=1))


if __name__ == "__main__":
    sample() if sys.argv[1] == "sample" else analyse(sys.argv[2])
