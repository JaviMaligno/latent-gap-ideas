"""Where the owner's novelty departs from the judges': per item, owner minus judges' mean, with the owner's note.

Reuses human_criteria.load; only batches with items.json (judges on the same items). Prints per-arm mean gap, the gap
split by the owner's "known" answer (patents P1 only asks it), and the largest disagreements with their pitches.

    uv run python scripts/human_novelty_gap.py
"""
import json
import sys

import pandas as pd

sys.path.insert(0, "scripts")
from human_criteria import BATCHES, DATA, load  # noqa: E402

rows = []
for d, b in BATCHES:
    h = load(d, b)
    if h.empty or "j_novelty" not in h:
        continue
    items = {f"{i['arm']}-{i['k']}": i for i in json.loads((DATA / d / b / "items.json").read_text())}
    h["pitch"] = h.key.map(lambda k: items.get(k, {}).get("pitch", ""))
    rows.append(h)
H = pd.concat(rows, ignore_index=True)
H["gap"] = H.novelty - H.j_novelty
print("Spearman owner~judges novelty per batch:")
print(H.groupby(["domain", "batch"]).apply(lambda g: g.novelty.corr(g.j_novelty, "spearman"), include_groups=False).round(2))
print("\nMean gap (owner - judges) by domain and arm:")
print(H.pivot_table("gap", "arm", "domain", aggfunc="mean").round(2))
if "known" in H:
    print("\nGap by 'known' answer:")
    print(H.dropna(subset=["known"]).groupby("known").agg(n=("gap", "size"), gap=("gap", "mean"),
                                                           owner=("novelty", "mean"), judges=("j_novelty", "mean")).round(2))
print("\nLargest disagreements:")
for _, r in H.reindex(H.gap.abs().sort_values(ascending=False).index).head(12).iterrows():
    note = " | ".join(x for x in (str(r.get("known_ref") or ""), str(r.get("note") or "")) if x and x != "nan")
    print(f"\n[{r.domain}/{r.batch} {r.key}] owner {r.novelty:.0f} vs judges {r.j_novelty:.1f} (gap {r.gap:+.1f})"
          f" coh {r.coherence:.0f}/{r.j_coherence:.1f}{'  note: ' + note if note else ''}")
    print("   " + r.pitch[:300].replace("\n", " "))
