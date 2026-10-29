#!/usr/bin/env bash
# Pre-cutoff generator (gpt-4o 2024-08-06, training cutoff Oct 2023) for the anticipation test: same targets as the
# grid, arms REAL (control), FAR, GAP, PLAIN, PLAIND. Gateway alias in ALIAS. Company spend: logged in ledger.jsonl.
set -u
ALIAS=${ALIAS:?gateway alias of gpt-4o 2024-08-06}
N=${N:-50}
cd "$(dirname "$0")/.."
for d in ${DOMAINS:-steam arxiv}; do
  L=~/Documents/experiment-data/latent-gap-ideas/runs/$d
  echo "$(date '+%H:%M') start $d"
  uv run python scripts/run_pilot.py --domain "$d" --n "$N" --out grid-gpt-4o-0806 --targets-from pilot \
    --arms REAL,FAR,GAP,PLAIN --generator "gateway:$ALIAS" --workers 8 > "$L/grid-gpt-4o-0806.log" 2>&1
  uv run python scripts/run_pilot.py --domain "$d" --n "$N" --out grid-gpt-4o-0806 --targets-from pilot \
    --arms PLAIND --generator "gateway:$ALIAS" > "$L/grid-gpt-4o-0806.plaind.log" 2>&1
  echo "$(date '+%H:%M') done  $d: $(grep -c '^\[' "$L/grid-gpt-4o-0806.log") + $(grep -c '^\[' "$L/grid-gpt-4o-0806.plaind.log") items"
done
