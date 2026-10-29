#!/usr/bin/env bash
# MEM (memory probe) on both domains for one generator. GEN=codex:gpt-6-luna OUT=grid-gpt-6-luna scripts/mem.sh
set -u
cd "$(dirname "$0")/.."
for d in ${DOMAINS:-arxiv steam}; do
  L=~/Documents/experiment-data/latent-gap-ideas/runs/$d
  uv run python scripts/run_pilot.py --domain "$d" --out "$OUT" --targets-from "${TARGETS_FROM:-pilot}" --arms MEM \
    --generator "$GEN" --workers "${WORKERS:-4}" > "$L/$OUT.mem.log" 2>&1
  echo "$(date '+%H:%M') done MEM $GEN $d: $(grep -c '^\[' "$L/$OUT.mem.log") items, $(grep -c FAILED "$L/$OUT.mem.log") failed"
done
