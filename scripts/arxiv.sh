#!/usr/bin/env bash
# arXiv domain, all arms on the arXiv pilot targets, one queue per CLI harness (separate quotas).
# Resumable.
set -u
N=${N:-50}
WORKERS=${WORKERS:-4}
ARMS=REAL,INTERP,FAR,NOISE,CROWD,GAP,PLAIN,PAIR
LOGS=~/Documents/experiment-data/latent-gap-ideas/runs/arxiv
cd "$(dirname "$0")/.."

run() {
  out=$1; g=$2
  echo "$(date '+%H:%M') start arxiv $g -> $out"
  uv run python scripts/run_pilot.py --domain arxiv --n "$N" --out "$out" --targets-from pilot --arms "$ARMS" \
    --workers "$WORKERS" --generator "$g" > "$LOGS/$out.log" 2>&1
  uv run python scripts/analyze.py --domain arxiv --run "$out" > "$LOGS/$out.analyze.log" 2>&1
  echo "$(date '+%H:%M') done  $g: $(grep -c '^\[' "$LOGS/$out.log") items, $(grep -c FAILED "$LOGS/$out.log") failed"
}

run grid-gpt-6-luna codex:gpt-6-luna &
run grid-claude-sonnet-5-5 claude:claude-sonnet-5-5 &
wait
