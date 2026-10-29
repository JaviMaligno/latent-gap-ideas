#!/usr/bin/env bash
# Generator grid on the pilot's targets: one queue per CLI harness (separate subscription quotas), run in parallel.
# Each run: runs/<domain>/grid-<model>/ (items.jsonl, metrics.csv, summary.csv), log next to it. Resumable.
set -u
DOMAIN=${DOMAIN:-steam}
N=${N:-50}
WORKERS=${WORKERS:-4}
LOGS=~/Documents/experiment-data/latent-gap-ideas/runs/$DOMAIN
cd "$(dirname "$0")/.."

queue() {
  for g in "$@"; do
    out=grid-${g#*:}
    out=${out//./-}
    echo "$(date '+%H:%M') start $g -> $out"
    uv run python scripts/run_pilot.py --domain "$DOMAIN" --n "$N" --out "$out" --targets-from pilot \
      --workers "$WORKERS" --generator "$g" > "$LOGS/$out.log" 2>&1
    uv run python scripts/analyze.py --domain "$DOMAIN" --run "$out" > "$LOGS/$out.analyze.log" 2>&1
    echo "$(date '+%H:%M') done  $g: $(grep -c '^\[' "$LOGS/$out.log") items, $(grep -c FAILED "$LOGS/$out.log") failed"
  done
}

queue codex:gpt-6-luna codex:gpt-6.1-sol &
queue claude:claude-sonnet-5-5 claude:claude-opus-5-5 claude:claude-haiku-5-5 &
wait
