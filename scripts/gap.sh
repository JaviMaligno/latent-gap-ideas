#!/usr/bin/env bash
# Study 2 GAP arm on the pilot and grid runs (same targets in every run: built from the corpus with a fixed seed).
# Gateway and Codex queues run in parallel with the Claude queue (separate quotas). Resumable.
set -u
DOMAIN=${DOMAIN:-steam}
N=${N:-50}
WORKERS=${WORKERS:-4}
LOGS=~/Documents/experiment-data/latent-gap-ideas/runs/$DOMAIN
cd "$(dirname "$0")/.."

queue() {
  for pair in "$@"; do
    out=${pair%%=*}; g=${pair#*=}
    if [ "$g" = gateway ] && ! curl -s -m 10 -o /dev/null "${LITELLM_BASE_URL:-http://localhost:4000/v1}"/../health/liveliness; then
      echo "$(date '+%H:%M') skip  $out: gateway unreachable (VPN?); rerun later"; continue
    fi
    echo "$(date '+%H:%M') start GAP $g -> $out"
    uv run python scripts/run_pilot.py --domain "$DOMAIN" --n "$N" --out "$out" --arms GAP \
      --workers "$WORKERS" --generator "$g" > "$LOGS/$out.gap.log" 2>&1
    uv run python scripts/analyze.py --domain "$DOMAIN" --run "$out" > "$LOGS/$out.analyze.log" 2>&1
    echo "$(date '+%H:%M') done  $g: $(grep -c '^\[' "$LOGS/$out.gap.log") items, $(grep -c FAILED "$LOGS/$out.gap.log") failed"
  done
}

queue pilot=gateway grid-gpt-6-luna=codex:gpt-6-luna grid-gpt-6-1-sol=codex:gpt-6.1-sol &
queue grid-claude-sonnet-5-5=claude:claude-sonnet-5-5 grid-claude-opus-5-5=claude:claude-opus-5-5 \
  grid-claude-haiku-5-5=claude:claude-haiku-5-5 &
wait
