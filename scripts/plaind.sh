#!/usr/bin/env bash
# PLAIND (PLAIN with diversity control) on the grid runs of one domain: sequential within a run, one queue per CLI
# quota. Resumable: finished k are skipped and the earlier ideas are reloaded from items.jsonl.
set -u
DOMAIN=${DOMAIN:-steam}
N=${N:-50}
LOGS=~/Documents/experiment-data/latent-gap-ideas/runs/$DOMAIN
cd "$(dirname "$0")/.."

queue() {
  for pair in "$@"; do
    out=${pair%%=*}; g=${pair#*=}
    echo "$(date '+%H:%M') start PLAIND $g -> $out"
    uv run python scripts/run_pilot.py --domain "$DOMAIN" --n "$N" --out "$out" --targets-from pilot --arms PLAIND \
      --generator "$g" > "$LOGS/$out.plaind.log" 2>&1
    echo "$(date '+%H:%M') done  $g: $(grep -c '^\[' "$LOGS/$out.plaind.log") items, $(grep -c FAILED "$LOGS/$out.plaind.log") failed"
  done
}

CODEX_Q=${CODEX_Q:-"grid-gpt-6-luna=codex:gpt-6-luna grid-gpt-6-1-sol=codex:gpt-6.1-sol"}
CLAUDE_Q=${CLAUDE_Q:-"grid-claude-sonnet-5-5=claude:claude-sonnet-5-5 grid-claude-opus-5-5=claude:claude-opus-5-5 grid-claude-haiku-5-5=claude:claude-haiku-5-5"}
queue $CODEX_Q &
queue $CLAUDE_Q &
wait
