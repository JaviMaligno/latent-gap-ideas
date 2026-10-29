#!/usr/bin/env bash
# Judge every generator-grid run with the four subscription judges (sol on the gateway is left out: company spend).
# Per run: pitches (gateway luna, cached), then one queue per CLI quota in parallel. Resumable.
set -u
DOMAIN=${DOMAIN:-steam}
WORKERS=${WORKERS:-4}
RUNS=${RUNS:-"grid-gpt-6-luna grid-gpt-6-1-sol grid-claude-opus-5-5 grid-claude-sonnet-5-5 grid-claude-haiku-5-5"}
LOGS=~/Documents/experiment-data/latent-gap-ideas/runs/$DOMAIN
cd "$(dirname "$0")/.."

j() { uv run python scripts/judge_multi.py --domain "$DOMAIN" --run "$1" --workers "$WORKERS" "${@:2}" >> "$LOGS/$1/judge.log" 2>&1; }

for r in $RUNS; do
  echo "$(date '+%H:%M') prepare $r"
  j "$r" --prepare ${PITCH_CODEX:+--pitch-codex} || { echo "prepare failed: $r"; continue; }
  (j "$r" --judge opus; j "$r" --judge sonnet; j "$r" --judge haiku) &
  j "$r" --judge gpt61sol &
  wait
  echo "$(date '+%H:%M') judged  $r: $(wc -l "$LOGS/$r"/judged_*.jsonl | tail -1)"
done
