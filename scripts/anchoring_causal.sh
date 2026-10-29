#!/usr/bin/env bash
# Round 5B. One queue per subscription, one decoder and one CLI child per queue.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4

if [[ "${1:-}" == "--dry-check" ]]; then
  exec uv run python scripts/anchoring_causal.py dry-check
fi
if [[ $# != 0 ]]; then
  echo "Usage: bash scripts/anchoring_causal.sh [--dry-check]" >&2
  exit 2
fi
uv run python scripts/anchoring_causal.py check-plan
command -v claude >/dev/null
command -v codex >/dev/null
ROOT="$HOME/Documents/experiment-data/latent-gap-ideas/runs"
LOCK="$ROOT/anchoring-causal.lock"
if ! mkdir "$LOCK"; then
  echo "Another runner holds $LOCK. If its PID is no longer alive, remove that stale lock directory before resuming." >&2
  exit 1
fi
echo "$$" > "$LOCK/pid"
PIDS=""
cleanup() {
  for pid in $PIDS; do kill "$pid" 2>/dev/null || true; done
  for pid in $PIDS; do wait "$pid" 2>/dev/null || true; done
  rm -f "$LOCK/pid"
  rmdir "$LOCK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

queue() {
  local gen="$1" spec="$2" offset="$3" domain arm cond order out log child=""
  local di=0 ai block_seed
  trap 'if [[ -n "$child" ]]; then kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 130' INT TERM
  for domain in steam arxiv; do
    ai=0
    for arm in REAL GAP FAR; do
      block_seed=$((20261028 + 100 * di + ai))
      # Latin order, balanced across the six domain/arm blocks; shifted between CLIs.
      case $(((di * 3 + ai + offset) % 3)) in
        0) order="true swap none" ;;
        1) order="swap none true" ;;
        2) order="none true swap" ;;
      esac
      for cond in $order; do
        out="anchoring-causal/$gen/$cond/$arm"
        log="$ROOT/$domain/$out/run.log"
        mkdir -p "$(dirname "$log")"
        echo "$(date '+%F %T') $domain $gen $arm $cond (resume)"
        uv run python scripts/run_pilot.py --domain "$domain" --arms "$arm" --n 20 --start-k 1 \
          --targets-from pilot --out "$out" --generator "$spec" --feedback score \
          --neighbours "$cond" --neighbour-seed "$block_seed" --workers 1 --candidate-workers 1 >> "$log" 2>&1 &
        child=$!
        if ! wait "$child"; then
          echo "Failed: $log; rerun this script to resume missing targets." >&2
          return 1
        fi
        child=""
      done
      ai=$((ai + 1))
    done
    di=$((di + 1))
  done
}
queue luna codex:gpt-6-luna 0 &
PIDS="$!"
queue sonnet claude:claude-sonnet-5-5 1 &
PIDS="$PIDS $!"
status=0
for pid in $PIDS; do wait "$pid" || status=1; done
PIDS=""
if [[ "$status" != 0 ]]; then exit "$status"; fi
uv run python scripts/anchoring_causal.py analyze
