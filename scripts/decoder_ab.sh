#!/bin/bash
# Decoder comparison on identical targets (REAL, NOISE, CROWD; 5 each); same 12 LLM calls per target.
set -e
cd "$(dirname "$0")/.."
uv run python scripts/run_pilot.py --domain steam --n 5 --out dec-residual --feedback residual --arms REAL,NOISE,CROWD --workers 6
uv run python scripts/run_pilot.py --domain steam --n 5 --out dec-score --feedback score --arms REAL,NOISE,CROWD --workers 6 --targets-from dec-residual
uv run python scripts/run_pilot.py --domain steam --n 5 --out dec-none --feedback none --arms REAL,NOISE,CROWD --workers 6 --targets-from dec-residual
for r in dec-residual dec-score dec-none; do echo "== $r"; uv run python scripts/analyze.py --domain steam --run $r | head -5; done
