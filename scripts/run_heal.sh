#!/usr/bin/env bash
# Evaluate the pruned checkpoint, heal it (layer-wise distillation), evaluate again.
#   setsid nohup scripts/run_heal.sh models/q35-reap-k128 router moe > logs/heal.log 2>&1 &
# It modifies the checkpoint in place; re-run 04_prune.py to undo.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
M=${1:?checkpoint}; TRAIN=${2:-router}; TARGET=${3:-moe}
$PY scripts/05_eval.py --model "$M" --tag "$(basename "$M")-pruned"
$PY scripts/06_heal.py --student "$M" --train "$TRAIN" --target "$TARGET"
$PY scripts/05_eval.py --model "$M" --tag "$(basename "$M")-healed-$TRAIN-$TARGET"
echo HEAL DONE
