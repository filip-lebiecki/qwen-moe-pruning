#!/usr/bin/env bash
# Pass 1 + a cheap comparison sweep (nothing written to disk except stats/results):
#   - expert statistics on the calibration set (skipped if stats/calib.pt exists)
#   - baseline perplexity of the unpruned model
#   - virtual pruning with REAP at each K given, plus the weaker criteria
#     (freq, random) at the first K
#   scripts/run_sweep.sh 128 96          (default: 128 64 48)
# Run detached so it survives closing the terminal:
#   setsid nohup scripts/run_sweep.sh 128 96 > logs/sweep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
KS=("$@")
[ ${#KS[@]} -eq 0 ] && KS=(128 64 48)

[ -f stats/calib.pt ] || $PY scripts/03_expert_stats.py
$PY scripts/05_eval.py --tag original
for K in "${KS[@]}"; do
    $PY scripts/05_eval.py --stats stats/calib.pt --keep "$K" --criterion reap
done
for C in freq random; do
    $PY scripts/05_eval.py --stats stats/calib.pt --keep "${KS[0]}" --criterion $C
done
echo SWEEP DONE
