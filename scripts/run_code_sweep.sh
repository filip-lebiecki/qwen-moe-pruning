#!/usr/bin/env bash
# Code-specialist experiment: does calibrating on code only keep a ~9B model useful for code?
#   - expert statistics on the code calibration set (stats/code.pt)
#   - code perplexity (data/eval_code.pt) of the original, and of virtual prunes at each K
#     with experts chosen from code statistics vs. from the general statistics
#   - general perplexity (data/eval.pt) of the code-calibrated prunes: what they give up
# Results: results/eval_code.jsonl (code eval) and results/eval.jsonl (general eval).
#   setsid nohup scripts/run_code_sweep.sh 64 48 32 > logs/code_sweep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
KS=("$@")
[ ${#KS[@]} -eq 0 ] && KS=(64 48 32)
CODE="--data data/eval_code.pt --results results/eval_code.jsonl"

[ -f stats/code.pt ] || $PY scripts/03_expert_stats.py --data data/calib_code.pt --out stats/code.pt
$PY scripts/05_eval.py $CODE --tag original
for K in "${KS[@]}"; do
    $PY scripts/05_eval.py $CODE --stats stats/code.pt --keep "$K" --tag "code-calib-keep$K"
    $PY scripts/05_eval.py $CODE --stats stats/calib.pt --keep "$K" --tag "general-calib-keep$K"
    $PY scripts/05_eval.py --stats stats/code.pt --keep "$K" --tag "code-calib-keep$K"
done
echo CODE SWEEP DONE
