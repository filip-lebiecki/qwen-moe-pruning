#!/usr/bin/env bash
# Step 5: causal ablation of single experts and of domain teams, then merging instead of deleting.
# Needs the original model (scripts/01_download.sh) and data/domains.pt (02_calib_data.py --preset domains).
#   setsid nohup scripts/run_step5.sh > logs/step5.log 2>&1 < /dev/null &        (~1 h 45 min)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
ts() { echo "[$(date +%T)] $*"; }

ts "ablation: one expert at a time"
$PY scripts/16_ablate.py

ts "expert similarity"
$PY scripts/17_expert_similarity.py

ts "original, unpruned (reference; moe_forward is deterministic since step 5, numbers shift by ~0.1%)"
$PY scripts/05_eval.py --tag step5-original

for K in 128 48; do
    ts "K=$K: prune (reference), merge naive, merge aligned, merge aligned by router similarity"
    $PY scripts/05_eval.py --stats stats/calib.pt --keep $K --tag step5-prune-k$K
    $PY scripts/05_eval.py --stats stats/calib.pt --keep $K --merge naive
    $PY scripts/05_eval.py --stats stats/calib.pt --keep $K --merge aligned
    $PY scripts/05_eval.py --stats stats/calib.pt --keep $K --merge aligned --sim-kind router
done

ts "ablation: domain teams"
$PY scripts/16_ablate.py --no-single --teams 4 16

for K in 128 48; do
    ts "K=$K: merge aligned, keeping the full router (M-SMoE style)"
    $PY scripts/05_eval.py --stats stats/calib.pt --keep $K --merge aligned-remap
done

$PY scripts/ablation_report.py --js
ts "STEP5 DONE"
