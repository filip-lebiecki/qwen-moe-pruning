#!/usr/bin/env bash
# Running a MoE that is bigger than VRAM: keep some layers' experts in system RAM (step 6).
#
#   scripts/offload_sweep.sh MODEL.gguf TAG [N ...]       (N = layers whose experts stay on the CPU)
#
# For each N: start llama-server with all layers on the GPU except the expert tensors of the first N layers
# (--n-cpu-moe N), run speed_test.py, and record the GPU memory llama.cpp projected. Also one run with plain
# --fit on (llama.cpp chooses the split itself) and one with --cpu-moe (all experts on the CPU).
# --load-mode none (no mmap): the CPU part is read into RAM once instead of being paged from the file during generation
# (matters when the file is on a slow drive such as /mnt/f). Rows -> results/offload.jsonl (+ results/speed.jsonl).
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1; TAG=$2; shift 2
NS=("$@"); [ ${#NS[@]} -eq 0 ] && NS=(12 16 20 24 30)
BIN=llama.cpp/build/bin
mkdir -p logs/offload results
run() {  # $1 = label, rest = extra llama-server arguments
    local LAB=$1; shift
    local LOG=logs/offload/$TAG-$LAB.log
    $BIN/llama-server -m "$M" -c 8192 -np 1 --load-mode none --port 8093 -v "$@" > "$LOG" 2>&1 &
    local PID=$!
    for _ in $(seq 600); do grep -q "listening on\|error:\|failed to" "$LOG" && break; sleep 1; done
    if ! grep -q "listening on" "$LOG"; then echo "{\"tag\": \"$TAG\", \"setting\": \"$LAB\", \"error\": true}" | tee -a results/offload.jsonl; kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true; return; fi
    local USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local GPU=$(grep -oE "CUDA0 model buffer size += +[0-9.]+ MiB" "$LOG" | grep -oE "[0-9.]+ MiB" | cut -d' ' -f1 | sort -n | tail -1)
    local CPU=$(grep -oE "CPU model buffer size += +[0-9.]+ MiB" "$LOG" | grep -oE "[0-9.]+ MiB" | cut -d' ' -f1 | sort -n | tail -1)
    .venv/bin/python -I scripts/speed_test.py --port 8093 --tag "$TAG-$LAB" > /dev/null
    local SP=$(tail -1 results/speed.jsonl)
    kill $PID; wait $PID 2>/dev/null || true
    echo "{\"tag\": \"$TAG\", \"setting\": \"$LAB\", \"gpu_weights_mib\": ${GPU:-null}, \"cpu_weights_mib\": ${CPU:-null}, \"gpu_used_mib\": $USED, \"speed\": $SP}" | tee -a results/offload.jsonl
    sleep 3
}
run fit --fit on
for N in "${NS[@]}"; do run "ncmoe$N" -ngl 99 --n-cpu-moe "$N" --fit off; done
run cpumoe -ngl 99 --cpu-moe --fit off
