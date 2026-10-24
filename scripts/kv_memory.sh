#!/usr/bin/env bash
# How much memory does context cost? Start llama-server at several context lengths (and KV-cache types),
# read the sizes llama.cpp reports for the KV cache (attention layers) and the recurrent state (Gated
# DeltaNet layers), llama.cpp's projected total, and the GPU memory in use, then stop it. (When the projection
# exceeds the card, the Windows driver silently spills the rest to system RAM, so nvidia-smi under-reports.) One JSON row per setting -> results/kv_memory.jsonl.
#
#   scripts/kv_memory.sh [GGUF]        (default: the 19 B Q4_K_M)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:-gguf/q35-reap-k128/q35-reap-k128-Q4_K_M.gguf}
BIN=llama.cpp/build/bin
mkdir -p logs/kv results
for KV in f16 q8_0; do
    for C in 4096 32768 131072 262144; do
        LOG=logs/kv/$KV-$C.log
        $BIN/llama-server -m "$M" -ngl 99 -c $C -np 1 -fa on -ctk $KV -ctv $KV --port 8091 -v > "$LOG" 2>&1 &   # -v: the memory lines are verbose-only
        PID=$!
        for _ in $(seq 120); do grep -q "listening on\|error:\|failed to" "$LOG" && break; sleep 1; done
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        kill $PID; wait $PID 2>/dev/null || true
        # each size is logged twice (a dry run for --fit, then the real allocation): take the largest
        num() { grep -oE "$1 += +[0-9.]+ MiB" "$LOG" | grep -oE "[0-9.]+ MiB" | cut -d' ' -f1 | sort -n | tail -1 || true; }
        PROJ=$(grep -oE "projected to use [0-9]+ MiB" "$LOG" | head -1 | grep -oE "[0-9]+" || true)
        KVMIB=$(num "llama_kv_cache: size")
        RSMIB=$(num "llama_memory_recurrent: size")
        CBMIB=$(num "CUDA0 compute buffer size")
        MODMIB=$(num "CUDA0 model buffer size")
        echo "{\"kv_type\": \"$KV\", \"ctx\": $C, \"kv_mib\": ${KVMIB:-null}, \"recurrent_mib\": ${RSMIB:-null}, \"compute_mib\": ${CBMIB:-null}, \"model_mib\": ${MODMIB:-null}, \"projected_mib\": ${PROJ:-null}, \"gpu_used_mib\": $USED}" | tee -a results/kv_memory.jsonl
        sleep 2
    done
done
