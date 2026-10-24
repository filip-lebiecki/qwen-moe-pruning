#!/usr/bin/env bash
# Speculative decoding sweep (after step 6): three ways to draft tokens, each against the same model without drafting.
#   1. MTP: the model's own multi-token-prediction head (Qwen3.5-2B; our 19 B and unsloth's 35 B GGUFs have none)
#   2. a small draft model: Qwen3.5-2B drafts for the 19 B (same tokenizer, so its guesses can be checked)
#   3. n-gram lookup: drafts copied from earlier text in the context, no extra model at all
#   scripts/spec_sweep.sh            -> results/spec.jsonl, logs/spec/
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=llama.cpp/build/bin
SMALL=gguf/Qwen3.5-2B-Q4_K_M.gguf                  # make_gguf.sh output; copy it back first:
# cp /mnt/f/qwen-scratch/step6/gguf-Qwen3.5-2B/Qwen3.5-2B-Q4_K_M.gguf gguf/
BIG=gguf/q35-reap-k128/q35-reap-k128-Q4_K_M.gguf
mkdir -p logs/spec
run() {  # $1 = tag, rest = llama-server arguments
    local TAG=$1; shift
    local LOG=logs/spec/$TAG.log
    $BIN/llama-server -ngl 99 -c 8192 -np 1 --port 8094 "$@" > "$LOG" 2>&1 &
    local PID=$!
    for _ in $(seq 120); do grep -q "listening on\|error:" "$LOG" && break; sleep 1; done
    .venv/bin/python -I scripts/spec_test.py --port 8094 --tag "$TAG" || echo "{\"tag\": \"$TAG\", \"error\": true}"
    kill $PID; wait $PID 2>/dev/null || true
    sleep 2
}
run 2b-none -m $SMALL
for N in 1 2 3 4; do run 2b-mtp$N -m $SMALL --spec-type draft-mtp --spec-draft-n-max $N; done
run 19b-none -m $BIG
run 19b-draft2b -m $BIG --spec-type draft-simple -md $SMALL -ngld 99
run 19b-ngram -m $BIG --spec-type ngram-simple
