#!/usr/bin/env bash
# Step 4: the quantization trade-off curve of one model. For each llama.cpp type: quantize the BF16
# GGUF with the model's importance matrix, measure it (15_measure_gguf.sh), then delete it again
# unless it is listed in KEEP (disk). The BF16 model itself is measured first, and its next-token
# distributions on 16 x 2048 wikitext tokens are saved once as the KL-divergence reference (~8 GB).
#
#   NO_BENCH_TYPES="Q8_0 Q6_K" scripts/14_quant_sweep.sh BF16.gguf IMATRIX.gguf TAG Q8_0 Q6_K Q5_K_M ...
set -euo pipefail
cd "$(dirname "$0")/.."
BF16=${1:?bf16 gguf}; IMAT=${2:?imatrix}; TAG=${3:?tag}; shift 3
BIN=llama.cpp/build/bin; OUT=gguf/quant-sweep; KLD=$OUT/$TAG-base.kld
mkdir -p $OUT logs/quant
if [ ! -f "$KLD" ]; then
    $BIN/llama-perplexity -m "$BF16" -f data/wikitext-2-test.txt -c 2048 --chunks 16 --fit on \
        --kl-divergence-base "$KLD" > logs/quant/$TAG-kldbase.log 2>&1
fi
grep -q "\"tag\": \"$TAG\", \"quant\": \"BF16\"" results/quant/sweep.jsonl 2>/dev/null || \
    NO_BENCH=1 scripts/15_measure_gguf.sh "$BF16" "$TAG" BF16
for Q in "$@"; do
    F=$OUT/$TAG-$Q.gguf
    [ -f "$F" ] || $BIN/llama-quantize --imatrix "$IMAT" "$BF16" "$F" "$Q" > logs/quant/$TAG-$Q-quantize.log 2>&1
    nb=0; [[ " ${NO_BENCH_TYPES:-} " == *" $Q "* ]] && nb=1
    NO_BENCH=$nb scripts/15_measure_gguf.sh "$F" "$TAG" "$Q" "$KLD"
    [[ " ${KEEP:-} " == *" $Q "* ]] || rm -f "$F"
done
echo "SWEEP $TAG DONE"
