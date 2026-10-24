#!/usr/bin/env bash
# Convert a (pruned) HF checkpoint to GGUF and quantize it for llama.cpp.
#
#   scripts/07_gguf.sh models/<pruned-dir> [QUANT ...]     (default: Q4_K_M IQ4_XS)
#   CALIB=data/calib_code.pt scripts/07_gguf.sh ...         (imatrix text for a specialist)
#
# 1. HF safetensors -> GGUF BF16           (llama.cpp/convert_hf_to_gguf.py)
# 2. importance matrix on calibration text (llama-imatrix): records which weight
#    columns see large activations so the quantizer spends its bits there.
#    IQ4_XS and other i-quants effectively need it; it also helps Q4_K_M.
# 3. llama-quantize BF16 -> each requested type
# The BF16 GGUF is kept (it's the source for any re-quantization); delete it to save disk.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:?usage: $0 <hf-checkpoint-dir> [QUANT ...]}
shift
QUANTS=("$@")
[ ${#QUANTS[@]} -eq 0 ] && QUANTS=(Q4_K_M IQ4_XS)

NAME=$(basename "$SRC")
OUT=gguf/$NAME
BIN=llama.cpp/build/bin
PY=.venv/bin/python
CALIB=${CALIB:-data/calib.pt}  # the imatrix should see the same kind of text the model is for
mkdir -p "$OUT"
BF16=$OUT/$NAME-BF16.gguf

# --no-mtp: 04_prune.py drops the multi-token-prediction head
if [ ! -f "$BF16" ]; then
    $PY llama.cpp/convert_hf_to_gguf.py "$SRC" --outtype bf16 --no-mtp --outfile "$BF16"
fi

# Calibration text for the imatrix: decode our calibration tokens back to text
if [ ! -f "$OUT/imatrix.gguf" ]; then
    $PY -I -c "
import torch, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
t = torch.load(sys.argv[2])['tokens']
print('\n'.join(tok.decode(s) for s in t[::2].tolist()))
" "$SRC" "$CALIB" > "$OUT/imatrix-calib.txt"
    # --fit on: let llama.cpp decide how many layers / experts fit in VRAM
    $BIN/llama-imatrix -m "$BF16" -f "$OUT/imatrix-calib.txt" -o "$OUT/imatrix.gguf" \
        -c 1024 --chunks 256 --parse-special --fit on
fi

for Q in "${QUANTS[@]}"; do
    $BIN/llama-quantize --imatrix "$OUT/imatrix.gguf" "$BF16" "$OUT/$NAME-$Q.gguf" "$Q"
done
ls -lh "$OUT"/*.gguf
