#!/usr/bin/env bash
# Any Hugging Face model -> GGUF -> importance matrix -> quantized files -> a perplexity check of each.
# The general recipe behind 07_gguf.sh, for models that were not pruned here (step 6).
#
#   scripts/make_gguf.sh Qwen/Qwen3.5-2B Q8_0 Q4_K_M IQ4_XS
#   OUT=/mnt/f/gguf/x scripts/make_gguf.sh org/model Q4_K_M      (put the files elsewhere)
#
# 1. hf download          the original safetensors (skipped if already there)
# 2. convert_hf_to_gguf   -> <name>-BF16.gguf (+ mmproj-<name>-F16.gguf: the vision part, if the model has one)
# 3. llama-imatrix        on wikitext-2 train text (generic English; use text like your own use for a specialist)
# 4. llama-quantize       -> one file per type, using the imatrix. If the model has an MTP head (an extra layer
#                         used only for speculative decoding, llama-server --spec-type draft-mtp), that block
#                         never runs in a normal forward pass, so the imatrix has no data for it and IQ1-IQ3
#                         types refuse to quantize it: it is stored at Q8_0 instead (it is small).
# 5. llama-perplexity     BF16 and every type on wikitext-2 test: the smoke test that the conversion is right
#                         and what each type costs. Rows -> results/make_gguf.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$1; shift
TYPES=("$@"); [ ${#TYPES[@]} -eq 0 ] && TYPES=(Q4_K_M)
NAME=$(basename "$REPO")
SRC=models/$NAME
OUT=${OUT:-gguf/$NAME}
BIN=llama.cpp/build/bin
PY=.venv/bin/python
mkdir -p "$OUT" data logs results

# 1. download (config.json present = already downloaded)
[ -f "$SRC/config.json" ] || hf download "$REPO" --local-dir "$SRC"
du -sh "$SRC"

# 2. convert. --mmproj writes the vision encoder to its own file; llama.cpp loads it with --mmproj when images are used.
BF16=$OUT/$NAME-BF16.gguf
if [ ! -f "$BF16" ]; then
    $PY llama.cpp/convert_hf_to_gguf.py "$SRC" --outtype bf16 --outfile "$BF16"
    if grep -q '"vision_config"' "$SRC/config.json"; then
        $PY llama.cpp/convert_hf_to_gguf.py "$SRC" --mmproj --outtype f16 --outfile "$OUT/mmproj-$NAME-F16.gguf" || echo "(no mmproj written)"
    fi
fi

# 3. calibration and test text (wikitext-2), written once
if [ ! -f data/wikitext-train.txt ]; then
    $PY -I -c "
from datasets import load_dataset
for split in ('train', 'test'):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    open(f'data/wikitext-{split}.txt', 'w').write(''.join(ds['text']))
"
fi
[ -f "$OUT/imatrix.gguf" ] || $BIN/llama-imatrix -m "$BF16" -f data/wikitext-train.txt -o "$OUT/imatrix.gguf" \
    -c 512 --chunks 200 --fit on > "logs/imatrix-$NAME.log" 2>&1

# 4. quantize (to a .part file first, so a failed run cannot leave a broken .gguf behind)
read -r NL NMTP < <($PY -I -c "
import json; c = json.load(open('$SRC/config.json')); t = c.get('text_config', c)
print(t['num_hidden_layers'], t.get('mtp_num_hidden_layers', 0))")
EXTRA=(); [ "$NMTP" -gt 0 ] && EXTRA=(--tensor-type "blk\\.$NL\\.=q8_0")   # the MTP block comes after the last layer
for T in "${TYPES[@]}"; do
    [ -f "$OUT/$NAME-$T.gguf" ] && continue
    $BIN/llama-quantize --imatrix "$OUT/imatrix.gguf" "${EXTRA[@]}" "$BF16" "$OUT/$NAME-$T.gguf.part" "$T" \
        > "logs/quantize-$NAME-$T.log" 2>&1 && mv "$OUT/$NAME-$T.gguf.part" "$OUT/$NAME-$T.gguf" \
        || { echo "quantize $T failed: see logs/quantize-$NAME-$T.log"; rm -f "$OUT/$NAME-$T.gguf.part"; }
done

# 5. perplexity of each file (lower is better; the quantized ones should be close to BF16)
for F in "$BF16" "${TYPES[@]/#/$OUT/$NAME-}"; do
    F=${F%.gguf}.gguf
    [ -f "$F" ] || continue
    T=$(basename "$F" .gguf); T=${T#"$NAME"-}
    PPL=$($BIN/llama-perplexity -m "$F" -f data/wikitext-test.txt -c 2048 --chunks 32 --fit on 2>&1 | tee "logs/ppl-$NAME-$T.log" \
        | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$" || true)
    GIB=$(du -b "$F" | awk '{printf "%.2f", $1 / 2^30}')
    echo "{\"repo\": \"$REPO\", \"type\": \"$T\", \"gib\": $GIB, \"ppl\": ${PPL:-null}}" | tee -a results/make_gguf.jsonl
done
