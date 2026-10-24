#!/usr/bin/env bash
# Measure one GGUF for the quantization curve (step 4) and append a row to results/quant/sweep.jsonl:
#   size      file bytes and bits per weight (bytes x 8 / N_PARAMS language-model parameters)
#   quality   wikitext perplexity (64 x 2048 tokens, as in run 2); KL divergence to the BF16 model's
#             next-token distributions and "same top token" rate, if a KLD reference file is given
#   speed     generation and prompt tokens/s (speed_test.py, via run_bench.sh)
#   tasks     the 714-question benchmark (08_bench.py), unless NO_BENCH=1
#
#   N_PARAMS=18.54e9 scripts/15_measure_gguf.sh FILE.gguf TAG QUANT [KLD_BASE]
set -euo pipefail
cd "$(dirname "$0")/.."
F=${1:?gguf}; TAG=${2:?tag}; QUANT=${3:?quant name}; KLD=${4:-}
BIN=llama.cpp/build/bin; WIKI=data/wikitext-2-test.txt; LOG=logs/quant/$TAG-$QUANT
mkdir -p logs/quant results/quant
pkill -x llama-server || true

$BIN/llama-perplexity -m "$F" -f $WIKI -c 2048 --chunks 64 --fit on > "$LOG-ppl.log" 2>&1
ppl=$(grep -oP "Final estimate: PPL = \K[0-9.]+" "$LOG-ppl.log")
kld=null; top=null
if [ -n "$KLD" ]; then
    $BIN/llama-perplexity -m "$F" -f $WIKI -c 2048 --chunks 16 --fit on --kl-divergence-base "$KLD" --kl-divergence > "$LOG-kld.log" 2>&1
    kld=$(grep -oP "Mean    KLD:\s+\K[0-9.]+" "$LOG-kld.log"); top=$(grep -oP "Same top p:\s+\K[0-9.]+" "$LOG-kld.log")
fi
if [ "${NO_BENCH:-0}" = 1 ]; then SKIP_BENCH=1 scripts/run_bench.sh "$F" "$TAG-$QUANT"; else scripts/run_bench.sh "$F" "$TAG-$QUANT"; fi

bytes=$(stat -L -c %s "$F")
python3 -I -c "
import json, sys
bytes_, n = int(sys.argv[1]), float(sys.argv[2])
row = {'tag': sys.argv[3], 'quant': sys.argv[4], 'bytes': bytes_, 'gib': round(bytes_ / 2**30, 2), 'bpw': round(bytes_ * 8 / n, 2),
       'ppl': float(sys.argv[5]), 'kld': None if sys.argv[6] == 'null' else float(sys.argv[6]),
       'same_top_pct': None if sys.argv[7] == 'null' else float(sys.argv[7])}
open('results/quant/sweep.jsonl', 'a').write(json.dumps(row) + '\n'); print(row)
" "$bytes" "${N_PARAMS:-18.54e9}" "$TAG" "$QUANT" "$ppl" "$kld" "$top"
