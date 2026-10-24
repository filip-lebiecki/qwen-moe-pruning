#!/usr/bin/env bash
# Baseline for the benchmark: the UNPRUNED Qwen3.5-35B-A3B as a published 4-bit GGUF (20.5 GiB),
# so pruned models can be compared at the same quantization. Qwen publishes no GGUF of this model;
# unsloth's is the most used one (note: its imatrix was computed on unsloth's own calibration text).
set -euo pipefail
cd "$(dirname "$0")/.."
# optional argument: another file from the same repo, e.g. Qwen3.5-35B-A3B-UD-IQ2_M.gguf (step 4)
# DEST=/mnt/f/... to put it elsewhere (step 6 kept the 20.5 GiB file on the F: drive)
hf download unsloth/Qwen3.5-35B-A3B-GGUF "${1:-Qwen3.5-35B-A3B-Q4_K_M.gguf}" --local-dir "${DEST:-gguf/original-unsloth}"
