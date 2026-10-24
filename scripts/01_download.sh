#!/usr/bin/env bash
# Download the original BF16 weights (72 GB = 67 GiB) of the MoE teacher.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download Qwen/Qwen3.5-35B-A3B --local-dir models/Qwen3.5-35B-A3B
