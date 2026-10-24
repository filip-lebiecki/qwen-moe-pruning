#!/usr/bin/env bash
# One-time environment setup (Ubuntu 24.04 on WSL2, NVIDIA RTX 50xx / Blackwell).
set -euo pipefail
cd "$(dirname "$0")/.."

# --- CUDA toolkit (nvcc) for building llama.cpp; the GPU driver comes from Windows.
if ! command -v /usr/local/cuda/bin/nvcc >/dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-13-0
fi
sudo apt-get install -y cmake ninja-build

# --- Python env: torch (CUDA 13), transformers with Qwen3.5-MoE, flash-linear-attention
#     (fast Gated DeltaNet kernels). Versions pinned to the ones
#     this project was run with: moe_stream.py uses transformers internals that can change.
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python transformers==5.20.2 flash-linear-attention==0.5.2 \
    accelerate datasets safetensors scipy sentencepiece protobuf numpy tqdm "huggingface_hub[hf_xet]"

# --- llama.cpp (GGUF conversion, quantization, inference), pinned to the commit used here.
#     sm_120 = RTX 50xx; change CMAKE_CUDA_ARCHITECTURES for other GPUs (e.g. 89 = RTX 40xx).
LLAMA_COMMIT=7775ffb7e48fa0c7dcffd5815743b3d2fb3b6761
if [ ! -d llama.cpp ]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp
    git -C llama.cpp checkout "$LLAMA_COMMIT"
fi
uv pip install --python .venv/bin/python -e llama.cpp/gguf-py
export PATH=/usr/local/cuda/bin:$PATH
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build llama.cpp/build -j "$(nproc)" --target \
    llama-cli llama-quantize llama-imatrix llama-perplexity llama-server llama-bench llama-gguf-split
