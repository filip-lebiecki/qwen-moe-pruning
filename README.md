# Shrinking Qwen3.5-35B-A3B: MoE expert pruning, distillation, quantization

A learning project: shrink the Mixture-of-Experts model `Qwen/Qwen3.5-35B-A3B` on one
16 GB GPU (RTX 5070 Ti, 32 GB RAM, WSL2) by removing experts, distilling routers and
quantizing to GGUF, with every command kept so the process can be repeated.

**Read the lessons online: https://filip-lebiecki.github.io/qwen-moe-pruning/learn/**

## Where to start
- [`learn/README.html`](learn/README.html): the full notes: pipeline, exact commands, results.
- [`learn/start.html`](learn/start.html): learning path and cheat sheet of key numbers.
- [`learn/index.html`](learn/index.html): 16 animated chapters (tokens, experts, router, REAP,
  perplexity, distillation, quantization, KV cache, speed, sampling...) plus appendices A-J.

The pages are plain HTML/JS and open straight from disk (`file://`). Clone the repo and open them in a browser.

## Results in brief
| Model | Params | GSM8K | HumanEval | MMLU |
|---|---|---|---|---|
| Original (Q4_K_M) | 35 B | 96.4 | 95.7 | 86.7 |
| REAP, keep 128 of 256 experts | 19 B | 95.6 | 79.9 | 73.7 |
| Code specialist, keep 48 | 8.9 B | 74.8 | 87.2 | 39.0 |

The main finding: at equal file size (~10.6 GiB), quantizing the unpruned 35 B model to ~2.6 bpw
(avg 91.4) beats pruning to 19 B and quantizing to Q4_K_M (avg 82.7).

## Scripts
`scripts/` holds the numbered pipeline (`00_setup.sh` ... `17_expert_similarity.py`) plus the
helpers and report generators. `moe_stream.py` is the shared engine: it streams one layer at a
time through the GPU, so the 67 GiB model fits a 16 GB card. Run from the repo root with
`.venv/bin/python -I scripts/...`.

`results/` holds the measured numbers behind the lessons: perplexity sweeps (`eval*.jsonl`),
every benchmark answer (`bench/*.jsonl`), quantization sweep, speed, ablation and distillation logs,
and the verbatim generation tests (`*.txt`).

The repo leaves out models, GGUFs, calibration data, stats, the `.pt` tensor dumps in `results/`,
`.venv` and `llama.cpp`.
`scripts/00_setup.sh` and `scripts/01_download.sh` rebuild the environment and fetch the model.
