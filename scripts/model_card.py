"""Write a model card (README.md) for the pruned 19 B GGUF files, from this project's own results (step 6).

A model card is the README.md of a Hugging Face repo. Its YAML header is read by the Hub: license, base model
(links the repo to the original's model tree), tags (how people find it), and the library. The text below the
header is for people: what the model is, how it was made, measured quality, how to run it, its limits.

    .venv/bin/python -I scripts/model_card.py --dir gguf/q35-reap-k128      # writes gguf/q35-reap-k128/README.md
"""
import argparse
import json
from pathlib import Path


def bench(tag):
    out = {}
    for line in open("results/bench.jsonl"):
        r = json.loads(line)
        if r["tag"] == tag:
            out.update({t: r[t] for t in ("gsm8k", "humaneval", "mmlu") if t in r})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="gguf/q35-reap-k128")
    ap.add_argument("--project-url", default="", help="link to where the scripts are published, if anywhere")
    args = ap.parse_args()
    d = Path(args.dir)
    files = sorted(f for f in d.glob("*.gguf") if "BF16" not in f.name)
    models = [f for f in files if f.name != "imatrix.gguf"]
    main_file = next((f.name for f in models if "Q4_K_M" in f.name), models[0].name if models else "MODEL.gguf")
    pr, og = bench("general-k128"), bench("original-q4")
    row = lambda name, b: f"| {name} | " + " | ".join(f"{b[t]['score']:.1f}" if t in b else "–" for t in ("gsm8k", "humaneval", "mmlu")) + " |"  # noqa: E731
    sizes = "\n".join(f"| `{f.name}` | {f.stat().st_size / 2**30:.1f} GiB |" for f in files)
    card = f"""---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE
base_model: Qwen/Qwen3.5-35B-A3B
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- moe
- pruned
- reap
- qwen3.5
---

# Qwen3.5-35B-A3B, expert-pruned to 19 B (REAP, 128 of 256 experts), GGUF

A learning-project prune of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B): in each of the
40 MoE layers the 128 experts with the lowest REAP score (router weight x expert output norm, measured on 524k tokens of
web, chat, math and code text) were deleted. No retraining. 18.99 B parameters, about 3.4 B active per token; the
vision encoder is unchanged but not included here (text only), and the MTP head was dropped.

## Files

| file | size |
|---|---|
{sizes}

Quantized with llama.cpp and an importance matrix computed on the same calibration text (`imatrix.gguf`).

## Quality (measured, % correct; greedy decoding, thinking off)

| model | GSM8K (250) | HumanEval (164) | MMLU (300) |
|---|---|---|---|
{row("this model, Q4_K_M", pr)}
{row("original 35B, Q4_K_M", og)}

Math survived the prune; **code and knowledge did not**: about 16 points less on HumanEval and 13 on MMLU than the
unpruned model. Perplexity on a held-out mix rose from 4.92 to 5.92. On a 16 GB GPU, the unpruned model at a 3-bit
quantization (about 12 GiB) scored higher than this one in the same tests; this prune is mainly of interest for
studying expert pruning, or for GPUs with less memory.

## Run

```bash
llama-server -m {main_file} -ngl 99 -c 16384
```

About 200 tokens per second on an RTX 5070 Ti (16 GB) with all layers on the GPU. Use the model's chat template (built
into the GGUF); Qwen's recommended sampling for thinking mode: temperature 1.0, top_p 0.95, top_k 20.

## How it was made
{("Scripts and notes: " + args.project_url) if args.project_url else "Expert statistics streamed layer by layer through the BF16 model on one 16 GB GPU; experts ranked by REAP; tensors sliced; converted with llama.cpp's convert_hf_to_gguf.py and quantized with llama-quantize."}

## License

Apache 2.0, as the base model.
"""
    (d / "README.md").write_text(card)
    print(card)
    print(f"wrote {d / 'README.md'}")


if __name__ == "__main__":
    main()
