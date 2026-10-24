"""Pass 1: stream the calibration set through the original model and record
how much every routed expert in every layer is used / contributes.

Output stats/<name>.pt:
  {criterion: float64 [n_layers, n_experts]}  raw sums (see moe_stream.moe_forward)
  freq_by_source [n_layers, n_sources, n_experts], sources, n_tokens
plus the calibration perplexity of the unpruned model as a sanity check.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import Checkpoint, ExpertStats, Streamer, gpu_mem, patch_moe, ppl_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--data", default="data/calib.pt")
    ap.add_argument("--out", default="stats/calib.pt")
    ap.add_argument("--micro-batch", type=int, default=16)
    args = ap.parse_args()

    ckpt = Checkpoint(args.model)
    data = torch.load(args.data)
    tokens, sources = data["tokens"], data["source"]
    src_names = list(dict.fromkeys(sources))
    src_id = torch.tensor([src_names.index(s) for s in sources])
    print(f"{tokens.numel():,} calibration tokens, sources {src_names}")

    s = Streamer(ckpt, tokens, args.micro_batch)
    s.embed()
    per_layer = []
    t0 = time.time()
    for i, layer in s.layers():
        stats = ExpertStats(ckpt.n_experts, len(src_names))
        patch_moe(layer, stats)

        def set_source(b):
            L = tokens.shape[1]
            stats.source = src_id[b].repeat_interleave(L).to(s.device)

        s.run_layer(layer, before_batch=set_source)
        per_layer.append(stats.to_cpu())
        f = per_layer[-1]["freq"]
        print(f"layer {i:2d} {ckpt.text_config.layer_types[i]:16s} "
              f"dead experts: {(f == 0).sum().item():3d}  "
              f"max/min usage: {f.max().item() / max(f.min().item(), 1):8.1f}  "
              f"[{time.time() - t0:5.0f}s, {gpu_mem()}]", flush=True)
        torch.cuda.reset_peak_memory_stats()

    out = {k: torch.stack([p[k] for p in per_layer]) for k in ExpertStats.KEYS}
    out["freq_by_source"] = torch.stack([p["freq_by_source"] for p in per_layer])
    out["sources"] = src_names
    out["n_tokens"] = tokens.numel()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)

    nll, cnt = s.nll()
    print("calibration perplexity (unpruned):", {k: round(v, 3) for k, v in ppl_report(nll, cnt, sources).items()})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
