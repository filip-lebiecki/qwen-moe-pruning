"""Pass 2: write a pruned checkpoint that keeps the top-K experts per layer.

No GPU needed - this only slices tensors:
  mlp.experts.gate_up_proj [E, 1024, 2048] -> [K, 1024, 2048]
  mlp.experts.down_proj    [E, 2048, 512]  -> [K, 2048, 512]
  mlp.gate.weight          [E, 2048]       -> [K, 2048]   (router rows)
Everything else (attention, DeltaNet, shared expert, embeddings, vision tower)
is copied unchanged.  The MTP (multi-token prediction) head is dropped: it has
its own 256 experts we have no statistics for, and it is only used for
speculative decoding.

The result is a normal HF checkpoint (config num_experts=K) that transformers,
vLLM and llama.cpp's converter load like any other Qwen3.5-MoE.

--merge naive|aligned merges each dropped expert into its most similar survivor instead of deleting it
(stats/similarity.pt from 17_expert_similarity.py); the file has the same shape either way.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import CRITERIA, TEXT, Checkpoint, merge_layer_state, prune_layer_state, select_experts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--stats", default="stats/calib.pt")
    ap.add_argument("--keep", type=int, required=True, help="experts kept per layer (of 256)")
    ap.add_argument("--criterion", choices=CRITERIA, default="reap")
    ap.add_argument("--merge", choices=("naive", "aligned"), help="merge dropped experts instead of deleting them")
    ap.add_argument("--similarity", default="stats/similarity.pt")
    ap.add_argument("--sim-kind", choices=("output", "router"), default="output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Checkpoint(args.model)
    assert args.keep >= ckpt.text_config.num_experts_per_tok, "must keep at least top_k experts"
    stats = torch.load(args.stats)
    keep = select_experts(stats, args.keep, args.criterion)
    sim = torch.load(args.similarity)[args.sim_kind] if args.merge else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_shards = ckpt.n_layers + 1
    shard = lambda k: f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"  # noqa: E731
    weight_map, n_params, n_bytes = {}, {"text": 0, "vision": 0}, 0

    def write(k, tensors):
        nonlocal n_bytes
        save_file(tensors, out / shard(k), metadata={"format": "pt"})
        for name, t in tensors.items():
            weight_map[name] = shard(k)
            n_params["vision" if ".visual." in name else "text"] += t.numel()
            n_bytes += t.numel() * t.element_size()

    for i in range(ckpt.n_layers):
        if args.merge:
            state = merge_layer_state(ckpt.layer_state(i), keep[i], sim[i], stats["freq"][i], aligned=args.merge == "aligned")
        else:
            state = prune_layer_state(ckpt.layer_state(i), keep[i])
        write(i, {f"{TEXT}layers.{i}.{k}": v for k, v in state.items()})
        print(f"layer {i:2d}: kept {args.keep} experts" + (f" (others merged, {args.merge})" if args.merge else ""), flush=True)

    rest = [n for n in ckpt.weight_map if not n.startswith(TEXT + "layers.") and not n.startswith("mtp.")]
    write(ckpt.n_layers, ckpt.tensors(rest))

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": n_bytes}, "weight_map": weight_map}, indent=2))

    config = json.loads((ckpt.path / "config.json").read_text())
    config["text_config"]["num_experts"] = args.keep
    config["text_config"]["mtp_num_hidden_layers"] = 0
    (out / "config.json").write_text(json.dumps(config, indent=2))

    for f in ckpt.path.iterdir():  # tokenizer, chat template, processor configs...
        if f.is_file() and not f.name.startswith("model.safetensors") and f.name != "config.json":
            shutil.copy(f, out / f.name)

    (out / "pruning_info.json").write_text(json.dumps({
        "source": str(ckpt.path), "stats": args.stats, "criterion": args.criterion,
        "keep": args.keep, "merge": args.merge, "sim_kind": args.sim_kind if args.merge else None, "experts_kept": {i: k.tolist() for i, k in enumerate(keep)},
    }))

    top_k = ckpt.text_config.num_experts_per_tok
    expert_size = sum(state[n][0].numel() for n in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
    active = n_params["text"] - ckpt.n_layers * (args.keep - top_k) * expert_size
    print(f"params: text {n_params['text'] / 1e9:.2f}B + vision {n_params['vision'] / 1e9:.2f}B "
          f"= {sum(n_params.values()) / 1e9:.2f}B  (active per token ~{active / 1e9:.2f}B)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
