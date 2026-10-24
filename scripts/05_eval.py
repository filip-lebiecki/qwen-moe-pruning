"""Held-out perplexity of a (pruned) checkpoint, per data source.

Two modes:
  --model DIR                         evaluate a checkpoint on disk
  --model ORIG --stats S --keep K     prune on the fly (nothing written to disk),
                                      handy for sweeping K / criteria cheaply
  ... --routers results/distill/routers.pt
                                      also replace the routers with distilled ones (10_distill.py)
  ... --merge naive|aligned           merge the dropped experts into their most similar survivor
                                      instead of deleting them (17_expert_similarity.py)
  ... --merge aligned-remap           the same, but keep the full router: choosing a dropped expert runs
                                      its group's merged expert (evaluation only, saves no memory)
Each run appends one JSON line to results/eval.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import CRITERIA, Checkpoint, Streamer, merge_layer_state, patch_moe, ppl_report, prune_layer_state, select_experts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--stats", help="prune on the fly using these stats")
    ap.add_argument("--keep", type=int)
    ap.add_argument("--criterion", choices=CRITERIA, default="reap")
    ap.add_argument("--routers", help="router weights from 10_distill.py (needs --stats/--keep)")
    ap.add_argument("--keep-file", help="JSON list with one K per layer (13_layer_sensitivity.py); needs --stats")
    ap.add_argument("--merge", choices=("naive", "aligned", "aligned-remap"), help="merge dropped experts instead of deleting them")
    ap.add_argument("--similarity", default="stats/similarity.pt")
    ap.add_argument("--sim-kind", choices=("output", "router"), default="output", help="which similarity picks the merge target")
    ap.add_argument("--data", default="data/eval.pt")
    ap.add_argument("--tag", help="name for the results row")
    ap.add_argument("--results", default="results/eval.jsonl")
    ap.add_argument("--micro-batch", type=int, default=16)
    args = ap.parse_args()
    if args.keep_file:
        args.keep = json.loads(Path(args.keep_file).read_text())
    if args.stats and not args.keep:
        ap.error("--stats needs --keep K (experts kept per layer) or --keep-file")

    ckpt = Checkpoint(args.model)
    data = torch.load(args.data)
    transform = None
    if args.stats:
        stats = torch.load(args.stats)
        keep = select_experts(stats, args.keep, args.criterion)
        sim = torch.load(args.similarity)[args.sim_kind] if args.merge else None
        routers = torch.load(args.routers) if args.routers else None
        if routers:
            assert all(torch.equal(a, b) for a, b in zip(keep, routers["keep"])), "routers were trained for other experts"

        def prune(i, state):
            if args.merge:
                state = merge_layer_state(state, keep[i], sim[i], stats["freq"][i], aligned=args.merge != "naive",
                                          remap=args.merge == "aligned-remap")
            else:
                state = prune_layer_state(state, keep[i])
            if routers:
                state["mlp.gate.weight"] = routers["gate"][i].to(torch.bfloat16)
                state["mlp.shared_expert_gate.weight"] = routers["shexp"][i].to(torch.bfloat16)
            return state
        transform = prune

    t0 = time.time()
    s = Streamer(ckpt, data["tokens"], args.micro_batch)
    s.embed()
    for i, layer in s.layers(transform):
        patch_moe(layer)
        s.run_layer(layer)
        print(f"\rlayer {i + 1}/{ckpt.n_layers}", end="", flush=True)
    nll, cnt = s.nll()
    ppl = ppl_report(nll, cnt, data["source"])

    experts = (round(sum(args.keep) / len(args.keep), 1) if isinstance(args.keep, list) else args.keep) if args.stats else ckpt.n_experts
    tag = args.tag or ((f"merge-{args.merge}-{args.sim_kind}-keep{experts}" if args.merge else f"{args.criterion}-keep{experts}")
                       if args.stats else Path(args.model).name)
    row = {"tag": tag, "experts": experts, "ppl": {k: round(v, 3) for k, v in ppl.items()},
           "seconds": round(time.time() - t0)}
    print("\n" + json.dumps(row))
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
