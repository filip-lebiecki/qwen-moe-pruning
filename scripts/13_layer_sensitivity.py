"""Which layers can lose experts, and which can't? Measure it, then allocate K per layer.

Statistics only say how much each expert contributes inside its own layer. Whether pruning a layer
hurts the OUTPUT has to be measured: prune just that one layer (to --test-k experts, the others
stay complete) and see how much the perplexity rises. 40 measurements, one per layer.

To save time the unpruned hidden states at every layer boundary are stored once (RAM); the test for
layer L starts from the stored input of layer L, so layers 0..L-1 are never recomputed.

Then the expert budget is shared out in proportion to sensitivity^alpha (more experts where pruning
hurts most), keeping the same total as a uniform K, and written as JSON lists for
05_eval.py --keep-file. Note: a model with a different K per layer cannot be saved in the standard
HF/GGUF formats (both hold one num_experts), so this is an experiment, evaluated by streaming.

    .venv/bin/python -I scripts/13_layer_sensitivity.py
    .venv/bin/python -I scripts/05_eval.py --stats stats/calib.pt --keep-file results/alloc_k48.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import Checkpoint, Streamer, patch_moe, prune_layer_state, select_experts  # noqa: E402


def allocate(sens, avg_k, alpha=0.5, lo=16, hi=256):
    """Share out n_layers * avg_k experts in proportion to sensitivity^alpha, clamped to [lo, hi]."""
    w = torch.tensor(sens).clamp(min=1e-6) ** alpha
    budget = avg_k * len(sens)
    scale = budget / w.sum()
    for _ in range(50):  # re-scale until the clamped total meets the budget
        k = (w * scale).round().clamp(lo, hi)
        if k.sum() == budget:
            break
        scale *= budget / k.sum()
    k = (w * scale).round().clamp(lo, hi).long()
    while k.sum() != budget:  # fix rounding on the most / least sensitive layer that still has room
        up = k.sum() < budget
        room = (k < hi) if up else (k > lo)
        j = int(torch.where(room, w, -1.0 if up else float("inf")).argmax() if up else torch.where(room, w, float("inf")).argmin())
        k[j] += 1 if up else -1
    return k.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--stats", default="stats/calib.pt")
    ap.add_argument("--data", default="data/train.pt", help="not the eval set: the allocation is scored on that later")
    ap.add_argument("--n-seqs", type=int, default=32, help="eval sequences, spread over all domains")
    ap.add_argument("--test-k", type=int, default=32, help="experts kept in the one pruned layer")
    ap.add_argument("--budgets", type=int, nargs="+", default=[48, 128], help="average K per layer to allocate")
    ap.add_argument("--micro-batch", type=int, default=16)
    args = ap.parse_args()

    ckpt = Checkpoint(args.model)
    data = torch.load(args.data)
    ids = torch.linspace(0, len(data["tokens"]) - 1, args.n_seqs).long()
    keep = select_experts(torch.load(args.stats), args.test_k, "reap")
    st = Streamer(ckpt, data["tokens"][ids], args.micro_batch)
    st.embed()

    # baseline pass: store the input of every layer
    t0 = time.time()
    saved = [st.hidden.clone()]
    for i, layer in st.layers():
        patch_moe(layer)
        st.run_layer(layer)
        saved.append(st.hidden.clone())
    nll, cnt = st.nll()
    base = float(torch.exp(nll.sum() / cnt.sum()))
    print(f"unpruned perplexity on {args.n_seqs} sequences: {base:.4f}  [{time.time() - t0:.0f}s]", flush=True)

    sens = []
    for L in range(ckpt.n_layers):
        st.hidden = saved[L].clone()
        for i, layer in st.layers(transform_state=lambda i, s: prune_layer_state(s, keep[i]) if i == L else s, start=L):
            patch_moe(layer)
            st.run_layer(layer)
        nll, cnt = st.nll()
        ppl = float(torch.exp(nll.sum() / cnt.sum()))
        sens.append(ppl / base - 1)
        print(f"layer {L:2d} pruned to {args.test_k}: perplexity {ppl:.4f}  (+{100 * sens[-1]:.2f}%)  [{time.time() - t0:.0f}s]", flush=True)

    out = {"base_ppl": base, "test_k": args.test_k, "n_seqs": args.n_seqs, "sensitivity": sens}
    for b in args.budgets:
        k = allocate(sens, b)
        out[f"alloc_k{b}"] = k
        Path(f"results/alloc_k{b}.json").write_text(json.dumps(k))
        print(f"budget {b} per layer on average: {k}")
    Path("results/layer_sensitivity.json").write_text(json.dumps(out, indent=1))
    print("wrote results/layer_sensitivity.json")


if __name__ == "__main__":
    main()
