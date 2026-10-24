"""Step 3a: the teacher's soft targets for distillation.

Stream the unpruned model over data/train.pt and keep, for every token position, the teacher's
top-32 next-token log-probabilities and their token ids. 10_distill.py trains the pruned student
to reproduce these distributions ("X 60%, Y 20%, Z 5% ..."), not just the one token that came next.
Top-32 keeps nearly all of the probability mass in 1/7,700 of the full 248k vocabulary:
1M tokens x 32 x 6 bytes = 192 MB instead of about 1 TB.

Output: data/teacher_top32.pt  {"idx": int32 [N, L, K], "logp": float16 [N, L, K], "mass": float32 [N, L]}
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import TEXT, Checkpoint, Qwen3_5MoeRMSNorm, Streamer, patch_moe  # noqa: E402


@torch.no_grad()
def top_k_targets(streamer, k, chunk=1024):
    """Final norm + lm_head on the streamed hidden states -> top-k log-probabilities per position."""
    ckpt = streamer.ckpt
    names = [TEXT + "norm.weight", "lm_head.weight"]
    t = ckpt.tensors(names, streamer.device)
    norm = Qwen3_5MoeRMSNorm(t[names[0]].shape[0], eps=ckpt.text_config.rms_norm_eps).to(streamer.device)
    norm.weight.data.copy_(t[names[0]])
    head = t[names[1]]
    N, L = streamer.tokens.shape
    idx = torch.empty(N, L, k, dtype=torch.int32)
    logp = torch.empty(N, L, k, dtype=torch.float16)
    mass = torch.empty(N, L, dtype=torch.float32)
    for n in range(N):
        h = norm(streamer.hidden[n].to(streamer.device))
        for c in range(0, L, chunk):
            lp = F.log_softmax(F.linear(h[c:c + chunk], head).float(), dim=-1)
            v, i = lp.topk(k, dim=-1)
            idx[n, c:c + chunk], logp[n, c:c + chunk] = i.int().cpu(), v.half().cpu()
            mass[n, c:c + chunk] = v.exp().sum(-1).cpu()
    return idx, logp, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--data", default="data/train.pt")
    ap.add_argument("--out", default="data/teacher_top32.pt")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--chunk-seqs", type=int, default=512, help="sequences per streaming pass (RAM: 2 GB per 512)")
    ap.add_argument("--micro-batch", type=int, default=16)
    args = ap.parse_args()

    ckpt = Checkpoint(args.model)
    tokens = torch.load(args.data)["tokens"]
    parts, t0 = [], time.time()
    for s in range(0, len(tokens), args.chunk_seqs):
        st = Streamer(ckpt, tokens[s:s + args.chunk_seqs], args.micro_batch)
        st.embed()
        for i, layer in st.layers():
            patch_moe(layer)
            st.run_layer(layer)
            print(f"\rsequences {s}-{s + len(st.tokens)}: layer {i + 1}/{ckpt.n_layers}  [{time.time() - t0:.0f}s]", end="", flush=True)
        parts.append(top_k_targets(st, args.topk))
        print()
    idx, logp, mass = (torch.cat(x) for x in zip(*parts))
    torch.save({"idx": idx, "logp": logp, "mass": mass, "topk": args.topk}, args.out)
    print(f"saved {args.out}: {tuple(idx.shape)}, top-{args.topk} mass mean {mass.mean():.3f}, "
          f"min {mass.min():.3f}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
