"""Pass 3 (optional): layer-wise knowledge distillation ("healing").

Full distillation backpropagates a KL loss through the whole student, which
needs the student in VRAM with optimizer state - far beyond 16 GB here.
Instead we distill *locally*, one MoE block at a time, while streaming
teacher and student side by side:

  for layer i:
      teacher: X_t -> layer_i (all 256 experts)     -> Y_t
      student: X_s -> attention/DeltaNet part -> r,  u = post_norm(r)
      target  (--target moe):   teacher MoE on the student's input, MoE_t(u)
              (--target layer): Y_t - r  (also corrects drift from earlier layers)
      train the student MoE block to minimise ||MoE_s(u) - target||^2
      X_s <- healed student layer(X_s),  X_t <- Y_t

--train router   only the router rows (+ shared-expert gate): few params, safe.
                 Pruning breaks the router's calibration - tokens whose favourite
                 experts are gone now fall back to arbitrary second choices -
                 and this is the cheapest thing to fix.
--train moe      router + kept experts + shared expert (more capacity, can overfit).

The student checkpoint is updated in place (one shard per layer, from 04_prune.py).

Measured in this project it did NOT help: perplexity K=128 5.92 -> 5.96 (target=moe),
K=48 11.7 -> 12.3 (moe) / 15.8 (layer). Lower local MSE is not better next-token
prediction; real recovery needs a KL loss on the logits through the whole model.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import TEXT, Checkpoint, Streamer, gpu_mem, moe_forward, patch_moe  # noqa: E402


def rel_err(pred, target):
    return (pred.float() - target.float()).pow(2).sum() / target.float().pow(2).sum()


def train_block(mlp, u, target, args):
    params = [mlp.gate.weight, mlp.shared_expert_gate.weight]
    if args.train == "moe":
        params = list(mlp.parameters())
    for p in mlp.parameters():
        p.requires_grad_(False)
    for p in params:
        p.data = p.data.float()
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    n = len(u)
    n_val = n // 10
    perm = torch.randperm(n, device=u.device)
    val, tr = perm[:n_val], perm[n_val:]

    def evaluate():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            errs = [rel_err(moe_forward(mlp, u[val[j:j + args.batch_tokens]]), target[val[j:j + args.batch_tokens]])
                    for j in range(0, n_val, args.batch_tokens)]
        return torch.stack(errs).mean().item()

    before = evaluate()
    steps = args.epochs * len(tr) // args.batch_tokens
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    for step in range(steps):
        j = tr[torch.randint(len(tr), (args.batch_tokens,), device=u.device)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = rel_err(moe_forward(mlp, u[j]), target[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    after = evaluate()

    for p in params:
        p.grad = None
        p.requires_grad_(False)
        p.data = p.data.to(torch.bfloat16)
    return before, after


def save_layer(ckpt, i, layer):
    """Rewrite layer i's shard (04_prune.py writes exactly one shard per layer)."""
    prefix = f"{TEXT}layers.{i}."
    files = {f for n, f in ckpt.weight_map.items() if n.startswith(prefix)}
    assert len(files) == 1, f"layer {i} spans {files}; expected a 04_prune.py checkpoint"
    file = ckpt.path / files.pop()
    others = [n for n, f in ckpt.weight_map.items() if f == file.name and not n.startswith(prefix)]
    assert not others, f"{file.name} also holds other tensors"
    state = {prefix + k: v.detach().to("cpu").contiguous() for k, v in layer.state_dict().items()}
    tmp = file.with_suffix(".tmp")
    save_file(state, tmp, metadata={"format": "pt"})
    os.replace(tmp, file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--student", required=True, help="checkpoint from 04_prune.py (modified in place)")
    ap.add_argument("--data", default="data/calib.pt")
    ap.add_argument("--n-seqs", type=int, default=128, help="calibration sequences used (spread over sources)")
    ap.add_argument("--train", choices=("router", "moe"), default="router")
    ap.add_argument("--target", choices=("moe", "layer"), default="moe")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)

    teacher, student = Checkpoint(args.teacher), Checkpoint(args.student)
    data = torch.load(args.data)
    idx = torch.linspace(0, len(data["tokens"]) - 1, args.n_seqs).long()
    tokens = data["tokens"][idx]
    print(f"healing on {tokens.numel():,} tokens, train={args.train} target={args.target} lr={args.lr}")

    st = Streamer(teacher, tokens, args.micro_batch)
    st.embed()
    ss = Streamer(student, tokens, args.micro_batch)
    ss.hidden = st.hidden.clone().pin_memory()  # embeddings are shared, so both start identical

    t0 = time.time()
    for (i, T), (_, S) in zip(st.layers(), ss.layers()):
        patch_moe(T)
        patch_moe(S)
        us, targets = [], []
        with torch.no_grad():
            for b in st.batches():
                ht = st.hidden[b].to(st.device)
                hs = ss.hidden[b].to(ss.device)
                pe, pos = st.rope(ht)
                yt = T(ht, position_embeddings=pe, attention_mask=None, position_ids=pos)
                st.hidden[b] = yt.cpu()
                # student up to the MoE input (decoder layer forward, split in two)
                x = S.input_layernorm(hs)
                if S.block_type == "linear_attention":
                    x = S.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
                else:
                    x, _ = S.self_attn(hidden_states=x, attention_mask=None, position_ids=pos,
                                       past_key_values=None, position_embeddings=pe)
                r = hs + x
                u = S.post_attention_layernorm(r)
                target = moe_forward(T.mlp, u) if args.target == "moe" else yt - r
                us.append(u.flatten(0, 1))
                targets.append(target.flatten(0, 1))
        u, target = torch.cat(us), torch.cat(targets)
        del us, targets
        before, after = train_block(S.mlp, u, target, args)
        del u, target
        with torch.no_grad():
            ss.run_layer(S)
        save_layer(student, i, S)
        print(f"layer {i:2d}: MoE relative error {before:.4f} -> {after:.4f}  "
              f"[{time.time() - t0:5.0f}s, {gpu_mem()}]", flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
