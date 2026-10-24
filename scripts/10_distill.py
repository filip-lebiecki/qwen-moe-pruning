"""Step 3b: end-to-end knowledge distillation of the routers, one layer on the GPU at a time.

Chapter 9 explained why layer-wise healing (06_heal.py) failed: matching hidden states inside each
layer is not the same goal as predicting the next token. Real distillation puts the loss on the
model's output: cross-entropy between the student's next-token distribution and the teacher's
(top-32 soft targets from 09_teacher_logits.py), backpropagated through all 40 layers.

The student does not fit on a 16 GB GPU, and full fine-tuning would need ~300 GB for weights,
gradients and Adam state. Two choices make it fit:

1. Train only the routers: 40 x 128 x 2048 = 10.5 M parameters (+ the 40 shared-expert gates).
   They are what pruning broke: tokens whose favourite experts were deleted must be re-routed
   well. Experts, attention, DeltaNet and embeddings stay frozen.
2. Stream the backward pass like the forward pass. For each batch:
     forward   layers 0..39, keep every layer's INPUT in CPU RAM
     loss      final norm + lm_head, cross-entropy to the teacher's top-32  ->  dL/d(hidden_40)
     backward  layers 39..0: load the layer again, recompute its forward from the saved input with
               autograd on, push dL/d(output) back to dL/d(input), collect dL/d(router)
   Only one layer is ever on the GPU, so memory stays as for evaluation; the price is reading every
   layer from disk twice per step. (Gradient checkpointing taken to the extreme: one checkpoint per
   layer, recomputed one layer at a time.)

The student is the teacher's checkpoint pruned on the fly (the same experts as 04_prune.py), so no
pruned copy is needed on disk. Outputs in results/distill/: routers_init.pt and routers.pt (42 MB
each), log.jsonl. 05_eval.py --routers and 11_patch_gguf.py use them.

  .venv/bin/python -I scripts/10_distill.py --gradcheck      # verify the streamed gradient first
  .venv/bin/python -I scripts/10_distill.py                  # train
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import (  # noqa: E402
    TEXT, VRAM_FRACTION, Checkpoint, Qwen3_5MoeRMSNorm, Qwen3_5MoeTextRotaryEmbedding,
    build_layer, patch_moe, prune_layer_state, select_experts,
)

GATE, SHEXP = "mlp.gate.weight", "mlp.shared_expert_gate.weight"


class Student:
    """The pruned model, streamed layer by layer, with its routers held as trainable fp32 tensors."""

    def __init__(self, ckpt, keep, micro_batch, device="cuda"):
        self.ckpt, self.keep, self.mb, self.device = ckpt, keep, micro_batch, device
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
        self.rotary = Qwen3_5MoeTextRotaryEmbedding(ckpt.text_config).to(device)
        n = ckpt.n_layers
        t = ckpt.tensors([f"{TEXT}layers.{i}.{GATE}" for i in range(n)] + [f"{TEXT}layers.{i}.{SHEXP}" for i in range(n)])
        self.gate = [t[f"{TEXT}layers.{i}.{GATE}"][keep[i]].float().to(device) for i in range(n)]
        self.shexp = [t[f"{TEXT}layers.{i}.{SHEXP}"].float().to(device) for i in range(n)]
        self.params = self.gate + self.shexp
        h = ckpt.tensors([TEXT + "embed_tokens.weight", TEXT + "norm.weight", "lm_head.weight"], device)
        self.embed_w, self.head = h[TEXT + "embed_tokens.weight"], h["lm_head.weight"]
        self.norm = Qwen3_5MoeRMSNorm(self.head.shape[1], eps=ckpt.text_config.rms_norm_eps).to(device)
        self.norm.weight.data.copy_(h[TEXT + "norm.weight"])
        self.norm.requires_grad_(False)

    def rope(self, h):
        pos = torch.arange(h.shape[1], device=self.device)[None].expand(h.shape[0], -1)
        return self.rotary(h, pos), pos

    def slices(self, n):
        return [slice(s, min(s + self.mb, n)) for s in range(0, n, self.mb)]

    def layers(self, order, train):
        """Yield (i, pruned layer on the GPU) in `order`, prefetching the next one from disk."""
        order = list(order)
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(self.ckpt.layer_state, order[0])
            for k, i in enumerate(order):
                state = fut.result()
                if k + 1 < len(order):
                    fut = pool.submit(self.ckpt.layer_state, order[k + 1])
                state = prune_layer_state(state, self.keep[i])
                state[GATE], state[SHEXP] = self.gate[i].to(torch.bfloat16), self.shexp[i].to(torch.bfloat16)
                layer = build_layer(self.ckpt.text_config, i, state, self.device)
                if train:  # the only trainable tensors: this layer's router and shared-expert gate
                    layer.mlp.gate.weight.requires_grad_(True)
                    layer.mlp.shared_expert_gate.weight.requires_grad_(True)
                patch_moe(layer)
                yield i, layer
                del layer, state
                torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, tokens):
        """Activations at every layer boundary (CPU, bf16): acts[i] is the input of layer i."""
        acts = [torch.empty(*tokens.shape, self.embed_w.shape[1], dtype=torch.bfloat16)]
        for b in self.slices(len(tokens)):
            acts[0][b] = F.embedding(tokens[b].to(self.device).long(), self.embed_w).cpu()
        for i, layer in self.layers(range(self.ckpt.n_layers), train=False):
            out = torch.empty_like(acts[i])
            for b in self.slices(len(tokens)):
                h = acts[i][b].to(self.device)
                pe, pos = self.rope(h)
                out[b] = layer(h, position_embeddings=pe, attention_mask=None, position_ids=pos).cpu()
            acts.append(out)
        return acts

    def loss(self, hidden, t_idx, t_logp, want_grad, chunk=1024):
        """Cross-entropy to the teacher's renormalised top-k distribution, mean per token.
        Returns (loss, kl, dL/d(hidden) on the CPU or None)."""
        n_tok = hidden.shape[0] * hidden.shape[1]
        grad = torch.empty(hidden.shape, dtype=torch.float32) if want_grad else None
        total = ent = 0.0
        for b in self.slices(len(hidden)):
            h = hidden[b].to(self.device).requires_grad_(want_grad)
            with torch.set_grad_enabled(want_grad):
                hn = self.norm(h)
            hn_leaf = hn.detach().reshape(-1, hn.shape[-1]).requires_grad_(want_grad)
            idx = t_idx[b].reshape(-1, t_idx.shape[-1]).to(self.device).long()
            pt = F.softmax(t_logp[b].reshape(-1, t_logp.shape[-1]).to(self.device).float(), dim=-1)
            for c in range(0, len(hn_leaf), chunk):
                with torch.set_grad_enabled(want_grad):
                    lp = F.log_softmax(F.linear(hn_leaf[c:c + chunk], self.head).float(), dim=-1)
                    loss = -(pt[c:c + chunk] * lp.gather(-1, idx[c:c + chunk])).sum() / n_tok
                if want_grad:
                    loss.backward()
                total += loss.item()
                ent += -(pt[c:c + chunk] * pt[c:c + chunk].clamp_min(1e-12).log()).sum().item() / n_tok
            if want_grad:
                hn.backward(hn_leaf.grad.reshape(hn.shape))
                grad[b] = h.grad.float().cpu()
        return total, total - ent, grad

    def backward(self, acts, grad):
        """Push dL/d(hidden_40) back through layers 39..0; return router gradients."""
        g_gate, g_shexp = [None] * self.ckpt.n_layers, [None] * self.ckpt.n_layers
        for i, layer in self.layers(reversed(range(self.ckpt.n_layers)), train=True):
            g_in = torch.empty_like(grad)
            for b in self.slices(len(grad)):
                x = acts[i][b].to(self.device).requires_grad_(True)
                pe, pos = self.rope(x)
                out = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos)
                out.backward(grad[b].to(self.device, out.dtype))
                g_in[b] = x.grad.float().cpu()
            g_gate[i] = layer.mlp.gate.weight.grad.float()
            g_shexp[i] = layer.mlp.shared_expert_gate.weight.grad.float()
            grad = g_in
        return g_gate + g_shexp

    def save(self, path, **meta):
        torch.save({"gate": [g.cpu() for g in self.gate], "shexp": [s.cpu() for s in self.shexp],
                    "keep": [k.cpu() for k in self.keep], **meta}, path)


def batch(data, targets, idx):
    return data[idx], targets["idx"][idx], targets["logp"][idx]


def gradcheck(student, data, targets, eps):
    """Compare the streamed gradient with finite differences, one router at a time.

    The step goes along that router's own gradient direction (unit length). It must be big enough
    to survive the bf16 rounding of the weights in the forward pass (~5e-5 at these magnitudes;
    eps=0.01 spreads to ~2e-5 per weight), yet small enough not to switch which experts tokens pick
    (a random step of ~2e-3 per weight did, and overstated the slope 3x)."""
    tok, ti, tl = batch(data, targets, torch.arange(2))
    acts = student.forward(tok)
    loss0, _, grad = student.loss(acts[-1], ti, tl, want_grad=True)
    grads = student.backward(acts, grad)
    print(f"loss {loss0:.5f}")
    for i in (39, 20, 5):  # last layer: no backprop through other layers; 5: through 34 of them
        d = grads[i] / grads[i].norm()
        vals = []
        for sign in (+1, -1):
            student.gate[i] += sign * eps * d
            vals.append(student.loss(student.forward(tok)[-1], ti, tl, want_grad=False)[0])
            student.gate[i] -= sign * eps * d
        numeric = (vals[0] - vals[1]) / (2 * eps)
        print(f"layer {i:2d}: streamed backward {grads[i].norm().item():.4e}, finite difference {numeric:.4e}, "
              f"ratio {grads[i].norm().item() / numeric:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--stats", default="stats/calib.pt")
    ap.add_argument("--keep", type=int, default=128)
    ap.add_argument("--criterion", default="reap")
    ap.add_argument("--data", default="data/train.pt")
    ap.add_argument("--targets", default="data/teacher_top32.pt")
    ap.add_argument("--out", default="results/distill")
    ap.add_argument("--batch-seqs", type=int, default=32, help="sequences per optimizer step (32 x 1024 tokens)")
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences on the GPU at once")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--val-seqs", type=int, default=32, help="held out from training, for the validation loss")
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gradcheck", action="store_true", help="only verify the streamed gradient")
    ap.add_argument("--eps", type=float, default=1e-2)
    args = ap.parse_args()
    torch.manual_seed(0)

    ckpt = Checkpoint(args.teacher)
    keep = select_experts(torch.load(args.stats), args.keep, args.criterion)
    data = torch.load(args.data)["tokens"]
    targets = torch.load(args.targets)
    student = Student(ckpt, keep, args.micro_batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.gradcheck:
        gradcheck(student, data, targets, args.eps)
        return

    # validation: evenly spread over the file (it is ordered by source), so it covers every domain
    val_ids = torch.linspace(0, len(data) - 1, args.val_seqs).long()
    train_ids = torch.tensor(sorted(set(range(len(data))) - set(val_ids.tolist())))
    n_train = len(train_ids)
    steps = int(args.epochs * n_train / args.batch_seqs)
    opt = torch.optim.AdamW(student.params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(  # short warm-up, then cosine decay to 10%
        opt, lambda s: min(1.0, (s + 1) / 3) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))))
    student.save(out / "routers_init.pt", step=0)
    log = open(out / "log.jsonl", "a")

    def validate(step):
        tok, ti, tl = batch(data, targets, val_ids)
        loss, kl, _ = student.loss(student.forward(tok)[-1], ti, tl, want_grad=False)
        row = {"step": step, "val_loss": round(loss, 5), "val_kl": round(kl, 5), "time": round(time.time() - t0)}
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"[validation] step {step:3d}  loss {loss:.4f}  KL to teacher {kl:.4f}", flush=True)
        return kl

    t0 = time.time()
    best = validate(0)
    order = train_ids[torch.randperm(n_train)]
    for step in range(1, steps + 1):
        k = (step - 1) * args.batch_seqs % n_train
        if k + args.batch_seqs > n_train:
            order, k = train_ids[torch.randperm(n_train)], 0
        tok, ti, tl = batch(data, targets, order[k:k + args.batch_seqs])
        acts = student.forward(tok)
        loss, kl, grad = student.loss(acts[-1], ti, tl, want_grad=True)
        grads = student.backward(acts, grad)
        del acts, grad
        for p, g in zip(student.params, grads):
            p.grad = g
        gnorm = torch.nn.utils.clip_grad_norm_(student.params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        row = {"step": step, "loss": round(loss, 5), "kl": round(kl, 5), "grad_norm": round(gnorm, 4),
               "lr": sched.get_last_lr()[0], "time": round(time.time() - t0)}
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"step {step:3d}/{steps}  loss {loss:.4f}  KL {kl:.4f}  |grad| {gnorm:.3f}  [{time.time() - t0:.0f}s]", flush=True)
        if step % args.eval_every == 0 or step == steps:
            kl_val = validate(step)
            if kl_val < best:
                best = kl_val
                student.save(out / "routers.pt", step=step, val_kl=kl_val)
    print(f"done; best validation KL {best:.4f}")


if __name__ == "__main__":
    main()
