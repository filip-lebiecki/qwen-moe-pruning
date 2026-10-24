"""Layer-by-layer ("streaming") execution of Qwen3.5-MoE checkpoints.

Qwen3.5-35B-A3B is ~67 GiB in BF16: more than RAM + VRAM on a 32 GB / 16 GB
machine.  But one decoder layer is only ~1.7 GB.  So we keep the activations
of the whole calibration set in CPU RAM ([N, L, 2048] bf16 = 4 KiB/token) and
push them through the network one layer at a time: load layer i to the GPU,
run every sequence through it, free it, load layer i+1 (prefetched from disk
in a background thread while layer i computes).  Calibration and evaluation
only need forward passes, so this is all we need - it is the same trick
GPTQ/AWQ-style quantizers use.

The MoE block is re-implemented in `moe_forward` (the same math as the HF one;
outputs agree to bf16 rounding) so we can (a) record per-expert statistics and (b) run it faster than
the HF eager loop.
"""
import copy
import json
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)

TEXT = "model.language_model."
EXPERT_TENSORS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj", "mlp.gate.weight")

# Per-expert token counts differ every batch, so the CUDA caching allocator
# fragments and its reserved pool creeps up.  On Windows/WSL the driver then
# silently backs the excess with system RAM ("sysmem fallback") and kernels run
# 10-30x slower instead of failing.  Expandable segments limit fragmentation,
# and a hard cap below physical VRAM makes PyTorch free its cache and retry.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
VRAM_FRACTION = float(os.environ.get("VRAM_FRACTION", "0.85"))


def gpu_mem():
    return (f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB alloc / "
            f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved")


class Checkpoint:
    """Lazy access to a sharded safetensors checkpoint."""

    def __init__(self, path):
        self.path = Path(path)
        self.config = AutoConfig.from_pretrained(path)
        self.text_config = self.config.text_config
        self.text_config._attn_implementation = "sdpa"
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]

    @property
    def n_layers(self):
        return self.text_config.num_hidden_layers

    @property
    def n_experts(self):
        return self.text_config.num_experts

    def tensors(self, names, device="cpu"):
        by_file = {}
        for n in names:
            by_file.setdefault(self.weight_map[n], []).append(n)
        out = {}
        for file, ns in by_file.items():
            with safe_open(self.path / file, framework="pt", device=str(device)) as f:
                for n in ns:
                    out[n] = f.get_tensor(n)
        return out

    def layer_state(self, i):
        prefix = f"{TEXT}layers.{i}."
        t = self.tensors([n for n in self.weight_map if n.startswith(prefix)])
        return {n[len(prefix):]: v for n, v in t.items()}


def prune_layer_state(state, keep):
    """Keep only experts `keep` (sorted LongTensor) in one layer's state dict.

    Pruning an MoE layer is just slicing: each expert owns one slice of the
    fused expert tensors and one row of the router matrix.  The router's
    softmax/top-k then simply runs over fewer candidates.
    """
    state = dict(state)
    for name in EXPERT_TENSORS:
        state[name] = state[name][keep].contiguous()
    return state


def align_neurons(gu_t, dn_t, gu_d, dn_d):
    """Line up expert d's hidden neurons with expert t's, so that averaging them pairs like with like.

    An expert computes down @ (silu(gate @ x) * (up @ x)) with 512 hidden neurons; neuron j owns gate row j,
    up row j and down column j. Two symmetries leave the function unchanged:
      permutation  list the neurons in any order
      sign         negate a neuron's up row AND down column together (silu(g) * (-u) * (-w) = silu(g) * u * w)
    Two separately trained experts have no reason to agree on either, so neuron j of one has nothing to do
    with neuron j of the other. Match them (Hungarian algorithm on the cosine similarity of the gate, up and
    down vectors), and return d's tensors permuted and sign-flipped into t's order.
    gu_*: [2I, H] (gate rows then up rows), dn_*: [H, I]."""
    from scipy.optimize import linear_sum_assignment
    I = dn_t.shape[1]
    unit = lambda m: F.normalize(m.float(), dim=-1)  # noqa: E731
    cg = unit(gu_t[:I]) @ unit(gu_d[:I]).T
    cuw = unit(gu_t[I:]) @ unit(gu_d[I:]).T + unit(dn_t.T) @ unit(dn_d.T).T
    sign = torch.where(cuw >= 0, 1.0, -1.0)
    score = cg + cuw.abs()  # [I, I]: best of "same sign" and "flipped" for each pair
    _, perm = linear_sum_assignment(score.cpu().numpy(), maximize=True)
    perm = torch.as_tensor(perm)
    flip = sign[torch.arange(I), perm].to(gu_d.dtype)
    gate, up = gu_d[:I][perm], gu_d[I:][perm] * flip[:, None]
    return torch.cat([gate, up]), dn_d[:, perm] * flip[None, :]


def merge_layer_state(state, keep, sim, weight, aligned=False, remap=False):
    """Like prune_layer_state, but every dropped expert is first MERGED into the kept expert it is most
    similar to (sim [E, E], higher = more alike) instead of being thrown away.

    The merged expert is the weighted average of the group's weights, weighted by `weight` [E] (how often
    each expert was used), with the neurons aligned first if `aligned` (see align_neurons). The survivor
    keeps its own router row: tokens that used to pick a dropped expert now go wherever the router sends
    them, and some of them land on the merged expert that now contains a share of the dropped one.

    remap=True keeps the full router instead (M-SMoE style): a dropped expert's row stays, and choosing it
    now runs the merged expert of its group. The state keeps all E slots (members of a group share the
    merged weights), so this is for evaluation only: it saves no memory in a standard checkpoint."""
    gu_name, dn_name = EXPERT_TENSORS[0], EXPERT_TENSORS[1]
    gu, dn = state[gu_name].clone(), state[dn_name].clone()
    E = gu.shape[0]
    kept = torch.zeros(E, dtype=torch.bool)
    kept[keep] = True
    dropped = (~kept).nonzero().flatten()
    target = keep[sim[dropped][:, keep].argmax(1)]  # most similar survivor of each dropped expert
    w = weight.double().clamp(min=1)
    for t in target.unique().tolist():
        group = dropped[target == t].tolist()
        acc_gu, acc_dn, tot = gu[t].float() * w[t], dn[t].float() * w[t], w[t]
        for d in group:
            g_d, d_d = align_neurons(gu[t], dn[t], gu[d], dn[d]) if aligned else (gu[d], dn[d])
            acc_gu += g_d.float() * w[d]
            acc_dn += d_d.float() * w[d]
            tot += w[d]
        gu[t], dn[t] = (acc_gu / tot).to(gu.dtype), (acc_dn / tot).to(dn.dtype)
        if remap:
            for d in group:
                gu[d], dn[d] = gu[t], dn[t]
    state = dict(state)
    state[gu_name], state[dn_name] = gu, dn
    return state if remap else prune_layer_state(state, keep)


CRITERIA = ("reap", "freq", "gate", "ean", "prob", "random")


def expert_scores(stats, criterion, seed=0):
    """Importance score [n_layers, n_experts] from the sums recorded in pass 1.

    reap   mean of g_j(x)*||E_j(x)|| over tokens routed to j  (REAP; usually best)
    freq   how often j is in the top-k                         (simplest baseline)
    gate   summed routing weight
    ean    summed output norm, ignoring the router weight
    prob   summed softmax probability over all tokens
    random lower bound - shows how much the criterion matters
    """
    if criterion == "reap":
        return stats["reap_sum"] / stats["freq"].clamp(min=1)
    if criterion == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.rand(stats["freq"].shape, generator=g, dtype=torch.float64)
    return stats[{"freq": "freq", "gate": "gate_sum", "ean": "ean_sum", "prob": "prob_sum"}[criterion]]


def select_experts(stats, keep, criterion):
    """Per layer, the sorted indices of the `keep` highest-scoring experts.
    `keep` is one K for every layer, or a list with one K per layer."""
    scores = expert_scores(stats, criterion)
    ks = keep if isinstance(keep, (list, tuple)) else [keep] * len(scores)
    return [row.topk(k).indices.sort().values for row, k in zip(scores, ks)]


def build_layer(text_config, i, state, device="cuda"):
    n_experts = state["mlp.gate.weight"].shape[0]
    if n_experts != text_config.num_experts:
        text_config = copy.deepcopy(text_config)
        text_config.num_experts = n_experts
    with torch.device("meta"):
        layer = Qwen3_5MoeDecoderLayer(text_config, i)
    state = {k: v.to(device, non_blocking=True) for k, v in state.items()}
    layer.load_state_dict(state, strict=True, assign=True)
    layer.requires_grad_(False)  # inference only; 06_heal.py re-enables what it trains
    return layer.eval()


class ExpertStats:
    """Per-expert accumulators for one layer (float64 on GPU)."""

    KEYS = ("freq", "gate_sum", "reap_sum", "ean_sum", "prob_sum")

    def __init__(self, n_experts, n_sources, device="cuda"):
        self.n_sources = n_sources
        for k in self.KEYS:
            setattr(self, k, torch.zeros(n_experts, dtype=torch.float64, device=device))
        self.freq_by_source = torch.zeros(n_sources, n_experts, dtype=torch.float64, device=device)
        self.source = None  # LongTensor [tokens] set by the caller per micro-batch

    def to_cpu(self):
        d = {k: getattr(self, k).cpu() for k in self.KEYS}
        d["freq_by_source"] = self.freq_by_source.cpu()
        return d


def moe_forward(mlp, hidden_states, stats=None):
    """Qwen3.5 sparse-MoE block: softmax router -> top-k -> renormalize, plus a
    sigmoid-gated shared expert.  Same math as Qwen3_5MoeSparseMoeBlock.

    Tokens are sorted by expert so every expert does one dense matmul over
    all its tokens.  If `stats` is given we record, per expert j:
      freq      number of tokens routed to j
      gate_sum  sum of routing weights g_j(x)
      ean_sum   sum of ||E_j(x)||            (expert activation norm)
      reap_sum  sum of g_j(x) * ||E_j(x)||   (REAP, Cerebras 2025)
      prob_sum  sum of softmax prob over *all* tokens (soft usage)
    """
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    gate_up, down = mlp.experts.gate_up_proj, mlp.experts.down_proj
    n_experts, top_k = gate_up.shape[0], mlp.gate.top_k

    logits = F.linear(x, mlp.gate.weight)
    probs = logits.float().softmax(dim=-1)
    top_w, top_i = probs.topk(top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)

    flat_e = top_i.flatten()
    order = flat_e.argsort(stable=True)
    token = order // top_k
    weight = top_w.flatten()[order]
    counts = torch.bincount(flat_e, minlength=n_experts).tolist()

    xs = x[token]
    ys = torch.empty_like(xs)
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        seg = slice(start, start + c)
        g, u = F.linear(xs[seg], gate_up[e]).chunk(2, dim=-1)
        ys[seg] = F.linear(F.silu(g) * u, down[e])
        start += c

    if stats is not None:
        with torch.no_grad():
            e_sorted = flat_e[order]
            norm = ys.float().norm(dim=-1).double()
            w = weight.double()
            stats.freq += torch.bincount(e_sorted, minlength=n_experts).double()
            stats.gate_sum.index_add_(0, e_sorted, w)
            stats.ean_sum.index_add_(0, e_sorted, norm)
            stats.reap_sum.index_add_(0, e_sorted, w * norm)
            stats.prob_sum += probs.sum(0).double()
            if stats.source is not None:
                src = stats.source[token]
                stats.freq_by_source.view(-1).index_add_(
                    0, src * n_experts + e_sorted, torch.ones_like(w))

    # Put each (token, slot) result back in place and sum each token's top_k slots in a fixed order.
    # (index_add_ into `out` would use GPU atomics: the add order, and so the bf16 rounding, changes from
    # run to run, which made identical runs differ by up to 0.15% perplexity per domain; see 16_ablate.py.)
    contrib = torch.empty_like(ys)
    contrib[order] = ys * weight[:, None].to(ys.dtype)
    out = contrib.view(-1, top_k, shape[-1]).sum(dim=1)
    shared = mlp.shared_expert(x) * torch.sigmoid(mlp.shared_expert_gate(x))
    return (out + shared).reshape(shape)


def patch_moe(layer, stats=None):
    """Route a decoder layer's MoE block through `moe_forward`.

    The override holds only a weak reference to the block: storing a closure
    over the module on the module itself is a reference cycle, which keeps each
    1.7 GB layer alive on the GPU until Python's cycle collector happens to run.
    """
    ref = weakref.ref(layer.mlp)
    layer.mlp.forward = lambda x: moe_forward(ref(), x, stats)


class Streamer:
    """Holds the hidden states of a token set and pushes them layer by layer."""

    def __init__(self, ckpt, tokens, micro_batch=16, device="cuda"):
        self.ckpt, self.tokens, self.mb, self.device = ckpt, tokens, micro_batch, device
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
        self.rotary = Qwen3_5MoeTextRotaryEmbedding(ckpt.text_config).to(device)
        self.hidden = None

    def batches(self):
        for s in range(0, len(self.tokens), self.mb):
            yield slice(s, min(s + self.mb, len(self.tokens)))

    def rope(self, h):
        pos = torch.arange(h.shape[1], device=self.device)[None].expand(h.shape[0], -1)
        return self.rotary(h, pos), pos

    @torch.no_grad()
    def embed(self):
        w = self.ckpt.tensors([TEXT + "embed_tokens.weight"], self.device)[TEXT + "embed_tokens.weight"]
        N, L = self.tokens.shape
        self.hidden = torch.empty(N, L, w.shape[1], dtype=w.dtype, pin_memory=True)
        for b in self.batches():
            self.hidden[b] = F.embedding(self.tokens[b].to(self.device).long(), w).cpu()
        del w

    @torch.no_grad()
    def run_layer(self, layer, hidden=None, before_batch=None):
        """Run every micro-batch of `hidden` (default: self.hidden) through `layer`, in place."""
        hidden = self.hidden if hidden is None else hidden
        for b in self.batches():
            h = hidden[b].to(self.device, non_blocking=True)
            if before_batch is not None:
                before_batch(b)
            pe, pos = self.rope(h)
            hidden[b] = layer(h, position_embeddings=pe, attention_mask=None, position_ids=pos).cpu()

    def layers(self, transform_state=None, start=0):
        """Yield (i, layer on GPU) for layers start..n-1, prefetching layer i+1 from disk meanwhile."""
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(self.ckpt.layer_state, start)
            for i in range(start, self.ckpt.n_layers):
                state = fut.result()
                if i + 1 < self.ckpt.n_layers:
                    fut = pool.submit(self.ckpt.layer_state, i + 1)
                if transform_state is not None:
                    state = transform_state(i, state)
                layer = build_layer(self.ckpt.text_config, i, state, self.device)
                yield i, layer
                del layer, state
                torch.cuda.empty_cache()

    @torch.no_grad()
    def nll(self, chunk=1024, per_token=False):
        """Per-sequence summed next-token NLL and token counts, from final hidden states.
        per_token=True instead returns every token's loss, float32 [N, L-1] (loss t predicts token t+1)."""
        names = [TEXT + "norm.weight", "lm_head.weight"]
        t = self.ckpt.tensors(names, self.device)
        norm = Qwen3_5MoeRMSNorm(t[names[0]].shape[0], eps=self.ckpt.text_config.rms_norm_eps).to(self.device)
        norm.weight.data.copy_(t[names[0]])
        head = t[names[1]]
        N, L = self.tokens.shape
        total = torch.zeros(N, dtype=torch.float64)
        tok_loss = torch.zeros(N, L - 1) if per_token else None
        for n in range(N):
            h = norm(self.hidden[n, :-1].to(self.device))
            target = self.tokens[n, 1:].to(self.device).long()
            s = 0.0
            for c in range(0, L - 1, chunk):
                logits = F.linear(h[c:c + chunk], head).float()
                loss = F.cross_entropy(logits, target[c:c + chunk], reduction="none")
                if per_token:
                    tok_loss[n, c:c + chunk] = loss.cpu()
                s += loss.sum().item()
            total[n] = s
        if per_token:
            return tok_loss
        return total, torch.full((N,), L - 1)


def ppl_report(nll, count, sources):
    """Perplexity per data source and overall."""
    rows = {}
    for src in dict.fromkeys(sources):
        idx = [i for i, s in enumerate(sources) if s == src]
        rows[src] = float(torch.exp(nll[idx].sum() / count[idx].sum()))
    rows["all"] = float(torch.exp(nll.sum() / count.sum()))
    return rows
