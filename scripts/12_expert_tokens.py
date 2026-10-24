"""Look inside the experts: which tokens does each expert handle?

One streamed pass of the original model over the calibration set records the top-8 experts of every
token in every layer. Then, per expert, it finds the tokens it is most specialised in, ranked by
LIFT: how much more often a token appears among the expert's tokens than in the text overall.
(Ranking by raw count would give "the" and "," for nearly every expert.)

    .venv/bin/python -I scripts/12_expert_tokens.py              # GPU pass + analysis
    .venv/bin/python -I scripts/12_expert_tokens.py --analyse    # analysis only, from the saved routing

Outputs: results/expert_routing.pt (top-8 ids per token per layer, int16, ~340 MB) and
learn/g-experts-data.js for the appendix G page.
"""
import argparse
import json
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import Checkpoint, Streamer, moe_forward  # noqa: E402


def record_routing(model, data, micro_batch):
    ckpt = Checkpoint(model)
    tokens = data["tokens"]
    N, L = tokens.shape
    routing = torch.empty(ckpt.n_layers, N * L, 8, dtype=torch.int16)
    st = Streamer(ckpt, tokens, micro_batch)
    st.embed()
    t0 = time.time()
    for i, layer in st.layers():
        ref = weakref.ref(layer.mlp)
        pos = {"k": 0}

        def forward(x, ref=ref, i=i, pos=pos):
            mlp = ref()
            flat = x.reshape(-1, x.shape[-1])
            top = F.linear(flat, mlp.gate.weight).float().softmax(-1).topk(mlp.gate.top_k, dim=-1).indices
            routing[i, pos["k"]:pos["k"] + len(flat)] = top.to(torch.int16).cpu()
            pos["k"] += len(flat)
            return moe_forward(mlp, x)
        layer.mlp.forward = forward
        st.run_layer(layer)
        print(f"\rlayer {i + 1}/{ckpt.n_layers}  [{time.time() - t0:.0f}s]", end="", flush=True)
    print()
    return routing


def analyse(routing, tokens, tok, top_n=10, min_count=20, examples=3):
    flat_tok = tokens.reshape(-1).long()
    V = int(flat_tok.max()) + 1
    base = torch.bincount(flat_tok, minlength=V).double()
    p_base = base / base.sum()
    n_layers, n_tokens, k = routing.shape
    out = []
    for i in range(n_layers):
        e = routing[i].long().reshape(-1)                     # expert of each (token, slot)
        t = flat_tok.repeat_interleave(k)                      # token of each (token, slot)
        key = e * V + t
        uniq, cnt = torch.unique(key, return_counts=True)
        ex, tk = uniq // V, uniq % V
        per_expert_total = torch.bincount(e, minlength=256).double()
        lift = (cnt.double() / per_expert_total[ex]) / p_base[tk]
        layer = []
        for j in range(256):
            m = (ex == j) & (cnt >= min_count)
            if m.sum() == 0:
                layer.append({"n": int(per_expert_total[j]), "top": []})
                continue
            order = lift[m].argsort(descending=True)[:top_n]
            ids, c, lf = tk[m][order], cnt[m][order], lift[m][order]
            layer.append({"n": int(per_expert_total[j]),
                          "top": [[tok.decode([int(a)]), int(b), round(float(d), 1)] for a, b, d in zip(ids, c, lf)]})
        # example contexts for every expert: occurrences of its most characteristic tokens (spread
        # through the text), so the examples show the speciality rather than its everyday tokens
        top_ids = {j: [int(tk[(ex == j) & (cnt >= min_count)][lift[(ex == j) & (cnt >= min_count)].argsort(descending=True)[:3]][q])
                       for q in range(min(3, int(((ex == j) & (cnt >= min_count)).sum())))] for j in range(256)}
        for j in range(256):
            if not top_ids[j]:
                continue
            routed = (routing[i] == j).any(-1)
            pos = (routed & torch.isin(flat_tok, torch.tensor(top_ids[j]))).nonzero().squeeze(-1)
            if len(pos) == 0:
                continue
            pick = pos[torch.linspace(0, len(pos) - 1, min(examples, len(pos))).long()]
            ctx = []
            for p in pick.tolist():
                a, b = max(0, p - 8), min(n_tokens, p + 4)
                ctx.append([tok.decode(flat_tok[a:p].tolist()), tok.decode([int(flat_tok[p])]), tok.decode(flat_tok[p + 1:b].tolist())])
            layer[j]["examples"] = ctx
        out.append(layer)
        print(f"\ranalysed layer {i + 1}/{n_layers}", end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--data", default="data/calib.pt")
    ap.add_argument("--routing", default="results/expert_routing.pt")
    ap.add_argument("--analyse", action="store_true", help="skip the GPU pass, use the saved routing")
    ap.add_argument("--micro-batch", type=int, default=16)
    args = ap.parse_args()
    data = torch.load(args.data)
    if not args.analyse:
        torch.save(record_routing(args.model, data, args.micro_batch), args.routing)
    routing = torch.load(args.routing)
    tok = AutoTokenizer.from_pretrained(args.model)
    experts = analyse(routing, data["tokens"], tok)
    Path("learn/g-experts-data.js").write_text(
        "/* Generated by scripts/12_expert_tokens.py: per layer, per expert, the tokens with the highest lift\n"
        "   [token, count, lift] and example contexts of its top tokens. */\nconst EXPERTS = " + json.dumps(experts, ensure_ascii=False) + ";\n")
    print("wrote learn/g-experts-data.js")


if __name__ == "__main__":
    main()
