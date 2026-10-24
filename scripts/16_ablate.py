"""Causal test of expert specialities: delete ONE expert and see where the loss goes up.

Appendix G found experts whose tokens look like a topic ("layer 20 expert 241 receives membrane, drug,
blood, cells"). That is correlation. The causal question is: does the model NEED that expert for that
topic? Test it the way pruning would: delete just that one expert (its router row and weights; its
tokens re-route to their 9th choice) and measure the loss on texts of different topics (data/domains.pt,
02_calib_data.py --preset domains). Biology, chemistry and physics are written in the same style, as are
algebra, geometry and number theory, so a difference between them is the topic, not the style.

Two controls tell signal from noise:
  repeat   the unablated model again from layer 5: GPU float sums are not bit-exact, this is the floor
  random   a few random experts of the same layers: what deleting any one expert costs

Per ablation, the loss change is reported per domain, and split into tokens the expert actually
handled in the unablated model (one of its top 8) against all others; for code, into tokens inside
'#' comments against the rest. Results: results/ablation.json (+ per-token deltas in ablation_tokens.pt).

To save time the unablated hidden states at the input of each tested layer are kept in RAM, so a
test of layer L only recomputes layers L..39.

--teams N: one expert is a small part of a token's computation (8 experts x 40 layers, with 255 others
to re-route to), so its effect may be tiny either way. The team test deletes, for each domain, N experts
PER LAYER (40 x N in total) that are both specific to it (picked at least 2x more often on that domain
than on average: lift >= 2) and carry the most of its traffic (largest share of the domain's picks), and
compares with N random experts per layer. Ranking by lift alone would pick experts for rare words that
handle under 1% of a domain's tokens; the single-expert test showed those barely matter.
Teams are chosen on the first half of each domain's sequences and measured on the second half, so the
choice is not fitted to the text it is judged on. Results: results/ablation_teams.json.

    .venv/bin/python -I scripts/16_ablate.py                          # 16 single experts, ~12 min
    .venv/bin/python -I scripts/16_ablate.py --no-single --teams 4 16  # 20 domain teams, ~24 min
"""
import argparse
import json
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import Checkpoint, Streamer, moe_forward, patch_moe, prune_layer_state  # noqa: E402

NAMED = [  # (layer, expert, what appendix G's token lists suggest)
    (5, 39, "algebra"), (5, 50, "code comments"),
    (20, 241, "biology and medicine"), (20, 97, "economics and environment"), (20, 109, "people and relationships"),
    (35, 204, "think tags and LaTeX"), (36, 173, "generalist, no speciality"),
]


def comment_mask(tok, tokens):
    """True for tokens inside a Python '#' comment (from the '#' to the end of the line). Approximate:
    a '#' inside a string also counts, but that is rare in this data."""
    mask = torch.zeros(tokens.shape, dtype=torch.bool)
    for n, row in enumerate(tokens.tolist()):
        inside = False
        for t, piece in enumerate(tok.convert_ids_to_tokens(row)):
            text = tok.convert_tokens_to_string([piece])
            if "#" in text:
                inside = True
            mask[n, t] = inside
            if "\n" in text:
                inside = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--data", default="data/domains.pt")
    ap.add_argument("--random", type=int, default=3, help="random control experts per tested layer")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--out", default="results/ablation.json")
    ap.add_argument("--no-single", action="store_true", help="skip the single-expert tests")
    ap.add_argument("--teams", type=int, nargs="*", default=[], help="team sizes (experts per layer) to test")
    ap.add_argument("--team-seeds", type=int, default=2, help="random teams of each size")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ckpt = Checkpoint(args.model)
    data = torch.load(args.data)
    sources = data["source"]
    N, L = data["tokens"].shape

    g = torch.Generator().manual_seed(0)
    layers = sorted({l for l, _, _ in NAMED if l in (5, 20, 35)})
    controls = []
    for l in layers:
        named = {e for ll, e, _ in NAMED if ll == l}
        pool = [e for e in torch.randperm(ckpt.n_experts, generator=g).tolist() if e not in named]
        controls += [(l, e, "random control") for e in pool[:args.random]]
    tests = [] if args.no_single else NAMED + controls
    test_layers = sorted({l for l, _, _ in tests} | ({0} if args.teams else set()))

    # which tokens did each tested expert handle in the unablated model? record during the baseline pass
    routed = {(l, e): torch.zeros(N, L, dtype=torch.bool) for l, e, _ in tests}
    domains = list(dict.fromkeys(sources))
    rows_of = {s: [n for n, x in enumerate(sources) if x == s] for s in domains}
    half_a = {s: r[:len(r) // 2] for s, r in rows_of.items()}  # teams are chosen here...
    half_b = {s: r[len(r) // 2:] for s, r in rows_of.items()}  # ...and measured here
    seq_dom = torch.tensor([domains.index(x) for x in sources])
    in_a = torch.zeros(N, dtype=torch.bool)
    in_a[[n for r in half_a.values() for n in r]] = True
    picks = torch.zeros(ckpt.n_layers, ckpt.n_experts, len(domains), dtype=torch.float64)  # routed slots, half A
    cur = {}

    def recorder(layer_idx, mlp):
        ref = weakref.ref(mlp)

        def fwd(x):
            m = ref()
            top = F.linear(x.reshape(-1, x.shape[-1]), m.gate.weight).topk(m.gate.top_k, dim=-1).indices
            top = top.view(x.shape[0], x.shape[1], -1).cpu()
            for (l, e), mask in routed.items():
                if l == layer_idx:
                    mask[cur["b"]] = (top == e).any(-1)
            if args.teams:
                b = cur["b"]
                rows = torch.arange(b.start, b.stop)[in_a[b]]
                if len(rows):
                    t = top[in_a[b]].reshape(len(rows), -1)  # [rows, L*k]
                    d = seq_dom[rows][:, None].expand_as(t)
                    picks[layer_idx].view(-1).index_add_(0, (t * len(domains) + d).flatten(), torch.ones(t.numel(), dtype=torch.float64))
            return moe_forward(m, x)
        mlp.forward = fwd

    st = Streamer(ckpt, data["tokens"], args.micro_batch)
    st.embed()
    t0 = time.time()
    saved = {}
    for i, layer in st.layers():
        if i in test_layers:
            saved[i] = st.hidden.clone()
        if i in test_layers or args.teams:
            recorder(i, layer.mlp)
        else:
            patch_moe(layer)
        st.run_layer(layer, before_batch=lambda b: cur.__setitem__("b", b))
    base = st.nll(per_token=True)
    print(f"baseline pass [{time.time() - t0:.0f}s]", flush=True)

    src_idx = {s: [n for n, x in enumerate(sources) if x == s] for s in dict.fromkeys(sources)}
    code_rows = src_idx.get("code", [])
    in_comment = torch.zeros(N, L, dtype=torch.bool)
    if code_rows:
        in_comment[code_rows] = comment_mask(tok, data["tokens"][code_rows])
    in_comment = in_comment[:, 1:]  # loss t predicts token t+1

    def run_from(l, transform=None):
        st.hidden = saved[l].clone()
        for i, layer in st.layers(transform, start=l):
            patch_moe(layer)
            st.run_layer(layer)
        return st.nll(per_token=True)

    def summarise(name, l, e, loss):
        d = loss - base  # per-token change in loss (nats); >0 = worse
        row = {"layer": l, "expert": e, "reading": name,
               "ppl_change_pct": {s: round(100 * (float(torch.exp(d[idx].mean())) - 1), 4) for s, idx in src_idx.items()}}
        row["ppl_change_pct"]["all"] = round(100 * (float(torch.exp(d.mean())) - 1), 4)
        if e is not None:
            m = routed[(l, e)][:, :-1]
            row["tokens_routed"] = int(m.sum())
            row["routed_share_pct"] = {s: round(100 * float(m[idx].float().mean()), 3) for s, idx in src_idx.items()}
            row["delta_routed"] = round(float(d[m].mean()), 5) if m.any() else None
            row["delta_other"] = round(float(d[~m].mean()), 5)
        if code_rows:
            c, cm = d[code_rows], in_comment[code_rows]
            row["code_comment_delta"] = round(float(c[cm].mean()), 5)
            row["code_other_delta"] = round(float(c[~cm].mean()), 5)
        return row, d.half()

    if args.teams:
        teams(args, ckpt, st, saved, base, picks, domains, half_b, run_from)
    if args.no_single:
        return

    rows, deltas = [], {}
    t1 = time.time()
    row, d = summarise("repeat (no deletion)", test_layers[0], None, run_from(test_layers[0]))
    rows.append(row)
    print(f"repeat: {row['ppl_change_pct']}  [{time.time() - t1:.0f}s]", flush=True)
    for l, e, name in tests:
        t1 = time.time()
        keep = torch.tensor([k for k in range(ckpt.n_experts) if k != e])
        loss = run_from(l, lambda i, state, l=l, keep=keep: prune_layer_state(state, keep) if i == l else state)
        row, deltas[f"{l}.{e}"] = summarise(name, l, e, loss)
        rows.append(row)
        print(f"{l:2d}.{e:<3d} {name:28s} {row['ppl_change_pct']}  routed {row['delta_routed']} other {row['delta_other']}"
              f"  [{time.time() - t1:.0f}s]", flush=True)
        Path(args.out).write_text(json.dumps({"n_tokens_per_domain": len(src_idx[sources[0]]) * (L - 1), "rows": rows}, indent=1))
    torch.save({"deltas": deltas, "routed": {f"{l}.{e}": m for (l, e), m in routed.items()}, "base": base.half(),
                "source": sources}, Path(args.out).with_name("ablation_tokens.pt"))
    print(f"wrote {args.out}  [{time.time() - t0:.0f}s total]")


def teams(args, ckpt, st, saved, base, picks, domains, half_b, run_from):
    """Delete each domain's most specific experts (N per layer, chosen on half A), measure on half B."""
    share_dom = picks / picks.sum(1, keepdim=True)                    # [layer, expert, domain]: share of the domain's picks
    share_all = picks.sum(2, keepdim=True) / picks.sum((1, 2), keepdim=True)  # [layer, expert, 1]: share of all picks
    lift = share_dom / share_all.clamp(min=1e-12)
    score = torch.where((lift >= 2) & (picks >= 20), share_dom, torch.full_like(share_dom, -1.0))
    g = torch.Generator().manual_seed(1)
    out = []

    def measure(name, team):
        t1 = time.time()
        carried = {s: round(100 * float(sum(share_dom[i, team[i], k].sum() for i in range(ckpt.n_layers)) / ckpt.n_layers), 2)
                   for k, s in enumerate(domains)}  # % of each domain's picks (half A) the team handled, per layer
        keep = [torch.tensor([e for e in range(ckpt.n_experts) if e not in set(team[i])]) for i in range(ckpt.n_layers)]
        d = run_from(0, lambda i, state: prune_layer_state(state, keep[i])) - base
        row = {"team": name, "per_layer": len(team[0]), "share_of_picks_pct": carried,
               "ppl_change_pct": {s: round(100 * (float(torch.exp(d[r].mean())) - 1), 3) for s, r in half_b.items()}}
        out.append(row)
        print(f"team {name:22s} {row['ppl_change_pct']}  [{time.time() - t1:.0f}s]", flush=True)
        Path(args.out).with_name("ablation_teams.json").write_text(json.dumps(out, indent=1))

    for n in args.teams:
        for k, dom in enumerate(domains):
            measure(f"{dom} x{n}", [score[i, :, k].topk(n).indices.tolist() for i in range(ckpt.n_layers)])
        for s in range(args.team_seeds):
            measure(f"random x{n} #{s + 1}", [torch.randperm(ckpt.n_experts, generator=g)[:n].tolist() for _ in range(ckpt.n_layers)])


if __name__ == "__main__":
    main()
