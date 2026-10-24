"""How alike are the experts of a layer? Needed to merge experts instead of deleting them.

Merging puts each dropped expert into the surviving expert that does the most similar job. "Similar job"
is measured on real inputs: stream a slice of the calibration set through the original model, take
--n-tokens MoE inputs per layer, run ALL 256 experts on them (not just the 8 the router picks), and
compare outputs:

  output   cosine similarity of two experts' outputs over the same tokens (functional similarity;
           close to what HC-SMoE clusters on)
  router   cosine similarity of the two router rows (do they get picked for the same inputs?
           close to M-SMoE's router-logit similarity)

Writes stats/similarity.pt {"output": [40, 256, 256], "router": [40, 256, 256]}, then:

    .venv/bin/python -I scripts/05_eval.py --stats stats/calib.pt --keep 128 --merge aligned
"""
import argparse
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import Checkpoint, Streamer, moe_forward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--data", default="data/calib.pt")
    ap.add_argument("--n-seqs", type=int, default=32, help="calibration sequences streamed (spread over sources)")
    ap.add_argument("--n-tokens", type=int, default=1024, help="MoE inputs per layer on which all experts run")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--out", default="stats/similarity.pt")
    args = ap.parse_args()

    ckpt = Checkpoint(args.model)
    data = torch.load(args.data)
    ids = torch.linspace(0, len(data["tokens"]) - 1, args.n_seqs).long()
    tokens = data["tokens"][ids]
    N, L = tokens.shape
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(N * L, generator=g)[:args.n_tokens].sort().values  # flat positions sampled per layer

    st = Streamer(ckpt, tokens, args.micro_batch)
    st.embed()
    out_sim, router_sim = [], []
    cur, grabbed = {}, []
    t0 = time.time()
    for i, layer in st.layers():
        ref = weakref.ref(layer.mlp)

        def fwd(x, ref=ref):  # keep the MoE inputs of the sampled positions, then run the block as usual
            b = cur["b"]
            lo, hi = b.start * L, b.stop * L
            sel = pick[(pick >= lo) & (pick < hi)] - lo
            grabbed.append(x.reshape(-1, x.shape[-1])[sel.to(x.device)])
            return moe_forward(ref(), x)
        layer.mlp.forward = fwd
        grabbed.clear()
        st.run_layer(layer, before_batch=lambda b: cur.__setitem__("b", b))
        x = torch.cat(grabbed)  # [n_tokens, 2048]
        with torch.no_grad():
            gu, dn = layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj
            y = []
            for e in range(gu.shape[0]):
                a, u = F.linear(x, gu[e]).chunk(2, dim=-1)
                y.append(F.normalize(F.linear(F.silu(a) * u, dn[e]).float().flatten(), dim=0))
            y = torch.stack(y)  # [E, n_tokens * 2048], unit length
            out_sim.append((y @ y.T).cpu())
            r = F.normalize(layer.mlp.gate.weight.float(), dim=-1)
            router_sim.append((r @ r.T).cpu())
        del y, x
        s = out_sim[-1].clone().fill_diagonal_(-1)
        print(f"layer {i:2d}: most similar pair {s.max():.3f}, median expert's best match {s.max(1).values.median():.3f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"output": torch.stack(out_sim), "router": torch.stack(router_sim), "n_tokens": args.n_tokens}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
