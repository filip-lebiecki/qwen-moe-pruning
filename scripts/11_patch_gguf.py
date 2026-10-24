"""Step 3c: put distilled routers into a GGUF without re-converting or re-quantizing.

Distillation changed only the routers (blk.N.ffn_gate_inp.weight, 128 x 2048) and the shared-expert
gates (blk.N.ffn_gate_inp_shexp.weight). llama.cpp keeps both unquantized (F32), so they can be
overwritten in a copy of the existing Q4_K_M file; every quantized expert stays byte-identical.

Safety check first: the GGUF's current routers must equal the untrained ones from 10_distill.py
(routers_init.pt). That proves both use the same experts in the same order.

  .venv/bin/python -I scripts/11_patch_gguf.py gguf/q35-reap-k128/q35-reap-k128-Q4_K_M.gguf \\
      results/distill/routers.pt gguf/q35-distill-k128/q35-distill-k128-Q4_K_M.gguf
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("routers")
    ap.add_argument("dst")
    ap.add_argument("--init", default="results/distill/routers_init.pt")
    args = ap.parse_args()

    init, new = torch.load(args.init), torch.load(args.routers)
    src = GGUFReader(args.src)
    t = {x.name: x for x in src.tensors}
    worst = 0.0
    for i, g in enumerate(init["gate"]):
        have = np.asarray(t[f"blk.{i}.ffn_gate_inp.weight"].data)
        worst = max(worst, float(np.abs(have - g.to(torch.bfloat16).float().numpy()).max()))
    assert worst == 0.0, f"GGUF routers differ from routers_init.pt (max diff {worst}); wrong model or expert order"
    print("check passed: the GGUF holds exactly the untrained routers")

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src, args.dst)
    dst = GGUFReader(args.dst, "r+")
    t = {x.name: x for x in dst.tensors}
    for i, (g, s) in enumerate(zip(new["gate"], new["shexp"])):
        gate, shexp = t[f"blk.{i}.ffn_gate_inp.weight"], t[f"blk.{i}.ffn_gate_inp_shexp.weight"]
        gate.data[...] = g.to(torch.bfloat16).float().numpy().reshape(gate.data.shape)
        shexp.data[...] = s.to(torch.bfloat16).float().numpy().reshape(shexp.data.shape)
    dst.data.flush()
    moved = max(float((a - b).abs().max()) for a, b in zip(new["gate"], init["gate"]))
    print(f"wrote {args.dst}: 40 routers + 40 shared-expert gates replaced (largest weight change {moved:.4f})")


if __name__ == "__main__":
    main()
