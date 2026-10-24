"""Look at the expert statistics from 03_expert_stats.py (CPU only).

  - how unevenly experts are used per layer (would pruning even be possible?)
  - how much the criteria disagree on which experts to keep
  - domain specialists: experts mostly used by one calibration source - these
    are the ones a narrow calibration set would wrongly delete
  - how much routing mass the kept experts cover at a given K
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import select_experts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="stats/calib.pt")
    ap.add_argument("--keep", type=int, nargs="+", default=[128, 64, 48])
    args = ap.parse_args()
    s = torch.load(args.stats)
    freq = s["freq"]                     # [layers, experts]
    n_layers, n_experts = freq.shape
    share = freq / freq.sum(1, keepdim=True)

    print("== usage concentration: share of routed tokens taken by the top-N experts of each layer")
    ranked = share.sort(1, descending=True).values.cumsum(1)
    for n in (8, 32, 64, 128):
        col = ranked[:, n - 1]
        print(f"  top-{n:3d}: mean {col.mean():.1%}  min {col.min():.1%} (layer {col.argmin()})  max {col.max():.1%} (layer {col.argmax()})")

    print("\n== routing mass kept: fraction of top-8 assignments that land on a kept expert (REAP selection)")
    for k in args.keep:
        keep = select_experts(s, k, "reap")
        kept = torch.stack([share[i, keep[i]].sum() for i in range(n_layers)])
        print(f"  K={k:3d}: mean {kept.mean():.1%}  worst layer {kept.min():.1%} (layer {kept.argmin()})")

    print("\n== criterion agreement at each K: overlap of kept sets with REAP")
    for k in args.keep:
        ref = select_experts(s, k, "reap")
        row = []
        for c in ("freq", "gate", "ean", "random"):
            other = select_experts(s, k, c)
            ov = sum(len(set(ref[i].tolist()) & set(other[i].tolist())) for i in range(n_layers)) / (k * n_layers)
            row.append(f"{c} {ov:.0%}")
        print(f"  K={k:3d}: " + "  ".join(row))

    print("\n== domain specialists: experts getting >60% of their tokens from one source")
    by_src = s["freq_by_source"]          # [layers, sources, experts]
    frac = by_src / by_src.sum(1, keepdim=True).clamp(min=1)
    for j, name in enumerate(s["sources"]):
        spec = (frac[:, j] > 0.6)
        print(f"  {name:12s}: {spec.sum().item():5d} experts total "
              f"(per layer: early {spec[:10].sum().item() / 10:.1f}, late {spec[-10:].sum().item() / 10:.1f})")
    for k in args.keep:
        keep = select_experts(s, k, "reap")
        lost = {name: 0 for name in s["sources"]}
        for i in range(n_layers):
            dropped = torch.ones(n_experts, dtype=torch.bool)
            dropped[keep[i]] = False
            for j, name in enumerate(s["sources"]):
                lost[name] += (dropped & (frac[i, j] > 0.6)).sum().item()
        print(f"  K={k:3d} drops specialists: " + "  ".join(f"{n} {v}" for n, v in lost.items()))


if __name__ == "__main__":
    main()
