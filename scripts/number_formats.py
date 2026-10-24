"""Real numbers for concept 14 (number formats): what the weights and activations of this model look like.

  1. A few real weights (layer 20, expert 241), with their BF16 bit patterns and what FP16 / FP32 store.
  2. The spread of weight magnitudes in one expert tensor (histogram of log2|w|): the range a format must cover.
  3. The largest hidden-state value after each layer on 8 calibration sequences: if it passed 65,504,
     FP16 would overflow to infinity; BF16 has the same range as FP32.
  4. Addition order in BF16: (a + b) + c against a + (b + c) on real values (why bug 4 happened).

Writes results/number_formats.json (read by learn/14-numbers.html).

    .venv/bin/python -I scripts/number_formats.py          # ~1 min (one streamed pass over 8 sequences)
"""
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from moe_stream import Checkpoint, Streamer, patch_moe  # noqa: E402


def bits(x, fmt):
    """Bit string of float x stored as fmt ('bf16', 'fp16', 'fp32')."""
    if fmt == "fp16":
        return format(struct.unpack("<H", struct.pack("<e", x))[0], "016b")
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return format(u >> 16, "016b") if fmt == "bf16" else format(u, "032b")


def main():
    ckpt = Checkpoint("models/Qwen3.5-35B-A3B")
    st = ckpt.layer_state(20)
    gu = st["mlp.experts.gate_up_proj"][241]  # [1024, 2048] bf16
    sample = gu[0, :8].float().tolist()
    weights = [{"bf16": v, "bf16_bits": bits(v, "bf16"), "fp16": float(torch.tensor(v).half()), "fp16_bits": bits(v, "fp16"),
                "fp32_bits": bits(v, "fp32")} for v in sample]
    a = gu.float().abs().flatten()
    a = a[a > 0]
    lg = torch.log2(a)
    hist = torch.histc(lg, bins=30, min=-24, max=6).tolist()
    weight_range = {"min_abs": float(a.min()), "max_abs": float(a.max()), "median_abs": float(a.median()),
                    "log2_hist": hist, "log2_range": [-24, 6], "n": a.numel(),
                    "below_fp16_normal": float((a < 2 ** -14).double().mean())}

    data = torch.load("data/calib.pt")
    idx = torch.linspace(0, len(data["tokens"]) - 1, 8).long()
    s = Streamer(ckpt, data["tokens"][idx], micro_batch=8)
    s.embed()
    act = [float(s.hidden.float().abs().max())]
    for i, layer in s.layers():
        patch_moe(layer)
        s.run_layer(layer)
        act.append(float(s.hidden.float().abs().max()))
        print(f"\rlayer {i + 1}/40 max |h| {act[-1]:.0f}", end="", flush=True)
    print()

    # addition order: three real hidden-state values, summed in two orders, in bf16 and fp32
    h = s.hidden[0, 5].float()
    vals = h[h.abs().argsort(descending=True)][[0, 400, 401]].tolist()  # one large value and two small ones
    x, y, z = (torch.tensor(v, dtype=torch.bfloat16) for v in vals)
    order = {"values": vals, "bf16_left": float((x + y) + z), "bf16_right": float(x + (y + z)),
             "fp32_left": float((torch.tensor(vals[0]) + vals[1]) + vals[2]), "fp32_right": float(vals[0] + (torch.tensor(vals[1]) + vals[2]))}

    out = {"weights": weights, "weight_range": weight_range, "max_abs_hidden_per_layer": act, "addition_order": order,
           "fp16_max": 65504.0, "bf16_max": float(torch.finfo(torch.bfloat16).max)}
    Path("results/number_formats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "weight_range"}, indent=1)[:1500])
    print("weights: min", weight_range["min_abs"], "max", weight_range["max_abs"], "median", weight_range["median_abs"],
          "below fp16 normal", weight_range["below_fp16_normal"])


if __name__ == "__main__":
    main()
