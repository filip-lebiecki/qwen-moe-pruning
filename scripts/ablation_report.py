"""Step 5 report: single-expert ablations (16_ablate.py) and prune vs merge (05_eval.py --merge).

    .venv/bin/python -I scripts/ablation_report.py          # print the tables
    .venv/bin/python -I scripts/ablation_report.py --js     # also write learn/i-ablation-data.js
"""
import argparse
import json
from pathlib import Path

import torch

DOMAINS = ["biology", "chemistry", "physics", "algebra", "geometry", "number_theory", "code", "web"]


def hurt_examples(tok, data, delta, n=6, ctx=24):
    """The n target tokens whose loss rose most, with the text before them."""
    flat = delta.float().flatten()
    out = []
    for idx in flat.topk(n).indices.tolist():
        row, t = divmod(idx, delta.shape[1])
        ids = data["tokens"][row].tolist()
        out.append({"domain": data["source"][row], "delta": round(float(flat[idx]), 3),
                    "before": tok.decode(ids[max(0, t + 1 - ctx):t + 1]), "token": tok.decode([ids[t + 1]])})
    return out


def similarity_summary(path, stats_path, ks=(128, 48)):
    sim = torch.load(path)
    from moe_stream import select_experts
    stats = torch.load(stats_path)
    res = {}
    for kind in ("output", "router"):
        s = sim[kind].clone()
        for m in s:
            m.fill_diagonal_(-1)
        res[kind] = {"best_match_median_per_layer": [round(float(m.max(1).values.median()), 4) for m in s]}
    for k in ks:
        keep = select_experts(stats, k, "reap")
        best, sizes, own = [], [], []
        for i, kp in enumerate(keep):
            dropped = torch.ones(sim["output"].shape[1], dtype=torch.bool)
            dropped[kp] = False
            m = sim["output"][i][dropped][:, kp]
            best += m.max(1).values.tolist()
            sizes += torch.bincount(m.argmax(1), minlength=len(kp)).tolist()
            f = stats["freq"][i].double().clamp(min=1)
            absorbed = torch.zeros(len(kp), dtype=torch.float64).index_add_(0, m.argmax(1), f[dropped])
            own += [float(f[t] / (f[t] + a)) for t, a in zip(kp.tolist(), absorbed.tolist()) if a > 0]
        hist = torch.histc(torch.tensor(best), bins=20, min=-0.1, max=0.9).tolist()
        res[f"k{k}"] = {"target_sim_hist": hist, "hist_range": [-0.1, 0.9], "target_sim_median": round(float(torch.tensor(best).median()), 4),
                        "group_size_max": max(sizes), "group_size_hist": torch.bincount(torch.tensor(sizes)).tolist(),
                        "own_weight_median": round(float(torch.tensor(own).median()), 3),
                        "own_weight_below_half_pct": round(100 * float((torch.tensor(own) < 0.5).double().mean()), 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--js", action="store_true")
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    abl = json.loads(Path("results/ablation.json").read_text())
    print(f"{'expert':8s} {'reading':28s} " + " ".join(f"{d[:7]:>8s}" for d in DOMAINS) + "   routed   other")
    for r in abl["rows"]:
        e = f"{r['layer']}.{r['expert']}" if r.get("expert") is not None else "-"
        p = r["ppl_change_pct"]
        print(f"{e:8s} {r['reading'][:28]:28s} " + " ".join(f"{p[d]:8.3f}" for d in DOMAINS)
              + (f"  {r['delta_routed']:7.4f} {r['delta_other']:7.4f}" if r.get("delta_routed") is not None else ""))

    evals = {}
    for line in open("results/eval.jsonl"):
        row = json.loads(line)
        evals[row["tag"]] = row
    variants = [("original", "step5-original")]
    for k in (128, 48):
        variants += [(f"prune K={k}", f"step5-prune-k{k}"), (f"merge naive K={k}", f"merge-naive-output-keep{k}"),
                     (f"merge aligned K={k}", f"merge-aligned-output-keep{k}"),
                     (f"merge aligned, router similarity K={k}", f"merge-aligned-router-keep{k}"),
                     (f"merge aligned, full router K={k}", f"merge-aligned-remap-output-keep{k}")]
    merge = []
    print()
    for name, tag in variants:
        if tag in evals:
            merge.append({"name": name, "tag": tag, "ppl": evals[tag]["ppl"]})
            print(f"{name:40s} " + "  ".join(f"{k} {v:.3f}" for k, v in evals[tag]["ppl"].items()))

    if args.js:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        data = torch.load("data/domains.pt")
        tokens = torch.load("results/ablation_tokens.pt")
        examples = {key: hurt_examples(tok, data, d) for key, d in tokens["deltas"].items()}
        sim = similarity_summary("stats/similarity.pt", "stats/calib.pt") if Path("stats/similarity.pt").exists() else None
        teams = json.loads(Path("results/ablation_teams.json").read_text()) if Path("results/ablation_teams.json").exists() else []
        # the control run before moe_forward was made deterministic (index_add_ with GPU atomics), saved from its log
        nondet = json.loads(Path("results/ablation_repeat_nondeterministic.json").read_text())
        Path("learn/i-ablation-data.js").write_text(
            "/* Generated by scripts/ablation_report.py --js from results/ablation.json, ablation_tokens.pt, eval.jsonl, stats/similarity.pt. */\n"
            "const ABL = " + json.dumps({"domains": DOMAINS, "rows": abl["rows"], "examples": examples, "merge": merge, "similarity": sim,
                                    "teams": teams, "repeat_nondeterministic": nondet}) + ";\n")
        print("wrote learn/i-ablation-data.js")


if __name__ == "__main__":
    main()
