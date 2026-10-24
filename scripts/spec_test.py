"""Speculative decoding: speed and draft acceptance on a running llama-server, per kind of text.

Speculative decoding lets something cheap guess the next few tokens (a "draft"), and the real model checks all of
them in ONE forward pass. Every guess that matches what the model would have produced anyway is kept for free; the first
wrong one is replaced by the model's own token. With greedy decoding the output is therefore exactly the same as without
drafting, only faster, so this script also records a hash of every answer to check that.

Four prompts (chat template, thinking off, greedy, 256 tokens): prose, code, math, and an edit task that repeats a
block of code from the prompt (the case n-gram drafting is made for). One JSON row per prompt -> results/spec.jsonl.

    llama-server -m MODEL.gguf -ngl 99 -c 8192 -np 1 [--spec-type draft-mtp | --spec-type draft-simple -md DRAFT.gguf | ...]
    .venv/bin/python -I scripts/spec_test.py --port 8094 --tag 2b-mtp3
"""
import argparse
import hashlib
import json
from urllib.request import Request, urlopen

CODE = '''def summarize(data):
    totals = {}
    counts = {}
    for row in data:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
    averages = {}
    for key in totals:
        averages[key] = totals[key] / counts[key]
    largest = max(averages, key=averages.get) if averages else None
    return {"averages": averages, "largest": largest, "rows": len(data)}'''
PROMPTS = {
    "prose": "Explain how a refrigerator works, in about 200 words.",
    "code": "Write a Python function that reads a CSV file and returns the average of every numeric column. Use type hints and a docstring.",
    "math": "Solve step by step: a train travels 120 km at 80 km/h and then 90 km at 60 km/h. What is its average speed for the whole trip?",
    "edit": f"Rewrite this Python function with the variable `data` renamed to `rows` and `totals` renamed to `sums`. Change nothing else and output only the code.\n\n```python\n{CODE}\n```",
}


def post(port, path, body):
    req = Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-predict", type=int, default=256)
    args = ap.parse_args()
    post(args.port, "/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0})  # warm-up
    for name, text in PROMPTS.items():
        prompt = post(args.port, "/apply-template", {"messages": [{"role": "user", "content": text}],
                                                      "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        res = post(args.port, "/completion", {"prompt": prompt, "n_predict": args.n_predict, "temperature": 0,
                                              "cache_prompt": False, "seed": 1})
        t = res["timings"]
        row = {"tag": args.tag, "prompt": name, "tokens": t["predicted_n"], "gen_tok_s": round(t["predicted_per_second"], 1),
               "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
               "sha": hashlib.sha1(res["content"].encode()).hexdigest()[:12], "start": res["content"][:80]}
        print(json.dumps(row), flush=True)
        with open("results/spec.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
