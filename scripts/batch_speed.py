"""Throughput against the number of simultaneous requests, on a running llama-server.

Generating one token for one request reads (almost) every active weight once; generating one token for 8
requests at the same time reads the same weights once and uses them 8 times. So while decoding is limited by
memory bandwidth, total tokens per second grow almost linearly with the batch, until compute or the
experts' spread (different tokens want different experts) catches up. Writes results/batch_speed.jsonl.

    llama.cpp/build/bin/llama-server -m gguf/q35-reap-k128/q35-reap-k128-Q4_K_M.gguf -ngl 99 -np 16 -c 65536 --port 8092
    .venv/bin/python -I scripts/batch_speed.py --port 8092
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

PROMPTS = ["Explain how a refrigerator works.", "Write a short story about a lighthouse keeper.",
           "Describe the water cycle.", "What are the causes of inflation?", "How do vaccines train the immune system?",
           "Summarise the history of the printing press.", "Explain recursion to a beginner.", "Why is the sky blue?"]


def one(port, k, n_predict):
    body = {"prompt": PROMPTS[k % len(PROMPTS)], "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
            "ignore_eos": True}
    req = Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as r:
        return json.loads(r.read())["timings"]["predicted_n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--tag", default="q128-Q4_K_M")
    args = ap.parse_args()
    one(args.port, 0, 16)  # warm-up
    for n in args.batches:
        t0 = time.time()
        with ThreadPoolExecutor(n) as pool:
            tokens = sum(pool.map(lambda k: one(args.port, k, args.n_predict), range(n)))
        dt = time.time() - t0
        row = {"tag": args.tag, "parallel": n, "tokens": tokens, "seconds": round(dt, 2),
               "total_tok_s": round(tokens / dt, 1), "per_request_tok_s": round(tokens / dt / n, 1)}
        print(json.dumps(row), flush=True)
        with open("results/batch_speed.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
