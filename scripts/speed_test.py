"""Generation speed of the model behind a running llama-server: one request, 256 new tokens.

The server reports its own timings: prompt processing (tokens/s read, measured on a ~1,500-token
prompt with the prompt cache off, so it is really processed) and generation (tokens/s written).
Run twice; the second, warm run is kept. Appends a row to results/speed.jsonl.

    .venv/bin/python -I scripts/speed_test.py --tag q128-Q4_K_M
"""
import argparse
import json
import urllib.request

CONTEXT = ("A hash map stores key-value pairs in an array of buckets. A hash function turns each key into an "
           "index; two keys with the same index collide and share a bucket. ") * 45
PROMPT = CONTEXT + ("\n\nUsing the notes above, explain in detail how a hash map works: hashing, buckets, "
                    "collisions, resizing, and the time complexity of each operation. Use examples.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 256, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    for _ in range(2):
        req = urllib.request.Request(f"http://localhost:{args.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = json.load(urllib.request.urlopen(req, timeout=600))["timings"]
    row = {"tag": args.tag, "prompt_tok_s": round(t["prompt_per_second"], 1), "gen_tok_s": round(t["predicted_per_second"], 1),
           "prompt_tokens": t["prompt_n"]}
    with open("results/speed.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{args.tag:20s} prompt {row['prompt_tok_s']:7.1f} tok/s   generation {row['gen_tok_s']:6.1f} tok/s")


if __name__ == "__main__":
    main()
