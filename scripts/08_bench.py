"""Step 2: a small but real benchmark, run against a llama-server that is already up.

  gsm8k      grade-school math word problems (openai/gsm8k, test): the final number must match
  humaneval  164 Python functions with hidden unit tests (openai/openai_humaneval): the tests
             must pass ("pass@1"). This RUNS model-written code in a `python -I` subprocess with
             a 10 s timeout - fine for a local model on a fixed benchmark, but not a sandbox.
  mmlu       4-choice questions from 57 subjects (cais/mmlu, all/test, fixed random sample).
             Scored by next-token probability, the usual way for MMLU: the answer is started as
             "The correct option is **" and the most likely of A/B/C/D is the model's pick. A
             free-text answer would mostly measure whether the model obeys "letter only" (the
             pruned code model writes explanations instead). Measures general knowledge.

Greedy decoding (temperature 0), thinking off, so a run is reproducible. Each score comes with
a 95% confidence interval: two models whose intervals overlap a lot are not clearly different.

  .venv/bin/python -I scripts/08_bench.py --tag code-k48
Writes one summary row to results/bench.jsonl and every answer to results/bench/<tag>-<task>.jsonl.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset


def post(port, path, body):
    req = urllib.request.Request(f"http://localhost:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def chat(port, prompt, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return (r["choices"][0]["message"].get("content") or "").strip(), r["usage"]["completion_tokens"]


# ---------------------------------------------------------------- gsm8k
def last_number(text):
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_items(n):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        prompt = (f"{row['question']}\n\nSolve it step by step. On the last line write the final answer as "
                  f"'#### <number>'.")
        yield {"id": f"gsm8k-{i}", "prompt": prompt, "gold": gold}


def gsm8k_grade(item, answer):
    got = last_number(answer)
    return got is not None and abs(got - item["gold"]) < 1e-6, got


# ---------------------------------------------------------------- humaneval
def humaneval_items(n):
    ds = load_dataset("openai/openai_humaneval", split="test")
    for row in ds.select(range(min(n, len(ds)))):
        prompt = ("Complete this Python function. Reply with the complete function, including its signature, "
                  f"in one ```python code block.\n\n```python\n{row['prompt']}```")
        yield {"id": row["task_id"], "prompt": prompt, "src": row["prompt"], "test": row["test"], "entry": row["entry_point"]}


def humaneval_grade(item, answer):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
    if not m:
        return False, "no code block"
    # the original prompt first: it holds the imports and any helper functions; the model's
    # definition then replaces the target function (its body in the prompt is only a docstring)
    program = f"{item['src']}\n    pass\n\n{m.group(1)}\n\n{item['test']}\ncheck({item['entry']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=10)
        ok = r.returncode == 0
        return ok, "" if ok else (r.stderr.strip().splitlines() or ["?"])[-1][:100]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(f.name)


# ---------------------------------------------------------------- mmlu
def mmlu_items(n):
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(0).sample(range(len(ds)), min(n, len(ds)))
    for i in idx:
        row = ds[i]
        opts = "\n".join(f"{'ABCD'[k]}. {c}" for k, c in enumerate(row["choices"]))
        prompt = (f"{row['question']}\n\n{opts}\n\nAnswer with the letter of the correct option (A, B, C or D) only.")
        yield {"id": f"{row['subject']}-{i}", "prompt": prompt, "gold": "ABCD"[row["answer"]], "subject": row["subject"]}


def mmlu_answer(port, item):
    """Most likely letter right after the forced answer start (thinking off, as in the chat runs)."""
    t = post(port, "/apply-template", {"messages": [{"role": "user", "content": item["prompt"]}],
                                        "chat_template_kwargs": {"enable_thinking": False}})
    r = post(port, "/completion", {"prompt": t["prompt"] + "The correct option is **", "n_predict": 1,
                                   "n_probs": 20, "temperature": 0, "cache_prompt": False})
    probs = {}
    for p in r["completion_probabilities"][0]["top_logprobs"]:
        tok = p["token"].strip()
        if tok in ("A", "B", "C", "D"):
            probs[tok] = max(probs.get(tok, 0.0), math.exp(p["logprob"]))
    return max(probs, key=probs.get) if probs else "", 1


def mmlu_grade(item, answer):
    return answer == item["gold"], answer or None


TASKS = {  # name: (items, grade, max_tokens, answer function; None = normal chat answer)
    "gsm8k": (gsm8k_items, gsm8k_grade, 2048, None),
    "humaneval": (humaneval_items, humaneval_grade, 2048, None),
    "mmlu": (mmlu_items, mmlu_grade, 1, mmlu_answer),
}


def ci95(k, n):
    """Wilson score interval for a proportion, in percent."""
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name for this model in the results")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--n", type=int, nargs=3, default=[250, 164, 300], metavar=("GSM8K", "HUMANEVAL", "MMLU"))
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--parallel", type=int, default=4, help="concurrent requests (match llama-server -np)")
    args = ap.parse_args()
    n_for = dict(zip(TASKS, args.n))

    Path("results/bench").mkdir(parents=True, exist_ok=True)
    row = {"tag": args.tag}
    for task in args.tasks:
        items_fn, grade_fn, max_tokens, answer_fn = TASKS[task]
        items = list(items_fn(n_for[task]))
        t0 = time.time()

        def run(item):
            answer, toks = answer_fn(args.port, item) if answer_fn else chat(args.port, item["prompt"], max_tokens)
            ok, detail = grade_fn(item, answer)
            return {"id": item["id"], "ok": ok, "detail": detail, "tokens": toks, "answer": answer}

        with ThreadPoolExecutor(args.parallel) as pool:
            results = list(pool.map(run, items))
        k = sum(r["ok"] for r in results)
        with open(f"results/bench/{args.tag}-{task}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        lo, hi = ci95(k, len(results))
        row[task] = {"score": round(100 * k / len(results), 1), "ci95": [lo, hi], "n": len(results),
                     "hit_token_limit": 0 if answer_fn else sum(r["tokens"] >= max_tokens - 4 for r in results),
                     "seconds": round(time.time() - t0)}
        if task == "mmlu":
            row[task]["unparsed"] = sum(r["detail"] is None for r in results)
        extra = f"  no letter in top 20 {row[task]['unparsed']}" if task == "mmlu" else ""
        print(f"{args.tag:16s} {task:10s} {k:4d}/{len(results):<4d} = {100 * k / len(results):5.1f}%  "
              f"(95% CI {lo}-{hi})  hit limit {row[task]['hit_token_limit']}{extra}  [{time.time() - t0:.0f}s]", flush=True)
    with open("results/bench.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
