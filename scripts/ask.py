"""Send test prompts to a running llama-server (OpenAI-compatible API).

    .venv/bin/python -I scripts/ask.py [--think] [--port 8089]       # 3 general prompts
    .venv/bin/python -I scripts/ask.py --set code --check            # 5 coding prompts

--check grades every coding prompt automatically:
  python    first ```python block (or the whole answer if it is valid Python) + unit tests,
            run in a separate `python -I` process with a 10 s timeout. That executes code the
            model wrote: fine for a local model answering these fixed prompts, not a sandbox.
  sql       first ```sql block run with sqlite3 on a small test database
  contains  the answer must contain the expected text
"""
import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request

GENERAL = [
    ("What is the capital of Australia, and why isn't it Sydney? Answer in 3 sentences.", None),
    ("Write a Python function that returns the n-th Fibonacci number iteratively.", None),
    ("If a train travels 180 km in 2.5 hours, what is its average speed in km/h?", None),
]
CODE = [
    ("Write a Python function fib(n) that returns the n-th Fibonacci number iteratively, with fib(0) = 0 and fib(1) = 1.",
     ("python", "assert [fib(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\nassert fib(50) == 12586269025")),
    ("Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards, "
     "ignoring case and any character that is not a letter or digit.",
     ("python", "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('hello')\nassert is_palindrome('')\nassert is_palindrome('No 1on')")),
    ("This function crashes on an empty list. Fix it so it returns 0.0 for an empty list and give the whole corrected function:\n\n"
     "def avg(xs):\n    return sum(xs) / len(xs)",
     ("python", "assert avg([]) == 0.0\nassert avg([1, 2, 3]) == 2\nassert avg([5]) == 5")),
    ("Write a SQL query that returns the names of the 3 customers with the highest total order amount, "
     "given tables customers(id, name) and orders(id, customer_id, amount).",
     ("sql", ["Dana", "Bob", "Eve"])),
    ("In one or two sentences, what does this Python expression produce: [x * x for x in range(10) if x % 2]",
     ("contains", "[1, 9, 25, 49, 81]")),
]


SQL_DB = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
INSERT INTO customers VALUES (1,'Ann'),(2,'Bob'),(3,'Cid'),(4,'Dana'),(5,'Eve');
INSERT INTO orders VALUES (1,1,10),(2,2,50),(3,2,40),(4,3,20),(5,4,70),(6,4,60),(7,5,30),(8,5,45),(9,1,5);
"""  # totals: Dana 130, Bob 90, Eve 75, Cid 20, Ann 15


def python_code(answer):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
    if m:
        return m.group(1)
    try:  # some answers are bare code without a fence
        ast.parse(answer)
        return answer
    except SyntaxError:
        return None


def check_python(answer, tests):
    code = python_code(answer)
    if code is None:
        return "FAIL (no code)"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\nprint('ALL TESTS PASSED')\n")
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "FAIL (timeout)"
    finally:
        os.unlink(f.name)
    if "ALL TESTS PASSED" in r.stdout:
        return "PASS"
    return f"FAIL ({(r.stderr.strip().splitlines() or ['?'])[-1][:80]})"


def check_sql(answer, expected):
    m = re.search(r"```sql\s*\n(.*?)```", answer, re.S | re.I)
    if not m:
        return "FAIL (no sql block)"
    db = sqlite3.connect(":memory:")
    db.executescript(SQL_DB)
    try:
        rows = db.execute(m.group(1).strip().rstrip(";")).fetchall()
    except sqlite3.Error as e:
        return f"FAIL (sqlite: {str(e)[:60]})"
    got = [r[0] for r in rows]
    return "PASS" if got == expected else f"FAIL (returned {got[:5]})"


def check(answer, test):
    kind, spec = test
    if kind == "python":
        return check_python(answer, spec)
    if kind == "sql":
        return check_sql(answer, spec)
    squash = lambda t: re.sub(r"\s+", "", t)  # noqa: E731  "[1,9, 25]" and "[1, 9, 25]" count the same
    return "PASS" if squash(spec) in squash(answer) else "FAIL (expected text missing)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("general", "code"), default="general")
    ap.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    ap.add_argument("--check", action="store_true", help="run the unit tests of coding prompts")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    passed = total = 0
    for q, tests in (CODE if args.set == "code" else GENERAL):
        body = {"messages": [{"role": "user", "content": q}], "max_tokens": args.max_tokens,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": args.think}}
        req = urllib.request.Request(f"http://localhost:{args.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req))
        m, t = r["choices"][0]["message"], r.get("timings", {})
        answer = (m.get("content") or "").strip()
        print("Q:", q)
        if m.get("reasoning_content"):
            print(f"THINK ({len(m['reasoning_content'])} chars): {m['reasoning_content'][:300]!r} ...")
        print("A:", answer)
        print(f"   [{r['usage']['completion_tokens']} tokens, {t.get('predicted_per_second', 0):.0f} tok/s]")
        if args.check and tests:
            verdict = check(answer, tests)
            total += 1
            passed += verdict == "PASS"
            print("   TESTS:", verdict)
        print()
    if args.check and total:
        print(f"unit tests passed: {passed}/{total}")


if __name__ == "__main__":
    main()
