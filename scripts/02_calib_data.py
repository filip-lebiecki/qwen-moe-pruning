"""Build tokenized calibration / evaluation sets for expert pruning.

Expert importance is measured on whatever text we feed the model, so the mix
matters: experts that only fire on (say) Chinese or Rust will look useless if
the calibration set has none.  We use a general mix:
  web   - fineweb-edu (English educational web text)
  chat  - ultrachat_200k (multi-turn assistant conversations)
  think - Mixture-of-Thoughts math/code/science (long <think> reasoning traces)
Chat-style data is rendered with the model's own chat template so the router
sees the same special tokens it sees at inference time.

Output: data/calib.pt  {"tokens": int32 [N, L], "source": [N]}
        data/eval.pt   same format, held-out samples + wikitext-2 test

--preset code builds a code-only calibration set instead (for a code specialist):
  think-code  - Mixture-of-Thoughts code (problem, <think> trace, solution), 3/4
  py-instruct - python_code_instructions_18k ("write a Python function ..." with a
                plain answer, no reasoning), 1/4
and writes data/calib_code.pt + data/eval_code.pt (held-out rows of the same two sets).

--preset domains builds data/domains.pt for the ablation test (16_ablate.py): 16 sequences each of
  biology / chemistry / physics  - camel-ai question + answer pairs (one writing style, three sciences)
  algebra / geometry / number_theory - MATH competition problems + worked solutions
  code - python_code_instructions (instruction + commented code), web - fineweb-edu
Matched pairs in one style are the point: if deleting the "biology expert" hurts biology more than
chemistry and physics, written the same way, the difference is the topic, not the style.

--preset train builds data/train.pt for distillation (10_distill.py): the four general sources,
2 x --calib-per-source sequences each (1M tokens by default), from rows that neither the
calibration nor the eval set uses.
"""
import argparse
import json
import random
import zipfile
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def web_docs(split_offset):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for row in islice(ds, split_offset, None):
        yield row["text"]


def chat_docs(tok, split, split_offset=0):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
    for row in islice(ds, split_offset, None):
        yield tok.apply_chat_template(row["messages"], tokenize=False)


def think_docs(tok, subset, split_offset):
    ds = load_dataset("open-r1/Mixture-of-Thoughts", subset, split="train", streaming=True)
    for row in islice(ds, split_offset, None):
        yield tok.apply_chat_template(row["messages"], tokenize=False)


def py_instruct_docs(tok, split_offset):
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    for row in islice(ds, split_offset, None):
        q = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
        yield tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": row["output"]}], tokenize=False)


def camel_docs(subject):
    """camel-ai/<subject>: ~20k JSON files of question (message_1) and answer (message_2), in a zip."""
    z = zipfile.ZipFile(hf_hub_download(f"camel-ai/{subject}", f"{subject}.zip", repo_type="dataset"))
    names = sorted(n for n in z.namelist() if n.endswith(".json"))
    random.Random(0).shuffle(names)
    for n in names:
        row = json.loads(z.read(n))
        yield row["message_1"] + "\n\n" + row["message_2"]


def math_docs(subject):
    ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
    for row in ds.shuffle(seed=0):
        yield row["problem"] + "\n\n" + row["solution"]


def code_docs(split_offset):
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    for row in islice(ds, split_offset, None):
        yield row["instruction"] + "\n\n" + row["output"]


def wikitext_docs():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # wikitext is one line per row; join it back into one long document
    yield "".join(ds["text"])


def pack(tok, docs, n_seqs, seq_len):
    """Concatenate documents (EOS-separated) and cut into fixed-length blocks."""
    buf, out = [], []
    for text in docs:
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)
        while len(buf) >= seq_len and len(out) < n_seqs:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(out) >= n_seqs:
            break
    if len(out) < n_seqs:
        print(f"  warning: only got {len(out)}/{n_seqs} sequences")
    return torch.tensor(out, dtype=torch.int32)


def build(tok, parts, seq_len):
    tokens, source = [], []
    for name, docs, n in parts:
        print(f"  {name}: {n} x {seq_len} tokens")
        t = pack(tok, docs, n, seq_len)
        tokens.append(t)
        source += [name] * len(t)
    return {"tokens": torch.cat(tokens), "source": source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-35B-A3B")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--calib-per-source", type=int, default=128,
                    help="sequences per source; 4 sources x 128 x 1024 = 524k tokens")
    ap.add_argument("--eval-per-source", type=int, default=32)
    ap.add_argument("--preset", choices=("general", "code", "train", "domains"), default="general")
    args = ap.parse_args()

    random.seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, e, L = args.calib_per_source, args.eval_per_source, args.seq_len

    if args.preset == "train":  # disjoint rows: calibration starts at 0, eval at 200k web / 300k think / test_sft
        print("distillation training set")
        train = build(tok, [("web", web_docs(100_000), 2 * n),
                            ("chat", chat_docs(tok, "train_sft", 50_000), 2 * n),
                            ("think-math", think_docs(tok, "math", 20_000), 2 * n),
                            ("think-code", think_docs(tok, "code", 20_000), 2 * n)], L)
        torch.save(train, out / "train.pt")
        print(f"saved {tuple(train['tokens'].shape)} train -> {out}/")
        return

    if args.preset == "domains":  # plain text (no chat template): the template's tokens are the same in every domain
        print("domain test set")
        d = 16
        ev = build(tok, [(s, camel_docs(s), d) for s in ("biology", "chemistry", "physics")]
                   + [(s, math_docs(s), d) for s in ("algebra", "geometry", "number_theory")]
                   + [("code", code_docs(16_000), d), ("web", web_docs(300_000), d)], L)
        torch.save(ev, out / "domains.pt")
        print(f"saved {tuple(ev['tokens'].shape)} domains -> {out}/")
        return

    if args.preset == "code":  # same total size as the general set: 4 x n sequences
        print("code calibration set")
        calib = build(tok, [("think-code", think_docs(tok, "code", 0), 3 * n),
                            ("py-instruct", py_instruct_docs(tok, 0), n)], L)
        torch.save(calib, out / "calib_code.pt")
        print("code eval set")  # rows far past what calibration consumed
        ev = build(tok, [("think-code", think_docs(tok, "code", 50_000), e),
                         ("py-instruct", py_instruct_docs(tok, 15_000), e)], L)
        torch.save(ev, out / "eval_code.pt")
        print(f"saved {tuple(calib['tokens'].shape)} calib_code, {tuple(ev['tokens'].shape)} eval_code -> {out}/")
        return

    print("calibration set")
    calib = build(tok, [
        ("web", web_docs(0), n),
        ("chat", chat_docs(tok, "train_sft"), n),
        ("think-math", think_docs(tok, "math", 0), n),
        ("think-code", think_docs(tok, "code", 0), n),
    ], L)
    torch.save(calib, out / "calib.pt")

    # eval: disjoint samples (skip far past what calibration consumed) + wikitext
    print("eval set")
    ev = build(tok, [
        ("web", web_docs(200_000), e),
        ("chat", chat_docs(tok, "test_sft"), e),
        ("think", think_docs(tok, "all", 300_000), e),
        ("wikitext", wikitext_docs(), e),
    ], L)
    torch.save(ev, out / "eval.pt")
    print(f"saved {tuple(calib['tokens'].shape)} calib, {tuple(ev['tokens'].shape)} eval -> {out}/")


if __name__ == "__main__":
    main()
