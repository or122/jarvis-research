"""Flow's model eval. Three gates. Any FAIL means not done.

Run:  .venv/bin/python model/eval.py
Exit code 0 = all passed, 1 = something failed.
"""
import math
import os
import re
import sys

import torch

from model import BLOCK_SIZE, FlowLM
from tokenizer import WordTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")
WORDS = "/usr/share/dict/words"
VOCAB_SIZE = 8192

results = []


def gate(n, name, passed, detail):
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  Gate {n} — {name}")
    print(f"      {detail}\n")


# --- Gate 1: wiring ---------------------------------------------------------
# An untrained model must guess uniformly across the vocabulary, which is a
# cross-entropy of exactly ln(vocab_size). Far from it means the model is
# miswired before training starts.
torch.manual_seed(1337)
fresh = FlowLM(VOCAB_SIZE)
x = torch.randint(0, VOCAB_SIZE, (4, BLOCK_SIZE))
_, untrained = fresh(x, x)
expected = math.log(VOCAB_SIZE)
gate(1, f"untrained loss ~= ln({VOCAB_SIZE})",
     8.7 < untrained.item() < 9.3,
     f"got {untrained.item():.4f}, expected {expected:.4f}, window (8.7, 9.3)")

# --- Gate 2: learning -------------------------------------------------------
if not os.path.exists(CKPT):
    gate(2, "trained val loss < 4.5", False, "no ckpt.pt — run train.py first")
    val_loss = None
else:
    ck = torch.load(CKPT, map_location="cpu")
    val_loss = ck["val_loss"]
    gate(2, "trained val loss < 4.5", val_loss < 4.5,
         f"got {val_loss:.4f} at iter {ck['iter']} "
         f"(random-guess baseline is {expected:.2f})")

# --- Gate 3: the text is actually English ----------------------------------
# Loss can look reasonable while the output is nonsense, so check the words.
text = ""
if val_loss is None:
    gate(3, "generated text is English", False, "no checkpoint to sample from")
else:
    from sample import generate

    torch.manual_seed(1337)
    text = generate("Once upon a time there was", max_new_tokens=100)
    with open(WORDS, encoding="utf-8", errors="ignore") as f:
        vocab = {w.strip().lower() for w in f}

    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    real = [w for w in tokens if w.lower() in vocab]
    pct = 100.0 * len(real) / max(len(tokens), 1)
    sentences = len(re.findall(r"[.!?]", text))

    # Word validity alone is too easy to pass: "the the and and the" is 100%
    # real words. Variety catches degenerate repetition, which is the actual
    # failure mode of an undertrained language model.
    variety = len(set(w.lower() for w in real)) / max(len(real), 1)

    ok = pct >= 80.0 and sentences >= 3 and variety >= 0.5
    gate(3, "generated text is English",
         ok,
         f"{pct:.1f}% real words ({len(real)}/{len(tokens)}), "
         f"{sentences} sentences, {variety:.0%} distinct — "
         f"need >=80%, >=3, >=50%")

if text:
    print("--- generated ---------------------------------------------")
    print("Once upon a time there was" + text)
    print("-" * 59 + "\n")

passed = sum(results)
print(f"{passed}/3 gates passed — {'FLOW MODEL READY' if passed == 3 else 'NOT READY'}")
sys.exit(0 if passed == 3 else 1)
