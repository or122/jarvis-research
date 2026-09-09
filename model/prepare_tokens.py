"""Encode the corpus once, save it as raw token ids.

Encoding 145M characters in Python takes minutes. Training needs to read the
data thousands of times, so it happens once here and training just memory-maps
the result.

Run:  .venv/bin/python model/prepare_tokens.py
"""
import os
import sys

import numpy as np

from tokenizer import CORPUS, DATA, WordTokenizer

TRAIN_BIN = os.path.join(DATA, "train.bin")
VAL_BIN = os.path.join(DATA, "val.bin")


def main():
    tok = WordTokenizer.load()
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} characters")

    # Encode in chunks so progress is visible and memory stays flat.
    chunk = 5_000_000
    ids = []
    for start in range(0, len(text), chunk):
        ids.extend(tok.encode(text[start:start + chunk]))
        pct = 100.0 * min(start + chunk, len(text)) / len(text)
        print(f"  encoding {pct:5.1f}%   {len(ids):,} tokens", flush=True)

    arr = np.array(ids, dtype=np.uint16)   # vocab is 8192, fits in uint16
    n = int(0.9 * len(arr))
    arr[:n].tofile(TRAIN_BIN)
    arr[n:].tofile(VAL_BIN)

    print(f"\ntokens:      {len(arr):,}")
    print(f"train:       {n:,}  -> {TRAIN_BIN}")
    print(f"val:         {len(arr) - n:,}  -> {VAL_BIN}")
    print(f"compression: {len(text) / len(arr):.2f} characters per token")

    ok = len(arr) > 20_000_000
    print(f"\n{'TOKENS READY' if ok else 'TOO FEW TOKENS'} (need > 20M)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
