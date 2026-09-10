"""Build the final dataset: stories + conversations + Python code.

One run over all three, rather than three separate runs. Training on code
alone would erode the English; training on English alone loses the code. The
mix is oversampled so each source keeps a meaningful share of what the model
sees.

Run:  .venv/bin/python model/prepare_all_tokens.py
"""
import os
import sys

import numpy as np

from tokenizer import CORPUS, DATA, WordTokenizer

CHAT = os.path.join(DATA, "chat.txt")
CODE = os.path.join(DATA, "code.txt")
BOOKS = os.path.join(DATA, "books.txt")
TRAIN_BIN = os.path.join(DATA, "all_train.bin")
VAL_BIN = os.path.join(DATA, "all_val.bin")

# Repeats chosen so no source is drowned out. Real books are only ~4 MB
# against 133 MB of TinyStories, and TinyStories is written deliberately
# simply for young children - excellent for coherence, useless for rich or
# creative language. Repeating the books 12x is what gives Flow a vocabulary
# beyond "the cat was happy".
SOURCES = [("stories", CORPUS, 1), ("chat", CHAT, 5),
           ("code", CODE, 2), ("books", BOOKS, 12)]


def encode_file(tok, name, path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = []
    for start in range(0, len(text), 5_000_000):
        ids.extend(tok.encode(text[start:start + 5_000_000]))
        print(f"  {name}: {len(ids):,} tokens", flush=True)
    return np.array(ids, dtype=np.uint16)


def main():
    tok = WordTokenizer.load()
    print(f"vocab: {tok.vocab_size}\n")

    trains, vals, report = [], [], []
    for name, path, repeat in SOURCES:
        if not os.path.exists(path):
            print(f"  {name}: MISSING at {path}")
            continue
        ids = encode_file(tok, name, path)
        # Validation is held out from every source in proportion, so the metric
        # tracks all three things the model is being asked to learn. Holding it
        # out of one source only is the bug that cost 30 minutes yesterday.
        cut = int(0.98 * len(ids))
        trains.append(np.tile(ids[:cut], repeat))
        vals.append(ids[cut:])
        report.append((name, len(ids[:cut]) * repeat, repeat))

    train = np.concatenate(trains)
    val = np.concatenate(vals)
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)

    print(f"\ntrain: {len(train):,} tokens")
    shares = {}
    for name, count, repeat in report:
        share = 100.0 * count / len(train)
        shares[name] = share
        print(f"  {name:8s} {count:>12,}  ({repeat}x)  {share:4.0f}%")
    print(f"val:   {len(val):,} tokens (all three sources)")

    checks = [
        ("train > 40M tokens", len(train) > 40_000_000, f"{len(train):,}"),
        ("code share 15-35%", 15 <= shares.get("code", 0) <= 35,
         f"{shares.get('code', 0):.0f}%"),
        ("chat share 20-40%", 20 <= shares.get("chat", 0) <= 40,
         f"{shares.get('chat', 0):.0f}%"),
        ("books share 8-25%", 8 <= shares.get("books", 0) <= 25,
         f"{shares.get('books', 0):.0f}%"),
    ]
    print("\n=== DATASET GATE ===")
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    ok = all(c[1] for c in checks)
    print(f"\n{'DATASET READY' if ok else 'NOT READY'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
