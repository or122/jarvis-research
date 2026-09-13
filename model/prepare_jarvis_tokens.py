"""Encode the Jarvis fine-tuning mix.

Assistant examples get most of the weight, because the point is to change the
model's manner. Story and chat data stay in as replay - fine-tuning on the new
data alone would teach the manner while quietly eroding the English underneath.
That trick is why the earlier chat fine-tune worked, and skipping it is a
classic way to ruin a model.

Validation is held out from EVERY source in proportion. Holding it out of one
source only is the bug that cost half an hour on 2026-09-10: validation
measured stories while the model was being trained to converse, so the metric
moved the wrong way and saving stopped.

Run:  .venv/bin/python model/prepare_jarvis_tokens.py
"""
import os
import sys

import numpy as np

from tokenizer import CORPUS, DATA, WordTokenizer

JARVIS = os.path.join(DATA, "jarvis.txt")
CHAT = os.path.join(DATA, "chat.txt")
BOOKS = os.path.join(DATA, "books.txt")
CODE = os.path.join(DATA, "code.txt")
TRAIN_BIN = os.path.join(DATA, "jv_train.bin")
VAL_BIN = os.path.join(DATA, "jv_val.bin")

# jarvis.txt is only 2.4 MB but must dominate the model's manner, so it is
# repeated hard. The rest is replay, at just enough weight to hold the English.
SOURCES = [
    ("jarvis", JARVIS, 8),
    ("chat", CHAT, 1),
    ("code", CODE, 1),
    ("books", BOOKS, 2),
    ("stories", CORPUS, 0),      # 0 = a slice only, see below
]
STORY_SLICE_MB = 20


def encode(tok, name, path, cap_mb=None):
    with open(path, encoding="utf-8") as f:
        text = f.read(cap_mb * 1024 * 1024) if cap_mb else f.read()
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
            print(f"  {name}: MISSING {path}")
            continue
        ids = encode(tok, name, path, STORY_SLICE_MB if repeat == 0 else None)
        cut = int(0.98 * len(ids))
        trains.append(np.tile(ids[:cut], max(repeat, 1)))
        vals.append(ids[cut:])
        report.append((name, len(ids[:cut]) * max(repeat, 1), max(repeat, 1)))

    train = np.concatenate(trains)
    val = np.concatenate(vals)
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)

    print(f"\ntrain: {len(train):,} tokens")
    shares = {}
    for name, count, repeat in report:
        share = 100.0 * count / len(train)
        shares[name] = share
        print(f"  {name:8s} {count:>12,}  (x{repeat})  {share:4.0f}%")
    print(f"val:   {len(val):,} tokens (every source)")

    checks = [
        ("train > 10M tokens", len(train) > 10_000_000, f"{len(train):,}"),
        ("jarvis share 15-45%", 15 <= shares.get("jarvis", 0) <= 45,
         f"{shares.get('jarvis', 0):.0f}%"),
        ("replay kept (stories+books > 20%)",
         shares.get("stories", 0) + shares.get("books", 0) > 20,
         f"{shares.get('stories', 0) + shares.get('books', 0):.0f}%"),
    ]
    print("\n=== JARVIS DATASET GATE ===")
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    ok = all(c[1] for c in checks)
    print(f"\n{'READY' if ok else 'NOT READY'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
