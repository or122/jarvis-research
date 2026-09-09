"""Encode the chat corpus for fine-tuning.

Mixes in a slice of the original stories on purpose. Training a model purely
on the new format makes it forget the old one — it would learn to produce
"Me: ..." lines while its English quietly got worse. Keeping some original
data in the mix is called replay, and it is the standard guard against that.

Uses the SAME vocabulary as the base model. Changing the vocabulary would
change what every token id means and make the trained weights meaningless.

Run:  .venv/bin/python model/prepare_chat_tokens.py
"""
import os
import sys

import numpy as np

from tokenizer import CORPUS, DATA, WordTokenizer

CHAT = os.path.join(DATA, "chat.txt")
TRAIN_BIN = os.path.join(DATA, "chat_train.bin")
VAL_BIN = os.path.join(DATA, "chat_val.bin")
REPLAY_MB = 8


def main():
    tok = WordTokenizer.load()

    with open(CHAT, encoding="utf-8") as f:
        chat = f.read()
    with open(CORPUS, encoding="utf-8") as f:
        replay = f.read(REPLAY_MB * 1024 * 1024)

    print(f"chat:   {len(chat):,} characters")
    print(f"replay: {len(replay):,} characters of original stories")

    ids = []
    for name, text in (("chat", chat), ("replay", replay)):
        for start in range(0, len(text), 5_000_000):
            ids.extend(tok.encode(text[start:start + 5_000_000]))
            print(f"  encoding {name}: {len(ids):,} tokens", flush=True)

    arr = np.array(ids, dtype=np.uint16)
    n = int(0.95 * len(arr))       # chat data is precious; keep validation small
    arr[:n].tofile(TRAIN_BIN)
    arr[n:].tofile(VAL_BIN)

    print(f"\ntokens: {len(arr):,}")
    print(f"train:  {n:,} -> {TRAIN_BIN}")
    print(f"val:    {len(arr) - n:,} -> {VAL_BIN}")

    ok = len(arr) > 3_000_000
    print(f"\n{'CHAT TOKENS READY' if ok else 'TOO FEW TOKENS'} (need > 3M)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
