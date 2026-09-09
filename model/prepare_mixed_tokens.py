"""Build the overnight dataset: the full corpus with conversations mixed in.

Phase 1 fine-tuned on conversations alone, which teaches the reply format fast
but stops improving the model's English. This mixes both so one long run
improves both at once.

Conversations are repeated 3x so they stay roughly a third of what the model
sees. Left at their natural 13%, the reply format would slowly wash out over a
long run on story text.

Run:  .venv/bin/python model/prepare_mixed_tokens.py
"""
import os
import sys

import numpy as np

from tokenizer import CORPUS, DATA, WordTokenizer

CHAT = os.path.join(DATA, "chat.txt")
TRAIN_BIN = os.path.join(DATA, "mixed_train.bin")
VAL_BIN = os.path.join(DATA, "mixed_val.bin")
CHAT_REPEAT = 3


def encode_all(tok, name, text):
    ids = []
    for start in range(0, len(text), 5_000_000):
        ids.extend(tok.encode(text[start:start + 5_000_000]))
        print(f"  encoding {name}: {len(ids):,} tokens", flush=True)
    return np.array(ids, dtype=np.uint16)


def main():
    tok = WordTokenizer.load()

    with open(CHAT, encoding="utf-8") as f:
        chat = f.read()
    with open(CORPUS, encoding="utf-8") as f:
        corpus = f.read()

    print(f"chat:   {len(chat):,} characters")
    print(f"corpus: {len(corpus):,} characters\n")

    chat_ids = encode_all(tok, "chat", chat)
    corpus_ids = encode_all(tok, "corpus", corpus)

    # Validation is held out from BOTH sources, in proportion — the run is
    # optimising conversation quality and English quality together, so the
    # metric has to see both. Getting this wrong once already cost 30 minutes.
    chat_cut = int(0.98 * len(chat_ids))
    corpus_cut = int(0.98 * len(corpus_ids))

    train = np.concatenate(
        [np.tile(chat_ids[:chat_cut], CHAT_REPEAT), corpus_ids[:corpus_cut]]
    )
    val = np.concatenate([chat_ids[chat_cut:], corpus_ids[corpus_cut:]])

    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)

    chat_tokens = len(chat_ids[:chat_cut]) * CHAT_REPEAT
    share = 100.0 * chat_tokens / len(train)

    print(f"\ntrain: {len(train):,} tokens")
    print(f"       {chat_tokens:,} chat ({CHAT_REPEAT}x repeated) = {share:.0f}%")
    print(f"       {corpus_cut:,} corpus = {100 - share:.0f}%")
    print(f"val:   {len(val):,} tokens (both sources)")

    ok = len(train) > 30_000_000 and 20 <= share <= 45
    print(f"\n=== MIXED DATASET GATE ===")
    print(f"{'PASS' if len(train) > 30_000_000 else 'FAIL'}  train > 30M tokens: {len(train):,}")
    print(f"{'PASS' if 20 <= share <= 45 else 'FAIL'}  chat share 20-45%: {share:.0f}%")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
