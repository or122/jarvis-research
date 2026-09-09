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

    def encode_all(name, text):
        ids = []
        for start in range(0, len(text), 5_000_000):
            ids.extend(tok.encode(text[start:start + 5_000_000]))
            print(f"  encoding {name}: {len(ids):,} tokens", flush=True)
        return np.array(ids, dtype=np.uint16)

    chat_ids = encode_all("chat", chat)
    replay_ids = encode_all("replay", replay)

    # Validation must be held out from the CHAT data specifically. Splitting
    # the concatenated file by position put the replay stories at the end, so
    # validation contained no conversations at all and measured story quality
    # while the model was being trained to converse — the metric moved the
    # wrong way and the "save on improvement" rule stopped saving.
    cut = int(0.95 * len(chat_ids))
    train = np.concatenate([chat_ids[:cut], replay_ids])
    val = chat_ids[cut:]

    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)

    print(f"\ntokens: {len(train) + len(val):,}")
    print(f"train:  {len(train):,} ({cut:,} chat + {len(replay_ids):,} replay)")
    print(f"val:    {len(val):,} (chat only — the thing being optimised)")
    arr = train

    ok = len(arr) > 3_000_000
    print(f"\n{'CHAT TOKENS READY' if ok else 'TOO FEW TOKENS'} (need > 3M)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
