"""Flow's tokenizer — word-level with byte fallback, written from zero.

Tiny-GPT used one character per token, which meant most of a small model's
capacity went into learning how to spell. This hands the model whole words
instead, so that capacity goes to meaning.

  ids   0-255   one raw byte each  (fallback: lets ANY text encode)
  ids 256-12287 the 12,032 most common words, each stored with its leading
                space, so "cat" and " cat" are the same token and decoding
                is plain concatenation

Run:  .venv/bin/python model/tokenizer.py     (builds vocab + runs the gate)
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CORPUS = os.path.join(DATA, "corpus.txt")
VOCAB = os.path.join(DATA, "vocab.json")

# 12,288 chosen by measurement, not taste: across stories/chat/code, going
# 8192 -> 12288 buys +0.22 chars/token on code for +2.1M parameters, while the
# next step up buys only +0.12 for the same cost.
VOCAB_SIZE = 12288
N_BYTES = 256
N_WORDS = VOCAB_SIZE - N_BYTES        # 12,032

# Splits text into words, numbers, punctuation runs and whitespace runs.
# Every character falls into one of those four, so the pieces always rejoin
# into the original string exactly.
PATTERN = re.compile(r"'[a-z]{1,2}| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def split(text):
    return PATTERN.findall(text)


class WordTokenizer:
    def __init__(self, words):
        self.words = list(words)
        self.stoi = {w: i + N_BYTES for i, w in enumerate(self.words)}
        self.itos = {i + N_BYTES: w for i, w in enumerate(self.words)}
        self.vocab_size = N_BYTES + len(self.words)

    # --- build / persist ---------------------------------------------------
    @classmethod
    def build(cls, text):
        counts = collections.Counter(split(text))
        top = [w for w, _ in counts.most_common(N_WORDS)]
        return cls(top)

    def save(self, path=VOCAB):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.words, f)

    @classmethod
    def load(cls, path=VOCAB):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # --- encode / decode ---------------------------------------------------
    def encode(self, text):
        ids = []
        for piece in split(text):
            i = self.stoi.get(piece)
            if i is not None:
                ids.append(i)
            else:
                # Unknown word: spell it out in bytes. Nothing is ever
                # unrepresentable, so there is no "unknown token" hole.
                ids.extend(piece.encode("utf-8"))
        return ids

    def decode(self, ids):
        out = []
        buf = bytearray()
        for i in ids:
            if i < N_BYTES:
                buf.append(i)
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(self.itos.get(i, ""))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


if __name__ == "__main__":
    CHAT = os.path.join(DATA, "chat.txt")
    CODE = os.path.join(DATA, "code.txt")

    if not os.path.exists(CORPUS):
        print(f"no corpus at {CORPUS} — run download_corpus.py first")
        sys.exit(1)

    # The vocabulary must be built from everything the model will ever see.
    # Built from stories alone, "def", "return", "self" and "import" were all
    # missing, and code tokenized at 1.15 characters per token instead of 4.2
    # — a 30-line function would not fit in a 128-token window.
    sources = []
    with open(CORPUS, encoding="utf-8") as f:
        sources.append(("stories", f.read(25 * 1024 * 1024)))
    for name, path in (("chat", CHAT), ("code", CODE)):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                sources.append((name, f.read(20 * 1024 * 1024)))

    print("building vocabulary from:")
    for name, text in sources:
        print(f"  {name:8s} {len(text):>12,} characters")

    tok = WordTokenizer.build("\n".join(t for _, t in sources))
    tok.save()
    print(f"\nvocab size: {tok.vocab_size}")
    print(f"most common: {tok.words[:10]}")

    for w in ("def", "return", "self", "import", "class", "None", "True"):
        mark = "in vocab" if w in tok.stoi else "MISSING"
        print(f"  {w!r:9} {mark}")

    # --- gate ----------------------------------------------------------
    # Measured per domain: one number hides a vocabulary that serves stories
    # well and code badly, which is exactly the failure being fixed.
    print("\n=== TOKENIZER GATE ===")
    ok = True
    for name, text in sources:
        sample = text[-150_000:]
        ids = tok.encode(sample)
        cpt = len(sample) / len(ids)
        real = 100.0 * sum(1 for i in ids if i >= N_BYTES) / len(ids)
        # Floors differ by domain because the domains differ. Chat cannot
        # reach prose compression at ANY vocabulary size — measured 3.15 at
        # 8k and only 3.30 at 24k — because every turn carries "You:", "Me:"
        # and newlines that no vocabulary can compress. A 3.5 floor there was
        # prose calibration wrongly applied to dialogue markup.
        floor = {"code": 2.5, "chat": 3.0}.get(name, 3.5)
        passed = cpt >= floor and tok.decode(ids) == sample
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:8s} "
              f"{cpt:.2f} chars/token (need >={floor})   "
              f"{real:.1f}% real words   round-trip "
              f"{'exact' if tok.decode(ids) == sample else 'BROKEN'}")

    demo = "def add(a, b):\n    return a + b"
    print(f"\ncode demo: {len(demo)} characters -> {len(tok.encode(demo))} tokens")
    print(f"  round-trip: {tok.decode(tok.encode(demo)) == demo}")

    print(f"\n{'TOKENIZER READY' if ok else 'TOKENIZER NOT GOOD ENOUGH'}")
    sys.exit(0 if ok else 1)
