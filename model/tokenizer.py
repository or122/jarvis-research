"""Flow's tokenizer — word-level with byte fallback, written from zero.

Tiny-GPT used one character per token, which meant most of a small model's
capacity went into learning how to spell. This hands the model whole words
instead, so that capacity goes to meaning.

  ids   0-255   one raw byte each  (fallback: lets ANY text encode)
  ids 256-8191  the 7,936 most common words, each stored with its leading
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

VOCAB_SIZE = 8192
N_BYTES = 256
N_WORDS = VOCAB_SIZE - N_BYTES        # 7,936

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
    if not os.path.exists(CORPUS):
        print(f"no corpus at {CORPUS} — run download_corpus.py first")
        sys.exit(1)

    print("reading corpus...")
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} characters")

    # Build the vocabulary from the first 40 MB. Word frequency converges long
    # before that; scanning all 150 MB would cost minutes for no better list.
    print("building vocabulary from the first 40 MB...")
    tok = WordTokenizer.build(text[:40 * 1024 * 1024])
    tok.save()
    print(f"  vocab size: {tok.vocab_size}")
    print(f"  most common: {tok.words[:12]}")

    # --- gate --------------------------------------------------------------
    # Test on text from the END of the corpus, which the vocabulary was not
    # built from — otherwise coverage would be flattered.
    sample = text[-200_000:]
    ids = tok.encode(sample)
    back = tok.decode(ids)

    round_trip = back == sample
    real_words = sum(1 for i in ids if i >= N_BYTES)
    word_pct = 100.0 * real_words / len(ids)
    chars_per_token = len(sample) / len(ids)

    checks = [
        ("exact round-trip on held-out text", round_trip,
         "identical" if round_trip else "MISMATCH"),
        ("real-word tokens >= 85%", word_pct >= 85.0, f"{word_pct:.1f}%"),
        ("chars per token >= 3.0", chars_per_token >= 3.0, f"{chars_per_token:.2f}"),
    ]

    print("\n=== TOKENIZER GATE ===")
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    demo = "The little girl smiled and said hello to her friend."
    print(f"\ndemo: {demo!r}")
    print(f"  -> {len(tok.encode(demo))} tokens: {tok.encode(demo)}")
    print(f"  -> {tok.decode(tok.encode(demo))!r}")

    ok = all(c[1] for c in checks)
    print(f"\n{'TOKENIZER READY' if ok else 'TOKENIZER BROKEN'}")
    sys.exit(0 if ok else 1)
