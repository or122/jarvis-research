"""Step 1 — build Flow's training corpus.

Downloads text from several sources, cleans it, and concatenates it into one
file. Every source is optional: if one is dead, it says so and keeps going.

Run:  .venv/bin/python model/download_corpus.py
Gate: >=100 MB collected, >=95% printable ASCII, >=500k sentences. Exits 1 if not.
"""
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "corpus.txt")

MB = 1024 * 1024
TARGET_MB = 150

# (name, url, byte cap). Capped because the CPU can only consume ~50M tokens
# overnight — more text than that is never read even once.
SOURCES = [
    ("tinystories-valid",
     "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
     25 * MB),
    ("tinystories-train",
     "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
     115 * MB),
    ("gutenberg-pride-prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", 2 * MB),
    ("gutenberg-alice", "https://www.gutenberg.org/cache/epub/11/pg11.txt", 2 * MB),
    ("gutenberg-sherlock", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt", 2 * MB),
    ("gutenberg-frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt", 2 * MB),
    ("gutenberg-moby-dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt", 3 * MB),
    ("gutenberg-great-expectations", "https://www.gutenberg.org/cache/epub/1400/pg1400.txt", 3 * MB),
]

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) flow-corpus-builder"}


def fetch(name, url, cap):
    """Stream a URL to a file, stopping at `cap` bytes. Returns bytes written."""
    dest = os.path.join(DATA, f"raw_{name}.txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        size = os.path.getsize(dest)
        print(f"  {name:32s} cached      {size / MB:7.1f} MB")
        return size

    try:
        req = urllib.request.Request(url, headers=UA)
        written = 0
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            while written < cap:
                chunk = r.read(min(1 * MB, cap - written))
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
        print(f"  {name:32s} downloaded  {written / MB:7.1f} MB")
        return written
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        print(f"  {name:32s} FAILED      {type(e).__name__}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return 0


def clean(text, is_gutenberg):
    """Strip licence boilerplate and normalise whitespace."""
    if is_gutenberg:
        # Gutenberg wraps every book in legal text. Keep only what's between
        # the markers; without this the model learns to recite the licence.
        start = re.search(r"\*\*\* ?START OF TH[EIS].*?\*\*\*", text)
        end = re.search(r"\*\*\* ?END OF TH[EIS].*?\*\*\*", text)
        if start:
            text = text[start.end():]
        if end:
            # `end` was found in the original string; re-find it after slicing.
            m = re.search(r"\*\*\* ?END OF TH[EIS].*?\*\*\*", text)
            if m:
                text = text[:m.start()]

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank-line runs
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
    return text.strip()


def main():
    os.makedirs(DATA, exist_ok=True)
    print(f"building corpus -> {OUT}\n")

    print("downloading:")
    for name, url, cap in SOURCES:
        fetch(name, url, cap)

    # Tiny Shakespeare, already on disk from the Tiny-GPT project.
    shakespeare = os.path.expanduser("~/tiny-gpt/data/input.txt")
    if os.path.exists(shakespeare):
        dest = os.path.join(DATA, "raw_shakespeare.txt")
        if not os.path.exists(dest):
            with open(shakespeare, encoding="utf-8") as src, open(dest, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        print(f"  {'shakespeare':32s} copied      {os.path.getsize(dest) / MB:7.1f} MB")

    print("\ncombining:")
    total = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for fname in sorted(os.listdir(DATA)):
            if not fname.startswith("raw_"):
                continue
            path = os.path.join(DATA, fname)
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = clean(f.read(), is_gutenberg="gutenberg" in fname)
            out.write(text)
            out.write("\n\n")
            total += len(text)
            print(f"  {fname:36s} {len(text) / MB:7.1f} MB")

    # --- gate --------------------------------------------------------------
    with open(OUT, encoding="utf-8") as f:
        corpus = f.read()

    size_mb = len(corpus) / MB
    printable = sum(1 for c in corpus[:2_000_000] if 32 <= ord(c) < 127 or c in "\n\t")
    ascii_pct = 100.0 * printable / min(len(corpus), 2_000_000)
    sentences = len(re.findall(r"[.!?]", corpus))

    checks = [
        ("size >= 100 MB", size_mb >= 100, f"{size_mb:.1f} MB"),
        ("printable ASCII >= 95%", ascii_pct >= 95.0, f"{ascii_pct:.1f}%"),
        ("sentences >= 500k", sentences >= 500_000, f"{sentences:,}"),
    ]

    print("\n=== CORPUS GATE ===")
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    words = len(corpus.split())
    print(f"\nwords: {words:,}   characters: {len(corpus):,}")
    ok = all(c[1] for c in checks)
    print(f"\n{'CORPUS READY' if ok else 'CORPUS NOT READY'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
