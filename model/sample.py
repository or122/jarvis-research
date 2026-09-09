"""Generate text from Flow's trained model.

Run:  .venv/bin/python model/sample.py ["your prompt here"]
"""
import os
import sys

import torch

from model import FlowLM
from tokenizer import WordTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")

_cache = {}


def load():
    """Load once and keep it — the server calls this on every request."""
    if "model" not in _cache:
        ckpt = torch.load(CKPT, map_location="cpu")
        tok = WordTokenizer.load()
        model = FlowLM(ckpt["vocab_size"])
        model.load_state_dict(ckpt["model"])
        model.eval()          # dropout off
        _cache["model"] = model
        _cache["tok"] = tok
        _cache["meta"] = ckpt
    return _cache["model"], _cache["tok"], _cache["meta"]


# The corpus separates its 155,520 stories with "<|endoftext|>", so the model
# correctly learns to emit it when a story finishes. That is a signal to stop,
# not text to show anyone — the same way every real LLM handles its own
# end-of-text token.
STOP = "<|"


def stream(prompt, max_new_tokens=120, temperature=0.8, top_k=40):
    """Yield decoded text one token at a time, stopping at end-of-story."""
    model, tok, _ = load()
    ids = tok.encode(prompt) or [tok.stoi.get(" the", 256)]
    idx = torch.tensor([ids], dtype=torch.long)

    # The stop marker spans several word-level tokens, so the tail has to be
    # buffered rather than checked one token at a time.
    tail = ""
    for token_id in model.generate(idx, max_new_tokens, temperature, top_k):
        piece = tok.decode([token_id])
        combined = tail + piece
        if STOP in combined:
            head = combined.split(STOP)[0]
            if head:
                yield head
            return
        # Hold back the last character in case it is the start of the marker.
        if combined.endswith(STOP[0]):
            tail = combined[-1:]
            emit = combined[:-1]
        else:
            tail, emit = "", combined
        if emit:
            yield emit


def generate(prompt, max_new_tokens=120, temperature=0.8, top_k=40):
    return "".join(stream(prompt, max_new_tokens, temperature, top_k))


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    _, _, meta = load()
    print(f"checkpoint: iter {meta['iter']}, val loss {meta['val_loss']:.4f}\n")
    print(prompt, end="", flush=True)
    for piece in stream(prompt, max_new_tokens=150):
        print(piece, end="", flush=True)
    print()
