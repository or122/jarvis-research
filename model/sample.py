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


def stream(prompt, max_new_tokens=120, temperature=0.8, top_k=40):
    """Yield decoded text one token at a time."""
    model, tok, _ = load()
    ids = tok.encode(prompt) or [tok.stoi.get(" the", 256)]
    idx = torch.tensor([ids], dtype=torch.long)
    for token_id in model.generate(idx, max_new_tokens, temperature, top_k):
        yield tok.decode([token_id])


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
