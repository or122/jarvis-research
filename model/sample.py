"""Generate text from Flow's trained model.

Two modes:
  continue  — the model carries your text on (what a raw language model does)
  chat      — the model replies to you (what Flow's app uses)

Run:  .venv/bin/python model/sample.py "hello"
      .venv/bin/python model/sample.py --continue "Once upon a time"
"""
import os
import sys

import torch

from model import FlowLM
from tokenizer import WordTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")

# Where a reply ends. "<|" begins the end-of-text marker the corpus uses
# between documents; "\nYou:" is the user's next turn; "\nMe:" is a second
# reply. Left alone, the model happily writes both sides of the conversation
# forever. Every real chat model has the same kind of stop list.
STOPS = ("<|", "\nYou:", "\nYou :", "\nMe:", "\nMe :")

_cache = {}


def load():
    """Load once and keep it — the server calls this on every request."""
    if "model" not in _cache:
        ckpt = torch.load(CKPT, map_location="cpu")
        tok = WordTokenizer.load()
        model = FlowLM(ckpt["vocab_size"])
        model.load_state_dict(ckpt["model"])
        model.eval()          # dropout off
        _cache.update(model=model, tok=tok, meta=ckpt)
    return _cache["model"], _cache["tok"], _cache["meta"]


def stream(prompt, max_new_tokens=120, temperature=0.8, top_k=40, stops=STOPS):
    """Yield decoded text as it is generated, ending at any stop sequence."""
    model, tok, _ = load()
    ids = tok.encode(prompt) or [tok.stoi.get(" the", 256)]
    idx = torch.tensor([ids], dtype=torch.long)

    # A stop sequence spans several word-level tokens, so text is accumulated
    # and a short tail is held back in case it turns out to be the beginning
    # of one. Without that, "\nYou" would be shown before ":" arrives.
    hold = max(len(s) for s in stops) - 1
    buffer, sent = "", 0

    for token_id in model.generate(idx, max_new_tokens, temperature, top_k):
        buffer += tok.decode([token_id])

        hits = [buffer.find(s) for s in stops]
        hits = [h for h in hits if h != -1]
        if hits:
            stop_at = min(hits)
            if stop_at > sent:
                yield buffer[sent:stop_at]
            return

        safe = len(buffer) - hold
        if safe > sent:
            yield buffer[sent:safe]
            sent = safe

    if len(buffer) > sent:
        yield buffer[sent:]


def chat_stream(message, **kw):
    """Ask the model to reply, rather than to carry on writing.

    The prompt ends mid-line at "Me:" on purpose: the model's job is to
    complete that line, and completing it is a reply.
    """
    return stream(f"You: {message}\nMe:", **kw)


def generate(prompt, **kw):
    return "".join(stream(prompt, **kw))


def chat(message, **kw):
    return "".join(chat_stream(message, **kw)).strip()


if __name__ == "__main__":
    args = sys.argv[1:]
    mode_continue = args and args[0] == "--continue"
    if mode_continue:
        args = args[1:]
    text = args[0] if args else ("Once upon a time" if mode_continue else "hello")

    _, _, meta = load()
    print(f"checkpoint: iter {meta['iter']}, val loss {meta['val_loss']:.4f}\n")

    if mode_continue:
        print(text, end="", flush=True)
        for piece in stream(text, max_new_tokens=150):
            print(piece, end="", flush=True)
    else:
        print(f"You:  {text}")
        print("Flow:", end=" ", flush=True)
        for piece in chat_stream(text, max_new_tokens=100):
            print(piece, end="", flush=True)
    print()
