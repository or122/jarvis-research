"""Generate text from Flow's trained model.

Two modes:
  continue  — the model carries your text on (what a raw language model does)
  chat      — the model replies to you (what Flow's app uses)

Run:  .venv/bin/python model/sample.py "hello"
      .venv/bin/python model/sample.py --continue "Once upon a time"
"""
import os
import re
import sys

import torch

from knowledge import look_up
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

        # A checkpoint and a vocabulary that disagree is a silent disaster:
        # every token id means something different, so the model emits fluent
        # nonsense and nothing raises. Fail loudly instead.
        if tok.vocab_size != ckpt["vocab_size"]:
            raise SystemExit(
                f"vocabulary mismatch: ckpt.pt was trained with "
                f"vocab_size={ckpt['vocab_size']} but data/vocab.json has "
                f"{tok.vocab_size}. They must come from the same training run."
            )

        model = FlowLM(ckpt["vocab_size"])
        model.load_state_dict(ckpt["model"])
        model.eval()          # dropout off
        _cache.update(model=model, tok=tok, meta=ckpt)
    return _cache["model"], _cache["tok"], _cache["meta"]


def stream(prompt, max_new_tokens=120, temperature=0.8, top_k=40, stops=STOPS,
           min_new_tokens=0):
    """Yield decoded text as it is generated, ending at any stop sequence.

    min_new_tokens ignores stop sequences until that many tokens have been
    produced. The chat data is full of short mined turns, so the model learned
    to stop after about a dozen words - fine for "hello", useless for "tell me
    a story". Every real chat model has the same control.
    """
    model, tok, _ = load()
    ids = tok.encode(prompt) or [tok.stoi.get(" the", 256)]
    idx = torch.tensor([ids], dtype=torch.long)

    # A stop sequence spans several word-level tokens, so text is accumulated
    # and a short tail is held back in case it turns out to be the beginning
    # of one. Without that, "\nYou" would be shown before ":" arrives.
    hold = max(len(s) for s in stops) - 1
    buffer, sent, produced = "", 0, 0

    for token_id in model.generate(idx, max_new_tokens, temperature, top_k):
        buffer += tok.decode([token_id])

        produced += 1
        hits = [] if produced < min_new_tokens else [buffer.find(s) for s in stops]
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


LONG_ASKS = ("story", "tell me about", "write about", "poem", "describe",
             "imagine", "once upon")


def story_opening(message):
    """Turn a request into the first words of a story to continue from.

    "tell me about a dragon" -> "Once upon a time there was a dragon." The
    model then has something to carry on, which is exactly what its story and
    book training taught it to do.
    """
    text = re.sub(r"^\s*(please\s+)?(can you\s+)?(tell|write|make)\s+"
                  r"(me\s+)?(a|an|the)?\s*", "", message.strip(), flags=re.I)
    text = re.sub(r"^(story|poem)\s*(about|of)?\s*", "", text, flags=re.I).strip(" .?!")
    if text:
        return f"Once upon a time there was {text}."
    return "Once upon a time"


def chat_stream(message, **kw):
    """Ask the model to reply, rather than to carry on writing.

    The prompt ends mid-line at "Me:" on purpose: the model's job is to
    complete that line, and completing it is a reply.
    """
    # A story request is not a conversational turn. Forced to keep going in
    # chat mode the model just writes both sides of another dialogue - it was
    # trained on short mined turns, so that is what "more" means to it.
    # Stories come from the story and book data instead, in continuation mode.
    if any(k in message.lower() for k in LONG_ASKS):
        kw.pop("min_new_tokens", None)
        opening = story_opening(message)

        def with_opening():
            # The opening is part of the story, not hidden scaffolding -
            # without it the reply starts mid-sentence on a comma.
            yield opening
            yield from stream(opening, stops=("<|", "\nYou:"), **kw)

        return with_opening()
    return stream(f"You: {message}\nMe:", **kw)


def generate(prompt, **kw):
    return "".join(stream(prompt, **kw))


def answer(message, **kw):
    """What Flow says, and where it came from.

    The database is checked first. An 11M-parameter model cannot hold facts,
    so anything it "remembers" is invented - a stored fact is always the
    better answer when one exists. The model handles everything else.

    Returns (text, source) where source is "memory" or "model".
    """
    fact, _ = look_up(message)
    if fact:
        return fact, "memory"
    return "".join(chat_stream(message, **kw)).strip(), "model"


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
