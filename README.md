# Flow

An AI chat app whose brain is a language model built and trained from zero on
one MacBook Pro.

No Claude API. No OpenAI. No rented model. Every weight came out of code in
this repository.

## What Flow can honestly claim

- **Free forever** — there is no API bill, so there is no per-message cost
- **Works offline** — the model lives on the machine; turn the wifi off
- **Completely private** — nothing typed ever leaves the computer
- **Actually owned** — the tokenizer, the attention maths, the training loop

## What Flow cannot do

It will not answer factual questions, follow instructions reliably, or reason.
It is ~11 million parameters trained on one CPU. Claude is roughly ten thousand
times larger and cost over $100M to train.

What it does: fluent, grammatical, conversational English, and code-shaped
Python. It sounds like someone talking without saying anything checkable.

Flow's marketing must never claim otherwise. "Private, offline, yours, free" is
true. "As smart as Claude" is not, and would be a lie to users.

## Run it

Two processes. The model is PyTorch so inference must be Python; everything
else is TypeScript.

```bash
# terminal 1 — the model
.venv/bin/python model/server.py

# terminal 2 — the app
cd web && npm run dev      # http://localhost:5173
```

First-time setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`torch` is pinned to **2.2.2** — the last version with a macOS Intel build.

## Build the model from scratch

```bash
.venv/bin/python model/download_corpus.py      # 138 MB of stories
.venv/bin/python model/build_code_corpus.py    # 31 MB of Python from this Mac
.venv/bin/python model/build_chat_corpus.py    # 151k conversation turns
.venv/bin/python model/tokenizer.py            # 12,288-word vocabulary
.venv/bin/python model/prepare_all_tokens.py   # 74.5M tokens
caffeinate -i .venv/bin/python model/train.py 6000 all_
.venv/bin/python model/eval.py                 # 3 gates
.venv/bin/python model/eval_code.py            # 4 code gates
```

## How it works

| File | Job |
|---|---|
| `model/tokenizer.py` | Words ↔ numbers. 12,032 words + 256 byte fallbacks |
| `model/model.py` | `Head` → `MultiHeadAttention` → `FeedForward` → `Block` → `FlowLM` |
| `model/train.py` | Training loop, warmup, cosine decay, resumes from checkpoint |
| `model/sample.py` | Generation, chat mode, stop sequences |
| `model/server.py` | Streaming HTTP server, standard library only |
| `model/eval.py` | The gates. Exit 1 = not done |
| `web/src/App.tsx` | The chat interface, free/paid tiers |

The model is a decoder-only transformer — the same family as GPT and Claude,
scaled down about ten thousand times.

```
vocab 12,288   context 128 tokens   n_embd 256
n_head 8       n_layer 6            11.07M parameters
```

## Training data

| Source | Size | Why |
|---|---|---|
| TinyStories | 133 MB | Simple, clean English. ~10M-parameter models produce coherent text on it |
| Project Gutenberg | 4 MB | Real sentence variety |
| Python stdlib + packages | 31 MB | 2,679 files already on this Mac. Learn what code looks like |
| Mined conversations | 17 MB | 151,695 turns, so Flow replies instead of rambling |

Mixed into 74.5M training tokens: 49% stories, 30% code, 21% chat.

## The two-stage recipe

Flow is trained the way every chat model is:

1. **Pretrain** — learn language from a large pile of text
2. **Fine-tune** — learn the shape of a conversation from `You:` / `Me:` pairs

Conversations are oversampled so they keep a meaningful share of what the model
sees, and story text is kept in the mix so its English does not erode. That
second trick is called **replay**, and skipping it is a classic way to ruin a
fine-tune.

## Pricing

Free tier: 20 messages a day. Paid: unlimited.

**No real payments in v1, on purpose.** The seat limit and the upgrade screen
are real; who counts as "paid" is a flag flipped by hand. Real money is v2, and
payments are a hot zone — ask before touching.

## Not in v1

Rooms, invites, other people, accounts, file uploads, deploying to the
internet. Rooms and per-seat pricing are v2.
