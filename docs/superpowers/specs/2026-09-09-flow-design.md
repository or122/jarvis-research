# Flow — Design

**Date:** 2026-09-09
**Owner:** Or Gefen
**Status:** Approved, building

---

## What Flow is

An AI chat app whose brain is a language model Or built and trained himself.

**No Claude API. No OpenAI. No rented model.** Or made this decision with the
limits fully explained (see "The honest limits" below) and reaffirmed it.

## What Flow can honestly claim that Claude cannot

These are real, and they follow directly from the decision above:

- **Free forever** — there is no API bill, so there is no per-message cost
- **Works offline** — the model lives on the machine; turn the wifi off
- **Completely private** — nothing typed ever leaves the computer
- **Actually owned** — every weight was produced by code in this repo

## The honest limits

Flow's model will **not** answer factual questions, follow instructions
reliably, or reason. It is roughly 9 million parameters trained overnight on
one CPU. Claude is ~10,000× larger and cost ~$100M+ to train.

What it *will* do: produce fluent, grammatical, conversational-sounding English.
It will sound like someone talking without saying anything checkable.

Flow's marketing must never claim otherwise. "Private, offline, yours, free" is
true. "As smart as Claude" is not, and would be a lie to users.

## The machine

MacBook Pro, **Intel** i5-1038NG7, 16 GB, **no GPU**. PyTorch pinned to
`2.2.2` (last macOS x86_64 build). Sustained training thermally throttles —
a 15-minute estimate ran 55 minutes on Tiny-GPT. Plan for 2× estimates.

---

## Part 1 — The model

Four upgrades over Tiny-GPT, ordered by how much they improve output:

| Lever | Tiny-GPT | Flow | Why it matters |
|---|---|---|---|
| Training data | 1 MB Shakespeare | ~150 MB mixed text | Biggest single win |
| Tokenizer | characters | **words** (8k vocab) | Capacity stops being spent on spelling |
| Parameters | 0.8M | ~9M | Longer coherence |
| Training | 55 min | overnight | More of all of the above |

### Corpus — "everything I can find", capped by reality

Or asked for the biggest pile of text possible. The real cap is not disk, it is
CPU: at roughly 1,500 tokens/second, an overnight run consumes about **50M
tokens**. More text than that is never seen even once, so the corpus targets
**~150 MB** (~35M words) — enough that the model never has to repeat itself.

Sources, in priority order. Each is optional; the downloader reports what it
actually got and continues past failures:

| Source | Size | Why |
|---|---|---|
| TinyStories | ~120 MB | Simple, clean English. The published result behind it is that ~10M-parameter models trained on it produce *coherent* text — exactly Flow's size class. |
| Project Gutenberg books | ~10 MB | Real sentence variety and vocabulary |
| Tiny Shakespeare | 1 MB | Already on disk; adds dialogue formatting |

**Gate:** ≥ 100 MB collected, ≥ 95% printable ASCII, ≥ 500k sentences.

### Tokenizer — word-level, written from zero

Character-level spends most of a small model's capacity learning to spell.
Word-level hands it whole words and frees that capacity for meaning.

- 7,936 most frequent words + 256 single-byte fallbacks = **8,192 vocab**
- Any word not in the vocab is spelled out in bytes, so **every possible input
  encodes** — no unknown-token holes
- Written by hand, like Tiny-GPT's. Not BPE: proper BPE training on 150 MB in
  pure Python takes hours, and at this scale word-level captures most of the
  benefit for a fraction of the complexity.

**Gate:** exact round-trip on held-out text; ≥ 85% of tokens are real words
rather than byte fallbacks; ≥ 3.0 characters per token.

### Model — ~9M parameters

```
vocab_size = 8192   block_size = 128   n_embd = 256
n_head = 8          n_layer = 6        dropout = 0.1
```

Sized so embeddings (~4.2M) and transformer blocks (~4.7M) are balanced.
Same `model.py` architecture as Tiny-GPT, scaled. Reused, not rewritten.

**Gates:** untrained loss ≈ `ln(8192)` = 9.011 (window 8.7–9.3); trained
validation loss < 4.5.

### Training

AdamW, `lr=3e-4` (lower than Tiny-GPT's 1e-3 — bigger models need gentler
steps), batch 32, overnight. Checkpoint saved only on validation improvement,
so an interrupted run still leaves the best model on disk.

**Gate:** 100 generated tokens are ≥ 80% real English words and contain
≥ 3 sentences ending in punctuation.

---

## Part 2 — The app

### Shape

```
React + Vite (TypeScript)  ──HTTP──>  Python inference server  ──>  model.pt
        the UI Or owns                 the only Python in Flow
```

Two processes, no database, no Express layer. The model is PyTorch, so
inference must be Python; everything Or will actually edit is TypeScript.

Skipping a Node backend is deliberate: it would only forward requests. One
fewer process, one fewer thing to break.

### v1 is exactly this

1. A chat window
2. Type a message, Flow's model replies, words streaming in as generated
3. History persists across refresh (`localStorage`)
4. Free tier: 20 messages/day. Paid tier: unlimited. Shown in the UI.
5. It looks good

**Not in v1:** rooms, invites, other people, accounts, real payments, file
uploads, deploying to the internet. Rooms and seats are v2.

### Subscriptions

Tiers exist in the UI and the limit is enforced, but **no real money changes
hands in v1** — "paid" is a flag flipped by hand. Real payments are v2 and are
a hot zone: ask before touching.

The pricing is honest by construction: free users get the small model, paid
users get bigger and more models. Both cost Or CPU time, which is a real cost.

**Gate:** page loads with **zero** console errors; a typed message returns a
reply in under 5 seconds; message 21 on the free tier is blocked.

---

## Build order

| # | Step | Estimate |
|---|---|---|
| 1 | Corpus downloader + gate | 1 hr |
| 2 | Word tokenizer from zero + gate | 2 hr |
| 3 | Scale model to 9M + wiring gate | 1 hr |
| 4 | Train overnight + gate | 10–14 hr unattended |
| 5 | Python inference server (streaming) | 2 hr |
| 6 | React + TypeScript chat UI | 4 hr |
| 7 | Tiers and limits | 1 hr |

Steps 5–7 can be built while step 4 trains.

## Risks

| Risk | Sign | Response |
|---|---|---|
| A download source is dead | Downloader reports < 100 MB | Continue with what arrived; add Gutenberg books until the gate passes |
| Training too slow | < 500 tokens/sec | Drop `n_layer` to 4 and `block_size` to 64 |
| Output still gibberish at loss < 4.5 | Gate 3 fails | Vocabulary too small or data too varied — retrain on TinyStories alone |
| Mac sleeps mid-run | Training stops early | `caffeinate -i` the training command |
