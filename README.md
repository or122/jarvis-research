# Jarvis — a language model built from zero, on one laptop

I'm Or Gefen. I'm 10. I trained this language model myself, from nothing, on my
MacBook — no API, no borrowed weights, no pretrained anything. Then I built an
assistant on top of it.

It runs with the wifi off.

```
You:  what is your name
Jarvis: My name is Jarvis. I am Or's assistant.

You:  what is 89 times 47
Jarvis: 4183

You:  open safari
Jarvis: Opening Safari.          ← actually opens it

You:  make me a game
Jarvis: I can't build a game. That needs hundreds of lines of working
        code and I write about ten. Ask Claude for that one.
```

That last answer is the most important one in the repo. More on that below.

---

## What it is

| | |
|---|---|
| **Model** | 11,070,976 parameters, decoder-only transformer, written by hand |
| **Trained on** | 104.5M tokens: children's stories, classic books, Python source, mined conversations |
| **Loss** | 9.51 → 2.59 (started at `ln(12288)` — pure random guessing) |
| **Hardware** | One Intel MacBook Pro. No GPU. |
| **Cost** | nothing |

## How it works

Most of what an assistant does isn't thinking. So the model is the **last**
thing tried, not the first:

```
      you say something
             ↓
      1. a command?    →  real code runs it     always correct
      2. a sum?        →  a calculator          always correct
      3. a known fact? →  a database            always correct
      4. anything else →  the model             sounds right, may be wrong
```

Because of that split, Jarvis scores **100% on maths** while the model alone
scores **0%** — a neural network has never counted. Knowing which problems
*not* to give the model turned out to matter as much as improving it.

## Honest scores

Measured by [`model/report_card.py`](model/report_card.py), which grades on
randomly generated sums and on questions held out of the database, so nothing
can be memorised.

| Subject | Whole app | Model alone |
|---|---|---|
| Maths | 100% | 0% |
| English | 89.9% | 89.9% |
| Java | 83.3% | 20.0% |
| Creativity | 78.9% | 78.9% |
| Python | 66.7% | 66.7% |
| **Overall** | **83.7%** | **51.1%** |

> The first version of this test said 78/100. It was wrong — a four-word reply
> scored 92.5% on English, and the Java test was grading answers I'd written
> myself. Rebuilding it honestly dropped the score to 42%, and immediately
> exposed a real bug. **An eval you can't fail teaches you nothing.**

## What it cannot do

It will not build you a game, an app or a website. It will not answer facts
nobody taught it. It cannot reason.

It's 11 million parameters. Claude is roughly ten thousand times bigger and
cost over $100M to train. That gap doesn't close on a laptop, and this repo
never pretends otherwise — Jarvis is taught to say so out loud.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # torch 2.2.2, numpy

# the assistant, in your terminal
.venv/bin/python model/jarvis.py "what time is it"

# the chat app
cd model && ../.venv/bin/python server.py      # terminal 1
cd web && npm install && npm run dev           # terminal 2 -> localhost:5173
```

`torch` is pinned to **2.2.2** — the last version with a macOS Intel build.

> The trained weights aren't in the repo (they're 130 MB). Train your own with
> the steps below, or ask me for the checkpoint.

## Build the model yourself

Every step, from an empty folder to a talking model:

```bash
.venv/bin/python model/download_corpus.py      # 138 MB of stories
.venv/bin/python model/build_code_corpus.py    # Python from your own machine
.venv/bin/python model/build_chat_corpus.py    # 151,695 conversation turns
.venv/bin/python model/build_jarvis_corpus.py  # 33,000 assistant turns
.venv/bin/python model/tokenizer.py            # 12,288-word vocabulary
.venv/bin/python model/prepare_all_tokens.py   # 104.5M tokens
./train_chunked.sh 400 20                      # train in bursts
.venv/bin/python model/report_card.py          # score it
```

`train_chunked.sh` trains ~35 minutes then rests 15. That isn't fussiness:
running flat out, this laptop fell from 190 to 28 GFLOPS and effectively
stopped. Chunked, it holds ~2,000 tokens/sec. **Same code, 4x the throughput,
purely from letting it cool.**

## What's in here

| File | Job |
|---|---|
| `model/model.py` | `Head` -> `MultiHeadAttention` -> `FeedForward` -> `Block` -> `FlowLM` |
| `model/tokenizer.py` | Words <-> numbers. 12,032 words + 256 byte fallbacks |
| `model/train.py` | Training loop: warmup, cosine decay, resumable checkpoints |
| `model/jarvis.py` | The assistant — 15 real commands, swappable brain |
| `model/knowledge.py` | Facts database. Where Jarvis's identity actually lives |
| `model/maths.py` | The calculator. Safe expression evaluation, never `eval()` |
| `model/rooms.py` | Shared rooms with paid seats |
| `model/report_card.py` | The honest test |
| `web/` | React + TypeScript chat interface |

## Things I learned the hard way

- **`<|endoftext|>` in the output wasn't a bug.** The corpus separates its
  155,520 stories with it, so the model was right to learn it. The fix was
  treating it as a stop signal — what every real LLM does.
- **A model has no idea when to stop.** Left alone it writes your next message
  too, then answers itself forever. Stopping is the program's job.
- **A model has no idea what it can do either.** Asked "can you build an app"
  it said yes, because agreement is a likely next word. Honesty has to be
  written down, not hoped for.
- **CPU% tells you a process is busy, never that it's fast.** Comparing
  CPU-time against wall-time is what exposed both the thermal throttling and
  the laptop silently sleeping through training.
- **Never edit a running shell script.** Bash reads it as it goes.

## Licence

MIT. Take any of it.
