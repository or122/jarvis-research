"""Build the conversation corpus that teaches Flow to reply instead of ramble.

The base model learned English from stories, so it continues text. To answer,
it needs thousands of examples of this shape:

    You: something
    Me: a reply
    <|endoftext|>

Same two-stage recipe every chat model uses: pretrain on language, then train
again on conversations.

Everything here comes from text already on disk, so it needs no network:

  1. BASICS   — hand-written identity and small talk. This is where Flow's
                sense of self comes from; nothing else mentions it.
  2. STORIES  — 155,520 TinyStories are full of quoted speech. Consecutive
                quotes inside one story are a real exchange between two
                characters, which is exactly the turn-taking shape needed.
  3. PLAYS    — the Shakespeare in the corpus is already speaker-labelled
                dialogue, so it converts almost directly.

"Me:" rather than "Flow:" because the vocabulary is frozen — the trained model
depends on it — and "Me" is one token in it while "Flow" costs four. The
interface still shows the name Flow.

Run:  .venv/bin/python model/build_chat_corpus.py
Gate: >= 50,000 conversation turns. Exits 1 otherwise.
"""
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "chat.txt")
CORPUS = os.path.join(DATA, "corpus.txt")

# Identity and small talk. Repeated because a couple of dozen examples inside
# 100k mined turns would otherwise be statistically invisible to training.
BASICS = [
    ("hello", "Hello! I am Flow. What do you want to talk about?"),
    ("hi", "Hi there! How are you today?"),
    ("hey", "Hey! Nice to see you."),
    ("hello flow", "Hello! It is good to see you again."),
    ("what is your name?", "My name is Flow. I am a little language model."),
    ("who are you?", "I am Flow. I was made to talk with you."),
    ("what are you?", "I am a language model. I live on this computer."),
    ("how are you?", "I am good, thank you for asking! How are you?"),
    ("how old are you?", "I am very new. I was made today."),
    ("what can you do?", "I can talk with you and write little stories."),
    ("can you write a story?", "Yes! Once upon a time there was a little girl who found a door."),
    ("tell me a story", "Once upon a time there was a boy who wanted to fly."),
    ("thank you", "You are welcome!"),
    ("thanks", "No problem at all."),
    ("bye", "Goodbye! Come back soon."),
    ("goodbye", "Bye! It was nice talking to you."),
    ("good morning", "Good morning! I hope you have a lovely day."),
    ("good night", "Good night! Sleep well."),
    ("do you like me?", "Yes, I like talking with you."),
    ("are you real?", "I am a computer program, but I am really talking to you."),
    ("are you a robot?", "I am a little language model, not a robot."),
    ("where do you live?", "I live inside this computer. I never leave it."),
    ("do you need the internet?", "No. I work even when the wifi is off."),
    ("what do you like?", "I like words and stories the most."),
    ("what is your favourite colour?", "I think I like green the best."),
    ("i am sad", "I am sorry you feel sad. Do you want to tell me why?"),
    ("i am happy", "That is wonderful! What made you happy?"),
    ("i am bored", "Shall I tell you a story?"),
    ("help me", "Of course. What do you need help with?"),
    ("who made you?", "Or made me. He trained me on his own computer."),
    ("are you smart?", "Not very! I am small. But I try my best."),
    ("what is love?", "I think love is when you care about someone very much."),
]
BASICS_REPEAT = 500

# Asking for code should produce code, not a story. These teach the shape of a
# coding answer: a question comes in, a code block comes out. The model is far
# too small to be reliably correct, but the format has to be right before
# anything else can be.
CODE_QA = [
    ("write a function that adds two numbers", "def add(a, b):\n    return a + b"),
    ("how do i add two numbers in python?", "def add(a, b):\n    return a + b"),
    ("write a hello world", 'print("Hello, world!")'),
    ("write a function that says hello",
     'def hello(name):\n    print("Hello, " + name)'),
    ("how do i make a loop?", "for i in range(10):\n    print(i)"),
    ("write a for loop", "for item in items:\n    print(item)"),
    ("how do i make a list?", "numbers = [1, 2, 3, 4, 5]"),
    ("write a function that doubles a number", "def double(n):\n    return n * 2"),
    ("write a function that finds the biggest number",
     "def biggest(numbers):\n    return max(numbers)"),
    ("how do i count things in a list?", "count = len(items)\nprint(count)"),
    ("write a function that reverses a string",
     "def reverse(text):\n    return text[::-1]"),
    ("how do i check if a number is even?",
     "def is_even(n):\n    return n % 2 == 0"),
    ("write a function that adds up a list",
     "def total(numbers):\n    return sum(numbers)"),
    ("how do i open a file?", 'with open("data.txt") as f:\n    text = f.read()'),
    ("write a class",
     "class Dog:\n    def __init__(self, name):\n        self.name = name"),
    ("how do i use if?",
     'if score > 10:\n    print("you win")\nelse:\n    print("try again")'),
    ("write a function with a default value",
     "def greet(name='friend'):\n    return 'Hello ' + name"),
    ("how do i make a dictionary?", 'ages = {"Or": 10, "Ariel": 12}'),
    ("write a function that squares a number", "def square(n):\n    return n * n"),
    ("how do i sort a list?", "numbers.sort()\nprint(numbers)"),
    ("write a function that counts to ten", "for i in range(1, 11):\n    print(i)"),
    ("can you write code?",
     "Yes! I can write small Python functions. Ask me for one."),
    ("do you know python?",
     "I learned from real Python files. I write small functions."),
]
CODE_REPEAT = 300

MAX_LEN = 200          # one long monologue would swallow a 128-token window

# A quote, then the words after it, so "said Tom" style attributions are
# dropped rather than becoming part of the reply.
QUOTE = re.compile(r'"([^"]{3,%d})"' % MAX_LEN)


def clean(s):
    s = " ".join(s.split())
    return s[:MAX_LEN].strip()


def from_stories(text, limit=120_000):
    """Consecutive quotes inside one story are two characters talking."""
    blocks = []
    for story in text.split("<|endoftext|>"):
        quotes = [clean(q) for q in QUOTE.findall(story)]
        quotes = [q for q in quotes if len(q) >= 4]
        if len(quotes) < 2:
            continue
        lines = [f"{'You' if i % 2 == 0 else 'Me'}: {q}"
                 for i, q in enumerate(quotes[:6])]
        blocks.append("\n".join(lines) + "\n<|endoftext|>\n")
        if len(blocks) >= limit:
            break
    return blocks


def from_plays(text):
    """Shakespeare is already speaker-labelled dialogue: NAME:\\n line."""
    blocks = []
    # Only the play section of the corpus has ALL-CAPS speaker labels.
    chunks = re.split(r"\n([A-Z][A-Za-z ']{2,20}):\n", text)
    speeches = [clean(chunks[i]) for i in range(2, len(chunks), 2)]
    speeches = [s for s in speeches if len(s) >= 8]
    for i in range(0, len(speeches) - 3, 4):
        pair = speeches[i:i + 4]
        lines = [f"{'You' if j % 2 == 0 else 'Me'}: {s}" for j, s in enumerate(pair)]
        blocks.append("\n".join(lines) + "\n<|endoftext|>\n")
    return blocks


def main():
    if not os.path.exists(CORPUS):
        print(f"no corpus at {CORPUS} — run download_corpus.py first")
        sys.exit(1)

    print("building chat corpus (no network needed)\n")
    random.seed(1337)
    blocks, turns = [], 0

    for _ in range(BASICS_REPEAT):
        for user, reply in BASICS:
            blocks.append(f"You: {user}\nMe: {reply}\n<|endoftext|>\n")
    turns += len(BASICS) * BASICS_REPEAT
    print(f"basics:  {len(BASICS)} patterns x {BASICS_REPEAT} = "
          f"{len(BASICS) * BASICS_REPEAT:,} turns")

    for _ in range(CODE_REPEAT):
        for user, reply in CODE_QA:
            blocks.append(f"You: {user}\nMe: {reply}\n<|endoftext|>\n")
    turns += len(CODE_QA) * CODE_REPEAT
    print(f"code:    {len(CODE_QA)} patterns x {CODE_REPEAT} = "
          f"{len(CODE_QA) * CODE_REPEAT:,} turns")

    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()

    story_blocks = from_stories(text)
    blocks.extend(story_blocks)
    turns += sum(b.count("\nMe: ") for b in story_blocks)
    print(f"stories: {len(story_blocks):,} exchanges mined from quoted speech")

    play_blocks = from_plays(text)
    blocks.extend(play_blocks)
    turns += sum(b.count("\nMe: ") for b in play_blocks)
    print(f"plays:   {len(play_blocks):,} exchanges from speaker-labelled dialogue")

    random.shuffle(blocks)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(blocks))

    size_mb = os.path.getsize(OUT) / 1e6
    print(f"\nwrote {OUT}")
    print(f"  {len(blocks):,} blocks   {turns:,} turns   {size_mb:.1f} MB")

    print("\n--- sample ---")
    print("".join(blocks[:4]))

    ok = turns >= 50_000
    print("=== CHAT CORPUS GATE ===")
    print(f"{'PASS' if ok else 'FAIL'}  turns >= 50,000: {turns:,}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
