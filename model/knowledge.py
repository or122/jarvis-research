"""Flow's memory — facts it looks up instead of trying to remember.

An 11M-parameter model physically cannot store facts. There is no room. So
Flow does what large systems do: it checks a database first, and only falls
back to the model when it finds nothing.

  facts     -> a real, correct answer
  no fact   -> the model talks

Facts always beat the model, because a small model will happily invent a
confident wrong answer and a database will not.

Anyone can teach Flow: `teach("question", "answer")` and it knows that
forever, instantly, with no retraining.

Run:  .venv/bin/python model/knowledge.py          seed and self-test
      .venv/bin/python model/knowledge.py "hello"  look one up
"""
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "knowledge.db")

# Below this share of matched words, the lookup is a guess rather than a hit,
# so the model answers instead. Tuned by the self-test at the bottom: 0.5 keeps
# "what is your name" matching while rejecting unrelated questions.
THRESHOLD = 0.5

# Words that appear in almost every question and so carry no signal.
STOP = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "will", "would", "i", "you", "me", "my", "your",
        "of", "to", "in", "on", "for", "it", "that", "this", "and", "or",
        "what", "whats", "how", "who", "please", "tell", "s"}

SEED = [
    # --- who Flow is ---
    ("what is your name", "My name is Flow."),
    ("who are you", "I am Flow, a language model built from zero on this Mac."),
    ("what are you", "I am a small language model. About 11 million parameters."),
    ("who made you", "Or Gefen made me. He wrote my code and trained me himself."),
    ("how were you made", "I was trained from zero on a MacBook Pro, with no API "
                          "and no borrowed model."),
    ("how big are you", "I have about 11 million parameters. Claude has roughly "
                        "10,000 times more."),
    ("are you claude", "No. I am Flow. I am much smaller, but I am entirely Or's."),
    ("do you need the internet", "No. I run on this computer and work with the "
                                 "wifi turned off."),
    ("is my chat private", "Yes. Nothing you type ever leaves this computer."),
    ("what can you do", "I can chat, write little stories, and look up facts I "
                        "have been taught."),
    ("what are you bad at", "Facts I was never taught, maths I cannot look up, "
                            "and writing code that really runs."),

    # --- simple maths, which a small model gets wrong constantly ---
    ("what is 2 plus 2", "4"),
    ("what is 10 times 10", "100"),
    ("how many days in a year", "365, or 366 in a leap year."),
    ("how many hours in a day", "24"),
    ("how many minutes in an hour", "60"),

    # --- things Or is likely to ask ---
    ("what is python", "A programming language. It is what I am written in."),
    ("what is javascript", "A programming language that runs in web browsers."),
    ("what is a language model", "A program that predicts the next word. Do that "
                                 "well enough and it looks like talking."),
    ("what is a token", "A piece of text a model reads at once. For me it is "
                        "usually a whole word."),
    ("what is training", "Showing a model lots of text so it learns to predict "
                         "what comes next."),
    ("what is israel", "A country in the Middle East. Or lives there."),

    # --- small talk worth getting exactly right ---
    ("hello", "Hello! I am Flow. What would you like to talk about?"),
    ("hi", "Hi! Good to see you."),
    ("goodbye", "Goodbye! Come back soon."),
    ("thank you", "You are welcome!"),
    ("how are you", "I am well, thank you. How are you?"),
]


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("""CREATE TABLE IF NOT EXISTS facts (
                    id       INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer   TEXT NOT NULL,
                    taught   TEXT DEFAULT CURRENT_TIMESTAMP)""")
    db.commit()
    return db


def keywords(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    kept = [w for w in words if w not in STOP]
    # If a question is nothing but stop words ("how are you"), keep them —
    # something must be matched on.
    return set(kept or words)


def teach(question, answer):
    """Add a fact. Replaces any existing answer to the same question."""
    db = connect()
    db.execute("DELETE FROM facts WHERE question = ?", (question.lower().strip(),))
    db.execute("INSERT INTO facts (question, answer) VALUES (?, ?)",
               (question.lower().strip(), answer.strip()))
    db.commit()
    db.close()


def look_up(question):
    """Return (answer, confidence) for the best match, or (None, 0.0).

    Scored by how much of the stored question's keywords the asked question
    covers. Asking a longer question than the stored one is fine; a stored
    fact matching only part of what was asked is not.
    """
    db = connect()
    rows = db.execute("SELECT question, answer FROM facts").fetchall()
    db.close()

    asked = keywords(question)
    if not asked:
        return None, 0.0

    best, best_score = None, 0.0
    for stored_q, answer in rows:
        stored = keywords(stored_q)
        if not stored:
            continue
        overlap = len(asked & stored)
        # Both directions matter: the stored fact must be well covered, and
        # the question must not be mostly about something else.
        score = (overlap / len(stored)) * 0.7 + (overlap / len(asked)) * 0.3
        if score > best_score:
            best, best_score = answer, score

    return (best, best_score) if best_score >= THRESHOLD else (None, best_score)


def count():
    db = connect()
    n = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    db.close()
    return n


def seed():
    for q, a in SEED:
        teach(q, a)
    return len(SEED)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer, score = look_up(sys.argv[1])
        print(f"asked:  {sys.argv[1]}")
        print(f"match:  {score:.2f}")
        print(f"answer: {answer if answer else '(no fact — the model would answer)'}")
        sys.exit(0)

    n = seed()
    print(f"seeded {n} facts -> {DB}\n")

    # --- gate ---------------------------------------------------------
    # Two halves that can each fail: known questions must be found, and
    # unknown ones must NOT be answered with a confident wrong fact.
    should_hit = [
        ("what is your name", "Flow"),
        ("who made you", "Or"),
        ("what is 2 plus 2", "4"),
        ("hello", "Hello"),
        ("do you need the internet", "wifi"),
        ("how many days in a year", "365"),
    ]
    should_miss = [
        "what is the capital of france",
        "tell me about dinosaurs",
        "why is the sky blue",
        "what did i eat yesterday",
    ]

    hits = 0
    print("questions Flow should KNOW:")
    for q, expect in should_hit:
        answer, score = look_up(q)
        ok = answer is not None and expect.lower() in answer.lower()
        hits += ok
        print(f"  {'OK  ' if ok else 'MISS'} {q:32s} {score:.2f}  {answer}")

    misses = 0
    print("\nquestions Flow should NOT pretend to know:")
    for q in should_miss:
        answer, score = look_up(q)
        ok = answer is None
        misses += ok
        print(f"  {'OK  ' if ok else 'BAD '} {q:32s} {score:.2f}  "
              f"{'(passed to the model)' if ok else answer}")

    hit_rate = hits / len(should_hit)
    miss_rate = misses / len(should_miss)
    print("\n=== KNOWLEDGE GATE ===")
    print(f"{'PASS' if hit_rate == 1.0 else 'FAIL'}  finds what it knows: "
          f"{hits}/{len(should_hit)}")
    print(f"{'PASS' if miss_rate == 1.0 else 'FAIL'}  refuses what it doesn't: "
          f"{misses}/{len(should_miss)}")

    ok = hit_rate == 1.0 and miss_rate == 1.0
    print(f"\n{'KNOWLEDGE READY' if ok else 'NOT READY'}  ({count()} facts stored)")
    sys.exit(0 if ok else 1)
