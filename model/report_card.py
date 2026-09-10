"""Flow's report card — honest scores, with the cheating removed.

The first version of this file was wrong in two ways, both found by Or asking
whether the scores were real:

  1. A four-word reply ("We love this toy,") scored 92.5% on English, because
     "100% distinct words" is trivially true when nothing has room to repeat.
     Length is now a hard requirement, not a bonus.

  2. Java scored 100% by grading answers that were written by hand and stored
     in the database. A test that grades its own answer key cannot fail.
     Every subject is now also asked HELD-OUT questions that are not in the
     database, so the model has to do the work.

Two scores per subject:

  APP    what a user actually gets - database, calculator and model together
  MODEL  what the neural network can do alone, with the database switched off

The APP score is what Flow delivers. The MODEL score is what was trained.
Both are true, and reporting only one of them would be misleading.

Run:  .venv/bin/python model/report_card.py
"""
import ast
import random
import re
import sys

import torch

WORDS = "/usr/share/dict/words"
MIN_STORY_WORDS = 40      # below this, prose scores are scaled down
TARGETS = {"Maths": 88, "English": 79, "Java": 80, "Python": 60, "Creativity": 70}


def bar(pct, width=20):
    n = round(width * max(pct, 0) / 100)
    return "█" * n + "·" * (width - n)


def balanced(text):
    stack, pairs = [], {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack


# --------------------------------------------------------------- maths
def grade_maths(ask):
    """Randomly generated sums, so nothing can be a memorised answer."""
    rnd = random.Random(20260910)
    tests = []
    for _ in range(20):
        a, b = rnd.randint(2, 99), rnd.randint(2, 99)
        op = rnd.choice(["plus", "minus", "times"])
        expect = {"plus": a + b, "minus": a - b, "times": a * b}[op]
        tests.append((f"what is {a} {op} {b}", str(expect)))

    right = 0
    for q, expect in tests:
        got = ask(q, max_new_tokens=12)
        # The answer must be the number, not merely contain a digit that
        # happens to appear inside a longer wrong number.
        if re.search(rf"(?<![\d.]){re.escape(expect)}(?![\d.])", got):
            right += 1
    return 100.0 * right / len(tests), f"{right}/{len(tests)} random sums correct"


# ------------------------------------------------------------- english
def grade_english(ask, dictionary):
    """Measured on generated prose, with length as a hard requirement."""
    text = ask("tell me a story", max_new_tokens=120, temperature=0.75, top_k=40)
    words = re.findall(r"[a-zA-Z']+", text)
    n = len(words)
    if n < 5:
        return 0.0, f"produced only {n} words - nothing to judge"

    real = sum(1 for w in words if w.lower().strip("'") in dictionary) / n
    distinct = len(set(w.lower() for w in words)) / n
    sents = [s for s in re.split(r"[.!?]", text) if s.strip()]
    avg = sum(len(s.split()) for s in sents) / max(len(sents), 1)
    shape = 1.0 if 5 <= avg <= 25 else 0.6 if 3 <= avg <= 35 else 0.2

    # Distinctness only counts once there is enough text for repetition to be
    # possible at all; below that it is meaningless and scores nothing.
    variety = min(distinct / 0.6, 1.0) if n >= 25 else 0.0
    enough = min(n / MIN_STORY_WORDS, 1.0)

    raw = 100 * (0.40 * real + 0.25 * variety + 0.20 * shape + 0.15 * enough)
    # Short output scales the whole score, so four words can never score high.
    pct = raw * enough
    return pct, (f"{n} words ({enough:.0%} of the {MIN_STORY_WORDS} needed), "
                 f"{real:.0%} real, {distinct:.0%} distinct, {avg:.0f}/sentence")


# ---------------------------------------------------------------- java
JAVA_KNOWN = ["write a java hello world", "write a java class",
              "how do i make a loop in java"]
# Not in the database. The model has to answer these itself.
JAVA_HELDOUT = ["write a java function that counts to five",
                "how do i make a while loop in java",
                "write a java method that returns a name"]


def score_java_answer(text):
    marks = 0.0
    marks += 0.3 * any(k in text for k in ("class", "static", "public", "void"))
    marks += 0.2 * (";" in text)
    marks += 0.2 * balanced(text)
    marks += 0.3 * (text.count("{") >= 1 and text.count("{") == text.count("}"))
    return marks


def grade_java(ask, held_out_only):
    qs = JAVA_HELDOUT if held_out_only else JAVA_KNOWN + JAVA_HELDOUT
    total = sum(score_java_answer(ask(q, max_new_tokens=90, temperature=0.3,
                                      top_k=20)) for q in qs)
    label = "held-out only" if held_out_only else f"{len(JAVA_KNOWN)} known + {len(JAVA_HELDOUT)} unseen"
    return 100.0 * total / len(qs), f"{total:.1f}/{len(qs)} well-formed ({label})"


# -------------------------------------------------------------- python
PY_QS = ["write a function that adds two numbers", "write a hello world",
         "how do i make a loop?", "write a function that doubles a number",
         "how do i make a list?", "write a class"]


def grade_python(ask):
    parsed = 0
    for q in PY_QS:
        r = ask(q, max_new_tokens=50, temperature=0.3, top_k=20)
        m = re.search(r"^(def |class |print|for |if |while |import |[a-z_]+ = )",
                      r, re.M)
        try:
            ast.parse((r[m.start():] if m else r).strip())
            parsed += 1
        except (SyntaxError, ValueError):
            pass
    return 100.0 * parsed / len(PY_QS), f"{parsed}/{len(PY_QS)} valid Python"


# ---------------------------------------------------------- creativity
COMMON = {"the", "a", "an", "and", "was", "were", "is", "are", "to", "of",
          "in", "it", "he", "she", "they", "said", "had", "his", "her",
          "you", "i", "that", "for", "on", "with", "but", "not", "be",
          "very", "so", "day", "one", "little", "big", "good", "happy",
          "went", "saw", "then", "there", "this", "have", "we", "at"}


def grade_creativity(ask, dictionary):
    pieces = [ask("tell me a story", max_new_tokens=120, temperature=0.9, top_k=50)
              for _ in range(2)]
    text = " ".join(pieces)
    words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
    n = len(words)
    if n < 10:
        return 0.0, f"produced only {n} words"

    real = [w for w in words if w.strip("'") in dictionary]
    rare = len([w for w in real if w not in COMMON]) / max(len(real), 1)
    distinct = len(set(real)) / max(len(real), 1)

    seen, run, best = set(), 0, 0
    for w in words:
        if w in seen:
            seen, run = set(), 0
        seen.add(w)
        run += 1
        best = max(best, run)

    # Same length requirement: two "stories" of four words are not creative.
    enough = min(n / (MIN_STORY_WORDS * 2), 1.0)
    raw = 100 * (0.4 * min(rare / 0.55, 1) + 0.35 * min(distinct / 0.65, 1) +
                 0.25 * min(best / 25.0, 1))
    return raw * enough, (f"{n} words, {rare:.0%} uncommon, {distinct:.0%} "
                          f"distinct, {best} before repeating")


def main():
    from knowledge import look_up
    from sample import chat, load

    model, _, meta = load()
    with open(WORDS, encoding="utf-8", errors="ignore") as f:
        dictionary = {w.strip().lower() for w in f}

    def ask_app(q, **kw):
        """What a user gets: database and calculator first, model second."""
        fact, _ = look_up(q)
        return fact if fact else chat(q, **kw)

    def ask_model(q, **kw):
        """The neural network alone, with the database switched off."""
        return chat(q, **kw)

    print(f"Flow — {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, "
          f"iteration {meta['iter']}, val loss {meta['val_loss']:.3f}\n")

    rows = []
    for name, fn in [
        ("Maths", lambda ask, ho: grade_maths(ask)),
        ("English", lambda ask, ho: grade_english(ask, dictionary)),
        ("Java", lambda ask, ho: grade_java(ask, ho)),
        ("Python", lambda ask, ho: grade_python(ask)),
        ("Creativity", lambda ask, ho: grade_creativity(ask, dictionary)),
    ]:
        torch.manual_seed(1337)
        app_pct, app_note = fn(ask_app, False)
        torch.manual_seed(1337)
        model_pct, _ = fn(ask_model, True)
        rows.append((name, app_pct, model_pct, app_note))

    print(f"{'':11s} {'FLOW (app)':>11s}   {'MODEL alone':>11s}")
    print("=" * 66)
    met = 0
    for name, app_pct, model_pct, note in rows:
        target = TARGETS[name]
        ok = app_pct >= target
        met += ok
        print(f"{name:11s} {app_pct:9.1f}%   {model_pct:9.1f}%   "
              f"target {target}%  {'MET' if ok else 'below'}")
        print(f"{'':11s} {bar(app_pct)}  {note}")
    print("=" * 66)

    app_avg = sum(r[1] for r in rows) / len(rows)
    model_avg = sum(r[2] for r in rows) / len(rows)
    print(f"\n{'OVERALL':11s} {app_avg:9.1f}%   {model_avg:9.1f}%   "
          f"{met}/{len(rows)} targets met\n")

    print("What the two columns mean")
    print("  FLOW (app)   what you get in the app: calculator for sums, stored")
    print("               snippets for Java, the model for everything else.")
    print("  MODEL alone  the neural network with the database switched off,")
    print("               and asked only questions it was never given answers to.")
    print("\nWhy they differ so much: an 11M-parameter model cannot do arithmetic")
    print("or write Java. Flow scores well by not asking it to.")


if __name__ == "__main__":
    main()
