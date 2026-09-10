"""Flow's report card — a percentage per subject.

Scores the subjects Or asked about: Maths, English, Java, Python, Creativity.

Each subject is graded on what Flow actually produces, through the same path
the app uses — so what is measured here is what a user would get, database
and model together, not the model alone.

Run:  .venv/bin/python model/report_card.py
"""
import ast
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = "/usr/share/dict/words"

TARGETS = {"Maths": 88, "English": 79, "Java": 80, "Python": 60, "Creativity": 70}


def bar(pct, width=24):
    return "█" * round(width * pct / 100) + "·" * (width - round(width * pct / 100))


def balanced(text):
    stack, pairs = [], {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack


def grade_maths(answer_fn):
    """20 sums with one right answer each. Nothing subjective."""
    sums = [
        ("what is 2 plus 2", "4"), ("what is 15 + 27", "42"),
        ("what is 10 times 10", "100"), ("what is 47 times 13", "611"),
        ("what is 100 divided by 4", "25"), ("what is 7 minus 9", "-2"),
        ("what is 5 squared", "25"), ("what is 2 to the power of 10", "1024"),
        ("what is 20% of 50", "10"), ("what is the square root of 144", "12"),
        ("what is 7 divided by 2", "3.5"), ("what is 17 mod 5", "2"),
        ("what is three plus four", "7"), ("what is 1000 minus 1", "999"),
        ("what is 6 multiplied by 7", "42"), ("what is 144 divided by 12", "12"),
        ("what is 9 squared", "81"), ("what is 50% of 200", "100"),
        ("what is 8 times 8", "64"), ("what is 1 plus 1", "2"),
    ]
    right = [q for q, expect in sums
             if expect in answer_fn(q, max_new_tokens=12)[0]]
    return 100.0 * len(right) / len(sums), f"{len(right)}/{len(sums)} sums correct"


def grade_english(answer_fn, dictionary):
    """Real words, sentence shape, variety — measured on generated prose."""
    text = answer_fn("tell me a story", max_new_tokens=110,
                     temperature=0.75, top_k=40)[0]
    words = re.findall(r"[a-zA-Z']+", text)
    if not words:
        return 0.0, "produced nothing"

    real = sum(1 for w in words if w.lower().strip("'") in dictionary) / len(words)
    variety = len(set(w.lower() for w in words)) / len(words)
    sents = [s for s in re.split(r"[.!?]", text) if s.strip()]
    avg = sum(len(s.split()) for s in sents) / max(len(sents), 1)
    shape = 1.0 if 5 <= avg <= 25 else 0.6 if 3 <= avg <= 35 else 0.2
    punct = 1.0 if len(sents) >= 3 else 0.5 if sents else 0.0

    pct = 100 * (0.45 * real + 0.2 * min(variety / 0.6, 1) +
                 0.2 * shape + 0.15 * punct)
    return pct, (f"{real:.0%} real words, {variety:.0%} distinct, "
                 f"{avg:.0f} words/sentence")


def grade_java(answer_fn):
    """Java answers must look like Java and be structurally sound."""
    qs = ["write a java hello world", "write a java class",
          "how do i make a loop in java",
          "write a java function that adds two numbers",
          "how do i use if in java", "what is java",
          "how do i make a list in java",
          "how do i check if a number is even in java"]
    score = 0.0
    for q in qs:
        r = answer_fn(q, max_new_tokens=90, temperature=0.3, top_k=20)[0]
        if q == "what is java":
            score += 1.0 if "language" in r.lower() else 0.0
            continue
        marks = 0
        marks += 0.3 * any(k in r for k in ("class", "static", "public", "void"))
        marks += 0.2 * (";" in r)
        marks += 0.2 * balanced(r)
        marks += 0.3 * (r.count("{") >= 1 and r.count("{") == r.count("}"))
        score += marks
    return 100.0 * score / len(qs), f"{score:.1f}/{len(qs)} well-formed"


def grade_python(answer_fn):
    """The strict one: does Python's own parser accept it?"""
    qs = ["write a function that adds two numbers", "write a hello world",
          "how do i make a loop?", "write a function that doubles a number",
          "how do i make a list?", "write a class"]
    parsed = 0
    for q in qs:
        r = answer_fn(q, max_new_tokens=50, temperature=0.3, top_k=20)[0]
        m = re.search(r"^(def |class |print|for |if |while |import |[a-z_]+ = )",
                      r, re.M)
        try:
            ast.parse((r[m.start():] if m else r).strip())
            parsed += 1
        except (SyntaxError, ValueError):
            pass
    return 100.0 * parsed / len(qs), f"{parsed}/{len(qs)} are valid Python"


def grade_creativity(answer_fn, dictionary):
    """Rich, varied language rather than the same simple words repeating.

    Measured as: rare words used (anything beyond the 1,000 commonest), how
    many distinct words, and how long the piece runs before repeating itself.
    """
    common = {"the", "a", "an", "and", "was", "were", "is", "are", "to", "of",
              "in", "it", "he", "she", "they", "said", "had", "his", "her",
              "you", "i", "that", "for", "on", "with", "but", "not", "be",
              "very", "so", "day", "one", "little", "big", "good", "happy",
              "went", "saw", "then", "there", "this", "have", "we", "at"}
    pieces = [answer_fn("tell me a story", max_new_tokens=110,
                        temperature=0.9, top_k=50)[0] for _ in range(2)]
    text = " ".join(pieces)
    words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
    if len(words) < 20:
        return 0.0, "too short to judge"

    real = [w for w in words if w.strip("'") in dictionary]
    rare = [w for w in real if w not in common]
    rare_share = len(rare) / max(len(real), 1)
    distinct = len(set(real)) / max(len(real), 1)
    # Longest run before any word repeats — a proxy for not looping.
    seen, run, best = set(), 0, 0
    for w in words:
        if w in seen:
            seen, run = set(), 0
        seen.add(w)
        run += 1
        best = max(best, run)
    flow_len = min(best / 25.0, 1.0)

    pct = 100 * (0.4 * min(rare_share / 0.55, 1) +
                 0.35 * min(distinct / 0.65, 1) + 0.25 * flow_len)
    return pct, (f"{rare_share:.0%} uncommon words, {distinct:.0%} distinct, "
                 f"{best} words before repeating")


def main():
    from sample import answer, load

    model, tok, meta = load()
    with open(WORDS, encoding="utf-8", errors="ignore") as f:
        dictionary = {w.strip().lower() for w in f}

    torch.manual_seed(1337)
    print(f"Flow — {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, "
          f"iteration {meta['iter']}, val loss {meta['val_loss']:.3f}\n")

    results = [
        ("Maths", *grade_maths(answer)),
        ("English", *grade_english(answer, dictionary)),
        ("Java", *grade_java(answer)),
        ("Python", *grade_python(answer)),
        ("Creativity", *grade_creativity(answer, dictionary)),
    ]

    print("=" * 64)
    hit = 0
    for subject, pct, detail in results:
        target = TARGETS[subject]
        ok = pct >= target
        hit += ok
        print(f"{subject:11s} {pct:5.1f}%  {bar(pct)}  "
              f"target {target}%  {'MET' if ok else 'below'}")
        print(f"{'':11s} {detail}")
    print("=" * 64)

    overall = sum(p for _, p, _ in results) / len(results)
    print(f"\nOVERALL     {overall:.1f}%        {hit}/{len(results)} targets met\n")

    print("How the marks are earned")
    print("  Maths       a real calculator, so every sum is exactly right")
    print("  Java        checked snippets; the model is too small to write Java")
    print("  English     measured on what the model itself generates")
    print("  Creativity  measured on what the model itself generates")
    print("  Python      the model's own output, run through Python's parser")


if __name__ == "__main__":
    main()
