"""Score Flow out of 100.

Four categories, 25 points each, all measured — nothing is a matter of taste:

  Prediction  how well it predicts real text (from validation loss)
  English     real words, sentence shape, variety
  Chatting    does it answer, stay sane, and stop
  Code        keywords, brackets, indentation, and whether Python parses it

The scale is "how good can a small model built on a laptop be", NOT "how close
to Claude". On that second scale Flow scores roughly 1, and no amount of
training on this machine changes that. See the note the script prints.

Run:  .venv/bin/python model/score.py
"""
import ast
import math
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")
WORDS = "/usr/share/dict/words"

CHAT_QS = [
    "hello", "what is your name?", "how are you?", "who made you?",
    "tell me a story", "what can you do?", "are you real?", "thank you",
]
CODE_QS = [
    "write a function that adds two numbers", "write a hello world",
    "how do i make a loop?", "write a function that doubles a number",
    "how do i make a list?", "write a class",
]
KEYWORDS = ("def", "return", "print", "for", "if", "class", "import",
            "while", "range", "len", "=")
IDENTITY = ("flow", "model", "computer", "made", "talk", "language", "or")


def bar(score, width=28):
    filled = round(width * score / 100)
    return "█" * filled + "·" * (width - filled)


def balanced(text):
    stack, pairs = [], {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack


def main():
    if not os.path.exists(CKPT):
        print("no ckpt.pt — train first")
        sys.exit(1)

    from sample import chat, load

    model, tok, meta = load()
    vocab = meta["vocab_size"]
    val_loss = meta["val_loss"]
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Flow — {n_params/1e6:.1f}M parameters, vocab {vocab}, "
          f"trained to iteration {meta['iter']}\n")

    # ---------------------------------------------------------------- 1/4
    # Prediction. An untrained model scores ln(vocab); 2.0 is about as good as
    # a model this size gets on this data. Linear between the two.
    baseline = math.log(vocab)
    target = 2.0
    predict = max(0.0, min(25.0, 25 * (baseline - val_loss) / (baseline - target)))

    # ---------------------------------------------------------------- 2/4
    torch.manual_seed(1337)
    with open(WORDS, encoding="utf-8", errors="ignore") as f:
        dictionary = {w.strip().lower() for w in f}

    prose = chat("tell me a story", max_new_tokens=90, temperature=0.7, top_k=30)
    words = re.findall(r"[a-zA-Z']+", prose)
    real = [w for w in words if w.lower().strip("'") in dictionary]

    real_pct = len(real) / max(len(words), 1)
    variety = len(set(w.lower() for w in words)) / max(len(words), 1)
    sentences = [s for s in re.split(r"[.!?]", prose) if s.strip()]
    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    ends_ok = 1.0 if re.search(r"[.!?]", prose) else 0.0
    length_ok = 1.0 if 4 <= avg_len <= 28 else 0.5 if 2 <= avg_len <= 40 else 0.0

    english = (10 * real_pct) + (5 * min(variety / 0.6, 1.0)) + (5 * length_ok) + (5 * ends_ok)
    english = min(english, 25.0)

    # ---------------------------------------------------------------- 3/4
    replies = [chat(q, max_new_tokens=40, temperature=0.7, top_k=30) for q in CHAT_QS]
    answered = sum(1 for r in replies if len(r.split()) >= 2) / len(replies)
    clean = sum(1 for r in replies if "You:" not in r and "Me:" not in r) / len(replies)
    sized = sum(1 for r in replies if 2 <= len(r.split()) <= 40) / len(replies)
    # The identity questions are the two the model was explicitly taught.
    id_hits = sum(1 for r in replies[1:4]
                  if any(k in r.lower() for k in IDENTITY)) / 3

    chatting = (7 * answered) + (5 * clean) + (5 * sized) + (8 * id_hits)
    chatting = min(chatting, 25.0)

    # ---------------------------------------------------------------- 4/4
    code_replies = [chat(q, max_new_tokens=45, temperature=0.4, top_k=20)
                    for q in CODE_QS]
    has_kw = sum(1 for r in code_replies if any(k in r for k in KEYWORDS)) / len(code_replies)
    brackets = sum(1 for r in code_replies if balanced(r)) / len(code_replies)

    ind_ok = ind_total = 0
    for r in code_replies:
        lines = r.split("\n")
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith(":"):
                ind_total += 1
                ind_ok += lines[i + 1].startswith((" ", "\t"))
    indent = ind_ok / ind_total if ind_total else 0.0

    parses = 0
    for r in code_replies:
        m = re.search(r"^(def |class |print|for |if |while |import |[a-z_]+ = )", r, re.M)
        try:
            ast.parse((r[m.start():] if m else r).strip())
            parses += 1
        except (SyntaxError, ValueError):
            pass
    parse_rate = parses / len(code_replies)

    code = (7 * has_kw) + (6 * brackets) + (6 * indent) + (6 * parse_rate)
    code = min(code, 25.0)

    # ---------------------------------------------------------------- total
    total = predict + english + chatting + code

    rows = [
        ("Prediction", predict, f"val loss {val_loss:.2f} "
                                f"(random guess = {baseline:.2f}, great = {target})"),
        ("English", english, f"{real_pct:.0%} real words, {variety:.0%} distinct, "
                             f"{avg_len:.0f} words/sentence"),
        ("Chatting", chatting, f"{answered:.0%} answered, {clean:.0%} clean stops, "
                               f"{id_hits:.0%} knew itself"),
        ("Code", code, f"{has_kw:.0%} used keywords, {brackets:.0%} balanced, "
                       f"{parse_rate:.0%} valid Python"),
    ]

    print("=" * 62)
    for name, score, detail in rows:
        print(f"{name:11s} {score:5.1f} / 25   {bar(score * 4)}")
        print(f"{'':11s} {detail}")
    print("=" * 62)
    print(f"\n{'FLOW SCORES':13s} {total:.0f} / 100\n")

    if total >= 80:
        verdict = "Excellent for a model this size."
    elif total >= 60:
        verdict = "Good. It talks and mostly makes sense."
    elif total >= 40:
        verdict = "Working, but rough. More training will help."
    elif total >= 20:
        verdict = "Early. It has learned words, not yet meaning."
    else:
        verdict = "Barely trained."
    print(verdict)

    print("\nWhat this score means")
    print("  100 = as good as a small laptop-trained model can get")
    print("  It is NOT a comparison with Claude or ChatGPT. Against those,")
    print("  Flow scores about 1 out of 100 — they are ~10,000x bigger and")
    print("  cost millions to train. That gap cannot be closed on a laptop,")
    print("  and no amount of training here will change it.")

    print("\nSample answers")
    for q, r in list(zip(CHAT_QS, replies))[:3]:
        print(f"  You:  {q}")
        print(f"  Flow: {r}")
    print(f"  You:  {CODE_QS[0]}")
    print(f"  Flow: {code_replies[0]}")


if __name__ == "__main__":
    main()
