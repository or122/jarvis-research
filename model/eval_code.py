"""Can Flow write code? Four gates, honestly scored.

A 11M-parameter model will not write correct programs. What it can plausibly
learn is the *shape* of code: keywords in the right places, balanced brackets,
consistent indentation, and text that Python's own parser accepts.

Gate 4 is the strict one — it runs the real Python parser. Nothing subjective
about that: the code either parses or it does not.

Run:  .venv/bin/python model/eval_code.py
"""
import ast
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")

PROMPTS = [
    "write a function that adds two numbers",
    "write a hello world",
    "how do i make a loop?",
    "write a function that doubles a number",
    "how do i make a list?",
    "write a class",
    "how do i check if a number is even?",
    "write a function that reverses a string",
]

KEYWORDS = ("def", "return", "print", "for", "if", "class", "import",
            "while", "in", "range", "len")


def main():
    if not os.path.exists(CKPT):
        print("no ckpt.pt — train first")
        sys.exit(1)

    from sample import chat

    torch.manual_seed(1337)
    replies = []
    print("=== what Flow writes ===\n")
    for prompt in PROMPTS:
        # Low temperature: code has far less acceptable variation than prose,
        # so sampling should stay close to the most likely token.
        reply = chat(prompt, max_new_tokens=60, temperature=0.4, top_k=20)
        replies.append(reply)
        print(f"You:  {prompt}")
        print(f"Flow: {reply}\n")

    # --- Gate 1: does it produce code at all, rather than prose? -----------
    with_kw = sum(1 for r in replies if any(k in r for k in KEYWORDS))
    kw_pct = 100.0 * with_kw / len(replies)

    # --- Gate 2: brackets balanced ----------------------------------------
    def balanced(text):
        stack = []
        pairs = {")": "(", "]": "[", "}": "{"}
        for ch in text:
            if ch in "([{":
                stack.append(ch)
            elif ch in pairs:
                if not stack or stack.pop() != pairs[ch]:
                    return False
        return not stack

    bal = sum(1 for r in replies if balanced(r))
    bal_pct = 100.0 * bal / len(replies)

    # --- Gate 3: indentation after a colon ---------------------------------
    # A line ending in ":" must be followed by an indented line. This is the
    # single most characteristic rule of Python's syntax.
    indent_ok = indent_total = 0
    for r in replies:
        lines = r.split("\n")
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith(":"):
                indent_total += 1
                if lines[i + 1].startswith((" ", "\t")):
                    indent_ok += 1
    indent_pct = 100.0 * indent_ok / max(indent_total, 1)

    # --- Gate 4: Python's own parser accepts it ----------------------------
    parses = 0
    for r in replies:
        # Take just the code-looking part: everything from the first line that
        # starts with a keyword, so a prose preamble doesn't fail the parse.
        m = re.search(r"^(def |class |print|for |if |while |import |[a-z_]+ = )",
                      r, re.M)
        snippet = r[m.start():] if m else r
        try:
            ast.parse(snippet.strip())
            parses += 1
        except (SyntaxError, ValueError):
            pass
    parse_pct = 100.0 * parses / len(replies)

    checks = [
        ("uses code keywords >= 75%", kw_pct >= 75, f"{kw_pct:.0f}% ({with_kw}/{len(replies)})"),
        ("brackets balanced >= 75%", bal_pct >= 75, f"{bal_pct:.0f}% ({bal}/{len(replies)})"),
        ("indents after ':' >= 70%", indent_pct >= 70,
         f"{indent_pct:.0f}% ({indent_ok}/{indent_total})"),
        ("Python parses it >= 50%", parse_pct >= 50, f"{parse_pct:.0f}% ({parses}/{len(replies)})"),
    ]

    print("=== CODE GATE ===")
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    n = sum(c[1] for c in checks)
    print(f"\n{n}/4 gates passed")
    print("Note: these measure code SHAPE, not correctness. A model this size")
    print("cannot write reliably correct programs, and no gate here claims it can.")
    sys.exit(0 if n >= 3 else 1)


if __name__ == "__main__":
    main()
