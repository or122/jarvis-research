"""Flow's calculator — real arithmetic, not a guess.

An 11M-parameter model cannot do maths. It has never "counted"; it only
predicts likely next words, so "2 + 2" comes out as whatever number usually
follows in text. A calculator is always right, so Flow uses one.

Questions are turned into an expression and evaluated with Python's own parser
over a whitelist of node types. `eval()` is never used: this file runs
whatever the user typed, and eval() on that would let anyone run any code.

Run:  .venv/bin/python model/maths.py            self-test
      .venv/bin/python model/maths.py "12 x 8"   try one
"""
import ast
import operator
import re
import sys

# Only these operations exist. Anything else in the parse tree is rejected,
# which is what makes running user input safe.
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

WORDS = {
    "plus": "+", "add": "+", "and": "+", "sum of": "+",
    "minus": "-", "subtract": "-", "take away": "-", "less": "-",
    "times": "*", "multiplied by": "*", "multiply": "*", "x": "*",
    "divided by": "/", "divide": "/", "over": "/",
    "squared": "**2", "cubed": "**3",
    "to the power of": "**", "power": "**",
    "remainder": "%", "mod": "%",
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100",
    "thousand": "1000", "half": "0.5",
}

# Text that means "please calculate", stripped before parsing.
NOISE = re.compile(
    r"\b(what|whats|what's|is|are|the|of|calculate|work out|tell me|"
    r"how much|how many|equals?|answer|result|please|can you|do)\b",
    re.I,
)


def _evaluate(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("not a number")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


def to_expression(question):
    """Turn an English maths question into something Python can parse."""
    text = question.lower().strip().rstrip("?.!")

    # Percentages: "20% of 50" -> "(20/100)*50"
    pct = re.match(r"^\s*([\d.]+)\s*%\s*(?:of\s+)?([\d.]+)\s*$",
                   NOISE.sub(" ", text).strip())
    if pct:
        return f"({pct.group(1)}/100)*{pct.group(2)}"

    # Square roots, before the general word replacements.
    root = re.search(r"(?:square root|sqrt)\s*(?:of\s*)?([\d.]+)", text)
    if root:
        return f"{root.group(1)}**0.5"

    # Operator phrases are replaced BEFORE the filler words are stripped:
    # NOISE removes "of", which would otherwise destroy "to the power of"
    # before it could be recognised.
    # Longest phrases first, so "multiplied by" is not eaten by "multiply".
    for phrase in sorted(WORDS, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(phrase)}\b", f" {WORDS[phrase]} ", text)
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    text = NOISE.sub(" ", text)

    text = text.replace("÷", "/").replace("×", "*").replace("^", "**")
    expr = re.sub(r"\s+", "", text)

    # Must be only numbers and operators, and must contain both.
    if not expr or not re.fullmatch(r"[\d.+\-*/%()**]+", expr):
        return None
    if not re.search(r"\d", expr) or not re.search(r"[+\-*/%]", expr):
        return None
    return expr


def solve(question):
    """Return the answer as a string, or None if this is not a maths question."""
    expr = to_expression(question)
    if not expr:
        return None
    try:
        value = _evaluate(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError,
            OverflowError, RecursionError):
        return None

    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        # 4.0 should read as 4, but 4.5 must keep its half.
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(solve(sys.argv[1]) or "(not a maths question)")
        sys.exit(0)

    should_solve = [
        ("what is 2 plus 2", "4"),
        ("2 + 2", "4"),
        ("what is 10 times 10", "100"),
        ("12 x 8", "96"),
        ("what is 100 divided by 4", "25"),
        ("what is 7 minus 9", "-2"),
        ("what is 5 squared", "25"),
        ("what is 2 to the power of 10", "1024"),
        ("what is 20% of 50", "10"),
        ("what is the square root of 144", "12"),
        ("what is 7 divided by 2", "3.5"),
        ("what is 17 mod 5", "2"),
        ("what is three plus four", "7"),
        ("what is 1000 minus 1", "999"),
        ("what is 6 multiplied by 7", "42"),
        ("calculate 15 + 27", "42"),
    ]
    should_refuse = [
        "hello", "what is your name", "tell me a story",
        "who made you", "what is love", "write a function",
    ]

    print("maths it must get right:")
    right = 0
    for q, expect in should_solve:
        got = solve(q)
        ok = got == expect
        right += ok
        print(f"  {'OK  ' if ok else 'WRONG'} {q:34s} -> {got}"
              f"{'' if ok else f'   (expected {expect})'}")

    print("\nnot maths — must be left to the model:")
    refused = 0
    for q in should_refuse:
        got = solve(q)
        ok = got is None
        refused += ok
        print(f"  {'OK  ' if ok else 'BAD '} {q:34s} -> "
              f"{'passed on' if ok else got}")

    acc = right / len(should_solve)
    ref = refused / len(should_refuse)
    print("\n=== MATHS GATE ===")
    print(f"{'PASS' if acc == 1.0 else 'FAIL'}  correct: "
          f"{right}/{len(should_solve)}  ({acc:.0%})")
    print(f"{'PASS' if ref == 1.0 else 'FAIL'}  refuses non-maths: "
          f"{refused}/{len(should_refuse)}")

    ok = acc == 1.0 and ref == 1.0
    print(f"\n{'MATHS READY' if ok else 'NOT READY'}")
    sys.exit(0 if ok else 1)
