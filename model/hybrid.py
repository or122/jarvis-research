"""The hybrid engine — electric and petrol.

Or's idea: Jarvis is a hybrid car. Flow, the model he trained, is the electric
motor — free, silent, always there. Claude Fable is the petrol engine, which
kicks in only when the job needs power it does not have.

    ELECTRIC (Flow)        chat, greetings, anything short
    PETROL   (Fable 5.1)   code, real questions, anything Flow would fumble

Commands, sums and stored facts never reach either engine — those are plain
code and are answered before this file is consulted.

Nothing here runs unless the petrol engine is actually needed, so a day of
greetings and sums costs nothing at all.

Run:  .venv/bin/python model/hybrid.py "write me a bubble sort"
      .venv/bin/python model/hybrid.py            routing self-test
"""
import os
import re
import sys

# claude-fable-5-1 is Or's explicit choice: Anthropic's most capable model.
# It is also the most expensive at $10/$50 per million tokens, which is why
# the router below sends as little to it as possible.
PETROL_MODEL = os.environ.get("JARVIS_PETROL", "claude-fable-5-1")

# A cost cap, not a quality choice: 4k is plenty for a function or an
# explanation, and stops one runaway answer costing a fortune.
MAX_TOKENS = int(os.environ.get("JARVIS_MAX_TOKENS", "4096"))

SYSTEM = (
    "You are Jarvis, Or Gefen's assistant. Or is 10 years old, lives in "
    "Israel, writes JavaScript, and trained the small language model that "
    "normally answers him. Be direct and warm. Explain things clearly without "
    "talking down to him. When you write code, keep it short and runnable, and "
    "say what it does in one line. Never pretend something works if it does not."
)

# Words that mean the electric motor will not cope. Code is the obvious one;
# the rest are questions needing knowledge an 11M-parameter model does not have.
PETROL_WORDS = (
    "write me", "write a", "write the", "build", "make me a", "create a",
    "code", "function", "class", "script", "program", "app", "game", "website",
    "debug", "fix", "error", "bug", "why does", "why is", "why do",
    "how do i", "how does", "how can i", "explain", "what is the difference",
    "compare", "best way", "should i", "help me with", "teach me",
)

# Explicitly summoning the petrol engine.
FORCE_PETROL = re.compile(r"^\s*(ask\s+)?(claude|fable)[,:]?\s+", re.I)


def needs_petrol(text):
    """Decide which engine answers. Returns (petrol: bool, why: str)."""
    t = text.lower().strip()

    if FORCE_PETROL.match(t):
        return True, "you asked for Claude"

    if any(w in t for w in PETROL_WORDS):
        return True, "needs real knowledge or code"

    # Long questions carry more than Flow's 128-token memory can hold.
    if len(t.split()) > 12:
        return True, "too long for the small model"

    return False, "the small model can handle this"


def strip_summons(text):
    return FORCE_PETROL.sub("", text).strip()


def ask_petrol(text, stream=True):
    """Run Claude Fable. Yields text chunks.

    Fable 5.1 notes that differ from older models: thinking is always on so the
    `thinking` parameter is omitted entirely, temperature is not accepted, and
    server-side fallbacks are enabled so a safety decline is answered by
    another model inside the same call rather than returning nothing.
    """
    import anthropic

    client = anthropic.Anthropic()
    prompt = strip_summons(text)

    with client.beta.messages.stream(
        model=PETROL_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as s:
        for chunk in s.text_stream:
            yield chunk

        final = s.get_final_message()
        # A refusal arrives as HTTP 200 with no usable content, so it has to be
        # checked rather than assumed away.
        if final.stop_reason == "refusal":
            yield "\n(Claude declined to answer that one.)"


def petrol_available():
    """Is there a key to run the petrol engine at all?"""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile also works, and the SDK finds it itself.
    return os.path.exists(os.path.expanduser("~/.config/anthropic"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        petrol, why = needs_petrol(question)
        print(f"engine: {'PETROL (Fable)' if petrol else 'ELECTRIC (Flow)'} — {why}\n")
        if petrol:
            if not petrol_available():
                print("No API key. Set ANTHROPIC_API_KEY to use the petrol engine.")
                sys.exit(1)
            for piece in ask_petrol(question):
                print(piece, end="", flush=True)
            print()
        else:
            from sample import chat
            print(chat(question, max_new_tokens=70))
        sys.exit(0)

    # --- routing gate --------------------------------------------------
    # The router is the whole design, so it is tested without spending money.
    electric = [
        "hello", "hi", "thanks", "how are you", "good morning",
        "i am bored", "goodbye", "who are you",
    ]
    petrol = [
        "write me a function that sorts a list",
        "why does my code crash",
        "how do i use async in javascript",
        "explain what a closure is",
        "build me a game",
        "claude, what is the capital of France",
        "fix this bug in my react component please it keeps rerendering forever",
    ]

    print("=== ROUTING GATE ===\n")
    right = 0
    for q in electric:
        got, why = needs_petrol(q)
        ok = not got
        right += ok
        print(f"  {'OK  ' if ok else 'FAIL'} electric: {q:38s} -> {why}")
    print()
    for q in petrol:
        got, why = needs_petrol(q)
        right += got
        print(f"  {'OK  ' if got else 'FAIL'} petrol:   {q[:38]:38s} -> {why}")

    total = len(electric) + len(petrol)
    print(f"\n{right}/{total} routed correctly")
    print(f"petrol engine available: {'yes' if petrol_available() else 'NO API KEY'}")
    print(f"{'HYBRID READY' if right == total else 'ROUTING NOT READY'}")
    sys.exit(0 if right == total else 1)
