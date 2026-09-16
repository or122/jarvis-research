"""Does the agent read the CLI's output correctly?

This exists because of a real bug. `claude --output-format json` returns a
JSON ARRAY of events - init, every tool call, then a final {"type": "result"}
carrying the answer. The first parser assumed one object, fell through to
"print the raw text", and dumped the whole event stream at the user. The build
had worked; only the reporting was broken.

Every case here uses fake events, so this costs nothing and needs no key. A
test you can run for free is a test you will actually run.

Run:  .venv/bin/python model/test_agent_parse.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent


class Fake:
    """Stands in for what subprocess.run gives back."""

    def __init__(self, out, err=""):
        self.stdout, self.stderr = out, err


# The real shape: an array of events with the result last.
GOOD = json.dumps([
    {"type": "system", "subtype": "init", "tools": ["Read", "Write"]},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "noise"}]}},
    {"type": "result", "subtype": "success", "is_error": False,
     "result": "I made guess.py.", "total_cost_usd": 0.19145375},
])
FAILED = json.dumps([
    {"type": "result", "subtype": "error_max_turns", "is_error": True,
     "result": "", "total_cost_usd": 0.02},
])
SOLO = json.dumps({"type": "result", "is_error": False,
                   "result": "single object still works", "total_cost_usd": 0.01})

CASES = [
    ("array -> clean answer", Fake(GOOD), "I made guess.py."),
    ("array -> cost read",    Fake(GOOD), 0.19145375),
    ("error event named",     Fake(FAILED), "The agent failed: error_max_turns"),
    ("lone dict still works", Fake(SOLO), "single object still works"),
    ("not JSON -> raw text",  Fake("crashed: no such file"), "crashed: no such file"),
    ("empty -> stderr",       Fake("", "boom"), "boom"),
    ("no result event",       Fake(json.dumps([{"type": "system"}])),
                              "The agent finished but did not say what it did."),
]


if __name__ == "__main__":
    print("=== PARSER GATE ===\n")
    passed = 0
    for name, fake, want in CASES:
        text, cost = agent._parse(fake)
        # The cost case checks the number; every other case checks the text.
        got = cost if isinstance(want, float) else text
        ok = got == want
        passed += ok
        print("{}  {}: {!r}".format("PASS" if ok else "FAIL", name, got))

    print("\n{}/{} parser checks passed".format(passed, len(CASES)))
    sys.exit(0 if passed == len(CASES) else 1)
