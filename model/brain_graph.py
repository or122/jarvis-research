"""Jarvis's memory, as a graph.

Every fact becomes a node. Two facts are linked when they share meaningful
words, so related knowledge clusters on its own rather than being filed by
hand — "what is python" and "what is a function" end up neighbours because
they talk about the same things.

The categories exist so the graph has colour and structure. A single grey
cloud of dots looks impressive and says nothing.

Run:  .venv/bin/python model/brain_graph.py     self-test
"""
import os
import re
import sys

from knowledge import STOP, connect, keywords

# Each fact goes to the first category it matches, so ORDER IS THE RULE.
# Specific categories come first; identity is last because words like "you"
# appear in most questions and would otherwise swallow everything - a first
# attempt put 39 of 62 facts under identity and the graph lost all structure.
# Common words are deliberately absent from these lists for the same reason.
CATEGORIES = [
    ("maths", "#80ed99", ("plus", "minus", "times", "divided", "squared",
                          "percent", "sum", "multiply", "subtract", "mod",
                          "days in", "hours in", "minutes in")),
    ("limits", "#ff9e00", ("cannot", "can't", "never taught", "nobody taught",
                           "beyond me", "too much for me", "i do not know",
                           "far from it", "homework")),
    ("code", "#4cc9f0", ("python", "javascript", "java", "function", "variable",
                         "loop", "bug", "class", "def ", "return", "print",
                         "list", "dictionary", "string", "reverse", "even",
                         "hello world", "scanner", "arraylist")),
    ("model", "#f72585", ("language model", "token", "training", "parameters",
                          "predicts the next", "neural", "weights", "vocab")),
    ("world", "#90e0ef", ("israel", "country", "school", "weather", "news",
                          "capital", "year", "day", "hour", "minute")),
    ("talk", "#ffd60a", ("hello", "hi", "hey", "morning", "night", "goodbye",
                         "thank", "bored", "stuck", "service", "welcome",
                         "any time", "of course")),
    ("identity", "#c77dff", ("jarvis", "my name", "who are you", "what are you",
                             "made me", "owner", "assistant", "claude", "smart",
                             "internet", "private", "mac", "trained")),
]
DEFAULT = ("notes", "#adb5bd")

# Below this many shared words, a link is coincidence rather than a
# relationship, and the graph turns into an unreadable hairball.
LINK_MIN = 2


def categorise(question, answer):
    text = f"{question} {answer}".lower()
    for name, colour, words in CATEGORIES:
        if any(re.search(rf"\b{re.escape(w)}", text) for w in words):
            return name, colour
    return DEFAULT


def build():
    db = connect()
    rows = db.execute("SELECT id, question, answer FROM facts ORDER BY id").fetchall()
    db.close()

    nodes = []
    words_by_id = {}
    for row in rows:
        fid, question, answer = row[0], row[1], row[2]
        name, colour = categorise(question, answer)
        # Both sides are used for linking: two facts can be related through
        # their answers even when the questions look nothing alike.
        words = keywords(question) | (keywords(answer) - STOP)
        words_by_id[fid] = {w for w in words if len(w) > 2}
        nodes.append({
            "id": fid,
            "label": question,
            "answer": answer,
            "group": name,
            "colour": colour,
            "size": 4 + min(len(answer) / 60.0, 5),
        })

    links = []
    ids = list(words_by_id)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = words_by_id[a] & words_by_id[b]
            if len(shared) >= LINK_MIN:
                links.append({"source": a, "target": b, "weight": len(shared)})

    # Degree drives node size, so the hubs of Jarvis's knowledge are visibly
    # the hubs rather than being the same dot as everything else.
    degree = {n["id"]: 0 for n in nodes}
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1
    for n in nodes:
        n["degree"] = degree[n["id"]]
        n["size"] = 4 + min(degree[n["id"]] * 0.9, 9)

    groups = {}
    for n in nodes:
        groups.setdefault(n["group"], {"group": n["group"],
                                       "colour": n["colour"], "count": 0})
        groups[n["group"]]["count"] += 1

    hubs = sorted(nodes, key=lambda n: -n["degree"])[:8]

    return {
        "nodes": nodes,
        "links": links,
        "groups": sorted(groups.values(), key=lambda g: -g["count"]),
        "hubs": [{"label": h["label"], "degree": h["degree"],
                  "colour": h["colour"]} for h in hubs],
    }


if __name__ == "__main__":
    g = build()
    print(f"nodes: {len(g['nodes'])}")
    print(f"links: {len(g['links'])}\n")

    print("groups:")
    for grp in g["groups"]:
        print(f"  {grp['group']:10s} {grp['count']:3d}  {grp['colour']}")

    print("\ntop hubs:")
    for h in g["hubs"]:
        print(f"  {h['degree']:3d}  {h['label'][:50]}")

    loners = sum(1 for n in g["nodes"] if n["degree"] == 0)
    checks = [
        ("has nodes", len(g["nodes"]) >= 40, f"{len(g['nodes'])}"),
        ("has links", len(g["links"]) >= 30, f"{len(g['links'])}"),
        # A graph where everything connects to everything is as useless as one
        # with no links at all.
        ("not a hairball", len(g["links"]) < len(g["nodes"]) ** 2 / 6,
         f"{len(g['links'])} < {int(len(g['nodes']) ** 2 / 6)}"),
        ("most nodes connected", loners < len(g["nodes"]) * 0.35,
         f"{loners} isolated of {len(g['nodes'])}"),
        ("several groups", len(g["groups"]) >= 4, f"{len(g['groups'])}"),
    ]
    print("\n=== BRAIN GRAPH GATE ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    good = all(c[1] for c in checks)
    print(f"\n{'GRAPH READY' if good else 'NOT READY'}")
    sys.exit(0 if good else 1)
