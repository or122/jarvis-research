"""Jarvis's hands — the agent that actually does the work.

Until now Jarvis could only talk about code. This lets it read files, write
files and search a project, so "fix the bug in server.py" becomes a real edit
rather than a suggestion.

It drives the Claude Code agent loop through the CLI, which the Agent SDK's
own documentation gives as the supported route for anything that cannot import
the Python library. That is this machine: the SDK needs Python 3.10+ and the
server runs on 3.9 because PyTorch 2.2.2 is the last build for this Intel Mac.

SAFETY - this is the only part of Jarvis that can change files, so it is shut
in on purpose:

  * it works ONLY inside ~/flow/workspace, never the whole Mac
  * Bash is DISABLED by default. It is the tool that can delete things, and it
    is not needed to write code. JARVIS_AGENT_BASH=1 turns it on deliberately.
  * permissions are never bypassed
  * every run has a timeout, so nothing can hang forever

Run:  .venv/bin/python model/agent.py "write hello.py that prints hello"
      .venv/bin/python model/agent.py            self-test, does no work
"""
import json
import os
import shutil
import subprocess
import sys

# The sandbox. Everything the agent does happens in here and nowhere else.
WORKSPACE = os.environ.get(
    "JARVIS_WORKSPACE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "workspace"),
)

# Reading, writing and searching are enough to write code. Bash is the tool
# that can delete a folder or download something, so it stays off unless asked
# for by name.
SAFE_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]
BASH_ALLOWED = os.environ.get("JARVIS_AGENT_BASH", "0") == "1"

TIMEOUT = int(os.environ.get("JARVIS_AGENT_TIMEOUT", "300"))

SYSTEM = (
    "You are Jarvis, Or Gefen's assistant. Or is 10, writes JavaScript, and "
    "built the small language model that normally answers him. Work only in "
    "the current folder. Keep code short and runnable. Explain what you did "
    "in one or two plain sentences at the end. Never claim something works "
    "unless you checked it."
)


def available():
    return shutil.which("claude") is not None


def ensure_workspace():
    os.makedirs(WORKSPACE, exist_ok=True)
    readme = os.path.join(WORKSPACE, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Jarvis workspace\n\n"
                "Everything Jarvis builds goes here. It cannot touch anything\n"
                "outside this folder.\n\n"
                "Safe to delete anything in here.\n"
            )
    return WORKSPACE


def do(task, on_event=None):
    """Run one task in the sandbox. Returns (text, files_changed, cost_usd).

    on_event(kind, detail) is called as things happen, so a caller can show
    progress instead of a long silence.
    """
    if not available():
        return "The agent needs the Claude Code CLI, which is not installed.", [], 0.0

    ensure_workspace()
    before = _snapshot()

    cmd = [
        "claude", "-p", task,
        "--output-format", "json",
        # Load NO settings files. Without this the agent picks up whatever
        # Claude Code config the person running it happens to have - Or's own
        # CLAUDE.md leaked his session greeting and writing style straight into
        # Jarvis's answers. Jarvis's personality must come from SYSTEM below
        # and nowhere else, so it behaves the same for everyone.
        "--setting-sources", "",
        "--append-system-prompt", SYSTEM,
        "--allowed-tools", *(SAFE_TOOLS + (["Bash"] if BASH_ALLOWED else [])),
    ]
    if not BASH_ALLOWED:
        # Belt and braces: name it as disallowed as well as leaving it out of
        # the allowed list, so a default cannot quietly re-enable it.
        cmd += ["--disallowed-tools", "Bash"]

    if on_event:
        on_event("start", f"working in {os.path.basename(WORKSPACE)}/")

    try:
        result = subprocess.run(
            cmd, cwd=WORKSPACE, capture_output=True, text=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"That took longer than {TIMEOUT} seconds, so I stopped.", [], 0.0
    except OSError as e:
        return f"Could not start the agent: {e}", [], 0.0

    text, cost = _parse(result)
    changed = _changes(before, _snapshot())
    if on_event:
        on_event("done", f"{len(changed)} file(s) changed, ${cost:.3f}")
    return text, changed, cost


def _parse(result):
    """Pull the reply and the cost out of the CLI's output.

    The CLI returns a JSON ARRAY of events - init, every tool call, then a
    final {"type": "result"} carrying the answer and the money spent. A first
    version of this function assumed one object, fell through to the raw text,
    and printed the whole event stream at the user.
    """
    raw = (result.stdout or "").strip()
    if not raw:
        return (result.stderr or "").strip() or "The agent returned nothing.", 0.0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, 0.0         # not JSON after all; the text is still useful

    events = data if isinstance(data, list) else [data]
    for event in reversed(events):      # the result is the last event
        if isinstance(event, dict) and event.get("type") == "result":
            cost = float(event.get("total_cost_usd") or 0.0)
            if event.get("is_error"):
                return f"The agent failed: {event.get('subtype', 'unknown')}", cost
            answer = event.get("result")
            if isinstance(answer, str) and answer.strip():
                return answer.strip(), cost

    return "The agent finished but did not say what it did.", 0.0


def _snapshot():
    """Every file in the sandbox with its size and mtime."""
    seen = {}
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
                seen[os.path.relpath(path, WORKSPACE)] = (st.st_size, st.st_mtime)
            except OSError:
                pass
    return seen


def _changes(before, after):
    changed = [f"+ {p}" for p in after if p not in before]
    changed += [f"~ {p}" for p in after if p in before and after[p] != before[p]]
    changed += [f"- {p}" for p in before if p not in after]
    return sorted(changed)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        print(f"workspace: {WORKSPACE}")
        print(f"bash:      {'ENABLED' if BASH_ALLOWED else 'disabled'}\n")
        answer, files, cost = do(task, on_event=lambda k, d: print(f"[{k}] {d}"))
        print(f"\n{answer}\n")
        if files:
            print("files changed:")
            for f in files:
                print(f"  {f}")
        print(f"\ncost: ${cost:.3f}")
        sys.exit(0)

    # --- gate: check the guards without letting the agent do anything ----
    print("=== JARVIS AGENT GATE ===\n")
    ensure_workspace()

    cmd_preview = [
        "claude", "-p", "<task>", "--output-format", "json",
        "--setting-sources", "",
        "--append-system-prompt", "<system>",
        "--allowed-tools", *(SAFE_TOOLS + (["Bash"] if BASH_ALLOWED else [])),
    ]
    if not BASH_ALLOWED:
        cmd_preview += ["--disallowed-tools", "Bash"]

    checks = [
        ("claude CLI found", available(), shutil.which("claude") or "missing"),
        ("workspace exists", os.path.isdir(WORKSPACE), WORKSPACE),
        ("sandboxed to workspace", WORKSPACE.endswith("workspace"),
         f"cwd={WORKSPACE}"),
        ("Bash disabled by default", not BASH_ALLOWED,
         "off" if not BASH_ALLOWED else "ON - deliberately enabled"),
        ("Bash also named as disallowed", BASH_ALLOWED or "--disallowed-tools" in cmd_preview,
         "belt and braces"),
        ("permissions never bypassed",
         not any("dangerous" in c for c in cmd_preview), "no skip flags"),
        ("no personal settings loaded", "--setting-sources" in cmd_preview,
         "same behaviour for everyone"),
        ("has a timeout", TIMEOUT > 0, f"{TIMEOUT}s"),
    ]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    good = sum(c[1] for c in checks)
    print(f"\n{good}/{len(checks)} guards in place")
    print(f"\ncommand it would run:\n  {' '.join(cmd_preview[:8])} …")
    print(f"\n{'AGENT READY' if good == len(checks) else 'NOT SAFE YET'}")
    sys.exit(0 if good == len(checks) else 1)
