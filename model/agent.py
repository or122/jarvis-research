"""Jarvis's hands — the agent that actually does the work.

Until now Jarvis could only talk about code. This lets it read files, write
files and search a project, so "fix the bug in server.py" becomes a real edit
rather than a suggestion.

It uses the real Claude Agent SDK. The first version drove the `claude` CLI as
a subprocess, because the SDK needs Python 3.10+ and the Mac ran 3.9 for
PyTorch. That turned out to be wrong: torch 2.2.2 has a Python 3.11 build for
this Intel Mac too, so the whole project moved to 3.11 and the SDK went in.

SAFETY - this is the only part of Jarvis that can change files, so it is shut
in on purpose:

  * it works ONLY inside ~/flow/workspace, never the whole Mac
  * Bash is DISABLED by default. It is the tool that can delete things, and it
    is not needed to write code. JARVIS_AGENT_BASH=1 turns it on deliberately.
  * permissions are never bypassed
  * no personal settings are loaded, so it behaves the same for everyone
  * every run has a turn limit and a timeout, so nothing can run forever

Run:  .venv311/bin/python model/agent.py "write hello.py that prints hello"
      .venv311/bin/python model/agent.py          self-test, does no work
"""
import asyncio
import os
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
MAX_TURNS = int(os.environ.get("JARVIS_AGENT_TURNS", "30"))

SYSTEM = (
    "You are Jarvis, Or Gefen's assistant. Or is 10, writes JavaScript, and "
    "built the small language model that normally answers him. Work only in "
    "the current folder. Keep code short and runnable. Explain what you did "
    "in one or two plain sentences at the end. Never claim something works "
    "unless you checked it."
)


def available():
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return False
    return True


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


def options():
    """Every safety rule, as one object the SDK understands."""
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        cwd=WORKSPACE,                  # the sandbox, and nowhere else
        allowed_tools=SAFE_TOOLS + (["Bash"] if BASH_ALLOWED else []),
        # Belt and braces: name Bash as disallowed as well as leaving it out of
        # the allowed list, so a default cannot quietly re-enable it.
        disallowed_tools=[] if BASH_ALLOWED else ["Bash"],
        permission_mode="default",      # never bypassPermissions
        max_turns=MAX_TURNS,
        # Load NO settings files. Without this the agent picks up whatever
        # Claude Code config the person running it happens to have - Or's own
        # CLAUDE.md leaked his session greeting and writing style straight into
        # Jarvis's answers, and cost 6x more because all his skills and memory
        # files were read in on every single build.
        setting_sources=[],
        system_prompt={"type": "preset", "preset": "claude_code",
                       "append": SYSTEM},
    )


async def _run(task, on_event=None):
    """Drive the agent, reporting each step as it happens."""
    from claude_agent_sdk import (AssistantMessage, ResultMessage, TextBlock,
                                  ToolUseBlock, query)

    said, cost = [], 0.0

    async for message in query(prompt=task, options=options()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                # A tool call is the interesting moment: this is where the
                # agent stops talking and actually changes a file. The old
                # subprocess version could not see these at all - it got one
                # lump of JSON at the very end.
                if isinstance(block, ToolUseBlock) and on_event:
                    # Only the tools that touch files are worth showing. The
                    # agent also makes internal housekeeping calls, and
                    # "tool: toolsearch" three times tells Or nothing.
                    said_it = _describe(block)
                    if said_it:
                        on_event("tool", said_it)
                elif isinstance(block, TextBlock):
                    said.append(block.text)

        elif isinstance(message, ResultMessage):
            cost = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
            # ResultMessage.result is the final answer. Prefer it, and fall
            # back to the text blocks if it is empty.
            final = (getattr(message, "result", "") or "").strip()
            if final:
                said = [final]

    text = "\n".join(s for s in said if s.strip()).strip()
    return text or "The agent finished but did not say what it did.", cost


VERBS = {"Write": "writing", "Edit": "editing", "Read": "reading",
         "Glob": "looking for files", "Grep": "searching",
         "Bash": "running a command"}


def _describe(block):
    """Turn a tool call into something a 10-year-old can read.

    Returns "" for anything that is not one of the tools we handed it, so
    internal housekeeping calls stay out of the progress line.
    """
    verb = VERBS.get(block.name)
    if not verb:
        return ""
    target = ""
    if isinstance(block.input, dict):
        path = block.input.get("file_path") or block.input.get("path") or ""
        target = os.path.basename(path) if path else ""
    return f"{verb} {target}".strip()


def do(task, on_event=None):
    """Run one task in the sandbox. Returns (text, files_changed, cost_usd).

    on_event(kind, detail) is called as things happen, so a caller can show
    progress instead of a long silence.
    """
    if not available():
        return "The agent needs the Claude Agent SDK, which is not installed.", [], 0.0

    ensure_workspace()
    before = _snapshot()

    if on_event:
        on_event("start", f"working in {os.path.basename(WORKSPACE)}/")

    async def guarded():
        # A timeout so a confused agent cannot run forever.
        return await asyncio.wait_for(_run(task, on_event), timeout=TIMEOUT)

    try:
        text, cost = asyncio.run(guarded())
    except asyncio.TimeoutError:
        return f"That took longer than {TIMEOUT} seconds, so I stopped.", [], 0.0
    except Exception as e:
        return f"The build failed: {type(e).__name__}: {e}", [], 0.0

    changed = _changes(before, _snapshot())
    if on_event:
        on_event("done", f"{len(changed)} file(s) changed, ${cost:.3f}")
    return text, changed, cost


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

    have_sdk = available()
    opts = options() if have_sdk else None

    checks = [
        ("Claude Agent SDK installed", have_sdk,
         "the real SDK, not a subprocess"),
        ("workspace exists", os.path.isdir(WORKSPACE), WORKSPACE),
        ("sandboxed to workspace", bool(opts) and opts.cwd == WORKSPACE,
         f"cwd={WORKSPACE}"),
        ("Bash disabled by default", not BASH_ALLOWED,
         "off" if not BASH_ALLOWED else "ON - deliberately enabled"),
        ("Bash also named as disallowed",
         BASH_ALLOWED or (bool(opts) and "Bash" in opts.disallowed_tools),
         "belt and braces"),
        ("permissions never bypassed",
         bool(opts) and opts.permission_mode == "default", "no skip flags"),
        ("no personal settings loaded",
         bool(opts) and opts.setting_sources == [],
         "same behaviour for everyone"),
        ("has a turn limit", bool(opts) and (opts.max_turns or 0) > 0,
         f"{MAX_TURNS} turns"),
        ("has a timeout", TIMEOUT > 0, f"{TIMEOUT}s"),
    ]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    good = sum(bool(c[1]) for c in checks)
    print(f"\n{good}/{len(checks)} guards in place")
    if opts:
        print(f"\ntools it may use: {', '.join(opts.allowed_tools)}")
    print(f"\n{'AGENT READY' if good == len(checks) else 'NOT SAFE YET'}")
    sys.exit(0 if good == len(checks) else 1)
