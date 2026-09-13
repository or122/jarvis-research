"""Jarvis — Or's assistant, built on Flow.

Most of what an assistant does is not thinking. Opening an app, telling the
time, setting a timer, doing a sum: all of that is ordinary code talking to the
operating system, and code is right every time. Only the leftover conversation
needs the model.

So requests are tried in this order:

    1. a command    real code runs it            always correct
    2. a sum        the calculator                always correct
    3. a known fact the database                  always correct
    4. anything else the model                    sounds right, may be wrong

The model is deliberately last, and it is swappable: BRAIN below decides
whether that fallback is Flow or something else, so switching later is one
line rather than a rewrite.

Run:  .venv/bin/python model/jarvis.py "what time is it"
      .venv/bin/python model/jarvis.py           self-test
"""
import datetime
import os
import re
import shutil
import subprocess
import sys

NAME = "Jarvis"          # change here and everywhere follows
OWNER = "Or"

# Which brain answers when no command matches. "flow" is the model Or trained;
# "none" makes Jarvis say it does not know rather than guess.
BRAIN = os.environ.get("JARVIS_BRAIN", "flow")

# Speaking is off by default so tests and the web app stay silent.
SPEAK = os.environ.get("JARVIS_SPEAK", "0") == "1"
VOICE = os.environ.get("JARVIS_VOICE", "Daniel")     # British, suits Jarvis


def say_aloud(text):
    """Speak through macOS. Never blocks and never crashes the answer."""
    if not SPEAK or not shutil.which("say"):
        return
    try:
        subprocess.Popen(["say", "-v", VOICE, text[:400]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _osascript(script):
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True,
                           text=True, timeout=15)
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)


# --------------------------------------------------------------- commands
# Each is (regex, function). The regex decides; the function does the work and
# returns what Jarvis should say.

def cmd_time(_m):
    return datetime.datetime.now().strftime("It is %-I:%M %p.")


def cmd_date(_m):
    return datetime.datetime.now().strftime("Today is %A, %-d %B %Y.")


def cmd_open(m):
    app = m.group("app").strip().title()
    ok, err = _osascript(f'tell application "{app}" to activate')
    return f"Opening {app}." if ok else f"I could not open {app}."


def cmd_quit(m):
    app = m.group("app").strip().title()
    ok, _ = _osascript(f'tell application "{app}" to quit')
    return f"Closing {app}." if ok else f"I could not close {app}."


def cmd_volume(m):
    level = max(0, min(100, int(m.group("n"))))
    ok, _ = _osascript(f"set volume output volume {level}")
    return f"Volume {level} percent." if ok else "I could not change the volume."


def cmd_mute(_m):
    _osascript("set volume with output muted")
    return "Muted."


def cmd_unmute(_m):
    _osascript("set volume without output muted")
    return "Unmuted."


def cmd_music(_m):
    ok, _ = _osascript('tell application "Music" to play')
    return "Playing music." if ok else "I could not start the music."


def cmd_pause(_m):
    ok, _ = _osascript('tell application "Music" to pause')
    return "Paused." if ok else "Nothing was playing."


def cmd_battery(_m):
    try:
        r = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                           text=True, timeout=10)
        pct = re.search(r"(\d+)%", r.stdout)
        state = "charging" if "AC Power" in r.stdout else "on battery"
        return f"Battery is at {pct.group(1)} percent, {state}." if pct \
            else "I could not read the battery."
    except (subprocess.TimeoutExpired, OSError):
        return "I could not read the battery."


def cmd_timer(m):
    amount = int(m.group("n"))
    unit = m.group("unit").lower()
    seconds = amount * (60 if unit.startswith("min") else
                        3600 if unit.startswith("hour") else 1)
    # Detached so the timer outlives this request.
    subprocess.Popen(
        ["bash", "-c",
         f'sleep {seconds}; osascript -e \'display notification '
         f'"Your {amount} {unit} timer is done" with title "{NAME}"\''],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"Timer set for {amount} {unit}."


def cmd_screenshot(_m):
    target = os.path.expanduser(
        f"~/Desktop/{NAME}-{datetime.datetime.now():%H%M%S}.png")
    try:
        subprocess.run(["screencapture", "-x", target], timeout=20)
        return "Screenshot saved to your Desktop."
    except (subprocess.TimeoutExpired, OSError):
        return "I could not take a screenshot."


def cmd_who(_m):
    return (f"I am {NAME}, {OWNER}'s assistant. I run on a language model "
            f"{OWNER} built and trained himself, on this Mac.")


def cmd_can(_m):
    return ("I can tell the time, open and close apps, play music, set timers, "
            "change the volume, check the battery, take screenshots, do maths, "
            "and answer things I have been taught. Anything else I try my best.")


COMMANDS = [
    (r"^(what('s| is)? the )?time( is it)?\??$", cmd_time),
    (r"^what time is it\??$", cmd_time),
    (r"^(what('s| is)? (the |today'?s? )?date|what day is it)\??$", cmd_date),
    (r"^(open|launch|start) (?P<app>[\w\s]+?)\s*$", cmd_open),
    (r"^(close|quit) (?P<app>[\w\s]+?)\s*$", cmd_quit),
    (r"^(set )?volume (to )?(?P<n>\d+)%?$", cmd_volume),
    (r"^mute$", cmd_mute),
    (r"^unmute$", cmd_unmute),
    (r"^(play|start) (some )?music$", cmd_music),
    (r"^(pause|stop)( the)?( music)?$", cmd_pause),
    (r"^(what('s| is)? the )?battery( level)?\??$", cmd_battery),
    (r"^(set a )?timer (for )?(?P<n>\d+) (?P<unit>seconds?|minutes?|mins?|hours?)$",
     cmd_timer),
    (r"^(take a )?screenshot$", cmd_screenshot),
    (r"^who are you\??$", cmd_who),
    (r"^what can you do\??$", cmd_can),
]

COMPILED = [(re.compile(p, re.I), f) for p, f in COMMANDS]

# "Jarvis, open safari" — the wake word is stripped before matching.
WAKE = re.compile(rf"^\s*(hey\s+)?{NAME}[,:]?\s*", re.I)


def run_command(text):
    """Return what a command would say, or None if nothing matched."""
    cleaned = WAKE.sub("", text).strip().rstrip("!.")
    for pattern, fn in COMPILED:
        m = pattern.match(cleaned)
        if m:
            try:
                return fn(m)
            except Exception as e:      # a broken command must not kill the reply
                return f"That went wrong: {type(e).__name__}"
    return None


def ask(text, speak=None):
    """The whole assistant. Returns (reply, source)."""
    reply = run_command(text)
    source = "command"

    if reply is None:
        from knowledge import look_up
        cleaned = WAKE.sub("", text).strip()
        fact, _ = look_up(cleaned)
        if fact:
            reply, source = fact, "memory"
        elif BRAIN == "flow":
            from sample import chat
            reply = chat(cleaned, max_new_tokens=70, temperature=0.7, top_k=30)
            source = "model"
        else:
            reply, source = "I do not know that one yet.", "none"

    if speak if speak is not None else SPEAK:
        say_aloud(reply)
    return reply, source


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer, src = ask(" ".join(sys.argv[1:]))
        print(f"{NAME}: {answer}   [{src}]")
        sys.exit(0)

    # --- gate: commands must run as code, not reach the model ------------
    print(f"=== {NAME} COMMAND GATE ===\n")
    cases = [
        ("what time is it", "command"),
        ("Jarvis, what time is it", "command"),
        ("what is the date", "command"),
        ("battery", "command"),
        ("volume 40", "command"),
        ("set a timer for 5 minutes", "command"),
        ("who are you", "command"),
        ("what can you do", "command"),
        ("what is 47 times 13", "memory"),
        ("what is your name", "memory"),
    ]
    right = 0
    for text, expect in cases:
        answer, src = ask(text, speak=False)
        ok = src == expect
        right += ok
        print(f"  {'OK  ' if ok else 'FAIL'} [{src:7s}] {text:32s} -> {answer[:52]}")

    print(f"\n{right}/{len(cases)} routed correctly")
    print(f"{'JARVIS READY' if right == len(cases) else 'NOT READY'}")
    sys.exit(0 if right == len(cases) else 1)
