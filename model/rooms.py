"""Shared rooms — several people talking with Flow at once.

The server owns the conversation, not the browsers. Every message becomes a
row with a sequence number the server assigns, and clients ask "what happened
after number N?". That is what makes two people typing at the same instant
safe: whoever's row is written first is simply first, and everyone sees the
same order.

The obvious alternative - each browser keeping its own list and sending the
whole thing - falls apart the moment two people type, because the two lists
disagree about what was said.

Seats are the product: a room has a limit, and inviting past it is blocked.
No real money in v1 - `paid` is a flag set by hand.

Run:  .venv/bin/python model/rooms.py     self-test
"""
import os
import random
import sqlite3
import string
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "rooms.db")

FREE_SEATS = 2      # you plus one. The third person is the upsell.
PAID_SEATS = 10


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS rooms (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL,
            code    TEXT NOT NULL UNIQUE,
            paid    INTEGER NOT NULL DEFAULT 0,
            created REAL NOT NULL);

        CREATE TABLE IF NOT EXISTS members (
            room_id INTEGER NOT NULL,
            name    TEXT NOT NULL,
            joined  REAL NOT NULL,
            PRIMARY KEY (room_id, name));

        -- seq is a single global counter, so ordering is decided in one place
        -- rather than negotiated between clients.
        CREATE TABLE IF NOT EXISTS messages (
            seq     INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER NOT NULL,
            author  TEXT NOT NULL,
            text    TEXT NOT NULL,
            created REAL NOT NULL);

        CREATE INDEX IF NOT EXISTS msg_room ON messages(room_id, seq);
    """)
    db.commit()
    return db


def _code():
    # Ambiguous characters left out: a code is read aloud across a room.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(5))


def create_room(name, owner):
    db = connect()
    for _ in range(20):
        code = _code()
        try:
            cur = db.execute(
                "INSERT INTO rooms (name, code, created) VALUES (?, ?, ?)",
                (name.strip()[:60] or "Room", code, time.time()))
            room_id = cur.lastrowid
            db.execute("INSERT INTO members (room_id, name, joined) VALUES (?, ?, ?)",
                       (room_id, owner.strip()[:40] or "You", time.time()))
            db.commit()
            db.close()
            return {"id": room_id, "code": code, "name": name}
        except sqlite3.IntegrityError:
            continue        # code collision, try another
    db.close()
    raise RuntimeError("could not allocate a room code")


def seats(room):
    return PAID_SEATS if room["paid"] else FREE_SEATS


def get_room(code_or_id):
    db = connect()
    row = db.execute("SELECT * FROM rooms WHERE code = ? OR id = ?",
                     (str(code_or_id).upper(), code_or_id)).fetchone()
    db.close()
    return row


def members(room_id):
    db = connect()
    rows = db.execute("SELECT name FROM members WHERE room_id = ? ORDER BY joined",
                      (room_id,)).fetchall()
    db.close()
    return [r["name"] for r in rows]


def join(code, name):
    """Returns (ok, detail). Refuses when the room is full - that refusal is
    the entire business model, so it has to actually work."""
    room = get_room(code)
    if not room:
        return False, {"error": "no room with that code"}

    name = name.strip()[:40] or "Guest"
    current = members(room["id"])
    if name in current:
        return True, {"room_id": room["id"], "name": room["name"],
                      "members": current, "seats": seats(room)}

    if len(current) >= seats(room):
        return False, {"error": "full", "seats": seats(room),
                       "members": current, "paid": bool(room["paid"])}

    db = connect()
    db.execute("INSERT INTO members (room_id, name, joined) VALUES (?, ?, ?)",
               (room["id"], name, time.time()))
    db.commit()
    db.close()
    return True, {"room_id": room["id"], "name": room["name"],
                  "members": members(room["id"]), "seats": seats(room)}


def say(room_id, author, text):
    db = connect()
    cur = db.execute(
        "INSERT INTO messages (room_id, author, text, created) VALUES (?, ?, ?, ?)",
        (room_id, author.strip()[:40], text.strip()[:2000], time.time()))
    db.commit()
    seq = cur.lastrowid
    db.close()
    return seq


def since(room_id, seq=0, limit=200):
    db = connect()
    rows = db.execute(
        "SELECT seq, author, text, created FROM messages "
        "WHERE room_id = ? AND seq > ? ORDER BY seq LIMIT ?",
        (room_id, seq, limit)).fetchall()
    db.close()
    return [dict(r) for r in rows]


def latest_seq(room_id):
    db = connect()
    row = db.execute("SELECT MAX(seq) AS s FROM messages WHERE room_id = ?",
                     (room_id,)).fetchone()
    db.close()
    return row["s"] or 0


def set_paid(room_id, paid=True):
    db = connect()
    db.execute("UPDATE rooms SET paid = ? WHERE id = ?", (1 if paid else 0, room_id))
    db.commit()
    db.close()


def recent_context(room_id, turns=4):
    """The last few messages, as the prompt Flow continues from.

    Only a handful: the model's whole memory is 128 tokens, so feeding it more
    would push the newest message out of view - the opposite of helpful.
    """
    db = connect()
    rows = db.execute(
        "SELECT author, text FROM messages WHERE room_id = ? "
        "ORDER BY seq DESC LIMIT ?", (room_id, turns)).fetchall()
    db.close()
    lines = [f"{r['author']}: {r['text']}" for r in reversed(rows)]
    return "\n".join(lines)


if __name__ == "__main__":
    if os.path.exists(DB):
        os.remove(DB)

    print("=== ROOMS SELF-TEST ===\n")
    checks = []

    room = create_room("Science Project", "Or")
    checks.append(("room created with a code", len(room["code"]) == 5,
                   f"code {room['code']}"))

    ok, info = join(room["code"], "Ariel")
    checks.append(("second person can join", ok, f"members {info.get('members')}"))

    # The third person must be refused: this is the product.
    ok3, info3 = join(room["code"], "Eden")
    checks.append(("third person BLOCKED on free", not ok3,
                   f"{info3.get('error')} at {info3.get('seats')} seats"))

    set_paid(room["id"], True)
    ok4, info4 = join(room["code"], "Eden")
    checks.append(("third person allowed once paid", ok4,
                   f"members {info4.get('members')}"))

    s1 = say(room["id"], "Or", "hello everyone")
    s2 = say(room["id"], "Ariel", "hi Or")
    checks.append(("messages get increasing sequence numbers", s2 > s1,
                   f"{s1} then {s2}"))

    new = since(room["id"], s1)
    checks.append(("clients can ask for what they missed", len(new) == 1
                   and new[0]["author"] == "Ariel",
                   f"{len(new)} message after seq {s1}"))

    everyone = since(room["id"], 0)
    checks.append(("everyone sees the same order", [m["author"] for m in everyone]
                   == ["Or", "Ariel"], " then ".join(m["author"] for m in everyone)))

    bad, binfo = join("ZZZZZ", "Nobody")
    checks.append(("bad code refused", not bad, binfo.get("error")))

    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    n = sum(c[1] for c in checks)
    print(f"\n{n}/{len(checks)} passed — {'ROOMS READY' if n == len(checks) else 'NOT READY'}")
    sys.exit(0 if n == len(checks) else 1)
