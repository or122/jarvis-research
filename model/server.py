"""Flow's inference server — the only Python the app talks to.

The model is PyTorch, so generation has to happen in Python. Everything Or
edits day to day is TypeScript in web/.

Uses only the standard library: no Flask, no FastAPI, nothing to install.

Run:  .venv/bin/python model/server.py
      GET  /health    -> is the model loaded, and how good is it
      POST /generate  -> {"prompt": "..."} streams back Server-Sent Events
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PORT = int(os.environ.get("FLOW_PORT", "8000"))
CKPT = os.path.join(HERE, "ckpt.pt")

# One generation at a time. Two concurrent runs on a CPU this size make both
# slower than running them in sequence.
gen_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        # The Vite dev server runs on a different port, so the browser treats
        # this as cross-origin and blocks it without these headers.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/rooms/"):
            self.handle_room_get()
            return
        if self.path == "/brain":
            # Rebuilt per request rather than cached: the graph must change the
            # moment someone teaches Jarvis something, and 62 facts is trivial
            # to recompute.
            from brain_graph import build
            self._json(build())
            return
        if self.path != "/health":
            self.send_error(404)
            return

        body = {"ready": False, "reason": "no checkpoint yet"}
        if os.path.exists(CKPT):
            try:
                from sample import load
                _, tok, meta = load()
                from knowledge import count
                body = {"ready": True, "val_loss": round(meta["val_loss"], 4),
                        "iter": meta["iter"], "vocab_size": tok.vocab_size,
                        "facts": count()}
            except Exception as e:
                body = {"ready": False, "reason": f"{type(e).__name__}: {e}"}

        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, body):
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if not (self.path in ("/generate", "/teach") or self.path.startswith("/rooms")):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "bad JSON")
            return

        if self.path.startswith("/rooms"):
            self.handle_room_post(req)
            return

        if self.path == "/teach":
            from knowledge import count, teach
            question = str(req.get("question", "")).strip()[:500]
            fact = str(req.get("answer", "")).strip()[:1000]
            if not question or not fact:
                self.send_error(400, "need a question and an answer")
                return
            teach(question, fact)
            self._json({"ok": True, "facts": count()})
            return

        prompt = str(req.get("prompt", ""))[:2000]
        max_tokens = min(int(req.get("max_tokens", 120)), 400)
        temperature = float(req.get("temperature", 0.8))
        # chat=True (the default) wraps the message as a conversation turn so
        # the model replies. chat=False makes it carry the text on instead.
        as_chat = bool(req.get("chat", True))

        if not os.path.exists(CKPT):
            self.send_error(503, "model not trained yet")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

        from knowledge import look_up
        from sample import chat_stream, stream

        # Memory first. A stored fact beats a small model's guess every time,
        # and it comes back instantly with no generation at all.
        if as_chat:
            fact, _ = look_up(prompt)
            if fact:
                frame = json.dumps({"t": fact, "src": "memory"})
                self.wfile.write(f"data: {frame}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

        producer = chat_stream if as_chat else stream
        try:
            with gen_lock:
                for piece in producer(prompt, max_tokens, temperature):
                    # SSE frame: "data: {json}\n\n". JSON-encoding the text
                    # keeps newlines from breaking the frame format.
                    self.wfile.write(f"data: {json.dumps({'t': piece})}\n\n".encode())
                    self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass   # the browser navigated away mid-generation; nothing to do

    # ---------------------------------------------------------------- rooms
    def handle_room_post(self, req):
        import rooms

        if self.path == "/rooms":
            self._json(rooms.create_room(str(req.get("name", "Room")),
                                         str(req.get("owner", "You"))))
            return

        if self.path == "/rooms/join":
            ok, info = rooms.join(str(req.get("code", "")),
                                  str(req.get("name", "Guest")))
            # 402 Payment Required is the honest status for "this room is full
            # and more seats cost money" — it is exactly what the code means.
            payload = json.dumps(info).encode()
            self.send_response(200 if ok else 402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self._cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[2] == "say":
            text = str(req.get("text", "")).strip()
            if not text:
                self.send_error(400, "empty message")
                return
            room_id = int(parts[1])
            seq = rooms.say(room_id, str(req.get("name", "Guest")), text)
            # Flow answers on a background thread so the sender's request
            # returns immediately — generation takes seconds on this CPU.
            threading.Thread(target=self._reply_later, args=(room_id, seq),
                             daemon=True).start()
            self._json({"seq": seq})
            return

        if len(parts) == 3 and parts[2] == "paid":
            rooms.set_paid(int(parts[1]), bool(req.get("paid", True)))
            self._json({"ok": True})
            return

        self.send_error(404)

    def _reply_later(self, room_id, seq):
        """Wait a moment, then answer once for everything just said.

        The pause means two people typing together get ONE reply that has read
        both, instead of two replies talking over each other. If someone else
        posts while this waits, this thread stands down and theirs answers.
        """
        import rooms
        from knowledge import look_up
        from sample import chat

        time.sleep(1.5)
        if rooms.latest_seq(room_id) != seq:
            return

        last = rooms.since(room_id, seq - 1)
        question = last[0]["text"] if last else ""

        fact, _ = look_up(question)
        if fact:
            rooms.say(room_id, "Flow", fact)
            return

        with gen_lock:
            try:
                reply = chat(rooms.recent_context(room_id),
                             max_new_tokens=80, temperature=0.8, top_k=40)
            except Exception as e:
                reply = f"(Flow could not answer: {type(e).__name__})"
        rooms.say(room_id, "Flow", reply.strip() or "...")

    def handle_room_get(self):
        import rooms

        path, _, query = self.path.partition("?")
        parts = path.strip("/").split("/")
        if len(parts) != 2 or not parts[1].isdigit():
            self.send_error(404)
            return

        after = 0
        for bit in query.split("&"):
            if bit.startswith("since="):
                after = int(bit[6:] or 0)

        room_id = int(parts[1])
        room = rooms.get_room(room_id)
        if not room:
            self.send_error(404, "no such room")
            return

        self._json({"name": room["name"], "code": room["code"],
                    "paid": bool(room["paid"]), "seats": rooms.seats(room),
                    "members": rooms.members(room_id),
                    "messages": rooms.since(room_id, after)})

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    ready = "trained" if os.path.exists(CKPT) else "NOT TRAINED YET"
    print(f"Flow inference server on http://localhost:{PORT}   model: {ready}")
    print("  GET  /health")
    print("  POST /generate  {\"prompt\": \"...\"}", flush=True)
    # 127.0.0.1 means "this Mac only" and is the safe default. Rooms need
    # other devices to reach the server, so FLOW_HOST=0.0.0.0 opens it to the
    # local network — anyone on the same wifi can then use it. That is the
    # point for a family room, and a reason not to run it on public wifi.
    HOST = os.environ.get("FLOW_HOST", "127.0.0.1")
    if HOST != "127.0.0.1":
        print(f"  WARNING: reachable by anyone on this network ({HOST})")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
