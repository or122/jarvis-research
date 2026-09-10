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
        if self.path not in ("/generate", "/teach"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "bad JSON")
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

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    ready = "trained" if os.path.exists(CKPT) else "NOT TRAINED YET"
    print(f"Flow inference server on http://localhost:{PORT}   model: {ready}")
    print("  GET  /health")
    print("  POST /generate  {\"prompt\": \"...\"}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
