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
                body = {"ready": True, "val_loss": round(meta["val_loss"], 4),
                        "iter": meta["iter"], "vocab_size": tok.vocab_size}
            except Exception as e:
                body = {"ready": False, "reason": f"{type(e).__name__}: {e}"}

        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "bad JSON")
            return

        prompt = str(req.get("prompt", ""))[:2000]
        max_tokens = min(int(req.get("max_tokens", 120)), 400)
        temperature = float(req.get("temperature", 0.8))

        if not os.path.exists(CKPT):
            self.send_error(503, "model not trained yet")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

        from sample import stream
        try:
            with gen_lock:
                for piece in stream(prompt, max_tokens, temperature):
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
