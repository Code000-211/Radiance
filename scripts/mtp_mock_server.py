#!/usr/bin/env python3
"""A stand-in radiance server for scripts/mtp_matrix.sh --mock-test.

Serves /health, /metrics and a chunked SSE /v1/completions with the shape the runner parses. It is
not a model and measures nothing: it exists so the runner's plumbing -- the prompt text in the
payload, the complete stream capture, client errors, both C edges, the lock and the signal cleanup
-- can be exercised without a card.

The counters follow the runner's own metric names, and `vllm:num_requests_running` is the number of
streams currently open, so a stream that ends before the window closes is visible as a closing edge
below C.

Environment:
  MOCK_CHUNKS     content chunks per request (default 20)
  MOCK_GAP_MS     milliseconds between chunks (default 50)
  MOCK_FAIL       the N-th request answers 500 instead of streaming (default 0, never)
  MOCK_JSON       "compact" (default) or "spaced": legal JSON with different spacing. Keys are
                  deliberately in an order the runner does not assume.
  MOCK_PIDFILE    write this process's PID here once it is listening
  MOCK_BODY_LOG   append every request body here, one per line
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHUNKS = int(os.environ.get("MOCK_CHUNKS", "20"))
GAP = float(os.environ.get("MOCK_GAP_MS", "50")) / 1000.0
FAIL_AT = int(os.environ.get("MOCK_FAIL", "0"))
SPACED = os.environ.get("MOCK_JSON", "compact") == "spaced"
PIDFILE = os.environ.get("MOCK_PIDFILE", "")
BODY_LOG = os.environ.get("MOCK_BODY_LOG", "")


def dumps(obj):
    if not SPACED:
        return json.dumps(obj, separators=(",", ":"))
    # A different legal rendering: spaces, and the top-level keys in the other order.
    return json.dumps({k: obj[k] for k in reversed(list(obj))}, separators=(", ", ": "))


lock = threading.Lock()
seen = {"active": 0, "requests": 0, "decode": 0, "steps": 0, "draft": 0, "acc": 0, "prefill": 0}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def plain(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.plain(200, "ok")
        if self.path == "/metrics":
            with lock:
                s = dict(seen)
            return self.plain(200, (
                "vllm:num_requests_running %d\n"
                "radiance:engine_steps_decode_total %d\n"
                "radiance:decode_tokens_total %d\n"
                "radiance:draft_tokens_total %d\n"
                "radiance:draft_accepted_total %d\n"
                "radiance:prefill_tokens_total %d\n") % (
                s["active"], s["steps"], s["decode"], s["draft"], s["acc"], s["prefill"]))
        self.plain(404, "no such path")

    def chunk(self, text):
        data = text.encode()
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        if BODY_LOG:
            with open(BODY_LOG, "ab") as f:
                f.write(body + b"\n")
        with lock:
            seen["requests"] += 1
            idx = seen["requests"]
            seen["active"] += 1
            seen["prefill"] += 32
        if FAIL_AT and idx == FAIL_AT:
            with lock:
                seen["active"] -= 1
            return self.plain(500, "mock failure")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for _ in range(CHUNKS):
                with lock:
                    seen["decode"] += 1
                    seen["steps"] += 1
                    seen["draft"] += 1
                    seen["acc"] += 1
                self.chunk("data: " + dumps(
                    {"choices": [{"index": 0, "text": "x"}]}) + "\n\n")
                time.sleep(GAP)
            # The usage line comes at the stream's end, after the window has closed.
            self.chunk("data: " + dumps(
                {"usage": {"completion_tokens": CHUNKS}, "choices": []}) + "\n\n")
            self.chunk("data: [DONE]\n\n")
            self.chunk("")
        finally:
            with lock:
                seen["active"] -= 1


def main():
    argv = sys.argv[1:]
    port = 8120
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if PIDFILE:
        with open(PIDFILE, "w") as f:
            f.write(str(os.getpid()))
    server.serve_forever()


if __name__ == "__main__":
    main()
