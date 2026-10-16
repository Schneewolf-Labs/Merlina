"""
A tiny stand-in for ``vllm serve`` used by tests/test_remote_serve_worker.py.

Accepts the same argv shape (``serve <model> --host H --port P ...``),
enforces ``VLLM_API_KEY`` on /v1 only (like real vLLM, which leaves e.g.
/invocations and /health open), and speaks just enough of the OpenAI API:
GET /v1/models and POST /v1/chat/completions (JSON or SSE streaming).

Behaviour is selected by environment variables:
  FAKE_VLLM_RECORD  path to write {"argv": [...], "env": {...}} on start
  FAKE_VLLM_MODE    "ok" (default) | "crash" (exit 3 before healthy)
                    | "die" (exit 4 shortly after becoming healthy)
                    | "slow" (never becomes healthy)
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main(argv):
    assert argv[0] == "serve", argv
    model = argv[1]
    host = argv[argv.index("--host") + 1]
    port = int(argv[argv.index("--port") + 1])
    served = argv[argv.index("--served-model-name") + 1]
    key = os.environ.get("VLLM_API_KEY", "")
    mode = os.environ.get("FAKE_VLLM_MODE", "ok")

    record = os.environ.get("FAKE_VLLM_RECORD")
    if record:
        with open(record, "w") as f:
            json.dump({"argv": argv, "env": {
                "VLLM_API_KEY": key,
                "MERLINA_WORKER_TOKEN": os.environ.get("MERLINA_WORKER_TOKEN"),
                "MERLINA_SERVE_API_KEY": os.environ.get("MERLINA_SERVE_API_KEY"),
                "MERLINA_REMOTE_JOB_B64": os.environ.get("MERLINA_REMOTE_JOB_B64"),
            }}, f)

    print(f"INFO Loading model {model}", flush=True)
    if mode == "crash":
        # Real vLLM may echo its config; the worker must scrub the key.
        print(f"DEBUG engine args api_key={key}", flush=True)
        print("torch.OutOfMemoryError: CUDA out of memory", flush=True)
        sys.exit(3)
    if mode == "slow":
        time.sleep(3600)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            if not self.path.startswith("/v1"):
                return True
            return self.headers.get("Authorization") == f"Bearer {key}"

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {})
            if not self._authorized():
                return self._json(401, {"error": "Unauthorized"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": served}]})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/invocations":           # unguarded in real vLLM
                return self._json(200, {"leaked": True})
            if not self._authorized():
                return self._json(401, {"error": "Unauthorized"})
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            text = f"echo: {body['messages'][-1]['content']}"
            if not body.get("stream"):
                return self._json(200, {"model": body.get("model"), "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for word in text.split():
                event = "data: " + json.dumps({"choices": [{"delta": {"content": word}}]}) + "\n\n"
                data = event.encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
                time.sleep(0.01)
            done = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"INFO Serving {served} on {host}:{port}", flush=True)
    if mode == "die":
        threading.Timer(0.5, lambda: os._exit(4)).start()
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
