#!/usr/bin/env python3
"""HTTP server with COOP/COEP headers required for wasm pthreads (SharedArrayBuffer).

Run from the directory that contains index.html (e.g. your CMake build folder):
  python path/to/tools/serve_emscripten.py
  python path/to/tools/serve_emscripten.py 8765
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Serving {os.getcwd()} on http://127.0.0.1:{port}/ (COOP/COEP for pthread wasm)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
