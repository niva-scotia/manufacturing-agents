#!/usr/bin/env python3
"""
Serve the FABWATCH dashboard — and ONLY the front_end/ folder.

Why this script exists:
  `python3 -m http.server` serves the directory you launch it from, recursively.
  Run from the repo root it would expose .env (Azure keys!), data/, agents/,
  chroma_db/ … over HTTP. This launcher pins the web root to front_end/ so the
  only things reachable are the dashboard HTML and its dashboard_data.json feed.
  Everything sensitive lives ABOVE the web root and is therefore unreachable.

Usage:
    python serve_dashboard.py            # http://localhost:8777/fabwatch_dashboard.html
    PORT=9000 python serve_dashboard.py  # custom port
"""
import functools
import http.server
import os
import socketserver
from pathlib import Path

WEB_ROOT = Path(__file__).parent / "front_end"   # the ONLY folder exposed
PORT     = int(os.environ.get("PORT", "8777"))
HOST     = "127.0.0.1"                            # localhost only — not the LAN


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Always fetch a fresh data feed (the dashboard polls every 5s).
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class Server(socketserver.TCPServer):
    allow_reuse_address = True   # avoid "Address already in use" on lingering sockets


def main():
    if not WEB_ROOT.is_dir():
        raise SystemExit(f"front_end/ not found at {WEB_ROOT}")
    handler = functools.partial(Handler, directory=str(WEB_ROOT))

    # Try the requested port, then roll forward to the next free one.
    httpd, port = None, PORT
    for port in range(PORT, PORT + 20):
        try:
            httpd = Server((HOST, port), handler)
            break
        except OSError:
            if port == PORT:
                print(f"Port {PORT} is busy — trying the next free port…")
    if httpd is None:
        raise SystemExit(f"No free port in {PORT}–{PORT + 19}. Set PORT=<n> and retry.")

    with httpd:
        print("─" * 60)
        print("FABWATCH dashboard")
        print(f"  Open:        http://localhost:{port}/fabwatch_dashboard.html")
        print(f"  Serving ONLY: {WEB_ROOT}")
        print("  (.env, data/, agents/, chroma_db/ are NOT exposed)")
        print("  Ctrl+C to stop.")
        print("─" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
