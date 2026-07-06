#!/usr/bin/env python3
"""
Serve the FABWATCH dashboard — and ONLY the front_end/ folder.

Why this script exists:
  `python3 -m http.server` serves the directory you launch it from, recursively.
  Run from the repo root it would expose .env (Azure keys!), data/, agents/,
  chroma_db/ … over HTTP. This launcher pins the web root to front_end/ so the
  only things reachable are the dashboard HTML and its dashboard_data.json feed.
  Everything sensitive lives ABOVE the web root and is therefore unreachable.

Real-time replay:
  On startup this launches agents/simulate_realtime.py as a managed background
  process, so every time you start the server the dashboard RESETS to the 06:02
  simulated start (empty board) and then fills in live as the day replays. The
  replay is stopped automatically when you Ctrl+C the server.

Usage:
    python serve_dashboard.py            # http://localhost:8777/fabwatch_dashboard.html
    PORT=9000 python serve_dashboard.py  # custom port
    SPEED=60 python serve_dashboard.py   # fast-forward the replay (see simulate_realtime.py)
    AUTO_REPLAY=0 python serve_dashboard.py  # serve only; don't auto-start the replay
"""
import atexit
import functools
import http.server
import os
import signal
import socketserver
import subprocess
import sys
from pathlib import Path

WEB_ROOT    = Path(__file__).parent / "front_end"           # the ONLY folder exposed
SIM_SCRIPT  = Path(__file__).parent / "agents" / "simulate_realtime.py"
PORT        = int(os.environ.get("PORT", "8777"))
HOST        = "127.0.0.1"                                    # localhost only — not the LAN
AUTO_REPLAY = os.environ.get("AUTO_REPLAY", "1") != "0"     # start the replay on launch


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Always fetch a fresh data feed (the dashboard polls every 5s).
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class Server(socketserver.TCPServer):
    allow_reuse_address = True   # avoid "Address already in use" on lingering sockets


def start_replay():
    """
    Launch the real-time replay in the background so the board resets to the
    06:02 start and fills in live every time the server starts.

    Uses THIS interpreter (so it picks up the same env's dependencies) and
    inherits the environment, so SPEED / START / TICK / LOOP still work.
    Returns the Popen, or None if disabled/unavailable.
    """
    if not AUTO_REPLAY:
        print("  Replay:      disabled (AUTO_REPLAY=0) — run simulate_realtime.py yourself.")
        return None
    if not SIM_SCRIPT.exists():
        print(f"  Replay:      [warn] {SIM_SCRIPT} not found — serving existing data as-is.")
        return None
    proc = subprocess.Popen([sys.executable, str(SIM_SCRIPT)], env=os.environ.copy())
    print(f"  Replay:      started (PID {proc.pid}) — board resets to 06:02 and fills in live.")
    print(f"               (SPEED/START/TICK env vars tune it; AUTO_REPLAY=0 disables.)")
    return proc


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
        sim = start_replay()

        def stop_sim():
            # Stop the background replay so it never outlives the server.
            if sim and sim.poll() is None:
                sim.terminate()
                try:
                    sim.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sim.kill()

        # Backstop: run cleanup on any interpreter exit.
        atexit.register(stop_sim)

        # Make BOTH Ctrl+C (SIGINT) and `kill` (SIGTERM) shut down cleanly.
        # Setting SIGINT explicitly also covers the case where the process was
        # started in the background with SIGINT inherited as "ignore".
        def _on_signal(signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        print("  Ctrl+C to stop.")
        print("─" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            stop_sim()


if __name__ == "__main__":
    main()
