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
import json
import os
import signal
import socketserver
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent                         # holds shift_config.json (NOT served)
WEB_ROOT    = REPO_ROOT / "front_end"                       # the ONLY folder exposed
SIM_SCRIPT  = REPO_ROOT / "agents" / "simulate_realtime.py"
SHIFT_CONFIG = REPO_ROOT / "shift_config.json"              # written by the pre-shift onboarding screen
PORT        = int(os.environ.get("PORT", "8777"))
HOST        = "127.0.0.1"                                    # localhost only — not the LAN
AUTO_REPLAY = os.environ.get("AUTO_REPLAY", "1") != "0"     # start the replay on launch

# Contract shared with front_end/onboarding.html and agents/production_agent.py.
ALLOWED_SENSORS = {"tcp_top_pwr", "bcl3_flow", "cl2_flow", "pressure", "rf_btm_pwr"}
ALLOWED_ROLES   = {"operator", "supervisor", "quality_engineer", "maintenance_lead"}


def sanitize_shift_config(payload):
    """
    Validate the onboarding payload BEFORE it touches disk.

    Only the five known sensors and four known roles are accepted, each min must
    be strictly below its max, and the output path is fixed (shift_config.json at
    the repo root). Nothing here can write outside that file or inject unknown
    keys — the endpoint is a narrow, single-purpose config writer.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    role = payload.get("role")
    if role not in ALLOWED_ROLES:
        raise ValueError(f"role must be one of {sorted(ALLOWED_ROLES)}")

    src = payload.get("thresholds")
    if not isinstance(src, dict):
        raise ValueError("thresholds must be a JSON object")

    thresholds = {}
    for sensor, rng in src.items():
        if sensor not in ALLOWED_SENSORS:
            continue
        try:
            lo, hi = float(rng["min"]), float(rng["max"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{sensor}: min and max must be numbers")
        if not lo < hi:
            raise ValueError(f"{sensor}: min ({lo}) must be < max ({hi})")
        thresholds[sensor] = {"min": lo, "max": hi}

    if not thresholds:
        raise ValueError("no valid sensor thresholds supplied")

    return {"role": role, "thresholds": thresholds, "saved_at": payload.get("saved_at")}


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Always fetch a fresh data feed (the dashboard polls every 5s).
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        # The pre-shift onboarding screen is the front door: hitting the site
        # root sends the operator to set thresholds + role first. From there the
        # "Enter Control Room" button links on to fabwatch_dashboard.html.
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/onboarding.html")
            self.end_headers()
            return
        super().do_GET()

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        # The one and only write endpoint: persist the pre-shift setup so the
        # agent chain reads the operator's thresholds instead of hard-coded ones.
        if self.path.split("?", 1)[0] != "/api/shift-config":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length  = int(self.headers.get("Content-Length", 0))
            raw     = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            clean   = sanitize_shift_config(payload)
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return
        try:
            SHIFT_CONFIG.write_text(json.dumps(clean, indent=2))
        except OSError as e:
            self._send_json(500, {"ok": False, "error": str(e)})
            return
        print(f"  [shift-config] saved role={clean['role']} "
              f"thresholds={len(clean['thresholds'])} → {SHIFT_CONFIG.name}")
        self._send_json(200, {"ok": True, "file": SHIFT_CONFIG.name})


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
        print(f"  Open:        http://localhost:{port}/   (→ pre-shift setup, then the control room)")
        print(f"  Control room:http://localhost:{port}/fabwatch_dashboard.html")
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
