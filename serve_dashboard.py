#!/usr/bin/env python3
"""
Serve the OI Argus dashboard — and ONLY the front_end/ folder.

Why this script exists:
  `python3 -m http.server` serves the directory you launch it from, recursively.
  Run from the repo root it would expose .env (Azure keys!), data/, agents/,
  chroma_db/ … over HTTP. This launcher pins the web root to front_end/ so the
  only things reachable are the dashboard HTML and its dashboard_data.json feed.
  Everything sensitive lives ABOVE the web root and is therefore unreachable.

Real-time replay:
  The detection/agent run does NOT start at server boot. It starts when the
  operator reaches the control room — oiargus.html calls POST
  /api/start on load, i.e. AFTER onboarding has set thresholds + role. So the
  run always reflects the operator's just-saved values. LIVE_AGENTS=1 runs the
  real 6-agent chain (agents/run_live_pipeline.py); otherwise the fast no-LLM
  replay (agents/simulate_realtime.py). Stopped automatically on Ctrl+C, and
  restarted on the next entry when new onboarding values are saved.

Usage:
    python serve_dashboard.py            # http://localhost:8777/  → onboarding → control room
    LIVE_AGENTS=1 python serve_dashboard.py  # real agent chain (LLM + RAG)
    PORT=9000 python serve_dashboard.py  # custom port
    SPEED=60 python serve_dashboard.py   # fast-forward pacing
    AUTO_REPLAY=0 python serve_dashboard.py  # serve only; never start agents
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
import threading
from pathlib import Path

REPO_ROOT   = Path(__file__).parent                         # holds shift_config.json (NOT served)
WEB_ROOT    = REPO_ROOT / "front_end"                       # the ONLY folder exposed
SIM_SCRIPT  = REPO_ROOT / "agents" / "simulate_realtime.py"
LIVE_SCRIPT = REPO_ROOT / "agents" / "run_live_pipeline.py"   # LIVE_AGENTS=1 → real 6-agent chain
SHIFT_CONFIG = REPO_ROOT / "shift_config.json"              # written by the pre-shift onboarding screen
PORT        = int(os.environ.get("PORT", "8777"))
HOST        = "127.0.0.1"                                    # localhost only — not the LAN
AUTO_REPLAY = os.environ.get("AUTO_REPLAY", "1") != "0"     # allow the dashboard to start agents

# ── Managed replay/agent process ──────────────────────────────────────────────
# The detection/agent run is NO LONGER started at server boot. It starts only
# when the operator reaches the control room (oiargus.html calls
# POST /api/start on load) — i.e. AFTER onboarding has set thresholds + role.
_replay_proc = None
_replay_lock = threading.Lock()


def _replay_alive():
    return _replay_proc is not None and _replay_proc.poll() is None


def write_blank_feed():
    """Reset dashboard_data.json to an EMPTY board *synchronously*, right now.

    The agent process takes several seconds to boot (imports + ChromaDB indexes)
    before it writes its own blank board. Without this, the dashboard's first
    poll on entry reads the PREVIOUS run's leftover feed and briefly shows stale
    anomalies until the new run resets it. Blanking here — before the subprocess
    even starts — means the board is already empty the instant the operator
    arrives, then fills in live. Reuses build_payload so the shape never drifts.
    """
    try:
        agents_dir = str(REPO_ROOT / "agents")
        if agents_dir not in sys.path:
            sys.path.insert(0, agents_dir)
        import pandas as pd
        from dashboard_export import build_payload, write_dashboard_json, resolve_thresholds
        # header-only frames → valid empty board without loading the full CSVs
        m = pd.read_csv(REPO_ROOT / "data" / "train_machine.csv", nrows=0)
        m.columns = [c.strip().lower().replace(" ", "_") for c in m.columns]
        s = pd.read_csv(REPO_ROOT / "data" / "train_summary.csv", nrows=0)
        blank = build_payload([], [], m, s, resolve_thresholds())
        blank["generated_at"] = blank["sim_clock"] = "--:--:--"
        blank["current_stage"] = "production"   # detection agent is coming online
        write_dashboard_json(blank, verbose=False)
    except Exception as e:
        print(f"  [agents] blank-feed skipped ({e})")


def start_agents():
    """Launch the replay/agent process if not already running. Idempotent, so a
    dashboard refresh won't reset a run in progress. Picks the LIVE 6-agent chain
    when LIVE_AGENTS=1, else the fast no-LLM replay. Reads shift_config.json, so
    the operator's just-saved onboarding thresholds/role take effect."""
    global _replay_proc
    with _replay_lock:
        if not AUTO_REPLAY:
            return {"ok": False, "status": "disabled"}
        if _replay_alive():
            return {"ok": True, "status": "already_running", "pid": _replay_proc.pid}
        live   = os.environ.get("LIVE_AGENTS") == "1"
        script = LIVE_SCRIPT if live else SIM_SCRIPT
        if not script.exists():
            return {"ok": False, "status": "missing", "detail": str(script)}
        # Wipe any stale feed BEFORE the (slow-to-boot) run starts, so the board
        # is empty the moment the operator enters — no leftover anomalies.
        write_blank_feed()
        _replay_proc = subprocess.Popen([sys.executable, str(script)], env=os.environ.copy())
        mode = "LIVE AGENTS (real chain)" if live else "no-LLM replay"
        print(f"  [agents] started {mode} (PID {_replay_proc.pid}) — operator entered the control room.")
        return {"ok": True, "status": "started", "pid": _replay_proc.pid, "mode": mode}


def stop_agents():
    """Terminate the replay/agent process if running (never outlives the server;
    also called on new onboarding so the next entry starts fresh)."""
    global _replay_proc
    with _replay_lock:
        if _replay_alive():
            _replay_proc.terminate()
            try:
                _replay_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _replay_proc.kill()
        _replay_proc = None


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
        # "Enter Control Room" button links on to oiargus.html.
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
        path = self.path.split("?", 1)[0]

        # Started by oiargus.html on load — i.e. once the operator has
        # finished onboarding and entered the control room. This (not server boot)
        # is what kicks off detection/the agent chain.
        if path == "/api/start":
            self._send_json(200, start_agents())
            return

        # The pre-shift setup writer: persist the operator's thresholds + role, and
        # stop any in-flight run so the NEXT control-room entry starts fresh on the
        # new values.
        if path == "/api/shift-config":
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
            stop_agents()   # new config → drop the old run; dashboard entry restarts it
            print(f"  [shift-config] saved role={clean['role']} "
                  f"thresholds={len(clean['thresholds'])} → {SHIFT_CONFIG.name}")
            self._send_json(200, {"ok": True, "file": SHIFT_CONFIG.name})
            return

        self._send_json(404, {"ok": False, "error": "not found"})


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
        print("OI Argus dashboard")
        print(f"  Open:        http://localhost:{port}/   (→ pre-shift setup, then the control room)")
        print(f"  Control room:http://localhost:{port}/oiargus.html")
        print(f"  Serving ONLY: {WEB_ROOT}")
        print("  (.env, data/, agents/, chroma_db/ are NOT exposed)")
        if AUTO_REPLAY:
            print("  Agents:      start when the operator ENTERS the control room")
            print("               (after onboarding sets thresholds + role) — not at boot.")
        else:
            print("  Agents:      disabled (AUTO_REPLAY=0) — serving existing data as-is.")

        # Backstop: never let the agent process outlive the server.
        atexit.register(stop_agents)

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
            stop_agents()


if __name__ == "__main__":
    main()
