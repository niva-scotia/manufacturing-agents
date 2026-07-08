#!/usr/bin/env python3
"""
Real-time playback of the anomaly-detection agent (time-warped replay).

The datasets carry no absolute clock, so we replay each detected event at its
recorded TIME OF DAY (train_summary.run_start) and advance a simulated clock
through the day. Every tick we rewrite front_end/dashboard_data.json to contain
only the events detected SO FAR — so the dashboard (which polls every 5s) shows
anomalies appear as time moves forward. Everything is derived from the real
CSVs via the same detection engine + payload builder; nothing is invented and
the UI is untouched.

How it works
------------
1. Run the real detector once over data/train_machine.csv  → full event list.
2. Give each event a sort key = seconds-since-midnight of its run_start.
3. Advance a simulated clock from START (default: first event − 13s = 06:02:20)
   at SPEED sim-seconds per real second.
4. Each tick, reveal events with time ≤ sim-now (and the wafers seen so far),
   rebuild the payload from that growing subset, and write the JSON.

Run it alongside the server:
    # terminal 1
    python serve_dashboard.py
    # terminal 2
    python agents/simulate_realtime.py           # SPEED=60 by default

Env knobs
---------
    SPEED   sim-seconds per real second (default 1 = true wall-clock real time:
            first anomaly appears ~13s after launch. Raise it, e.g. SPEED=60,
            to fast-forward ~3.5h of history into ~3.5min.)
    TICK    seconds between rewrites            (default 1.0)
    START   "HH:MM:SS" simulated start          (default: first event − 13s)
    LOOP    "1" to restart the day when it ends (default off)
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent

from dashboard_export import (          # noqa: E402
    detect_offline, build_payload, write_dashboard_json,
    THRESHOLDS, TREND_CONFIG, _wafer_start, _event_time,
    select_production_day, filter_machine_to_day, resolve_thresholds,
)


def _tod(hms):
    """'HH:MM:SS' → seconds since midnight."""
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def _fmt(sec):
    """seconds since midnight → 'HH:MM:SS'."""
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def main():
    speed = float(os.environ.get("SPEED", "1"))
    tick  = float(os.environ.get("TICK", "1.0"))
    loop  = os.environ.get("LOOP", "0") == "1"

    machine_df = pd.read_csv(_ROOT / "data" / "train_machine.csv")
    machine_df.columns = [c.strip().lower().replace(" ", "_") for c in machine_df.columns]
    summary_df = pd.read_csv(_ROOT / "data" / "train_summary.csv")
    # Replay ONE production day so the sim clock is a real, monotonic timeline
    # (the CSV holds multiple dates; collapsing them onto one 24h clock is the
    # source of the "06:02:15 isn't in the data" confusion — see ISSUES.md #1).
    summary_df, day = select_production_day(summary_df)
    machine_df = filter_machine_to_day(machine_df, summary_df)
    print(f"[sim] production day: {day}  ({len(summary_df)} wafers, {len(machine_df)} rows)")
    wafer_start = _wafer_start(summary_df)
    # Use the operator's onboarding thresholds (falls back to defaults if none).
    thresholds = resolve_thresholds()
    print(f"[sim] thresholds: {thresholds}")

    # Clear the board immediately so the dashboard starts EMPTY (no errors, no
    # trends, both chambers nominal) while detection runs — otherwise it would
    # keep showing the previous full-day snapshot for the first few seconds.
    blank = build_payload([], [], machine_df.iloc[0:0], summary_df.iloc[0:0], thresholds)
    blank["generated_at"] = blank["sim_clock"] = "--:--:--"
    write_dashboard_json(blank, verbose=False)

    print("[sim] running detection over full history (real data, no LLM)…")
    anoms, trends = detect_offline(machine_df, thresholds, TREND_CONFIG,
                                   wafer_start=wafer_start)

    # Chronological order + a seconds-of-day key, then STABLE ids so entries
    # never renumber as the feed grows.
    for e in anoms + trends:
        e["_tod"] = _tod(e["timestamp"])
    anoms.sort(key=lambda e: e["_tod"])
    trends.sort(key=lambda e: e["_tod"])
    for i, a in enumerate(anoms):
        a["_id"] = f"ANM-{4401 + i}"
    for i, t in enumerate(trends):
        t["_id"] = f"TRD-{1101 + i}"

    all_ev = anoms + trends
    if not all_ev:
        print("[sim] no events detected — nothing to replay.")
        return
    first = min(e["_tod"] for e in all_ev)
    last  = max(e["_tod"] for e in all_ev)
    start = _tod(os.environ["START"]) if os.environ.get("START") else first - 13

    # wafer → seconds-of-day of its run_start (to reveal chamber context in step)
    wafer_tod = {w: _tod(_event_time(rs)) for w, rs in wafer_start.items()
                 if _event_time(rs)}

    print(f"[sim] {len(anoms)} anomalies, {len(trends)} trends | "
          f"day window {_fmt(start)}–{_fmt(last)} | SPEED={speed}x  TICK={tick}s")
    print("[sim] open http://localhost:8777/fabwatch_dashboard.html and watch it fill in.")

    while True:
        wall0 = time.time()
        while True:
            sim = start + (time.time() - wall0) * speed

            revealed_a = [a for a in anoms  if a["_tod"] <= sim]
            revealed_t = [t for t in trends if t["_tod"] <= sim]
            seen_wafers = {w for w, td in wafer_tod.items() if td <= sim}

            # newest-first so the just-detected event shows at the top of the feed
            ra = list(reversed(revealed_a))
            rt = list(reversed(revealed_t))
            mdf = machine_df[machine_df["wafer_id"].isin(seen_wafers)]
            sdf = summary_df[summary_df["wafer_id"].isin(seen_wafers)]

            payload = build_payload(ra, rt, mdf, sdf, thresholds)
            payload["generated_at"] = _fmt(sim)
            payload["sim_clock"] = _fmt(sim)
            write_dashboard_json(payload, verbose=False)

            print(f"[sim] {_fmt(sim)}   anomalies={len(revealed_a):>3}  "
                  f"trends={len(revealed_t)}   ", end="\r", flush=True)

            if sim >= last + 5:
                break
            time.sleep(tick)

        print(f"\n[sim] reached end of day ({_fmt(last)}). "
              f"{'restarting…' if loop else 'done.'}")
        if not loop:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[sim] stopped.")
