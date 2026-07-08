# # Dashboard Export
# Builds the JSON payload that backs front_end/fabwatch_dashboard.html.
#
# Two entry points:
#   1. build_payload(anomaly_log, trend_log, machine_df, summary_df, thresholds)
#      — called from production_agent.py after a full run so the dashboard
#        reflects the real anomalies/trends (with LLM explanations).
#   2. `python agents/dashboard_export.py` — runs the SAME deterministic
#      detection engine offline (no Azure / no LLM) over data/train_machine.csv
#      and writes front_end/dashboard_data.json. Fast, free, repeatable.
#
# The detection math here mirrors the engine in production_agent.py exactly.
# Only the LLM explanation step is omitted in the offline path.

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).parent.parent

# ── Sensors surfaced on the dashboard (label + unit), in display order. ───────
# Mirrors USER_THRESHOLDS in production_agent.py.
SENSORS = [
    ("tcp_top_pwr", "TCP TOP PWR", "W",     {"min": 334, "max": 360}),
    ("pressure",    "PRESSURE",    "mTorr", {"min": 942, "max": 1420}),
    ("rf_btm_pwr",  "RF BTM PWR",  "W",     {"min": 124, "max": 142}),
    ("bcl3_flow",   "BCl3 FLOW",   "sccm",  {"min": 740, "max": 765}),
]
THRESHOLDS = {s[0]: s[3] for s in SENSORS}
THRESHOLDS["cl2_flow"] = {"min": 748, "max": 758}
LABELS = {s[0]: s[1] for s in SENSORS}
UNITS  = {s[0]: s[2] for s in SENSORS}
# cl2_flow is monitored but not one of the 4 chamber-card sensors; give it a
# label/unit so per-event error tiles for it still render cleanly.
LABELS.setdefault("cl2_flow", "CL2 FLOW")
UNITS.setdefault("cl2_flow", "sccm")

TREND_CONFIG = {
    "min_steps": 8, "min_range_fraction": 0.15, "min_r_squared": 0.85,
    "min_seconds_to_breach": 5, "max_seconds_to_breach": 86400,
}

# ── Pre-shift onboarding config (thresholds + role the operator set) ──────────
# front_end/onboarding.html POSTs these to serve_dashboard.py, which writes
# shift_config.json at the repo root. Every feed producer resolves thresholds
# through resolve_thresholds() so the USER'S limits — not the hard-coded ones —
# label anomalies and trends. Absent/malformed file → hard-coded defaults.
SHIFT_CONFIG_PATH = _ROOT / "shift_config.json"


def load_shift_config():
    """Return the onboarding config dict ({thresholds, role, ...}) or {}."""
    try:
        if SHIFT_CONFIG_PATH.exists():
            return json.loads(SHIFT_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def resolve_thresholds(base=None):
    """THRESHOLDS with the operator's onboarding overrides applied (per sensor,
    only when min < max). Returns a fresh dict; never mutates the base."""
    thresholds = {k: dict(v) for k, v in (base or THRESHOLDS).items()}
    for sensor, rng in (load_shift_config().get("thresholds") or {}).items():
        if sensor not in thresholds:
            continue
        try:
            lo, hi = float(rng["min"]), float(rng["max"])
        except (KeyError, TypeError, ValueError):
            continue
        if lo < hi:
            thresholds[sensor] = {"min": lo, "max": hi}
    return thresholds


def shift_role():
    """The role/position the operator selected at onboarding, or None."""
    return load_shift_config().get("role")

# Static plant-floor context for the Live View modal (support tools that the
# agent does not monitor — CHA / CHB are filled in from live status).
SUPPORT_TOOLS = [
    {"id": "EFEM-LP1", "status": "idle",    "monitored": False, "machine": "EFEM Load Port",   "bay": "BAY 2"},
    {"id": "MET-1",    "status": "nominal", "monitored": False, "machine": "Metrology / CD-SEM","bay": "BAY 3"},
    {"id": "AMHS-S2",  "status": "nominal", "monitored": False, "machine": "AMHS Stocker",      "bay": "AISLE"},
    {"id": "WB-2",     "status": "nominal", "monitored": False, "machine": "Wet Bench Clean",   "bay": "BAY 3"},
]
CHAMBER_MACHINE = {"CHA": "Multi-Chamber Etch System", "CHB": "PECVD Deposition Cluster"}
CHAMBER_BAY     = {"CHA": "BAY 2 · ETCH",              "CHB": "BAY 3 · DEPOSITION"}


# ════════════════════════════════════════════════════════════════════════════
#  Detection engine (offline mirror of production_agent.py — no LLM)
# ════════════════════════════════════════════════════════════════════════════
def _check_anomaly(value, rng):
    return float(value) < rng["min"] or float(value) > rng["max"]


def _format_time_to_breach(steps):
    s = int(steps)
    if s < 60:    return f"{s} seconds"
    if s < 3600:  return f"{s // 60} minutes"
    return f"{s // 3600} hours"


def _compute_trend(values, rng, cfg):
    if len(values) < cfg["min_steps"]:
        return None
    values = np.array(values, dtype=float)
    current = values[-1]
    if current < rng["min"] or current > rng["max"]:
        return None
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((values - y_pred) ** 2)
    ss_tot = np.sum((values - values.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    if r2 < cfg["min_r_squared"]:
        return None
    allowed = rng["max"] - rng["min"]
    if abs(slope * cfg["min_steps"]) / allowed < cfg["min_range_fraction"]:
        return None
    if slope > 0:
        direction, distance = "increasing", rng["max"] - current
    else:
        direction, distance = "decreasing", current - rng["min"]
    if abs(slope) < 1e-9:
        return None
    steps = int(distance / abs(slope))
    if steps < cfg["min_seconds_to_breach"] or steps > cfg["max_seconds_to_breach"]:
        return None
    return {
        "direction": direction, "slope": round(float(slope), 6),
        "r_squared": round(float(r2), 4), "current_value": round(float(current), 3),
        "threshold_min": rng["min"], "threshold_max": rng["max"],
        "steps_to_breach": steps, "time_to_breach": _format_time_to_breach(steps),
        "values_seen": [round(float(v), 2) for v in values.tolist()],
    }


def detect_offline(machine_df, thresholds, cfg, wafer_start=None):
    """Pure-Python detection (no LLM). Returns (anomaly_log, trend_log) with the
    same record shape production_agent.py produces. Each record is timestamped
    with the wafer's real run_start (via wafer_start), not the scan clock."""
    anomaly_log, trend_log = [], []
    history, trend_warned = {}, set()
    wafer_start = wafer_start or {}
    now_str = datetime.now().strftime("%H:%M:%S")
    for _, row in machine_df.iterrows():
        wid, step = int(row["wafer_id"]), int(row["step"])
        ts = _event_time(wafer_start.get(wid), now_str)   # real run_start time
        if wid not in history:
            history[wid] = {s: [] for s in thresholds}
            trend_warned = {k for k in trend_warned if k[0] == wid}
        for sensor, rng in thresholds.items():
            if sensor not in machine_df.columns:
                continue
            value = float(row[sensor])
            if _check_anomaly(value, rng):
                anomaly_log.append({
                    "timestamp": ts, "wafer_id": wid, "step": step, "sensor": sensor,
                    "value": value, "threshold_min": rng["min"], "threshold_max": rng["max"],
                    "explanation": "",
                })
            else:
                buf = history[wid][sensor]
                buf.append(value)
                if len(buf) > cfg["min_steps"] * 3:
                    history[wid][sensor] = buf[-cfg["min_steps"] * 3:]
                key = (wid, sensor)
                if key not in trend_warned:
                    t = _compute_trend(history[wid][sensor], rng, cfg)
                    if t is not None:
                        trend_log.append({
                            "timestamp": ts, "wafer_id": wid, "step": step, "sensor": sensor,
                            "current_value": t["current_value"], "threshold_min": t["threshold_min"],
                            "threshold_max": t["threshold_max"], "trend_direction": t["direction"],
                            "rate_per_step": t["slope"], "r_squared": t["r_squared"],
                            "steps_to_breach": t["steps_to_breach"], "time_to_breach": t["time_to_breach"],
                            "values_seen": t["values_seen"], "explanation": "",
                        })
                        trend_warned.add(key)
    return anomaly_log, trend_log


# ════════════════════════════════════════════════════════════════════════════
#  Payload builder  (logs + raw data → dashboard JSON)
# ════════════════════════════════════════════════════════════════════════════
def select_production_day(summary_df, day=None):
    """Restrict the run to a SINGLE production day so the replay clock is coherent
    and every timestamp maps to exactly one real wafer.

    data/train_summary.csv holds multiple calendar dates (e.g. 2024-03-04 and
    2024-03-25). The feed elsewhere keeps only HH:MM:SS, so without this filter
    both days collapse onto one 24h clock and a later day's wafer (06:02:15) sorts
    ahead of the earlier day's — the "fake timestamp" bug.

    Returns (filtered_summary_df, day_str). Default day = earliest date present;
    override with the `day` arg or the SIM_DATE=YYYY-MM-DD environment variable.
    """
    if "run_start" not in summary_df.columns or summary_df.empty:
        return summary_df, None
    dates = pd.to_datetime(summary_df["run_start"], errors="coerce").dt.date
    day = day or os.environ.get("SIM_DATE")
    target = pd.to_datetime(day).date() if day else dates.dropna().min()
    filtered = summary_df[dates == target].copy()
    return filtered, str(target)


def filter_machine_to_day(machine_df, summary_df):
    """Keep only the machine rows whose wafers belong to the selected-day summary."""
    wafers = set(summary_df["wafer_id"])
    return machine_df[machine_df["wafer_id"].isin(wafers)].copy()


def _wafer_chamber(summary_df):
    return summary_df.set_index("wafer_id")["chamber_id"].to_dict()


def _wafer_lot(summary_df):
    """wafer_id → lot_id, so each incoming-error tile can name its own lot."""
    return (summary_df.set_index("wafer_id")["lot_id"].to_dict()
            if "lot_id" in summary_df.columns else {})


def _wafer_start(summary_df):
    """wafer_id → run_start (raw ISO string) from the summary, for real event times."""
    return (summary_df.set_index("wafer_id")["run_start"].to_dict()
            if "run_start" in summary_df.columns else {})


def _event_time(raw, fallback=""):
    """Format a run_start value ("2024-03-04T06:02:33") to "06:02:33"."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return fallback
    s = str(raw)
    if "T" in s: return s.split("T", 1)[1][:8]
    if " " in s: return s.split(" ", 1)[1][:8]
    return fallback or s


def _event_time_offset(raw, offset_sec, fallback=""):
    """run_start + offset_sec → "HH:MM:SS" (time-of-day).

    offset_sec is the reading's time_sec — seconds INTO the wafer run — so the
    stamp is the real moment the breach occurred, not the wafer's start time
    (which every step of a wafer shares). Keeps to a 24h clock, matching the
    single-production-day design.
    """
    base = _event_time(raw, "")
    if not base or ":" not in base:
        return fallback
    try:
        h, m, s = (int(x) for x in base.split(":"))
    except ValueError:
        return fallback or base
    total = (h * 3600 + m * 60 + s + int(round(float(offset_sec or 0)))) % 86400
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _fmt(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _dev_string(value, rng):
    if value > rng["max"]:
        return f"+{value - rng['max']:.2f} above max"
    return f"-{rng['min'] - value:.2f} below min"


def _secs(hms):
    """'HH:MM:SS' → seconds since midnight (for ranking events/wafers by time)."""
    try:
        h, m, s = (int(x) for x in str(hms).split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return -1


def _aggregate_anomalies(anomaly_log, machine_df, summary_df):
    """Collapse per-step breaches into ONE episode per (wafer, sensor).

    The detection engine flags every out-of-range reading, so one wafer whose
    sensor sits over the limit at several steps yields many near-identical
    records. Operators think in wafer excursions ("wafer 2911 TCP out of range"),
    not individual step readings — so we group by (wafer_id, sensor) and emit a
    single event with:
      • timestamp  = the FIRST breach's true time (run_start + time_sec)
      • value      = the PEAK reading (furthest outside the band)
      • breach_count / first_step / last_step / duration_sec
      • steps[]    = every individual breach (step, value, true time) for drill-in

    Centralised here so both the offline feed (detect_offline) and the real
    chain (production_agent.run_agent) collapse identically.
    """
    if not anomaly_log:
        return []

    ws = _wafer_start(summary_df)
    # (wafer_id, step) → time_sec, so each breach gets its real in-run offset.
    tmap = {}
    if "time_sec" in machine_df.columns:
        for wid, step, tsec in zip(machine_df["wafer_id"], machine_df["step"],
                                   machine_df["time_sec"]):
            tmap[(int(wid), int(step))] = float(tsec)

    groups = {}
    for a in anomaly_log:
        groups.setdefault((a["wafer_id"], a["sensor"]), []).append(a)

    episodes = []
    for (wid, sensor), items in groups.items():
        rng = {"min": items[0]["threshold_min"], "max": items[0]["threshold_max"]}
        for it in items:
            it["_t"] = tmap.get((int(wid), int(it["step"])), 0.0)
        ordered = sorted(items, key=lambda x: x["_t"])
        overshoot = lambda v: max(v - rng["max"], rng["min"] - v, 0)
        peak = max(items, key=lambda x: overshoot(x["value"]))
        first, last = ordered[0], ordered[-1]
        episodes.append({
            "timestamp": _event_time_offset(ws.get(wid), first["_t"]),
            "wafer_id": wid, "sensor": sensor, "step": peak["step"],
            "value": peak["value"],
            "threshold_min": rng["min"], "threshold_max": rng["max"],
            "explanation": peak.get("explanation", ""),
            "breach_count": len(items),
            "first_step": first["step"], "last_step": last["step"],
            "duration_sec": round(last["_t"] - first["_t"], 1),
            "steps": [{"step": it["step"], "value": round(float(it["value"]), 2),
                       "time": _event_time_offset(ws.get(wid), it["_t"])}
                      for it in ordered],
        })
    episodes.sort(key=lambda e: e["timestamp"])
    return episodes


def build_payload(anomaly_log, trend_log, machine_df, summary_df, thresholds=None):
    thresholds = thresholds or THRESHOLDS
    machine_df = machine_df.copy()
    machine_df.columns = [c.strip().lower().replace(" ", "_") for c in machine_df.columns]
    w2c = _wafer_chamber(summary_df)
    w2l = _wafer_lot(summary_df)

    # Collapse per-step breaches into one episode per wafer+sensor (see helper).
    anomaly_log = _aggregate_anomalies(anomaly_log, machine_df, summary_df)

    # Event times are already carried on each record (production_agent stamps
    # them from run_start; detect_offline does the same). We just pull them
    # through here — no override.

    # tag every log entry with its chamber
    for a in anomaly_log: a["chamber"] = w2c.get(a["wafer_id"], "CHA")
    for t in trend_log:   t["chamber"] = w2c.get(t["wafer_id"], "CHA")

    chambers, errors, trends, tools = {}, [], [], []

    # ── incoming-error feed. IDs are assigned only if the caller hasn't already
    #    set one (the real-time simulator pre-assigns stable IDs so entries don't
    #    renumber as the feed grows). ──
    for i, a in enumerate(anomaly_log):
        a.setdefault("_id", f"ANM-{4401 + i}")
        rng = {"min": a["threshold_min"], "max": a["threshold_max"]}
        label = LABELS.get(a["sensor"], a["sensor"])
        errors.append({
            "chamber": a["chamber"], "pc": label, "type": "anomaly",
            "text": f"{label} out of range — {_dev_string(a['value'], rng)}",
            "id": a["_id"], "time": a["timestamp"],
            # ── per-event identity: makes every tile unique + traceable to the
            #    dataset, and lets the UI open THIS event (not the chamber's
            #    latest breach) when a tile is clicked. ──
            "wafer_id": a["wafer_id"],
            "lot": str(w2l.get(a["wafer_id"], "—")),
            "sensor": a["sensor"],
            "value": round(float(a["value"]), 2),
            "unit": UNITS.get(a["sensor"], ""),
            "threshold_min": rng["min"], "threshold_max": rng["max"],
            "deviation": _dev_string(a["value"], rng),
            "step": a.get("step"),
            "explanation": a.get("explanation", ""),
            # episode aggregation — the tile is one wafer excursion, not one step
            "breach_count": a.get("breach_count", 1),
            "first_step": a.get("first_step", a.get("step")),
            "last_step": a.get("last_step", a.get("step")),
            "duration_sec": a.get("duration_sec", 0),
            "steps": a.get("steps", []),
        })
    for i, t in enumerate(trend_log):
        t.setdefault("_id", f"TRD-{1101 + i}")

    # ── trend watch (up to 3 most advanced) ──────────────────────────────────
    def _progress(t):
        rng = {"min": t["threshold_min"], "max": t["threshold_max"]}
        band = rng["max"] - rng["min"]
        cur = t["current_value"]
        pct = (cur - rng["min"]) / band * 100 if t["trend_direction"] == "increasing" else (rng["max"] - cur) / band * 100
        return max(2, min(99, int(round(pct))))

    ranked = sorted(trend_log, key=_progress, reverse=True)[:3]
    for t in ranked:
        pct = _progress(t)
        tone = "red" if pct >= 80 else "yellow" if pct >= 50 else "green"
        rising = t["trend_direction"] == "increasing"
        trends.append({
            "label": f"{t['chamber']} · {LABELS.get(t['sensor'], t['sensor'])}",
            "pct": pct, "tone": tone,
            "status": f"{'rising' if rising else 'falling'} — monitoring",
            "data": t.get("values_seen", [])[-20:] or [pct - 4, pct - 2, pct],
        })

    # ── per-chamber rollup ───────────────────────────────────────────────────
    for ch in ["CHA", "CHB"]:
        srows = summary_df[summary_df["chamber_id"] == ch]
        ch_anoms  = [a for a in anomaly_log if a["chamber"] == ch]
        ch_trends = [t for t in trend_log   if t["chamber"] == ch]
        status = "fault" if ch_anoms else "watch" if ch_trends else "nominal"

        # Meta comes from the CURRENT wafer chosen BY TIME (latest run_start),
        # not by wafer_id — wafer_id order ≠ time order across production days.
        srows = srows.copy()
        total = len(srows)
        fails = int((srows["pass_fail"] == "FAIL").sum()) if total else 0
        if total:
            srows["_tod"] = srows["run_start"].map(lambda r: _secs(_event_time(r)))
            meta = srows.sort_values("_tod").iloc[-1]      # most recent wafer BY TIME
        else:
            meta = None

        # Last reading of that current-by-time wafer — used only for sensors that
        # have NOT tripped an alarm (so they show a live in-range value).
        last = None
        if meta is not None:
            wrows = machine_df[machine_df["wafer_id"] == int(meta["wafer_id"])]
            if len(wrows):
                last = wrows.sort_values("step").iloc[-1]

        # For each sensor, the MOST RECENT breach for this chamber (the reading
        # that tripped the alarm). Updated automatically as new breaches arrive,
        # because we keep the anomaly with the latest timestamp per sensor.
        latest_breach = {}
        for a in ch_anoms:
            k = a["sensor"]
            if k not in latest_breach or _secs(a["timestamp"]) >= _secs(latest_breach[k]["timestamp"]):
                latest_breach[k] = a
        trended_sensors = {t["sensor"] for t in ch_trends}

        sensors = []
        for key, label, unit, rng in SENSORS:
            if key in latest_breach:
                # show the value that tripped the alarm (most recent breach)
                sensors.append({"label": label, "value": _fmt(latest_breach[key]["value"], 1),
                                "unit": unit, "status": "fault"})
            elif last is not None and key in last:
                val = float(last[key])
                # Use the ACTIVE thresholds (user-set via onboarding) so a live
                # value's colour matches what detection would flag, not the
                # hard-coded SENSORS range.
                srng = thresholds.get(key, rng)
                sstat = "fault" if _check_anomaly(val, srng) else "watch" if key in trended_sensors else "ok"
                sensors.append({"label": label, "value": _fmt(val, 1), "unit": unit, "status": sstat})
            else:
                sensors.append({"label": label, "value": "—", "unit": unit, "status": "ok"})

        modules = [{"ok": "green", "watch": "yellow", "fault": "red"}[s["status"]] for s in sensors]
        # If the only breached sensor isn't one of the 4 displayed (e.g. cl2_flow),
        # still flag a module red so a faulted chamber reads as faulted.
        if status == "fault" and "red" not in modules:
            anom_sensors = {a["sensor"] for a in ch_anoms}
            idx = next((i for i, (k, *_ ) in enumerate(SENSORS) if k in anom_sensors), 0)
            modules[idx] = "red"

        findings = []
        for a in ch_anoms[:4]:
            findings.append({"type": "anomaly", "pc": LABELS.get(a["sensor"], a["sensor"]),
                             "text": a["explanation"] or f"Value {a['value']:.2f} outside [{a['threshold_min']}–{a['threshold_max']}] — fault declared",
                             "id": a["_id"], "time": a["timestamp"]})
        for t in ch_trends[:4]:
            arrow = "rising" if t["trend_direction"] == "increasing" else "falling"
            findings.append({"type": "trend", "pc": LABELS.get(t["sensor"], t["sensor"]),
                             "text": t["explanation"] or f"{LABELS.get(t['sensor'], t['sensor'])} {arrow} — breach in {t['time_to_breach']} (R²={t['r_squared']})",
                             "id": t["_id"], "time": t["timestamp"]})

        chambers[ch] = {
            "name": ch, "status": status, "monitored": True,
            "machine": CHAMBER_MACHINE[ch], "bay": CHAMBER_BAY[ch],
            "chamber": ch,
            "lot": str(meta["lot_id"]) if meta is not None else "—",
            "recipe": str(meta["recipe_name"]) if meta is not None else "—",
            "wafers": f"{total - fails}/{total}" if total else "—",
            "oee": f"{float(meta['oee']) * 100:.0f}%" if meta is not None else "—",
            "rfHrs": f"{float(meta['rf_generator_hrs']):.0f}h" if meta is not None else "—",
            "workOrder": str(meta["open_work_order"]) if meta is not None else "NONE",
            "sensors": sensors, "modules": modules, "findings": findings,
        }

    # ── tools for Live View ──────────────────────────────────────────────────
    tools = [
        {"id": "CHA", "status": chambers["CHA"]["status"], "monitored": True,
         "machine": CHAMBER_MACHINE["CHA"], "bay": "BAY 2"},
        SUPPORT_TOOLS[0],
        {"id": "CHB", "status": chambers["CHB"]["status"], "monitored": True,
         "machine": CHAMBER_MACHINE["CHB"], "bay": "BAY 3"},
        SUPPORT_TOOLS[1], SUPPORT_TOOLS[2], SUPPORT_TOOLS[3],
    ]

    # ── critical banner: most severe anomaly ─────────────────────────────────
    critical = None
    if anomaly_log:
        def _overshoot(a):
            rng = {"min": a["threshold_min"], "max": a["threshold_max"]}
            return max(a["value"] - rng["max"], rng["min"] - a["value"], 0) / (rng["max"] - rng["min"])
        worst = max(anomaly_log, key=_overshoot)
        critical = {
            "id": worst["_id"], "chamber": worst["chamber"],
            "pc": LABELS.get(worst["sensor"], worst["sensor"]),
            "text": f"{LABELS.get(worst['sensor'], worst['sensor'])} {_dev_string(worst['value'], {'min': worst['threshold_min'], 'max': worst['threshold_max']})}",
            "time": worst["timestamp"],
        }

    counts = {
        "fault": sum(1 for c in chambers.values() if c["status"] == "fault"),
        "watch": sum(1 for c in chambers.values() if c["status"] == "watch"),
        "ok":    sum(1 for c in chambers.values() if c["status"] == "nominal"),
    }

    return {
        "generated_at": datetime.now().strftime("%H:%M:%S"),
        "counts": counts, "critical": critical,
        "chambers": chambers, "errors": errors, "trends": trends, "tools": tools,
        # Live-pipeline progress (run_live_pipeline.py overrides these per stage).
        # None on the offline/simulate/production paths → frontend shows idle.
        "current_stage": None, "active_event_id": None,
    }


def write_dashboard_json(payload, path=None, verbose=True):
    path = Path(path) if path else (_ROOT / "front_end" / "dashboard_data.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: fully write a temp file, then rename it into place. A reader
    # (the web server) therefore never sees a half-written JSON.
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    if verbose:
        print(f"[dashboard_export] wrote {path}  "
              f"(anomalies feed: {len(payload['errors'])}, trends: {len(payload['trends'])})")
    return str(path)


def main():
    machine_df = pd.read_csv(_ROOT / "data" / "train_machine.csv")
    machine_df.columns = [c.strip().lower().replace(" ", "_") for c in machine_df.columns]
    summary_df = pd.read_csv(_ROOT / "data" / "train_summary.csv")
    # Single production day only — otherwise two calendar days collapse onto one
    # 24h clock and timestamps look fabricated (see ISSUES.md #1).
    summary_df, day = select_production_day(summary_df)
    machine_df = filter_machine_to_day(machine_df, summary_df)
    thresholds = resolve_thresholds()   # user-set onboarding limits, or defaults
    print(f"[dashboard_export] production day: {day}  ({len(summary_df)} wafers)")
    print(f"[dashboard_export] thresholds: {thresholds}")
    print(f"[dashboard_export] scanning {len(machine_df)} rows offline (no LLM)…")
    anomaly_log, trend_log = detect_offline(machine_df, thresholds, TREND_CONFIG,
                                            wafer_start=_wafer_start(summary_df))
    print(f"[dashboard_export] anomalies={len(anomaly_log)} trends={len(trend_log)}")
    payload = build_payload(anomaly_log, trend_log, machine_df, summary_df, thresholds)
    write_dashboard_json(payload)


if __name__ == "__main__":
    main()
