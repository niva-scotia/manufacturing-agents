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

TREND_CONFIG = {
    "min_steps": 8, "min_range_fraction": 0.15, "min_r_squared": 0.85,
    "min_seconds_to_breach": 5, "max_seconds_to_breach": 86400,
}

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


def detect_offline(machine_df, thresholds, cfg):
    """Pure-Python detection (no LLM). Returns (anomaly_log, trend_log) with the
    same record shape production_agent.py produces."""
    anomaly_log, trend_log = [], []
    history, trend_warned = {}, set()
    ts = datetime.now().strftime("%H:%M:%S")
    for _, row in machine_df.iterrows():
        wid, step = int(row["wafer_id"]), int(row["step"])
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
def _wafer_chamber(summary_df):
    return summary_df.set_index("wafer_id")["chamber_id"].to_dict()


def _fmt(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _dev_string(value, rng):
    if value > rng["max"]:
        return f"+{value - rng['max']:.2f} above max"
    return f"-{rng['min'] - value:.2f} below min"


def build_payload(anomaly_log, trend_log, machine_df, summary_df, thresholds=None):
    thresholds = thresholds or THRESHOLDS
    machine_df = machine_df.copy()
    machine_df.columns = [c.strip().lower().replace(" ", "_") for c in machine_df.columns]
    w2c = _wafer_chamber(summary_df)

    # tag every log entry with its chamber
    for a in anomaly_log: a["chamber"] = w2c.get(a["wafer_id"], "CHA")
    for t in trend_log:   t["chamber"] = w2c.get(t["wafer_id"], "CHA")

    chambers, errors, trends, tools = {}, [], [], []
    anm_n, trd_n = 4400, 1100

    # ── incoming-error feed (anomalies first, then trends), newest-ish first ──
    for a in anomaly_log:
        anm_n += 1
        a["_id"] = f"ANM-{anm_n}"
        errors.append({
            "chamber": a["chamber"], "pc": LABELS.get(a["sensor"], a["sensor"]), "type": "anomaly",
            "text": f"{LABELS.get(a['sensor'], a['sensor'])} out of range — {_dev_string(a['value'], {'min': a['threshold_min'], 'max': a['threshold_max']})}",
            "id": a["_id"], "time": a["timestamp"],
        })
    for t in trend_log:
        trd_n += 1
        t["_id"] = f"TRD-{trd_n}"

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

        # latest summary row for meta
        meta = srows.sort_values("wafer_id").iloc[-1] if len(srows) else None
        total = len(srows)
        fails = int((srows["pass_fail"] == "FAIL").sum()) if total else 0

        # latest sensor snapshot for this chamber's wafers
        ch_wafers = [w for w, c in w2c.items() if c == ch]
        snap = machine_df[machine_df["wafer_id"].isin(ch_wafers)]
        last = snap.sort_values(["wafer_id", "step"]).iloc[-1] if len(snap) else None

        trended_sensors = {t["sensor"] for t in ch_trends}
        anom_sensors    = {a["sensor"] for a in ch_anoms}
        sensors = []
        for key, label, unit, rng in SENSORS:
            if last is not None and key in last:
                val = float(last[key])
                # tile colour reflects the CURRENT reading (out of range = fault),
                # or an active trend on that sensor (watch).
                sstat = "fault" if _check_anomaly(val, rng) else "watch" if key in trended_sensors else "ok"
                sensors.append({"label": label, "value": _fmt(val, 1), "unit": unit, "status": sstat})
            else:
                sensors.append({"label": label, "value": "—", "unit": unit, "status": "ok"})

        modules = [{"ok": "green", "watch": "yellow", "fault": "red"}[s["status"]] for s in sensors]
        # A faulted chamber should read as faulted on the diagram even if the
        # latest snapshot is back in range — flag the module of a sensor that
        # actually faulted during the scan.
        if status == "fault" and "red" not in modules:
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
    }


def write_dashboard_json(payload, path=None):
    path = Path(path) if path else (_ROOT / "front_end" / "dashboard_data.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[dashboard_export] wrote {path}  "
          f"(anomalies feed: {len(payload['errors'])}, trends: {len(payload['trends'])})")
    return str(path)


def main():
    machine_df = pd.read_csv(_ROOT / "data" / "train_machine.csv")
    machine_df.columns = [c.strip().lower().replace(" ", "_") for c in machine_df.columns]
    summary_df = pd.read_csv(_ROOT / "data" / "train_summary.csv")
    print(f"[dashboard_export] scanning {len(machine_df)} rows offline (no LLM)…")
    anomaly_log, trend_log = detect_offline(machine_df, THRESHOLDS, TREND_CONFIG)
    print(f"[dashboard_export] anomalies={len(anomaly_log)} trends={len(trend_log)}")
    payload = build_payload(anomaly_log, trend_log, machine_df, summary_df, THRESHOLDS)
    write_dashboard_json(payload)


if __name__ == "__main__":
    main()
