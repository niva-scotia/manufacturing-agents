#!/usr/bin/env python3
"""
LIVE agentic replay — runs the REAL 6-agent chain per event as data "arrives".

Opt-in counterpart to simulate_realtime.py. Where the fast path pre-computes every
event with detect_offline() (no LLM) and replays them on a fake clock, this script
runs the genuine chain for each detected event:

    Production (LLM explanation) → Quality (RAG) → Maintenance (CMMS + RAG)
    → SOP (RAG) → Sustainability (deterministic, anomalies only) → Supervisor (LLM)

so LLM latency and RAG retrieval are real, and the dashboard's header pipeline
indicator lights up each real stage via the payload's `current_stage` field.

Design notes
------------
- Detection reuses dashboard_export.detect_offline() (import-safe, same engine,
  already day-aware). production_agent.py is NOT importable (it runs the whole
  batch at module import), so its get_llm_explanation / _severity / normalize_alert
  are replicated below as small deterministic helpers.
- Anomalies are grouped into ONE episode per (wafer_id, sensor) — one chain run =
  one dashboard tile — matching build_payload's "one tile per wafer excursion"
  aggregation. Each episode's PEAK reading is the chain's representative alert.
- Pacing reuses the SPEED-multiplied sim clock as a floor: an event is revealed
  once the sim clock reaches its timestamp, so a fast chain tops-up-sleeps to the
  next timestamp and a slow chain (real LLM latency) simply dominates.

Env knobs
---------
    LIVE_LIMIT   max wafer-excursion events to run: 5|10|50|100|all (default 10;
                 all/0 → no cap). Controls demo RUNTIME, not cost.
    SPEED        sim-seconds per real second (default 1; raise it, e.g. 60, so the
                 pacing floor is small and the demo flows at chain speed).
    TICK         max seconds between pacing sleeps        (default 1.0)
    START        "HH:MM:SS" simulated start               (default: first event − 13s)
    LOOP         "1" to restart the day when it ends      (default off)

Usage (via the server, opt-in):
    LIVE_AGENTS=1 LIVE_LIMIT=5 SPEED=60 python serve_dashboard.py
Or standalone:
    LIVE_LIMIT=5 python agents/run_live_pipeline.py
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

from dotenv import load_dotenv
from openai import AzureOpenAI

from dashboard_export import (          # noqa: E402
    detect_offline, build_payload, write_dashboard_json,
    THRESHOLDS, TREND_CONFIG, _wafer_start, _event_time,
    select_production_day, filter_machine_to_day,
    resolve_thresholds, shift_role,
)
# The five downstream agent modules ARE import-safe (no pipeline runs at import).
from quality_agent import build_index, run_quality_agent
from maintenance_agent import build_index as build_wo_index, run_maintenance_agent
from sop_agent import build_index as build_sop_index, run_sop_agent
from sustainability_agent import run_impact_agent, load_roster
from supervisor_agent import run_supervisor_agent


# ── module globals wired up in main() (mirrors production_agent.py) ───────────
_EXPL_CLIENT     = None    # AzureOpenAI client for the Production explanation
AZURE_DEPLOYMENT = None
WAFER_LOOKUP     = {}       # wafer_id → {lot_id, chamber_id}, for normalize_alert


# ════════════════════════════════════════════════════════════════════════════
#  Time helpers (copied from simulate_realtime.py — kept local to avoid coupling)
# ════════════════════════════════════════════════════════════════════════════
def _tod(hms):
    """'HH:MM:SS' → seconds since midnight."""
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def _fmt(sec):
    """seconds since midnight → 'HH:MM:SS'."""
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# ════════════════════════════════════════════════════════════════════════════
#  Replicated deterministic helpers (production_agent.py isn't import-safe)
# ════════════════════════════════════════════════════════════════════════════
def get_llm_explanation(event_type, sensor, details):
    """Ask the LLM to explain a confirmed anomaly/trend in plain language. Python
    has already verified it is real — the LLM only narrates. Mirrors the function
    in production_agent.py."""
    if event_type == "anomaly":
        prompt = (
            f"A sensor anomaly has been detected in a plasma etch chamber.\n"
            f"Sensor        : {sensor}\n"
            f"Current value : {details['value']}\n"
            f"Allowed range : {details['threshold_min']} to {details['threshold_max']}\n"
            f"Direction     : {'above maximum' if details['value'] > details['threshold_max'] else 'below minimum'}\n"
            f"\n"
            f"In 1-2 sentences, explain what this sensor measures physically "
            f"and why this out-of-range value is concerning for the etch process."
        )
    else:  # trend
        prompt = (
            f"A sustained sensor trend has been detected in a plasma etch chamber.\n"
            f"Sensor           : {sensor}\n"
            f"Trend direction  : {details['direction']}\n"
            f"Rate of change   : {details['slope']:+.4f} per step\n"
            f"Current value    : {details['current_value']}\n"
            f"Allowed range    : {details['threshold_min']} to {details['threshold_max']}\n"
            f"Projected breach : boundary {details['boundary']} in "
            f"{details['time_to_breach']}\n"
            f"Trend consistency: R² = {details['r_squared']} (1.0 = perfectly linear)\n"
            f"\n"
            f"In 1-2 sentences, explain what this trend means physically "
            f"and what might be causing the drift."
        )
    try:
        response = _EXPL_CLIENT.chat.completions.create(
            model    = AZURE_DEPLOYMENT,
            messages = [
                {"role": "system", "content": (
                    "You are a plasma etch process engineer. Give brief, "
                    "technically accurate explanations of sensor anomalies and "
                    "trends. Be concise.")},
                {"role": "user", "content": prompt},
            ],
            max_tokens = 120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(explanation unavailable: {e})"


def _severity(value, rng):
    """Severity from how far outside the band the value is, as a fraction of the
    allowed range. Mirrors production_agent.py."""
    allowed = rng["max"] - rng["min"]
    if allowed == 0:
        return "HIGH"
    overshoot = max(value - rng["max"], rng["min"] - value, 0)
    frac = overshoot / allowed
    if frac >= 0.20:  return "CRITICAL"
    if frac >= 0.10:  return "HIGH"
    if frac >= 0.04:  return "MEDIUM"
    return "LOW"


def normalize_alert(raw, alert_type):
    """Convert a raw anomaly/trend dict into the standard alert the chain expects.
    Mirrors production_agent.py; uses the module-level WAFER_LOOKUP."""
    wafer_id = raw["wafer_id"]
    sensor   = raw["sensor"]
    rng      = {"min": raw["threshold_min"], "max": raw["threshold_max"]}
    meta     = WAFER_LOOKUP.get(wafer_id, {})
    lot_id   = meta.get("lot_id", "UNKNOWN")
    chamber  = meta.get("chamber_id", "UNKNOWN")

    if alert_type == "ANOMALY":
        value = raw["value"]
        sign  = "+" if value > rng["max"] else "-"
        delta = abs(value - (rng["max"] if value > rng["max"] else rng["min"]))
        return {
            "alert_type": "ANOMALY", "wafer_id": wafer_id, "lot_id": lot_id,
            "chamber_id": chamber, "sensor": sensor, "value": value,
            "threshold_min": rng["min"], "threshold_max": rng["max"],
            "deviation": f"{sign}{delta:.2f} outside threshold",
            "severity": _severity(value, rng), "step": raw["step"],
            "time_to_breach": None,
        }
    else:  # TREND
        return {
            "alert_type": "TREND", "wafer_id": wafer_id, "lot_id": lot_id,
            "chamber_id": chamber, "sensor": sensor, "value": raw["current_value"],
            "threshold_min": rng["min"], "threshold_max": rng["max"],
            "deviation": (f"trending {raw['trend_direction']} toward "
                          f"{'upper' if raw['trend_direction'] == 'increasing' else 'lower'} limit"),
            "severity": "MEDIUM", "step": raw["step"],
            "time_to_breach": raw["time_to_breach"],
        }


# ════════════════════════════════════════════════════════════════════════════
#  Setup helpers
# ════════════════════════════════════════════════════════════════════════════
def _parse_limit(raw):
    """LIVE_LIMIT → int cap or None (no cap). unset→10; all/0→None; +int→that."""
    if raw is None or raw.strip() == "":
        return 10
    v = raw.strip().lower()
    if v in ("all", "0"):
        return None
    try:
        n = int(v)
        if n > 0:
            return n
    except ValueError:
        pass
    print(f"[live] LIVE_LIMIT={raw!r} invalid — defaulting to 10")
    return 10


def _overshoot(value, rng):
    return max(value - rng["max"], rng["min"] - value, 0)


def _make_client():
    return AzureOpenAI(
        api_key        = os.getenv("AZURE_API_KEY"),
        azure_endpoint = os.getenv("AZURE_ENDPOINT"),
        api_version    = os.getenv("AZURE_API_VERSION"),
    )


def _build_events(anoms, trends):
    """Group anomalies into ONE episode per (wafer, sensor); trends are already one
    per (wafer, sensor). Returns a combined list sorted by time-of-day, each item:
      {type, tod, peak (representative raw row), members (raw rows to reveal)}."""
    groups = {}
    for a in anoms:
        groups.setdefault((a["wafer_id"], a["sensor"]), []).append(a)

    events = []
    for (_wid, _sensor), members in groups.items():
        rng  = {"min": members[0]["threshold_min"], "max": members[0]["threshold_max"]}
        peak = max(members, key=lambda m: _overshoot(m["value"], rng))
        tod  = min(_tod(m["timestamp"]) for m in members)   # members share run_start
        events.append({"type": "anomaly", "tod": tod, "peak": peak, "members": members})

    for t in trends:
        events.append({"type": "trend", "tod": _tod(t["timestamp"]),
                       "peak": t, "members": [t]})

    events.sort(key=lambda e: e["tod"])
    return events


# ════════════════════════════════════════════════════════════════════════════
#  Shape the agent outputs for the dashboard modals
# ════════════════════════════════════════════════════════════════════════════
# Display order + labels for the Supervisor's per-role guidance. Kept here so a
# future "show only the covering role" filter is a one-liner over this list.
_ROLE_ORDER = [
    ("operator",         "Operator"),
    ("supervisor",       "Shift Supervisor"),
    ("quality_engineer", "Quality Engineer"),
    ("maintenance_lead", "Maintenance Lead"),
]

# Minimum real-time a stage stays on the board so the frontend poll (~1s) can
# catch it. Only matters for the deterministic (instant) Sustainability stage;
# the LLM/RAG stages already take longer than this on their own.
STAGE_MIN_DWELL_S = 1.5


def _sustainability_box(impact):
    """Sustainability Agent output → the chamber's 'Cost of Continued Run' box.
    Real numbers + the agent's own narrative; no placeholders."""
    if not impact or not impact.get("estimable"):
        reason = (impact or {}).get("reason", "not applicable")
        return {"estimable": False, "level": "nominal", "badge": "NOT ESTIMABLE",
                "figure": "—", "per": "", "desc": f"Scrap exposure not estimable ({reason})."}
    ec = impact["expected_cost"]
    return {
        "estimable": True, "level": "critical", "badge": "SCRAP EXPOSURE",
        "figure": f"${ec['value']:,.0f}", "per": "rest-of-lot",
        "desc": impact.get("narrative", ""),
        # raw numbers carried through for the frontend subtitle / future use
        "midpoint": ec["value"], "low": ec["low"], "high": ec["high"],
        "remaining_wafers": impact.get("remaining_wafers"),
        "per_wafer": (impact.get("per_wafer_cost") or {}).get("value"),
        "lot": impact.get("lot_id"), "chamber": impact.get("chamber_id"),
    }


def _supervisor_playbook(sup):
    """Supervisor Agent output → the Resolution Playbook modal. For now we surface
    EVERY role block the Supervisor emits; a later change can filter to just the
    operator's covering role (see _ROLE_ORDER)."""
    if not sup:
        return None
    roles = []
    for key, label in _ROLE_ORDER:
        blk = sup.get(key) or {}
        summary = (blk.get("summary") or "").strip()
        actions = [str(a).strip() for a in (blk.get("actions") or []) if str(a).strip()]
        if not summary and not actions:
            continue
        roles.append({"key": key, "label": label, "summary": summary, "actions": actions})
    return {
        "priority":   sup.get("priority", "N/A"),
        "sensor":     sup.get("sensor", ""),
        "chamber":    sup.get("chamber_id", ""),
        "alert_type": sup.get("alert_type", ""),
        "roles":      roles,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    global _EXPL_CLIENT, AZURE_DEPLOYMENT, WAFER_LOOKUP

    speed = float(os.environ.get("SPEED", "1"))
    tick  = float(os.environ.get("TICK", "1.0"))
    loop  = os.environ.get("LOOP", "0") == "1"
    limit = _parse_limit(os.environ.get("LIVE_LIMIT"))
    print(f"[live] LIVE_LIMIT={'all (no cap)' if limit is None else limit}")

    load_dotenv(dotenv_path=_ROOT / ".env")
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")

    # ── data (single production day, like simulate_realtime.py) ──
    machine_df = pd.read_csv(_ROOT / "data" / "train_machine.csv")
    machine_df.columns = [c.strip().lower().replace(" ", "_") for c in machine_df.columns]
    summary_df = pd.read_csv(_ROOT / "data" / "train_summary.csv")
    summary_df, day = select_production_day(summary_df)
    machine_df = filter_machine_to_day(machine_df, summary_df)
    WAFER_LOOKUP = summary_df.set_index("wafer_id")[["lot_id", "chamber_id"]].to_dict("index")
    wafer_start = _wafer_start(summary_df)
    # wafer → seconds-of-day of its run_start, so the board reveals EVERY wafer as
    # it "arrives" by time (not just anomaly wafers). This is what lets a nominal
    # chamber still show its current live sensor readings. (Mirrors simulate.)
    wafer_tod = {w: _tod(_event_time(rs)) for w, rs in wafer_start.items() if _event_time(rs)}
    print(f"[live] production day: {day}  ({len(summary_df)} wafers, {len(machine_df)} rows)")

    # Thresholds + covering role from the pre-shift onboarding screen. The USER'S
    # limits label anomalies/trends; the role tailors which supervisor guidance
    # the dashboard surfaces.
    thresholds = resolve_thresholds()
    SHIFT_ROLE = shift_role()
    print(f"[live] thresholds: {thresholds}")
    print(f"[live] covering role: {SHIFT_ROLE or '(none — showing all roles)'}")

    # ── agent stores + clients (mirrors production_agent.py init) ──
    print("[live] building indexes + clients (Quality / Maintenance / SOP)…")
    quality_store      = build_index(str(_ROOT / "data/quality_records.csv"))
    quality_client     = _make_client()
    wo_store           = build_wo_index(str(_ROOT / "data/cmms_work_orders.csv"))
    pm_df              = pd.read_csv(_ROOT / "data/cmms_pm_schedule.csv")
    parts_df           = pd.read_csv(_ROOT / "data/cmms_spare_parts.csv")
    calib_df           = pd.read_csv(_ROOT / "data/cmms_calibration.csv")
    maintenance_client = _make_client()
    sop_store          = build_sop_index({
        "sop":       str(_ROOT / "data/sop_procedures.csv"),
        "guides":    str(_ROOT / "data/troubleshooting_guides.csv"),
        "incidents": str(_ROOT / "data/incident_resolutions.csv"),
        "manuals":   str(_ROOT / "data/equipment_manuals.csv"),
    })
    sop_client        = _make_client()
    impact_roster     = load_roster()
    supervisor_client = _make_client()
    _EXPL_CLIENT      = _make_client()
    print("[live] agents ready.")

    # ── blank the board immediately (starts empty, then fills in). Show the
    #    Production/detection agent as already monitoring so the pipeline reads
    #    as LIVE from the first paint, not dead until the first anomaly. ──
    blank = build_payload([], [], machine_df.iloc[0:0], summary_df.iloc[0:0], thresholds)
    blank["generated_at"] = blank["sim_clock"] = "--:--:--"
    blank["current_stage"] = "production"
    write_dashboard_json(blank, verbose=False)

    # ── detection (cheap, no LLM) → episodes → cap ──
    print("[live] scanning for events (no LLM)…")
    anoms, trends = detect_offline(machine_df, thresholds, TREND_CONFIG, wafer_start=wafer_start)
    events = _build_events(anoms, trends)
    total_detected = len(events)
    if limit is not None:
        events = events[:limit]
    if not events:
        print("[live] no events detected — nothing to run.")
        return

    ai, ti = 4401, 1101
    for e in events:
        if e["type"] == "anomaly":
            e["_id"], ai = f"ANM-{ai}", ai + 1
        else:
            e["_id"], ti = f"TRD-{ti}", ti + 1

    first = min(e["tod"] for e in events)
    last  = max(e["tod"] for e in events)
    start = _tod(os.environ["START"]) if os.environ.get("START") else first - 13
    print(f"[live] {len(events)} event(s) to run (of {total_detected} detected) | "
          f"window {_fmt(start)}–{_fmt(last)} | SPEED={speed}x")
    print("[live] open the dashboard and watch each real agent stage light up.")

    STAGE_ORDER = ["production", "quality", "maintenance", "sop", "sustainability", "supervisor"]

    while True:
        revealed_a, revealed_t = [], []
        # Latest Sustainability / Supervisor output per chamber, so the chamber
        # modal shows real agent data for its most recent anomaly.
        chamber_cost, chamber_playbook = {}, {}
        wall0 = time.time()

        def write_payload(stage, active_id):
            """Write the current board state with the running stage highlighted."""
            sim = start + (time.time() - wall0) * speed
            # Reveal EVERY wafer that has run by the sim clock (not just anomaly
            # wafers) so both chambers show live sensor readings even when nominal.
            seen = {w for w, td in wafer_tod.items() if td <= sim}
            mdf = machine_df[machine_df["wafer_id"].isin(seen)]
            sdf = summary_df[summary_df["wafer_id"].isin(seen)]
            payload = build_payload(list(reversed(revealed_a)),
                                    list(reversed(revealed_t)), mdf, sdf, thresholds)
            payload["generated_at"] = payload["sim_clock"] = _fmt(sim)
            payload["current_stage"] = stage
            payload["active_event_id"] = active_id
            payload["shift_role"] = SHIFT_ROLE   # covering role → playbook filter
            # Attach the real agent outputs onto their chamber (build_payload
            # rebuilds chambers each call, so re-inject every write).
            for ch_id, box in chamber_cost.items():
                if ch_id in payload["chambers"]:
                    payload["chambers"][ch_id]["sustainability"] = box
            for ch_id, pb in chamber_playbook.items():
                if ch_id in payload["chambers"]:
                    payload["chambers"][ch_id]["playbook"] = pb
            write_dashboard_json(payload, verbose=False)

        for idx, e in enumerate(events, 1):
            # ── pacing floor: reveal once the sim clock reaches this event. While
            #    we wait, the Production/detection agent is monitoring the stream,
            #    so keep the board alive (clock ticking, PRODUCTION lit) with a
            #    heartbeat write each tick instead of freezing between events. ──
            while True:
                sim = start + (time.time() - wall0) * speed
                if sim >= e["tod"]:
                    break
                write_payload("production", None)
                time.sleep(min(tick, max(0.0, (e["tod"] - sim) / speed)))

            peak    = e["peak"]
            is_anom = e["type"] == "anomaly"
            sensor  = peak["sensor"]
            eid     = e["_id"]

            # ── Stage 1: Production (real LLM explanation) ──
            write_payload("production", eid)
            if is_anom:
                details = {"value": peak["value"],
                           "threshold_min": peak["threshold_min"],
                           "threshold_max": peak["threshold_max"]}
                expl = get_llm_explanation("anomaly", sensor, details)
            else:
                details = {"direction": peak["trend_direction"],
                           "slope": peak["rate_per_step"],
                           "current_value": peak["current_value"],
                           "threshold_min": peak["threshold_min"],
                           "threshold_max": peak["threshold_max"],
                           "boundary": (peak["threshold_max"] if peak["trend_direction"] == "increasing"
                                        else peak["threshold_min"]),
                           "time_to_breach": peak["time_to_breach"],
                           "r_squared": peak["r_squared"]}
                expl = get_llm_explanation("trend", sensor, details)
            for m in e["members"]:
                m["explanation"] = expl

            # Commit detection + explanation early so the tile & chamber appear now,
            # and the downstream stages light up for this active event.
            (revealed_a if is_anom else revealed_t).extend(e["members"])
            alert = normalize_alert(peak, "ANOMALY" if is_anom else "TREND")

            # ── Stage 2: Quality (RAG) ──
            write_payload("quality", eid)
            quality_report = run_quality_agent(alert, quality_store, quality_client)

            # ── Stage 3: Maintenance (CMMS + RAG) ──
            write_payload("maintenance", eid)
            recommendation = run_maintenance_agent(
                alert, quality_report, wo_store, pm_df, parts_df, calib_df, maintenance_client)

            # ── Stage 4: SOP (RAG) ──
            write_payload("sop", eid)
            sop_report = run_sop_agent(alert, quality_report, recommendation, sop_store, sop_client)

            chamber = alert["chamber_id"]

            # ── Stage 5: Sustainability (deterministic; anomalies only) ──
            sustainability_report = None
            if is_anom:
                write_payload("sustainability", eid)
                impact = run_impact_agent(alert, quality_report, recommendation, sop_report,
                                          roster=impact_roster)
                sustainability_report = {
                    "scrap_cost_estimate": (
                        {k: impact["expected_cost"][k] for k in ("value", "low", "high")}
                        if impact.get("estimable") else None),
                    "material_waste_estimate": impact.get("remaining_wafers"),
                    "notes": impact.get("narrative"),
                }
                # Real Sustainability output → the chamber's Cost of Continued Run box.
                chamber_cost[chamber] = _sustainability_box(impact)
                # This agent is deterministic (no LLM) so it returns instantly —
                # without a brief dwell the frontend's ~1s poll never catches the
                # "sustainability" stage and the header appears to skip SOP→Supervisor.
                time.sleep(STAGE_MIN_DWELL_S)

            # ── Stage 6: Supervisor (real LLM synthesis) ──
            write_payload("supervisor", eid)
            supervisor_summary = run_supervisor_agent(
                alert, quality_report, recommendation, sop_report,
                sustainability_report=sustainability_report,
                client=supervisor_client)
            # Real Supervisor output → the chamber's Resolution Playbook modal.
            chamber_playbook[chamber] = _supervisor_playbook(supervisor_summary)

            # ── chain done for this event → back to Production monitoring ──
            write_payload("production", None)
            print(f"[live] {idx}/{len(events)}  {eid}  {alert['chamber_id']} · {sensor} "
                  f"· wafer {peak['wafer_id']}  ✓", flush=True)

        print(f"\n[live] finished {len(events)} event(s). {'restarting…' if loop else 'done.'}")
        if not loop:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[live] stopped.")
