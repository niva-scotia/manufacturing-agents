# # Anomaly Detection Agent
# ### Plasma Etch Chamber — Real-Time Sensor Monitoring
# 
# Uses Python for precise threshold checking and trend detection.
# Uses GPT-4o-mini only for reasoning and plain language explanation.
# 
# Two capabilities:
# - **Anomaly detection** — flags any sensor currently outside its threshold
# - **Trend detection** — flags sensors on a sustained trajectory toward a breach
# 

# ## Cell 1 — Install dependencies

import sys, subprocess
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent
if str(_Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(_Path(__file__).parent))

# install all required packages including python-dotenv
subprocess.run([sys.executable, "-m", "pip", "install", 
                "openai", "pandas", "numpy", "python-dotenv", "--quiet"], 
               check=True)
print("Dependencies ready.")

import os
import json
import numpy as np
import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from quality_agent import build_index, run_quality_agent
from maintenance_agent import (
    build_index as build_wo_index,
    run_maintenance_agent,
    print_recommendation,
)
from sop_agent import (
    build_index as build_sop_index,
    run_sop_agent,
)
from sustainability_agent import run_impact_agent, load_roster
from supervisor_agent import run_supervisor_agent, print_supervisor_summary
from openai import OpenAI

print("Imports successful.")

# load .env from the project root (manufacturing-agents/)
env_path = _ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# verify .env was found
print(f"Looking for .env at : {env_path}")
print(f".env file exists    : {env_path.exists()}")

AZURE_API_KEY     = os.environ.get("AZURE_API_KEY")
AZURE_ENDPOINT    = os.environ.get("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = os.environ.get("AZURE_DEPLOYMENT")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")

print(f"AZURE_API_KEY found : {AZURE_API_KEY is not None}")
print(f"AZURE_ENDPOINT found: {AZURE_ENDPOINT is not None}")

# stop immediately if any variable is missing
missing = [name for name, val in {
    "AZURE_API_KEY"    : AZURE_API_KEY,
    "AZURE_ENDPOINT"   : AZURE_ENDPOINT,
    "AZURE_DEPLOYMENT" : AZURE_DEPLOYMENT,
    "AZURE_API_VERSION": AZURE_API_VERSION,
}.items() if not val]

if missing:
    raise ValueError(
        f"Missing environment variables: {missing}\n"
        f"Check your .env file at: {env_path}"
    )

client = AzureOpenAI(
    api_key        = AZURE_API_KEY.strip(),
    azure_endpoint = AZURE_ENDPOINT.strip(),
    api_version    = AZURE_API_VERSION.strip(),
)

print(f"Azure OpenAI client ready.")

# ── Quality Agent initialisation ──────────────────────────────────────────────
# Build the vector index once at startup — stays in memory for the full run.
# The quality_agent.py file must be in the same directory as this script.
QUALITY_RECORDS_PATH = str(_ROOT / "data/quality_records.csv")
quality_store  = build_index(QUALITY_RECORDS_PATH)
quality_client = AzureOpenAI(api_key=os.getenv("AZURE_API_KEY"), azure_endpoint=os.getenv("AZURE_ENDPOINT"), api_version=os.getenv("AZURE_API_VERSION"))
print("Quality Agent vector store ready.")

# ── Maintenance Agent initialisation ──────────────────────────────────────────
# Build the work-order vector index once at startup, and load the three CMMS
# lookup tables (PM schedule, spare parts, calibration) into memory.
# The maintenance_agent.py file must be in the same directory as this script.
CMMS_WO_PATH    = str(_ROOT / "data/cmms_work_orders.csv")
CMMS_PM_PATH    = str(_ROOT / "data/cmms_pm_schedule.csv")
CMMS_PARTS_PATH = str(_ROOT / "data/cmms_spare_parts.csv")
CMMS_CALIB_PATH = str(_ROOT / "data/cmms_calibration.csv")

wo_store           = build_wo_index(CMMS_WO_PATH)
pm_df              = pd.read_csv(CMMS_PM_PATH)
parts_df           = pd.read_csv(CMMS_PARTS_PATH)
calib_df           = pd.read_csv(CMMS_CALIB_PATH)
maintenance_client = AzureOpenAI(api_key=os.getenv("AZURE_API_KEY"), azure_endpoint=os.getenv("AZURE_ENDPOINT"), api_version=os.getenv("AZURE_API_VERSION"))
print("Maintenance Agent vector store ready.")

SOP_KB_PATHS = {
    "sop":       str(_ROOT / "data/sop_procedures.csv"),
    "guides":    str(_ROOT / "data/troubleshooting_guides.csv"),
    "incidents": str(_ROOT / "data/incident_resolutions.csv"),
    "manuals":   str(_ROOT / "data/equipment_manuals.csv"),
}
sop_store  = build_sop_index(SOP_KB_PATHS)
sop_client = AzureOpenAI(
    api_key        = os.getenv("AZURE_API_KEY"),
    azure_endpoint = os.getenv("AZURE_ENDPOINT"),
    api_version    = os.getenv("AZURE_API_VERSION"),
)
print("SOP Agent vector store ready.")

# ── Impact Agent initialisation ───────────────────────────────────────────────
# Loads the wafer/lot roster once — used to count how many wafers remain in the
# current lot on the faulty chamber. The Impact Agent runs on ANOMALIES only (an
# active fault already producing out-of-spec wafers), not on pre-fault trends.
impact_roster = load_roster()
print("Impact Agent roster ready.")

# ── Supervisor Agent initialisation ───────────────────────────────────────────
# Final agent in the chain — synthesises every upstream output (Production,
# Quality, Maintenance, SOP, Impact/Sustainability) into role-specific
# recommendations for the operator, shift supervisor, quality engineer, and
# maintenance lead. No critical action is auto-executed.
supervisor_client = AzureOpenAI(
    api_key        = os.getenv("AZURE_API_KEY"),
    azure_endpoint = os.getenv("AZURE_ENDPOINT"),
    api_version    = os.getenv("AZURE_API_VERSION"),
)
print("Supervisor Agent ready.")

# ## Cell 4 — Load and prepare data

DATA_PATH = str(_ROOT / "data/train_machine.csv")

data = pd.read_csv(DATA_PATH)
def to_snake_case(name):
    return name.strip().lower().replace(' ', '_')

data.columns = [to_snake_case(c) for c in data.columns]

ID_COLS     = ["wafer_id", "experiment", "step", "time_sec", "step_number"]
sensor_cols = [c for c in data.columns if c not in ID_COLS]

print(f"Data loaded  : {data.shape[0]} rows, {data.shape[1]} columns")

# Build wafer → lot_id / chamber_id lookup from summary file.
# Used to enrich fault alerts before passing them to the Quality Agent.
_summary = pd.read_csv(str(_ROOT / "data/train_summary.csv"))
WAFER_LOOKUP = _summary.set_index("wafer_id")[["lot_id","chamber_id"]].to_dict("index")

# Build wafer → run_start lookup so each detected event is timestamped with
# WHEN THE WAFER ACTUALLY RAN (from train_summary.csv), not the scan's clock.
WAFER_START = _summary.set_index("wafer_id")["run_start"].to_dict()

def event_time(wafer_id):
    """Real event time (HH:MM:SS) from train_summary.run_start for this wafer.
    Falls back to the current clock only if the wafer has no run_start."""
    raw = WAFER_START.get(int(wafer_id))
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return datetime.now().strftime("%H:%M:%S")
    s = str(raw)                          # e.g. "2024-03-04T06:02:33"
    if "T" in s: return s.split("T", 1)[1][:8]
    if " " in s: return s.split(" ", 1)[1][:8]
    return s
print(f"Wafers       : {data['wafer_id'].nunique()}")
print(f"Sensors      : {len(sensor_cols)}")
print(f"\nAvailable sensor names:")
for s in sensor_cols:
    print(f"  {s}")

# ## Cell 5 — Set thresholds and trend configuration
# Define the acceptable range for each sensor you want to monitor.
# Also configure the trend detection parameters.
# 

# ── User-defined thresholds ───────────────────────────────────────────────────
# These are the DEFAULTS. The pre-shift onboarding screen
# (front_end/onboarding.html) lets the operator set these ranges for the shift
# and writes them to shift_config.json at the repo root; _load_shift_config()
# below overrides the matching defaults so detection reflects what the user set.
USER_THRESHOLDS = {
    "tcp_top_pwr" : {"min": 334,  "max": 360},
    "bcl3_flow"   : {"min": 740,  "max": 765},
    "cl2_flow"    : {"min": 748,  "max": 758},
    "pressure"    : {"min": 942,  "max": 1420},
    "rf_btm_pwr"  : {"min": 124,  "max": 142},
}

# ── Shift config override (from the onboarding screen) ────────────────────────
# If shift_config.json exists at the repo root, replace the default ranges above
# with the operator's chosen values and remember the role they're covering (for
# the Supervisor Agent's role-specific output). Absent/malformed file → silently
# keep the hard-coded defaults, so the pipeline still runs standalone.
SHIFT_CONFIG_PATH = _ROOT / "shift_config.json"
SHIFT_ROLE = None   # e.g. "operator" | "supervisor" | "quality_engineer" | "maintenance_lead"

def _load_shift_config(thresholds):
    """Merge shift_config.json onto the default thresholds. Returns the (possibly
    updated) thresholds dict and sets the module-level SHIFT_ROLE."""
    global SHIFT_ROLE
    if not SHIFT_CONFIG_PATH.exists():
        return thresholds
    try:
        cfg = json.loads(SHIFT_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[shift_config] ignored ({e}); using default thresholds.")
        return thresholds

    SHIFT_ROLE = cfg.get("role")
    user_thresholds = cfg.get("thresholds", {})
    applied = 0
    for sensor, rng in user_thresholds.items():
        if sensor not in thresholds:            # only the five known sensors
            continue
        try:
            lo, hi = float(rng["min"]), float(rng["max"])
        except (KeyError, TypeError, ValueError):
            continue
        if lo < hi:                             # ignore inverted/degenerate ranges
            thresholds[sensor] = {"min": lo, "max": hi}
            applied += 1

    print(f"[shift_config] applied {applied} sensor override(s) from "
          f"{SHIFT_CONFIG_PATH.name}"
          + (f"; role = {SHIFT_ROLE}" if SHIFT_ROLE else ""))
    return thresholds

USER_THRESHOLDS = _load_shift_config(USER_THRESHOLDS)

# ── Trend detection configuration ────────────────────────────────────────────
TREND_CONFIG = {
    # minimum number of consecutive steps to consider a trend genuine
    "min_steps"          : 8,

    # fraction of the threshold range that must be covered by the trend
    # e.g. 0.20 means the sensor must have moved at least 20% of its
    # allowed range during the trend window to be considered significant
    "min_range_fraction" : 0.15,

    # R-squared threshold — how linear/consistent the trend must be
    # 0.85 means 85% of variance must be explained by a straight line
    # higher = stricter, fewer false positives
    "min_r_squared"      : 0.85,

    "min_seconds_to_breach" : 5,      # ignore if this breach is within 5 seconds away

    "max_seconds_to_breach": 86400,    # ignore if breach is beyond 24 hours
}

print("Thresholds:")
for sensor, rng in USER_THRESHOLDS.items():
    print(f"  {sensor:<20} min={rng['min']}, max={rng['max']}")
print(f"\nTrend config:")
for k, v in TREND_CONFIG.items():
    print(f"  {k:<25} {v}")

# ## Cell 6 — Python detection engine
# All threshold and trend checks happen here in Python — not in the LLM.
# This guarantees precise, deterministic results with no hallucinations.
# The LLM is only called afterwards to explain what Python already confirmed.
# 

def check_anomaly(value, sensor, thresholds):
    """
    Exact threshold check in Python.
    Returns True only if the value is strictly outside the defined range.
    """
    rng = thresholds.get(sensor)
    if rng is None:
        return False
    return float(value) < rng["min"] or float(value) > rng["max"]

def format_time_to_breach(steps):
    """
    Convert steps to a human readable time string.
    Each step is approximately 1 second.
    Automatically picks the most meaningful unit.
    """
    total_seconds = int(steps)

    if total_seconds < 60:
        return f"{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        if seconds == 0:
            return f"{minutes} minutes"
        return f"{minutes} minutes {seconds} seconds"
    else:
        hours   = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if minutes == 0 and seconds == 0:
            return f"{hours} hours"
        elif seconds == 0:
            return f"{hours} hours {minutes} minutes"
        return f"{hours} hours {minutes} minutes {seconds} seconds"

def compute_trend(values, sensor, thresholds, config):
    """
    Determine whether a sequence of values shows a genuine sustained
    trend toward a threshold breach.

    Returns a dict with trend details if a real trend is found,
    or None if no significant trend exists.

    A trend is only flagged when ALL of the following are true:
      1. There are at least min_steps readings in the window
      2. The trend is consistent — R-squared >= min_r_squared
      3. The total movement covers at least min_range_fraction
         of the allowed threshold range
      4. The projected breach is within max_minutes_to_breach
      5. The current value is still within range (not already an anomaly)
    """
    rng = thresholds.get(sensor)
    if rng is None or len(values) < config["min_steps"]:
        return None

    values = np.array(values, dtype=float)
    current = values[-1]

    # must be within range — if already breached use flag_anomaly instead
    if current < rng["min"] or current > rng["max"]:
        return None

    # fit a linear trend
    x         = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)

    # compute R-squared — measures how consistently linear the trend is
    y_pred    = slope * x + intercept
    ss_res    = np.sum((values - y_pred) ** 2)
    ss_tot    = np.sum((values - values.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # check linearity threshold
    if r_squared < config["min_r_squared"]:
        return None

    # check if slope is meaningful — must cover min_range_fraction
    # of the allowed range over the observation window
    allowed_range    = rng["max"] - rng["min"]
    total_movement   = abs(slope * config["min_steps"])
    range_fraction   = total_movement / allowed_range

    if range_fraction < config["min_range_fraction"]:
        return None

    # determine direction and distance to nearest boundary
    if slope > 0:
        direction       = "increasing"
        distance        = rng["max"] - current
        boundary        = rng["max"]
    else:
        direction       = "decreasing"
        distance        = current - rng["min"]
        boundary        = rng["min"]

    # slope is near zero — no meaningful trend
    if abs(slope) < 1e-9:
        return None

    steps_to_breach  = int(distance / abs(slope))
    time_to_breach   = format_time_to_breach(steps_to_breach)

    # ignore if breach is too soon — likely already an anomaly
    if steps_to_breach < config["min_seconds_to_breach"]:
        return None
    
    # ignore if breach is too far away to be actionable
    if steps_to_breach > config["max_seconds_to_breach"]:
        return None

    return {
        "direction"        : direction,
        "slope"            : round(float(slope), 6),
        "r_squared"        : round(float(r_squared), 4),
        "range_fraction"   : round(float(range_fraction), 4),
        "current_value"    : round(float(current), 3),
        "boundary"         : boundary,
        "threshold_min"    : rng["min"],
        "threshold_max"    : rng["max"],
        "steps_to_breach"  : steps_to_breach,
        "time_to_breach"   : time_to_breach,
        "window_size"      : len(values),
        "values_seen"      : values.tolist(),
    }


print("Detection engine ready.")

# ## Cell 7 — LLM explanation function
# The LLM is called only when Python has already confirmed a real anomaly
# or trend. It adds plain language explanation — nothing more.
# 

def get_llm_explanation(event_type, sensor, details):
    """
    Ask the LLM to explain a confirmed anomaly or trend in plain language.
    Python has already verified this is real — the LLM only explains it.
    """
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
            f"Trend consistency: R² = {details['r_squared']} "
            f"(1.0 = perfectly linear)\n"
            f"\n"
            f"In 1-2 sentences, explain what this trend means physically "
            f"and what might be causing the drift."
        )

    try:
        response = client.chat.completions.create(
            model     = AZURE_DEPLOYMENT,
            messages  = [
                {
                    "role"   : "system",
                    "content": (
                        "You are a plasma etch process engineer. "
                        "Give brief, technically accurate explanations "
                        "of sensor anomalies and trends. Be concise."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens = 120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(explanation unavailable: {e})"


print("LLM explanation function ready.")


def _severity(value, rng):
    """
    Derive severity from how far outside the threshold the value is,
    expressed as a fraction of the allowed range.
    """
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
    """
    Convert a raw anomaly or trend dict from the Production Agent into
    the standard alert format the Quality Agent expects.

    This is the handoff contract between Agent 1 and Agent 2.
    One call = one fault event.
    """
    wafer_id = raw["wafer_id"]
    sensor   = raw["sensor"]
    rng      = {"min": raw["threshold_min"], "max": raw["threshold_max"]}

    # Look up lot and chamber from the summary data
    meta     = WAFER_LOOKUP.get(wafer_id, {})
    lot_id   = meta.get("lot_id",   "UNKNOWN")
    chamber  = meta.get("chamber_id","UNKNOWN")

    if alert_type == "ANOMALY":
        value    = raw["value"]
        sign     = "+" if value > rng["max"] else "-"
        delta    = abs(value - (rng["max"] if value > rng["max"] else rng["min"]))
        dev_str  = f"{sign}{delta:.2f} outside threshold"
        return {
            "alert_type"   : "ANOMALY",
            "wafer_id"     : wafer_id,
            "lot_id"       : lot_id,
            "chamber_id"   : chamber,
            "sensor"       : sensor,
            "value"        : value,
            "threshold_min": rng["min"],
            "threshold_max": rng["max"],
            "deviation"    : dev_str,
            "severity"     : _severity(value, rng),
            "step"         : raw["step"],
            "time_to_breach": None
            #"explanation"  : raw["explanation"],
        }
    else:  # TREND
        return {
            "alert_type"   : "TREND",
            "wafer_id"     : wafer_id,
            "lot_id"       : lot_id,
            "chamber_id"   : chamber,
            "sensor"       : sensor,
            "value"        : raw["current_value"],
            "threshold_min": rng["min"],
            "threshold_max": rng["max"],
            "deviation"    : (f"trending {raw['trend_direction']} toward "
                           f"{'upper' if raw['trend_direction'] == 'increasing' else 'lower'} limit"),
            "severity"     : "MEDIUM",   # trends are pre-fault by definition
            "step"         : raw["step"],
            "time_to_breach": raw["time_to_breach"]
            #"explanation"  : raw["explanation"],
        }


print("Monitoring function ready.")


# ## Cell 8 — Main monitoring function
# Scans every row. Python checks thresholds and trends precisely.
# LLM explains findings. No false positives from LLM reasoning.
# 

def run_agent(data, thresholds, trend_config, max_rows=None):
    """
    Scan sensor data for anomalies and trends.

    Python handles all detection logic precisely.
    LLM handles explanation only.

    Parameters
    ----------
    data         : pd.DataFrame
    thresholds   : dict — {sensor: {min, max}}
    trend_config : dict — trend detection parameters
    max_rows     : int or None

    Returns
    -------
    anomaly_log : list
    trend_log   : list
    """
    if max_rows:
        data = data.head(max_rows).copy()

    total_rows  = len(data)
    anomaly_log = []
    trend_log   = []

    # rolling history buffer per wafer per sensor
    # stores the last N readings for trend analysis
    history = {}   # {wafer_id: {sensor: [values]}}

    # track which (wafer, sensor) combinations already raised a trend
    # warning so we do not spam the same warning every step
    trend_warned = set()

    print("=" * 60)
    print("ANOMALY DETECTION AGENT  —  GPT-4o-mini (Azure)")
    print("=" * 60)
    print(f"  Rows to scan     : {total_rows}")
    print(f"  Wafers           : {data['wafer_id'].nunique()}")
    print(f"  Sensors monitored: {list(thresholds.keys())}")
    print(f"  Trend window     : {trend_config['min_steps']} steps minimum")
    print(f"  Trend R² minimum : {trend_config['min_r_squared']}")
    print(f"  Start time       : {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print()

    for idx, row in data.iterrows():
        wafer_id = int(row["wafer_id"])
        step     = int(row["step"])

        # initialise history buffer for new wafer
        if wafer_id not in history:
            history[wafer_id] = {s: [] for s in thresholds}
            # clear trend warnings for previous wafer
            trend_warned = {k for k in trend_warned
                            if k[0] == wafer_id}

        for sensor, rng in thresholds.items():
            if sensor not in data.columns:
                continue

            value = float(row[sensor])

            # ── 1. ANOMALY CHECK (Python — exact) ────────────────────────────
            if check_anomaly(value, sensor, thresholds):
                details = {
                    "value"        : value,
                    "threshold_min": rng["min"],
                    "threshold_max": rng["max"],
                }
                explanation = get_llm_explanation("anomaly", sensor, details)

                anomaly = {
                    "timestamp"     : event_time(wafer_id),
                    "wafer_id"      : wafer_id,
                    "step"          : step,
                    "sensor"        : sensor,
                    "value"         : value,
                    "threshold_min" : rng["min"],
                    "threshold_max" : rng["max"],
                    "explanation"   : explanation,
                }
                anomaly_log.append(anomaly)

                print(f"\n{'🚨' * 3}  ANOMALY — VALUE OUT OF RANGE  {'🚨' * 3}")
                print(f"  Time      : {anomaly['timestamp']}")
                print(f"  Wafer     : {wafer_id}")
                print(f"  Step      : {step}")
                print(f"  Sensor    : {sensor}")
                print(f"  Value     : {value}")
                print(f"  Range     : {rng['min']} — {rng['max']}")
                print(f"  Note      : {explanation}")
                print(f"{'─' * 52}")

                # ── Hand off to Quality Agent ─────────────────────────────
                # Normalise this single anomaly into the standard alert format
                # and pass it to the Quality Agent immediately.
                # The Quality Agent processes one fault at a time.
                alert = normalize_alert(anomaly, "ANOMALY")
                quality_report = run_quality_agent(
                    alert, quality_store, quality_client
                )
                print("\n[Quality Agent Report]\n")
                print(quality_report)
                print(f"{'─' * 52}")

                # ── Hand off to Maintenance Agent ─────────────────────────
                # The Maintenance Agent receives the accumulated outputs of both
                # prior agents: the Production alert (Agent 1) AND the Quality
                # report (Agent 2). It adds CMMS context (PM, parts, calibration)
                # and a prioritised recommendation.
                recommendation = run_maintenance_agent(
                    alert, quality_report, wo_store, pm_df, parts_df,
                    calib_df, maintenance_client
                )
                print_recommendation(recommendation)

                # ── Hand off to SOP Agent ─────────────────────────────
                sop_report = run_sop_agent(
                    alert, quality_report, recommendation,
                    sop_store, sop_client
                )
                print("\n[SOP Agent Report]\n")
                print(sop_report)
                print(f"{'─' * 52}")

                # ── Hand off to Impact Agent (Agent 5) ────────────────
                # Active anomaly → estimate the scrap cost if this faulty chamber
                # keeps running the REST OF THE CURRENT LOT uncorrected.
                impact = run_impact_agent(
                    alert, quality_report, recommendation, sop_report,
                    roster=impact_roster
                )
                print("\n[Impact Agent Report]\n")
                print(impact["narrative"])
                if impact.get("estimable"):
                    ec = impact["expected_cost"]
                    print(f"  Remaining {impact['chamber_id']} wafers in lot: "
                          f"{impact['remaining_wafers']}")
                    print(f"  Expected scrap cost: ${ec['low']:,.0f}–${ec['high']:,.0f} "
                          f"(mid ${ec['value']:,.0f})")
                print(f"{'─' * 52}")

                # ── Hand off to Supervisor Agent (final agent) ─────────────
                # Synthesises every upstream output into role-specific
                # recommendations. Reuses the Impact Agent estimate already
                # computed above instead of recomputing it.
                sustainability_report = {
                    "scrap_cost_estimate": (
                        {k: impact["expected_cost"][k] for k in ("value", "low", "high")}
                        if impact.get("estimable") else None
                    ),
                    "material_waste_estimate": impact.get("remaining_wafers"),
                    "notes": impact["narrative"],
                }
                supervisor_summary = run_supervisor_agent(
                    alert, quality_report, recommendation, sop_report,
                    sustainability_report=sustainability_report,
                    client=supervisor_client,
                )
                print_supervisor_summary(supervisor_summary)

            else:
                # value is within range — update history for trend analysis
                history[wafer_id][sensor].append(value)

                # keep only the last N * 2 values to avoid unbounded growth
                max_history = trend_config["min_steps"] * 3
                if len(history[wafer_id][sensor]) > max_history:
                    history[wafer_id][sensor] = (
                        history[wafer_id][sensor][-max_history:]
                    )

                # ── 2. TREND CHECK (Python — exact) ──────────────────────────
                # only check if we have enough history
                # and have not already warned for this wafer+sensor
                warn_key = (wafer_id, sensor)
                if warn_key not in trend_warned:
                    trend = compute_trend(
                        history[wafer_id][sensor],
                        sensor,
                        thresholds,
                        trend_config
                    )

                    if trend is not None:
                        explanation = get_llm_explanation(
                            "trend", sensor, trend
                        )

                        trend_entry = {
                            "timestamp"        : event_time(wafer_id),
                            "wafer_id"         : wafer_id,
                            "step"             : step,
                            "sensor"           : sensor,
                            "current_value"    : trend["current_value"],
                            "threshold_min"    : trend["threshold_min"],
                            "threshold_max"    : trend["threshold_max"],
                            "trend_direction"  : trend["direction"],
                            "rate_per_step"    : trend["slope"],
                            "r_squared"        : trend["r_squared"],
                            "steps_to_breach"  : trend["steps_to_breach"],
                            "time_to_breach"   : trend["time_to_breach"],   # ← updated
                            "values_seen"      : trend["values_seen"],      # for dashboard sparkline
                            "explanation"      : explanation,
                        }
                        trend_log.append(trend_entry)

                        # mark as warned so we do not repeat for same wafer+sensor
                        trend_warned.add(warn_key)

                        arrow = "↑" if trend["direction"] == "increasing" else "↓"

                        print(f"\n{'⚠️ ' * 3}  TREND WARNING  {'⚠️ ' * 3}")
                        print(f"  Time          : {trend_entry['timestamp']}")
                        print(f"  Wafer         : {wafer_id}")
                        print(f"  Step          : {step}")
                        print(f"  Sensor        : {sensor}  {arrow}")
                        print(f"  Current value : {trend['current_value']}")
                        print(f"  Range         : {rng['min']} — {rng['max']}")
                        print(f"  Rate          : {trend['slope']:+.4f} per step")
                        print(f"  R² (linearity): {trend['r_squared']}")
                        print(
                            f"\n  ⏱  If the {sensor} continues on this trend, "
                            f"in {trend['time_to_breach']} "
                            f"it will produce an error."
                        )
                        print(f"  Note          : {explanation}")
                        print(f"{'─' * 52}")

                        # ── Hand off to Quality Agent ─────────────────────
                        # Normalise this single trend alert into the standard
                        # format and pass it to the Quality Agent immediately.
                        # The Quality Agent processes one fault at a time.
                        alert = normalize_alert(trend_entry, "TREND")
                        quality_report = run_quality_agent(
                            alert, quality_store, quality_client
                        )
                        print("\n[Quality Agent Report]\n")
                        print(quality_report)
                        print(f"{'─' * 52}")

                        # ── Hand off to Maintenance Agent ─────────────────
                        # The Maintenance Agent receives the accumulated outputs
                        # of both prior agents: the Production alert (Agent 1)
                        # AND the Quality report (Agent 2). It adds CMMS context
                        # (PM, parts, calibration) and a prioritised recommendation.
                        recommendation = run_maintenance_agent(
                            alert, quality_report, wo_store, pm_df, parts_df,
                            calib_df, maintenance_client
                        )
                        print_recommendation(recommendation)

                        # ── Hand off to SOP Agent ─────────────────────
                        sop_report = run_sop_agent(
                            alert, quality_report, recommendation,
                            sop_store, sop_client
                        )
                        print("\n[SOP Agent Report]\n")
                        print(sop_report)
                        print(f"{'─' * 52}")

                        # ── Hand off to Supervisor Agent (final agent) ─────
                        # Trends are pre-fault, so the Impact Agent is not run
                        # here — the Supervisor Agent's sustainability step
                        # will report "not applicable" for TREND alerts.
                        supervisor_summary = run_supervisor_agent(
                            alert, quality_report, recommendation, sop_report,
                            client=supervisor_client,
                        )
                        print_supervisor_summary(supervisor_summary)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"  Rows scanned     : {total_rows}")
    print(f"  Anomalies found  : {len(anomaly_log)}")
    print(f"  Trend warnings   : {len(trend_log)}")

    if anomaly_log:
        print(f"\n  Anomaly log:")
        for a in anomaly_log:
            print(f"    [{a['timestamp']}]  "
                  f"Wafer {a['wafer_id']} | "
                  f"Step {a['step']:>3} | "
                  f"{a['sensor']:<20} = {a['value']:<10} "
                  f"(range: {a['threshold_min']}–{a['threshold_max']})")

    if trend_log:
        print(f"\n  Trend warning log:")
        for t in trend_log:
            arrow = "↑" if t["trend_direction"] == "increasing" else "↓"
            print(f"    [{t['timestamp']}]  "
                  f"Wafer {t['wafer_id']} | "
                  f"Step {t['step']:>3} | "
                  f"{t['sensor']:<20} {arrow} "
                  f"→ breach in {t['time_to_breach']}"
                  f"(R²={t['r_squared']})")

    print("=" * 60)
    return anomaly_log, trend_log


# ## Cell 9 — Run the agent
# Set `max_rows=None` to scan the full dataset.
# 

anomalies, trends = run_agent(
    data         = data,
    thresholds   = USER_THRESHOLDS,
    trend_config = TREND_CONFIG,
    max_rows     = None     # set to None for full dataset
)

# ## Cell 10 — Inspect results

print(f"Anomalies     : {len(anomalies)}")
print(f"Trend warnings: {len(trends)}")

if anomalies:
    df_anomalies = pd.DataFrame(anomalies)
    print("\nAnomalies:")
    print(df_anomalies[[
        'timestamp', 'wafer_id', 'step',
        'sensor', 'value', 'threshold_min', 'threshold_max'
    ]].to_string(index=False))

if trends:
    df_trends = pd.DataFrame(trends)
    print("\nTrend warnings:")
    print(df_trends[[
        'timestamp', 'wafer_id', 'step', 'sensor',
        'current_value', 'trend_direction',
        'rate_per_step', 'r_squared', 'time_to_breach'
    ]].to_string(index=False))

# ## Cell 11 — Export data for the OI Argus dashboard
# Maps this run's anomalies + trends (with LLM explanations) and the chamber
# summary into front_end/dashboard_data.json. The dashboard
# (front_end/oiargus.html) fetches this file and auto-refreshes,
# so the UI reflects the agent's real findings.
try:
    from dashboard_export import (
        build_payload, write_dashboard_json,
        select_production_day, filter_machine_to_day,
    )
    # Export a SINGLE production day so the dashboard timeline is coherent (the
    # CSV spans multiple dates; see ISSUES.md #1). Detection above still ran over
    # the full data — we only scope what the dashboard shows.
    _summary_day, _day = select_production_day(_summary)
    _day_wafers = set(_summary_day["wafer_id"])
    _data_day = data[data["wafer_id"].isin(_day_wafers)].copy()
    _anoms_day  = [a for a in anomalies if a["wafer_id"] in _day_wafers]
    _trends_day = [t for t in trends    if t["wafer_id"] in _day_wafers]
    print(f"[dashboard_export] exporting production day {_day}: "
          f"{len(_anoms_day)} anomalies, {len(_trends_day)} trends")
    _payload = build_payload(_anoms_day, _trends_day, _data_day, _summary_day, USER_THRESHOLDS)
    write_dashboard_json(_payload, str(_ROOT / "front_end" / "dashboard_data.json"))
except Exception as _e:
    print(f"[dashboard_export] skipped: {_e}")