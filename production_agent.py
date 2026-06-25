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

print("Imports successful.")

# load .env from the same folder as this script
env_path = Path(__file__).parent / ".env"
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

# ## Cell 4 — Load and prepare data

DATA_PATH = "data/train_machine.csv"

data = pd.read_csv(DATA_PATH)

def to_snake_case(name):
    return name.strip().lower().replace(' ', '_')

data.columns = [to_snake_case(c) for c in data.columns]

ID_COLS     = ["wafer_id", "experiment", "step", "time_sec", "step_number"]
sensor_cols = [c for c in data.columns if c not in ID_COLS]

print(f"Data loaded  : {data.shape[0]} rows, {data.shape[1]} columns")
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
USER_THRESHOLDS = {
    "tcp_top_pwr" : {"min": 334,  "max": 360},
    "bcl3_flow"   : {"min": 740,  "max": 765},
    "cl2_flow"    : {"min": 748,  "max": 758},
    "pressure"    : {"min": 942,  "max": 1420},
    "rf_btm_pwr"  : {"min": 124,  "max": 142},
}

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
                    "timestamp"     : datetime.now().strftime("%H:%M:%S"),
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
                            "timestamp"        : datetime.now().strftime("%H:%M:%S"),
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


print("Monitoring function ready.")

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

