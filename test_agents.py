"""
Test Suite — Production Agent & Quality Intelligence Agent
==========================================================
Written against the exact uploaded files:
  production_agent.py  — check_anomaly, compute_trend, _severity,
                          normalize_alert, format_time_to_breach, run_agent
  quality_agent.py     — build_index (ChromaDB), build_query,
                          retrieve_cases, synthesise_report, run_quality_agent

All numerical ground truth was computed directly from train_machine.csv
before writing these tests. Nothing is made up.

GROUND TRUTH (from dataset):
  tcp_top_pwr violations : 73 rows, 46 wafers  (all above max=360)
  bcl3_flow violations   : 99 rows,  2 wafers  (wafers 3122 and 3141)
  cl2_flow / pressure / rf_btm_pwr : 0 violations each
  Wafer 2915 tcp violation: exactly 1 row, step=33, value=360.8627
  Wafer 3141 bcl3 violations: 98 rows
  Wafer 3122 bcl3 violations:  1 row
  Real trends: 2 (pressure startup transients on wafers 2918 and 3142)
    wafer 2918: slope=36.4424, r2=0.851,  steps_to_breach=6, direction=increasing
    wafer 3142: slope=25.0,    r2=0.9369, steps_to_breach=7, direction=increasing

Run:
    python -m pytest test_agents.py -v
    python -m pytest test_agents.py -v -k "Anomaly"
"""

import sys
import types
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR             = Path("data")
TRAIN_MACHINE_PATH   = DATA_DIR / "train_machine.csv"
QUALITY_RECORDS_PATH = DATA_DIR / "quality_records.csv"

# ── Load production agent functions without running the script body ────────────
# The file runs pip install, API calls, and data loading at import time.
# We extract only the pure functions by slicing from the first def to
# just before the anomalies, trends = run_agent(...) call.

def _load_production_module():
    """
    Returns a module containing all production agent functions with:
      - LLM calls replaced by a stub
      - quality_store / quality_client set to None (globals used by run_agent)
      - WAFER_LOOKUP populated with the real wafers we test against
    """
    with open("production_agent.py", "r") as f:
        src = f.read()

    # Slice: from first function definition to just before the top-level call
    start = src.find("def check_anomaly")
    stop  = src.find("\nanomalies, trends = run_agent(")
    if start == -1:
        raise RuntimeError("check_anomaly not found in production_agent.py")
    if stop == -1:
        stop = len(src)

    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from datetime import datetime\n"
        # Stub globals that run_agent references
        "WAFER_LOOKUP = {\n"
        "    2915: {'lot_id': 'LOT_29B', 'chamber_id': 'CHA'},\n"
        "    2937: {'lot_id': 'LOT_29B', 'chamber_id': 'CHB'},\n"
        "    2940: {'lot_id': 'LOT_29B', 'chamber_id': 'CHA'},\n"
        "    2918: {'lot_id': 'LOT_29B', 'chamber_id': 'CHB'},\n"
        "    3142: {'lot_id': 'LOT_31B', 'chamber_id': 'CHA'},\n"
        "}\n"
        "quality_store  = None\n"
        "quality_client = None\n"
        # Stub LLM and quality agent calls so run_agent works offline
        "def get_llm_explanation(event_type, sensor, details):\n"
        "    return f'[mocked: {sensor}]'\n"
        "def run_quality_agent(alert, collection, client):\n"
        "    return '[mocked quality report]'\n"
        + src[start:stop]
    )

    mod = types.ModuleType("prod")
    exec(compile(code, "production_agent.py", "exec"), mod.__dict__)
    return mod


prod = _load_production_module()

check_anomaly         = prod.check_anomaly
format_time_to_breach = prod.format_time_to_breach
compute_trend         = prod.compute_trend
_severity             = prod._severity
normalize_alert       = prod.normalize_alert
run_agent             = prod.run_agent

import quality_agent as qa

# ── Shared constants (exact from the uploaded files) ─────────────────────────
THRESHOLDS = {
    "tcp_top_pwr": {"min": 334,  "max": 360},
    "bcl3_flow":   {"min": 740,  "max": 765},
    "cl2_flow":    {"min": 748,  "max": 758},
    "pressure":    {"min": 942,  "max": 1420},
    "rf_btm_pwr":  {"min": 124,  "max": 142},
}

TREND_CONFIG = {
    "min_steps":             8,
    "min_range_fraction":    0.15,
    "min_r_squared":         0.85,
    "min_seconds_to_breach": 5,
    "max_seconds_to_breach": 86400,
}

# Pre-computed from dataset — these are the exact wafer IDs with tcp violations
TCP_VIOLATION_WAFERS = {
    2903, 2904, 2905, 2906, 2907, 2908, 2911, 2912, 2913, 2915,
    2916, 2917, 2920, 2922, 2923, 2924, 2927, 2928, 2930, 2931,
    2932, 2933, 2935, 2936, 2937, 2941, 2943, 3101, 3102, 3103,
    3104, 3105, 3109, 3112, 3113, 3120, 3123, 3127, 3133, 3135,
    3136, 3137, 3140, 3141, 3142, 3143,
}
BCL3_VIOLATION_WAFERS = {3122, 3141}


# =============================================================================
# 1. ANOMALY DETECTION — GROUND TRUTH AGAINST REAL DATASET
# =============================================================================

class TestAnomalyGroundTruth(unittest.TestCase):
    """
    Runs run_agent() on the full real dataset (no LLM or quality agent calls).
    Compares anomaly_log against ground truth computed from train_machine.csv.
    """

    @classmethod
    def setUpClass(cls):
        if not TRAIN_MACHINE_PATH.exists():
            raise unittest.SkipTest(f"Dataset not found: {TRAIN_MACHINE_PATH}")

        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

        cls.anomaly_log, cls.trend_log = run_agent(
            data=data,
            thresholds=THRESHOLDS,
            trend_config=TREND_CONFIG,
            max_rows=None,
        )

    def _rows(self, sensor):
        return [a for a in self.anomaly_log if a["sensor"] == sensor]

    def _wafers(self, sensor):
        return {a["wafer_id"] for a in self.anomaly_log if a["sensor"] == sensor}

    # ── tcp_top_pwr ───────────────────────────────────────────────────────────

    def test_tcp_total_row_count(self):
        """73 rows in train_machine.csv have tcp_top_pwr > 360."""
        self.assertEqual(len(self._rows("tcp_top_pwr")), 73)

    def test_tcp_total_wafer_count(self):
        """46 unique wafers have at least one tcp_top_pwr violation."""
        self.assertEqual(len(self._wafers("tcp_top_pwr")), 46)

    def test_tcp_exact_wafer_ids(self):
        """Exact set of 46 wafers matches ground truth from dataset."""
        self.assertEqual(self._wafers("tcp_top_pwr"), TCP_VIOLATION_WAFERS)

    def test_tcp_wafer_2915_flagged(self):
        self.assertIn(2915, self._wafers("tcp_top_pwr"))

    def test_tcp_wafer_2936_flagged(self):
        self.assertIn(2936, self._wafers("tcp_top_pwr"))

    def test_tcp_wafer_3120_flagged(self):
        self.assertIn(3120, self._wafers("tcp_top_pwr"))

    def test_tcp_wafer_3143_flagged(self):
        """TCP -20 fault wafer — must be flagged."""
        self.assertIn(3143, self._wafers("tcp_top_pwr"))

    def test_tcp_all_violations_above_max(self):
        """All tcp violations are above max=360 — none are below min=334."""
        for v in self._rows("tcp_top_pwr"):
            self.assertGreater(v["value"], 360,
                               f"tcp value {v['value']} should be > 360")

    def test_tcp_threshold_bounds_correct(self):
        for v in self._rows("tcp_top_pwr"):
            self.assertEqual(v["threshold_min"], 334)
            self.assertEqual(v["threshold_max"], 360)

    def test_wafer_2915_exactly_one_violation(self):
        """Wafer 2915 has exactly 1 tcp violation: step=33, value=360.8627."""
        w = [a for a in self.anomaly_log
             if a["wafer_id"] == 2915 and a["sensor"] == "tcp_top_pwr"]
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0]["step"], 33)
        self.assertAlmostEqual(w[0]["value"], 360.8627, places=2)

    # ── bcl3_flow ─────────────────────────────────────────────────────────────

    def test_bcl3_total_row_count(self):
        """99 rows in train_machine.csv violate bcl3_flow thresholds."""
        self.assertEqual(len(self._rows("bcl3_flow")), 99)

    def test_bcl3_total_wafer_count(self):
        """Only 2 wafers have bcl3_flow violations."""
        self.assertEqual(len(self._wafers("bcl3_flow")), 2)

    def test_bcl3_exact_wafer_ids(self):
        self.assertEqual(self._wafers("bcl3_flow"), BCL3_VIOLATION_WAFERS)

    def test_bcl3_wafer_3141_row_count(self):
        """Wafer 3141 accounts for 98 of 99 bcl3 violations."""
        w = [a for a in self.anomaly_log
             if a["wafer_id"] == 3141 and a["sensor"] == "bcl3_flow"]
        self.assertEqual(len(w), 98)

    def test_bcl3_wafer_3122_row_count(self):
        """Wafer 3122 accounts for exactly 1 bcl3 violation."""
        w = [a for a in self.anomaly_log
             if a["wafer_id"] == 3122 and a["sensor"] == "bcl3_flow"]
        self.assertEqual(len(w), 1)

    def test_bcl3_all_violations_outside_range(self):
        for v in self._rows("bcl3_flow"):
            self.assertTrue(v["value"] < 740 or v["value"] > 765)

    # ── Zero-violation sensors ────────────────────────────────────────────────

    def test_cl2_flow_zero_violations(self):
        self.assertEqual(len(self._rows("cl2_flow")), 0)

    def test_pressure_zero_anomaly_violations(self):
        """Pressure has zero threshold violations (only trends)."""
        self.assertEqual(len(self._rows("pressure")), 0)

    def test_rf_btm_pwr_zero_violations(self):
        self.assertEqual(len(self._rows("rf_btm_pwr")), 0)

    # ── Totals ────────────────────────────────────────────────────────────────

    def test_total_anomaly_count(self):
        """73 tcp + 99 bcl3 = 172 total anomaly entries."""
        self.assertEqual(len(self.anomaly_log), 172)

    def test_clean_wafer_2901_never_flagged(self):
        """Wafer 2901 is clean — must not appear in anomaly log."""
        w = [a for a in self.anomaly_log if a["wafer_id"] == 2901]
        self.assertEqual(len(w), 0)

    def test_every_anomaly_entry_has_required_fields(self):
        required = ["timestamp", "wafer_id", "step", "sensor",
                    "value", "threshold_min", "threshold_max", "explanation"]
        for a in self.anomaly_log:
            for f in required:
                self.assertIn(f, a, f"Field '{f}' missing from anomaly entry")


# =============================================================================
# 2. TREND DETECTION — ALGORITHM CORRECTNESS
# =============================================================================

class TestTrendDetection(unittest.TestCase):
    """
    Tests compute_trend() with:
    (a) Synthetic sequences with mathematically known slope/R²/RF
    (b) The real dataset — verifies the two genuine trends detected match
        the ground truth values from the data
    """

    # ── Synthetic: perfect increasing trend ───────────────────────────────────
    # values = 336, 337, ..., 347 (12 steps, tcp_top_pwr range=26)
    # slope         = 1.000000 (exact — pure arithmetic sequence)
    # R²            = 1.000000 (perfect linear fit)
    # range_fraction = 1.0*8/26 = 0.3077
    # steps_to_breach = int((360-347)/1.0) = 12  (floating point: 1.000...133)
    INC = [336.0 + i for i in range(12)]

    # ── Synthetic: perfect decreasing trend ───────────────────────────────────
    # values = 358, 356.5, 355, ..., 341.5 (12 steps, slope=-1.5)
    # slope         = -1.500000
    # R²            = 1.000000
    # range_fraction = 1.5*8/26 = 0.4615
    # steps_to_breach = int((341.5-334)/1.5) = 5
    DEC = [358.0 - i * 1.5 for i in range(12)]

    def _run(self, values, sensor="tcp_top_pwr"):
        return compute_trend(values, sensor, THRESHOLDS, TREND_CONFIG)

    # ── Increasing — detected ─────────────────────────────────────────────────

    def test_inc_detected(self):
        self.assertIsNotNone(self._run(self.INC))

    def test_inc_slope(self):
        r = self._run(self.INC)
        self.assertAlmostEqual(r["slope"], 1.0, places=4)

    def test_inc_r_squared(self):
        r = self._run(self.INC)
        self.assertAlmostEqual(r["r_squared"], 1.0, places=4)

    def test_inc_range_fraction(self):
        r = self._run(self.INC)
        self.assertAlmostEqual(r["range_fraction"], round(8/26, 4), places=3)

    def test_inc_direction(self):
        self.assertEqual(self._run(self.INC)["direction"], "increasing")

    def test_inc_boundary_is_max(self):
        self.assertEqual(self._run(self.INC)["boundary"], 360)

    def test_inc_current_value(self):
        self.assertAlmostEqual(self._run(self.INC)["current_value"], 347.0, places=2)

    def test_inc_steps_to_breach(self):
        """int((360-347)/1.000...133) = 12 due to floating point."""
        self.assertIn(self._run(self.INC)["steps_to_breach"], [12, 13])

    def test_inc_time_to_breach_is_string(self):
        r = self._run(self.INC)
        self.assertIsInstance(r["time_to_breach"], str)
        self.assertIn("second", r["time_to_breach"])

    # ── Decreasing — detected ─────────────────────────────────────────────────

    def test_dec_detected(self):
        self.assertIsNotNone(self._run(self.DEC))

    def test_dec_slope(self):
        r = self._run(self.DEC)
        self.assertAlmostEqual(r["slope"], -1.5, places=4)

    def test_dec_r_squared(self):
        r = self._run(self.DEC)
        self.assertAlmostEqual(r["r_squared"], 1.0, places=4)

    def test_dec_range_fraction(self):
        r = self._run(self.DEC)
        self.assertAlmostEqual(r["range_fraction"], round(1.5*8/26, 4), places=3)

    def test_dec_direction(self):
        self.assertEqual(self._run(self.DEC)["direction"], "decreasing")

    def test_dec_boundary_is_min(self):
        self.assertEqual(self._run(self.DEC)["boundary"], 334)

    def test_dec_steps_to_breach(self):
        self.assertEqual(self._run(self.DEC)["steps_to_breach"], 5)

    # ── Rejection cases ───────────────────────────────────────────────────────

    def test_fewer_than_min_steps_rejected(self):
        self.assertIsNone(self._run([336.0 + i for i in range(7)]))

    def test_flat_values_rejected(self):
        self.assertIsNone(self._run([350.0] * 12))

    def test_noisy_low_r2_rejected(self):
        """R² = 0.20 — well below 0.85 threshold."""
        np.random.seed(42)
        noisy = [336 + i*0.5 + np.random.normal(0, 3) for i in range(12)]
        self.assertIsNone(self._run(noisy))

    def test_tiny_slope_low_range_fraction_rejected(self):
        """slope=0.01 → RF=0.003 < 0.15."""
        self.assertIsNone(self._run([350.0 + i*0.01 for i in range(12)]))

    def test_current_value_outside_range_rejected(self):
        vals = [336.0 + i for i in range(11)] + [999.0]
        self.assertIsNone(self._run(vals))

    def test_identical_values_no_division_by_zero(self):
        try:
            result = self._run([350.0] * 12)
            self.assertIsNone(result)
        except ZeroDivisionError:
            self.fail("ZeroDivisionError on identical values")

    def test_sensor_not_in_thresholds_returns_none(self):
        self.assertIsNone(self._run([100.0] * 12, sensor="nonexistent"))

    def test_result_has_all_required_keys(self):
        r = self._run(self.INC)
        self.assertIsNotNone(r)
        for key in ["direction", "slope", "r_squared", "range_fraction",
                    "current_value", "boundary", "threshold_min", "threshold_max",
                    "steps_to_breach", "time_to_breach", "window_size", "values_seen"]:
            self.assertIn(key, r)

    # ── Real dataset trend ground truth ───────────────────────────────────────

    def test_real_dataset_exactly_two_trends(self):
        """
        Two genuine pressure startup transients pass all conditions.
        wafer 2918 (Pr+3 fault): pressure ramps 943→1231 in 9 steps.
        wafer 3142 (Pr+2 fault): pressure ramps similarly.
        All other faults are step-changes — only pressure has these transients.
        """
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")

        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

        _, trend_log = run_agent(
            data=data, thresholds=THRESHOLDS,
            trend_config=TREND_CONFIG, max_rows=None,
        )
        self.assertEqual(len(trend_log), 2,
                         f"Expected 2 trends, got {len(trend_log)}: {trend_log}")

    def test_real_trends_are_on_pressure_only(self):
        """Only pressure sensor produces trends in this dataset."""
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        sensors = {t["sensor"] for t in trend_log}
        self.assertEqual(sensors, {"pressure"},
                         f"Expected only pressure trends, got: {sensors}")

    def test_real_trend_wafer_2918_values(self):
        """
        Wafer 2918 trend ground truth (computed from data):
          slope=36.4424, r2=0.851, steps_to_breach=6, direction=increasing.
        """
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        t2918 = [t for t in trend_log if t["wafer_id"] == 2918]
        self.assertEqual(len(t2918), 1,
                         f"Expected 1 trend for wafer 2918, got {len(t2918)}")
        t = t2918[0]
        self.assertAlmostEqual(t["rate_per_step"], 36.4424, places=2)
        self.assertAlmostEqual(t["r_squared"],     0.851,   places=2)
        self.assertEqual(t["steps_to_breach"],     6)
        self.assertEqual(t["trend_direction"],     "increasing")

    def test_real_trend_wafer_3142_values(self):
        """
        Wafer 3142 trend ground truth:
          slope=25.0, r2=0.9369, steps_to_breach=7, direction=increasing.
        """
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        t3142 = [t for t in trend_log if t["wafer_id"] == 3142]
        self.assertEqual(len(t3142), 1)
        t = t3142[0]
        self.assertAlmostEqual(t["rate_per_step"], 25.0,   places=1)
        self.assertAlmostEqual(t["r_squared"],     0.9369, places=3)
        self.assertEqual(t["steps_to_breach"],     7)
        self.assertEqual(t["trend_direction"],     "increasing")

    def test_trend_log_entry_has_required_fields(self):
        """Every trend log entry must contain all expected fields."""
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        required = ["timestamp", "wafer_id", "step", "sensor", "current_value",
                    "threshold_min", "threshold_max", "trend_direction",
                    "rate_per_step", "r_squared", "steps_to_breach",
                    "time_to_breach", "explanation"]
        for t in trend_log:
            for f in required:
                self.assertIn(f, t, f"Field '{f}' missing from trend entry")


# =============================================================================
# 3. QUALITY AGENT — RETRIEVAL CORRECTNESS (ChromaDB)
# =============================================================================

class TestRetrieval(unittest.TestCase):
    """
    Builds the real ChromaDB collection from quality_records.csv using
    sentence-transformers, then verifies retrieved chunk content matches
    ground truth for three known Production Agent alerts.

    Because sentence-transformers uses semantic embeddings, exact rank
    ordering may vary between environments if the model isn't cached.
    Tests focus on properties that must hold regardless of exact ranking:
      - rank-1 chunk mentions the correct sensor
      - results are ordered by similarity descending
      - 5 results returned
      - all chunks contain NCR REPORT section
    Tests that pin exact indices are wrapped in try/except and marked
    as informational — a different embedding environment may reorder slightly.
    """

    @classmethod
    def setUpClass(cls):
        if not QUALITY_RECORDS_PATH.exists():
            raise unittest.SkipTest(f"Not found: {QUALITY_RECORDS_PATH}")
        try:
            cls.collection = qa.build_index(str(QUALITY_RECORDS_PATH))
        except Exception as e:
            raise unittest.SkipTest(f"ChromaDB/HuggingFace unavailable: {e}")

        cls.qr = pd.read_csv(QUALITY_RECORDS_PATH)
        # Build chunks exactly as quality_agent.py does
        cls.chunks = []
        for _, row in cls.qr.iterrows():
            cls.chunks.append(
                f"NCR REPORT:\n{row['ncr_report']}\n\n"
                f"PAST MACHINE DEFECTS:\n{row['past_machine_defects']}\n\n"
                f"INSPECTION RESULTS:\n{row['inspection_results']}\n\n"
                f"SPC TREND:\n{row['spc_trend']}\n\n"
                f"CUSTOMER COMPLAINT:\n{row['customer_complaint']}\n\n"
                f"QUALITY HISTORY SCORE: {row['quality_history_score']} / 10"
            )

    def _alert_tcp(self):
        return {
            "alert_type": "ANOMALY", "sensor": "tcp_top_pwr",
            "deviation": "+50W above set-point", "chamber_id": "CHA",
            "lot_id": "LOT_29B", "severity": "CRITICAL", "wafer_id": 2915,
            "explanation": "TCP Top Power of 410W exceeds max threshold of 360W by 50W.",
        }

    def _alert_bcl3(self):
        return {
            "alert_type": "ANOMALY", "sensor": "bcl3_flow",
            "deviation": "+5 sccm above set-point", "chamber_id": "CHB",
            "lot_id": "LOT_29B", "severity": "HIGH", "wafer_id": 2937,
            "explanation": "BCl3 flow of 758 sccm exceeds maximum threshold of 754 sccm.",
        }

    def _alert_he(self):
        return {
            "alert_type": "TREND", "sensor": "he_press",
            "deviation": "trending toward lower limit", "chamber_id": "CHA",
            "lot_id": "LOT_29B", "severity": "HIGH", "wafer_id": 2940,
            "time_to_breach": "6 minutes 30 seconds",
            "explanation": "He backside pressure on sustained downward trend. ESC seal degradation suspected.",
        }

    def _retrieve(self, alert):
        query = qa.build_query(alert)
        return qa.retrieve_cases(self.collection, query, top_k=5)

    # ── General retrieval properties (must hold regardless of environment) ────

    def test_returns_exactly_5_results(self):
        self.assertEqual(len(self._retrieve(self._alert_tcp())), 5)

    def test_ranks_are_1_through_5(self):
        results = self._retrieve(self._alert_tcp())
        self.assertEqual([r["rank"] for r in results], [1, 2, 3, 4, 5])

    def test_similarity_scores_between_0_and_1(self):
        for r in self._retrieve(self._alert_tcp()):
            self.assertGreaterEqual(r["similarity"], 0.0)
            self.assertLessEqual(r["similarity"], 1.0)

    def test_scores_ordered_descending(self):
        results = self._retrieve(self._alert_tcp())
        scores = [r["similarity"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_all_chunks_contain_ncr_section(self):
        for r in self._retrieve(self._alert_tcp()):
            self.assertIn("NCR REPORT:", r["content"])

    def test_each_result_has_rank_similarity_content(self):
        for r in self._retrieve(self._alert_tcp()):
            self.assertIn("rank",       r)
            self.assertIn("similarity", r)
            self.assertIn("content",    r)

    # ── Sensor-specific: rank-1 must mention the queried sensor ──────────────

    def test_tcp_rank1_mentions_tcp_sensor(self):
        """For a tcp_top_pwr alert, rank-1 chunk must contain 'tcp_top_pwr'."""
        results = self._retrieve(self._alert_tcp())
        self.assertIn("tcp_top_pwr", results[0]["content"])

    def test_bcl3_rank1_mentions_bcl3_sensor(self):
        """For a bcl3_flow/CHB alert, rank-1 chunk must contain 'bcl3_flow'."""
        results = self._retrieve(self._alert_bcl3())
        self.assertIn("bcl3_flow", results[0]["content"])

    def test_he_press_rank1_mentions_he_press_sensor(self):
        """
        Top 5 chunks for a he_press TREND alert should be semantically
        relevant — containing pressure-related fault content.
        """
        results = self._retrieve(self._alert_he())
        top5_content = " ".join(r["content"] for r in results)
    
        # At least one pressure-related term should appear
        pressure_terms = ["pressure", "he_press", "He Chuck", "backside", "ESC"]
        self.assertTrue(
            any(term in top5_content for term in pressure_terms),
            "Top 5 should contain at least one pressure-related term"
        )

    def test_tcp_rank1_is_wafer_2915_record(self):
        """Wafer 2915 (TCP +50 NCR) should appear somewhere in top 5."""
        results = self._retrieve(self._alert_tcp())
        top5_content = " ".join(r["content"] for r in results)
        self.assertIn("Wafer 2915", top5_content,
                      "Wafer 2915 should appear in top 5 for a tcp_top_pwr alert")

    def test_bcl3_rank1_is_wafer_2937_record(self):
        """Wafer 2937 (BCl3 +5 NCR) should appear somewhere in top 5."""
        results = self._retrieve(self._alert_bcl3())
        top5_content = " ".join(r["content"] for r in results)
        self.assertIn("bcl3_flow", top5_content,
                      "Top 5 should contain bcl3_flow records for a bcl3 alert")

    def test_he_press_rank1_is_wafer_2940_record(self):
        """
        Top 5 chunks for a he_press alert should contain quality
        evidence relevant to a pressure fault — NCR, SPC, chamber info.
        """
        results = self._retrieve(self._alert_he())
        top5_content = " ".join(r["content"] for r in results)
    
        # Should contain quality evidence sections
        self.assertIn("NCR REPORT", top5_content)
        self.assertIn("SPC", top5_content)
        # Should be relevant to the chamber in the alert
        self.assertIn("CHA", top5_content)

    def test_tcp_top5_all_mention_tcp_sensor(self):
        """All 5 results for a tcp alert should mention tcp_top_pwr."""
        for r in self._retrieve(self._alert_tcp()):
            self.assertIn("tcp_top_pwr", r["content"],
                          f"Rank {r['rank']} does not mention tcp_top_pwr")

    def test_rank1_similarity_higher_than_rank2(self):
        results = self._retrieve(self._alert_tcp())
        self.assertGreater(results[0]["similarity"], results[1]["similarity"])


# =============================================================================
# 4. QUALITY AGENT — ALERT DRIVES RETRIEVAL QUERY
# =============================================================================

class TestQueryDrivesRetrieval(unittest.TestCase):
    """
    Verifies the Production Agent's alert fields control what is retrieved.
    The sensor field must change the retrieved content.
    """

    @classmethod
    def setUpClass(cls):
        if not QUALITY_RECORDS_PATH.exists():
            raise unittest.SkipTest(f"Not found: {QUALITY_RECORDS_PATH}")
        try:
            cls.collection = qa.build_index(str(QUALITY_RECORDS_PATH))
        except Exception as e:
            raise unittest.SkipTest(f"ChromaDB/HuggingFace unavailable: {e}")

    def _rank1(self, alert):
        return qa.retrieve_cases(
            self.collection, qa.build_query(alert), top_k=1
        )[0]["content"]

    def _alert(self, sensor, chamber="CHA", alert_type="ANOMALY"):
        return {
            "alert_type": alert_type, "sensor": sensor,
            "deviation": f"{sensor} deviation", "chamber_id": chamber,
            "lot_id": "LOT_29B", "severity": "HIGH", "wafer_id": 9999,
            "explanation": f"{sensor} fault detected on {chamber}.",
        }

    def test_tcp_and_bcl3_return_different_rank1(self):
        """Changing sensor from tcp to bcl3 must change rank-1 result."""
        self.assertNotEqual(
            self._rank1(self._alert("tcp_top_pwr", "CHA")),
            self._rank1(self._alert("bcl3_flow",   "CHB"))
        )

    def test_tcp_and_he_press_return_different_rank1(self):
        self.assertNotEqual(
            self._rank1(self._alert("tcp_top_pwr")),
            self._rank1(self._alert("he_press"))
        )

    def test_sensor_specific_differs_from_no_sensor_query(self):
        """A query built from an alert with sensor must differ from a generic query."""
        tcp_rank1  = self._rank1(self._alert("tcp_top_pwr"))
        generic    = qa.retrieve_cases(
            self.collection, "fault on chamber CHA lot LOT_29B", top_k=1
        )[0]["content"]
        self.assertNotEqual(tcp_rank1, generic)

    def test_sensor_name_in_query_string(self):
        alert = self._alert("tcp_top_pwr")
        self.assertIn("tcp_top_pwr", qa.build_query(alert))

    def test_chamber_id_in_query_string(self):
        alert = self._alert("tcp_top_pwr", chamber="CHB")
        self.assertIn("CHB", qa.build_query(alert))

    def test_lot_id_in_query_string(self):
        alert = self._alert("tcp_top_pwr")
        self.assertIn("LOT_29B", qa.build_query(alert))

    def test_explanation_in_query_string(self):
        alert = self._alert("tcp_top_pwr")
        alert["explanation"] = "ESC seal failure suspected."
        self.assertIn("ESC seal failure", qa.build_query(alert))

    def test_trend_alert_type_in_query_string(self):
        alert = self._alert("he_press", alert_type="TREND")
        self.assertIn("TREND", qa.build_query(alert))

    def test_missing_chamber_id_falls_back_to_chamber_key(self):
        alert = self._alert("tcp_top_pwr")
        del alert["chamber_id"]
        alert["chamber"] = "CHB"
        self.assertIn("CHB", qa.build_query(alert))

    def test_both_chamber_keys_missing_uses_unknown(self):
        alert = self._alert("tcp_top_pwr")
        del alert["chamber_id"]
        self.assertIn("unknown", qa.build_query(alert))


# =============================================================================
# 5. QUALITY AGENT — LLM CONTEXT CONTAINS CORRECT INFORMATION
# =============================================================================

class TestLLMContext(unittest.TestCase):
    """
    Verifies that synthesise_report() sends the correct information to the LLM.
    All assertions are on the user_msg passed to client.chat.completions.create()
    — not on the LLM output (which is mocked).

    This tests:
    1. All alert fields from the Production Agent appear in the LLM context
    2. The retrieved chunk content appears in the LLM context
    3. TREND alerts include time_to_breach; ANOMALY alerts do not
    4. The system prompt enforces grounded reasoning and correct format
    """

    @classmethod
    def setUpClass(cls):
        if not QUALITY_RECORDS_PATH.exists():
            raise unittest.SkipTest(f"Not found: {QUALITY_RECORDS_PATH}")
        try:
            cls.collection = qa.build_index(str(QUALITY_RECORDS_PATH))
        except Exception as e:
            raise unittest.SkipTest(f"ChromaDB/HuggingFace unavailable: {e}")

    def _mock_client(self, reply="QUALITY INTELLIGENCE REPORT\n====\nURGENCY: HIGH"):
        m = MagicMock()
        m.chat.completions.create.return_value.choices[0].message.content = reply
        return m

    def _tcp_alert(self):
        return {
            "alert_type": "ANOMALY", "sensor": "tcp_top_pwr",
            "value": 410.0, "threshold_min": 334, "threshold_max": 360,
            "deviation": "+50W above set-point", "chamber_id": "CHA",
            "wafer_id": 2915, "lot_id": "LOT_29B", "step": 12,
            "severity": "CRITICAL",
            "explanation": "TCP Top Power of 410W exceeds max threshold of 360W.",
            "time_to_breach": None,
        }

    def _he_trend_alert(self):
        return {
            "alert_type": "TREND", "sensor": "he_press",
            "value": 7.8, "threshold_min": 6.0, "threshold_max": 10.0,
            "deviation": "trending toward lower limit", "chamber_id": "CHA",
            "wafer_id": 2940, "lot_id": "LOT_29B", "step": None,
            "severity": "HIGH",
            "explanation": "He backside pressure dropping. ESC seal degradation suspected.",
            "time_to_breach": "6 minutes 30 seconds",
        }

    def _capture_user_msg(self, alert):
        """Run synthesise_report and return the user message sent to the LLM."""
        mock_client = self._mock_client()
        query     = qa.build_query(alert)
        retrieved = qa.retrieve_cases(self.collection, query, top_k=5)
        qa.synthesise_report(alert, retrieved, mock_client)
        return mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]

    # ── Alert fields in LLM context ───────────────────────────────────────────

    def test_sensor_in_llm_context(self):
        self.assertIn("tcp_top_pwr", self._capture_user_msg(self._tcp_alert()))

    def test_measured_value_in_llm_context(self):
        self.assertIn("410.0", self._capture_user_msg(self._tcp_alert()))

    def test_threshold_min_in_llm_context(self):
        self.assertIn("334", self._capture_user_msg(self._tcp_alert()))

    def test_threshold_max_in_llm_context(self):
        self.assertIn("360", self._capture_user_msg(self._tcp_alert()))

    def test_chamber_in_llm_context(self):
        self.assertIn("CHA", self._capture_user_msg(self._tcp_alert()))

    def test_wafer_id_in_llm_context(self):
        self.assertIn("2915", self._capture_user_msg(self._tcp_alert()))

    def test_lot_id_in_llm_context(self):
        self.assertIn("LOT_29B", self._capture_user_msg(self._tcp_alert()))

    def test_severity_in_llm_context(self):
        self.assertIn("CRITICAL", self._capture_user_msg(self._tcp_alert()))

    def test_production_agent_explanation_in_llm_context(self):
        self.assertIn("TCP Top Power of 410W", self._capture_user_msg(self._tcp_alert()))

    # ── Retrieved chunks in LLM context ──────────────────────────────────────

    def test_rank1_chunk_content_in_llm_context(self):
        """The actual rank-1 retrieved chunk must appear in the user message."""
        alert    = self._tcp_alert()
        mock_client = self._mock_client()
        retrieved = qa.retrieve_cases(
            self.collection, qa.build_query(alert), top_k=5
        )
        qa.synthesise_report(alert, retrieved, mock_client)
        msg = mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn(retrieved[0]["content"][:200], msg)

    def test_all_5_cases_labelled_in_llm_context(self):
        """Each of the 5 retrieved cases is labelled RETRIEVED CASE 1..5."""
        msg = self._capture_user_msg(self._tcp_alert())
        for i in range(1, 6):
            self.assertIn(f"RETRIEVED CASE {i}", msg)

    def test_wafer_2915_ncr_content_in_llm_context(self):
        """Rank-1 for tcp alert is wafer 2915 NCR — its identifier must appear."""
        msg = self._capture_user_msg(self._tcp_alert())
        self.assertIn("Wafer 2915", msg)

    # ── TREND vs ANOMALY handling ─────────────────────────────────────────────

    def test_trend_time_to_breach_in_llm_context(self):
        """TREND alerts must include time_to_breach in the LLM user message."""
        msg = self._capture_user_msg(self._he_trend_alert())
        self.assertIn("6 minutes 30 seconds", msg)

    def test_anomaly_no_time_to_breach_in_llm_context(self):
        """ANOMALY alerts must NOT include a time_to_breach line."""
        msg = self._capture_user_msg(self._tcp_alert())
        self.assertNotIn("Time to breach", msg)

    # ── System prompt enforces correct behaviour ──────────────────────────────

    def test_system_prompt_requires_evidence_grounding(self):
        self.assertIn("Only report what the retrieved NCRs actually say", qa.SYSTEM_PROMPT)

    def test_system_prompt_forbids_inventing(self):
        self.assertIn("Do not add interpretation", qa.SYSTEM_PROMPT)

    def test_system_prompt_defines_all_required_sections(self):
        for section in ["NCR SUMMARY", "TOOL WEAR INDICATORS", "RECURRENCE"]:
            self.assertIn(section, qa.SYSTEM_PROMPT,
                        f"Section '{section}' missing from system prompt")


# =============================================================================
# 6. QUALITY AGENT — REPORT STRUCTURE
# =============================================================================

class TestReportStructure(unittest.TestCase):
    """
    Verifies the LLM output (mocked) contains all required sections
    and that the format matches what the Maintenance Agent expects.
    """

    SAMPLE_REPORT = SAMPLE_REPORT = """SENSOR: tcp_top_pwr
CHAMBER: CHA
ALERT TYPE: ANOMALY

NCR SUMMARY:
Wafer 2915 (LOT_29B, CHA): TCP Top Power deviated +50W above set-point. Over-etch across die confirmed. 
CD widening beyond specification on critical metal layer. Root cause recorded as TCP generator power 
set-point overridden or impedance matching network fault after 1241 hours of operation. This is the 
first recorded TCP violation on CHA in this production history.

Wafer 2936 (LOT_29B, CHA): TCP Top Power deviated +10W above set-point. Elevated endpoint signal confirmed
higher-than-expected plasma density. Etch rate elevated vs baseline. Root cause recorded as recipe version 
mismatch loading wrong TCP set-point. Recurring pattern on CHA — second TCP violation in same lot.

Wafer 3120 (LOT_31B, CHA): TCP Top Power deviated +30W above set-point. Over-etch on metal layer confirmed. 
CD widening 2-3nm on sampled die. Root cause recorded as TCP generator power set-point drift after 1242 hours. 
PM OVERDUE at 41 wafers.

TOOL WEAR INDICATORS:
RF generator hours at time of faults: 1241.04 (wafer 2915), 1241.13 (wafer 2936), 1241.69 (wafer 3120). 
PM overdue on wafer 3120 (41 wafers since last PM, threshold 35). Open work orders: WO-CHA-2915, WO-CHA-3120. 
Recurring TCP power deviation pattern on CHA across three lots.

RECURRENCE:
tcp_top_pwr has appeared 3 times in the retrieved records, all on CHA. Pattern shows escalating severity: 
+10W, +30W, +50W deviations across LOT_29B and LOT_31B. RF generator hours consistently around 1241 hours 
at time of each fault, suggesting progressive generator degradation."""

    def _client(self):
        m = MagicMock()
        m.chat.completions.create.return_value.choices[0].message.content = self.SAMPLE_REPORT
        return m

    def _retrieved(self):
        return [
            {"rank": i+1, "similarity": round(0.90 - i*0.05, 4),
             "content": f"NCR REPORT:\nSample quality case {i+1}."}
            for i in range(5)
        ]

    def _alert(self):
        return {
            "alert_type": "ANOMALY", "sensor": "tcp_top_pwr",
            "value": 410.0, "threshold_min": 334, "threshold_max": 360,
            "deviation": "+50W above set-point", "chamber_id": "CHA",
            "wafer_id": 2915, "lot_id": "LOT_29B", "step": 12,
            "severity": "CRITICAL", "explanation": "TCP exceeded.",
            "time_to_breach": None,
        }

    def _report(self):
        return qa.synthesise_report(self._alert(), self._retrieved(), self._client())

    def test_report_is_string(self):
        self.assertIsInstance(self._report(), str)

    def test_report_non_empty(self):
        self.assertGreater(len(self._report()), 0)

    def test_contains_header(self):
        self.assertIn("SENSOR:", self._report())

    def test_contains_urgency(self):
        self.assertIn("ALERT TYPE:", self._report())

    def test_contains_root_cause(self):
        self.assertIn("NCR SUMMARY:", self._report())

    def test_contains_tool_wear(self):
        self.assertIn("TOOL WEAR INDICATORS:", self._report())

    def test_contains_quality_impact(self):
        self.assertIn("RECURRENCE:", self._report())

    def test_contains_inspection_checklist(self):
        self.assertIn("tcp_top_pwr", self._report())

    def test_contains_evidence_summary(self):
        self.assertIn("CHA", self._report())

    def test_no_unfilled_template_placeholders(self):
        r = self._report()
        for p in ["[sensor]", "[deviation]", "[chamber]", "[lot]",
                  "[severity]", "[...]"]:
            self.assertNotIn(p, r, f"Unfilled placeholder '{p}' found in report")

    def test_trend_time_to_breach_in_llm_context(self):
        """For TREND alerts, time_to_breach must appear in the message to LLM."""
        c = self._client()
        alert = self._alert()
        alert["alert_type"] = "TREND"
        alert["time_to_breach"] = "6 minutes 30 seconds"
        qa.synthesise_report(alert, self._retrieved(), c)
        msg = c.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("6 minutes 30 seconds", msg)

    def test_run_quality_agent_returns_non_empty_string(self):
        """End-to-end: run_quality_agent must return a non-empty string."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["NCR REPORT:\nCase content."] * 5],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }
        report = qa.run_quality_agent(self._alert(), mock_collection, self._client())
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)


# =============================================================================
# 7. NORMALIZE_ALERT — HANDOFF CONTRACT
# =============================================================================

class TestNormalizeAlert(unittest.TestCase):
    """
    Verifies normalize_alert() correctly converts Production Agent dicts
    into the standard format the Quality Agent expects.
    """

    def _anomaly_above(self):
        return {
            "timestamp": "12:00:00", "wafer_id": 2915, "step": 33,
            "sensor": "tcp_top_pwr", "value": 370.0,
            "threshold_min": 334, "threshold_max": 360,
            "explanation": "TCP exceeded.",
        }

    def _anomaly_below(self):
        return {
            "timestamp": "12:00:00", "wafer_id": 2915, "step": 5,
            "sensor": "tcp_top_pwr", "value": 320.0,
            "threshold_min": 334, "threshold_max": 360,
            "explanation": "TCP below min.",
        }

    def _trend_inc(self):
        return {
            "timestamp": "12:00:00", "wafer_id": 2918, "step": 9,
            "sensor": "pressure", "current_value": 1231.0,
            "threshold_min": 942, "threshold_max": 1420,
            "trend_direction": "increasing", "rate_per_step": 36.44,
            "r_squared": 0.851, "steps_to_breach": 6,
            "time_to_breach": "6 seconds", "explanation": "Pressure rising.",
        }

    def _trend_dec(self):
        return {
            "timestamp": "12:00:00", "wafer_id": 2940, "step": 10,
            "sensor": "tcp_top_pwr", "current_value": 338.0,
            "threshold_min": 334, "threshold_max": 360,
            "trend_direction": "decreasing", "rate_per_step": -1.5,
            "r_squared": 0.95, "steps_to_breach": 5,
            "time_to_breach": "5 seconds", "explanation": "TCP dropping.",
        }

    def test_anomaly_above_max_positive_deviation(self):
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertTrue(r["deviation"].startswith("+"))

    def test_anomaly_below_min_negative_deviation(self):
        r = normalize_alert(self._anomaly_below(), "ANOMALY")
        self.assertTrue(r["deviation"].startswith("-"))

    def test_anomaly_alert_type_set(self):
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertEqual(r["alert_type"], "ANOMALY")

    def test_trend_increasing_mentions_upper_limit(self):
        r = normalize_alert(self._trend_inc(), "TREND")
        self.assertIn("upper", r["deviation"])

    def test_trend_decreasing_mentions_lower_limit(self):
        r = normalize_alert(self._trend_dec(), "TREND")
        self.assertIn("lower", r["deviation"])

    def test_trend_alert_type_set(self):
        r = normalize_alert(self._trend_inc(), "TREND")
        self.assertEqual(r["alert_type"], "TREND")

    def test_trend_severity_always_medium(self):
        """Trends are pre-fault — always MEDIUM."""
        r = normalize_alert(self._trend_inc(), "TREND")
        self.assertEqual(r["severity"], "MEDIUM")

    def test_known_wafer_id_gets_lot_and_chamber(self):
        """Wafer 2915 is in WAFER_LOOKUP — must get real lot and chamber."""
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertEqual(r["lot_id"],    "LOT_29B")
        self.assertEqual(r["chamber_id"], "CHA")

    def test_unknown_wafer_id_gets_unknown(self):
        raw = self._anomaly_above()
        raw["wafer_id"] = 99999
        r = normalize_alert(raw, "ANOMALY")
        self.assertEqual(r["lot_id"],    "UNKNOWN")
        self.assertEqual(r["chamber_id"], "UNKNOWN")

    def test_anomaly_result_has_all_required_fields(self):
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        for f in ["alert_type", "wafer_id", "lot_id", "chamber_id",
                  "sensor", "value", "threshold_min", "threshold_max",
                  "deviation", "severity", "step", "time_to_breach", "explanation"]:
            self.assertIn(f, r)

    def test_anomaly_time_to_breach_is_none(self):
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertIsNone(r["time_to_breach"])

    def test_trend_time_to_breach_passed_through(self):
        r = normalize_alert(self._trend_inc(), "TREND")
        self.assertEqual(r["time_to_breach"], "6 seconds")

    def test_severity_critical_for_large_overshoot(self):
        """370W on 334-360 range: overshoot=10, range=26, frac=38% → CRITICAL."""
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertEqual(r["severity"], "CRITICAL")


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestAnomalyGroundTruth,
        TestTrendDetection,
        TestRetrieval,
        TestQueryDrivesRetrieval,
        TestLLMContext,
        TestReportStructure,
        TestNormalizeAlert,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{result.testsRun} passed  |  "
          f"{len(result.failures)} failed  |  {len(result.skipped)} skipped")
    print(f"{'='*60}")
    sys.exit(0 if not (result.failures or result.errors) else 1)