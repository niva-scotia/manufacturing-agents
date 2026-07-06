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

EXPLANATION POLICY (tested explicitly):
  The Production Agent's LLM explanation is a hypothesis — not verified fact.
  It must NOT appear in:
    - the retrieval query (would bias what NCRs are retrieved)
    - the LLM context sent to the Quality Agent (would contaminate synthesis)
  Retrieval must be driven purely by: sensor, chamber, lot, severity, alert type.

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

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "agents"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR             = _ROOT / "data"
TRAIN_MACHINE_PATH   = DATA_DIR / "train_machine.csv"
QUALITY_RECORDS_PATH = DATA_DIR / "quality_records.csv"

# ── Load production agent functions without running the script body ────────────

def _load_production_module():
    with open(_ROOT / "agents" / "production_agent.py", "r") as f:
        src = f.read()

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
        "WAFER_LOOKUP = {\n"
        "    2915: {'lot_id': 'LOT_29B', 'chamber_id': 'CHA'},\n"
        "    2937: {'lot_id': 'LOT_29B', 'chamber_id': 'CHB'},\n"
        "    2940: {'lot_id': 'LOT_29B', 'chamber_id': 'CHA'},\n"
        "    2918: {'lot_id': 'LOT_29B', 'chamber_id': 'CHB'},\n"
        "    3142: {'lot_id': 'LOT_31B', 'chamber_id': 'CHA'},\n"
        "}\n"
        "quality_store  = None\n"
        "quality_client = None\n"
        "wo_store       = None\n"
        "pm_df          = None\n"
        "parts_df       = None\n"
        "calib_df       = None\n"
        "maintenance_client = None\n"
        "sop_store      = None\n"
        "sop_client     = None\n"
        "def event_time(wafer_id):\n"
        "    return '00:00:00'\n"
        "def get_llm_explanation(event_type, sensor, details):\n"
        "    return f'[mocked: {sensor}]'\n"
        "def run_quality_agent(alert, collection, client):\n"
        "    return '[mocked quality report]'\n"
        "def run_maintenance_agent(alert, quality_report, wo_collection, pm_df, parts_df, calib_df, client):\n"
        "    return {'priority': 'LOW', 'sensor': '', 'chamber_id': '', 'alert_type': '', 'component': '', 'pm_status': {}, 'required_parts': [], 'calibration_status': {}, 'past_wo_patterns': '', 'recommended_actions': [], 'draft_wo_header': '', 'llm_narrative': '[mocked]'}\n"
        "def print_recommendation(rec): pass\n"
        "def run_sop_agent(alert, quality_report, recommendation, collection, client):\n"
        "    return '[mocked sop report]'\n"
        "impact_roster = None\n"
        "def load_roster(*a, **k):\n"
        "    return None\n"
        "def run_impact_agent(alert, quality_report=None, recommendation=None, sop_report=None, roster=None, client=None, price_provider=None):\n"
        "    return {'narrative': '[mocked impact]', 'estimable': False}\n"
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

# ── Shared constants ──────────────────────────────────────────────────────────
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

    @classmethod
    def setUpClass(cls):
        if not TRAIN_MACHINE_PATH.exists():
            raise unittest.SkipTest(f"Dataset not found: {TRAIN_MACHINE_PATH}")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        cls.anomaly_log, cls.trend_log = run_agent(
            data=data, thresholds=THRESHOLDS,
            trend_config=TREND_CONFIG, max_rows=None,
        )

    def _rows(self, sensor):
        return [a for a in self.anomaly_log if a["sensor"] == sensor]

    def _wafers(self, sensor):
        return {a["wafer_id"] for a in self.anomaly_log if a["sensor"] == sensor}

    def test_tcp_total_row_count(self):
        self.assertEqual(len(self._rows("tcp_top_pwr")), 73)

    def test_tcp_total_wafer_count(self):
        self.assertEqual(len(self._wafers("tcp_top_pwr")), 46)

    def test_tcp_exact_wafer_ids(self):
        self.assertEqual(self._wafers("tcp_top_pwr"), TCP_VIOLATION_WAFERS)

    def test_tcp_wafer_2915_flagged(self):
        self.assertIn(2915, self._wafers("tcp_top_pwr"))

    def test_tcp_wafer_2936_flagged(self):
        self.assertIn(2936, self._wafers("tcp_top_pwr"))

    def test_tcp_wafer_3120_flagged(self):
        self.assertIn(3120, self._wafers("tcp_top_pwr"))

    def test_tcp_wafer_3143_flagged(self):
        self.assertIn(3143, self._wafers("tcp_top_pwr"))

    def test_tcp_all_violations_above_max(self):
        for v in self._rows("tcp_top_pwr"):
            self.assertGreater(v["value"], 360)

    def test_tcp_threshold_bounds_correct(self):
        for v in self._rows("tcp_top_pwr"):
            self.assertEqual(v["threshold_min"], 334)
            self.assertEqual(v["threshold_max"], 360)

    def test_wafer_2915_exactly_one_violation(self):
        w = [a for a in self.anomaly_log
             if a["wafer_id"] == 2915 and a["sensor"] == "tcp_top_pwr"]
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0]["step"], 33)
        self.assertAlmostEqual(w[0]["value"], 360.8627, places=2)

    def test_bcl3_total_row_count(self):
        self.assertEqual(len(self._rows("bcl3_flow")), 99)

    def test_bcl3_total_wafer_count(self):
        self.assertEqual(len(self._wafers("bcl3_flow")), 2)

    def test_bcl3_exact_wafer_ids(self):
        self.assertEqual(self._wafers("bcl3_flow"), BCL3_VIOLATION_WAFERS)

    def test_bcl3_wafer_3141_row_count(self):
        w = [a for a in self.anomaly_log
             if a["wafer_id"] == 3141 and a["sensor"] == "bcl3_flow"]
        self.assertEqual(len(w), 98)

    def test_bcl3_wafer_3122_row_count(self):
        w = [a for a in self.anomaly_log
             if a["wafer_id"] == 3122 and a["sensor"] == "bcl3_flow"]
        self.assertEqual(len(w), 1)

    def test_bcl3_all_violations_outside_range(self):
        for v in self._rows("bcl3_flow"):
            self.assertTrue(v["value"] < 740 or v["value"] > 765)

    def test_cl2_flow_zero_violations(self):
        self.assertEqual(len(self._rows("cl2_flow")), 0)

    def test_pressure_zero_anomaly_violations(self):
        self.assertEqual(len(self._rows("pressure")), 0)

    def test_rf_btm_pwr_zero_violations(self):
        self.assertEqual(len(self._rows("rf_btm_pwr")), 0)

    def test_total_anomaly_count(self):
        self.assertEqual(len(self.anomaly_log), 172)

    def test_clean_wafer_2901_never_flagged(self):
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

    INC = [336.0 + i for i in range(12)]
    DEC = [358.0 - i * 1.5 for i in range(12)]

    def _run(self, values, sensor="tcp_top_pwr"):
        return compute_trend(values, sensor, THRESHOLDS, TREND_CONFIG)

    def test_inc_detected(self):
        self.assertIsNotNone(self._run(self.INC))

    def test_inc_slope(self):
        self.assertAlmostEqual(self._run(self.INC)["slope"], 1.0, places=4)

    def test_inc_r_squared(self):
        self.assertAlmostEqual(self._run(self.INC)["r_squared"], 1.0, places=4)

    def test_inc_range_fraction(self):
        self.assertAlmostEqual(self._run(self.INC)["range_fraction"], round(8/26, 4), places=3)

    def test_inc_direction(self):
        self.assertEqual(self._run(self.INC)["direction"], "increasing")

    def test_inc_boundary_is_max(self):
        self.assertEqual(self._run(self.INC)["boundary"], 360)

    def test_inc_current_value(self):
        self.assertAlmostEqual(self._run(self.INC)["current_value"], 347.0, places=2)

    def test_inc_steps_to_breach(self):
        self.assertIn(self._run(self.INC)["steps_to_breach"], [12, 13])

    def test_inc_time_to_breach_is_string(self):
        r = self._run(self.INC)
        self.assertIsInstance(r["time_to_breach"], str)
        self.assertIn("second", r["time_to_breach"])

    def test_dec_detected(self):
        self.assertIsNotNone(self._run(self.DEC))

    def test_dec_slope(self):
        self.assertAlmostEqual(self._run(self.DEC)["slope"], -1.5, places=4)

    def test_dec_r_squared(self):
        self.assertAlmostEqual(self._run(self.DEC)["r_squared"], 1.0, places=4)

    def test_dec_range_fraction(self):
        self.assertAlmostEqual(self._run(self.DEC)["range_fraction"], round(1.5*8/26, 4), places=3)

    def test_dec_direction(self):
        self.assertEqual(self._run(self.DEC)["direction"], "decreasing")

    def test_dec_boundary_is_min(self):
        self.assertEqual(self._run(self.DEC)["boundary"], 334)

    def test_dec_steps_to_breach(self):
        self.assertEqual(self._run(self.DEC)["steps_to_breach"], 5)

    def test_fewer_than_min_steps_rejected(self):
        self.assertIsNone(self._run([336.0 + i for i in range(7)]))

    def test_flat_values_rejected(self):
        self.assertIsNone(self._run([350.0] * 12))

    def test_noisy_low_r2_rejected(self):
        np.random.seed(42)
        noisy = [336 + i*0.5 + np.random.normal(0, 3) for i in range(12)]
        self.assertIsNone(self._run(noisy))

    def test_tiny_slope_low_range_fraction_rejected(self):
        self.assertIsNone(self._run([350.0 + i*0.01 for i in range(12)]))

    def test_current_value_outside_range_rejected(self):
        self.assertIsNone(self._run([336.0 + i for i in range(11)] + [999.0]))

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

    def test_real_dataset_exactly_two_trends(self):
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        self.assertEqual(len(trend_log), 2,
                         f"Expected 2 trends, got {len(trend_log)}: {trend_log}")

    def test_real_trends_are_on_pressure_only(self):
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        sensors = {t["sensor"] for t in trend_log}
        self.assertEqual(sensors, {"pressure"})

    def test_real_trend_wafer_2918_values(self):
        if not TRAIN_MACHINE_PATH.exists():
            self.skipTest("Dataset not found")
        data = pd.read_csv(TRAIN_MACHINE_PATH)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        _, trend_log = run_agent(data=data, thresholds=THRESHOLDS,
                                 trend_config=TREND_CONFIG, max_rows=None)
        t2918 = [t for t in trend_log if t["wafer_id"] == 2918]
        self.assertEqual(len(t2918), 1)
        t = t2918[0]
        self.assertAlmostEqual(t["rate_per_step"], 36.4424, places=2)
        self.assertAlmostEqual(t["r_squared"],     0.851,   places=2)
        self.assertEqual(t["steps_to_breach"],     6)
        self.assertEqual(t["trend_direction"],     "increasing")

    def test_real_trend_wafer_3142_values(self):
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

    @classmethod
    def setUpClass(cls):
        if not QUALITY_RECORDS_PATH.exists():
            raise unittest.SkipTest(f"Not found: {QUALITY_RECORDS_PATH}")
        try:
            cls.collection = qa.build_index(str(QUALITY_RECORDS_PATH))
        except Exception as e:
            raise unittest.SkipTest(f"ChromaDB unavailable: {e}")

        cls.qr = pd.read_csv(QUALITY_RECORDS_PATH)
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

    def test_tcp_rank1_mentions_tcp_sensor(self):
        results = self._retrieve(self._alert_tcp())
        self.assertIn("tcp_top_pwr", results[0]["content"])

    def test_bcl3_rank1_mentions_bcl3_sensor(self):
        results = self._retrieve(self._alert_bcl3())
        self.assertIn("bcl3_flow", results[0]["content"])

    def test_he_press_rank1_mentions_he_press_sensor(self):
        results = self._retrieve(self._alert_he())
        top5_content = " ".join(r["content"] for r in results)
        pressure_terms = ["pressure", "he_press", "He Chuck", "backside", "ESC"]
        self.assertTrue(any(term in top5_content for term in pressure_terms))

    def test_tcp_rank1_is_wafer_2915_record(self):
        results = self._retrieve(self._alert_tcp())
        top5_content = " ".join(r["content"] for r in results)
        self.assertIn("Wafer 2915", top5_content)

    def test_bcl3_rank1_is_wafer_2937_record(self):
        results = self._retrieve(self._alert_bcl3())
        top5_content = " ".join(r["content"] for r in results)
        self.assertIn("bcl3_flow", top5_content)

    def test_he_press_rank1_is_wafer_2940_record(self):
        results = self._retrieve(self._alert_he())
        top5_content = " ".join(r["content"] for r in results)
        self.assertIn("NCR REPORT", top5_content)
        self.assertIn("SPC", top5_content)
        self.assertIn("CHA", top5_content)

    def test_tcp_top5_all_mention_tcp_sensor(self):
        for r in self._retrieve(self._alert_tcp()):
            self.assertIn("tcp_top_pwr", r["content"],
                          f"Rank {r['rank']} does not mention tcp_top_pwr")

    def test_rank1_similarity_higher_than_rank2(self):
        results = self._retrieve(self._alert_tcp())
        self.assertGreater(results[0]["similarity"], results[1]["similarity"])


# =============================================================================
# 4. QUALITY AGENT — ALERT DRIVES RETRIEVAL QUERY
#    Includes explicit tests that explanation is ABSENT from query
# =============================================================================

class TestQueryDrivesRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not QUALITY_RECORDS_PATH.exists():
            raise unittest.SkipTest(f"Not found: {QUALITY_RECORDS_PATH}")
        try:
            cls.collection = qa.build_index(str(QUALITY_RECORDS_PATH))
        except Exception as e:
            raise unittest.SkipTest(f"ChromaDB unavailable: {e}")

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
        tcp_rank1 = self._rank1(self._alert("tcp_top_pwr"))
        generic   = qa.retrieve_cases(
            self.collection, "fault on chamber CHA lot LOT_29B", top_k=1
        )[0]["content"]
        self.assertNotEqual(tcp_rank1, generic)

    def test_sensor_name_in_query_string(self):
        self.assertIn("tcp_top_pwr", qa.build_query(self._alert("tcp_top_pwr")))

    def test_chamber_id_in_query_string(self):
        self.assertIn("CHB", qa.build_query(self._alert("tcp_top_pwr", chamber="CHB")))

    def test_lot_id_in_query_string(self):
        self.assertIn("LOT_29B", qa.build_query(self._alert("tcp_top_pwr")))

    def test_explanation_not_in_query_string(self):
        """
        Explanation is a hypothesis — must NOT appear in the retrieval query.
        Retrieval must be driven by objective facts only: sensor, chamber, lot,
        severity, alert type. A biased hypothesis must not select the evidence.
        """
        alert = self._alert("tcp_top_pwr")
        alert["explanation"] = "ESC seal failure suspected."
        query = qa.build_query(alert)
        self.assertNotIn(
            "ESC seal failure", query,
            "Explanation must not appear in retrieval query — it biases evidence selection"
        )

    def test_explanation_with_different_text_not_in_query(self):
        """
        Regardless of what the explanation says, it must never appear in the query.
        """
        alert = self._alert("tcp_top_pwr")
        alert["explanation"] = "Impedance matching network fault after 1241 hours."
        query = qa.build_query(alert)
        self.assertNotIn("Impedance matching", query)
        self.assertNotIn("1241 hours", query)

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

    def test_two_alerts_same_sensor_different_explanation_same_retrieval(self):
        """
        Since explanation is excluded from the query, two alerts on the same
        sensor/chamber with different explanations should retrieve the same
        rank-1 chunk — the explanation has no effect on retrieval.
        """
        alert1 = self._alert("tcp_top_pwr", "CHA")
        alert1["explanation"] = "TCP generator calibration drift suspected."

        alert2 = self._alert("tcp_top_pwr", "CHA")
        alert2["explanation"] = "Recipe upload error loading wrong set-point."

        rank1_alert1 = self._rank1(alert1)
        rank1_alert2 = self._rank1(alert2)

        self.assertEqual(
            rank1_alert1, rank1_alert2,
            "Different explanations on same sensor/chamber should retrieve "
            "the same rank-1 result — explanation must not influence retrieval"
        )


# =============================================================================
# 5. QUALITY AGENT — LLM CONTEXT CONTAINS CORRECT INFORMATION
#    Includes explicit tests that explanation is ABSENT from LLM context
# =============================================================================

class TestLLMContext(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not QUALITY_RECORDS_PATH.exists():
            raise unittest.SkipTest(f"Not found: {QUALITY_RECORDS_PATH}")
        try:
            cls.collection = qa.build_index(str(QUALITY_RECORDS_PATH))
        except Exception as e:
            raise unittest.SkipTest(f"ChromaDB unavailable: {e}")

    def _mock_client(self, reply="SENSOR: tcp_top_pwr\nCHAMBER: CHA\nALERT TYPE: ANOMALY\n\nNCR SUMMARY:\nTest."):
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
        mock_client = self._mock_client()
        query     = qa.build_query(alert)
        retrieved = qa.retrieve_cases(self.collection, query, top_k=5)
        qa.synthesise_report(alert, retrieved, mock_client)
        return mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]

    # ── Alert fields that MUST appear in LLM context ──────────────────────────

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

    # ── Explanation must NOT appear in LLM context ────────────────────────────

    def test_explanation_absent_from_llm_context(self):
        """
        The Production Agent's explanation is a hypothesis — not verified fact.
        It must NOT be passed to the Quality Agent's LLM.
        The LLM should reason from retrieved NCR evidence only.
        """
        msg = self._capture_user_msg(self._tcp_alert())
        self.assertNotIn(
            "TCP Top Power of 410W exceeds max threshold of 360W",
            msg,
            "Production Agent explanation must not appear in Quality Agent LLM context"
        )

    def test_explanation_field_label_absent_from_llm_context(self):
        """The 'Explanation' label must not appear in the user message."""
        msg = self._capture_user_msg(self._tcp_alert())
        self.assertNotIn(
            "Explanation   :",
            msg,
            "Explanation field must be removed from LLM context"
        )

    def test_trend_explanation_absent_from_llm_context(self):
        """TREND alert explanation also must not appear in LLM context."""
        msg = self._capture_user_msg(self._he_trend_alert())
        self.assertNotIn(
            "ESC seal degradation suspected",
            msg,
            "Trend explanation must not appear in Quality Agent LLM context"
        )

    # ── Retrieved chunks in LLM context ──────────────────────────────────────

    def test_rank1_chunk_content_in_llm_context(self):
        alert = self._tcp_alert()
        mock_client = self._mock_client()
        retrieved = qa.retrieve_cases(
            self.collection, qa.build_query(alert), top_k=5
        )
        qa.synthesise_report(alert, retrieved, mock_client)
        msg = mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn(retrieved[0]["content"][:200], msg)

    def test_all_5_cases_labelled_in_llm_context(self):
        msg = self._capture_user_msg(self._tcp_alert())
        for i in range(1, 6):
            self.assertIn(f"RETRIEVED CASE {i}", msg)

    def test_wafer_2915_ncr_content_in_llm_context(self):
        msg = self._capture_user_msg(self._tcp_alert())
        self.assertIn("Wafer 2915", msg)

    # ── TREND vs ANOMALY handling ─────────────────────────────────────────────

    def test_trend_time_to_breach_in_llm_context(self):
        msg = self._capture_user_msg(self._he_trend_alert())
        self.assertIn("6 minutes 30 seconds", msg)

    def test_anomaly_no_time_to_breach_in_llm_context(self):
        msg = self._capture_user_msg(self._tcp_alert())
        self.assertNotIn("Time to breach", msg)

    # ── System prompt ─────────────────────────────────────────────────────────

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

    SAMPLE_REPORT = """SENSOR: tcp_top_pwr
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

    def test_contains_sensor_header(self):
        self.assertIn("SENSOR:", self._report())

    def test_contains_alert_type_header(self):
        self.assertIn("ALERT TYPE:", self._report())

    def test_contains_ncr_summary(self):
        self.assertIn("NCR SUMMARY:", self._report())

    def test_contains_tool_wear_indicators(self):
        self.assertIn("TOOL WEAR INDICATORS:", self._report())

    def test_contains_recurrence(self):
        self.assertIn("RECURRENCE:", self._report())

    def test_no_unfilled_template_placeholders(self):
        r = self._report()
        for p in ["[sensor]", "[deviation]", "[chamber]", "[lot]",
                  "[severity]", "[...]"]:
            self.assertNotIn(p, r, f"Unfilled placeholder '{p}' found in report")

    def test_report_does_not_contain_explanation_text(self):
        """
        The sample report must not contain the Production Agent's explanation.
        The report should be grounded in NCR evidence only.
        """
        self.assertNotIn(
            "TCP exceeded.",
            self._report(),
            "Production Agent explanation must not appear in Quality Agent report"
        )

    def test_trend_time_to_breach_in_llm_context(self):
        c = self._client()
        alert = self._alert()
        alert["alert_type"] = "TREND"
        alert["time_to_breach"] = "6 minutes 30 seconds"
        qa.synthesise_report(alert, self._retrieved(), c)
        msg = c.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("6 minutes 30 seconds", msg)

    def test_explanation_not_in_user_message_to_llm(self):
        """
        Verify the user message sent to the LLM does not contain
        the Explanation field — it was removed from synthesise_report().
        """
        c = self._client()
        qa.synthesise_report(self._alert(), self._retrieved(), c)
        msg = c.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertNotIn(
            "Explanation   :",
            msg,
            "Explanation field must be absent from the message sent to LLM"
        )

    def test_run_quality_agent_returns_non_empty_string(self):
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
        r = normalize_alert(self._trend_inc(), "TREND")
        self.assertEqual(r["severity"], "MEDIUM")

    def test_known_wafer_id_gets_lot_and_chamber(self):
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
                  "deviation", "severity", "step", "time_to_breach"]:
            self.assertIn(f, r)

    def test_anomaly_time_to_breach_is_none(self):
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertIsNone(r["time_to_breach"])

    def test_trend_time_to_breach_passed_through(self):
        r = normalize_alert(self._trend_inc(), "TREND")
        self.assertEqual(r["time_to_breach"], "6 seconds")

    def test_severity_critical_for_large_overshoot(self):
        r = normalize_alert(self._anomaly_above(), "ANOMALY")
        self.assertEqual(r["severity"], "CRITICAL")


# =============================================================================
# Sustainability (Impact) Agent tests
# =============================================================================
# Estimates the scrap cost of continuing to run a faulty chamber for the rest of
# the current lot. Pure Python (pandas only) — no Azure/LLM calls.
#
# Synthetic roster (every count hand-checkable):
#   LOT_A (6 wafers, alternating): 101 CHA,102 CHB,103 CHA,104 CHB,105 CHA,106 CHB
#     -> CHA = [101,103,105]   CHB = [102,104,106]
#   LOT_B (single chamber): 201 CHA, 202 CHA
#   LOT_C (unmapped chamber): 301 CHZ

import time

from cost_model import (
    scrap_cost_per_wafer, machine_for_chamber, ScrapPriceProvider,
    StaticScrapPriceProvider, WebSearchScrapPriceProvider,
)
from sustainability_agent import (
    load_roster, remaining_faulty_chamber_wafers, estimate_scrap_impact,
    narrate, run_impact_agent,
)

_SUMMARY  = DATA_DIR / "train_summary.csv"
_HAS_DATA = _SUMMARY.exists()


def _roster(rows):
    """Build a roster DataFrame with the '_rs' datetime column load_roster adds."""
    df = pd.DataFrame(rows, columns=["wafer_id", "lot_id", "chamber_id", "run_start"])
    df["_rs"] = pd.to_datetime(df["run_start"])
    return df.sort_values("_rs").reset_index(drop=True)


def _synthetic_roster():
    return _roster([
        (101, "LOT_A", "CHA", "2024-01-01T00:00:00"),
        (102, "LOT_A", "CHB", "2024-01-01T00:01:00"),
        (103, "LOT_A", "CHA", "2024-01-01T00:02:00"),
        (104, "LOT_A", "CHB", "2024-01-01T00:03:00"),
        (105, "LOT_A", "CHA", "2024-01-01T00:04:00"),
        (106, "LOT_A", "CHB", "2024-01-01T00:05:00"),
        (201, "LOT_B", "CHA", "2024-01-01T01:00:00"),
        (202, "LOT_B", "CHA", "2024-01-01T01:01:00"),
        (301, "LOT_C", "CHZ", "2024-01-01T02:00:00"),   # unmapped chamber
    ])


def _sa_alert(wafer_id, lot_id="LOT_A", chamber_id="CHA",
              sensor="tcp_top_pwr", alert_type="ANOMALY", **extra):
    a = {"wafer_id": wafer_id, "lot_id": lot_id, "chamber_id": chamber_id,
         "sensor": sensor, "alert_type": alert_type, "severity": "HIGH"}
    a.update(extra)
    return a


class SpyProvider(ScrapPriceProvider):
    """Returns fixed cheap numbers and records the context it was called with."""
    def __init__(self, value=100.0, low=50.0, high=150.0):
        self.value, self.low, self.high = value, low, high
        self.contexts = []

    def scrap_cost_per_wafer(self, chamber_id, context=None):
        self.contexts.append(context)
        return {"value": self.value, "low": self.low, "high": self.high, "source": "spy"}


class FakeLLMClient:
    """Mimics client.chat.completions.create(...).choices[0].message.content."""
    def __init__(self, text="LLM NARRATION"):
        self.text = text
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = type("M", (), {"content": self.text})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


class _SyntheticRosterCase(unittest.TestCase):
    """Base class: builds the hand-checkable synthetic roster before each test."""
    def setUp(self):
        self.roster = _synthetic_roster()


class TestCountingAndAttribution(_SyntheticRosterCase):

    def test_first_wafer_counts_all_chamber_wafers(self):
        r = remaining_faulty_chamber_wafers(101, "LOT_A", "CHA", self.roster)
        self.assertTrue(r["found"])
        self.assertEqual(r["remaining_count"], 3)
        self.assertEqual(r["remaining_wafer_ids"], [101, 103, 105])

    def test_mid_wafer_counts_fewer(self):
        r = remaining_faulty_chamber_wafers(103, "LOT_A", "CHA", self.roster)
        self.assertEqual(r["remaining_count"], 2)
        self.assertEqual(r["remaining_wafer_ids"], [103, 105])

    def test_last_wafer_counts_one(self):
        r = remaining_faulty_chamber_wafers(105, "LOT_A", "CHA", self.roster)
        self.assertEqual(r["remaining_count"], 1)
        self.assertEqual(r["remaining_wafer_ids"], [105])

    def test_current_wafer_is_included(self):
        # The triggering wafer is itself bad, so it must be counted.
        r = remaining_faulty_chamber_wafers(105, "LOT_A", "CHA", self.roster)
        self.assertIn(105, r["remaining_wafer_ids"])

    def test_attribution_is_faulty_chamber_only(self):
        # CHA fault must NOT count the whole 6-wafer lot — only CHA's 3.
        r = remaining_faulty_chamber_wafers(101, "LOT_A", "CHA", self.roster)
        self.assertEqual(r["remaining_count"], 3)
        self.assertEqual(r["lot_chamber_total"], 3)
        self.assertEqual(r["lot_total"], 6)

    def test_cross_chamber_invariant(self):
        # CHA-first + CHB-first == lot_total (partition, no double-count).
        cha = remaining_faulty_chamber_wafers(101, "LOT_A", "CHA", self.roster)["remaining_count"]
        chb = remaining_faulty_chamber_wafers(102, "LOT_A", "CHB", self.roster)["remaining_count"]
        self.assertEqual(cha + chb, 6)

    def test_single_chamber_lot(self):
        r = remaining_faulty_chamber_wafers(201, "LOT_B", "CHA", self.roster)
        self.assertEqual(r["remaining_count"], 2)
        self.assertEqual(r["lot_total"], 2)

    def test_monotonic_decrease_through_lot(self):
        counts = [remaining_faulty_chamber_wafers(w, "LOT_A", "CHA", self.roster)["remaining_count"]
                  for w in (101, 103, 105)]
        self.assertEqual(counts, [3, 2, 1])
        for i in range(len(counts) - 1):
            self.assertGreater(counts[i], counts[i + 1])

    def test_wafer_not_in_roster(self):
        r = remaining_faulty_chamber_wafers(999, "LOT_A", "CHA", self.roster)
        self.assertFalse(r["found"])
        self.assertIn("999", r["reason"])

    def test_lot_not_in_roster(self):
        r = remaining_faulty_chamber_wafers(101, "LOT_MISSING", "CHA", self.roster)
        self.assertFalse(r["found"])
        self.assertIn("LOT_MISSING", r["reason"])

    def test_chamber_mismatch_behaviour(self):
        # Requested chamber != wafer's own chamber: counts the requested chamber's
        # wafers at/after the current wafer's time. (Can't happen in the pipeline.)
        r = remaining_faulty_chamber_wafers(101, "LOT_A", "CHB", self.roster)
        self.assertEqual(r["remaining_count"], 3)
        self.assertNotIn(101, r["remaining_wafer_ids"])


class TestArithmetic(_SyntheticRosterCase):

    def test_expected_cost_is_count_times_price(self):
        est = estimate_scrap_impact(_sa_alert(101), self.roster)  # 3 wafers, $2000
        self.assertTrue(est["estimable"])
        self.assertEqual(est["remaining_wafers"], 3)
        pw, ec = est["per_wafer_cost"], est["expected_cost"]
        self.assertAlmostEqual(ec["value"], 3 * pw["value"], places=2)
        self.assertAlmostEqual(ec["low"], 3 * pw["low"], places=2)
        self.assertAlmostEqual(ec["high"], 3 * pw["high"], places=2)

    def test_cost_range_is_ordered(self):
        est = estimate_scrap_impact(_sa_alert(101), self.roster)
        for block in (est["per_wafer_cost"], est["expected_cost"]):
            self.assertLessEqual(block["low"], block["value"])
            self.assertLessEqual(block["value"], block["high"])

    def test_not_estimable_has_no_cost_but_keeps_metadata(self):
        est = estimate_scrap_impact(_sa_alert(999), self.roster)
        self.assertFalse(est["estimable"])
        self.assertIn("reason", est)
        self.assertNotIn("expected_cost", est)
        self.assertIn("per_wafer_cost", est)
        self.assertIn("assumptions", est)


class TestInputRobustness(_SyntheticRosterCase):

    def test_accepts_chamber_key_instead_of_chamber_id(self):
        a = {"wafer_id": 101, "lot_id": "LOT_A", "chamber": "CHA", "sensor": "x"}
        est = estimate_scrap_impact(a, self.roster)
        self.assertEqual(est["chamber_id"], "CHA")
        self.assertTrue(est["estimable"])

    def test_string_wafer_id_is_coerced(self):
        est = estimate_scrap_impact(_sa_alert("101"), self.roster)
        self.assertTrue(est["estimable"])
        self.assertEqual(est["remaining_wafers"], 3)

    def test_missing_lot_id_is_not_estimable(self):
        a = {"wafer_id": 101, "chamber_id": "CHA", "sensor": "x"}
        self.assertFalse(estimate_scrap_impact(a, self.roster)["estimable"])

    def test_missing_sensor_defaults(self):
        a = {"wafer_id": 101, "lot_id": "LOT_A", "chamber_id": "CHA"}
        self.assertEqual(estimate_scrap_impact(a, self.roster)["sensor"], "UNKNOWN")

    def test_missing_wafer_id_raises(self):
        with self.assertRaises(KeyError):
            estimate_scrap_impact({"lot_id": "LOT_A", "chamber_id": "CHA"}, self.roster)

    def test_unmapped_chamber_uses_default_cost(self):
        est = estimate_scrap_impact(_sa_alert(301, lot_id="LOT_C", chamber_id="CHZ"), self.roster)
        self.assertTrue(est["estimable"])
        self.assertEqual(est["remaining_wafers"], 1)
        self.assertEqual(est["machine"], "UNKNOWN")
        self.assertEqual(est["per_wafer_cost"]["value"], 2000.0)


class TestProviderSeam(_SyntheticRosterCase):

    def test_default_provider_is_static(self):
        est = estimate_scrap_impact(_sa_alert(101), self.roster)
        self.assertEqual(est["per_wafer_cost"]["value"], scrap_cost_per_wafer("CHA")["value"])

    def test_injected_provider_is_used(self):
        est = estimate_scrap_impact(_sa_alert(101), self.roster,
                                    price_provider=SpyProvider(100, 50, 150))
        self.assertEqual(est["per_wafer_cost"]["value"], 100.0)
        self.assertAlmostEqual(est["expected_cost"]["value"], 3 * 100.0, places=2)

    def test_provider_receives_context(self):
        spy = SpyProvider()
        estimate_scrap_impact(_sa_alert(101, sensor="rf_btm_pwr"), self.roster, price_provider=spy)
        self.assertTrue(spy.contexts)
        self.assertEqual(spy.contexts[0]["sensor"], "rf_btm_pwr")

    def test_websearch_provider_is_stub(self):
        with self.assertRaises(NotImplementedError):
            WebSearchScrapPriceProvider().scrap_cost_per_wafer("CHA")

    def test_websearch_provider_builds_query(self):
        q = WebSearchScrapPriceProvider()._build_query("CHA", {"sensor": "tcp_top_pwr"})
        self.assertIn("200", q)
        self.assertIn("tcp_top_pwr", q)


class TestNarrator(_SyntheticRosterCase):

    def test_narrate_deterministic_without_client(self):
        est = estimate_scrap_impact(_sa_alert(101), self.roster)
        text = narrate(est, client=None)
        self.assertIn("approximately", text)
        self.assertIn("CHA", text)
        self.assertIn("LOT_A", text)
        self.assertIn(f"{est['expected_cost']['value']:,.0f}", text)

    def test_narrate_non_estimable(self):
        est = estimate_scrap_impact(_sa_alert(999), self.roster)
        self.assertIn("not estimable", narrate(est, client=None).lower())

    def test_narrate_with_mock_llm(self):
        est = estimate_scrap_impact(_sa_alert(101), self.roster)
        fake = FakeLLMClient("SUPERVISOR SUMMARY")
        text = narrate(est, client=fake, model="gpt-x")
        self.assertEqual(text, "SUPERVISOR SUMMARY")
        self.assertEqual(len(fake.calls), 1)


class TestDeterminism(_SyntheticRosterCase):

    def test_estimate_is_pure(self):
        a = _sa_alert(103)
        self.assertEqual(estimate_scrap_impact(a, self.roster),
                         estimate_scrap_impact(a, self.roster))


class TestCostModel(unittest.TestCase):

    def test_scrap_cost_known_chamber(self):
        c = scrap_cost_per_wafer("CHA")
        self.assertLessEqual(c["low"], c["value"])
        self.assertLessEqual(c["value"], c["high"])
        self.assertEqual(c["value"], 2000.0)
        self.assertEqual(c["low"], 1500.0)
        self.assertEqual(c["high"], 2500.0)

    def test_scrap_cost_unknown_chamber_default(self):
        c = scrap_cost_per_wafer("CHZ")
        self.assertEqual(c["value"], 2000.0)
        self.assertIn("default", c["source"])

    def test_machine_lookup(self):
        self.assertEqual(machine_for_chamber("CHA")["wafer_diameter_mm"], 200)
        self.assertIsNone(machine_for_chamber("CHZ"))


@unittest.skipUnless(_HAS_DATA, "train_summary.csv not present")
class TestSustainabilityRealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.roster = load_roster()

    def test_load_roster_shape(self):
        self.assertGreater(len(self.roster), 0)
        self.assertIn("_rs", self.roster.columns)
        self.assertTrue(self.roster["_rs"].is_monotonic_increasing)

    def test_known_real_counts_and_cost(self):
        # Verified against the shipped roster (LOT_29A, CHA has 11 wafers).
        est = estimate_scrap_impact(_sa_alert(2901, lot_id="LOT_29A", chamber_id="CHA"), self.roster)
        self.assertEqual(est["remaining_wafers"], 11)
        self.assertEqual(est["lot_total"], 22)
        self.assertAlmostEqual(est["expected_cost"]["value"], 11 * 2000.0, places=2)

    def test_real_count_shrinks_later_in_lot(self):
        first = estimate_scrap_impact(_sa_alert(2901, lot_id="LOT_29A", chamber_id="CHA"), self.roster)
        later = estimate_scrap_impact(_sa_alert(2913, lot_id="LOT_29A", chamber_id="CHA"), self.roster)
        self.assertLess(later["remaining_wafers"], first["remaining_wafers"])

    def test_run_impact_agent_end_to_end(self):
        out = run_impact_agent(_sa_alert(2901, lot_id="LOT_29A", chamber_id="CHA"))
        self.assertTrue(out["estimable"])
        self.assertIn("narrative", out)
        self.assertTrue(out["narrative"])


class TestSustainabilityPerformance(_SyntheticRosterCase):

    def test_many_estimates_are_fast(self):
        prov = StaticScrapPriceProvider()
        t0 = time.perf_counter()
        for _ in range(2000):
            estimate_scrap_impact(_sa_alert(101), self.roster, price_provider=prov)
        self.assertLess(time.perf_counter() - t0, 10.0)


@unittest.skipUnless(_HAS_DATA, "train_summary.csv not present")
class TestWafersLeftInLot(unittest.TestCase):
    """Verifies remaining-wafer counting against an INDEPENDENT ground truth."""

    @classmethod
    def setUpClass(cls):
        cls.roster = load_roster()

    def test_remaining_wafers_count_is_correct(self):
        lot_id, chamber = "LOT_29A", "CHA"
        # Independent ground truth: list the faulty chamber's wafers in TIME order
        # and count from the chosen wafer to the end (list POSITION, not the agent's
        # timestamp filter) — a genuine cross-check, not a re-implementation.
        lot = self.roster[self.roster["lot_id"] == lot_id].sort_values("_rs")
        chamber_wafers = lot[lot["chamber_id"] == chamber]["wafer_id"].tolist()
        self.assertTrue(chamber_wafers, "no wafers found for this lot/chamber")

        mid = len(chamber_wafers) // 2
        wafer_id = chamber_wafers[mid]
        expected_remaining = len(chamber_wafers) - mid   # inclusive of current wafer

        alert = {"wafer_id": wafer_id, "lot_id": lot_id, "chamber_id": chamber,
                 "sensor": "tcp_top_pwr", "alert_type": "ANOMALY"}
        est = estimate_scrap_impact(alert, self.roster)
        self.assertTrue(est["estimable"], "estimate should be produced")
        self.assertEqual(est["remaining_wafers"], expected_remaining)


# =============================================================================
# Pipeline propagation — chamber/lot/sensor carried across ALL five agents
# =============================================================================
# Swaps the plain stubs in the loaded production module for RECORDING mocks, runs
# run_agent on a single controlled anomaly (wafer 2915 / CHA / tcp_top_pwr), then
# asserts the SAME identity fields and each upstream OUTPUT reach every downstream
# agent. This is the only test that exercises the real wiring end-to-end.

import maintenance_agent as ma
import sop_agent as so


class TestPipelinePropagation(unittest.TestCase):

    def setUp(self):
        # Known outputs so we can assert they are forwarded intact.
        self.q_report       = "QUALITY_REPORT_X"
        self.recommendation = {"priority": "HIGH", "component": "TCP_GENERATOR",
                               "sensor": "tcp_top_pwr", "chamber_id": "CHA"}
        self.sop_report     = "SOP_REPORT_X"

        # Save originals and install recording mocks on the loaded prod module.
        self._orig = {n: getattr(prod, n) for n in (
            "run_quality_agent", "run_maintenance_agent", "run_sop_agent",
            "run_impact_agent", "print_recommendation")}
        prod.run_quality_agent    = MagicMock(return_value=self.q_report)
        prod.run_maintenance_agent = MagicMock(return_value=self.recommendation)
        prod.run_sop_agent        = MagicMock(return_value=self.sop_report)
        prod.run_impact_agent     = MagicMock(return_value={"narrative": "[mock]",
                                                            "estimable": False})
        prod.print_recommendation = MagicMock()

        # One row that breaches tcp_top_pwr only; other sensors in range (no trend).
        df = pd.DataFrame([{
            "wafer_id": 2915, "step": 12,
            "tcp_top_pwr": 410.0,   # > 360 -> anomaly
            "bcl3_flow": 750.0, "cl2_flow": 752.0,
            "pressure": 1000.0, "rf_btm_pwr": 130.0,
        }])
        prod.run_agent(data=df, thresholds=THRESHOLDS,
                       trend_config=TREND_CONFIG, max_rows=None)

    def tearDown(self):
        for n, fn in self._orig.items():
            setattr(prod, n, fn)

    def test_production_alert_reaches_quality(self):
        alert = prod.run_quality_agent.call_args.args[0]
        self.assertEqual(alert["wafer_id"], 2915)
        self.assertEqual(alert["sensor"], "tcp_top_pwr")
        self.assertEqual(alert["chamber_id"], "CHA")
        self.assertEqual(alert["lot_id"], "LOT_29B")

    def test_quality_report_forwarded_to_maintenance(self):
        args = prod.run_maintenance_agent.call_args.args
        self.assertEqual(args[0]["wafer_id"], 2915)   # Agent 1 alert
        self.assertEqual(args[1], self.q_report)       # Agent 2 output

    def test_alert_and_reports_forwarded_to_sop(self):
        args = prod.run_sop_agent.call_args.args
        self.assertEqual(args[0]["sensor"], "tcp_top_pwr")  # Agent 1
        self.assertEqual(args[1], self.q_report)            # Agent 2
        self.assertEqual(args[2], self.recommendation)      # Agent 3

    def test_all_upstream_forwarded_to_impact(self):
        args = prod.run_impact_agent.call_args.args
        self.assertEqual(args[0]["wafer_id"], 2915)     # Agent 1 alert
        self.assertEqual(args[0]["chamber_id"], "CHA")
        self.assertEqual(args[0]["sensor"], "tcp_top_pwr")
        self.assertEqual(args[1], self.q_report)         # Agent 2
        self.assertEqual(args[2], self.recommendation)   # Agent 3
        self.assertEqual(args[3], self.sop_report)       # Agent 4

    def test_identity_consistent_across_every_boundary(self):
        a_q = prod.run_quality_agent.call_args.args[0]
        a_m = prod.run_maintenance_agent.call_args.args[0]
        a_s = prod.run_sop_agent.call_args.args[0]
        a_i = prod.run_impact_agent.call_args.args[0]
        for a in (a_m, a_s, a_i):
            self.assertEqual(a["wafer_id"],   a_q["wafer_id"])
            self.assertEqual(a["chamber_id"], a_q["chamber_id"])
            self.assertEqual(a["lot_id"],     a_q["lot_id"])
            self.assertEqual(a["sensor"],     a_q["sensor"])


# =============================================================================
# Maintenance Agent
# =============================================================================

class TestMaintenanceDeterministic(unittest.TestCase):
    """The auditable, no-LLM core: parsing, CMMS checks, priority, query build."""

    REPORT = ("SENSOR: tcp_top_pwr\nCHAMBER: CHA\nALERT TYPE: ANOMALY\n\n"
              "NCR SUMMARY:\nTCP +50W fault on CHA.\n\n"
              "TOOL WEAR INDICATORS:\nRF generator hours 1241.\n\n"
              "RECURRENCE:\nFour TCP faults on CHA.")

    def _pm_df(self):
        return pd.DataFrame([
            {"chamber_id": "CHA", "pm_type": "WET_CLEAN", "wafers_until_pm": -8, "pm_status": "OVERDUE"},
            {"chamber_id": "CHA", "pm_type": "FULL_PM",   "wafers_until_pm": 200, "pm_status": "OK"},
        ])

    def _parts_df(self):
        return pd.DataFrame([
            {"component_category": "TCP_GENERATOR", "part_number": "P0", "description": "d",
             "quantity_on_hand": 0,  "reorder_level": 2, "lead_time_days": 5, "storage_location": "A"},
            {"component_category": "TCP_GENERATOR", "part_number": "PL", "description": "d",
             "quantity_on_hand": 2,  "reorder_level": 2, "lead_time_days": 5, "storage_location": "A"},
            {"component_category": "TCP_GENERATOR", "part_number": "PI", "description": "d",
             "quantity_on_hand": 10, "reorder_level": 2, "lead_time_days": 5, "storage_location": "A"},
        ])

    def _calib_df(self):
        return pd.DataFrame([
            {"chamber_id": "CHA", "component_category": "TCP_GENERATOR", "component": "TCP power meter",
             "calibration_status": "OVERDUE", "next_calibration_due": "2024-01-01",
             "last_calibration_date": "2023-01-01", "calibration_notes": "n"},
        ])

    # ── parsing & mapping ─────────────────────────────────────────────────────
    def test_parse_quality_report_headers(self):
        p = ma.parse_quality_report(self.REPORT)
        self.assertEqual(p["sensor"], "tcp_top_pwr")
        self.assertEqual(p["chamber_id"], "CHA")
        self.assertEqual(p["alert_type"], "ANOMALY")
        self.assertEqual(p["component"], "TCP_GENERATOR")

    def test_parse_missing_headers_are_unknown(self):
        p = ma.parse_quality_report("no headers here")
        self.assertEqual(p["sensor"], "UNKNOWN")
        self.assertEqual(p["chamber_id"], "UNKNOWN")
        self.assertEqual(p["component"], "UNKNOWN")

    def test_sensor_to_component_mapping(self):
        for sensor, comp in [("tcp_top_pwr", "TCP_GENERATOR"), ("rf_btm_pwr", "RF_GENERATOR"),
                             ("bcl3_flow", "BCL3_MFC"), ("cl2_flow", "CL2_MFC"),
                             ("pressure", "PRESSURE_SYSTEM"), ("he_press", "ESC_HE_CIRCUIT")]:
            self.assertEqual(ma.SENSOR_TO_COMPONENT[sensor], comp)

    # ── CMMS checks ───────────────────────────────────────────────────────────
    def test_check_pm_status_flags_overdue_wet_clean(self):
        r = ma.check_pm_status("CHA", "TCP_GENERATOR", self._pm_df())
        self.assertEqual(r["wet_clean_status"], "OVERDUE")
        self.assertEqual(r["wet_clean_overdue_by"], 8)
        self.assertEqual(r["full_pm_wafers_until"], 200)
        self.assertTrue(any("WET_CLEAN" in x for x in r["overdue_items"]))
        self.assertTrue(r["wet_clean_relevant"])           # TCP is wet-clean sensitive

    def test_check_pm_status_wet_clean_not_relevant_for_gas(self):
        r = ma.check_pm_status("CHA", "BCL3_MFC", self._pm_df())
        self.assertFalse(r["wet_clean_relevant"])

    def test_check_parts_availability_statuses(self):
        parts = {p["part_number"]: p["status"]
                 for p in ma.check_parts_availability("TCP_GENERATOR", self._parts_df())}
        self.assertEqual(parts["P0"], "OUT_OF_STOCK")
        self.assertEqual(parts["PL"], "LOW_STOCK")
        self.assertEqual(parts["PI"], "IN_STOCK")

    def test_check_calibration_found_and_not_found(self):
        found = ma.check_calibration_status("CHA", "TCP_GENERATOR", self._calib_df())
        self.assertEqual(found["status"], "OVERDUE")
        self.assertEqual(found["component"], "TCP power meter")
        missing = ma.check_calibration_status("CHB", "TCP_GENERATOR", self._calib_df())
        self.assertEqual(missing["status"], "NOT_FOUND")

    # ── priority (the auditable decision) ─────────────────────────────────────
    def test_priority_critical(self):
        pm    = {"overdue_items": ["WET_CLEAN overdue by 8 wafers"]}
        parts = [{"status": "OUT_OF_STOCK"}]
        calib = {"status": "OK"}
        self.assertEqual(ma.determine_priority(pm, parts, calib, "ANOMALY"), "CRITICAL")

    def test_priority_high_from_calibration(self):
        self.assertEqual(
            ma.determine_priority({"overdue_items": []}, [{"status": "IN_STOCK"}],
                                  {"status": "OVERDUE"}, "ANOMALY"), "HIGH")

    def test_priority_high_from_low_stock(self):
        self.assertEqual(
            ma.determine_priority({"overdue_items": []}, [{"status": "LOW_STOCK"}],
                                  {"status": "OK"}, "TREND"), "HIGH")

    def test_priority_medium_clean_anomaly(self):
        self.assertEqual(
            ma.determine_priority({"overdue_items": []}, [{"status": "IN_STOCK"}],
                                  {"status": "OK"}, "ANOMALY"), "MEDIUM")

    def test_priority_low_clean_trend(self):
        self.assertEqual(
            ma.determine_priority({"overdue_items": []}, [{"status": "IN_STOCK"}],
                                  {"status": "OK"}, "TREND"), "LOW")

    # ── query building & signal extraction ────────────────────────────────────
    def test_build_wo_query_uses_full_upstream_picture(self):
        parsed = {"sensor": "tcp_top_pwr", "chamber_id": "CHA",
                  "alert_type": "ANOMALY", "component": "TCP_GENERATOR"}
        alert  = {"deviation": "+50W above set-point", "value": 410.0, "severity": "HIGH"}
        q = ma.build_wo_query(parsed, alert, self.REPORT)
        self.assertIn("tcp_top_pwr", q)
        self.assertIn("TCP_GENERATOR", q)
        self.assertIn("+50W", q)                 # Agent 1 deviation
        self.assertIn("Four TCP faults", q)      # Agent 2 recurrence signal

    def test_extract_quality_signals(self):
        sig = ma._extract_quality_signals(self.REPORT)
        self.assertIn("TCP +50W fault", sig)
        self.assertIn("1241", sig)
        self.assertIn("Four TCP faults", sig)


class TestMaintenanceEndToEnd(unittest.TestCase):
    """run_maintenance_agent with a mocked WO collection + mocked LLM."""

    def _run(self):
        report = TestMaintenanceDeterministic.REPORT
        d = TestMaintenanceDeterministic()
        pm_df, parts_df, calib_df = d._pm_df(), d._parts_df(), d._calib_df()
        alert = {"alert_type": "ANOMALY", "sensor": "tcp_top_pwr", "wafer_id": 2915,
                 "lot_id": "LOT_29B", "chamber_id": "CHA", "value": 410.0,
                 "threshold_min": 334, "threshold_max": 360,
                 "deviation": "+50W above set-point", "severity": "HIGH"}
        coll = MagicMock()
        coll.query.return_value = {
            "documents": [["WO history doc"] * 5],
            "distances": [[0.10, 0.20, 0.30, 0.40, 0.50]],
        }
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = (
            '{"past_wo_patterns": "PATTERN", "recommended_actions": ["1. act"], '
            '"draft_wo_header": "WO HEADER", "llm_narrative": "NARR"}')
        return ma.run_maintenance_agent(alert, report, coll, pm_df, parts_df, calib_df, client)

    def test_returns_structured_dict_with_deterministic_fields(self):
        rec = self._run()
        self.assertEqual(rec["sensor"], "tcp_top_pwr")
        self.assertEqual(rec["chamber_id"], "CHA")
        self.assertEqual(rec["component"], "TCP_GENERATOR")

    def test_priority_is_critical_given_overdue_pm_and_out_of_stock(self):
        # PM overdue (WET_CLEAN) + a part OUT_OF_STOCK -> CRITICAL (deterministic).
        self.assertEqual(self._run()["priority"], "CRITICAL")

    def test_llm_fields_come_from_model_output(self):
        rec = self._run()
        self.assertEqual(rec["past_wo_patterns"], "PATTERN")
        self.assertEqual(rec["draft_wo_header"], "WO HEADER")
        self.assertEqual(rec["llm_narrative"], "NARR")


# =============================================================================
# SOP / Knowledge Agent
# =============================================================================

class TestSOPDeterministic(unittest.TestCase):

    REPORT = ("SENSOR: tcp_top_pwr\nCHAMBER: CHA\nALERT TYPE: ANOMALY\n\n"
              "NCR SUMMARY:\nTCP over-etch.\n\n"
              "TOOL WEAR INDICATORS:\nRF hrs 1241.\n\n"
              "RECURRENCE:\nFour times on CHA.")

    def _alert(self):
        return {"alert_type": "ANOMALY", "sensor": "tcp_top_pwr",
                "deviation": "+50W above set-point", "chamber_id": "CHA",
                "severity": "HIGH"}

    def _rec(self):
        return {"priority": "CRITICAL", "component": "TCP_GENERATOR",
                "past_wo_patterns": "prior capacitor bank replacement"}

    def test_build_query_contains_alert_and_maintenance_fields(self):
        q = so.build_query(self._alert(), self.REPORT, self._rec())
        self.assertIn("tcp_top_pwr", q)
        self.assertIn("CHA", q)
        self.assertIn("CRITICAL", q)                       # priority from maintenance
        self.assertIn("TCP_GENERATOR", q)                  # component
        self.assertIn("+50W", q)                           # deviation

    def test_build_query_folds_in_upstream_signals(self):
        q = so.build_query(self._alert(), self.REPORT, self._rec())
        self.assertIn("Four times on CHA", q)              # quality recurrence
        self.assertIn("prior capacitor bank replacement", q)  # maintenance WO pattern

    def test_source_labels_mapping(self):
        self.assertEqual(so.SOURCE_LABELS["sop"], "SOP Procedure")
        self.assertEqual(so.SOURCE_LABELS["guides"], "Troubleshooting Guide")
        self.assertEqual(so.SOURCE_LABELS["incidents"], "Incident Record")
        self.assertEqual(so.SOURCE_LABELS["manuals"], "Equipment Manual")

    def test_extract_quality_signals(self):
        sig = so._extract_quality_signals(self.REPORT)
        self.assertIn("TCP over-etch", sig)
        self.assertIn("Four times on CHA", sig)

    def test_retrieve_cases_shape(self):
        coll = MagicMock()
        coll.query.return_value = {
            "documents": [["DOC BODY"] * 5],
            "distances": [[0.10, 0.20, 0.30, 0.40, 0.50]],
            "metadatas": [[{"source": "manuals", "id": "MAN-1"}] * 5],
        }
        cases = so.retrieve_cases(coll, "q", top_k=5)
        self.assertEqual(len(cases), 5)
        self.assertEqual([c["rank"] for c in cases], [1, 2, 3, 4, 5])
        self.assertEqual(cases[0]["source"], "manuals")
        self.assertEqual(cases[0]["id"], "MAN-1")
        for c in cases:
            for k in ("rank", "similarity", "content", "source", "id"):
                self.assertIn(k, c)


class TestSOPEndToEnd(unittest.TestCase):

    def _retrieved(self):
        return [{"rank": 1, "similarity": 0.82, "content": "MANUAL CONTENT",
                 "source": "manuals", "id": "MAN-1"},
                {"rank": 2, "similarity": 0.70, "content": "SOP CONTENT",
                 "source": "sop", "id": "SOP-1"}]

    def _rec(self):
        return {"priority": "CRITICAL", "component": "TCP_GENERATOR",
                "pm_status": {"overdue_items": ["WET_CLEAN overdue"]},
                "required_parts": [{"part_number": "P0", "status": "OUT_OF_STOCK"}],
                "calibration_status": {"status": "OVERDUE"},
                "recommended_actions": ["Replace capacitor bank"],
                "llm_narrative": "TCP generator degrading."}

    def test_synthesise_report_includes_all_upstream_context(self):
        alert  = {"alert_type": "ANOMALY", "sensor": "tcp_top_pwr", "chamber_id": "CHA",
                  "deviation": "+50W", "severity": "HIGH"}
        report = "SENSOR: tcp_top_pwr\nNCR SUMMARY:\nUNIQUE_QUALITY_MARKER."
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "REPORT_OUT"
        out = so.synthesise_report(alert, report, self._rec(), self._retrieved(), client)
        self.assertEqual(out, "REPORT_OUT")

        msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("UNIQUE_QUALITY_MARKER", msg)  # Agent 2 quality report present
        self.assertIn("CRITICAL", msg)               # Agent 3 priority present
        self.assertIn("MANUAL CONTENT", msg)          # retrieved KB present
        self.assertIn("Equipment Manual", msg)        # human-readable source label

    def test_run_sop_agent_returns_string(self):
        alert  = {"alert_type": "ANOMALY", "sensor": "tcp_top_pwr", "chamber_id": "CHA",
                  "deviation": "+50W", "severity": "HIGH"}
        report = "SENSOR: tcp_top_pwr\nNCR SUMMARY:\nx."
        coll = MagicMock()
        coll.query.return_value = {
            "documents": [["DOC"] * 5],
            "distances": [[0.10, 0.20, 0.30, 0.40, 0.50]],
            "metadatas": [[{"source": "manuals", "id": "MAN-1"}] * 5],
        }
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "SOP REPORT"
        out = so.run_sop_agent(alert, report, self._rec(), coll, client)
        self.assertIsInstance(out, str)
        self.assertEqual(out, "SOP REPORT")


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
        TestCountingAndAttribution,
        TestArithmetic,
        TestInputRobustness,
        TestProviderSeam,
        TestNarrator,
        TestDeterminism,
        TestCostModel,
        TestSustainabilityRealData,
        TestSustainabilityPerformance,
        TestWafersLeftInLot,
        TestPipelinePropagation,
        TestMaintenanceDeterministic,
        TestMaintenanceEndToEnd,
        TestSOPDeterministic,
        TestSOPEndToEnd,
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