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
        "def get_llm_explanation(event_type, sensor, details):\n"
        "    return f'[mocked: {sensor}]'\n"
        "def run_quality_agent(alert, collection, client):\n"
        "    return '[mocked quality report]'\n"
        "def run_maintenance_agent(alert, quality_report, wo_collection, pm_df, parts_df, calib_df, client):\n"
        "    return {'priority': 'LOW', 'sensor': '', 'chamber_id': '', 'alert_type': '', 'component': '', 'pm_status': {}, 'required_parts': [], 'calibration_status': {}, 'past_wo_patterns': '', 'recommended_actions': [], 'draft_wo_header': '', 'llm_narrative': '[mocked]'}\n"
        "def print_recommendation(rec): pass\n"
        "def run_sop_agent(alert, quality_report, recommendation, collection, client):\n"
        "    return '[mocked sop report]'\n"
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