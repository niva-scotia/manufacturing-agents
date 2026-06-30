"""
Maintenance Agent — Hybrid CMMS System
=======================================
Pipeline position:
  Production Agent → Quality Agent → [THIS AGENT]

Design philosophy (mirrors the Production Agent):
  Python does the deterministic, auditable checks.
  The LLM only writes language.

  - PM overdue status, spare-parts availability, calibration status, and the
    final priority are all computed in plain Python from the CMMS data.
    These are FACTS — the LLM cannot invent or override them.
  - Historical work orders are retrieved by semantic similarity (RAG), exactly
    like the Quality Agent retrieves NCRs.
  - The LLM synthesises the recommendation narrative, corrective-action steps,
    and draft work-order header from the facts and retrieved WOs it is given.

Inputs / outputs:
  Input  : the Quality Agent's structured text report (one fault at a time)
  Output : a structured dict — ready for the web app UI to render directly
           (priority badge, PM status, parts table, action checklist, draft WO)

Vector database:
  ChromaDB (persistent) — same chroma_db/ folder as the Quality Agent, but a
  separate collection ("cmms_work_orders"). Built once, survives restarts.
"""

import os
import re
import json
import pandas as pd
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
from chromadb.api.types import EmbeddingFunction

# Load .env
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

CMMS_WO_PATH     = str(_ROOT / "data/cmms_work_orders.csv")
CMMS_PM_PATH     = str(_ROOT / "data/cmms_pm_schedule.csv")
CMMS_PARTS_PATH  = str(_ROOT / "data/cmms_spare_parts.csv")
CMMS_CALIB_PATH  = str(_ROOT / "data/cmms_calibration.csv")

CHROMA_DB_PATH   = str(_ROOT / "chroma_db")
COLLECTION_NAME  = "cmms_work_orders"    # but a SEPARATE collection
EMBEDDING_MODEL  = "text-embedding-3-small"
TOP_K            = 5
OPENAI_MODEL     = os.getenv("AZURE_DEPLOYMENT")

# Maps a Production Agent sensor name to the CMMS component category.
# Drives every deterministic CMMS lookup (PM, parts, calibration).
SENSOR_TO_COMPONENT = {
    "tcp_top_pwr": "TCP_GENERATOR",
    "rf_btm_pwr":  "RF_GENERATOR",
    "bcl3_flow":   "BCL3_MFC",
    "cl2_flow":    "CL2_MFC",
    "pressure":    "PRESSURE_SYSTEM",
    "he_press":    "ESC_HE_CIRCUIT",
}

# Faults on these sensors degrade the chamber via polymer buildup, so an overdue
# wet clean is a relevant contributing factor for them.
WET_CLEAN_SENSITIVE = {"TCP_GENERATOR", "RF_GENERATOR", "PRESSURE_SYSTEM"}


class OpenAIEmbedder(EmbeddingFunction):
    def __init__(self):
        self._client = AzureOpenAI(
            api_key        = os.getenv("EMBEDDING_MODEL_KEY"),
            azure_endpoint = os.getenv("EMBEDDING_ENDPOINT"),
            api_version    = os.getenv("AZURE_API_VERSION"),
        )
        self._model = EMBEDDING_MODEL

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            raise ValueError("Empty input passed to embedder")
        response = self._client.embeddings.create(
            input=input,
            model=self._model
        )
        embeddings = [r.embedding for r in response.data]
        if not embeddings:
            raise ValueError(f"Azure returned empty embeddings for input: {input[:1]}")
        return embeddings


# ── ChromaDB vector store (historical work orders) ─────────────────────────────

def build_index(csv_path: str) -> chromadb.Collection:
    """
    Builds (or loads) the persistent ChromaDB collection of historical work
    orders. The rich text columns (fault description, root cause, work performed,
    resolution) are what gets embedded and retrieved.
    """
    # Guard: never touch the Quality Agent's collection by mistake.
    assert COLLECTION_NAME == "cmms_work_orders", \
        "Maintenance Agent must only build the 'cmms_work_orders' collection."

    embed_fn = OpenAIEmbedder()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    df = pd.read_csv(csv_path)

    if collection.count() == len(df):
        print(f"[VectorStore] Loaded {collection.count()} existing work orders from disk.")
        return collection

    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    documents, ids, metadatas = [], [], []
    for idx, row in df.iterrows():
        chunk = (
            f"WO: {row['WO_number']} | CHAMBER: {row['chamber_id']} | "
            f"COMPONENT: {row['equipment_component']}\n"
            f"FAULT TYPE: {row['fault_type']} | DATE: {row['fault_date']}\n"
            f"FAULT DESCRIPTION:\n{row['fault_description']}\n\n"
            f"ROOT CAUSE:\n{row['root_cause']}\n\n"
            f"WORK PERFORMED:\n{row['work_description']}\n\n"
            f"RESOLUTION:\n{row['resolution']}\n\n"
            f"PARTS USED: {row['parts_used']}\n"
            f"TIME TO REPAIR: {row['time_to_repair_hrs']} hrs | "
            f"DOWNTIME: {row['downtime_hours']} hrs"
        )
        documents.append(chunk)
        ids.append(f"wo_{idx}")
        metadatas.append({
            "fault_type":         str(row["fault_type"]),
            "chamber_id":         str(row["chamber_id"]),
            "equipment_component": str(row["equipment_component"]),
        })

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"[VectorStore] Indexed {len(documents)} work orders and saved to disk.")

    return collection


# ── Parse the Quality Agent report ──────────────────────────────────────────────

def parse_quality_report(report: str) -> dict:
    """
    Extracts the header fields from the Quality Agent's structured text output.
    The Quality Agent guarantees these on the first lines:
        SENSOR: ...
        CHAMBER: ...
        ALERT TYPE: ...
    Pure regex — no LLM. Falls back to "UNKNOWN" if a field is missing.
    """
    def grab(label: str) -> str:
        m = re.search(rf"^{label}\s*:\s*(.+)$", report, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else "UNKNOWN"

    sensor     = grab("SENSOR")
    chamber    = grab("CHAMBER")
    alert_type = grab("ALERT TYPE").upper()

    if sensor == "UNKNOWN" or chamber == "UNKNOWN":
        print("[Maintenance Agent] WARNING: could not fully parse Quality report "
              f"(sensor={sensor}, chamber={chamber}).")

    component = SENSOR_TO_COMPONENT.get(sensor, "UNKNOWN")

    return {
        "sensor":     sensor,
        "chamber_id": chamber,
        "alert_type": alert_type,
        "component":  component,
    }


# ── Deterministic CMMS checks (no LLM) ──────────────────────────────────────────

def check_pm_status(chamber_id: str,
                    component: str,
                    pm_df: pd.DataFrame) -> dict:
    """
    Reports PM status for the chamber. Always includes the wet clean (relevant to
    every chamber-condition fault) plus the full PM, and flags anything overdue.
    Pure Python.
    """
    rows = pm_df[pm_df["chamber_id"] == chamber_id]

    result = {
        "chamber_id":            chamber_id,
        "wet_clean_status":      "UNKNOWN",
        "wet_clean_overdue_by":  0,        # positive = overdue by N wafers
        "full_pm_status":        "UNKNOWN",
        "full_pm_wafers_until":  None,
        "overdue_items":         [],
        "wet_clean_relevant":    component in WET_CLEAN_SENSITIVE,
    }

    for _, r in rows.iterrows():
        pm_type = r["pm_type"]
        until   = int(r["wafers_until_pm"])          # negative = overdue
        status  = str(r["pm_status"])

        if pm_type == "WET_CLEAN":
            result["wet_clean_status"]     = status
            result["wet_clean_overdue_by"] = max(0, -until)
        elif pm_type == "FULL_PM":
            result["full_pm_status"]       = status
            result["full_pm_wafers_until"] = until

        if status == "OVERDUE":
            result["overdue_items"].append(
                f"{pm_type} overdue by {abs(until)} wafers"
            )

    return result


def check_parts_availability(component: str,
                             parts_df: pd.DataFrame) -> list[dict]:
    """
    Returns the spare parts for the relevant component category, each tagged with
    a stock status. Pure Python.
        OUT_OF_STOCK : qty == 0
        LOW_STOCK    : qty <= reorder_level
        IN_STOCK     : otherwise
    """
    rows = parts_df[parts_df["component_category"] == component]

    parts = []
    for _, r in rows.iterrows():
        qty     = int(r["quantity_on_hand"])
        reorder = int(r["reorder_level"])
        if qty == 0:
            status = "OUT_OF_STOCK"
        elif qty <= reorder:
            status = "LOW_STOCK"
        else:
            status = "IN_STOCK"

        parts.append({
            "part_number":      r["part_number"],
            "description":      r["description"],
            "quantity_on_hand": qty,
            "status":           status,
            "lead_time_days":   int(r["lead_time_days"]),
            "storage_location": r["storage_location"],
        })

    return parts


def check_calibration_status(chamber_id: str,
                             component: str,
                             calib_df: pd.DataFrame) -> dict:
    """
    Returns the calibration record for the relevant component on this chamber.
    Pure Python — the status is read straight from the CMMS data.
    """
    rows = calib_df[
        (calib_df["chamber_id"] == chamber_id) &
        (calib_df["component_category"] == component)
    ]

    if rows.empty:
        return {
            "component":  "UNKNOWN",
            "status":     "NOT_FOUND",
            "next_due":   "N/A",
            "last_date":  "N/A",
            "notes":      "No calibration record found for this component.",
        }

    r = rows.iloc[0]
    return {
        "component":  r["component"],
        "status":     str(r["calibration_status"]),
        "next_due":   str(r["next_calibration_due"]),
        "last_date":  str(r["last_calibration_date"]),
        "notes":      str(r["calibration_notes"]),
    }


def determine_priority(pm_status: dict,
                       parts_status: list[dict],
                       calib_status: dict,
                       alert_type: str) -> str:
    """
    Deterministic, auditable priority. The LLM never sets this.
        CRITICAL : a PM is overdue AND a required part is out of stock
        HIGH     : PM overdue, OR calibration overdue, OR any part out/low stock
        MEDIUM   : all checks pass and this is an ANOMALY
        LOW      : all checks pass and this is a TREND
    """
    pm_overdue    = len(pm_status.get("overdue_items", [])) > 0
    calib_overdue = calib_status.get("status") == "OVERDUE"
    out_of_stock  = any(p["status"] == "OUT_OF_STOCK" for p in parts_status)
    low_stock     = any(p["status"] == "LOW_STOCK"    for p in parts_status)

    if pm_overdue and out_of_stock:
        return "CRITICAL"
    if pm_overdue or calib_overdue or out_of_stock or low_stock:
        return "HIGH"
    if alert_type == "ANOMALY":
        return "MEDIUM"
    return "LOW"


# ── Build retrieval query + retrieve work orders ────────────────────────────────

def build_wo_query(parsed: dict) -> str:
    """
    Builds the ChromaDB query for historical work-order retrieval. Focuses on the
    component, sensor, and chamber rather than copying the Quality report verbatim.
    """
    return (
        f"{parsed['alert_type']} on sensor {parsed['sensor']} in chamber "
        f"{parsed['chamber_id']}. Component: {parsed['component']}. "
        f"Looking for past work orders, root cause, corrective work performed, "
        f"parts replaced, and time to repair for {parsed['sensor']} faults on a "
        f"plasma etch chamber. Recurrence and resolution history."
    )


def retrieve_work_orders(collection: chromadb.Collection,
                         query: str,
                         top_k: int = TOP_K) -> list[dict]:
    """
    Queries ChromaDB for the top_k most semantically similar work orders.
    Same structure as the Quality Agent's retrieve_cases().
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances"]
    )

    wos = []
    for i in range(len(results["documents"][0])):
        wos.append({
            "rank":       i + 1,
            "similarity": round(1 - results["distances"][0][i], 4),
            "content":    results["documents"][0][i],
        })

    return wos


# ── System prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Maintenance Agent in a semiconductor manufacturing pipeline for a
plasma etch chamber.

You receive:
  1. A Quality Agent report — sensor, chamber, NCR summary, tool wear indicators,
     recurrence.
  2. CMMS FACTS that have ALREADY been computed deterministically before you were
     called: PM status, spare-parts availability, calibration status, and the
     final priority.
  3. Retrieved historical work orders — the most relevant past WOs, found by
     semantic similarity.

Your job is to synthesise a maintenance recommendation.

Rules:
  - Priority has already been determined by the system. Do NOT change it.
  - PM wafer counts, parts quantities, lead times, and calibration dates are
    FACTS. Use them exactly as given. Never invent or alter a number.
  - Only use information from the CMMS facts and the retrieved work orders.
    Do not reference procedures, part numbers, or failure modes not present in
    the input.
  - Recommended actions must be concrete, executable steps. Each step should
    reference a specific component, part number, measurement, or PM task drawn
    from the facts or retrieved WOs.
  - The draft WO header must reflect the actual chamber, component, fault, and
    priority provided.

Output ONLY valid JSON (no markdown, no code fences) with exactly these keys:

{
  "past_wo_patterns":    "2-3 sentences: what the retrieved WOs reveal about
                          recurrence and the resolutions that worked.",
  "recommended_actions": ["1. ...", "2. ...", "3. ...", "4. ...", "5. ..."],
  "draft_wo_header":     "4-6 lines of plain text for a new work order.",
  "llm_narrative":       "3-5 sentence prose recommendation for the supervisor."
}
""".strip()


# ── Synthesise recommendation (the only LLM call) ───────────────────────────────

def _strip_code_fence(text: str) -> str:
    """GPT sometimes wraps JSON in ```json ... ``` — remove it before parsing."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def synthesise_recommendation(parsed: dict,
                              pm_status: dict,
                              parts_status: list[dict],
                              calib_status: dict,
                              retrieved_wos: list[dict],
                              quality_report: str,
                              priority: str,
                              client: AzureOpenAI) -> dict:
    """
    Sends every deterministic fact + the retrieved WOs to the LLM and asks for the
    recommendation narrative/actions/draft WO. Returns the full output dict.
    The deterministic fields (priority, pm_status, parts, calibration) are injected
    by the caller — they are never taken from the LLM.
    """
    wo_block = "\n\n---\n\n".join([
        f"RETRIEVED WORK ORDER {w['rank']} "
        f"(similarity: {w['similarity']}):\n\n{w['content']}"
        for w in retrieved_wos
    ])

    parts_lines = "\n".join([
        f"  - {p['part_number']} ({p['description']}): {p['status']}, "
        f"qty {p['quantity_on_hand']}, lead time {p['lead_time_days']} days, "
        f"loc {p['storage_location']}"
        for p in parts_status
    ]) or "  - No parts mapped to this component category."

    pm_lines = (
        f"  Wet clean : {pm_status['wet_clean_status']}"
        + (f" (OVERDUE by {pm_status['wet_clean_overdue_by']} wafers)"
           if pm_status['wet_clean_overdue_by'] > 0 else "")
        + (" [relevant to this fault type]"
           if pm_status['wet_clean_relevant'] else "")
        + "\n"
        f"  Full PM   : {pm_status['full_pm_status']}"
        + (f" ({pm_status['full_pm_wafers_until']} wafers until due)"
           if pm_status['full_pm_wafers_until'] is not None else "")
    )

    user_msg = (
        f"QUALITY AGENT REPORT\n{'='*50}\n{quality_report}\n\n"
        f"{'='*50}\n"
        f"CMMS FACTS (deterministic — use verbatim)\n{'='*50}\n"
        f"Priority (FINAL, do not change): {priority}\n"
        f"Sensor / Component: {parsed['sensor']} / {parsed['component']}\n"
        f"Chamber: {parsed['chamber_id']}\n\n"
        f"PM STATUS:\n{pm_lines}\n\n"
        f"SPARE PARTS AVAILABILITY:\n{parts_lines}\n\n"
        f"CALIBRATION STATUS:\n"
        f"  Component: {calib_status['component']}\n"
        f"  Status   : {calib_status['status']}\n"
        f"  Last cal : {calib_status['last_date']} | Next due: {calib_status['next_due']}\n"
        f"  Notes    : {calib_status['notes']}\n\n"
        f"{'='*50}\n"
        f"RETRIEVED WORK-ORDER HISTORY "
        f"({len(retrieved_wos)} most relevant past WOs):\n\n{wo_block}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ]
    )

    raw = _strip_code_fence(response.choices[0].message.content)
    try:
        llm = json.loads(raw)
    except json.JSONDecodeError:
        print("[Maintenance Agent] WARNING: LLM did not return valid JSON. "
              "Falling back to raw text in llm_narrative.")
        llm = {
            "past_wo_patterns":    "",
            "recommended_actions": [],
            "draft_wo_header":     "",
            "llm_narrative":       raw,
        }

    # Assemble the final output dict. Deterministic fields override anything the
    # LLM might have produced for them.
    return {
        "priority":            priority,             # from determine_priority()
        "sensor":              parsed["sensor"],
        "chamber_id":          parsed["chamber_id"],
        "alert_type":          parsed["alert_type"],
        "component":           parsed["component"],
        "pm_status":           pm_status,
        "required_parts":      parts_status,
        "calibration_status":  calib_status,
        "past_wo_patterns":    llm.get("past_wo_patterns", ""),
        "recommended_actions": llm.get("recommended_actions", []),
        "draft_wo_header":     llm.get("draft_wo_header", ""),
        "llm_narrative":       llm.get("llm_narrative", ""),
    }


# ── Main pipeline function ──────────────────────────────────────────────────────

def run_maintenance_agent(quality_report: str,
                          wo_collection: chromadb.Collection,
                          pm_df: pd.DataFrame,
                          parts_df: pd.DataFrame,
                          calib_df: pd.DataFrame,
                          client: AzureOpenAI) -> dict:
    """
    Full Maintenance Agent pipeline. Called by the Production Agent after the
    Quality Agent produces its report.

    Args:
      quality_report : structured text from the Quality Agent (one fault)
      wo_collection  : ChromaDB collection of historical work orders
      pm_df          : CMMS PM schedule
      parts_df       : CMMS spare parts inventory
      calib_df       : CMMS calibration records
      client         : AzureOpenAI client

    Returns:
      A structured recommendation dict (ready for the web app UI).
    """
    parsed = parse_quality_report(quality_report)
    print(f"\n[Maintenance Agent] Fault: {parsed['sensor']} "
          f"({parsed['component']}) on {parsed['chamber_id']} "
          f"[{parsed['alert_type']}]")

    # Deterministic CMMS checks
    pm_status    = check_pm_status(parsed["chamber_id"], parsed["component"], pm_df)
    parts_status = check_parts_availability(parsed["component"], parts_df)
    calib_status = check_calibration_status(parsed["chamber_id"], parsed["component"], calib_df)
    priority     = determine_priority(pm_status, parts_status, calib_status, parsed["alert_type"])

    print(f"[Maintenance Agent] Priority: {priority} | "
          f"PM overdue: {pm_status['overdue_items'] or 'none'} | "
          f"Calibration: {calib_status['status']}")

    # RAG retrieval of historical work orders
    query     = build_wo_query(parsed)
    retrieved = retrieve_work_orders(wo_collection, query, top_k=TOP_K)
    print(f"[Maintenance Agent] Retrieved {len(retrieved)} work orders. "
          f"Best similarity: {retrieved[0]['similarity']:.4f}")

    # LLM synthesis
    return synthesise_recommendation(
        parsed, pm_status, parts_status, calib_status,
        retrieved, quality_report, priority, client
    )


# ── Pretty printer (console demo) ───────────────────────────────────────────────

def print_recommendation(rec: dict) -> None:
    print(f"\n{'='*70}")
    print(f"MAINTENANCE RECOMMENDATION  —  PRIORITY: {rec['priority']}")
    print(f"{'='*70}")
    print(f"Fault     : {rec['sensor']} ({rec['component']}) on "
          f"{rec['chamber_id']} [{rec['alert_type']}]")

    pm = rec["pm_status"]
    print(f"\nPM STATUS:")
    print(f"  Wet clean : {pm['wet_clean_status']}"
          + (f" — OVERDUE by {pm['wet_clean_overdue_by']} wafers"
             if pm['wet_clean_overdue_by'] > 0 else ""))
    print(f"  Full PM   : {pm['full_pm_status']}"
          + (f" ({pm['full_pm_wafers_until']} wafers until due)"
             if pm['full_pm_wafers_until'] is not None else ""))

    cal = rec["calibration_status"]
    print(f"\nCALIBRATION: {cal['component']} — {cal['status']} "
          f"(next due {cal['next_due']})")

    print(f"\nREQUIRED PARTS:")
    for p in rec["required_parts"]:
        print(f"  [{p['status']:<12}] {p['part_number']:<18} "
              f"qty {p['quantity_on_hand']} | lead {p['lead_time_days']}d | "
              f"{p['description']}")

    print(f"\nPAST WO PATTERNS:\n  {rec['past_wo_patterns']}")

    print(f"\nRECOMMENDED ACTIONS:")
    for action in rec["recommended_actions"]:
        print(f"  {action}")

    print(f"\nDRAFT WORK ORDER:\n{rec['draft_wo_header']}")

    print(f"\nNARRATIVE:\n  {rec['llm_narrative']}")
    print(f"{'='*70}\n")


# ── Entry point (standalone demo / test) ────────────────────────────────────────

if __name__ == "__main__":

    wo_collection = build_index(CMMS_WO_PATH)
    pm_df    = pd.read_csv(CMMS_PM_PATH)
    parts_df = pd.read_csv(CMMS_PARTS_PATH)
    calib_df = pd.read_csv(CMMS_CALIB_PATH)

    client = AzureOpenAI(
        api_key        = os.getenv("AZURE_API_KEY"),
        azure_endpoint = os.getenv("AZURE_ENDPOINT"),
        api_version    = os.getenv("AZURE_API_VERSION"),
    )

    # Sample Quality Agent reports — same shape as quality_agent.py produces.
    quality_reports = [
        # Case 1: TCP anomaly on CHA — should hit the overdue wet clean,
        # overdue TCP calibration, and LOW_STOCK capacitor bank → HIGH priority.
        (
            "SENSOR: tcp_top_pwr\n"
            "CHAMBER: CHA\n"
            "ALERT TYPE: ANOMALY\n\n"
            "NCR SUMMARY:\n"
            "Wafer 2915 recorded a TCP Top Power deviation of +50W above set-point. "
            "Over-etch and CD widening observed. Root cause logged as TCP generator "
            "impedance matching fault. This sensor has deviated repeatedly on CHA.\n\n"
            "TOOL WEAR INDICATORS:\n"
            "RF generator hours ~1241 at time of past faults. Multiple TCP work "
            "orders raised on CHA.\n\n"
            "RECURRENCE:\n"
            "TCP power deviation has appeared four times on CHA in the retrieved records."
        ),
        # Case 2: RF anomaly on CHA — RF generator module is OUT_OF_STOCK → HIGH.
        (
            "SENSOR: rf_btm_pwr\n"
            "CHAMBER: CHA\n"
            "ALERT TYPE: ANOMALY\n\n"
            "NCR SUMMARY:\n"
            "Wafer 3122 recorded an RF Bottom Power deviation of +8W above set-point, "
            "with elevated RF Phase Error. Root cause logged as RF matching network "
            "capacitor drift.\n\n"
            "TOOL WEAR INDICATORS:\n"
            "RF matching instability noted across multiple lots on CHA.\n\n"
            "RECURRENCE:\n"
            "RF power deviation has appeared twice on CHA in the retrieved records."
        ),
        # Case 3: pressure trend on CHB — everything current → LOW priority.
        (
            "SENSOR: pressure\n"
            "CHAMBER: CHB\n"
            "ALERT TYPE: TREND\n\n"
            "NCR SUMMARY:\n"
            "Chamber pressure trending slowly toward the upper control limit. No "
            "defects yet. Possible early APC valve drift.\n\n"
            "TOOL WEAR INDICATORS:\n"
            "None found in retrieved records for CHB.\n\n"
            "RECURRENCE:\n"
            "First pressure trend recorded on CHB in the retrieved records."
        ),
    ]

    for i, report in enumerate(quality_reports, 1):
        print(f"\n{'#'*70}")
        print(f"TEST CASE {i}/{len(quality_reports)}")
        print(f"{'#'*70}")
        rec = run_maintenance_agent(report, wo_collection, pm_df, parts_df, calib_df, client)
        print_recommendation(rec)
