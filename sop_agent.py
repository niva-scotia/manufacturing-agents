"""
SOP / Knowledge Agent — RAG System
====================================
Pipeline position:
  Production Agent → Quality Agent → Maintenance Agent → [THIS AGENT]

Embedding strategy:
  Azure text-embedding-3-small
  — same model used by quality_agent.py for consistent vector space

Vector database:
  ChromaDB (persistent) — vectors saved to disk in chroma_db/
  — separate collection ("sop_knowledge_base") from quality_records
  — built once, survives restarts

Knowledge base:
  4 CSVs in data/:
    sop_procedures.csv        — 15 SOPs (one per sensor × alert type + emergency/PM/escalation)
    troubleshooting_guides.csv — 12 diagnostic trees (root cause + corrective actions per sensor fault)
    incident_resolutions.csv   — 12 prior incident records (real wafer IDs, confirmed root causes)
    equipment_manuals.csv      — 15 manual excerpts (TCP gen, BCl3/Cl2 gas systems, pressure control)

All data is grounded in the LAM Research 9600 TCP Metal Etcher — BCl3/Cl2 chemistry,
TCP Top Power (334–360 W), RF Bottom Power (124–142 W), chamber pressure (942–1420 mTorr).
"""

import os
import pandas as pd
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
from chromadb.api.types import EmbeddingFunction

load_dotenv(Path(__file__).parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "sop_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K           = 5
OPENAI_MODEL    = os.getenv("AZURE_DEPLOYMENT")

TOTAL_EXPECTED_DOCS = 54  # 15 SOPs + 12 guides + 12 incidents + 15 manuals

DEFAULT_CSV_PATHS = {
    "sop":       "data/sop_procedures.csv",
    "guides":    "data/troubleshooting_guides.csv",
    "incidents": "data/incident_resolutions.csv",
    "manuals":   "data/equipment_manuals.csv",
}


# ── Embedder (mirrors quality_agent.py exactly) ───────────────────────────────

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


# ── Document formatters (one per CSV schema) ──────────────────────────────────

def _fmt_sop(row) -> str:
    return (
        f"SOP ID: {row['sop_id']}\n"
        f"TITLE: {row['title']}\n"
        f"SENSOR: {row['applies_to_sensor']}\n"
        f"ALERT TYPE: {row['applies_to_alert_type']}\n"
        f"SEVERITY THRESHOLD: {row['severity_threshold']}\n"
        f"PROCEDURE:\n{row['procedure_steps']}\n"
        f"SAFETY REQUIREMENTS:\n{row['safety_requirements']}\n"
        f"ESTIMATED DOWNTIME: {row['estimated_downtime_min']} minutes"
    )


def _fmt_guide(row) -> str:
    return (
        f"GUIDE ID: {row['guide_id']}\n"
        f"FAULT: {row['fault_description']}\n"
        f"SENSOR: {row['sensor']}\n"
        f"SYMPTOMS: {row['symptoms']}\n"
        f"ROOT CAUSES: {row['possible_root_causes']}\n"
        f"DIAGNOSTIC STEPS:\n{row['diagnostic_steps']}\n"
        f"CORRECTIVE ACTIONS:\n{row['corrective_actions']}\n"
        f"ESCALATION TRIGGER: {row['escalation_trigger']}"
    )


def _fmt_incident(row) -> str:
    return (
        f"INCIDENT: {row['incident_id']}\n"
        f"SENSOR: {row['sensor']}\n"
        f"FAULT: {row['fault_summary']}\n"
        f"ROOT CAUSE: {row['root_cause_confirmed']}\n"
        f"STEPS TAKEN:\n{row['resolution_steps_taken']}\n"
        f"OUTCOME: {row['outcome']}\n"
        f"LESSONS LEARNED: {row['lessons_learned']}"
    )


def _fmt_manual(row) -> str:
    return (
        f"DOC: {row['doc_id']}\n"
        f"EQUIPMENT: {row['equipment_name']}\n"
        f"SECTION: {row['section_title']}\n"
        f"KEYWORDS: {row['keywords']}\n"
        f"CONTENT:\n{row['content']}"
    )


# ── ChromaDB vector store ─────────────────────────────────────────────────────

def build_index(csv_paths: dict = None) -> chromadb.Collection:
    """
    Build (or reload) the SOP knowledge base vector store.

    Args:
      csv_paths : dict of {label: path} for the 4 knowledge base CSVs.
                  Defaults to DEFAULT_CSV_PATHS if not provided.

    Returns:
      ChromaDB collection with all 54 documents embedded.
    """
    if csv_paths is None:
        csv_paths = DEFAULT_CSV_PATHS

    embed_fn      = OpenAIEmbedder()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_or_create_collection(
        name             = COLLECTION_NAME,
        embedding_function = embed_fn,
        metadata         = {"hnsw:space": "cosine"}
    )

    if collection.count() == TOTAL_EXPECTED_DOCS:
        print(f"[SOP VectorStore] Loaded {collection.count()} existing documents from disk.")
        return collection

    # Rebuild from scratch
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name             = COLLECTION_NAME,
        embedding_function = embed_fn,
        metadata         = {"hnsw:space": "cosine"}
    )

    formatters = {
        "sop":       (_fmt_sop,      "sop_id"),
        "guides":    (_fmt_guide,    "guide_id"),
        "incidents": (_fmt_incident, "incident_id"),
        "manuals":   (_fmt_manual,   "doc_id"),
    }

    documents, ids, metadatas = [], [], []

    for source_key, path in csv_paths.items():
        fmt_fn, id_col = formatters[source_key]
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            documents.append(fmt_fn(row))
            ids.append(f"{source_key}_{idx}")
            metadatas.append({
                "source": source_key,
                "id":     str(row[id_col]),
            })

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"[SOP VectorStore] Indexed {len(documents)} documents and saved to disk.")

    return collection


# ── Build retrieval query ─────────────────────────────────────────────────────

def build_query(alert: dict,
                quality_report: str = "",
                recommendation: dict = None) -> str:
    """
    Converts a Production Agent alert (plus optional upstream context) into a
    retrieval query. The 'explanation' field is deliberately excluded — retrieval
    must be driven by confirmed facts, not the Production Agent's hypothesis.
    """
    chamber   = alert.get("chamber_id") or alert.get("chamber", "unknown")
    priority  = (recommendation or {}).get("priority", alert.get("severity", "unknown"))
    component = (recommendation or {}).get("component", "")
    return (
        f"{alert.get('alert_type', 'ANOMALY')} on sensor {alert['sensor']}. "
        f"Deviation: {alert.get('deviation', '')}. "
        f"Priority: {priority}. Chamber: {chamber}. "
        + (f"Component: {component}. " if component else "")
        + f"Looking for SOPs, troubleshooting procedures, corrective actions, "
        f"and prior incident resolutions for {alert['sensor']} faults. "
        f"Safety requirements and escalation criteria."
    )


# ── Retrieve relevant documents ───────────────────────────────────────────────

def retrieve_cases(collection: chromadb.Collection,
                   query: str,
                   top_k: int = TOP_K) -> list[dict]:
    """
    Queries ChromaDB for the top_k most semantically similar documents.
    Returns list of dicts with rank, similarity, content, and source metadata.
    """
    results = collection.query(
        query_texts = [query],
        n_results   = top_k,
        include     = ["documents", "distances", "metadatas"]
    )

    cases = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        cases.append({
            "rank":       i + 1,
            "similarity": round(1 - results["distances"][0][i], 4),
            "content":    results["documents"][0][i],
            "source":     meta.get("source", "unknown"),
            "id":         meta.get("id", ""),
        })

    return cases


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the SOP/Knowledge Agent for the LAM Research 9600 TCP Metal Etcher.

You receive:
  1. A production alert — sensor, deviation, severity, chamber
  2. A quality history report — past NCRs, tool wear indicators, recurrence patterns
  3. A maintenance recommendation — PM status, parts availability, calibration status, priority
  4. Retrieved knowledge base documents — SOPs, troubleshooting guides, prior incidents, manual excerpts

Your job: synthesize a prioritized list of exactly 3–5 next-best-action recommendations.

Rules:
- Each recommendation must cite the specific document(s) it comes from (SOP ID, guide ID, incident ID, or doc ID)
- Order recommendations by urgency: safety first, then fault resolution, then prevention
- Be specific to the sensor values, component names, and thresholds in the retrieved documents
- Do not add steps not supported by the retrieved content
- Output exactly 3–5 numbered recommendations — no more, no fewer

Output format (strict):

SENSOR: [sensor name]
CHAMBER: [chamber ID]
ALERT TYPE: [ANOMALY or TREND]
PRIORITY: [CRITICAL / HIGH / MEDIUM / LOW — from maintenance recommendation]

RECOMMENDATIONS:

1. [Short action title]
   [2–3 sentences: what to do, why, and what to check]
   Source: [document ID] — "[document title or section]"

2. [Short action title]
   [2–3 sentences]
   Source: [document ID] — "[document title or section]"

[Continue for 3–5 total]
""".strip()


# ── Synthesise report ─────────────────────────────────────────────────────────

def synthesise_report(alert: dict,
                      quality_report: str,
                      recommendation: dict,
                      retrieved: list[dict],
                      client: AzureOpenAI) -> str:
    """
    Sends all three upstream outputs + retrieved KB docs to the LLM.
    Returns 3–5 cited recommendations as structured plain text.
    """
    cases_block = "\n\n---\n\n".join([
        f"DOCUMENT {c['rank']} (source: {c['source']}, id: {c['id']}, "
        f"similarity: {c['similarity']}):\n\n{c['content']}"
        for c in retrieved
    ])

    chamber  = alert.get("chamber_id") or alert.get("chamber", "unknown")
    atype    = alert.get("alert_type", "ANOMALY")

    maint_block = ""
    if recommendation:
        maint_block = (
            f"MAINTENANCE CONTEXT\n"
            f"Priority       : {recommendation.get('priority', 'unknown')}\n"
            f"Component      : {recommendation.get('component', 'unknown')}\n"
            f"PM status      : {recommendation.get('pm_status', {})}\n"
            f"Parts status   : {recommendation.get('required_parts', [])}\n"
            f"Calibration    : {recommendation.get('calibration_status', {})}\n"
            f"Actions (CMMS) : {recommendation.get('recommended_actions', [])}\n"
            f"Narrative      : {recommendation.get('llm_narrative', '')}\n"
        )

    user_msg = (
        f"PRODUCTION ALERT\n{'='*40}\n"
        f"Alert type    : {atype}\n"
        f"Sensor        : {alert['sensor']}\n"
        f"Measured value: {alert.get('value', 'N/A')}\n"
        f"Threshold     : min={alert.get('threshold_min', 'N/A')} / "
        f"max={alert.get('threshold_max', 'N/A')}\n"
        f"Deviation     : {alert.get('deviation', '')}\n"
        f"Chamber       : {chamber}\n"
        f"Wafer ID      : {alert.get('wafer_id', 'N/A')}\n"
        f"Lot           : {alert.get('lot_id', 'unknown')}\n"
        f"Step          : {alert.get('step', 'N/A')}\n"
        f"Severity      : {alert.get('severity', 'unknown')}\n"
        + (f"Time to breach: {alert.get('time_to_breach')}\n" if atype == "TREND" else "")
        + f"\n{'='*40}\n\nQUALITY HISTORY REPORT\n{quality_report}\n\n"
        + f"{'='*40}\n\n{maint_block}\n"
        + f"{'='*40}\n\nRETRIEVED KNOWLEDGE BASE ({len(retrieved)} documents):\n\n"
        + cases_block
    )

    response = client.chat.completions.create(
        model      = OPENAI_MODEL,
        max_tokens = 1400,
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ]
    )

    return response.choices[0].message.content.strip()


# ── Main pipeline function ────────────────────────────────────────────────────

def run_sop_agent(alert: dict,
                  quality_report: str,
                  recommendation: dict,
                  collection: chromadb.Collection,
                  client: AzureOpenAI) -> str:
    """
    Full SOP Agent pipeline. Called by the Production Agent after the
    Quality Agent and Maintenance Agent have both produced their outputs.

    Args:
      alert          : normalized alert dict from the Production Agent
      quality_report : structured text report from the Quality Agent
      recommendation : recommendation dict from the Maintenance Agent
      collection     : ChromaDB collection (built once at startup via build_index())
      client         : AzureOpenAI client

    Returns:
      report : 3–5 prioritized recommendations, each citing the source document
    """
    chamber = alert.get("chamber_id") or alert.get("chamber", "unknown")
    print(f"\n[SOP Agent] Alert: {alert['sensor']} "
          f"{alert.get('deviation', '')} on {chamber}")

    query     = build_query(alert, quality_report, recommendation)
    retrieved = retrieve_cases(collection, query, top_k=TOP_K)

    print(f"[SOP Agent] Retrieved {len(retrieved)} documents. "
          f"Best similarity: {retrieved[0]['similarity']:.4f}")

    return synthesise_report(alert, quality_report, recommendation, retrieved, client)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    collection = build_index(DEFAULT_CSV_PATHS)
    client     = AzureOpenAI(
        api_key        = os.getenv("AZURE_API_KEY"),
        azure_endpoint = os.getenv("AZURE_ENDPOINT"),
        api_version    = os.getenv("AZURE_API_VERSION"),
    )

    test_alerts = [
        {
            "alert_type":    "ANOMALY",
            "wafer_id":      2915,
            "lot_id":        "LOT_29B",
            "chamber_id":    "CHA",
            "sensor":        "tcp_top_pwr",
            "value":         410.0,
            "threshold_min": 334.0,
            "threshold_max": 360.0,
            "deviation":     "+50W above set-point",
            "severity":      "CRITICAL",
            "step":          12,
            "time_to_breach": None,
        },
        {
            "alert_type":    "ANOMALY",
            "wafer_id":      2937,
            "lot_id":        "LOT_29B",
            "chamber_id":    "CHB",
            "sensor":        "bcl3_flow",
            "value":         778.0,
            "threshold_min": 740.0,
            "threshold_max": 765.0,
            "deviation":     "+13 sccm above set-point",
            "severity":      "HIGH",
            "step":          8,
            "time_to_breach": None,
        },
        {
            "alert_type":    "TREND",
            "wafer_id":      2918,
            "lot_id":        "LOT_29B",
            "chamber_id":    "CHA",
            "sensor":        "pressure",
            "value":         1380.0,
            "threshold_min": 942.0,
            "threshold_max": 1420.0,
            "deviation":     "trending increasing toward upper limit",
            "severity":      "MEDIUM",
            "step":          15,
            "time_to_breach": "2 minutes 30 seconds",
        },
    ]

    # Stub upstream outputs for standalone testing
    stub_quality_report = (
        "SENSOR: tcp_top_pwr\nCHAMBER: CHA\nALERT TYPE: ANOMALY\n\n"
        "NCR SUMMARY:\nWafer 2915 had TCP +50W fault. RF generator impedance "
        "network failed at 1241 hours. Pattern: 3 TCP faults in last 90 days.\n\n"
        "TOOL WEAR INDICATORS:\nRF generator at 1241 hours (threshold 1200). "
        "PM overdue by 41 hours.\n\n"
        "RECURRENCE:\n3 occurrences on tcp_top_pwr in retrieved records."
    )
    stub_recommendation = {
        "priority":           "CRITICAL",
        "sensor":             "tcp_top_pwr",
        "chamber_id":         "CHA",
        "alert_type":         "ANOMALY",
        "component":          "TCP_GENERATOR",
        "pm_status":          {"overdue_items": ["WET_CLEAN overdue by 8 wafers"]},
        "required_parts":     [{"part_number": "IMP-CAP-7730", "status": "LOW_STOCK"}],
        "calibration_status": {"status": "OVERDUE", "notes": "TCP power meter overdue 10 days"},
        "recommended_actions": ["Replace capacitor bank", "Schedule wet clean"],
        "llm_narrative":      "TCP generator approaching end of service life. Immediate PM required.",
    }

    for i, alert in enumerate(test_alerts, 1):
        print(f"\n{'='*70}")
        print(f"TEST ALERT {i}/{len(test_alerts)} "
              f"[{alert['alert_type']}] {alert['sensor']} on {alert['chamber_id']}")
        print(f"{'='*70}")

        report = run_sop_agent(
            alert, stub_quality_report, stub_recommendation, collection, client
        )
        print(f"\n{report}")
