"""
Quality Intelligence Agent — RAG System
========================================
Pipeline position:
  Production Agent → [THIS AGENT] → Maintenance Agent

Embedding strategy:
  sentence-transformers/all-MiniLM-L6-v2 from HuggingFace
  — understands meaning, not just word overlap
  — "power exceeded" and "set-point overridden" will match correctly

Vector database:
  ChromaDB (persistent) — vectors saved to disk in chroma_db/
  — built once, survives restarts
  — no need to re-embed on every run
"""

import os
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

QUALITY_RECORDS_PATH = str(_ROOT / "data/quality_records.csv")
CHROMA_DB_PATH       = str(_ROOT / "chroma_db")
COLLECTION_NAME      = "quality_records"   # name of the collection inside ChromaDB
EMBEDDING_MODEL      = "text-embedding-3-small" # HuggingFace model for embeddings
TOP_K                = 5
# Cosine-similarity floor: genuine matches ~0.65+, noise ~0.15, so 0.35 keeps
# real records and drops irrelevant ones. Empty result => no relevant history.
MIN_SIMILARITY       = 0.35
OPENAI_MODEL         = os.getenv("AZURE_DEPLOYMENT")

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


# ── ChromaDB vector store ─────────────────────────────────────────────────────

def build_index(csv_path: str) -> chromadb.Collection:

    embed_fn = OpenAIEmbedder()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    df = pd.read_csv(csv_path)

    if collection.count() == len(df):
        print(f"[VectorStore] Loaded {collection.count()} existing documents from disk.")
        return collection

    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    documents, ids = [], []
    for idx, row in df.iterrows():
        chunk = (
            f"NCR REPORT:\n{row['ncr_report']}\n\n"
            f"PAST MACHINE DEFECTS:\n{row['past_machine_defects']}\n\n"
            f"INSPECTION RESULTS:\n{row['inspection_results']}\n\n"
            f"SPC TREND:\n{row['spc_trend']}\n\n"
            f"CUSTOMER COMPLAINT:\n{row['customer_complaint']}\n\n"
            f"QUALITY HISTORY SCORE: {row['quality_history_score']} / 10"
        )
        documents.append(chunk)
        ids.append(f"case_{idx}")

    collection.add(documents=documents, ids=ids)
    print(f"[VectorStore] Indexed {len(documents)} documents and saved to disk.")
   
    return collection

# ── Build retrieval query ─────────────────────────────────────────────────────

def build_query(alert: dict) -> str:
    """
    Converts the Production Agent's alert into a natural language query.
    ChromaDB will embed this query using the same model used at index time,
    then find the most semantically similar documents.
    """
    chamber = alert.get("chamber_id") or alert.get("chamber", "unknown")
    return (
        f"{alert.get('alert_type', 'ANOMALY')} on sensor {alert['sensor']}. "
        f"Deviation: {alert.get('deviation', '')}. "
        f"Chamber: {chamber}. Lot: {alert.get('lot_id', 'unknown')}. "
        f"Severity: {alert.get('severity', 'unknown')}. "
        #f"{alert.get('explanation', '')} " remove explanation from query; make retrieval happen based on sensor, wafer, chamber and lot
        f"Looking for NCR reports, SPC violations, inspection failures, "
        f"and customer complaints related to {alert['sensor']} on {chamber}. "
        f"Root cause and tool wear history."
    )


# ── Retrieve relevant cases ───────────────────────────────────────────────────

def retrieve_cases(collection: chromadb.Collection,
                   query: str,
                   top_k: int = TOP_K) -> list[dict]:
    """
    Queries ChromaDB for the top_k most semantically similar documents.

    ChromaDB embeds the query string using the same HuggingFace model,
    then compares it against all stored vectors using cosine similarity.
    Returns the top_k closest matches.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances"]
    )

    docs      = results["documents"][0] if results.get("documents") else []
    distances = results["distances"][0] if results.get("distances") else []

    cases = []
    for i in range(len(docs)):
        # ChromaDB returns distance (lower = more similar); convert to similarity.
        similarity = round(1 - distances[i], 4)
        if similarity < MIN_SIMILARITY:        # drop weak / irrelevant matches
            continue
        cases.append({
            "rank":       len(cases) + 1,
            "similarity": similarity,
            "content":    docs[i],
        })

    return cases


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Quality Intelligence Agent in a semiconductor manufacturing pipeline.

You receive:
  1. A Production Agent alert — a sensor that just exceeded its threshold
  2. Retrieved NCR records — the most relevant past non-conformance reports
     from the quality database, found by semantic similarity to the alert

Your job is to extract and summarise the NCR evidence related to the sensor
and tool that produced the anomaly. This summary will be passed to the
Maintenance Agent as context.

Be concise and factual. Only report what the retrieved NCRs actually say.
Do not add interpretation, recommendations, or sections not supported by
the evidence.

Output format:

SENSOR: [sensor name]
CHAMBER: [chamber ID]
ALERT TYPE: [ANOMALY or TREND]

NCR SUMMARY:
[For each relevant retrieved NCR, one short paragraph covering:
 - what fault was recorded
 - what the sensor deviation was
 - what defects or quality impact were observed
 - what root cause was recorded at the time
 - whether this is a recurring pattern on this sensor or chamber]

TOOL WEAR INDICATORS:
[Any evidence from the NCRs of tool wear, PM overdue status,
 RF generator hours, or open work orders at the time of past faults.
 If none recorded, state: None found in retrieved records.]

RECURRENCE:
[How many times has this sensor or a related fault appeared in the
 retrieved records. State the pattern if one exists.]
""".strip()


# ── Synthesise report ─────────────────────────────────────────────────────────

def synthesise_report(alert: dict,
                      retrieved: list[dict],
                      client: AzureOpenAI) -> str:
    """
    Sends the alert and retrieved cases to the LLM.
    Returns the structured text report directly — no JSON, no parsing.
    """
    cases_block = "\n\n---\n\n".join([
        f"RETRIEVED CASE {c['rank']} "
        f"(similarity: {c['similarity']}):\n\n{c['content']}"
        for c in retrieved
    ]) or ("No sufficiently relevant quality records were found for this fault. "
           "Do NOT invent NCRs, defects, or history — report that no matching "
           "records exist and base the summary only on the alert.")

    chamber = alert.get("chamber_id") or alert.get("chamber", "unknown")
    atype   = alert.get("alert_type", "ANOMALY")
    ttb     = alert.get("time_to_breach", "N/A")

    user_msg = (
        f"PRODUCTION AGENT ALERT\n"
        f"{'='*40}\n"
        f"Alert type    : {atype}\n"
        f"Sensor        : {alert['sensor']}\n"
        f"Measured value: {alert.get('value', 'N/A')}\n"
        f"Threshold     : min={alert.get('threshold_min','N/A')} / "
        f"max={alert.get('threshold_max','N/A')}\n"
        f"Deviation     : {alert.get('deviation', '')}\n"
        f"Chamber       : {chamber}\n"
        f"Wafer ID      : {alert.get('wafer_id', 'N/A')}\n"
        f"Lot           : {alert.get('lot_id', 'unknown')}\n"
        f"Step          : {alert.get('step', 'N/A')}\n"
        f"Severity      : {alert.get('severity', 'unknown')}\n"
        + (f"Time to breach: {ttb}\n" if atype == "TREND" else "")
        #+ f"Explanation   : {alert.get('explanation', 'Not provided.')}\n\n" 
        + f"{'='*60}\n\n"
        + f"RETRIEVED QUALITY HISTORY "
        + f"({len(retrieved)} most relevant past cases):\n\n"
        + f"{cases_block}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ]
    )

    return response.choices[0].message.content.strip()


# ── Main pipeline function ────────────────────────────────────────────────────

def run_quality_agent(alert: dict,
                      collection: chromadb.Collection,
                      client: AzureOpenAI) -> str:
    """
    Full Quality Agent pipeline. Called directly by the Production Agent.

    Args:
      alert      : dict from the Production Agent (one fault at a time)
      collection : ChromaDB collection (built once at startup)
      client     : AzureOpenAI client

    Returns:
      report : structured text passed directly to the Maintenance Agent
    """
    chamber = alert.get("chamber_id") or alert.get("chamber", "unknown")
    print(f"\n[Quality Agent] Alert: {alert['sensor']} "
          f"{alert.get('deviation','')} on {chamber}")

    query     = build_query(alert)
    retrieved = retrieve_cases(collection, query, top_k=TOP_K)

    if retrieved:
        print(f"[Quality Agent] Retrieved {len(retrieved)} cases "
              f"(floor {MIN_SIMILARITY}):")
        for c in retrieved:
            # Skip the generic "NCR REPORT:" header; use the NCR line that
            # carries wafer/lot/chamber so each row is distinguishable.
            lines = [l for l in c["content"].splitlines() if l.strip()]
            head = (lines[1] if len(lines) > 1 else lines[0])[:70]
            print(f"    similarity = {c['similarity']:.4f}  |  {head}")
    else:
        print("[Quality Agent] No sufficiently relevant quality records found.")

    # Add this block to see the retrieved cases
    #print("\n[Quality Agent] Top 5 retrieved cases:")
    #for case in retrieved:
        #print(f"\n--- Rank {case['rank']} | Similarity: {case['similarity']} ---")
        #print(case['content'][:500])  # first 500 characters of each case
        #print("...")  

        
    return synthesise_report(alert, retrieved, client)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    collection = build_index(QUALITY_RECORDS_PATH)
    client     = AzureOpenAI(
        api_key        = os.getenv("AZURE_API_KEY"),
        azure_endpoint = os.getenv("AZURE_ENDPOINT"),
        api_version    = os.getenv("AZURE_API_VERSION"),
    )

    alerts = [
        {
            "alert_type": "ANOMALY", "wafer_id": 2915, "lot_id": "LOT_29B",
            "chamber_id": "CHA", "sensor": "tcp_top_pwr", "value": 410.0,
            "threshold_min": 300.0, "threshold_max": 360.0,
            "deviation": "+50W above set-point", "severity": "CRITICAL",
            "step": 12, "time_to_breach": None,
            "explanation": "TCP Top Power of 410W exceeds max threshold of 360W by 50W."
        },
        {
            "alert_type": "ANOMALY", "wafer_id": 2937, "lot_id": "LOT_29B",
            "chamber_id": "CHB", "sensor": "bcl3_flow", "value": 758.0,
            "threshold_min": 748.0, "threshold_max": 754.0,
            "deviation": "+5 sccm above set-point", "severity": "HIGH",
            "step": 8, "time_to_breach": None,
            "explanation": "BCl3 flow of 758 sccm exceeds maximum threshold of 754 sccm."
        },
        {
            "alert_type": "TREND", "wafer_id": 2940, "lot_id": "LOT_29B",
            "chamber_id": "CHA", "sensor": "he_press", "value": 7.8,
            "threshold_min": 6.0, "threshold_max": 10.0,
            "deviation": "trending -2 Torr toward lower limit", "severity": "HIGH",
            "step": None, "time_to_breach": "6 minutes 30 seconds",
            "explanation": "He backside pressure on sustained downward trend. ESC seal degradation suspected."
        },
    ]

    for i, alert in enumerate(alerts, 1):
        print(f"\n{'='*70}")
        print(f"TEST ALERT {i}/{len(alerts)} "
              f"[{alert['alert_type']}] {alert['sensor']} on {alert['chamber_id']}")
        print(f"{'='*70}")

        report = run_quality_agent(alert, collection, client)
        print(f"\n{report}")