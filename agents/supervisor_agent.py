"""
Human-in-the-Loop Supervisor Agent
===================================
Pipeline position:
  Production Agent → Quality Agent → Maintenance Agent → SOP Agent → Sustainability Agent → [THIS AGENT]

Design philosophy (mirrors the other agents):
  This agent does not detect faults, retrieve records, or check facts itself —
  it synthesises everything the upstream agents already produced (plus a
  Sustainability Agent placeholder) into short, role-specific recommendations
  for the four humans who actually act on an event: the operator, the shift
  supervisor, the quality engineer, and the maintenance lead.

  No critical action is auto-executed. This agent only summarizes and
  recommends — a human approves and carries out every action.

Inputs / outputs:
  Input  : the Production alert, Quality report, Maintenance recommendation,
           SOP report, and (placeholder) Sustainability report.
  Output : a structured dict with one recommendation block per role — ready
           for a web app UI or console printer to render directly.
"""

# import required modules and libraries
import os 
import re
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path

try:
    from sustainability_agent import run_impact_agent
except ImportError:  # allow "python agents/supervisor_agent.py" and package import
    from agents.sustainability_agent import run_impact_agent

# Load .env
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

AZURE_API_KEY     = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT     = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT   = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION  = os.getenv("AZURE_API_VERSION")
OPENAI_MODEL       = AZURE_DEPLOYMENT

ROLES = ("operator", "supervisor", "quality_engineer", "maintenance_lead")


def _check_env() -> None:
    missing = [name for name, val in {
        "AZURE_API_KEY"    : AZURE_API_KEY,
        "AZURE_ENDPOINT"   : AZURE_ENDPOINT,
        "AZURE_DEPLOYMENT" : AZURE_DEPLOYMENT,
        "AZURE_API_VERSION": AZURE_API_VERSION,
    }.items() if not val]
    if missing:
        raise ValueError(
            f"Missing environment variables: {missing}\n"
            f"Check your .env file at: {_ROOT / '.env'}"
        )


def get_client() -> AzureOpenAI:
    _check_env()
    return AzureOpenAI(
        api_key        = AZURE_API_KEY,
        azure_endpoint = AZURE_ENDPOINT,
        api_version    = AZURE_API_VERSION,
    )


# ── Sustainability Agent ─────────────────────────────────────────────────────────
# Backed by the Impact Agent (sustainability_agent.py / Agent 5), which computes
# a deterministic scrap-cost estimate for an ACTIVE anomaly (see that file for the
# counting/pricing logic). Trends are pre-fault by definition — no wafers have
# been damaged yet — so this returns a simple "not applicable" result for TRENDs
# without calling the Impact Agent.

def run_sustainability_agent(alert: dict,
                              quality_report: str = None,
                              recommendation: dict = None,
                              roster=None,
                              client: AzureOpenAI = None,
                              price_provider=None) -> dict:
    """
    Wraps the Impact Agent's deterministic scrap-cost estimate into the
    Supervisor Agent's sustainability-report shape:
      {"scrap_cost_estimate": {...} or None,
       "material_waste_estimate": int or None,
       "notes": str}
    """
    alert = alert or {}
    if alert.get("alert_type") != "ANOMALY":
        return {
            "scrap_cost_estimate":     None,
            "material_waste_estimate": None,
            "notes": ("Not applicable — this is a pre-fault trend; no wafers "
                      "have been scrapped yet."),
        }

    estimate = run_impact_agent(
        alert, quality_report=quality_report, recommendation=recommendation,
        roster=roster, client=client, price_provider=price_provider,
    )

    if not estimate.get("estimable"):
        return {
            "scrap_cost_estimate":     None,
            "material_waste_estimate": None,
            "notes": estimate.get(
                "narrative",
                f"Scrap impact not estimable: {estimate.get('reason', 'unknown')}."
            ),
        }

    ec = estimate["expected_cost"]
    return {
        "scrap_cost_estimate": {
            "value": ec["value"], "low": ec["low"], "high": ec["high"],
        },
        "material_waste_estimate": estimate.get("remaining_wafers"),
        "notes": estimate.get("narrative", ""),
    }


# ── LLM synthesis ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Human-in-the-Loop Supervisor Agent for a semiconductor fab's
manufacturing copilot. You are the last stage in a chain of specialized agents:
Production (fault detection) -> Quality (historical NCRs) -> Maintenance (CMMS
facts + recommendation) -> SOP (relevant procedures) -> Sustainability (scrap cost estimates) -> you.

Your job is NOT to detect faults, invent facts, or verify anything — every
prior agent already did that. Your job is to translate the accumulated,
already-verified information into short, concrete, role-specific next steps
for four different humans, so each person immediately knows what to do without
reading the full upstream reports.

The four roles and their responsibilities:
  - operator          : works the line right now. Needs immediate, physical,
                         line-level actions (e.g. pause the line, isolate a
                         station, stop feeding a specific lot).
  - supervisor         : owns the shift. Needs a concise situation summary and
                         coordination actions (e.g. who to notify, what to
                         approve, what to escalate).
  - quality_engineer   : owns product disposition. Needs quality-specific
                         actions (e.g. quarantine lots, review NCRs, dispatch
                         inspection).
  - maintenance_lead   : owns the equipment fix. Needs maintenance-specific
                         actions (e.g. inspect a station, run a named SOP,
                         create a CMMS work order, order/replace a part).

Rules:
  - Use ONLY the facts given to you below (the alert, the Maintenance Agent's
    deterministic CMMS facts and narrative, the SOP report, and the
    sustainability placeholder). Do not invent part numbers, SOP IDs, chamber
    IDs, or facts not present in the input.
  - Treat the Production Agent's plain-language "explanation" field and the
    Quality Agent's narrative as hypotheses, not verified fact — do not let
    them override the Maintenance Agent's deterministic priority, parts, PM,
    or calibration facts.
  - Every action must be short, imperative, and specific (reference an actual
    sensor, chamber, SOP ID, part number, or CMMS fact given to you).
  - TECHNICAL SPECIFICITY (critical): the reader is a trained technician and
    already knows the obvious generic responses. Do NOT fill the list with
    generic, self-evident advice such as "stop the line", "isolate the chamber",
    "notify maintenance", "monitor the situation", "follow safety procedures",
    or "quarantine the lot" UNLESS you attach the specific technical detail that
    makes it actionable. Instead, tell them the concrete technical fix drawn from
    the upstream agents' data, for example:
      * the exact SOP/troubleshooting procedure to run — cite its ID/title and
        the specific step(s) from the SOP Agent report (e.g. "Run SOP-TCP-014
        step 3: reseat the TCP match network and re-strike plasma");
      * the specific part to replace and its status/location from the Maintenance
        Agent (e.g. "Replace TCP match capacitor PN-4471 — in stock, loc A-12");
      * the specific PM/calibration task that is due or overdue (e.g. "Perform
        wet clean — overdue by 320 wafers", "Recalibrate TCP generator — cal
        overdue, next-due date passed");
      * the specific fix that resolved this fault in the retrieved past work
        orders (from PAST WORK-ORDER PATTERNS);
      * concrete inspection targets tied to the component (e.g. "Inspect TCP
        generator output stage and RF strap for arcing"), not "inspect the tool".
    If a generic containment step (stop line / isolate chamber / quarantine lot)
    is genuinely warranted by the priority, keep AT MOST one such step and spend
    the remaining actions on the specific technical fixes above.
  - Base every technical action on a fact actually present in the inputs (SOP
    Agent report, Maintenance recommended CMMS actions, parts, PM, calibration,
    past WO patterns, draft WO). Do NOT invent step numbers, part numbers, SOP
    IDs, or torque/spec values that are not given. If the inputs do not contain
    a specific fix for a role, give the most specific next diagnostic step the
    facts DO support rather than a generic platitude.
  - Keep each role's list to 2-5 actions. Do not repeat the same action
    verbatim across roles unless it genuinely requires both people to act.
  - The supervisor's actions should include notifying/coordinating with the
    other roles when appropriate.
  - IMPORTANT — severity/urgency wording: the personnel MUST be told how urgent
    this is, and the UI also shows the FINAL priority (CRITICAL / HIGH / MEDIUM /
    LOW) as a badge next to your summary. These two must NEVER contradict each
    other. Therefore:
      * If you state a severity or urgency level in the summary prose, use the
        Maintenance Agent's FINAL priority verbatim (e.g. a HIGH-priority issue
        is described as "high-priority" / "urgent", never as "low-severity" or
        "minor"). The word you use and the badge must match.
      * NEVER derive the severity wording from the "Deviation magnitude" field
        below. That field reflects only how far the sensor drifted from its
        threshold — it is NOT the overall urgency. A small deviation can still
        be HIGH priority (e.g. overdue PM/calibration or a part shortage), so
        calling such an issue "low-severity" would dangerously understate it.
      * State the urgency plainly so the reader knows how serious it is, then
        describe what is happening and what is at risk.

Proportionality (important):
  - Match the response to the FINAL priority and scope given to you. A single
    LOW or MEDIUM priority anomaly on one sensor/chamber is a localized issue —
    do NOT recommend halting the line, stopping all production, or pausing all
    operations for it. Prefer the narrowest effective action: isolate or flag
    the specific chamber/tool/lot involved, keep the rest of the line running.
  - Reserve line-wide or production-halting language ("pause the line", "stop
    all processing", "halt production") for when the given priority is
    CRITICAL (or HIGH with a large expected scrap cost / clear safety issue
    stated in the facts). Otherwise, prefer scoped actions such as "isolate
    [chamber]", "hold/quarantine [lot]", "route new lots away from [chamber]",
    or "flag [chamber] for inspection before its next lot".
  - Do not escalate severity beyond what the Maintenance Agent's priority and
    the sustainability figures actually support. If the facts don't justify
    it, say so plainly rather than defaulting to the most drastic action.

Output ONLY valid JSON (no markdown, no code fences) with exactly these keys:

{
  "operator":          {"summary": "1-2 sentences.", "actions": ["...", "..."]},
  "supervisor":        {"summary": "1-2 sentences.", "actions": ["...", "..."]},
  "quality_engineer":  {"summary": "1-2 sentences.", "actions": ["...", "..."]},
  "maintenance_lead":  {"summary": "1-2 sentences.", "actions": ["...", "..."]}
}
""".strip()


def _strip_code_fence(text: str) -> str:
    """GPT sometimes wraps JSON in ```json ... ``` — remove it before parsing."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _fallback_roles(narrative: str) -> dict:
    """Used only if the LLM does not return valid JSON."""
    return {role: {"summary": narrative, "actions": []} for role in ROLES}


# ── Grounding guard: no invented part numbers / SOP IDs / WO numbers ─────────────

# Citeable identifiers in this domain look like PART/SOP/WO/INC/NCR codes:
# hyphenated uppercase-alphanumeric groups that contain at least one digit,
# e.g. IMP-CAP-7730, TCP-SEAL-O42, SOP-TCP-001, WO-CHA-2915, INC-2915-001,
# NCR-CHA-2915. Plain words and chamber/sensor names are intentionally excluded.
_IDENT_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)+\b")


def _normalize_ident(tok: str) -> str:
    """Case/separator-insensitive form so 'LOT_29A' and 'LOT-29a' compare equal."""
    return tok.upper().replace("_", "-").replace("/", "-")


def _cited_idents(text: str) -> list[str]:
    """Identifier-like citations in a piece of text (must contain a digit)."""
    return [m.group(0) for m in _IDENT_RE.finditer(text or "")
            if any(c.isdigit() for c in m.group(0))]


def _verify_grounding(roles: dict, corpus: str) -> dict:
    """
    Deterministic anti-hallucination guard. Every part number, SOP ID, WO/INC/NCR
    number the LLM cites MUST appear verbatim in `corpus` (the exact upstream data
    the LLM was given). An ACTION that cites an identifier not present in the data
    is treated as invented and dropped — the operator never sees a fabricated part
    or procedure number. Ungrounded identifiers in a SUMMARY are logged and the
    identifier is scrubbed from the prose (the summary text is kept, since it is
    narrative rather than an executable instruction).

    This runs AFTER the LLM and does not trust it: grounding is decided by string
    membership against the source data, not by the model's own claims.
    """
    corpus_norm = _normalize_ident(corpus)
    dropped, scrubbed = [], []

    for role, block in roles.items():
        if not isinstance(block, dict):
            continue

        kept_actions = []
        for action in block.get("actions", []) or []:
            bad = [tok for tok in _cited_idents(str(action))
                   if _normalize_ident(tok) not in corpus_norm]
            if bad:
                dropped.append((role, str(action), bad))
            else:
                kept_actions.append(action)
        block["actions"] = kept_actions

        summary = str(block.get("summary", "") or "")
        bad_sum = [tok for tok in _cited_idents(summary)
                   if _normalize_ident(tok) not in corpus_norm]
        if bad_sum:
            for tok in bad_sum:
                summary = summary.replace(tok, "the referenced item")
            summary = re.sub(r"\s{2,}", " ", summary).strip()
            block["summary"] = summary
            scrubbed.append((role, bad_sum))

    if dropped:
        print(f"[Supervisor Agent] GROUNDING GUARD: dropped {len(dropped)} action(s) "
              f"citing identifiers not found in the upstream data:")
        for role, action, bad in dropped:
            print(f"    - [{role}] invented {bad}: {action!r}")
    if scrubbed:
        print(f"[Supervisor Agent] GROUNDING GUARD: scrubbed ungrounded identifiers "
              f"from {len(scrubbed)} summary/summaries: {scrubbed}")
    if not dropped and not scrubbed:
        print("[Supervisor Agent] GROUNDING GUARD: all cited identifiers verified "
              "against upstream data.")

    return roles


def synthesise_recommendations(alert: dict,
                                recommendation: dict,
                                sop_report: str,
                                sustainability_report: dict,
                                client: AzureOpenAI) -> dict:
    """
    Sends the accumulated upstream outputs to the LLM and asks for role-specific
    recommendations. Returns a dict with one block per role in ROLES.
    """
    alert = alert or {}
    recommendation = recommendation or {}

    pm_status    = recommendation.get("pm_status", {}) or {}
    calib_status = recommendation.get("calibration_status", {}) or {}
    parts        = recommendation.get("required_parts", []) or []

    parts_lines = "\n".join([
        f"  - {p.get('part_number')} ({p.get('description')}): "
        f"{p.get('status')}, qty {p.get('quantity_on_hand')}, "
        f"lead time {p.get('lead_time_days')} days"
        + (f", loc {p.get('storage_location')}" if p.get('storage_location') else "")
        for p in parts
    ]) or "  - No parts mapped to this component category."

    prod_block = (
        f"PRODUCTION AGENT ALERT (Agent 1 output)\n{'='*50}\n"
        f"Alert type    : {alert.get('alert_type', 'N/A')}\n"
        f"Sensor        : {alert.get('sensor', 'N/A')}\n"
        f"Chamber       : {alert.get('chamber_id', 'N/A')}\n"
        f"Wafer / Lot   : {alert.get('wafer_id', 'N/A')} / {alert.get('lot_id', 'N/A')}\n"
        f"Deviation magnitude (how far the sensor drifted from threshold — "
        f"NOT the overall urgency): {alert.get('severity', 'N/A')}\n"
        f"NOTE: The overall urgency is the Maintenance Agent's FINAL priority "
        f"below, which may be higher than this deviation magnitude (e.g. a small "
        f"drift can still be HIGH priority due to overdue PM/calibration or a "
        f"part shortage). Communicate urgency using the FINAL priority only.\n"
    )

    cmms_actions = recommendation.get("recommended_actions", []) or []
    cmms_action_lines = "\n".join(f"  - {a}" for a in cmms_actions) \
        or "  - (none provided)"

    maint_block = (
        f"\n{'='*50}\n"
        f"MAINTENANCE AGENT RECOMMENDATION (Agent 3 output — deterministic facts)\n"
        f"{'='*50}\n"
        f"Priority (FINAL, do not change): {recommendation.get('priority', 'N/A')}\n"
        f"Component: {recommendation.get('component', 'N/A')}\n\n"
        f"PM STATUS:\n"
        f"  Wet clean : {pm_status.get('wet_clean_status', 'N/A')}"
        + (f" (OVERDUE by {pm_status.get('wet_clean_overdue_by')} wafers)"
           if pm_status.get('wet_clean_overdue_by') else "")
        + "\n"
        f"  Full PM   : {pm_status.get('full_pm_status', 'N/A')}\n\n"
        f"SPARE PARTS AVAILABILITY:\n{parts_lines}\n\n"
        f"CALIBRATION STATUS:\n"
        f"  Component: {calib_status.get('component', 'N/A')}\n"
        f"  Status   : {calib_status.get('status', 'N/A')}\n"
        f"  Last cal : {calib_status.get('last_date', 'N/A')} | "
        f"Next due: {calib_status.get('next_due', 'N/A')}\n"
        f"  Notes    : {calib_status.get('notes', 'N/A')}\n\n"
        f"RECOMMENDED CMMS ACTIONS (specific technical steps from the Maintenance "
        f"Agent — use these as the basis for the maintenance_lead's actions):\n"
        f"{cmms_action_lines}\n\n"
        f"PAST WORK-ORDER PATTERNS (what resolved this fault before — reuse the "
        f"specific fix if applicable):\n  {recommendation.get('past_wo_patterns', 'N/A')}\n\n"
        f"DRAFT WORK ORDER:\n{recommendation.get('draft_wo_header', 'N/A')}\n\n"
        f"MAINTENANCE NARRATIVE:\n  {recommendation.get('llm_narrative', 'N/A')}\n"
    )

    sop_block = f"\n{'='*50}\nSOP AGENT REPORT (Agent 4 output)\n{'='*50}\n{sop_report}\n"

    sce = sustainability_report.get("scrap_cost_estimate")
    sce_line = (
        f"${sce['low']:,.0f}–${sce['high']:,.0f} (mid ${sce['value']:,.0f})"
        if sce else "N/A"
    )
    sust_block = (
        f"\n{'='*50}\nSUSTAINABILITY / IMPACT AGENT REPORT (Agent 5 output)\n{'='*50}\n"
        f"Expected scrap cost      : {sce_line}\n"
        f"Remaining wafers at risk : {sustainability_report.get('material_waste_estimate')}\n"
        f"Notes                    : {sustainability_report.get('notes')}\n"
    )

    user_msg = f"{prod_block}{maint_block}{sop_block}{sust_block}"

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
        roles = json.loads(raw)
    except json.JSONDecodeError:
        print("[Supervisor Agent] WARNING: LLM did not return valid JSON. "
              "Falling back to raw text for all roles.")
        roles = _fallback_roles(raw)

    roles = {role: roles.get(role, {"summary": "", "actions": []}) for role in ROLES}

    # Deterministic anti-hallucination pass: verify every cited part/SOP/WO/NCR
    # identifier actually exists in the data we handed the LLM (user_msg). Any
    # action citing an invented identifier is dropped before it reaches the UI.
    roles = _verify_grounding(roles, user_msg)

    return roles


# ── Main pipeline function ──────────────────────────────────────────────────────

def run_supervisor_agent(alert: dict,
                          quality_report: str,
                          recommendation: dict,
                          sop_report: str,
                          sustainability_report: dict = None,
                          client: AzureOpenAI = None,
                          roster=None,
                          price_provider=None) -> dict:
    """
    Full Supervisor Agent pipeline (final agent in the chain). Called after the
    SOP Agent produces its report.

    This agent receives the accumulated outputs of every prior agent: the
    Production alert (Agent 1), the Quality report (Agent 2), the Maintenance
    recommendation (Agent 3), the SOP report (Agent 4), and a Sustainability
    report from the Impact Agent (Agent 5).

    Args:
      alert                  : normalized alert dict from the Production Agent
      quality_report         : structured text from the Quality Agent
      recommendation         : structured dict from the Maintenance Agent
      sop_report             : structured text from the SOP Agent
      sustainability_report  : dict from the Impact Agent, or None to compute
                                it here via run_sustainability_agent()
      client                 : AzureOpenAI client (created via get_client() if
                                not supplied)
      roster                 : wafer/lot roster passed through to the Impact
                                Agent (see sustainability_agent.load_roster());
                                only used if sustainability_report is None
      price_provider         : ScrapPriceProvider passed through to the Impact
                                Agent; only used if sustainability_report is None

    Returns:
      A dict with keys: priority, sensor, chamber_id, alert_type,
      sustainability, and one block per role in ROLES
      ({"summary": str, "actions": [str, ...]}).
    """
    alert = alert or {}
    client = client or get_client()

    if sustainability_report is None:
        sustainability_report = run_sustainability_agent(
            alert, quality_report, recommendation,
            roster=roster, price_provider=price_provider,
        )

    print(f"\n[Supervisor Agent] Synthesising role-specific recommendations for "
          f"{alert.get('sensor', 'N/A')} on {alert.get('chamber_id', 'N/A')} "
          f"[priority: {(recommendation or {}).get('priority', 'N/A')}]")

    role_blocks = synthesise_recommendations(
        alert, recommendation, sop_report, sustainability_report, client
    )

    result = {
        "priority":       (recommendation or {}).get("priority", "N/A"),
        "sensor":         alert.get("sensor", "N/A"),
        "chamber_id":     alert.get("chamber_id", "N/A"),
        "alert_type":     alert.get("alert_type", "N/A"),
        "sustainability": sustainability_report,
    }
    result.update(role_blocks)
    return result


# ── Pretty printer (console demo) ───────────────────────────────────────────────

ROLE_LABELS = {
    "operator":         "OPERATOR",
    "supervisor":       "SHIFT SUPERVISOR",
    "quality_engineer": "QUALITY ENGINEER",
    "maintenance_lead": "MAINTENANCE LEAD",
}


def print_supervisor_summary(rec: dict) -> None:
    print(f"\n{'='*70}")
    print(f"SUPERVISOR SUMMARY  —  PRIORITY: {rec['priority']}")
    print(f"{'='*70}")
    print(f"Fault     : {rec['sensor']} on {rec['chamber_id']} [{rec['alert_type']}]")

    sust = rec.get("sustainability", {}) or {}
    sce = sust.get("scrap_cost_estimate")
    print(f"\nSUSTAINABILITY / IMPACT:")
    if sce:
        print(f"  Expected scrap cost      : ${sce['low']:,.0f}–${sce['high']:,.0f} "
              f"(mid ${sce['value']:,.0f})")
        print(f"  Remaining wafers at risk : {sust.get('material_waste_estimate')}")
    print(f"  {sust.get('notes', 'N/A')}")

    for role in ROLES:
        block = rec.get(role, {})
        print(f"\n{'-'*70}")
        print(f"{ROLE_LABELS[role]}")
        print(f"{'-'*70}")
        print(f"  {block.get('summary', '')}")
        for action in block.get("actions", []):
            print(f"  - {action}")

    print(f"\n{'='*70}\n")


# ── Entry point (standalone demo) ────────────────────────────────────────────────

if __name__ == "__main__":
    demo_alert = {
        "alert_type":    "TREND",
        "sensor":        "tcp_top_pwr",
        "chamber_id":    "ETCH-04",
        "wafer_id":      "W3122",
        "lot_id":        "LOT-8841",
        "severity":      "HIGH",
    }

    demo_quality_report = (
        "SENSOR: tcp_top_pwr\nCHAMBER: ETCH-04\nALERT TYPE: TREND\n\n"
        "NCR SUMMARY:\n  3 prior NCRs linked to TCP generator drift on this "
        "chamber over the last 60 days.\n\nTOOL WEAR INDICATORS:\n  Repeated "
        "generator power drift preceding wet clean events.\n\nRECURRENCE:\n"
        "  Recurring approximately every 400-500 wafers since last wet clean."
    )

    demo_recommendation = {
        "priority":     "HIGH",
        "sensor":       "tcp_top_pwr",
        "chamber_id":   "ETCH-04",
        "alert_type":   "TREND",
        "component":    "TCP_GENERATOR",
        "pm_status": {
            "chamber_id": "ETCH-04",
            "wet_clean_status": "OVERDUE",
            "wet_clean_overdue_by": 9,
            "full_pm_status": "OK",
            "full_pm_wafers_until": 340,
            "overdue_items": ["wet_clean"],
            "wet_clean_relevant": True,
        },
        "required_parts": [
            {
                "part_number": "TCP-GEN-2200",
                "description": "TCP RF Generator Match Assembly",
                "quantity_on_hand": 2,
                "status": "IN_STOCK",
                "lead_time_days": 0,
                "storage_location": "CRIB-A3",
            }
        ],
        "calibration_status": {
            "component": "TCP_GENERATOR",
            "status": "CURRENT",
            "next_due": "2026-09-01",
            "last_date": "2026-03-01",
            "notes": "No calibration concerns.",
        },
        "past_wo_patterns": (
            "Similar past WOs show TCP power drift resolved by wet clean + "
            "generator recalibration."
        ),
        "recommended_actions": [],
        "draft_wo_header": (
            "WO TYPE: Corrective Maintenance\nChamber: ETCH-04\n"
            "Component: TCP_GENERATOR\nPriority: HIGH\n"
            "Fault: tcp_top_pwr upward trend, wet clean overdue by 9 wafers"
        ),
        "llm_narrative": (
            "TCP generator power is trending upward on ETCH-04 with wet clean "
            "9 wafers overdue, a known contributing factor. Recommend running "
            "the wet clean and verifying generator match tuning before the next lot."
        ),
    }

    demo_sop_report = (
        "SENSOR: tcp_top_pwr\nCHAMBER: ETCH-04\nALERT TYPE: TREND\nPRIORITY: HIGH\n\n"
        "RELEVANT INFORMATION:\n"
        "1. TCP Generator Wet Clean & Calibration Procedure\n"
        "   Perform chamber wet clean followed by generator match calibration.\n"
        "   Source: SOP — SOP-147 — \"TCP Generator Wet Clean & Calibration\""
    )

    result = run_supervisor_agent(
        demo_alert, demo_quality_report, demo_recommendation, demo_sop_report
    )
    print_supervisor_summary(result)
