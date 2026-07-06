"""
Impact Agent (Agent 5) — scrap-cost projection for an ACTIVE anomaly
=====================================================================
Pipeline position:
  Production → Quality → Maintenance → SOP → [THIS AGENT]

Goal (locked with the user)
  When the Production Agent flags an ACTIVE anomaly on a chamber, estimate how
  much SCRAP it will cost if that faulty chamber keeps running uncorrected for
  the REST OF THE CURRENT LOT.

      expected_cost = remaining_faulty_chamber_wafers_in_lot × scrap_cost_per_wafer

Design decisions (all deliberate)
  - Trigger      : active anomaly (not a trend/breach — the line is already bad).
  - Attribution  : the FAULTY CHAMBER's wafers only. A wafer runs on exactly one
                   chamber (verified in train_summary: 1 chamber per wafer), so a
                   fault on CHA cannot damage a wafer that ran on CHB. If CHB is
                   also faulty, that is a SEPARATE anomaly with its own estimate.
  - Horizon      : remainder of the CURRENT lot (lots run contiguously in time and
                   are ~21–25 wafers, split ~50/50 across the two chambers).
  - Affected     : ALL remaining faulty-chamber wafers, each FULLY SCRAPPED (worst
                   case "if nothing changes"; etch damage is largely irreversible).
  - Cost basis   : full processed-wafer value (material embedded), a RANGE, from
                   the online-sourced cited constants in cost_model.py.
  - Output       : a MODELED ESTIMATE reported as a range, never false precision.

Why wafers, not clock time
  Scrap is caused per-wafer; an idle tool costs nothing. So the axis is wafers,
  and the horizon is "wafers left in this lot on this chamber", not hours.

Design philosophy (mirrors the other agents)
  Python does the deterministic counting and arithmetic (these are FACTS from the
  lot roster). The LLM only writes the supervisor-facing narrative.
"""

import os
from pathlib import Path

import pandas as pd

try:
    from cost_model import (machine_for_chamber, StaticScrapPriceProvider,
                            ScrapPriceProvider)
except ImportError:  # allow "python agents/impact_agent.py" and package import
    from agents.cost_model import (machine_for_chamber, StaticScrapPriceProvider,
                                   ScrapPriceProvider)

_ROOT = Path(__file__).parent.parent
SUMMARY_PATH = str(_ROOT / "data" / "train_summary.csv")


# ── Lot roster ────────────────────────────────────────────────────────────────

def load_roster(summary_path: str = SUMMARY_PATH) -> pd.DataFrame:
    """
    Load the wafer roster (train_summary) used to count how many wafers remain in
    a lot on a given chamber. Sorted by run_start so 'remaining' is well-defined.
    Kept as a small, explicit dependency: in a live system this is replaced by the
    tool's cassette/lot plan, but the interface (lot_id, chamber_id, order) is the
    same.
    """
    df = pd.read_csv(summary_path, usecols=["wafer_id", "lot_id", "chamber_id", "run_start"])
    df["_rs"] = pd.to_datetime(df["run_start"], errors="coerce")
    return df.sort_values("_rs").reset_index(drop=True)


# ── Deterministic count (no LLM) ────────────────────────────────────────────────

def remaining_faulty_chamber_wafers(wafer_id: int,
                                    lot_id: str,
                                    chamber_id: str,
                                    roster: pd.DataFrame) -> dict:
    """
    Count the wafers still to be processed in THIS lot on THIS (faulty) chamber,
    INCLUDING the current wafer (it already triggered the anomaly, so it is bad).

    Returns a dict with the counts and the wafer ids, or a 'not_found' flag if the
    current wafer isn't in the roster.
    """
    lot = roster[roster["lot_id"] == lot_id]
    if lot.empty:
        return {"found": False, "reason": f"lot {lot_id} not in roster"}

    cur = lot[lot["wafer_id"] == wafer_id]
    if cur.empty:
        return {"found": False, "reason": f"wafer {wafer_id} not in lot {lot_id}"}
    cur_rs = cur["_rs"].iloc[0]

    # Wafers in this lot, on the faulty chamber, at or after the current wafer's run.
    remaining = lot[(lot["chamber_id"] == chamber_id) & (lot["_rs"] >= cur_rs)]

    lot_chamber_total = int((lot["chamber_id"] == chamber_id).sum())
    return {
        "found":               True,
        "remaining_count":     int(len(remaining)),
        "remaining_wafer_ids": remaining["wafer_id"].tolist(),
        "lot_chamber_total":   lot_chamber_total,   # total wafers this lot runs on this chamber
        "lot_total":           int(len(lot)),        # total wafers in the lot (both chambers)
    }


def estimate_scrap_impact(alert: dict,
                          roster: pd.DataFrame,
                          price_provider: ScrapPriceProvider = None) -> dict:
    """
    Full deterministic estimate for one active anomaly. Pure Python — no LLM.

    Reads the fault identity (wafer_id, lot_id, chamber_id) from the alert, counts
    remaining faulty-chamber wafers in the lot, and multiplies by the ranged
    scrap-cost-per-wafer supplied by `price_provider`.

    price_provider : where the per-wafer cost comes from. Defaults to the static
                     (cited-constant) provider. Swap in WebSearchScrapPriceProvider
                     later to source it live — the counting/arithmetic below is
                     unaffected.

    Returns a structured dict ready for the UI / narrator.
    """
    if price_provider is None:
        price_provider = StaticScrapPriceProvider()

    wafer_id   = int(alert["wafer_id"])
    chamber_id = alert.get("chamber_id") or alert.get("chamber", "UNKNOWN")
    lot_id     = alert.get("lot_id", "UNKNOWN")

    machine = machine_for_chamber(chamber_id)
    counts  = remaining_faulty_chamber_wafers(wafer_id, lot_id, chamber_id, roster)
    # Context is passed through so a future web-search provider can build its query.
    price_context = {
        "sensor":      alert.get("sensor"),
        "alert_type":  alert.get("alert_type"),
        "machine":     (machine or {}).get("description"),
    }
    cost = price_provider.scrap_cost_per_wafer(chamber_id, context=price_context)

    result = {
        "sensor":      alert.get("sensor", "UNKNOWN"),
        "alert_type":  alert.get("alert_type", "ANOMALY"),
        "wafer_id":    wafer_id,
        "lot_id":      lot_id,
        "chamber_id":  chamber_id,
        "machine":     (machine or {}).get("description", "UNKNOWN"),
        "per_wafer_cost": {
            "value": cost["value"], "low": cost["low"], "high": cost["high"],
            "source": cost["source"],
        },
        "assumptions": [
            "All remaining faulty-chamber wafers in the lot are fully scrapped.",
            "Attribution is to the faulty chamber only (1 wafer = 1 chamber).",
            "Horizon = remainder of the current lot.",
            "Per-wafer cost is a public-data estimate (material embedded), not fab-actual.",
        ],
    }

    if not counts["found"]:
        result.update({"estimable": False, "reason": counts["reason"]})
        return result

    n = counts["remaining_count"]
    result.update({
        "estimable":            True,
        "remaining_wafers":     n,
        "lot_chamber_total":    counts["lot_chamber_total"],
        "lot_total":            counts["lot_total"],
        "expected_cost": {
            "value": round(n * cost["value"], 2),
            "low":   round(n * cost["low"], 2),
            "high":  round(n * cost["high"], 2),
        },
    })
    return result


# ── LLM narrator (optional — mirrors the other agents) ──────────────────────────

SYSTEM_PROMPT = """
You are the Impact Agent in a semiconductor etch pipeline. You are given a
DETERMINISTIC scrap-cost estimate that has ALREADY been computed in Python:
the faulty chamber, the number of wafers remaining in the current lot on that
chamber, the per-wafer scrap cost (a range), and the total cost (a range).

Write a 2–4 sentence supervisor-facing summary. Rules:
  - Use the numbers EXACTLY as given. Never invent or alter a number.
  - Lead with the APPROXIMATE cost (the given midpoint), then show the tight
    range in parentheses as the confidence band, e.g. "approximately $18,000
    (est. $13,500–$22,500)". Do not present a huge/uncertain spread.
  - State the core assumption: this is what it costs if the faulty chamber keeps
    running uncorrected for the rest of the current lot, with every remaining
    wafer treated as scrap.
  - Do not recommend a fix or add steps — impact only.
Output plain text, no markdown.
""".strip()


def narrate(estimate: dict, client=None, model: str = None) -> str:
    """
    Optional LLM narration of the deterministic estimate. If no client is given,
    returns a plain deterministic sentence (no API call) so the agent is fully
    usable without the LLM.
    """
    if not estimate.get("estimable"):
        return (f"Impact not estimable for wafer {estimate.get('wafer_id')}: "
                f"{estimate.get('reason', 'unknown')}.")

    ec = estimate["expected_cost"]
    det = (f"If {estimate['chamber_id']} keeps running the {estimate['sensor']} "
           f"fault uncorrected for the rest of lot {estimate['lot_id']}, the "
           f"{estimate['remaining_wafers']} remaining wafers on that chamber would "
           f"be scrapped — approximately ${ec['value']:,.0f} "
           f"(est. ${ec['low']:,.0f}–${ec['high']:,.0f}).")

    if client is None:
        return det

    user_msg = (
        f"Faulty chamber : {estimate['chamber_id']} ({estimate['machine']})\n"
        f"Sensor / fault : {estimate['sensor']} [{estimate['alert_type']}]\n"
        f"Lot            : {estimate['lot_id']}\n"
        f"Remaining wafers on this chamber (incl. current): {estimate['remaining_wafers']}\n"
        f"Per-wafer scrap cost: ${estimate['per_wafer_cost']['low']:,.0f}"
        f"–${estimate['per_wafer_cost']['high']:,.0f} "
        f"(mid ${estimate['per_wafer_cost']['value']:,.0f})\n"
        f"TOTAL expected scrap cost: ${ec['low']:,.0f}–${ec['high']:,.0f} "
        f"(mid ${ec['value']:,.0f})\n"
    )
    resp = client.chat.completions.create(
        model=model or os.getenv("AZURE_DEPLOYMENT"),
        max_tokens=250,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
    )
    return resp.choices[0].message.content.strip()


# ── Main pipeline function ──────────────────────────────────────────────────────

def run_impact_agent(alert: dict,
                     quality_report: str = None,
                     recommendation: dict = None,
                     sop_report: str = None,
                     roster: pd.DataFrame = None,
                     client=None,
                     price_provider: ScrapPriceProvider = None) -> dict:
    """
    Agent 5 entry point. Called by the Production Agent after the SOP Agent.

    Consistent with the chain, it accepts the accumulated upstream outputs, but the
    scrap estimate only needs the Production alert (wafer_id, lot_id, chamber_id).

    price_provider : source of the per-wafer cost (default: static cited constants).
                     Pass a WebSearchScrapPriceProvider() later to source it live.

    Returns the deterministic estimate dict with an added 'narrative' field.
    """
    if roster is None:
        roster = load_roster()
    estimate = estimate_scrap_impact(alert, roster, price_provider=price_provider)
    estimate["narrative"] = narrate(estimate, client=client)
    return estimate


# ── Standalone demo (real data, no LLM) ─────────────────────────────────────────

if __name__ == "__main__":
    roster = load_roster()

    # Sample active anomalies (shape matches the Production Agent's normalized alert).
    demo_alerts = [
        {"alert_type": "ANOMALY", "wafer_id": 2901, "lot_id": "LOT_29A",
         "chamber_id": "CHA", "sensor": "tcp_top_pwr", "severity": "HIGH"},
        {"alert_type": "ANOMALY", "wafer_id": 2913, "lot_id": "LOT_29A",
         "chamber_id": "CHA", "sensor": "rf_btm_pwr", "severity": "HIGH"},
        {"alert_type": "ANOMALY", "wafer_id": 2902, "lot_id": "LOT_29A",
         "chamber_id": "CHB", "sensor": "pressure", "severity": "MEDIUM"},
    ]

    for a in demo_alerts:
        est = run_impact_agent(a, roster=roster)   # no client → deterministic narrative
        print("=" * 72)
        print(f"ALERT: {a['sensor']} on {a['chamber_id']} (wafer {a['wafer_id']}, {a['lot_id']})")
        if est["estimable"]:
            print(f"  remaining {a['chamber_id']} wafers in lot: {est['remaining_wafers']} "
                  f"(of {est['lot_chamber_total']} this chamber runs; lot total {est['lot_total']})")
            ec = est["expected_cost"]
            print(f"  per-wafer scrap: ${est['per_wafer_cost']['low']:,.0f}"
                  f"–${est['per_wafer_cost']['high']:,.0f}")
            print(f"  EXPECTED SCRAP COST: ${ec['low']:,.0f}–${ec['high']:,.0f} "
                  f"(mid ${ec['value']:,.0f})")
        else:
            print(f"  not estimable: {est['reason']}")
        print(f"\n  {est['narrative']}")
