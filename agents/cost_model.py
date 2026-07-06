"""
Cost Model — reference constants for the Impact Agent (Agent 5)
================================================================
Scope (locked with the user):
  The Impact Agent estimates the SCRAP cost of continuing to run a FAULTY CHAMBER
  for the remainder of the current lot. Every affected wafer is treated as FULLY
  SCRAPPED, so the only cost constant needed is the value of one lost processed
  wafer — which already EMBEDS the material invested in it (substrate + processing
  + consumed process gas). We do NOT track material separately (that would
  double-count a fully-scrapped wafer).

Where the numbers come from
  These are EXTERNAL reference values sourced from public/online data (cited
  below), NOT measured from the line and NOT your internal financials. They are
  reviewed/tuned by an engineer and are the ONLY place external numbers enter the
  estimate. The LLM never invents or changes them. Every value carries a plausible
  RANGE, and all downstream outputs are reported as ranged, MODELED ESTIMATES.

  (Energy/carbon pathways were considered earlier but de-scoped — this model is
  scrap-only per the current goal.)
"""

# ── Machines ──────────────────────────────────────────────────────────────────
# Tool-level facts. A chamber id maps to a machine so a fault on CHA/CHB resolves
# to the right per-wafer scrap value.

MACHINES = {
    "LAM_9600_TCP": {
        "description":       "LAM Research 9600 TCP metal etch, 200 mm Al etch",
        "wafer_diameter_mm": 200,
        # Value of ONE finished 200 mm wafer if scrapped (material embedded).
        # Public anchors: a BLANK/reclaimed 200 mm substrate is only ~$27–98; a
        # PROCESSED wafer is worth far more (processing adds the bulk of the value).
        # The range below is a public-data placeholder — SET value TO YOUR FAB'S
        # ACTUAL processed-wafer/lot value when known.
        "scrap_cost_per_wafer_usd": {
            "value": 2000.0,
            "range": [1500.0, 2500.0],
            "source": "Approx. for a fully-processed 200mm mature/legacy-node wafer: "
                      "mature-node foundry wafers ~$3,000–4,000 at 300mm (Silicon "
                      "Analysts 2026, ~$3,200 avg); scaled to 200mm (~0.44x area) and "
                      "legacy Al metal-etch node -> ~$2,000 (+/-25%). Override with "
                      "your fab's actual processed-wafer value when known.",
        },
    },
}

CHAMBER_TO_MACHINE = {
    "CHA": "LAM_9600_TCP",
    "CHB": "LAM_9600_TCP",
}

# Fallback when a chamber isn't mapped — keeps the agent robust rather than crashing.
_DEFAULT_SCRAP = {"value": 2000.0, "range": [1500.0, 2500.0],
                  "source": "default (chamber not mapped to a machine)"}


# ── Accessors ─────────────────────────────────────────────────────────────────

def machine_for_chamber(chamber_id: str) -> dict:
    """Resolve a chamber id to its machine parameter block (or None)."""
    key = CHAMBER_TO_MACHINE.get(chamber_id)
    return MACHINES.get(key) if key else None


def scrap_cost_per_wafer(chamber_id: str) -> dict:
    """
    Full scrap cost of one wafer on this chamber's machine, as a dict:
        {"value": float, "low": float, "high": float, "source": str}
    Returns a safe default (with a note) if the chamber isn't mapped.
    """
    m = machine_for_chamber(chamber_id)
    spec = m["scrap_cost_per_wafer_usd"] if m else _DEFAULT_SCRAP
    return {
        "value":  spec["value"],
        "low":    spec["range"][0],
        "high":   spec["range"][1],
        "source": spec["source"],
    }


# ── Price providers (swappable source of the per-wafer cost) ────────────────────
# The Impact Agent depends on this INTERFACE, not on where the number comes from.
# Swapping the provider changes the SOURCE of the price without touching the
# agent's counting or arithmetic. Today we use the static (cited-constant) provider;
# later, drop in the web-search provider — a one-line change at the call site.

from typing import Optional


class ScrapPriceProvider:
    """Interface: return the scrap cost of ONE wafer as
    {'value': float, 'low': float, 'high': float, 'source': str}."""

    def scrap_cost_per_wafer(self, chamber_id: str,
                             context: Optional[dict] = None) -> dict:
        raise NotImplementedError


class StaticScrapPriceProvider(ScrapPriceProvider):
    """CURRENT approach — engineer-reviewed cited constants from this module.
    Deterministic and reproducible. Ignores `context`."""

    def scrap_cost_per_wafer(self, chamber_id: str,
                             context: Optional[dict] = None) -> dict:
        return scrap_cost_per_wafer(chamber_id)


class WebSearchScrapPriceProvider(ScrapPriceProvider):
    """
    FUTURE approach — fetch the price LIVE via OpenAI's Responses API built-in
    `web_search` tool, so the model itself sources the cost from the internet.

    NOT IMPLEMENTED YET. This is a drop-in seam: once implemented, switching is a
    one-line change — pass WebSearchScrapPriceProvider() instead of the static one.
    Requires an OpenAI PLATFORM key (the web_search tool is not on Azure chat).

    Intended implementation:

        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            input=self._build_query(chamber_id, context),
        )
        # then parse resp.output into {value, low, high, source(url)}
    """

    def __init__(self, client=None, model: str = "gpt-4o"):
        self.client = client          # an openai.OpenAI (Responses API) client
        self.model = model

    def _build_query(self, chamber_id: str, context: Optional[dict]) -> str:
        m = machine_for_chamber(chamber_id) or {}
        desc = m.get("description", "semiconductor wafer")
        sensor = (context or {}).get("sensor")
        fault_note = f" (fault: {sensor})" if sensor else ""
        return (f"Current USD cost to scrap one fully processed wafer from a "
                f"{desc}{fault_note}. Give a typical value and a low-high range, "
                f"and cite the source.")

    def scrap_cost_per_wafer(self, chamber_id: str,
                             context: Optional[dict] = None) -> dict:
        raise NotImplementedError(
            "WebSearchScrapPriceProvider is a placeholder. Implement it with the "
            "OpenAI Responses API web_search tool (needs an OpenAI platform key). "
            "See the class docstring. Until then, use StaticScrapPriceProvider."
        )


if __name__ == "__main__":
    for ch in ("CHA", "CHB", "CHZ"):
        c = scrap_cost_per_wafer(ch)
        print(f"{ch}: ${c['value']:,.0f}  (range ${c['low']:,.0f}–${c['high']:,.0f})")
