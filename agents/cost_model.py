"""
Cost / Impact Model — reference constants for the Impact Agent (Agent 5)
=========================================================================
Pipeline position:
  Production → Quality → Maintenance → SOP → [Impact Agent uses THIS module]

What this file is
-----------------
The Impact Agent estimates scrap, rework, material loss, energy, and carbon for
a fault. To turn a physical deviation into a cost you need two things:

  1. The FAULT IDENTITY — which machine and which sensor produced the error, and
     how far out of range it is. This arrives LIVE with each fault event (from
     the Production Agent alert); it is never pre-loaded.
  2. CONVERSION FACTORS — $/kWh, kg CO2/kWh, wafer value, gas price, etc. These
     are EXTERNAL reference values (public sources, cited below). They are
     reviewed/tuned by an engineer and are the ONLY place external numbers enter
     the estimate. The LLM never invents them.

Why the model is keyed by (machine, sensor)
--------------------------------------------
A deviation means a physically different thing on each sensor, so each routes to
a different cost pathway:

  - power sensors (W)      → extra electrical energy   → $ + CO2   (ENERGY pathway)
  - gas-flow sensors (sccm)→ wasted / over-used gas     → $         (MATERIAL pathway)
  - pressure (mTorr/Torr)  → no direct consumable       → cost is SCRAP-driven only

And the machine sets the tool-level facts the conversion needs (wafer size,
wafer value, baseline energy per wafer).

IMPORTANT
---------
Every constant below carries a value, a plausible range, and a source. The
placeholder values marked "SET TO YOUR FAB'S ACTUAL" are the ones only your
finance/facilities data can pin down — the public web only gives ranges. Treat
all outputs built on these as MODELED ESTIMATES WITH STATED ASSUMPTIONS.
"""

# ── Machines ────────────────────────────────────────────────────────────────
# Tool-level facts. Chamber ids map to a machine so a fault on CHA/CHB resolves
# to the right wafer size and per-wafer economics.

MACHINES = {
    "LAM_9600_TCP": {
        "description":        "LAM Research 9600 TCP metal etch, 200 mm Al etch",
        "wafer_diameter_mm":  200,
        # Value of one FINISHED 200 mm wafer if scrapped. Blank 200 mm substrate is
        # only ~$27–98; a processed wafer is worth far more (processing adds value).
        # SET TO YOUR FAB'S ACTUAL wafer/lot value.
        "wafer_value_usd":    {"value": 1500.0, "range": [500.0, 5000.0],
                                "source": "SET TO YOUR FAB'S ACTUAL "
                                          "(blank 200mm $27–98 per reclaim market)"},
        # Reference energy consumed per 200 mm wafer (whole-tool, not just RF).
        "energy_per_wafer_kwh": {"value": 361.0, "range": [250.0, 400.0],
                                 "source": "ScienceDirect S2666445323000041 "
                                           "(8-inch wafer ≈ 361 kWh, ≈264 kg CO2)"},
    },
}

CHAMBER_TO_MACHINE = {
    "CHA": "LAM_9600_TCP",
    "CHB": "LAM_9600_TCP",
}


# ── Sensors ─────────────────────────────────────────────────────────────────
# Per-sensor: the deviation unit and which cost pathway it drives.
#   pathway "energy"     → deviation is extra power (W); integrate over cycle time
#   pathway "material"   → deviation is wasted gas flow (sccm); price the excess
#   pathway "scrap_only" → no direct consumable; impact comes only from scrap risk

SENSORS = {
    "tcp_top_pwr": {"unit": "W",     "pathway": "energy",
                    "note": "TCP top power — plasma density; deviation = extra RF watts"},
    "rf_btm_pwr":  {"unit": "W",     "pathway": "energy",
                    "note": "RF bottom power — ion energy; deviation = extra RF watts"},
    "bcl3_flow":   {"unit": "sccm",  "pathway": "material", "gas": "BCl3",
                    "note": "BCl3 mass-flow; deviation = over/under gas delivery"},
    "cl2_flow":    {"unit": "sccm",  "pathway": "material", "gas": "Cl2",
                    "note": "Cl2 mass-flow; deviation = over/under gas delivery"},
    "pressure":    {"unit": "mTorr", "pathway": "scrap_only",
                    "note": "chamber pressure — impact is scrap-risk only"},
    "he_press":    {"unit": "Torr",  "pathway": "scrap_only",
                    "note": "He backside pressure — impact is scrap-risk only"},
}


# ── Shared conversion constants ────────────────────────────────────────────────
# Not machine- or sensor-specific. Each: value, range, unit, source.

SHARED = {
    "electricity_usd_per_kwh": {
        "value": 0.086, "range": [0.06, 0.15], "unit": "USD/kWh",
        "source": "US industrial avg ~8.56 c/kWh (2025); up to 30% of fab opex",
    },
    "grid_kgco2_per_kwh": {
        "value": 0.40, "range": [0.01, 0.96], "unit": "kgCO2/kWh",
        "source": "IPCC/IEA: coal 0.961, gas 0.483, renewables <0.01; set per your grid",
    },
    # Price of process gas per standard-litre-minute delivered for one hour. Rough
    # placeholder — SET FROM YOUR GAS CONTRACT (BCl3/Cl2 differ).
    "gas_usd_per_slm_hour": {
        "value": 2.0, "range": [0.5, 10.0], "unit": "USD/(slm·hr)",
        "source": "SET TO YOUR FAB'S ACTUAL gas contract price",
    },
    # Fallback wafer cycle time when the live value is not supplied on the event.
    # Prefer the actual cycle_time_sec from the stream when available.
    "cycle_time_sec_default": {
        "value": 110.0, "range": [90.0, 130.0], "unit": "s/wafer",
        "source": "typical single-wafer etch cycle; OVERRIDE with live value",
    },
}


# ── Accessors ───────────────────────────────────────────────────────────────

def machine_for_chamber(chamber_id: str) -> dict:
    """Resolve a chamber id to its machine parameter block (or None)."""
    key = CHAMBER_TO_MACHINE.get(chamber_id)
    return MACHINES.get(key) if key else None


def sensor_spec(sensor: str) -> dict:
    """Per-sensor unit + cost pathway (or None if the sensor is unknown)."""
    return SENSORS.get(sensor)


def const(name: str) -> float:
    """Value of a SHARED constant by name."""
    return SHARED[name]["value"]


# ── Pure conversions (deterministic — the LLM never does arithmetic) ──────────

def energy_kwh_from_power_delta(delta_w: float,
                                cycle_time_sec: float,
                                n_wafers: float) -> float:
    """
    Extra electrical energy (kWh) from running `n_wafers` at `delta_w` watts above
    set-point for `cycle_time_sec` each.  kWh = W · s · wafers / 3.6e6.
    """
    return (delta_w * cycle_time_sec * n_wafers) / 3_600_000.0


def usd_from_kwh(kwh: float) -> float:
    return kwh * const("electricity_usd_per_kwh")


def kgco2_from_kwh(kwh: float) -> float:
    return kwh * const("grid_kgco2_per_kwh")


def usd_from_scrap(chamber_id: str, n_wafers: float) -> float:
    """Cost of scrapping `n_wafers` finished wafers on this chamber's machine."""
    m = machine_for_chamber(chamber_id)
    wafer_value = m["wafer_value_usd"]["value"] if m else 1500.0
    return wafer_value * n_wafers


def usd_from_gas_excess(delta_sccm: float,
                        duration_hours: float) -> float:
    """
    Cost of gas over-delivered at `delta_sccm` (≈ slm/1000) for `duration_hours`.
    Material pathway for gas-flow sensors.
    """
    slm = abs(delta_sccm) / 1000.0
    return slm * duration_hours * const("gas_usd_per_slm_hour")


# ── Standalone sanity demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: tcp_top_pwr +50 W over set-point, 20 wafers at risk, 110 s each.
    spec = sensor_spec("tcp_top_pwr")
    print(f"tcp_top_pwr → pathway={spec['pathway']} unit={spec['unit']}")
    kwh = energy_kwh_from_power_delta(delta_w=50, cycle_time_sec=110, n_wafers=20)
    print(f"  extra energy : {kwh:.4f} kWh")
    print(f"  extra cost   : ${usd_from_kwh(kwh):.4f}")
    print(f"  extra carbon : {kgco2_from_kwh(kwh):.4f} kg CO2")
    print(f"  scrap 3 wafers on CHA: ${usd_from_scrap('CHA', 3):,.2f}")
