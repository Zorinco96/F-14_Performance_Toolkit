# ============================================================
# perf_core_v2.py — Core orchestrator for F-14 Toolkit
# Built from scratch per LOCK SPEC
# ============================================================

from __future__ import annotations
import math
from typing import Dict, Any, Optional, List

# --- Hard imports from your repo modules
from derate import DerateModel
from takeoff_model import TakeoffDeck
from landing_model import landing_performance
from cruise_model import cruise_point
from climb_model import climb_profile, aeo_gradient_ft_per_nm_to_1000
from f14_aero import F14Aero
from engine_f110 import F110Deck
from isa import isa_atm

__version__ = "v2.0.0"

# ===============================
# Constants / Policy (LOCK SPEC)
# ===============================
DERATE_FLOORS = {"UP": 85, "MAN": 90, "FULL": 96}
DERATE_ABS_MIN = 85
CLIMB_FLOOR_FTPNM = 300.0

# Map flaps mode → degrees
FLAP_MAP = {"UP": 0, "MAN": 20, "FULL": 40}


# ===============================
# Utilities
# ===============================
def _pressure_alt_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    """Convert field elev + QNH to pressure altitude (ft)."""
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0


# ===============================
# Takeoff
# ===============================
def compute_takeoff(inputs: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    gw   = float(inputs.get("gw_lbs", 60000))
    oat  = float(inputs.get("oat_c", 15))
    qnh  = float(inputs.get("qnh_inhg", 29.92))
    elev = float(inputs.get("field_elev_ft", 0))
    head = float(inputs.get("headwind_kts_component", 0))
    tora = float(inputs.get("tora_ft", 999999))
    asda = float(inputs.get("asda_ft", 999999))
    flap_mode = str(inputs.get("flap_mode", "MAN")).upper()
    thrust_pref = str(inputs.get("thrust_pref", "Auto-Select"))
    manual_pct = inputs.get("manual_derate_pct", None)

    pa_ft = _pressure_alt_ft(elev, qnh)
    flap_deg = FLAP_MAP.get(flap_mode, 20)

    # Enforce derate floors
    floor_pct = DERATE_FLOORS.get(flap_mode, 90)
    if manual_pct is not None:
        pct = max(int(manual_pct), floor_pct, DERATE_ABS_MIN)
    else:
        pct = floor_pct

    deck = TakeoffDeck()
    result = None
    resolved_label = "?"

    if thrust_pref == "Manual AB":
        result = deck.lookup(flap_deg, "AFTERBURNER", gw, pa_ft, oat)
        resolved_label = "MAX AB"

    elif thrust_pref == "Manual MIL":
        result = deck.lookup(flap_deg, "MILITARY", gw, pa_ft, oat)
        resolved_label = "MIL"

    elif thrust_pref == "Manual DERATE":
        dm = DerateModel()
        dres = dm.clamp_pct(pct, flap_mode)
        # Derate is modeled as MIL w/ thrust multiplier
        result = deck.lookup(flap_deg, "MILITARY", gw, pa_ft, oat)
        resolved_label = f"DERATE {dres['derate_pct']}%"

    else:  # Auto-Select
        mil = deck.lookup(flap_deg, "MILITARY", gw, pa_ft, oat)
        ab = deck.lookup(flap_deg, "AFTERBURNER", gw, pa_ft, oat)

        # Check dispatchability: ASDA/TORA + climb floor
        def _dispatchable(r):
            asdr_ok = r.ASD_ft <= asda and r.TODR_ft <= tora
            climb_grad = aeo_gradient_ft_per_nm_to_1000(gw, r.V2_kts, "MIL", None)
            climb_ok = climb_grad >= CLIMB_FLOOR_FTPNM
            return asdr_ok and climb_ok

        if _dispatchable(mil):
            result = mil; resolved_label = "MIL"
        elif _dispatchable(ab):
            result = ab; resolved_label = "MAX AB"
        else:
            # Pick the lesser TODR
            result = mil if mil.TODR_ft <= ab.TODR_ft else ab
            resolved_label = "MIL" if result is mil else "MAX AB"

    return {
        "Vs_kts": result.Vs_kts,
        "V1_kts": result.V1_kts,
        "Vr_kts": result.Vr_kts,
        "V2_kts": result.V2_kts,
        "ASDR_ft": result.ASD_ft,
        "TODR_OEI_35ft_ft": result.TODR_ft,
        "Dispatchable": True,  # already filtered in auto-select
        "ResolvedThrustLabel": resolved_label,
        "Flaps": flap_mode,
        "GW_lbs": gw,
        "source": "TakeoffDeck.lookup",
        "version": __version__,
    }


# ===============================
# Landing
# ===============================
def compute_landing(inputs: Dict[str, Any]) -> Dict[str, Any]:
    flap = str(inputs.get("flap_setting", "FULL"))
    gw = float(inputs.get("gross_weight_lbs", 52000))
    pa = float(inputs.get("pressure_alt_ft", 0))
    oat_c = float(inputs.get("oat_c", 15))
    headwind = float(inputs.get("headwind_kt", 0))

    res = landing_performance(gw, pa, oat_c, headwind, flap_mode=flap)
    return {
        "Vref_kts": res.get("Vref_kts"),
        "ldr_unfactored_ft": res.get("GroundRoll_ft"),
        "source": "landing_performance",
        "version": __version__,
    }


# ===============================
# Cruise
# ===============================
def compute_cruise(inputs: Dict[str, Any]) -> Dict[str, Any]:
    gw = float(inputs.get("gross_weight_lbs", 55000))
    di = float(inputs.get("drag_index", 0))

    best = None
    for alt in range(20000, 41000, 2000):
        for mach in [round(x * 0.01, 2) for x in range(60, 96, 5)]:
            res = cruise_point(gw, alt, mach, power="MIL")
            sr = res.get("SpecificRange_nm_per_lb", 0)
            if best is None or sr > best["sr"]:
                best = {"alt": alt, "mach": mach, "sr": sr}

    return {
        "optimum_alt_ft": best["alt"],
        "optimum_mach": best["mach"],
        "source": "cruise_point",
        "version": __version__,
    }


# ===============================
# Climb
# ===============================
def compute_climb(inputs: Dict[str, Any]) -> Dict[str, Any]:
    profile = str(inputs.get("profile", "Economy"))
    respect_250 = bool(inputs.get("respect_250", True))
    gw = float(inputs.get("gross_weight_lbs", 55000))

    res = climb_profile(gw, 0, 30000, profile=profile, respect_250=respect_250)
    return {
        "profile": profile,
        "respect_250": respect_250,
        "schedule": res.get("schedule"),
        "source": "climb_profile",
        "version": __version__,
    }
