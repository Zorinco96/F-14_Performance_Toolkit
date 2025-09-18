# f14_takeoff_core.py — v1.2.0-core-skel (stable)
from __future__ import annotations

from typing import Dict, Tuple, Optional
import math

# -----------------------------
# Constants (placeholder values)
# -----------------------------
F14B_BEW_LB = 43_735.0          # Basic empty weight (approx placeholder)
EXT_TANK_EMPTY_LB = 1_100.0     # Structural weight per external tank, no fuel
ISA_LAPSE_C_PER_FT = 1.98 / 1000.0  # °C per foot

# Simple MAC/trim placeholders for UI wiring
DEFAULT_CG_PERCENT_MAC = 24.5
DEFAULT_STAB_TRIM_UNITS = 0.0


# ---------------------------------------
# Atmosphere / conversions (UI scaffolds)
# ---------------------------------------
def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: Optional[float]) -> float:
    """
    Compute pressure altitude from field elevation (ft) and QNH (inHg).
    PA(ft) ≈ elev + (29.92 - QNH) * 1000. If QNH is None, assume 29.92.
    """
    elev = float(field_elev_ft or 0.0)
    if qnh_inhg is None:
        return elev
    try:
        qnh = float(qnh_inhg)
    except Exception:
        qnh = 29.92
    return elev + (29.92 - qnh) * 1000.0


def density_ratio_sigma(press_alt_ft: float, oat_c: float) -> float:
    """
    ISA-based density ratio σ = δ/θ (troposphere approximation).
    δ ≈ (1 - 6.87535e-6 * h)^4.2561 with h in ft.
    θ = (T / T_ISA_at_h), where T = OAT in Kelvin, T_ISA_at_h = ISA temp at altitude.
    Clamped to a reasonable range to avoid UI weirdness.
    """
    h = max(0.0, float(press_alt_ft or 0.0))
    # Pressure ratio
    delta = (1.0 - 6.87535e-6 * h) ** 4.2561
    # ISA temperature at altitude (°C)
    isa_c = 15.0 - ISA_LAPSE_C_PER_FT * h
    # Temperature ratio
    try:
        theta = ((float(oat_c) + 273.15) / (isa_c + 273.15))
        if theta <= 0:
            theta = 1.0
    except Exception:
        theta = 1.0
    sigma = delta / theta
    return max(0.2, min(1.2, float(sigma)))


# ---------------------------------------
# Weight & Balance aggregation (scaffold)
# ---------------------------------------
def build_loadout_totals(
    stations: Dict[str, Dict[str, object]],
    fuel_lb: float,
    ext_tanks: Tuple[bool, bool],
    mode_simple_gw: Optional[float] = None,
) -> Dict[str, float]:
    """
    Aggregate gross weights for UI display.
    Returns keys used by the app:
      - 'gw_tow_lb'       : takeoff gross weight (lb)
      - 'gw_ldg_lb'       : landing gross weight (lb) [fuel landing = min(fuel_tow, 3000 lb)]
      - 'zf_weight_lb'    : zero-fuel weight (lb)
      - 'fuel_tow_lb'     : fuel at takeoff (lb)
      - 'fuel_ldg_lb'     : fuel at landing (lb) (placeholder 3000 lb cap)
      - 'cg_percent_mac'  : placeholder CG %MAC
      - 'stab_trim_units' : placeholder stabilizer trim units

    Notes:
      - In Simple mode, we trust the provided GW for takeoff and infer ZFW by subtracting fuel.
      - In Detailed mode, we sum station store/pylon weights + BEW + external tank structure.
      - External tank *fuel* is not counted here; app passes total fuel separately.
    """
    # Normalize inputs
    fuel_tow = max(0.0, float(fuel_lb or 0.0))
    fuel_ldg = min(fuel_tow, 3000.0)  # landing fuel placeholder
    left_tank, right_tank = bool(ext_tanks[0]), bool(ext_tanks[1])
    tank_struct = (EXT_TANK_EMPTY_LB if left_tank else 0.0) + (EXT_TANK_EMPTY_LB if right_tank else 0.0)

    if mode_simple_gw is not None:
        # Simple mode: user-provided takeoff GW; infer ZFW
        gw_tow = max(0.0, float(mode_simple_gw))
        zfw = max(0.0, gw_tow - fuel_tow)
        gw_ldg = max(0.0, zfw + fuel_ldg)
    else:
        # Detailed mode: sum stations (store + pylon), add BEW and tank structural weight
        stores = 0.0
        pylons = 0.0
        try:
            for _, d in (stations or {}).items():
                qty = int(d.get("qty", 0) or 0)
                store_w = float(d.get("store_weight_lb", 0.0) or 0.0)
                pylon_w = float(d.get("pylon_weight_lb", 0.0) or 0.0)
                if qty > 0:
                    stores += qty * store_w
                    pylons += pylon_w  # one pylon per occupied station (approx)
        except Exception:
            # If anything odd comes in, fall back gracefully
            stores = 0.0
            pylons = 0.0

        zfw = max(0.0, F14B_BEW_LB + stores + pylons + tank_struct)
        gw_tow = max(0.0, zfw + fuel_tow)
        gw_ldg = max(0.0, zfw + fuel_ldg)

    # Placeholder CG/trim for UI wiring
    cg_percent_mac = float(DEFAULT_CG_PERCENT_MAC)
    stab_trim_units = float(DEFAULT_STAB_TRIM_UNITS)

    return {
        "gw_tow_lb": float(gw_tow),
        "gw_ldg_lb": float(gw_ldg),
        "zf_weight_lb": float(zfw),
        "fuel_tow_lb": float(fuel_tow),
        "fuel_ldg_lb": float(fuel_ldg),
        "cg_percent_mac": cg_percent_mac,
        "stab_trim_units": stab_trim_units,
    }
