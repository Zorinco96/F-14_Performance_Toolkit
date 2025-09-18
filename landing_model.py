
from __future__ import annotations
import math
from typing import Dict, Optional, Literal
from isa import isa_atm
from f14_aero import F14Aero

S_WING_FT2 = 565.0
S_WING_M2 = S_WING_FT2 * 0.09290304
G = 9.80665

def landing_performance(
    weight_lbf: float,
    alt_ft: float,
    oat_c: float,
    headwind_kts: float,
    mode: Literal["DCS","FAA"] = "DCS",
    calib: Optional[Dict[str, Dict[str,float]]] = None,
    config: str = "LDG_FLAPS",
    sweep_deg: float = 20.0
):
    T, p, rho, a = isa_atm(alt_ft)
    rho *= max(0.6, 1.0 - 0.0032*(oat_c - 15.0))
    aero = F14Aero()
    CLmax, CD0, k = aero.polar(config, sweep_deg)

    W = weight_lbf * 4.4482216153
    Vs = ((2*W)/(rho*S_WING_M2*CLmax))**0.5
    Vref = 1.3*Vs
    TAS = max(0.1, Vref - headwind_kts*0.514444)

    def get(name, default):
        if calib and name in calib: return calib[name].get(mode, default)
        return default
    mu_b = get("BrakeCoefficient", 0.30 if mode=="DCS" else 0.40)
    brake_lag = get("BrakeLag_sec", 0.5 if mode=="DCS" else 0.3)
    anti_skid = get("AntiSkidEffectiveness", 0.85 if mode=="DCS" else 0.95)
    spoiler_cd0 = get("SpoilerBrake_CD0", 0.008 if mode=="DCS" else 0.006)

    s_air = Vref*4.0
    q = 0.5*rho*Vref*Vref
    CD = CD0 + spoiler_cd0
    D = q*S_WING_M2*CD
    a_decel = (mu_b * W * anti_skid + D) / (W/9.80665)
    t_brake = Vref/max(0.5, a_decel)
    s_brake = 0.5*Vref*t_brake

    return {
        "Vref_kts": Vref/0.514444,
        "Airborne_ft": s_air/0.3048,
        "GroundRoll_ft": s_brake/0.3048,
        "Total_ft": (s_air + s_brake)/0.3048
    }
