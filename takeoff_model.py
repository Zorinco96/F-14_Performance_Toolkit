
from __future__ import annotations
import math
from typing import Dict, Literal, Optional
from isa import isa_atm
from engine_f110 import F110Deck
from f14_aero import F14Aero

S_WING_FT2 = 565.0
S_WING_M2 = S_WING_FT2 * 0.09290304
G = 9.80665

def kts_to_mps(kts: float) -> float: return kts * 0.514444
def mps_to_kts(mps: float) -> float: return mps / 0.514444
def m_to_ft(m: float) -> float: return m / 0.3048

def stall_speed_kt(weight_lbf: float, rho: float, clmax: float) -> float:
    W = weight_lbf * 4.4482216153
    Vs = math.sqrt((2*W)/(rho*S_WING_M2*clmax))
    return mps_to_kts(Vs)

def v_speeds(weight_lbf: float, rho: float, clmax: float, mode="DCS") -> Dict[str, float]:
    Vs = stall_speed_kt(weight_lbf, rho, clmax)
    VR = 1.10 * Vs
    VLOF = 1.13 * Vs
    V2 = 1.20 * Vs
    return {"Vs": Vs, "VR": VR, "VLOF": VLOF, "V2": V2}

def takeoff_run(
    weight_lbf: float,
    alt_ft: float,
    oat_c: float,
    headwind_kts: float,
    runway_slope: float,
    config: Literal["TO_FLAPS","CLEAN"] = "TO_FLAPS",
    sweep_deg: float = 20.0,
    power: Literal["MIL","MAX"] = "MAX",
    mode: Literal["DCS","FAA"] = "DCS",
    calib: Optional[Dict[str, Dict[str,float]]] = None,
    stores: Optional[list[str]] = None,
    dt: float = 0.05
):
    eng = F110Deck()
    aero = F14Aero()
    T, p, rho, a = isa_atm(alt_ft)
    rho *= max(0.7, 1.0 - 0.0032*(oat_c - 15.0))

    def get(name, default):
        if calib and name in calib: return calib[name].get(mode, default)
        return default

    mu_r = get("RollingResistance", 0.025 if mode=="DCS" else 0.020)
    ground_lift_frac = get("GroundLiftFraction", 0.25 if mode=="DCS" else 0.20)
    gear_cd0 = get("GearDragDelta_CD0", 0.015 if mode=="DCS" else 0.010)
    rot_lag = get("RotationLag_sec", 2.5 if mode=="DCS" else 1.5)
    pitch_rate_dps = get("RotationPitchRate_dps", 3.0 if mode=="DCS" else 4.0)
    thrust_scale_AB = get("ThrustScaleLowAlt", 0.95 if mode=="DCS" else 1.00)
    mil_scale = get("MilThrustScale", 0.98 if mode=="DCS" else 1.00)

    CLmax, CD0, k = aero.polar(config, sweep_deg)
    CD_misc = gear_cd0

    if stores:
        for s in stores:
            try:
                dCL, dCD0, dk = aero.polar(f"STORE_DELTA_{s}", 0)
                CLmax += dCL; CD_misc += dCD0; k += dk
            except Exception:
                pass

    vs = v_speeds(weight_lbf, rho, CLmax, mode=mode)
    VRtas = kts_to_mps(vs["VR"] - max(0.0, headwind_kts))
    VLOFtas = kts_to_mps(vs["VLOF"] - max(0.0, headwind_kts))

    W = weight_lbf * 4.4482216153
    m = W / G
    V = max(0.1, kts_to_mps(max(0.0, headwind_kts)))
    s = 0.0; t = 0.0

    while V < VRtas:
        M = V / a
        thrust_each = eng.thrust_lbf(alt_ft, M, power)
        if power in ["MAX","MIL"]:
            thrust_each *= (thrust_scale_AB if power=="MAX" else mil_scale)
        T_total = thrust_each * 2 * 4.4482216153
        Lg = ground_lift_frac * W
        N = max(0.0, W - Lg) * math.cos(math.atan(runway_slope))
        Frr = mu_r * N
        CL = 0.4*CLmax
        CD = CD0 + k*CL*CL + CD_misc
        q = 0.5*rho*V*V
        D = q*S_WING_M2*CD
        Fnet = T_total - D - Frr - W*math.sin(math.atan(runway_slope))
        a_mps2 = max(0.0, Fnet/m)
        V += a_mps2 * dt
        s += V * dt
        t += dt
        if t > 120: break

    ground_roll_ft = m_to_ft(s)
    pitch_rate = math.radians(pitch_rate_dps)
    time_rotate = max(1.0, (5.0/pitch_rate_dps))
    s_rot = V * (rot_lag + time_rotate)

    Vair = max(V, VLOFtas)
    M = Vair / a
    thrust_each = eng.thrust_lbf(alt_ft, M, power)
    if power in ["MAX","MIL"]:
        thrust_each *= (thrust_scale_AB if power=="MAX" else mil_scale)
    T_total = thrust_each * 2 * 4.4482216153
    CL_liftoff = 0.9*CLmax
    q = 0.5*rho*Vair*Vair
    CD_liftoff = CD0 + k*CL_liftoff*CL_liftoff + CD_misc
    D = q*S_WING_M2*CD_liftoff
    excess = max(0.0, T_total - D)
    roc = (excess * Vair)/W
    t_air = 10.7/max(0.1, roc)
    s_air = Vair*t_air

    return {
        "Vs_kts": vs["Vs"],
        "VR_kts": vs["VR"],
        "VLOF_kts": vs["VLOF"],
        "V2_kts": vs["V2"],
        "GroundRoll_ft": ground_roll_ft,
        "DistanceTo35ft_ft": m_to_ft((s + s_rot + s_air)),
        "TimeTo35ft_s": t + (rot_lag + time_rotate) + t_air
    }
