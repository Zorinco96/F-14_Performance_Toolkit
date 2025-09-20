
from __future__ import annotations
import math
from typing import Literal, Dict
from isa import isa_atm
from engine_f110 import F110Deck
from f14_aero import F14Aero

def knots_to_mps(kts: float) -> float:
    return kts * 0.514444

def mps_to_kts(mps: float) -> float:
    return mps / 0.514444

def fps_to_mps(fps: float) -> float:
    return fps * 0.3048

def mps_to_fps(mps: float) -> float:
    return mps / 0.3048

def ground_takeoff_run(
    weight_lbf: float,
    alt_ft: float,
    oat_c: float,
    headwind_kts: float,
    runway_slope: float,
    config: Literal["TO_FLAPS","CLEAN"] = "TO_FLAPS",
    sweep_deg: float = 20.0,
    power: Literal["MIL","MAX"] = "MAX",
    mode: Literal["DCS","FAA"] = "DCS",
    calib: Dict[str, Dict[str,float]] | None = None,
    vr_kts: float | None = None,
    dt: float = 0.1
):
    """
    Minimal integrate-to-VR + estimate distance to 35 ft AGL.
    Returns dict with VR, ground_roll_ft, distance_to_35ft_ft.
    """
    eng = F110Deck()
    aero = F14Aero()
    T, p, rho, a = isa_atm(alt_ft)
    # simple density correction for non-ISA OAT
    # (small effect; full ISA variant later)
    rho *= max(0.7, 1.0 - 0.0032*(oat_c - 15.0))

    # Calibration defaults
    def get(name, default):
        if calib and name in calib:
            return calib[name][mode]
        return default

    mu_r = get("RollingResistance", 0.025 if mode=="DCS" else 0.020)
    ground_lift_frac = get("GroundLiftFraction", 0.25 if mode=="DCS" else 0.20)
    gear_cd0 = get("GearDragDelta_CD0", 0.015 if mode=="DCS" else 0.010)
    rot_lag = get("RotationLag_sec", 2.5 if mode=="DCS" else 1.5)

    # Aero polar
    CLmax, CD0, k = aero.polar(config, sweep_deg)
    CD0 += gear_cd0  # gear down during roll

    # Integrate ground roll until VR
    V = max(0.1, knots_to_mps(max(0.0, headwind_kts)))  # TAS start ~ headwind
    s = 0.0  # meters
    t = 0.0
    W = weight_lbf * 4.4482216153  # N
    m = W / 9.80665

    # If VR unknown, set VR ~ 1.1 * stall (placeholder; will be refined/calibrated)
    if vr_kts is None:
        # Estimate stall from CLmax at takeoff
        Vstall = math.sqrt((2*W) / (rho * (CLmax) * (565.0*0.09290304)))
        VR = 1.1 * Vstall
    else:
        VR = knots_to_mps(vr_kts)

    while V < VR:
        M = V / a
        T_eng = eng.thrust_lbf(alt_ft, M, power)
        T_total = T_eng * 2 * 4.4482216153  # N

        # Approx ground lift (wing carries fraction of weight with flaps)
        L_ground = ground_lift_frac * W
        N = max(0.0, W - L_ground) * math.cos(math.atan(runway_slope))  # normal force
        F_rr = mu_r * N

        # Drag at small CL (in ground run)
        CL = 0.4 * CLmax  # heuristic during roll
        CD = CD0 + k * CL**2
        q = 0.5 * rho * V**2
        D = q * (565.0*0.09290304) * CD

        # Net force
        F_net = T_total - D - F_rr - W*math.sin(math.atan(runway_slope))
        a_mps2 = max(0.0, F_net / m)
        V += a_mps2 * dt
        s += V * dt
        t += dt

        # Simple failsafe
        if t > 120.0:
            break

    ground_roll_ft = s / 0.3048

    # Rotation + liftoff + climb to 35 ft (very compact placeholder)
    # Add rotation-lag distance ~ average speed * rot_lag, plus airborne
    s_rot = V * rot_lag  # meters
    # Airborne distance to 35ft: approximate with 3-second segment at ~VR speed
    s_air = V * 3.0
    distance_35_ft = (s + s_rot + s_air) / 0.3048

    return {
        "VR_kts": mps_to_kts(VR),
        "GroundRoll_ft": ground_roll_ft,
        "DistanceTo35ft_ft": distance_35_ft,
        "Time_s": t + rot_lag + 3.0,
    }
