
from __future__ import annotations
import math
from typing import Dict, Literal, Optional
from isa import isa_atm
from engine_f110 import F110Deck
from f14_aero import F14Aero

S_WING_FT2 = 565.0
S_WING_M2 = S_WING_FT2 * 0.09290304

def climb_profile(
    weight_lbf: float,
    alt_start_ft: float,
    alt_end_ft: float,
    oat_dev_c: float = 0.0,
    schedule: Literal["NAVY","DISPATCH"] = "NAVY",
    mode: Literal["DCS","FAA"] = "DCS",
    power: Literal["MIL","MAX"] = "MIL",
    sweep_deg: float = 20.0,
    config: str = "CLEAN",
    dt: float = 1.0
):
    eng = F110Deck()
    aero = F14Aero()

    if schedule == "NAVY":
        IAS_kts = 350.0; Mach_target = 0.90
    else:
        IAS_kts = 300.0; Mach_target = 0.78

    alt = alt_start_ft
    t = 0.0; fuel = 0.0; dist_m = 0.0
    W = weight_lbf * 4.4482216153
    clmax, cd0, k = aero.polar(config, sweep_deg)

    while alt < alt_end_ft - 1.0:
        T, p, rho, a = isa_atm(alt)
        rho *= max(0.6, 1.0 - 0.0032*(oat_dev_c))
        Vias = IAS_kts * 0.514444
        Vtas = Vias * ( (1.225/rho) ** 0.5 )
        M = Vtas / a
        if M > Mach_target:
            M = Mach_target; Vtas = M * a

        thrust_each = eng.thrust_lbf(alt, M, power)
        Ttot = thrust_each * 2 * 4.4482216153
        q = 0.5*rho*Vtas*Vtas
        CL = W/(q*S_WING_M2)
        CD = cd0 + k*CL*CL
        D = q*S_WING_M2*CD
        excess = max(0.0, Ttot - D)
        roc = (excess * Vtas)/W  # m/s
        dh = max(0.1, roc) * dt
        alt += dh / 0.3048
        t += dt
        dist_m += Vtas * dt
        FF_each = eng.fuel_flow_pph(alt, M, power)
        fuel += (FF_each*2) * (dt/3600.0)
        if t > 7200: break

    return {
        "Time_s": t,
        "Fuel_lb": fuel,
        "Distance_nm": dist_m / 1852.0,
        "AvgROC_fpm": (alt_end_ft - alt_start_ft) / (t/60.0) if t>0 else 0.0
    }
