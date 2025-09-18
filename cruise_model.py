
from __future__ import annotations
from isa import isa_atm
from engine_f110 import F110Deck
from f14_aero import F14Aero

S_WING_FT2 = 565.0
S_WING_M2 = S_WING_FT2 * 0.09290304

def cruise_point(weight_lbf: float, alt_ft: float, mach: float, power: str = "MIL", sweep_deg: float = 20.0, config: str = "CLEAN"):
    eng = F110Deck(); aero = F14Aero()
    T, p, rho, a = isa_atm(alt_ft)
    V = mach * a
    q = 0.5*rho*V*V
    clmax, cd0, k = aero.polar(config, sweep_deg)
    L = weight_lbf * 4.4482216153
    CL = L/(q*S_WING_M2)
    CD = cd0 + k*CL*CL
    D = q*S_WING_M2*CD
    FF_each = eng.fuel_flow_pph(alt_ft, mach, power)
    TAS_kts = V/0.514444
    return {
        "Drag_lbf": D/4.4482216153,
        "FuelFlow_pph_total": FF_each*2,
        "TAS_kts": TAS_kts,
        "SpecificRange_nm_per_lb": (TAS_kts)/(FF_each*2) if FF_each>0 else 0.0
    }
