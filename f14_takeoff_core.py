# f14_takeoff_core.py — v1.2.0-core-skel
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math


# ---- Constants (placeholders; refine later) ----
F14B_BEW_LB = 43735.0 # basic empty weight placeholder
EXT_TANK_EMPTY_LB = 1100.0 # each, structural weight only
ISA_LAPSE_C_PER_FT = 1.98 / 1000.0


# -------- Atmosphere / conversions --------
def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: Optional[float]) -> float:
"""Compute pressure altitude from field elevation and QNH inHg.
PA(ft) ≈ elev + (29.92 - QNH)*1000. If QNH is None, assume 29.92 (=> PA=elev).
"""
if qnh_inhg is None:
return float(field_elev_ft)
return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0




def density_ratio_sigma(press_alt_ft: float, oat_c: float) -> float:
"""ISA-based density ratio σ = δ/θ (troposphere approx).
Uses ft-based approximation: δ ≈ (1 - 6.87535e-6 * h)^4.2561 ;
θ = (T/T0) with T0=288.15 K and T = (ISA at PA) adjusted to OAT.
This is a pragmatic placeholder adequate for v1.2 scaffolding.
"""
h = max(0.0, float(press_alt_ft))
delta = (1.0 - 6.87535e-6 * h) ** 4.2561
# ISA temp at altitude (°C)
isa_c = 15.0 - ISA_LAPSE_C_PER_FT * h
theta = (float(oat_c) + 273.15) / (isa_c + 273.15)
sigma = max(0.2, min(1.2, delta / theta))
return sigma




def tas_from_ias_kt(ias_kt: float, sigma: float) -> float:
"""Approx TAS from IAS and density ratio: TAS ≈ IAS / sqrt(σ)."""
s = max(0.1, float(sigma))
return float(ias_kt) / math.sqrt(s)




# -------- Weight & Balance --------
def build_loadout_totals(
stations: Dict[str, Dict[str, object]],
fuel_lb: float,
ext_tanks: Tuple[bool, bool],
mode_simple_gw: Optional[float] = None,
) -> Dict[str, float]:
"""
Aggregate gross weights. Placeholder CG/trim mapping.


stations: { name: { 'store_weight_lb': float, 'pylon_weight_lb': float, 'qty': int } }
ext_tanks: (left_has_tank, right_has_tank) — structural weight only here.


Returns keys:
'gw_tow_lb','gw_ldg_lb','zf_weight_lb','fuel_tow_lb','fuel_ldg_lb',
'cg_percent_mac','stab_trim_units'
"""
if mode_simple_gw is not None:
gw_tow = max(0.0, float(mode_simple_gw))
# Assume simple-mode fuel is as provided; landing fuel placeholder 3000 lb
fuel_tow = max(0.0, float(fuel_lb))
fuel_ldg = min(fuel_tow, 3000.0)
zfw = gw_tow - fuel_tow
else:
# Sum detailed station weights
stores = 0.0
pylons = 0.0
for _, d in stations.items():
qty = int(d.get('qty', 0) or 0)
stores += qty * float(d.get('store_weight_lb', 0.0) or 0.0)
pylons += (qty > 0) * float(d.get('pylon_weight_lb', 0.0) or 0.0)
tank_struct = (EXT_TANK_EMPTY_LB if ext_tanks[0] else 0.0) + (EXT_TANK_EMPTY_LB if ext_tanks[1] else 0.0)
zfw = F14B_BEW_LB + stores + pylons + tank_struct
fuel_tow = max(0.0, float(fuel_lb))
return {'dispatchable': 'YES' if ok else 'NO', 'limiter': limiting, 'note': note}
