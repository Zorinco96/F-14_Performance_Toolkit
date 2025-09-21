# f14_takeoff_core.py — v1.5.1
# - AEO uses *ambient* atmosphere from QNH & OAT (not ISA-only)
# - Engine deck lookup uses computed Pressure Altitude (PA = elev + (29.92-QNH)*1000)
# - Keeps 1% DERATE search, flap priority, runway scaling, Mach falloff, install drag

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import math

from takeoff_model import TakeoffDeck
from derate import DerateModel
from f14_aero import F14Aero
from engine_f110 import F110Deck

S_WING_FT2 = 565.0
S_WING_M2 = S_WING_FT2 * 0.09290304
KTS_TO_MS = 0.514444

GAMMA = 1.4
R_AIR = 287.05
LAPSE = 0.0065          # K/m
T0_K = 288.15           # ISA sea-level temperature
INHG_TO_PA = 3386.389

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    """PA ≈ Elev + (29.92 - QNH)*1000 (standard practical formula)."""
    return float(field_elev_ft + (29.92 - float(qnh_inhg)) * 1000.0)

def ambient_atm(field_elev_ft: float, qnh_inhg: float, oat_c: float) -> Dict[str, float]:
    """
    Use QNH (sea-level pressure) and elevation to compute station pressure,
    combine with OAT to get density and speed of sound.
    """
    h_m = float(field_elev_ft) * 0.3048
    p0 = float(qnh_inhg) * INHG_TO_PA  # sea-level pressure from QNH
    # Barometric formula with standard lapse to get station pressure from p0
    # p = p0 * (1 - L*h/T0)^(g/(R*L))
    expo = 9.80665 / (R_AIR * LAPSE)
    p = p0 * pow(max(1.0 - LAPSE * h_m / T0_K, 0.5), expo)
    T = float(oat_c) + 273.15
    rho = p / (R_AIR * T)
    a = math.sqrt(GAMMA * R_AIR * T)
    return {"p_Pa": p, "T_K": T, "rho": rho, "a": a}

@dataclass
class CandidateResult:
    flap_deg: int
    thrust_mode: str
    derate_pct: Optional[int]
    clamped_to_floor: bool
    v1_kts: float
    vr_kts: float
    v2_kts: float
    vs_kts: float
    asd_ft: float
    todr_ft: float
    aeo_grad_ft_per_nm: float
    limiter: str
    dispatchable: bool
    v1_mode: str
    _debug: Optional[Dict[str, float]] = None

class CorePlanner:
    def __init__(self):
        self.deck = TakeoffDeck()
        self.derate = DerateModel()
        self.aero = F14Aero()
        self.eng = F110Deck()
        self.k_mach = 0.35  # MIL thrust falloff with Mach

    # ---------- Helpers ----------
    def _map_flap_to_config(self, flap_deg: int) -> str:
        if flap_deg == 0: return "CLEAN"
        if flap_deg == 20: return "TO_PARTIAL"
        if flap_deg in (35, 40): return "TO_FLAPS"
        return "CLEAN"

    def _cd0_increment(self, config: str) -> float:
        if config == "CLEAN": return 0.005
        if config in ("TO_PARTIAL", "TO_FLAPS"): return 0.010
        return 0.0

    def _aeo_gradient(self, flap_deg: int, gw_lbs: float,
                      field_elev_ft: float, qnh_inhg: float, oat_c: float,
                      V2_kts: float,
                      thrust_mode: str, applied_pct: Optional[int],
                      want_debug: bool = False) -> (float, Dict[str,float]):
        Vref_kts = V2_kts + 15.0

        amb = ambient_atm(field_elev_ft, qnh_inhg, oat_c)
        rho = amb["rho"]; a = amb["a"]

        # Convert KIAS to KTAS via rho; then to m/s
        Vtas = Vref_kts * KTS_TO_MS * math.sqrt(1.225 / rho)
        M = Vtas / a

        # Engine thrust with Mach falloff; use *pressure altitude* for deck lookup
        pa_ft = pressure_altitude_ft(field_elev_ft, qnh_inhg)
        power = "MIL" if thrust_mode in ("MILITARY", "DERATE") else "MAX"
        T_per_lbf = self.eng.thrust_lbf(pa_ft, M, power)
        T_per_lbf = T_per_lbf * max(0.7, (1.0 - self.k_mach * M))

        thrust_mult = (applied_pct / 100.0) if (thrust_mode == "DERATE" and applied_pct) else 1.0
        T_tot_N = T_per_lbf * 2 * 4.4482216153 * thrust_mult

        # Aero polar
        config = self._map_flap_to_config(flap_deg)
        clmax, cd0, k = self.aero.polar(config, 20.0)
        cd0 += self._cd0_increment(config)

        W_N = gw_lbs * 4.4482216153
        q = 0.5 * rho * Vtas * Vtas
        CL_req = W_N / max(q * S_WING_M2, 1e-8)
        CL = min(CL_req, clmax)
        CD = cd0 + k * CL * CL
        D_N = q * S_WING_M2 * CD

        excess = max(0.0, T_tot_N - D_N)
        grad = 6076.0 * (excess / max(W_N, 1.0))

        debug = {}
        if want_debug:
            debug = {"rho": float(rho), "Vtas_ms": float(Vtas), "Mach": float(M),
                     "T_per_lbf": float(T_per_lbf), "thrust_mult": float(thrust_mult),
                     "T_tot_N": float(T_tot_N), "clmax": float(clmax),
                     "cd0": float(cd0), "k": float(k), "q": float(q),
                     "CL_req": float(CL_req), "CL_used": float(CL), "CD_used": float(CD),
                     "D_N": float(D_N), "excess_N": float(excess),
                     "grad_ft_per_nm": float(grad), "Vref_kts": float(Vref_kts),
                     "p_Pa": float(amb["p_Pa"]), "T_K": float(amb["T_K"]), "a_mps": float(a),
                     "PA_ft": float(pa_ft)}
        return grad, debug

    def _scale_runway_for_derate(self, asd_ft: float, todr_ft: float, applied_pct: float) -> (float, float):
        r = max(0.5, min(1.0, applied_pct / 100.0))
        asd_scale = 1.0 / (r ** 1.1)
        todr_scale = 1.0 / (r ** 1.6)
        return asd_ft * asd_scale, todr_ft * todr_scale

    def _build_candidate(self, flap_deg:int, mode:str, gw_lbs:float,
                         field_elev_ft: float, qnh_inhg: float, oat_c: float,
                         tora_ft:float, asda_ft:float, req_aeo:float,
                         derate_pct: Optional[int], want_debug:bool) -> CandidateResult:
        # Use PA for table lookup (matches deck expectations)
        pa_ft = pressure_altitude_ft(field_elev_ft, qnh_inhg)
        base_thrust = "MILITARY" if mode in ("MILITARY", "DERATE") else "AFTERBURNER"
        pt = self.deck.lookup(flap_deg, base_thrust, gw_lbs, pa_ft, oat_c)

        asd = float(pt.ASD_ft); todr = float(pt.TODR_ft)
        clamped=False; resolved_pct=None; applied_pct=None
        if mode=="DERATE":
            if derate_pct is None: derate_pct = 95
            resolved_pct = int(derate_pct)
            default_floors = {"0":85,"20":88,"35":90,"40":90}
            floor = int(self.derate.cfg.get("min_pct_by_flap_deg", default_floors).get(str(flap_deg),85))
            if resolved_pct < floor: clamped = True
            applied_pct = max(resolved_pct, floor)
            asd, todr = self._scale_runway_for_derate(asd, todr, applied_pct)

        aeo, dbg = self._aeo_gradient(flap_deg, gw_lbs, field_elev_ft, qnh_inhg, oat_c,
                                      pt.V2_kts, base_thrust, applied_pct, want_debug)
        limiter = "BALANCED" if abs(asd - todr) / max(todr, 1.0) < 0.02 else ("ASD" if asd > todr else "TODR")
        dispatch = bool((asd <= asda_ft) and (todr <= tora_ft) and (aeo >= req_aeo))

        return CandidateResult(
            flap_deg=int(flap_deg), thrust_mode=mode, derate_pct=resolved_pct,
            clamped_to_floor=bool(clamped),
            v1_kts=float(pt.V1_kts), vr_kts=float(pt.Vr_kts), v2_kts=float(pt.V2_kts), vs_kts=float(pt.Vs_kts),
            asd_ft=asd, todr_ft=todr, aeo_grad_ft_per_nm=float(aeo),
            limiter=str(limiter), dispatchable=dispatch, v1_mode="table",
            _debug=dbg if want_debug else None
        )

    def plan_takeoff_with_optional_derate(self,
                                          flap_deg: int,
                                          gw_lbs: float,
                                          field_elev_ft: float,
                                          qnh_inhg: float,
                                          oat_c: float,
                                          tora_ft: float,
                                          asda_ft: float,
                                          required_aeo_ft_per_nm: float = 200.0,
                                          allow_ab: bool = False,
                                          debug: bool = False,
                                          compare_all: bool = True,
                                          search_1pct: bool = True) -> Dict[str, Any]:

        tried: List[CandidateResult] = []
        flaps_priority = [0, 20, 40]  # prefer lower flap; 40° last resort

        default_floors = {"0":85,"20":88,"35":90,"40":90}
        floors = self.derate.cfg.get("min_pct_by_flap_deg", default_floors)

        best: Optional[CandidateResult] = None

        for f in flaps_priority:
            floor = int(floors.get(str(f), 85))
            if search_1pct:
                for pct in range(floor, 101):
                    cand_der = self._build_candidate(f, "DERATE", gw_lbs,
                                                     field_elev_ft, qnh_inhg, oat_c,
                                                     tora_ft, asda_ft, required_aeo_ft_per_nm,
                                                     pct, debug)
                    if compare_all: tried.append(cand_der)
                    if cand_der.dispatchable:
                        best = cand_der
                        break
                if best:
                    break
            else:
                # heuristic path (unused now but kept for completeness)
                pa_ft = pressure_altitude_ft(field_elev_ft, qnh_inhg)
                mil = self.deck.lookup(f, "MILITARY", gw_lbs, pa_ft, oat_c)
                der_guess = self.derate.compute_derate_from_groundroll(f, pa_ft, mil.AGD_ft/1.15,
                                                                       min(tora_ft, asda_ft), allow_ab)
                cand_der = self._build_candidate(f, "DERATE", gw_lbs,
                                                 field_elev_ft, qnh_inhg, oat_c,
                                                 tora_ft, asda_ft, required_aeo_ft_per_nm,
                                                 der_guess.get("derate_pct"), debug)
                if compare_all: tried.append(cand_der)
                if cand_der.dispatchable:
                    best = cand_der
                    break

            # Try MIL at this flap
            cand_mil = self._build_candidate(f, "MILITARY", gw_lbs,
                                             field_elev_ft, qnh_inhg, oat_c,
                                             tora_ft, asda_ft, required_aeo_ft_per_nm,
                                             None, debug)
            if compare_all: tried.append(cand_mil)
            if cand_mil.dispatchable:
                best = cand_mil
                break

        if not best and allow_ab:
            for f in flaps_priority:
                cand_ab = self._build_candidate(f, "AFTERBURNER", gw_lbs,
                                                field_elev_ft, qnh_inhg, oat_c,
                                                tora_ft, asda_ft, required_aeo_ft_per_nm,
                                                None, debug)
                if compare_all: tried.append(cand_ab)
                if cand_ab.dispatchable:
                    best = cand_ab
                    break

        verdict = "OK" if best else "NOT_DISPATCHABLE"
        return {
            "inputs": {"gw_lbs": float(gw_lbs),
                       "field_elev_ft": float(field_elev_ft), "qnh_inhg": float(qnh_inhg),
                       "oat_c": float(oat_c), "tora_ft": int(tora_ft), "asda_ft": int(asda_ft)},
            "tried": [asdict(c) for c in tried],
            "best": asdict(best) if best else None,
            "verdict": verdict
        }

def plan_takeoff_with_optional_derate(**kwargs) -> Dict[str, Any]:
    accepted = {"flap_deg","gw_lbs","field_elev_ft","qnh_inhg","oat_c","tora_ft","asda_ft",
                "required_aeo_ft_per_nm","allow_ab","debug","compare_all","search_1pct"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    filtered.setdefault("required_aeo_ft_per_nm", 200.0)
    filtered.setdefault("allow_ab", False)
    filtered.setdefault("debug", False)
    filtered.setdefault("compare_all", True)
    filtered.setdefault("search_1pct", True)
    return CorePlanner().plan_takeoff_with_optional_derate(**filtered)
