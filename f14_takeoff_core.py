# f14_takeoff_core.py — v1.6.0
# Adds a configurable drag increment model to depress optimistic performance:
# - Config-driven CD0 increments by flap config (CLEAN / TO_PARTIAL / TO_FLAPS)
# - Optional "stores" CD0 increment applied to all configs (placeholder, tunable)
# - Richer debug: report cd0_base, cd0_inc_config, cd0_inc_stores, cd0_total
# Keeps:
# - 1% DERATE search with flap priority (0 -> 20 -> 40)
# - Ambient atmosphere (QNH+OAT), PA engine lookup, Mach falloff, temp/density thrust sensitivity
# - Runway distance scaling under derate
#
# NOTE on "placeholders": defaults here are conservative and clearly labeled,
# meant to be calibration knobs; see DRAG_CFG below.

from __future__ import annotations

# --- Import path hardening for Streamlit/Cloud environments ---
import os as _os, sys as _sys
_here = _os.path.dirname(__file__)
if _here and _here not in _sys.path:
    _sys.path.insert(0, _here)
# Robust sibling imports (script vs package)
try:
    from takeoff_model import TakeoffDeck
except Exception:
    from .takeoff_model import TakeoffDeck  # type: ignore
try:
    from derate import DerateModel
except Exception:
    from .derate import DerateModel  # type: ignore
try:
    from f14_aero import F14Aero
except Exception:
    from .f14_aero import F14Aero  # type: ignore
try:
    from engine_f110 import F110Deck
except Exception:
    from .engine_f110 import F110Deck  # type: ignore

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
RHO0 = 1.225            # ISA SL density (kg/m^3)
INHG_TO_PA = 3386.389

# ---- Drag Increment Configuration (CALIBRATION KNOBS) ----
DRAG_CFG = {
    # Increment added to CD0 based on flap config mapping
    "base_cd0_inc": {
        "CLEAN": 0.010,       # baseline "installed" / pylons / bleed
        "TO_PARTIAL": 0.015,  # maneuver flaps down, more exposed bits
        "TO_FLAPS": 0.020     # full flaps, max incremental drag
    },
    # Global "stores" increment (e.g., pylons, tanks). Tune with DCS/NATOPS.
    "stores_cd0_inc": 0.000   # start at zero; raise to 0.005–0.015 if needed
}

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - float(qnh_inhg)) * 1000.0)

def ambient_atm(field_elev_ft: float, qnh_inhg: float, oat_c: float) -> Dict[str, float]:
    h_m = float(field_elev_ft) * 0.3048
    p0 = float(qnh_inhg) * INHG_TO_PA
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

        # MIL thrust Mach falloff factor
        self.k_mach = 0.35
        # Ambient thrust sensitivity exponents (tunable)
        self.a_sigma = 0.6
        self.b_theta = 0.2

        # Drag increments (can be externalized later)
        self.drag_cfg = DRAG_CFG.copy()

    # ---------- Helpers ----------
    def _map_flap_to_config(self, flap_deg: int) -> str:
        if flap_deg == 0: return "CLEAN"
        if flap_deg == 20: return "TO_PARTIAL"
        if flap_deg in (35, 40): return "TO_FLAPS"
        return "CLEAN"

    def _cd0_increment_config(self, config: str) -> float:
        base = self.drag_cfg.get("base_cd0_inc", {})
        return float(base.get(config, 0.0))

    def _cd0_increment_stores(self) -> float:
        return float(self.drag_cfg.get("stores_cd0_inc", 0.0))

    def _aeo_gradient(self, flap_deg: int, gw_lbs: float,
                      field_elev_ft: float, qnh_inhg: float, oat_c: float,
                      V2_kts: float,
                      thrust_mode: str, applied_pct: Optional[int],
                      want_debug: bool = False) -> (float, Dict[str,float]):
        Vref_kts = V2_kts + 15.0

        amb = ambient_atm(field_elev_ft, qnh_inhg, oat_c)
        rho = amb["rho"]; a = amb["a"]
        sigma = rho / RHO0
        theta = amb["T_K"] / T0_K

        # Convert KIAS to KTAS via rho; then to m/s
        Vtas = Vref_kts * KTS_TO_MS * math.sqrt(RHO0 / rho)
        M = Vtas / a

        # Engine thrust with Mach falloff; use *pressure altitude* for deck lookup
        pa_ft = pressure_altitude_ft(field_elev_ft, qnh_inhg)
        power = "MIL" if thrust_mode in ("MILITARY", "DERATE") else "MAX"
        T_per_lbf = self.eng.thrust_lbf(pa_ft, M, power)
        T_per_lbf = T_per_lbf * max(0.7, (1.0 - self.k_mach * M))

        # Ambient sensitivity (reduced thrust in hot/thin air)
        T_per_lbf = T_per_lbf * (sigma ** self.a_sigma) * (theta ** (-self.b_theta))

        thrust_mult = (applied_pct / 100.0) if (thrust_mode == "DERATE" and applied_pct) else 1.0
        T_tot_N = T_per_lbf * 2 * 4.4482216153 * thrust_mult

        # Aero polar with drag increments
        config = self._map_flap_to_config(flap_deg)
        clmax, cd0_base, k = self.aero.polar(config, 20.0)
        cd0_inc_cfg = self._cd0_increment_config(config)
        cd0_inc_store = self._cd0_increment_stores()
        cd0_total = cd0_base + cd0_inc_cfg + cd0_inc_store

        W_N = gw_lbs * 4.4482216153
        q = 0.5 * rho * Vtas * Vtas
        CL_req = W_N / max(q * S_WING_M2, 1e-8)
        CL = min(CL_req, clmax)
        CD = cd0_total + k * CL * CL
        D_N = q * S_WING_M2 * CD

        excess = max(0.0, T_tot_N - D_N)
        grad = 6076.0 * (excess / max(W_N, 1.0))

        debug = {}
        if want_debug:
            debug = {"rho": float(rho), "sigma": float(sigma), "theta": float(theta),
                     "Vtas_ms": float(Vtas), "Mach": float(M),
                     "T_per_lbf_final": float(T_per_lbf),
                     "thrust_mult": float(thrust_mult), "T_tot_N": float(T_tot_N),
                     "clmax": float(clmax),
                     "cd0_base": float(cd0_base),
                     "cd0_inc_config": float(cd0_inc_cfg),
                     "cd0_inc_stores": float(cd0_inc_store),
                     "cd0_total": float(cd0_total),
                     "k": float(k), "q": float(q),
                     "CL_req": float(CL_req), "CL_used": float(CL), "CD_used": float(CD),
                     "D_N": float(D_N), "excess_N": float(excess),
                     "grad_ft_per_nm": float(grad), "Vref_kts": float(Vref_kts),
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
        flaps_priority = [0, 20, 40]

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


# === Dict-based facade expected by UI (stable contract) ===
def perf_compute_takeoff(inputs: Dict[str, Any], overrides: Optional[Dict[str,Any]]=None) -> Dict[str, Any]:
    """
    UI-facing facade: accepts an inputs dict and optional overrides, returns a standard result dict.
    Leaves ASDR/TODR unfactored; UI applies any additional margins (e.g., +10%).
    """
    # Extract inputs with defaults
    gw = float(inputs.get("gw_lbs", inputs.get("gw_lb", 60000)))
    field_elev = float(inputs.get("field_elev_ft", 0.0))
    qnh = float(inputs.get("qnh_inhg", 29.92))
    oat = float(inputs.get("oat_c", 15.0))
    headwind = float(inputs.get("headwind_kts_component", 0.0))
    tora = float(inputs.get("tora_ft", 999999))
    asda = float(inputs.get("asda_ft", 999999))
    flap_mode = str(inputs.get("flap_mode", "MAN")).upper()
    thrust_pref = str(inputs.get("thrust_pref", "Auto-Select"))
    manual_pct = inputs.get("manual_derate_pct", None)

    # Overrides
    req_aeo = float((overrides or {}).get("climb_floor_ftpnm", 300.0))

    cfg_to_deg = {"UP":0, "MAN":20, "FULL":40}
    flap_deg = cfg_to_deg.get(flap_mode, 20)

    planner = CorePlanner()

    # Choose thrust pathway
    if thrust_pref == "Manual DERATE" or (thrust_pref == "Auto-Select" and manual_pct is not None):
        pct = int(manual_pct or 95)
        res = planner.plan_takeoff_with_optional_derate(
            flap_deg=flap_deg, gw_lbs=gw,
            field_elev_ft=field_elev, qnh_inhg=qnh, oat_c=oat,
            tora_ft=tora, asda_ft=asda, required_aeo_ft_per_nm=req_aeo,
            allow_ab=False, debug=False, compare_all=False, search_1pct=True
        )
        best = res.get("best") or {}
        vs  = best.get("Vs_kts") or 0.0
        v1  = best.get("V1_kts") or 0.0
        vr  = best.get("Vr_kts") or 0.0
        v2  = best.get("V2_kts") or 0.0
        asd = best.get("ASD_ft") or 0.0
        d35 = best.get("TODR_ft") or 0.0
        dispatch = bool(best.get("dispatchable", True))
        thrust_label = f"DERATE {pct}%"
    else:
        use_ab = (thrust_pref == "Manual AB")
        mode = "AFTERBURNER" if use_ab else "MILITARY"
        cand = planner._build_candidate(flap_deg, mode, gw, field_elev, qnh, oat, tora, asda, req_aeo, None, False)
        vs, v1, vr, v2 = cand.vs_kts, cand.v1_kts, cand.vr_kts, cand.v2_kts
        asd, d35, dispatch = cand.asd_ft, cand.todr_ft, cand.dispatchable
        thrust_label = "MAX AB" if use_ab else "MIL"

    return {
        "GroundRoll_ft": 0.0,
        "DistanceTo35ft_ft": float(d35 or 0.0),
        "ASDR_ft": float(asd or 0.0),
        "TODR_OEI_35ft_ft": float(d35 or 0.0),
        "Vs_kts": float(vs or 0.0),
        "Vr_kts": float(vr or 0.0),
        "V2_kts": float(v2 or 0.0),
        "Dispatchable": bool(dispatch),
        "ResolvedThrustLabel": thrust_label,
        "Flaps": flap_mode,
        "GW_lbs": gw,
    }

