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
            default_floors = {"0":85,"20":90,"35":90,"40":96}  # LOCK SPEC floors
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
                                          required_aeo_ft_per_nm: float =  300.0,
                                          allow_ab: bool = False,
                                          debug: bool = False,
                                          compare_all: bool = True,
                                          search_1pct: bool = True) -> Dict[str, Any]:

        tried: List[CandidateResult] = []
        flaps_priority = [0, 20, 40]

        default_floors = {"0":85,"20":90,"35":90,"40":96}  # LOCK SPEC floors
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
    filtered.setdefault("required_aeo_ft_per_nm", 300.0)
    filtered.setdefault("allow_ab", False)
    filtered.setdefault("debug", False)
    filtered.setdefault("compare_all", True)
    filtered.setdefault("search_1pct", True)
    return CorePlanner().plan_takeoff_with_optional_derate(**filtered)


# ============================================================================
# Compatibility facade for UI — exposes perf_compute_takeoff() used by UI
# ============================================================================
def perf_compute_takeoff(gw_lb: float,
                         field_elev_ft: float,
                         oat_c: float,
                         headwind_kts: float = 0.0,
                         runway_slope: float = 0.0,
                         thrust_mode: str = "MIL",
                         mode: str = "AUTO",
                         config: str = "TO_FLAPS",
                         sweep_deg: float = 20.0,
                         stores: list | None = None) -> dict:
    """Returns a dict with UI-expected keys:
       - GroundRoll_ft (optional; 0 if unknown)
       - DistanceTo35ft_ft
       - ASDR_ft
       - TODR_OEI_35ft_ft
       - Vs_kts, VR_kts, V2_kts
       - Dispatchable (bool)
       Notes:
       * headwind/slope not yet modeled in core; preserved for signature compatibility.
       * 'config' maps to flap_deg: CLEAN->0, TO_PARTIAL->20, TO_FLAPS->40
       * thrust_mode: "MIL" or "MAX" (AB). If 'mode' == 'DERATE', DERATE search is used.
    """
    cfg_to_flap = {"CLEAN":0, "TO_PARTIAL":20, "PARTIAL":20, "TO_FLAPS":40, "FLAPS":40, "FULL":40}
    flap_deg = cfg_to_flap.get(str(config).upper(), 20)

    # Determine planning mode
    allow_ab = str(thrust_mode).upper().startswith("MAX")
    derate_requested = (str(mode).upper() == "DERATE")
    planner = CorePlanner()

    # For now, provide TORA/ASDA as very large numbers; UI passes runway lengths elsewhere.
    # The planner will only use req_aeo and will mark dispatchability based on those and distances.
    # We return distances regardless.
    tora_ft = 999999
    asda_ft = 999999

    # Plan — if DERATE mode, search 1% increments; else MIL/AB fixed.
    if derate_requested:
        res = planner.plan_takeoff_with_optional_derate(
            flap_deg=flap_deg, gw_lbs=gw_lb,
            field_elev_ft=field_elev_ft, qnh_inhg=29.92, oat_c=oat_c,
            tora_ft=tora_ft, asda_ft=asda_ft,
            required_aeo_ft_per_nm=300.0, allow_ab=False,  # DERATE search is MIL only
            debug=False, compare_all=False, search_1pct=True
        )
        best = res.get("best") or {}
        vs = best.get("vs_kts") or best.get("Vs_kts") or 0.0
        v1 = best.get("v1_kts") or best.get("V1_kts") or 0.0
        vr = best.get("vr_kts") or best.get("Vr_kts") or 0.0
        v2 = best.get("v2_kts") or best.get("V2_kts") or 0.0
        asd = best.get("asd_ft") or best.get("ASD_ft") or 0.0
        d35 = best.get("todr_ft") or best.get("TODR_ft") or 0.0
        dispatch = bool(best.get("dispatchable", True))
    else:
        # MIL or AB plan with no derate search
        mode_str = "MILITARY" if not allow_ab else "AFTERBURNER"
        cand = planner._build_candidate(flap_deg, mode_str, gw_lb,
                                        field_elev_ft, 29.92, oat_c,
                                        tora_ft, asda_ft, 300.0, None, False)
        vs = cand.vs_kts; v1 = cand.v1_kts; vr = cand.vr_kts; v2 = cand.v2_kts
        asd = cand.asd_ft; d35 = cand.todr_ft; dispatch = cand.dispatchable

    # Build UI-compatible dict
    out = {
        "GroundRoll_ft": 0,                         # unknown -> let UI fallback to 1.15*GR or d35*1.10
        "DistanceTo35ft_ft": float(d35 or 0.0),
        "ASDR_ft": float(asd or 0.0),
        "TODR_OEI_35ft_ft": float(d35 or 0.0),     # lacking OEI-specific deck, reuse d35
        "Vs_kts": float(vs or 0.0),
        "VR_kts": float(vr or 0.0),
        "V2_kts": float(v2 or 0.0),
        "Dispatchable": bool(dispatch),
    }
    return out
