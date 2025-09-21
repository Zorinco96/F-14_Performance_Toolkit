# f14_takeoff_core.py — v1.3.1a (tolerant wrapper)
# Same as v1.3.1, but the top-level function now accepts legacy kwargs
# (e.g., headwind_kts_component, runway_slope) and ignores unknown keys.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from takeoff_model import TakeoffDeck
from climb_model import aeo_gradient_ft_per_nm_to_1000
from derate import DerateModel

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

class CorePlanner:
    def __init__(self):
        self.deck = TakeoffDeck()
        self.derate = DerateModel()

    def compute_candidate(self,
                          flap_deg: int,
                          thrust_mode: str,
                          gw_lbs: float,
                          pa_ft: float,
                          oat_c: float,
                          tora_ft: float,
                          asda_ft: float,
                          required_aeo_ft_per_nm: float,
                          derate_pct: Optional[int] = None) -> CandidateResult:
        base_thrust = "MILITARY" if thrust_mode in ("MILITARY","DERATE") else "AFTERBURNER"
        pt = self.deck.lookup(flap_deg, base_thrust, gw_lbs, pa_ft, oat_c)

        asd = pt.ASD_ft
        todr = pt.TODR_ft
        clamped = False
        resolved_pct = None

        if thrust_mode == "DERATE":
            if derate_pct is None:
                derate_pct = 95
            resolved_pct = derate_pct
            floor = int(self.derate.cfg.get("min_pct_by_flap_deg",
                                            {"0":85,"20":88,"35":90,"40":90}).get(str(flap_deg),85))
            clamped = resolved_pct <= floor

        aeo = aeo_gradient_ft_per_nm_to_1000(gw_lbs, pa_ft, oat_c)

        limiter = "BALANCED" if abs(asd - todr) / max(todr, 1) < 0.02 else ("ASD" if asd > todr else "TODR")
        dispatch = (asd <= asda_ft) and (todr <= tora_ft) and (aeo >= required_aeo_ft_per_nm)

        return CandidateResult(
            flap_deg=flap_deg,
            thrust_mode=thrust_mode,
            derate_pct=resolved_pct,
            clamped_to_floor=clamped,
            v1_kts=pt.V1_kts,
            vr_kts=pt.Vr_kts,
            v2_kts=pt.V2_kts,
            vs_kts=pt.Vs_kts,
            asd_ft=asd,
            todr_ft=todr,
            aeo_grad_ft_per_nm=aeo,
            limiter=limiter,
            dispatchable=dispatch,
            v1_mode="table"
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
                                          allow_ab: bool = False) -> Dict[str, Any]:
        pa_ft = field_elev_ft
        tried: List[CandidateResult] = []
        best: Optional[CandidateResult] = None

        mil = self.deck.lookup(flap_deg, "MILITARY", gw_lbs, pa_ft, oat_c)
        der = self.derate.compute_derate_from_groundroll(flap_deg, pa_ft,
                                                         mil.AGD_ft/1.15,
                                                         min(tora_ft, asda_ft),
                                                         allow_ab)
        cand_der = self.compute_candidate(flap_deg, "DERATE", gw_lbs, pa_ft, oat_c,
                                          tora_ft, asda_ft,
                                          required_aeo_ft_per_nm,
                                          der["derate_pct"])
        tried.append(cand_der)
        if cand_der.dispatchable:
            best = cand_der
        else:
            cand_mil = self.compute_candidate(flap_deg, "MILITARY", gw_lbs, pa_ft, oat_c,
                                              tora_ft, asda_ft, required_aeo_ft_per_nm)
            tried.append(cand_mil)
            if cand_mil.dispatchable:
                best = cand_mil
            elif allow_ab:
                cand_ab = self.compute_candidate(flap_deg, "AFTERBURNER", gw_lbs, pa_ft, oat_c,
                                                 tora_ft, asda_ft, required_aeo_ft_per_nm)
                tried.append(cand_ab)
                if cand_ab.dispatchable:
                    best = cand_ab

        verdict = "OK" if best else "NOT_DISPATCHABLE"
        return {
            "inputs": {"flap_deg": flap_deg, "gw_lbs": gw_lbs,
                       "field_elev_ft": field_elev_ft, "qnh_inhg": qnh_inhg,
                       "oat_c": oat_c, "tora_ft": tora_ft, "asda_ft": asda_ft},
            "tried": [c.__dict__ for c in tried],
            "best": best.__dict__ if best else None,
            "verdict": verdict
        }

# ---- tolerant top-level wrapper ----
def plan_takeoff_with_optional_derate(**kwargs) -> Dict[str, Any]:
    """
    Tolerant wrapper that drops unknown legacy keys so older test_app pages keep working.
    Accepted keys: flap_deg, gw_lbs, field_elev_ft, qnh_inhg, oat_c, tora_ft, asda_ft,
                   required_aeo_ft_per_nm (optional), allow_ab (optional).
    Everything else is ignored.
    """
    accepted = {"flap_deg","gw_lbs","field_elev_ft","qnh_inhg","oat_c","tora_ft","asda_ft",
                "required_aeo_ft_per_nm","allow_ab"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    # defaults
    filtered.setdefault("required_aeo_ft_per_nm", 200.0)
    filtered.setdefault("allow_ab", False)
    return CorePlanner().plan_takeoff_with_optional_derate(**filtered)
