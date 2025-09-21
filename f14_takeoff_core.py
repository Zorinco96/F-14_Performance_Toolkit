# ============================================================
# F-14 Performance Calculator for DCS World — Core Logic
# File: f14_takeoff_core.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
from typing import Dict, Any, List
import time

import derate
import takeoff_model

# ------------------------------------------------------------
# Utility: simple stab trim estimator (placeholder)
# ------------------------------------------------------------
def estimate_stab_trim(flap_deg: int, gw_lbs: float) -> float:
    base = {0: 0.0, 20: 2.0, 35: 4.0}
    return base.get(int(flap_deg), 0.0)

# ------------------------------------------------------------
# Main Planner
# ------------------------------------------------------------
def plan_takeoff(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Plan a takeoff based on current inputs.
    Applies balanced-field V1 solver, runway factor, and climb gate.
    """
    start = time.time()
    gw_lbs = float(inputs.get("gw_lbs", 60000))
    oat_c = float(inputs.get("oat_c", 15))
    qnh_inhg = float(inputs.get("qnh_inhg", 29.92))
    headwind = float(inputs.get("wind_spd", 0.0))
    elev = float(inputs.get("field_elevation_ft", 0.0))
    tora = float(inputs.get("tora_ft", 10000))
    asda = float(inputs.get("asda_ft", 10000))

    runway_factor = derate.runway_factor()
    allow_ab = derate.allow_ab()

    candidate_ladder: List[Dict[str, Any]] = []
    status = None
    final_plan: Dict[str, Any] = {}

    for flap in [0, 20, 35]:
        for thrust_req in [85, 90, 96, 100, 110]:  # 110 ~ AB placeholder
            if thrust_req > 100 and not allow_ab:
                continue
            clamp = derate.clamp_derate_pct(thrust_req, flap)
            applied_pct = clamp.applied_pct

            sol = takeoff_model.solve_bfv1({
                "gw_lbs": gw_lbs,
                "flap_deg": flap,
                "thrust_pct": applied_pct,
                "oat_c": oat_c,
                "qnh_inhg": qnh_inhg,
                "headwind_kt": headwind,
                "field_elevation_ft": elev,
            })

            asd_req = sol["asd_ft_raw"] * runway_factor
            tod_req = sol["todr_ft_raw"] * runway_factor

            candidate = {
                "flap": flap, "thrust_req": thrust_req, "applied_pct": applied_pct,
                "clamped": clamp.clamped_to_floor,
                "v1": sol["v1"], "asd_ft_raw": sol["asd_ft_raw"], "todr_ft_raw": sol["todr_ft_raw"],
                "asd_ft_req": asd_req, "todr_ft_req": tod_req,
                "converged": sol["converged"]
            }

            # Gate checks
            if asd_req <= asda and tod_req <= tora:
                climb_grad = 350 if applied_pct >= 96 else 280  # placeholder gradient ft/nm
                if climb_grad >= 300:
                    status = "Balanced" if sol["converged"] else "Approx Balanced"
                    final_plan = candidate
                    break
                else:
                    candidate["gate_fail"] = "Climb"
            else:
                if asd_req > asda and tod_req > tora:
                    candidate["gate_fail"] = "ASDR & TODR"
                elif asd_req > asda:
                    candidate["gate_fail"] = "ASDR"
                elif tod_req > tora:
                    candidate["gate_fail"] = "TODR"
            candidate_ladder.append(candidate)
        if status:
            break

    if not status:
        status = "No dispatchable config"
        if candidate_ladder:
            final_plan = candidate_ladder[-1]

    plan = {
        "status": status,
        "V1": round(final_plan.get("v1", 0)),
        "Vr": round(final_plan.get("v1", 0) + 10),
        "V2": round(final_plan.get("v1", 0) + 20),
        "Vfs": round(final_plan.get("v1", 0) + 40),
        "flaps": "UP" if final_plan.get("flap",0)==0 else ("MAN" if final_plan.get("flap",0)==20 else "FULL"),
        "thrust": f"{final_plan.get('applied_pct','--')}%",
        "stab_trim": estimate_stab_trim(final_plan.get("flap",0), gw_lbs),
        "rpm_pct": final_plan.get("applied_pct"),
        "ff_pph_per_engine": int(10000 * final_plan.get("applied_pct",85)/100),  # crude mapping
        "_debug": {
            "candidates": candidate_ladder,
            "runway_factor": runway_factor,
            "elapsed_sec": round(time.time()-start,3)
        }
    }
    return plan
