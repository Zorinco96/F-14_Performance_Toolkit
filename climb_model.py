# ============================================================
# F-14 Performance Calculator for DCS World — Climb Model
# File: climb_model.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
from typing import Dict, Any, List

def plan_climb(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Plan a climb profile based on mode and conditions.
    Inputs:
      gw_lbs: gross weight (lb)
      oat_c: OAT (C)
      qnh_inhg: QNH (inHg)
      profile: 'economy' or 'interceptor'
      respect_250: bool, enforce 250 KIAS below 10k
    """
    gw = float(inputs.get("gw_lbs", 60000))
    oat = float(inputs.get("oat_c", 15))
    profile = str(inputs.get("profile", "economy")).lower()
    respect_250 = bool(inputs.get("respect_250", True))

    # --- Placeholder simple climb performance model ---
    # This will later be calibrated against NATOPS / DCS data.
    segments: List[Dict[str, Any]] = []
    total_time_min = 0.0
    total_fuel = 0.0
    total_dist_nm = 0.0

    # Define climb schedule segments
    if profile == "economy":
        # Economy climb: slower speeds, less fuel burn
        sched = [
            {"from_ft": 35, "to_ft": 1000, "ias": "V2+15", "rpm": 90, "ff_pph_per_engine": 6000},
            {"from_ft": 1000, "to_ft": 10000, "ias": 250 if respect_250 else 300, "rpm": 92, "ff_pph_per_engine": 6500},
            {"from_ft": 10000, "to_ft": 30000, "mach": 0.70, "rpm": 85, "ff_pph_per_engine": 5000},
        ]
    else:
        # Interceptor climb: higher thrust, higher speed
        sched = [
            {"from_ft": 35, "to_ft": 1000, "ias": "V2+15", "rpm": 95, "ff_pph_per_engine": 7000},
            {"from_ft": 1000, "to_ft": 10000, "ias": 300, "rpm": 97, "ff_pph_per_engine": 8000},
            {"from_ft": 10000, "to_ft": 30000, "mach": 0.90, "rpm": 99, "ff_pph_per_engine": 9500},
        ]

    # Loop through schedule to build outputs
    for seg in sched:
        from_ft = seg.get("from_ft")
        to_ft = seg.get("to_ft")
        delta_ft = (to_ft - from_ft) if (from_ft is not None and to_ft is not None) else 0
        roc_fpm = 3000 if profile == "economy" else 5000
        time_min = delta_ft / roc_fpm / 60.0
        fuel_lb = time_min * seg.get("ff_pph_per_engine", 6000) * 2 / 60.0
        dist_nm = time_min * (seg.get("ias", seg.get("mach", 250)) if isinstance(seg.get("ias", None), (int,float)) else 250) / 60.0

        total_time_min += time_min
        total_fuel += fuel_lb
        total_dist_nm += dist_nm

        seg_out = {
            "From(ft)": from_ft,
            "To(ft)": to_ft,
            "IAS/Mach": seg.get("ias", seg.get("mach","--")),
            "RPM%": seg.get("rpm"),
            "FF/eng(pph)": seg.get("ff_pph_per_engine"),
            "ROC(fpm)": roc_fpm,
            "Time(min)": round(time_min,2),
            "Fuel(lb)": int(round(fuel_lb)),
            "Dist(nm)": round(dist_nm,1),
        }
        segments.append(seg_out)

    avg_roc = (30000/total_time_min) if total_time_min>0 else None

    return {
        "time_min": round(total_time_min,1),
        "fuel_lb": int(round(total_fuel)),
        "dist_nm": round(total_dist_nm,1),
        "avg_roc_fpm": int(round(avg_roc)) if avg_roc else None,
        "schedule": segments,
        "_debug": {
            "inputs": inputs,
            "profile": profile,
            "respect_250": respect_250,
            "gw": gw,
            "oat": oat,
        }
    }
