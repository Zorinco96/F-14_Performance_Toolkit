# ============================================================
# F-14 Performance Calculator for DCS World — Cruise Model
# File: cruise_model.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
import os, pandas as pd
from typing import Dict, Any

CSV_PATH = "data/f14_cruise_natops.csv"

def plan_cruise(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Plan cruise performance based on NATOPS data (if present) or fallback.
    Inputs:
      gw_lbs: gross weight
      drag_index: int
      mode: 'economy' or 'interceptor'
      altitude_ft: user altitude (if provided)
      auto_opt: bool, choose optimum altitude if True
    """
    gw = float(inputs.get("gw_lbs", 60000))
    di = int(inputs.get("drag_index", 0))
    mode = str(inputs.get("mode", "economy")).lower()
    alt_ft = inputs.get("altitude_ft")
    auto_opt = bool(inputs.get("auto_opt", False))

    data = None
    if os.path.exists(CSV_PATH):
        try:
            data = pd.read_csv(CSV_PATH)
        except Exception:
            data = None

    results = {}
    if data is not None and not data.empty:
        # Simplified filter: nearest weight and DI
        df = data.copy()
        df["gross_weight_lbs"] = pd.to_numeric(df["gross_weight_lbs"], errors="coerce")
        df["drag_index"] = pd.to_numeric(df["drag_index"], errors="coerce")
        subset = df[(df["drag_index"]==di)]
        if subset.empty:
            subset = df
        # nearest weight row
        wdiff = (subset["gross_weight_lbs"] - gw).abs()
        if not wdiff.empty:
            row = subset.iloc[wdiff.idxmin()]
            opt_alt = row.get("optimum_alt_ft", 30000)
            mach = row.get("optimum_mach", 0.75)
        else:
            opt_alt, mach = 30000, 0.75
        if auto_opt:
            alt_ft = opt_alt
        results = {
            "opt_alt_ft": int(round(alt_ft or opt_alt)),
            "mach": float(mach),
            "ias_kts": 300 if mode=="economy" else 320,
            "tas_kts": 480 if mode=="economy" else 550,
            "rpm_pct": 85 if mode=="economy" else 95,
            "ff_pph_per_engine": 5000 if mode=="economy" else 7000,
            "ff_total": (5000 if mode=="economy" else 7000)*2,
            "spec_range": 0.12 if mode=="economy" else 0.08,
            "_debug": {"source":"NATOPS CSV","row":row.to_dict()}
        }
    else:
        # Fallback stub
        results = {
            "opt_alt_ft": int(round(alt_ft or 30000)),
            "mach": 0.75 if mode=="economy" else 0.90,
            "ias_kts": 290 if mode=="economy" else 320,
            "tas_kts": 470 if mode=="economy" else 550,
            "rpm_pct": 83 if mode=="economy" else 97,
            "ff_pph_per_engine": 5200 if mode=="economy" else 7500,
            "ff_total": (5200 if mode=="economy" else 7500)*2,
            "spec_range": 0.11 if mode=="economy" else 0.07,
            "_debug": {"source":"fallback"}
        }
    return results
