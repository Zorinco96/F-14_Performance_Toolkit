# ============================================================
# F-14 Performance Calculator for DCS World — Landing Model
# File: landing_model.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
import os, math, pandas as pd
from typing import Dict, Any

CSV_PATH = "data/f14_landing_natops.csv"
FAA_FACTOR = 1.67

def compute_vref_and_ldr(gw_lbs: float, flaps: str, table: pd.DataFrame|None):
    vref = None
    ldr = None
    if table is not None and not table.empty:
        df = table.copy()
        df["gross_weight_lbs"] = pd.to_numeric(df["gross_weight_lbs"], errors="coerce")
        df = df[df["flap_setting"].str.upper() == flaps.upper()]
        if not df.empty:
            wdiff = (df["gross_weight_lbs"] - gw_lbs).abs()
            row = df.iloc[wdiff.idxmin()]
            vref = float(row.get("vref_kts", float("nan")))
            ldr = float(row.get("ground_roll_ft_unfactored", float("nan")))
    if (vref is None) or math.isnan(vref):
        vs0 = 110.0; w0 = 50000.0
        vs = vs0 * math.sqrt(max(gw_lbs,1.0)/w0)
        vref = 1.3*vs
    if (ldr is None) or math.isnan(ldr):
        base = 4500.0; w0 = 50000.0
        ldr = base * (max(gw_lbs,1.0)/w0)**1.1
    return int(round(vref)), int(round(ldr))

def plan_landing(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Plan landing scenarios A/B/C.
    Inputs:
      gw_lbs: current gross weight
      lda_ft: landing distance available
    """
    gw = float(inputs.get("gw_lbs",60000))
    lda_ft = int(inputs.get("lda_ft",8000))

    data = None
    if os.path.exists(CSV_PATH):
        try:
            data = pd.read_csv(CSV_PATH)
        except Exception:
            data = None

    results = {}
    scenarios = {}

    # Scenario A: 3000 lb fuel + stores kept
    gw_a = max(40000, gw-3000)
    vref_a, ldr_a = compute_vref_and_ldr(gw_a,"DOWN",data)
    fact_a = int(round(ldr_a*FAA_FACTOR))
    scenarios["A"] = {"gw":gw_a,"flaps":"DOWN","vref":vref_a,"ldr":ldr_a,"factored":fact_a,
        "dispatchable": fact_a<=lda_ft}

    # Scenario B: 3000 lb fuel + weapons expended, pods/tanks kept
    gw_b = max(40000, gw-3000)
    vref_b, ldr_b = compute_vref_and_ldr(gw_b,"DOWN",data)
    fact_b = int(round(ldr_b*FAA_FACTOR))
    scenarios["B"] = {"gw":gw_b,"flaps":"DOWN","vref":vref_b,"ldr":ldr_b,"factored":fact_b,
        "dispatchable": fact_b<=lda_ft}

    # Scenario C: custom
    gw_c = float(inputs.get("gw_custom",gw-3000))
    flaps_c = str(inputs.get("flaps_custom","DOWN")).upper()
    vref_c, ldr_c = compute_vref_and_ldr(gw_c,flaps_c,data)
    fact_c = int(round(ldr_c*FAA_FACTOR))
    scenarios["C"] = {"gw":gw_c,"flaps":flaps_c,"vref":vref_c,"ldr":ldr_c,"factored":fact_c,
        "dispatchable": fact_c<=lda_ft}

    results.update({
        "lda_ft": lda_ft,
        "scenarios": scenarios,
        "_debug": {"source":"NATOPS CSV" if data is not None else "fallback"}
    })
    return results
