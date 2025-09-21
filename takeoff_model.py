# takeoff_model.py — v1.3.1
# Table-driven takeoff performance using f14_perf.csv (+ optional SL overlay).
# Now supports synthetic MANEUVER (20°) flap band, derived from 0° and 40° anchors.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

from data_loaders import resolve_data_path

@dataclass
class TakeoffPoint:
    Vs_kts: float
    V1_kts: float
    Vr_kts: float
    V2_kts: float
    ASD_ft: float
    AGD_ft: float
    TODR_ft: float  # proxy to 35 ft after overlay/bias
    synthetic: bool = False

class TakeoffDeck:
    def __init__(self,
                 perf_csv: str = "f14_perf.csv",
                 overlay_csv: Optional[str] = "f14_perf_calibrated_SL_overlay.csv"):
        self.df = pd.read_csv(resolve_data_path(perf_csv, "f14_perf.csv"))
        self.overlay = None
        if overlay_csv:
            try:
                self.overlay = pd.read_csv(resolve_data_path(overlay_csv, "f14_perf_calibrated_SL_overlay.csv"))
            except Exception:
                self.overlay = None
        # Normalize
        self.df["thrust"] = self.df["thrust"].str.upper().str.strip()
        for c in ["flap_deg", "gw_lbs", "press_alt_ft", "oat_c",
                  "Vs_kt", "V1_kt", "Vr_kt", "V2_kt", "ASD_ft", "AGD_ft"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

    def _filter_slice(self, flap_deg: int, thrust: str) -> pd.DataFrame:
        return self.df[(self.df["flap_deg"] == flap_deg) &
                       (self.df["thrust"] == thrust.upper())].copy()

    def _interp3d(self, d: pd.DataFrame, gw: float, pa: float, oat: float) -> Dict[str, float]:
        # Simple nearest-neighbor if exact row not found
        if d.empty:
            return {}
        d["dist"] = ((d["gw_lbs"] - gw).abs() +
                     (d["press_alt_ft"] - pa).abs() +
                     (d["oat_c"] - oat).abs())
        row = d.sort_values("dist").iloc[0]
        return {
            "Vs": float(row["Vs_kt"]),
            "V1": float(row["V1_kt"]),
            "Vr": float(row["Vr_kt"]),
            "V2": float(row["V2_kt"]),
            "ASD": float(row["ASD_ft"]),
            "AGD": float(row["AGD_ft"])
        }

    def _apply_overlay(self, flap_deg: int, gw: float, pa: float, oat: float, agd: float) -> float:
        if self.overlay is None:
            return agd
        o = self.overlay.copy()
        o["score"] = (o["gw_lbs"] - gw).abs() + \
                     (o["press_alt_ft"] - pa).abs() + \
                     (o["oat_c"] - oat).abs()
        row = o.sort_values("score").iloc[0]
        bias = float(row.get("AGD_bias_factor", 1.0) or 1.0)
        delta = float(row.get("delta_to35_ft", 0.0) or 0.0)
        return max(1.0, agd * bias + delta)

    def _synthesize_20(self, gw: float, pa: float, oat: float, thrust: str) -> TakeoffPoint:
        # Pull 0° and 40° anchors
        d0 = self._filter_slice(0, thrust)
        d4 = self._filter_slice(40, thrust)
        if d0.empty or d4.empty:
            raise ValueError("Cannot synthesize MANEUVER flap without 0° and 40° anchors")
        vals0 = self._interp3d(d0, gw, pa, oat)
        vals4 = self._interp3d(d4, gw, pa, oat)
        # Physics-guided blend
        # Speeds scale ~ sqrt(CL). Approx with midpoint bias.
        Vs20 = np.mean([vals0["Vs"], vals4["Vs"] * 0.95])
        Vr20 = np.mean([vals0["Vr"], vals4["Vr"] * 0.95])
        V220 = np.mean([vals0["V2"], vals4["V2"] * 0.95])
        V120 = np.mean([vals0["V1"], vals4["V1"]])
        # Distances: energy & drag aware
        rV = Vr20 / vals0["Vr"] if vals0["Vr"] > 0 else 1.0
        agd20 = vals0["AGD"] * (rV**2) * 0.9 + vals4["AGD"] * 0.1
        asd20 = vals0["ASD"] * 1.05
        todr20 = self._apply_overlay(20, gw, pa, oat, agd20)
        return TakeoffPoint(
            Vs_kts=Vs20,
            V1_kts=V120,
            Vr_kts=Vr20,
            V2_kts=V220,
            ASD_ft=asd20,
            AGD_ft=agd20,
            TODR_ft=todr20,
            synthetic=True
        )

    def lookup(self, flap_deg: int, thrust: str,
               gw: float, pa: float, oat: float) -> TakeoffPoint:
        if flap_deg == 20:  # MANEUVER synthetic
            return self._synthesize_20(gw, pa, oat, thrust)
        d = self._filter_slice(flap_deg, thrust)
        vals = self._interp3d(d, gw, pa, oat)
        to35 = self._apply_overlay(flap_deg, gw, pa, oat, vals["AGD"])
        return TakeoffPoint(
            Vs_kts=vals["Vs"],
            V1_kts=vals["V1"],
            Vr_kts=vals["Vr"],
            V2_kts=vals["V2"],
            ASD_ft=vals["ASD"],
            AGD_ft=vals["AGD"],
            TODR_ft=to35,
            synthetic=False
        )
