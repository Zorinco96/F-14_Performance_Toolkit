# takeoff_model.py — v1.3.0
# Table-driven takeoff performance using f14_perf.csv (+ optional SL overlay).
# Provides: lookup of V-speeds, Accelerate-Stop Distance (ASD_ft), Accelerate-Go (AGD_ft),
#           and a to-35-ft proxy (TODR_ft ~= AGD_ft with overlay/bias applied).
#
# Notes:
# - Flap bands in data: 0 and 40. We'll map UI 35° -> 40° until we have 35° rows.
# - Thrust: 'MILITARY' and 'AFTERBURNER' present in data. 'DERATE' is applied upstream in core
#   by scaling distances via the derate exponent policy.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
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
    TODR_ft: float  # proxy to 35 ft (after overlay/bias if available)

class TakeoffDeck:
    def __init__(self, perf_csv: str = "f14_perf.csv", overlay_csv: Optional[str] = "f14_perf_calibrated_SL_overlay.csv"):
        self.df = pd.read_csv(resolve_data_path(perf_csv, "f14_perf.csv"))
        self.overlay = None
        if overlay_csv:
            try:
                self.overlay = pd.read_csv(resolve_data_path(overlay_csv, "f14_perf_calibrated_SL_overlay.csv"))
            except Exception:
                self.overlay = None
        # Normalize columns
        # Ensure numeric
        for c in ["flap_deg", "gw_lbs", "press_alt_ft", "oat_c", "Vs_kt", "V1_kt", "Vr_kt", "V2_kt", "ASD_ft", "AGD_ft"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        # Map thrust names to canonical forms
        self.df["thrust"] = self.df["thrust"].str.upper().str.strip()

    def _nearest_flap_in_table(self, flap_deg: float) -> int:
        # Data has 0 and 40; map 35->40 (FULL), 0 stays 0
        f = int(round(flap_deg))
        if f < 20:
            return 0
        return 40

    def _filter_slice(self, flap_deg: float, thrust: str, model: str = "F-14B") -> pd.DataFrame:
        f = self._nearest_flap_in_table(flap_deg)
        t = thrust.upper().strip()
        d = self.df[(self.df["model"] == model) & (self.df["flap_deg"] == f) & (self.df["thrust"] == t)].copy()
        if d.empty:
            # Loosen thrust filter if needed
            d = self.df[(self.df["model"] == model) & (self.df["flap_deg"] == f)].copy()
        return d

    def _interp4(self, d: pd.DataFrame, gw_lbs: float, pa_ft: float, oat_c: float) -> Dict[str, float]:
        # Interpolate in 3D (gw, pa, oat). We'll do successive 1D interpolations.
        # Expect dense-enough grid; otherwise fall back to nearest.
        cols = ["gw_lbs", "press_alt_ft", "oat_c"]
        for c in cols:
            if c not in d.columns:
                raise ValueError(f"Missing required column {c} in performance table")

        # Helper: safe 1D interp over a given key with grouping
        def interp_along(key: str, df: pd.DataFrame, target: float) -> pd.DataFrame:
            xs = np.sort(df[key].dropna().unique())
            if len(xs) == 0:
                return df.iloc[[0]].copy()
            lo = xs[xs <= target].max(initial=xs.min())
            hi = xs[xs >= target].min(initial=xs.max())
            if lo == hi:
                return df[df[key] == lo].copy()
            # Linear blend between two slices
            d0 = df[df[key] == lo]
            d1 = df[df[key] == hi]
            # Align by remaining keys
            on = [c for c in cols if c != key]
            m = pd.merge(d0, d1, on=on, suffixes=("_lo","_hi"))
            w = (target - lo) / (hi - lo)
            out = m.copy()
            for name in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
                if name+"_lo" in m.columns and name+"_hi" in m.columns:
                    out[name] = (1-w)*m[name+"_lo"] + w*m[name+"_hi"]
            for c in on:
                out[c] = out[c+"_lo"]
            keep = cols + ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]
            out = out[keep].drop_duplicates()
            return out

        s = d.copy()
        s = interp_along("gw_lbs", s, gw_lbs)
        s = interp_along("press_alt_ft", s, pa_ft)
        s = interp_along("oat_c", s, oat_c)
        if s.empty:
            # Fallback to nearest row
            idx = ((d["gw_lbs"]-gw_lbs).abs() + (d["press_alt_ft"]-pa_ft).abs() + (d["oat_c"]-oat_c).abs()).idxmin()
            s = d.loc[[idx]]
        row = s.iloc[0]
        return {
            "Vs_kts": float(row["Vs_kt"]),
            "V1_kts": float(row["V1_kt"]),
            "Vr_kts": float(row["Vr_kt"]),
            "V2_kts": float(row["V2_kt"]),
            "ASD_ft": float(row["ASD_ft"]),
            "AGD_ft": float(row["AGD_ft"]),
        }

    def _apply_overlay(self, flap_deg: float, gw_lbs: float, pa_ft: float, oat_c: float, agd_ft: float) -> float:
        if self.overlay is None:
            return agd_ft
        # Overlay schema: press_alt_ft, flap_deg, gw_lbs, oat_c, AGD_bias_factor, delta_to35_ft
        o = self.overlay.copy()
        # Nearest match on keys present in overlay
        keys = []
        for key in ["press_alt_ft","flap_deg","gw_lbs","oat_c"]:
            if key in o.columns:
                keys.append(key)
                o[key+"_diff"] = (o[key].astype(float) - (locals()[key] if key in locals() else 0.0)).abs()
        if not keys:
            return agd_ft
        o["score"] = sum(o[k+"_diff"] for k in keys)
        best = o.sort_values("score").iloc[0]
        bias = float(best.get("AGD_bias_factor", 1.0) or 1.0)
        delta = float(best.get("delta_to35_ft", 0.0) or 0.0)
        return max(1.0, agd_ft * bias + delta)

    def lookup(self, flap_deg: float, thrust: str, gw_lbs: float, pa_ft: float, oat_c: float) -> TakeoffPoint:
        d = self._filter_slice(flap_deg, thrust)
        vals = self._interp4(d, gw_lbs, pa_ft, oat_c)
        to35 = self._apply_overlay(flap_deg, gw_lbs, pa_ft, oat_c, vals["AGD_ft"])
        return TakeoffPoint(
            Vs_kts = vals["Vs_kts"],
            V1_kts = vals["V1_kts"],
            Vr_kts = vals["Vr_kts"],
            V2_kts = vals["V2_kts"],
            ASD_ft = vals["ASD_ft"],
            AGD_ft = vals["AGD_ft"],
            TODR_ft = to35
        )
