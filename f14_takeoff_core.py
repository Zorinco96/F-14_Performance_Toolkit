# ============================================================
# F-14B Performance Calculator for DCS World
# File: f14_takeoff_core.py
# Version: v0.9.0 (2025-09-14)
#
# Changelog:
# - Baseline structure + data loaders
# - Landing module: Vref/Vapp/Va_c/Vfs + LDR + Max-LDW (placeholder model)
# - Light takeoff/climb placeholders (no behavior change yet)
#
# ============================================================
# 🚨 Bogged Down Protocol 🚨
# If development chat becomes slow or confusing:
# 1. STOP — Do not keep patching endlessly.
# 2. REVERT — Roll back to last saved checkpoint (Git tag vX.Y.Z).
# 3. RESET — Start a new chat if needed, say "continue from vX.Y.Z".
# 4. SCOPE — Focus on one module/card at a time.
# 5. SAVE — Commit working versions often with clear tags.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import pandas as pd

# ---- GitHub raw fallbacks (Option 1) ----
RAW_F14_PERF = (
    "https://raw.githubusercontent.com/Zorinco96/f14_takeoff_app.py/main/f14_perf.csv"
)
RAW_DCS_AIRPORTS = (
    "https://raw.githubusercontent.com/Zorinco96/f14_takeoff_app.py/main/dcs_airports.csv"
)

# ---- Local fallback paths (uploaded in-session) ----
LOCAL_F14_PERF = "/mnt/data/f14_perf (1).csv"
LOCAL_DCS_AIRPORTS = "/mnt/data/dcs_airports.csv"

# =====================
# Data Loading
# =====================

def load_perf_table() -> pd.DataFrame:
    """Load f14 performance CSV (try local, then GitHub raw)."""
    for p in (LOCAL_F14_PERF, RAW_F14_PERF):
        try:
            df = pd.read_csv(p)
            # Normalize key columns
            if "gw_lbs" in df.columns:
                df["gw_lbs"] = pd.to_numeric(df["gw_lbs"], errors="coerce")
            if "flap_deg" in df.columns:
                df["flap_deg"] = pd.to_numeric(df["flap_deg"], errors="coerce")
            if "Vs_kt" in df.columns:
                df["Vs_kt"] = pd.to_numeric(df["Vs_kt"], errors="coerce")
            return df.dropna(subset=["gw_lbs"]).copy()
        except Exception:
            continue
    raise RuntimeError("Unable to load f14_perf.csv from local or GitHub raw.")


def load_airports_table() -> pd.DataFrame:
    """Load DCS airports CSV (try local, then GitHub raw)."""
    for p in (LOCAL_DCS_AIRPORTS, RAW_DCS_AIRPORTS):
        try:
            df = pd.read_csv(p)
            # Normalize numeric runway fields if present
            for col in ("length_ft", "tora_ft", "toda_ft", "asda_ft", "threshold_elev_ft"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.copy()
        except Exception:
            continue
    raise RuntimeError("Unable to load dcs_airports.csv from local or GitHub raw.")

# =====================
# Utilities
# =====================

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def kts_to_fps(kts: float) -> float:
    return kts * 1.68781

# =====================
# Landing Performance (v0.9.0 placeholder model)
# =====================

@dataclass
class LandingInputs:
    gw_lbs: float
    flap_deg: Optional[float] = None  # If None, will infer "max flap" from table
    pa_ft: float = 0.0  # pressure altitude (ft)
    oat_c: float = 15.0
    steady_wind_kts: float = 0.0  # headwind positive, tailwind negative
    gust_increment_kts: float = 0.0  # gust additive (above steady)
    slope_percent: float = 0.0  # +uphill, -downhill (currently unused)
    vapp_min_additive_kts: float = 5.0  # baseline Vapp additive beyond Vref
    calibration_factor: float = 1.00  # global fudge to match DCS/FAA toggle later


@dataclass
class LandingSpeeds:
    Vs_kt: float
    Vref_kt: float
    Vapp_kt: float
    Vac_kt: float
    Vfs_kt: float


@dataclass
class LandingResult:
    speeds: LandingSpeeds
    ldr_50ft_ft: float  # Landing distance required from 50 ft


def _infer_landing_flap(perf: pd.DataFrame) -> float:
    """Pick the highest flap_deg available as 'landing' if not provided."""
    if "flap_deg" not in perf.columns:
        return 35.0  # sensible default
    vals = perf["flap_deg"].dropna().unique()
    if len(vals) == 0:
        return 35.0
    return float(pd.Series(vals).max())


def _vs_from_table(perf: pd.DataFrame, gw_lbs: float, flap_deg: Optional[float]) -> float:
    """Get/estimate Vs at given weight & flap from table using nearest-weight within that flap bucket.
    If flap not present, fall back to the max flap bucket, then overall nearest.
    """
    df = perf.dropna(subset=["gw_lbs"]).copy()
    target_flap = flap_deg if flap_deg is not None else _infer_landing_flap(df)

    if "flap_deg" in df.columns:
        bucket = df.loc[(df["flap_deg"].round(1) == round(target_flap, 1)) & df["Vs_kt"].notna()]
        if bucket.empty:
            # fallback to max flap available
            target_flap = _infer_landing_flap(df)
            bucket = df.loc[(df["flap_deg"].round(1) == round(target_flap, 1)) & df["Vs_kt"].notna()]
    else:
        bucket = df.loc[df["Vs_kt"].notna()]

    if bucket.empty:
        # last resort — overall min Vs in table
        if "Vs_kt" in df.columns and df["Vs_kt"].notna().any():
            return float(df["Vs_kt"].dropna().min())
        raise RuntimeError("Vs_kt column not found or empty in performance table.")

    # nearest by weight
    bucket = bucket.sort_values("gw_lbs")
    idx = (bucket["gw_lbs"] - gw_lbs).abs().idxmin()
    vs = float(bucket.loc[idx, "Vs_kt"])

    # simple scaling with sqrt(weight) around nearest value
    gw_ref = float(bucket.loc[idx, "gw_lbs"])
    if gw_ref > 0:
        vs *= math.sqrt(gw_lbs / gw_ref)
    return float(vs)


def calc_landing_speeds(perf: pd.DataFrame, inputs: LandingInputs) -> LandingSpeeds:
    Vs = _vs_from_table(perf, inputs.gw_lbs, inputs.flap_deg)

    # Vref = 1.3 * Vs (typical)
    Vref = 1.3 * Vs

    # Approach additive: baseline + 1/2 steady headwind + full gust increment (clamped)
    headwind = max(0.0, inputs.steady_wind_kts)
    wind_add = 0.5 * headwind + inputs.gust_increment_kts
    wind_add = clamp(wind_add, 0.0, 20.0)  # safety clamp
    Vapp = Vref + max(inputs.vapp_min_additive_kts, wind_add)

    # Heuristic for go-around / final-segment speeds (to be refined once climb linked)
    Vac = Vref + 10.0
    Vfs = Vref + 20.0

    return LandingSpeeds(Vs_kt=float(Vs), Vref_kt=float(Vref), Vapp_kt=float(Vapp), Vac_kt=float(Vac), Vfs_kt=float(Vfs))


def calc_ldr_50ft(perf: pd.DataFrame, inputs: LandingInputs, speeds: Optional[LandingSpeeds] = None) -> float:
    """Conservative placeholder model for LDR from 50 ft.
    Intuition: LDR scales ~ with kinetic energy (V^2) and weight; increases with PA and tailwind.
    We anchor a nominal value and scale:
      - LDR0 = 4200 ft at 44,000 lb, Vref0 inferred, sea level, no wind.
      - LDR ∝ (W/44k)^1.12 * (Vref/Vref0)^0.6 * (1 + 0.07 per 1000 ft PA)
      - Wind: -3% per 10 kt headwind, +5% per 10 kt tailwind (capped at ±20 kt effect)
    """
    spd = speeds or calc_landing_speeds(perf, inputs)

    W = max(10000.0, float(inputs.gw_lbs))
    W_ref = 44000.0

    # Establish a reference Vref at W_ref using the same method
    ref_inputs = LandingInputs(gw_lbs=W_ref, flap_deg=inputs.flap_deg)
    Vref0 = calc_landing_speeds(perf, ref_inputs).Vref_kt

    LDR0 = 4200.0  # anchor — tune later
    weight_term = (W / W_ref) ** 1.12
    v_term = (spd.Vref_kt / max(1.0, Vref0)) ** 0.6

    # PA factor ~7% per 1000 ft (placeholder)
    pa_term = 1.0 + 0.07 * max(0.0, inputs.pa_ft) / 1000.0

    # Wind factor (headwind positive -> reduces distance)
    hw = clamp(inputs.steady_wind_kts, -20.0, 20.0)
    if hw >= 0:
        wind_term = 1.0 - 0.03 * (hw / 10.0)
    else:
        wind_term = 1.0 + 0.05 * (abs(hw) / 10.0)

    LDR = LDR0 * weight_term * v_term * pa_term * wind_term
    LDR *= max(0.5, min(1.5, inputs.calibration_factor))  # safety clamp around calibrator

    return float(LDR)


def landing_required_distance(perf: pd.DataFrame, inputs: LandingInputs) -> LandingResult:
    speeds = calc_landing_speeds(perf, inputs)
    ldr = calc_ldr_50ft(perf, inputs, speeds)
    return LandingResult(speeds=speeds, ldr_50ft_ft=float(ldr))


def max_landing_weight_for_runway(perf: pd.DataFrame, runway_available_ft: float, inputs: LandingInputs) -> Tuple[float, LandingResult]:
    """Binary search for maximum landing weight such that LDR <= runway_available_ft.
    Returns (max_weight_lbs, LandingResult at that weight).
    """
    lo, hi = 20000.0, 80000.0  # plausible F-14B landing weight range
    best = None
    for _ in range(40):  # sufficient precision
        mid = 0.5 * (lo + hi)
        trial = LandingInputs(
            gw_lbs=mid,
            flap_deg=inputs.flap_deg,
            pa_ft=inputs.pa_ft,
            oat_c=inputs.oat_c,
            steady_wind_kts=inputs.steady_wind_kts,
            gust_increment_kts=inputs.gust_increment_kts,
            slope_percent=inputs.slope_percent,
            vapp_min_additive_kts=inputs.vapp_min_additive_kts,
            calibration_factor=inputs.calibration_factor,
        )
        res = landing_required_distance(perf, trial)
        if res.ldr_50ft_ft <= runway_available_ft:
            best = (mid, res)
            lo = mid
        else:
            hi = mid
    if best is None:
        # Even the minimum weight doesn't fit
        min_trial = LandingInputs(**{**inputs.__dict__, "gw_lbs": lo})
        return lo, landing_required_distance(perf, min_trial)
    return best

# =====================
# Takeoff & Climb placeholders (stubs)
# =====================

@dataclass
class TakeoffInputs:
    gw_lbs: float
    flap_deg: float
    pa_ft: float
    oat_c: float
    runway_cond: str = "DRY"


def takeoff_stub_example(perf: pd.DataFrame, x: TakeoffInputs) -> Dict[str, float]:
    """Placeholder to keep interface stable."""
    return {"Vr_kt": 140.0, "V2_kt": 150.0, "TOD_ft": 8000.0}


@dataclass
class ClimbInputs:
    gw_lbs: float
    pa_ft: float
    oat_c: float


def climb_stub_example(x: ClimbInputs) -> Dict[str, float]:
    return {"Vy_kt": 220.0, "ROC_fpm": 6000.0}
