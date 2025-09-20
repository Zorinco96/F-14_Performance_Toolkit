# f14_takeoff_core.py — v1.2.1-core (DERATE integration & SL overlay; version bump only)
from __future__ import annotations

from typing import Dict, Tuple, Optional
import math

# ============ BEGIN: physics bridge wrappers (safe add) ============
# These wrappers call the small, calibrated modules we built:
# takeoff_model, climb_model, cruise_model, landing_model + calibration loader.

from data_loaders import load_calibration_csv
from takeoff_model import takeoff_run
from climb_model import climb_profile
from cruise_model import cruise_point
from landing_model import landing_performance
from functools import lru_cache

__version__ = "1.2.2-core+debug-fix"

@lru_cache(maxsize=1)
def get_calib():
    """LRU-cached calibration to avoid repeated CSV reads during V1 sweeps."""
    return load_calibration_csv("calibration.csv")

def perf_compute_takeoff(
    *,
    gw_lb: float,
    field_elev_ft: float,
    oat_c: float,
    headwind_kts: float,
    runway_slope: float,
    thrust_mode: str,             # "MIL" or "MAX"
    mode: str = "DCS",            # "DCS" or "FAA"
    config: str = "TO_FLAPS",     # "TO_FLAPS" or "CLEAN"
    sweep_deg: float = 20.0,
    stores: list[str] | None = None,
) -> dict:
    """Return dict with Vs/VR/VLOF/V2, ground roll, 35 ft, time."""
    calib = get_calib()
    return takeoff_run(
        weight_lbf=gw_lb,
        alt_ft=field_elev_ft,
        oat_c=oat_c,
        headwind_kts=headwind_kts,
        runway_slope=runway_slope,
        config=config,
        sweep_deg=sweep_deg,
        power=("MAX" if thrust_mode.upper().startswith("AB") or thrust_mode.upper()=="MAX" else "MIL"),
        mode=("DCS" if mode.upper()=="DCS" else "FAA"),
        calib=calib,
        stores=stores or [],
        dt=0.05,
    )

def perf_compute_climb(
    *,
    gw_lb: float,
    alt_start_ft: float,
    alt_end_ft: float,
    oat_dev_c: float = 0.0,
    schedule: str = "NAVY",       # "NAVY" or "DISPATCH"
    mode: str = "DCS",
    power: str = "MIL",
    sweep_deg: float = 20.0,
    config: str = "CLEAN",
) -> dict:
    """Return dict with time, fuel, distance, avg ROC."""
    return climb_profile(
        weight_lbf=gw_lb,
        alt_start_ft=alt_start_ft,
        alt_end_ft=alt_end_ft,
        oat_dev_c=oat_dev_c,
        schedule=("NAVY" if schedule.upper()=="NAVY" else "DISPATCH"),
        mode=("DCS" if mode.upper()=="DCS" else "FAA"),
        power=("MAX" if power.upper()=="MAX" else "MIL"),
        sweep_deg=sweep_deg,
        config=config,
        dt=1.0,
    )

def perf_compute_cruise(
    *,
    gw_lb: float,
    alt_ft: float,
    mach: float,
    power: str = "MIL",
    sweep_deg: float = 20.0,
    config: str = "CLEAN",
) -> dict:
    """Return dict with drag, FF total, TAS, specific range."""
    return cruise_point(
        weight_lbf=gw_lb,
        alt_ft=alt_ft,
        mach=mach,
        power=("MAX" if power.upper()=="MAX" else "MIL"),
        sweep_deg=sweep_deg,
        config=config,
    )

def perf_compute_landing(
    *,
    gw_lb: float,
    field_elev_ft: float,
    oat_c: float,
    headwind_kts: float,
    mode: str = "DCS",
    config: str = "LDG_FLAPS",
    sweep_deg: float = 20.0,
) -> dict:
    """Return dict with Vref, airborne, ground roll, total."""
    calib = get_calib()
    return landing_performance(
        weight_lbf=gw_lb,
        alt_ft=field_elev_ft,
        oat_c=oat_c,
        headwind_kts=headwind_kts,
        mode=("DCS" if mode.upper()=="DCS" else "FAA"),
        calib=calib,
        config=config,
        sweep_deg=sweep_deg,
    )
# ============ END: physics bridge wrappers ============

# -----------------------------
# Constants (placeholder values)
# -----------------------------
F14B_BEW_LB = 43_735.0          # Basic empty weight (approx placeholder)
EXT_TANK_EMPTY_LB = 1_100.0     # Structural weight per external tank, no fuel
ISA_LAPSE_C_PER_FT = 1.98 / 1000.0  # °C per foot

# Simple MAC/trim placeholders for UI wiring
DEFAULT_CG_PERCENT_MAC = 24.5
DEFAULT_STAB_TRIM_UNITS = 0.0

# ---------------------------------------
# Atmosphere / conversions (UI scaffolds)
# ---------------------------------------
def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: Optional[float]) -> float:
    """
    Compute pressure altitude from field elevation (ft) and QNH (inHg).
    PA(ft) ≈ elev + (29.92 - QNH) * 1000. If QNH is None, assume 29.92.
    """
    elev = float(field_elev_ft or 0.0)
    if qnh_inhg is None:
        return elev
    try:
        qnh = float(qnh_inhg)
    except Exception:
        qnh = 29.92
    return elev + (29.92 - qnh) * 1000.0

def density_ratio_sigma(press_alt_ft: float, oat_c: float) -> float:
    """
    ISA-based density ratio σ = δ/θ (troposphere approximation).
    δ ≈ (1 - 6.87535e-6 * h)^4.2561 with h in ft.
    θ = (T / T_ISA_at_h), where T = OAT in Kelvin, T_ISA_at_h = ISA temp at altitude.
    Clamped to a reasonable range to avoid UI weirdness.
    """
    h = max(0.0, float(press_alt_ft or 0.0))
    # Pressure ratio
    delta = (1.0 - 6.87535e-6 * h) ** 4.2561
    # ISA temperature at altitude (°C)
    isa_c = 15.0 - ISA_LAPSE_C_PER_FT * h
    # Temperature ratio
    try:
        theta = ((float(oat_c) + 273.15) / (isa_c + 273.15))
        if theta <= 0:
            theta = 1.0
    except Exception:
        theta = 1.0
    sigma = delta / theta
    return max(0.2, min(1.2, float(sigma)))

# ---------------------------------------
# Weight & Balance aggregation (scaffold)
# ---------------------------------------
def build_loadout_totals(
    stations: Dict[str, Dict[str, object]],
    fuel_lb: float,
    ext_tanks: Tuple[bool, bool],
    mode_simple_gw: Optional[float] = None,
) -> Dict[str, float]:
    """
    Aggregate gross weights for UI display.
    Returns keys used by the app:
      - 'gw_tow_lb'       : takeoff gross weight (lb)
      - 'gw_ldg_lb'       : landing gross weight (lb) [fuel landing = min(fuel_tow, 3000 lb)]
      - 'zf_weight_lb'    : zero-fuel weight (lb)
      - 'fuel_tow_lb'     : fuel at takeoff (lb)
      - 'fuel_ldg_lb'     : fuel at landing (lb) (placeholder 3000 lb cap)
      - 'cg_percent_mac'  : placeholder CG %MAC
      - 'stab_trim_units' : placeholder stabilizer trim units

    Notes:
      - In Simple mode, we trust the provided GW for takeoff and infer ZFW by subtracting fuel.
      - In Detailed mode, we sum station store/pylon weights + BEW + external tank structure.
      - External tank *fuel* is not counted here; app passes total fuel separately.
    """
    # Normalize inputs
    fuel_tow = max(0.0, float(fuel_lb or 0.0))
    fuel_ldg = min(fuel_tow, 3000.0)  # landing fuel placeholder
    left_tank, right_tank = bool(ext_tanks[0]), bool(ext_tanks[1])
    tank_struct = (EXT_TANK_EMPTY_LB if left_tank else 0.0) + (EXT_TANK_EMPTY_LB if right_tank else 0.0)

    if mode_simple_gw is not None:
        # Simple mode: user-provided takeoff GW; infer ZFW
        gw_tow = max(0.0, float(mode_simple_gw))
        zfw = max(0.0, gw_tow - fuel_tow)
        gw_ldg = max(0.0, zfw + fuel_ldg)
    else:
        # Detailed mode: sum stations (store + pylon), add BEW and tank structural weight
        stores = 0.0
        pylons = 0.0
        try:
            for _, d in (stations or {}).items():
                qty = int(d.get("qty", 0) or 0)
                store_w = float(d.get("store_weight_lb", 0.0) or 0.0)
                pylon_w = float(d.get("pylon_weight_lb", 0.0) or 0.0)
                if qty > 0:
                    stores += qty * store_w
                    pylons += pylon_w  # one pylon per occupied station (approx)
        except Exception:
            # If anything odd comes in, fall back gracefully
            stores = 0.0
            pylons = 0.0

        zfw = max(0.0, F14B_BEW_LB + stores + pylons + tank_struct)
        gw_tow = max(0.0, zfw + fuel_tow)
        gw_ldg = max(0.0, zfw + fuel_ldg)

    # Placeholder CG/trim for UI wiring
    cg_percent_mac = float(DEFAULT_CG_PERCENT_MAC)
    stab_trim_units = float(DEFAULT_STAB_TRIM_UNITS)

    return {
        "gw_tow_lb": float(gw_tow),
        "gw_ldg_lb": float(gw_ldg),
        "zf_weight_lb": float(zfw),
        "fuel_tow_lb": float(fuel_tow),
        "fuel_ldg_lb": float(fuel_ldg),
        "cg_percent_mac": cg_percent_mac,
        "stab_trim_units": stab_trim_units,
    }

# ======================================================================
# Auto-Select & Balanced-Field (BFL) selection pipeline — v1.0 (spec-frozen)
# ======================================================================
from dataclasses import dataclass
from typing import Any, Callable, List

# ----------------------------
# Policy: Wind credit settings
# ----------------------------
@dataclass(frozen=True)
class WindPolicy:
    use_headwind_fraction: float  # e.g., 0.5 for "50% headwind credit"
    tailwind_multiplier: float    # e.g., 1.5 for "150% tailwind penalty"

WIND_POLICY_50_150 = WindPolicy(0.5, 1.5)
WIND_POLICY_0_150  = WindPolicy(0.0, 1.5)

# ----------------------------
# Inputs for auto-selection
# ----------------------------
@dataclass(frozen=True)
class AutoSelectInputs:
    available_tora_ft: int
    available_asda_ft: int
    runway_heading_deg: float
    headwind_kts_raw: float           # prior to policy (e.g., component along rwy)
    aeo_required_ft_per_nm: float = 200.0
    wind_policy: WindPolicy = WIND_POLICY_50_150

    # Derate caps by flap, 1% steps
    up_min_pct: int = 85
    man_min_pct: int = 90
    full_min_pct: int = 98
    pct_step: int = 1

# ----------------------------
# Result shape
# ----------------------------
@dataclass
class AutoSelectResult:
    dispatchable: bool
    reason: str = ""
    flaps: str = ""
    thrust_label: str = ""         # "DERATE (xx%)" or "MILITARY (100% RPM)"
    derate_pct: int | None = None  # None when MIL
    balanced_label: str = ""       # "Balanced" or "Governing: ASDR/TODR"
    governing_side: str = ""       # "ASDR" or "TODR" (when not balanced)
    notes: List[str] = None
    perf: dict = None              # pass-through from compute function

# --------------------------------------------------------
# Helper: apply conservative wind credit to headwind value
# --------------------------------------------------------
def _apply_wind_policy(headwind_kts_raw: float, pol: WindPolicy) -> float:
    if headwind_kts_raw >= 0.0:
        return headwind_kts_raw * pol.use_headwind_fraction
    else:
        return headwind_kts_raw * pol.tailwind_multiplier  # more negative

# --------------------------------------------------------
# Candidate compute callback contract (wire your model here)
# --------------------------------------------------------
"""
You will provide `compute_candidate` to `auto_select_flaps_thrust`.
Signature:

compute_candidate(
    *,
    flaps: str,                 # "UP" | "MANEUVER" | "FULL"
    thrust_pct: int,            # 85..99 for derate; 100 for MIL
    v1_kt: float | None,        # for BFL sweep; pass None to let the model guess
    context: dict               # scenario inputs
) -> dict with keys:
    - "ASDR_ft"               : accelerate-stop distance at that V1
    - "TODR_OEI_35ft_ft"      : accelerate-go (OEI) distance to 35 ft at that V1
    - "balanced_diff_ratio"   : abs(ASDR-TODR)/max(ASDR,TODR)  (for the 1% test)
    - "AEO_min_grad_ft_per_nm_to_1000" : min AEO gradient to 1000 AFE at selected thrust
    - "OEI_second_seg_gross_pct"       : second-segment gross gradient (%)
    - "OEI_final_seg_gross_pct"        : final-segment gross gradient (%)
    - (optional) any V-speeds/FF/etc for display; pass back in "perf"
If you don't have these yet, return placeholders and mark as not computed.
"""

# --------------------------------------------------------
# BFL V1-sweep (minimize max(ASDR, TODR_OEI)) over discrete grid
# --------------------------------------------------------
def _bfl_solve_minimax_v1(
    *, flaps: str, thrust_pct: int, context: dict,
    compute_candidate: Callable[..., dict],
    v1_grid: List[float]
) -> tuple[float, dict, str, float]:
    """
    Returns: (best_v1, best_result_dict, governing_side, diff_ratio)
      - governing_side: "ASDR" or "TODR"
      - diff_ratio: abs(ASDR-TODR)/max(ASDR,TODR)
    """
    best_v1 = None
    best = None
    best_govern = ""
    best_max = float("inf")
    best_diff_ratio = 1.0

    for v1 in v1_grid:
        r = compute_candidate(flaps=flaps, thrust_pct=thrust_pct, v1_kt=float(v1), context=context)
        if ("ASDR_ft" not in r) or ("TODR_OEI_35ft_ft" not in r):
            # model not wired → skip
            continue
        asdr = float(r["ASDR_ft"]); todr = float(r["TODR_OEI_35ft_ft"])
        m = max(asdr, todr)
        if m < best_max:
            best_max = m
            best_v1 = v1
            best = r
            best_govern = "ASDR" if asdr > todr else "TODR"
            # recompute diff ratio for the "Balanced" badge
            if m > 0:
                best_diff_ratio = abs(asdr - todr) / m

    return best_v1, best, best_govern, best_diff_ratio

# --------------------------------------------------------
# Main selector implementing your frozen spec
# --------------------------------------------------------
def auto_select_flaps_thrust(
    *,
    sel: AutoSelectInputs,
    scenario_context: dict,
    compute_candidate: Callable[..., dict],
) -> AutoSelectResult:
    """
    Implements:
      - Prefer derate over MIL
      - Flap priority UP → MANEUVER → FULL
      - Flap caps: UP ≥85, MAN ≥90, FULL ≥98; 1% steps
      - Special tie-breaker: prefer MAN DERATE over UP MIL if both pass
      - Afterburner: evaluated only to report "would pass (prohibited)" → ND
      - Pass gates: BFL ≤ ASDA/TORA, AEO ≥ 200 ft/NM to 1000 AFE, OEI segments (gross; report net)
      - Balanced label if within 1%
    """
    from typing import List  # local import to keep top-level typing minimal
    notes: List[str] = []

    # Apply wind policy to the already-computed headwind component
    hw_policy_kts = _apply_wind_policy(sel.headwind_kts_raw, sel.wind_policy)
    ctx = dict(scenario_context)
    ctx["headwind_kts_policy"] = hw_policy_kts

    # Build the per-flap derate ranges (descending thrust from min→99, then 100 MIL last)
    flap_bands = [
        ("UP",       sel.up_min_pct,   99),
        ("MANEUVER", sel.man_min_pct,  99),
        ("FULL",     sel.full_min_pct, 99),
    ]

    # Helper to test one (flaps, pct) across BFL + climb gates
    def try_candidate(flaps: str, pct: int) -> tuple[bool, dict, float, str]:
        # Discrete V1 grid (you can refine once the model is wired)
        v1_grid = list(range(90, 171, 5))  # TEMP: 90..170 kt, 5-kt step
        v1, r, govern, diff_ratio = _bfl_solve_minimax_v1(
            flaps=flaps, thrust_pct=pct, context=ctx, compute_candidate=compute_candidate, v1_grid=v1_grid
        )
        if r is None:
            return False, {"__diagnostic__": "model_not_wired"}, 1.0, "ASDR"
        # Field-length gate
        balanced_req_ft = max(float(r["ASDR_ft"]), float(r["TODR_OEI_35ft_ft"]))
        r["balanced_v1_kt"] = v1
        r["balanced_required_ft"] = balanced_req_ft
        if balanced_req_ft > max(sel.available_asda_ft, sel.available_tora_ft):
            return False, r, diff_ratio, govern

        # AEO 200 ft/NM to 1000 AFE
        aeo_min = r.get("AEO_min_grad_ft_per_nm_to_1000", None)
        if aeo_min is None:
            return False, {**r, "__diagnostic__": "AEO_not_wired"}, diff_ratio, govern
        if aeo_min < sel.aeo_required_ft_per_nm:
            return False, r, diff_ratio, govern

        # OEI segments (gross)
        seg2 = r.get("OEI_second_seg_gross_pct", None)
        seg4 = r.get("OEI_final_seg_gross_pct", None)
        if (seg2 is None) or (seg4 is None):
            return False, {**r, "__diagnostic__": "OEI_not_wired"}, diff_ratio, govern
        if (seg2 < 2.4) or (seg4 < 1.2):
            return False, r, diff_ratio, govern

        return True, r, diff_ratio, govern

    # 1) DERATE search by flap band (lowest flap wins)
    for flap_name, pmin, pmax in flap_bands:
        for pct in range(pmin, pmax + 1, sel.pct_step):  # Ascending = minimum required thrust
            ok, r, diff_ratio, govern = try_candidate(flap_name, pct)
            if ok:
                bal_label = "Balanced" if diff_ratio <= 0.01 else (f"Governing: {govern}")
                return AutoSelectResult(
                    dispatchable=True,
                    flaps=flap_name,
                    thrust_label=(f"DERATE ({pct}%)" if pct < 100 else "MILITARY (100% RPM)"),
                    derate_pct=(pct if pct < 100 else None),
                    balanced_label=bal_label,
                    governing_side=("" if diff_ratio <= 0.01 else govern),
                    notes=notes,
                    perf=r
                )

    # 2) Special tie-breaker vs UP MIL:
    for flap_name in ["UP", "MANEUVER", "FULL"]:
        ok, r, diff_ratio, govern = try_candidate(flap_name, 100)
        if ok:
            bal_label = "Balanced" if diff_ratio <= 0.01 else (f"Governing: {govern}")
            return AutoSelectResult(
                dispatchable=True,
                flaps=flap_name,
                thrust_label="MILITARY (100% RPM)",
                derate_pct=None,
                balanced_label=bal_label,
                governing_side=("" if diff_ratio <= 0.01 else govern),
                notes=notes,
                perf=r
            )

    # 3) AB evaluated only to report prohibition (informational)
    return AutoSelectResult(
        dispatchable=False,
        reason="Not Dispatchable — Would pass with Afterburner (Prohibited)",
        notes=notes or ["Auto-select evaluated: all DERATE and MIL options failed the BFL + AEO + OEI gates."],
        perf={}
    )

# ======================================================================
# ============ DERATE INTEGRATION (NEW) =================================
# ======================================================================

# This section wires in:
# - SL overlay corrections (f14_perf_calibrated_SL_overlay.csv)
# - A small interpolator to read MIL ground roll for the user’s scenario
# - A thin wrapper that calls derate.DerateModel to return FF/RPM targets

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
def _csv(name: str) -> str:
    return os.path.join(DATA_DIR, name)

# Optional assets (load safely if present)
def _load_df_safe(path: str, required: bool = False) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        if required:
            raise
        return pd.DataFrame()

# Baseline NATOPS perf grid (must exist in your repo)
PERF = _load_df_safe(_csv("f14_perf.csv"), required=False)

# SL overlay and calibration summary (optional)
SL_OVERLAY = _load_df_safe(_csv("f14_perf_calibrated_SL_overlay.csv"))
CAL_SL     = _load_df_safe(_csv("calibration_sl_summary.csv"))

def apply_sl_overlay(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Apply SL bias and delta-to-35ft to baseline rows (press_alt_ft == 0) if overlay present."""
    if perf_df.empty or SL_OVERLAY.empty:
        return perf_df
    key = ["press_alt_ft", "flap_deg", "gw_lbs", "oat_c"]
    df = perf_df.copy()
    if "AGD_ft" not in df.columns:
        return df
    merged = df.merge(SL_OVERLAY[key + ["AGD_bias_factor", "delta_to35_ft"]], on=key, how="left")
    mask = (merged["press_alt_ft"] == 0) & merged["AGD_bias_factor"].notna()
    merged.loc[mask, "AGD_ft"] = merged.loc[mask, "AGD_ft"] * merged.loc[mask, "AGD_bias_factor"]
    if "AGD_to35_ft" not in merged.columns:
        merged["AGD_to35_ft"] = pd.NA
    merged.loc[mask, "AGD_to35_ft"] = merged.loc[mask, "AGD_ft"] + merged.loc[mask, "delta_to35_ft"].fillna(0)
    return merged

PERF_CAL = apply_sl_overlay(PERF)

def _interp3(perf_df: pd.DataFrame, flap_deg: int, gw_lbs: float, pa_ft: float, oat_c: float, value_col: str = "AGD_ft") -> Optional[float]:
    """Tri-linear-like interpolation across press_alt_ft × gw_lbs × oat_c at fixed flap."""
    if perf_df.empty:
        return None
    sub = perf_df[perf_df["flap_deg"] == int(flap_deg)]
    if sub.empty or value_col not in sub.columns:
        return None

    def brack(vals: np.ndarray, x: float) -> tuple[float, float]:
        vals = np.unique(np.sort(vals))
        lo = vals[vals <= x]
        hi = vals[vals >= x]
        v_lo = lo.max() if lo.size else float(vals.min())
        v_hi = hi.min() if hi.size else float(vals.max())
        return float(v_lo), float(v_hi)

    pa_lo, pa_hi   = brack(sub["press_alt_ft"].values, float(pa_ft))
    gw_lo, gw_hi   = brack(sub["gw_lbs"].values,       float(gw_lbs))
    ot_lo, ot_hi   = brack(sub["oat_c"].values,        float(oat_c))

    def corner(pa, gw, ot):
        q = sub[(sub["press_alt_ft"]==pa) & (sub["gw_lbs"]==gw) & (sub["oat_c"]==ot)]
        if q.empty:
            return None
        return float(q.iloc[0][value_col])

    c000 = corner(pa_lo, gw_lo, ot_lo); c001 = corner(pa_lo, gw_lo, ot_hi)
    c010 = corner(pa_lo, gw_hi, ot_lo); c011 = corner(pa_lo, gw_hi, ot_hi)
    c100 = corner(pa_hi, gw_lo, ot_lo); c101 = corner(pa_hi, gw_lo, ot_hi)
    c110 = corner(pa_hi, gw_hi, ot_lo); c111 = corner(pa_hi, gw_hi, ot_hi)
    if any(v is None for v in [c000,c001,c010,c011,c100,c101,c110,c111]):
        return None

    def lerp(a,b,t): return a + (b-a)*t
    tx = 0.0 if pa_hi==pa_lo else (float(pa_ft)-pa_lo)/(pa_hi-pa_lo)
    ty = 0.0 if gw_hi==gw_lo else (float(gw_lbs)-gw_lo)/(gw_hi-gw_lo)
    tz = 0.0 if ot_hi==ot_lo else (float(oat_c)-ot_lo)/(ot_hi-ot_lo)

    # Interpolate
    c00 = lerp(c000, c001, tz)
    c01 = lerp(c010, c011, tz)
    c10 = lerp(c100, c101, tz)
    c11 = lerp(c110, c111, tz)
    c0  = lerp(c00, c01, ty)
    c1  = lerp(c10, c11, ty)
    return lerp(c0, c1, tx)

def mil_ground_roll_ft(flap_deg: int, gw_lbs: float, pa_ft: float, oat_c: float) -> Optional[float]:
    """Baseline MIL ground roll from the calibrated (overlay-applied) table."""
    if PERF_CAL.empty:
        return None
    # Accept 'thrust' string column variations
    if "thrust" in PERF_CAL.columns:
        thrust_mask = PERF_CAL["thrust"].str.upper().isin(["MIL","MILITARY"])
        df = PERF_CAL[thrust_mask]
    else:
        df = PERF_CAL.copy()
    return _interp3(df, flap_deg, gw_lbs, pa_ft, oat_c, value_col="AGD_ft")

# --- Derate engine wrapper (derate.py) ---
try:
    from derate import DerateModel
    DM = DerateModel(
        tff_model_csv=_csv("f110_tff_model.csv"),
        ff_rpm_knots_csv=_csv("f110_ff_to_rpm_knots.csv"),
        calibration_sl_csv=_csv("calibration_sl_summary.csv"),
        config_json=os.path.join(os.path.dirname(__file__), "derate_config.json"),
    )
except Exception:
    DM = None  # keep app functional even if derate files not present

def compute_derate_for_run(
    *,
    flap_deg: int,
    gw_lbs: float,
    pa_ft: float,
    oat_c: float,
    runway_available_ft: float,
    allow_ab: Optional[bool] = None
) -> Optional[dict]:
    """
    Returns dict:
      { 'thrust_multiplier', 'T_required_lbf', 'FF_required_pph', 'RPM_required_pct',
        'T_MIL_lbf', 'FF_MIL_pph', 'alpha_used', 'alt_used_ft' }
    or None if unavailable.
    """
    if DM is None:
        return None
    gr_mil = mil_ground_roll_ft(flap_deg, gw_lbs, pa_ft, oat_c)
    if gr_mil is None:
        return None
    return DM.compute_derate_from_groundroll(
        flap_deg=int(flap_deg),
        pa_ft=float(pa_ft),
        mil_ground_roll_ft=float(gr_mil),
        runway_available_ft=float(runway_available_ft),
        allow_ab=allow_ab
    )

def plan_takeoff_with_optional_derate(
    *,
    flap_deg: int,                 # 0 or 35 for your app
    gw_lbs: float,
    field_elev_ft: float,
    qnh_inhg: float | None,
    oat_c: float,
    headwind_kts_component: float, # already projected onto runway
    runway_slope: float,
    tora_ft: int,                  # available TORA
    asda_ft: int,                  # available ASDA
    allow_ab: bool = False,
    do_derate: bool = True
) -> dict:
    """
    One-stop call for your UI:
      - Computes baseline MIL performance
      - If do_derate=True, computes derated thrust/RPM to meet runway
      - Returns a dict your UI can render directly
    """

    # 1) Baseline MIL performance (existing call)
    mil_perf = perf_compute_takeoff(
        gw_lb=gw_lbs,
        field_elev_ft=field_elev_ft,
        oat_c=oat_c,
        headwind_kts=headwind_kts_component,
        runway_slope=runway_slope,
        thrust_mode="MIL",
        mode="DCS",
        config=("TO_FLAPS" if flap_deg >= 35 else "CLEAN"),
        sweep_deg=20.0,
        stores=[]
    )

    # 2) Limiting runway distance for derate (conservative)
    runway_available_ft = min(int(tora_ft), int(asda_ft))

    # 3) Pressure altitude
    pa_ft = pressure_altitude_ft(field_elev_ft, qnh_inhg)

    # 4) Optional derate
    der = None
    derate_debug = None
    if do_derate:
        try:
            if DM is None:
                derate_debug = 'DM_none'
            else:
                # Prefer 35-ft if available, and map 35->nearest table flap (e.g., 40)
                try:
                    base = mil_ground_roll_or_to35_ft(flap_deg, gw_lbs, pa_ft, oat_c)
                except NameError:
                    # Fallback to AGD_ft-only helper if the file doesn't have the 35-ft-aware one
                    base = mil_ground_roll_ft(flap_deg, gw_lbs, pa_ft, oat_c)
                if base is None:
                    try:
                        flap_present = not PERF_CAL[PERF_CAL['flap_deg'] == int(flap_deg)].empty
                    except Exception:
                        flap_present = False
                    if not flap_present:
                        derate_debug = f'table_missing_flap:{int(flap_deg)}'
                    else:
                        derate_debug = 'table_missing_corners'
                else:
                    der = DM.compute_derate_from_groundroll(
                        flap_deg=int(flap_deg),
                        pa_ft=float(pa_ft),
                        mil_ground_roll_ft=float(base),
                        runway_available_ft=float(runway_available_ft),
                        allow_ab=allow_ab
                    )
        except Exception as e:
            der = None
            derate_debug = f'exception:{type(e).__name__}:{e}'

    # 5) Build UI payload
    out = {
        "inputs": {
            "flap_deg": flap_deg,
            "gw_lbs": gw_lbs,
            "field_elev_ft": field_elev_ft,
            "qnh_inhg": qnh_inhg,
            "oat_c": oat_c,
            "headwind_kts_component": headwind_kts_component,
            "runway_slope": runway_slope,
            "tora_ft": tora_ft,
            "asda_ft": asda_ft,
            "pressure_alt_ft": pa_ft,
            "runway_available_ft": runway_available_ft,
        },
        "baseline_MIL": mil_perf,
        "derate": der,
        "derate_debug": (None if der is not None else (derate_debug or ("disabled" if not do_derate else "unknown"))),
    }
    return out
