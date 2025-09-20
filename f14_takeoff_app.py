# ============================================================
# F-14 Performance Calculator for DCS World — UI-first build
# File: f14_takeoff_app.py
# Version: v1.1.3-hotfix8 (2025-09-18)  ← perf-optimized + Intersection Margins
#
# Hotfix8 (perf):
# - Cached wrappers for perf_* calls (compute once, reuse).
# - Vectorized Intersection Margins section (no .apply / no loops).
# - Scoped execution (margins compute only when expander opened & data ready).
# - Lightweight timing captions for <2.5s budget checks.
#
# Prior changes retained (from your v1.1.3-hotfix7):
# - UI wording, layout, toggles, presets, labels, sections.
# - NATOPS/DCS baselines & compute wiring (unchanged).
# - DERATE slider 85%–100% RPM, Landing scenarios, etc.
# ============================================================
# 🚨 Bogged Down Protocol (BDP) 🚨
# 1) STOP  2) REVERT to last good tag  3) RESET chat if needed  4) SCOPE small
# 5) SAVE often with clear tags
# ============================================================

from __future__ import annotations
import re
import json
import time
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import streamlit as st

# Robust import for Streamlit Cloud
try:
    import f14_takeoff_core as core
except Exception:
    import sys, os, importlib
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    core = importlib.import_module("f14_takeoff_core")

# (Optional) show where Python loaded the core from for quick sanity-check
try:
    _core_path = getattr(core, "__file__", "?")
except Exception:
    _core_path = "?"

# =========================
# Page + global settings
# =========================
st.set_page_config(
    page_title="F-14 Performance — DCS (UI)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data path helpers (local-first with optional dev fallback) ---
from pathlib import Path as _Path
_APP_DIR = _Path(__file__).resolve().parent
_DATA_DIR = _APP_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
def ensure_csv(name: str, allow_fallback: bool = False) -> str:
    p = _DATA_DIR / name
    if p.exists():
        return str(p)
    if not allow_fallback:
        try:
            contents = sorted([x.name for x in _DATA_DIR.iterdir()])
        except Exception:
            contents = ["<unavailable>"]
        raise FileNotFoundError(
            "Required data file not found.\n"
            f"Expected file: {p}\n"
            f"Data folder:   {_DATA_DIR}\n"
            f"Folder exists: {_DATA_DIR.exists()}\n"
            f"Contents:      {contents}\n\n"
            "Fix: add the CSV to your repo at F-14_Performance_Toolkit/data/<name> "
            "and redeploy. Alternatively, enable the sidebar toggle "
            "'Allow network fallback (dev)' to fetch from GitHub raw."
        )
    try:
        import requests as _requests
        raw_url = ("https://raw.githubusercontent.com/"
                   "Zorinco96/F-14_Performance_Toolkit/refs/heads/main/data/"
                   + name)
        r = _requests.get(raw_url, timeout=15); r.raise_for_status()
        p.write_bytes(r.content)
        return str(p)
    except Exception as e:
        try: st.warning(f"Dev fallback failed for {name}: {e}")
        except Exception: pass
        raise
try:
    _DEV_FALLBACK = st.sidebar.toggle("Allow network fallback (dev)", value=False,
        help="If a CSV is missing under ./data, fetch from GitHub raw and cache it.")
except Exception:
    _DEV_FALLBACK = False
with st.sidebar.expander("Diagnostics", expanded=False):
    st.write("App dir:", _APP_DIR); st.write("Data dir:", _DATA_DIR)
    try:
        st.write("Data dir exists:", _DATA_DIR.exists())
        st.write("Data dir contents:", sorted([p.name for p in _DATA_DIR.iterdir()]))
    except Exception as _e:
        st.write("Cannot list data dir:", _e)
    st.write("Dev fallback enabled:", _DEV_FALLBACK)

# =========================
# Constants (UI placeholders)
# =========================
FT_PER_NM = 6076.11549
ISA_LAPSE_C_PER_1000FT = 1.98

# Simple fuel model placeholders
INTERNAL_FUEL_MAX_LB = 16200
EXT_TANK_FUEL_LB = 1800

# Simple W&B defaults
DEFAULT_GTOW = 74349
DEFAULT_LDW  = 60000

# Station list (mirrors DCS/Heatblur naming for glove A/B and tunnel)
STATIONS: List[str] = ["1A", "1B", "2", "3", "4", "5", "6", "7", "8A", "8B"]
SYMMETRY = {"1A": "8A", "8A": "1A", "1B": "8B", "8B": "1B", "2": "7", "7": "2", "3": "6", "6": "3", "4": "5", "5": "4"}

# Catalog (category tagging only used for future filtering)
STORES_CATALOG: Dict[str, str] = {
    "—": "—",
    "AIM-9M": "AIR-TO-AIR",
    "AIM-7M": "AIR-TO-AIR",
    "AIM-54C": "AIR-TO-AIR",
    "Mk-82": "BOMBS",
    "Mk-83": "BOMBS",
    "GBU-12": "BOMBS",
    "ZUNI LAU-10": "ROCKETS",
    "Drop Tank 267 gal": "FUEL TANKS",
    "LANTIRN": "PODS",
}

# Auto-quantity mapping used in Detailed W&B and Landing Scenario C
AUTO_QTY_BY_STORE = {
    "—": 0,
    "AIM-9M": 1,
    "AIM-7M": 1,
    "AIM-54C": 1,
    "Mk-82": 1,
    "Mk-83": 1,
    "GBU-12": 1,
    "ZUNI LAU-10": 1,
    "Drop Tank 267 gal": 1,
    "LANTIRN": 1,
}

# --- Stores → drag-deltas helper ---------------------------------------------
def get_stores_drag_list() -> list[str]:
    totals = {"AIM-9M": 0, "AIM-7M": 0, "AIM-54C": 0, "Drop Tank 267 gal": 0}
    pylon_pair = False
    for sta in STATIONS:
        store = st.session_state.get(f"store_{sta}", "—")
        qty_default = AUTO_QTY_BY_STORE.get(store, 0)
        try:
            qty = int(st.session_state.get(f"qty_{sta}", qty_default))
        except Exception:
            qty = qty_default
        if store in totals:
            totals[store] += max(0, qty)
        if bool(st.session_state.get(f"pylon_{sta}", False)):
            pylon_pair = True

    if bool(st.session_state.get("ext_left_full", False)):
        totals["Drop Tank 267 gal"] += 1
    if bool(st.session_state.get("ext_right_full", False)):
        totals["Drop Tank 267 gal"] += 1

    deltas: list[str] = []
    if pylon_pair: deltas.append("PylonPair")
    if totals["AIM-9M"]   >= 2: deltas.append("2xSidewinders")
    if totals["AIM-7M"]   >= 2: deltas.append("2xSparrows")
    if totals["AIM-54C"]  >= 2: deltas.append("2xPhoenix")
    if totals["Drop Tank 267 gal"] >= 2: deltas.append("FuelTank2x")
    seen = set(); out = []
    for d in deltas:
        if d not in seen:
            out.append(d); seen.add(d)
    return out
# ----------------------------------------------------------------------------- 

# =========================
# Robust numeric helpers
# =========================
def _s_int(val, default: int = 0) -> int:
    try:
        if val is None:
            return default
        if isinstance(val, (pd.Series,)):
            val = val.max(skipna=True)
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        try:
            v = float(val)
            return 0 if pd.isna(v) else int(v)
        except Exception:
            return default

def _s_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        if isinstance(val, (pd.Series,)):
            val = val.max(skipna=True)
        v = pd.to_numeric(val, errors="coerce")
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def _series_max(df: pd.DataFrame, col: str):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").max(skipna=True)
    return None

def _calibration_badge(text: str, tone: str = "warning"):
    palette = {
        "warning": ("#664200", "#FFF4CC", "#FFD666"),
        "ok":      ("#0B5E29", "#E5F7EC", "#6ED69E"),
        "info":    ("#133C7A", "#E7F1FF", "#86B7FE"),
        "error":   ("#7A1F1F", "#FFE8E6", "#FFA39E"),
    }
    fg, bg, bd = palette.get(tone, palette["info"])
    st.caption(
        f"<span style='color:{fg};background:{bg};border:1px solid {bd};"
        f"padding:2px 6px;border-radius:6px;font-size:12px;'>{text}</span>",
        unsafe_allow_html=True
    )

# =========================
# Data loading (CSV only)
# =========================
@st.cache_data(show_spinner=False)
def load_airports(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    for col in ("length_ft", "tora_ft", "toda_ft", "asda_ft", "threshold_elev_ft", "heading_deg", "lda_ft"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_perf(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    for col in ("gw_lbs", "flap_deg", "Vs_kt", "V1_kt", "Vr_kt", "V2_kt", "press_alt_ft", "oat_c"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Local-first CSV (with optional dev fallback)
try:
    airports = load_airports(ensure_csv("dcs_airports.csv", allow_fallback=_DEV_FALLBACK))
except FileNotFoundError as e:
    st.error("Airport database missing."); st.code(str(e)); st.stop()
try:
    perf = load_perf(ensure_csv("f14_perf.csv", allow_fallback=_DEV_FALLBACK))
except FileNotFoundError as e:
    st.error("Performance table missing."); st.code(str(e)); st.stop()
# =========================
# Reference V-speed helpers (from f14_perf.csv)
# =========================
def _closest_row_by_weight(df: pd.DataFrame, gw_lb: float, flap_deg_target: Optional[int]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    wcol = "gw_lbs"
    if wcol not in df.columns:
        return None
    tmp = df.copy()
    # optional flap filter if present in dataset
    if flap_deg_target is not None and "flap_deg" in tmp.columns:
        tmp = tmp[pd.to_numeric(tmp["flap_deg"], errors="coerce") == float(flap_deg_target)]
        if tmp.empty:
            tmp = df.copy()  # fallback: ignore flap if not present
    tmp["_werr"] = (pd.to_numeric(tmp[wcol], errors="coerce") - float(gw_lb)).abs()
    tmp = tmp.sort_values("_werr")
    return tmp.iloc[0] if not tmp.empty else None

def get_reference_vspeeds_from_csv(
    perf_df: Optional[pd.DataFrame],
    gw_lb: float,
    flaps_label: str
) -> Optional[dict]:
    """
    Pull Vs/V1/Vr/V2 from f14_perf.csv by nearest weight and flap setting if available.
    For flap mapping:
      - "UP"        -> flap_deg_target = 0
      - "MANEUVER"  -> flap_deg_target = 20  (approx UI baseline)
      - "FULL"      -> flap_deg_target = 25  (approx UI baseline)
    If any key is missing, returns None to allow model fallback.
    """
    if perf_df is None or perf_df.empty:
        return None

    flap_map = {"UP": 0, "MANEUVER": 20, "FULL": 25}
    flap_deg_target = flap_map.get(str(flaps_label).upper(), None)
    row = _closest_row_by_weight(perf_df, gw_lb, flap_deg_target)
    if row is None:
        return None

    needed = ["Vs_kt", "V1_kt", "Vr_kt", "V2_kt"]
    if not all(col in row.index for col in needed):
        return None

    try:
        return dict(
            Vs=float(row["Vs_kt"]),
            V1=float(row["V1_kt"]),
            Vr=float(row["Vr_kt"]),
            V2=float(row["V2_kt"]),
        )
    except Exception:
        return None
        
# --- Helper: V-speed lookup from f14_perf.csv (nearest neighbor by flaps & GW)
def _vs_lookup_from_perf_table(gw_lb: float, flaps_label: str) -> dict:
    """
    Returns a dict of any of: {"VR_kts": float, "V2_kts": float, "Vs_kts": float}
    pulled from your f14_perf.csv by nearest neighbor on flap_deg and gw_lbs.
    If the table or columns are missing, returns {} (no override).
    """
    try:
        if perf is None or getattr(perf, "empty", True):
            return {}

        # Map UI flaps to approximate flap_deg used in the perf table
        target_flap = {"UP": 0, "MANEUVER": 10, "FULL": 20}.get(str(flaps_label).upper(), 0)

        df = perf.copy()

        # Ensure numeric types for selection
        if "flap_deg" in df.columns:
            df["flap_deg"] = pd.to_numeric(df["flap_deg"], errors="coerce")
        if "gw_lbs" in df.columns:
            df["gw_lbs"] = pd.to_numeric(df["gw_lbs"], errors="coerce")

        # Narrow to closest flap_deg if present, otherwise keep all
        if "flap_deg" in df.columns and df["flap_deg"].notna().any():
            diff = (df["flap_deg"] - target_flap).abs()
            df = df.loc[diff == diff.min()]

        # Now choose the closest weight row
        if "gw_lbs" in df.columns and df["gw_lbs"].notna().any():
            idx = (df["gw_lbs"] - float(gw_lb)).abs().idxmin()
            row = df.loc[idx]
        else:
            # Fallback to first row if weight column isn't there
            row = df.iloc[0]

        out = {}
        # Copy over available columns → canonical keys used by UI (`t_res` keys)
        if "Vr_kt" in row and pd.notna(row["Vr_kt"]):
            out["VR_kts"] = float(row["Vr_kt"])
        if "V2_kt" in row and pd.notna(row["V2_kt"]):
            out["V2_kts"] = float(row["V2_kt"])
        if "Vs_kt" in row and pd.notna(row["Vs_kt"]):
            out["Vs_kts"] = float(row["Vs_kt"])
        return out
    except Exception:
        # Never break the app on lookup
        return {}
        
# --- V-Speeds lookup from f14_perf.csv (cached) ------------------------------
@st.cache_data(show_spinner=False)
def _vs_lookup_from_perf_table(gw_lb: float, flaps_label: str, thrust_mode: str) -> Optional[dict]:
    """
    Returns dict like {'Vs_kts': float, 'Vr_kts': float, 'V2_kts': float} from f14_perf.csv
    using simple linear interpolation on weight. If no suitable rows, returns None.
    """
    try:
        df = perf.copy()
        if df is None or df.empty:
            return None
        # Map flaps label to a flap_deg key used in the CSV
        flap_map = {"UP": 0, "MANEUVER": 20, "FULL": 35}
        flap_deg = flap_map.get(str(flaps_label).upper(), 0)

        # If the CSV has 'flap_deg', filter by it; otherwise keep all rows
        if "flap_deg" in df.columns:
            df = df[pd.to_numeric(df["flap_deg"], errors="coerce") == flap_deg]
            if df.empty:
                return None

        # If thrust is encoded in your CSV, you could filter here (e.g., a 'thrust_mode' column).
        # We skip thrust filtering for now because your shared sheet didn’t show such column.

        # Require needed columns
        needed = {"gw_lbs", "Vr_kt", "V2_kt"}
        if not needed.issubset(set(df.columns)):
            return None

        # Clean and sort by weight
        df = df[["gw_lbs", "Vs_kt", "Vr_kt", "V2_kt"]].dropna()
        df["gw_lbs"] = pd.to_numeric(df["gw_lbs"], errors="coerce")
        df = df.dropna(subset=["gw_lbs"]).sort_values("gw_lbs")

        # Edge cases
        if df.empty:
            return None
        w = float(gw_lb)
        # Exact or bracketed interpolation
        # Find rows just below and above w
        lower = df[df["gw_lbs"] <= w].tail(1)
        upper = df[df["gw_lbs"] >= w].head(1)

        if lower.empty and upper.empty:
            return None
        if lower.empty:
            row = upper.iloc[0]
            return {"Vs_kts": float(row.get("Vs_kt", float("nan"))),
                    "Vr_kts": float(row["Vr_kt"]),
                    "V2_kts": float(row["V2_kt"])}
        if upper.empty:
            row = lower.iloc[0]
            return {"Vs_kts": float(row.get("Vs_kt", float("nan"))),
                    "Vr_kts": float(row["Vr_kt"]),
                    "V2_kts": float(row["V2_kt"])}

        l = lower.iloc[0]; u = upper.iloc[0]
        if u["gw_lbs"] == l["gw_lbs"]:
            # same weight row
            return {"Vs_kts": float(l.get("Vs_kt", u.get("Vs_kt", float("nan")))),
                    "Vr_kts": float(l["Vr_kt"]),
                    "V2_kts": float(l["V2_kt"])}

        # Linear interpolation on weight
        t = (w - l["gw_lbs"]) / (u["gw_lbs"] - l["gw_lbs"])
        def lerp(a, b): 
            if pd.isna(a) or pd.isna(b): 
                return float("nan")
            return float(a + t * (b - a))

        vs = lerp(l.get("Vs_kt", float("nan")), u.get("Vs_kt", float("nan")))
        vr = lerp(l["Vr_kt"], u["Vr_kt"])
        v2 = lerp(l["V2_kt"], u["V2_kt"])
        return {"Vs_kts": vs, "Vr_kts": vr, "V2_kts": v2}
    except Exception:
        return None
# -----------------------------------------------------------------------------

# =========================
# Cached wrappers for heavy perf calls (hashable args only)
# =========================
perf_takeoff   = getattr(core, "perf_compute_takeoff", None)
perf_climb     = getattr(core, "perf_compute_climb", None)
perf_cruise    = getattr(core, "perf_compute_cruise", None)
perf_landing   = getattr(core, "perf_compute_landing", None)

@st.cache_data(show_spinner=False)
def cached_perf_takeoff(gw_lb: float, field_elev_ft: float, oat_c: float, headwind_kts: float,
                        runway_slope: float, thrust_mode: str, mode: str,
                        config: str, sweep_deg: float, stores: tuple) -> dict:
    if not callable(perf_takeoff): return {}
    return perf_takeoff(
        gw_lb=gw_lb, field_elev_ft=field_elev_ft, oat_c=oat_c, headwind_kts=headwind_kts,
        runway_slope=runway_slope, thrust_mode=thrust_mode, mode=mode,
        config=config, sweep_deg=sweep_deg, stores=list(stores)
    )

@st.cache_data(show_spinner=False)
def cached_perf_climb(gw_lb: float, alt_start_ft: float, alt_end_ft: float, oat_dev_c: float,
                      schedule: str, mode: str, power: str, sweep_deg: float, config: str) -> dict:
    if not callable(perf_climb): return {}
    return perf_climb(
        gw_lb=gw_lb, alt_start_ft=alt_start_ft, alt_end_ft=alt_end_ft, oat_dev_c=oat_dev_c,
        schedule=schedule, mode=mode, power=power, sweep_deg=sweep_deg, config=config
    )

@st.cache_data(show_spinner=False)
def cached_perf_landing(gw_lb: float, field_elev_ft: float, oat_c: float, headwind_kts: float,
                        mode: str, config: str, sweep_deg: float) -> dict:
    if not callable(perf_landing): return {}
    return perf_landing(
        gw_lb=gw_lb, field_elev_ft=field_elev_ft, oat_c=oat_c, headwind_kts=headwind_kts,
        mode=mode, config=config, sweep_deg=sweep_deg
    )

# Helpers (unit detect, parsing)
def detect_length_unit(text: str) -> Tuple[Optional[float], str]:
    if text is None: return None, ""
    s = text.strip().lower()
    if not s: return None, ""
    nm_match = re.search(r"([0-9]*\.?[0-9]+)\s*(nm|nmi)", s)
    if nm_match:
        nm = float(nm_match.group(1)); return nm * FT_PER_NM, "NM (auto)"
    num_match = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not num_match: return None, ""
    val = float(num_match.group(1))
    if val <= 5: return val * FT_PER_NM, "NM (heuristic)"
    return val, "ft (auto)"

def detect_pressure(qnh_text: str) -> Tuple[Optional[float], str]:
    if not qnh_text: return None, ""
    s = qnh_text.strip().lower()
    hpa_match = re.search(r"([0-9]{3,4})\s*(hpa|mb)", s)
    inhg_match = re.search(r"([0-9]*\.?[0-9]+)\s*(inhg|hg)", s)
    num_match = re.search(r"([0-9]*\.?[0-9]+)", s)
    if hpa_match:
        hpa = float(hpa_match.group(1)); return hpa * 0.0295299830714, "hPa → inHg"
    if inhg_match: return float(inhg_match.group(1)), "inHg"
    if num_match:
        val = float(num_match.group(1))
        if 900 <= val <= 1100: return val * 0.0295299830714, "hPa (heuristic) → inHg"
        return val, "inHg (assumed)"
    return None, ""

def parse_wind(text: str) -> Dict[str, Any]:
    if not text: return {"dir_deg": None, "spd_kts": None, "unit": ""}
    s = text.strip().lower()
    m = re.search(r"(\d{2,3})\s*[/@]??\s*([0-9]*\.?[0-9]+)\s*(m/s|ms|kt|kts)?", s)
    if not m: return {"dir_deg": None, "spd_kts": None, "unit": ""}
    deg = int(m.group(1)); val = float(m.group(2)); unit = (m.group(3) or "kt").replace("ms","m/s")
    spd_kts = val * 1.94384 if unit == "m/s" else val
    return {"dir_deg": deg, "spd_kts": spd_kts, "unit": "m/s→kt" if unit == "m/s" else "kt"}

def temp_at_elevation(temp_sl_c: Optional[float], elev_ft: float, lapse_c_per_1000ft: float = ISA_LAPSE_C_PER_1000FT) -> Optional[float]:
    if temp_sl_c is None: return None
    return float(temp_sl_c - lapse_c_per_1000ft * (elev_ft / 1000.0))

def hw_xw_components(wind_dir: Optional[int], wind_kts: Optional[float], rwy_heading_deg: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    import math
    if None in (wind_dir, wind_kts, rwy_heading_deg): return None, None
    angle = math.radians((wind_dir - rwy_heading_deg) % 360)
    hw = wind_kts * math.cos(angle); xw = wind_kts * math.sin(angle)
    return hw, abs(xw)

# --- Wind policy application (App Scope Definition) ---------------------------
def apply_wind_policy(hw_raw_kts: float, use_50_150: bool) -> float:
    # +HW is headwind, -HW tailwind
    if use_50_150:
        return 0.5 * max(hw_raw_kts, 0.0) + 1.5 * min(hw_raw_kts, 0.0)
    else:
        return 0.0 * max(hw_raw_kts, 0.0) + 1.5 * min(hw_raw_kts, 0.0)

# --- Evaluate one (flaps, thrust) candidate against gates --------------------
def _evaluate_candidate(flaps_label: str,
                        thrust_label: str,   # "DERATE (xx%)", "MILITARY", or "AFTERBURNER (required)"
                        thrust_mode: str,    # "MIL" or "MAX" for perf engine
                        hw_eff_kts: float,
                        ctx: dict) -> dict:
    """
    Returns a dict with:
      - todr_ft, asdr_ft, diff_ratio
      - pass_runway, pass_climb, dispatchable
      - v_speeds dict
    Uses perf_* wrappers directly (cached).
    """
    cfg = "TO_FLAPS" if flaps_label in ("MANEUVER", "FULL") else "CLEAN"

    # TAKEOFF perf (memoized wrapper)
    t_res = cached_perf_takeoff(
        gw_lb=ctx["gw_lb"],
        field_elev_ft=ctx["field_elev_ft"],
        oat_c=ctx["oat_c"],
        headwind_kts=hw_eff_kts,
        runway_slope=ctx["runway_slope"],
        thrust_mode=("MAX" if thrust_mode == "MAX" else "MIL"),
        mode=ctx["mode"],
        config=cfg,
        sweep_deg=20.0,
        stores=tuple(ctx["stores"]),
    )

    # Distances
    gr  = float(t_res.get("GroundRoll_ft", 0.0) or 0.0)
    d35 = float(t_res.get("DistanceTo35ft_ft", 0.0) or 0.0)

    # Prefer true ASDR if core exposes; else surrogate
    asdr_ft = float(t_res.get("ASDR_ft", 0.0) or 0.0)
    if asdr_ft <= 0.0:
        asdr_ft = gr * 1.15 if gr > 0 else d35 * 1.10

    # Prefer explicit OEI TODR if available; else AEO d35 baseline
    todr_ft = float(t_res.get("TODR_OEI_35ft_ft", 0.0) or 0.0)
    if todr_ft <= 0.0:
        todr_ft = d35

    # Balanced-Field diff ratio
    m = max(asdr_ft, todr_ft) if max(asdr_ft, todr_ft) > 0 else 1.0
    diff_ratio = abs(asdr_ft - todr_ft) / m

    # CLIMB perf (AEO gradient to 1000 ft) — only if runway gates pass
    aeo_grad = None
    tora = float(ctx.get("available_tora_ft", 0.0))
    asda = float(ctx.get("available_asda_ft", tora))
    pass_runway_precheck = (todr_ft <= tora) and (asdr_ft <= asda)

    if pass_runway_precheck:
        try:
            cres = cached_perf_climb(
                gw_lb=ctx["gw_lb"],
                alt_start_ft=max(0.0, ctx["field_elev_ft"]),
                alt_end_ft=max(1000.0 + ctx["field_elev_ft"], ctx["field_elev_ft"] + 1.0),
                oat_dev_c=0.0,
                schedule="NAVY",
                mode="DCS",
                power=("MAX" if thrust_mode == "MAX" else "MIL"),
                sweep_deg=20.0,
                config=("CLEAN" if cfg == "CLEAN" else "TO_FLAPS"),
            )
            # Prefer explicit gradient if provided; else simple estimate from distance
            explicit = cres.get("AEO_min_grad_ft_per_nm_to_1000") or cres.get("Grad_ft_per_nm")
            if explicit is not None:
                aeo_grad = float(explicit)
            else:
                d_nm = float(cres.get("Distance_nm", 0.0) or 0.0)
                aeo_grad = (1000.0 / d_nm) if d_nm > 0 else None
        except Exception:
            aeo_grad = None

    # Gates
    req_grad = float(ctx.get("req_grad_ft_nm", 200.0))
    pass_runway = pass_runway_precheck
    pass_climb  = (aeo_grad is None) or (aeo_grad >= req_grad)  # unknown climb => pass

    dispatchable = pass_runway and pass_climb

    # Base V-speeds from engine result
    v_speeds = {
        "V1_kts": float(t_res.get("VR_kts", 0.0) or 0.0),  # placeholder = Vr until explicit V1 provided
        "Vr_kts": float(t_res.get("VR_kts", 0.0) or 0.0),
        "V2_kts": float(t_res.get("V2_kts", 0.0) or 0.0),
        "Vfs_kts": float(max(
            (t_res.get("V2_kts") or 0.0) * 1.1,
            (t_res.get("VLOF_kts") or 0.0) * 1.15
        )),
    }

    # Gentle override from f14_perf.csv (if available)
    try:
        vs_tbl = _vs_lookup_from_perf_table(ctx["gw_lb"], flaps_label, thrust_mode)
        if vs_tbl:
            if not pd.isna(vs_tbl.get("Vs_kts", float("nan"))):
                v_speeds["Vs_kts"] = float(vs_tbl["Vs_kts"])
            if not pd.isna(vs_tbl.get("Vr_kts", float("nan"))):
                v_speeds["Vr_kts"] = float(vs_tbl["Vr_kts"])
                v_speeds["V1_kts"] = float(vs_tbl["Vr_kts"])  # mirror Vr until we have V1
            if not pd.isna(vs_tbl.get("V2_kts", float("nan"))):
                v_speeds["V2_kts"] = float(vs_tbl["V2_kts"])
                v_speeds["Vfs_kts"] = float(max(
                    v_speeds["V2_kts"] * 1.1,
                    (t_res.get("VLOF_kts") or 0.0) * 1.15
                ))
    except Exception:
        pass  # never block on table lookup

    return {
        "flaps": flaps_label,
        "thrust_label": thrust_label,
        "thrust_mode": thrust_mode,
        "t_res": t_res,
        "todr_ft": todr_ft,
        "asdr_ft": asdr_ft,
        "diff_ratio": diff_ratio,
        "aeo_grad_ft_nm": aeo_grad,
        "pass_runway": pass_runway,
        "pass_climb": pass_climb,
        "dispatchable": dispatchable,
        "margins": {
            "tora_margin_ft": tora - todr_ft,
            "asda_margin_ft": asda - asdr_ft,
        },
        "v": v_speeds,
    }


    # Distances
    gr  = float(t_res.get("GroundRoll_ft", 0.0) or 0.0)
    d35 = float(t_res.get("DistanceTo35ft_ft", 0.0) or 0.0)

    # Prefer true ASDR if core exposes; else surrogate (documented)
    asdr_ft = float(t_res.get("ASDR_ft", 0.0) or 0.0)
    if asdr_ft <= 0.0:
        asdr_ft = gr * 1.15 if gr > 0 else d35 * 1.10  # surrogate only if core ASDR not provided

    # Prefer explicit OEI TODR if available; else AEO d35 per baseline
    todr_ft = float(t_res.get("TODR_OEI_35ft_ft", 0.0) or 0.0)
    if todr_ft <= 0.0:
        todr_ft = d35

    # Balanced-Field diff ratio
    m = max(asdr_ft, todr_ft) if max(asdr_ft, todr_ft) > 0 else 1.0
    diff_ratio = abs(asdr_ft - todr_ft) / m

    # CLIMB perf (AEO gradient to 1000 ft) — only if runway gates pass
    aeo_grad = None
    tora = float(ctx.get("available_tora_ft", 0.0))
    asda = float(ctx.get("available_asda_ft", tora))
    pass_runway_precheck = (todr_ft <= tora) and (asdr_ft <= asda)

    if pass_runway_precheck:
        try:
            cres = cached_perf_climb(
                gw_lb=ctx["gw_lb"],
                alt_start_ft=max(0.0, ctx["field_elev_ft"]),
                alt_end_ft=max(1000.0 + ctx["field_elev_ft"], ctx["field_elev_ft"] + 1.0),
                oat_dev_c=0.0,
                schedule="NAVY",        # baseline schedule for AEO check
                mode="DCS",
                power=("MAX" if thrust_mode == "MAX" else "MIL"),
                sweep_deg=20.0,
                config=("CLEAN" if cfg == "CLEAN" else "TO_FLAPS"),
            )

            # Prefer explicit gradient if core provides one
            explicit = cres.get("AEO_min_grad_ft_per_nm_to_1000", None)
            if explicit is None:
                explicit = cres.get("Grad_ft_per_nm", None)

            if explicit is not None:
                aeo_grad = float(explicit)
            else:
                # Fallback estimate: 1000 ft gained over the horizontal distance to 1000 ft
                d_nm = float(cres.get("Distance_nm", 0.0) or 0.0)
                if d_nm > 0:
                    aeo_grad = 1000.0 / d_nm
                else:
                    # No usable info → leave as None (unknown), do not fail climb gate
                    aeo_grad = None

        except Exception:
            aeo_grad = None

    # Gates
    tora = float(ctx.get("available_tora_ft", 0.0))
    asda = float(ctx.get("available_asda_ft", tora))
    req_grad = float(ctx.get("req_grad_ft_nm", 200.0))

    pass_runway = pass_runway_precheck
    pass_climb  = (aeo_grad is None) or (aeo_grad >= req_grad)  # if climb not computed (because runway failed), this will be True

    dispatchable = pass_runway and pass_climb
    # --- CSV V-speed override (Auto-Select and manual share the same source) ---
    # If the perf table has a better Vr/V2/Vs for this (flaps, weight), gently overwrite.
    _csv_vs = _vs_lookup_from_perf_table(ctx["gw_lb"], flaps_label)
    if _csv_vs:
        try:
            # Merge into t_res using canonical keys VR_kts / V2_kts / Vs_kts (if provided)
            for _k in ("VR_kts", "V2_kts", "Vs_kts"):
                if _k in _csv_vs and _csv_vs[_k] is not None:
                    t_res[_k] = float(_csv_vs[_k])
        except Exception:
            # non-fatal if types are odd
            pass

    # Package
    

    # Override with table-based V-speeds if available (gentle override)
    try:
        vs_tbl = _vs_lookup_from_perf_table(ctx["gw_lb"], flaps_label, thrust_mode)
        if vs_tbl:
            if not pd.isna(vs_tbl.get("Vs_kts", float("nan"))):
                v_speeds["Vs_kts"] = float(vs_tbl["Vs_kts"])
            if not pd.isna(vs_tbl.get("Vr_kts", float("nan"))):
                v_speeds["Vr_kts"] = float(vs_tbl["Vr_kts"])
                v_speeds["V1_kts"] = float(vs_tbl["Vr_kts"])  # still mirror Vr until explicit V1 exists
            if not pd.isna(vs_tbl.get("V2_kts", float("nan"))):
                v_speeds["V2_kts"] = float(vs_tbl["V2_kts"])
                v_speeds["Vfs_kts"] = float(max(
                    v_speeds["V2_kts"] * 1.1,
                    (t_res.get("VLOF_kts") or 0.0) * 1.15
                ))
    except Exception:
        # Don’t block app if CSV lookup fails
        pass


    return dict(
        flaps=flaps_label,
        thrust_label=thrust_label,
        thrust_mode=thrust_mode,
        t_res=t_res,
        todr_ft=todr_ft,
        asdr_ft=asdr_ft,
        diff_ratio=diff_ratio,
        aeo_grad_ft_nm=aeo_grad,
        pass_runway=pass_runway,
        pass_climb=pass_climb,
        dispatchable=dispatchable,
        margins=dict(
            tora_margin_ft=tora - todr_ft,
            asda_margin_ft=asda - asdr_ft,
        ),
        v=v_speeds,
    )


# Fuel helpers
def compute_total_fuel_lb(from_percent: Optional[float], ext_left_full: bool, ext_right_full: bool) -> Optional[float]:
    if from_percent is None: return None
    internal = INTERNAL_FUEL_MAX_LB * max(0.0, min(100.0, from_percent)) / 100.0
    ext = (EXT_TANK_FUEL_LB if ext_left_full else 0) + (EXT_TANK_FUEL_LB if ext_right_full else 0)
    return internal + ext

def compute_percent_from_total(total_lb: Optional[float], ext_left_full: bool, ext_right_full: bool) -> Optional[float]:
    if total_lb is None: return None
    ext = (EXT_TANK_FUEL_LB if ext_left_full else 0) + (EXT_TANK_FUEL_LB if ext_right_full else 0)
    internal = max(0.0, total_lb - ext)
    return max(0.0, min(100.0, (internal / INTERNAL_FUEL_MAX_LB) * 100.0))

# =========================
# Sidebar: curated presets + flags
# =========================
with st.sidebar:
    st.title("F-14 Performance — DCS")
    st.caption("UI skeleton • v1.1.3-hotfix8 (perf-optimized; math preserved)")

    st.subheader("Quick Presets (F-14B)")
    preset = st.selectbox(
        "Load preset",
        [
            "—",
            "Fleet CAP: 2× AIM-54C, 2× AIM-7M, 2× AIM-9M, 2× tanks",
            "Heavy Intercept: 6× AIM-54C, 2× AIM-9M",
            "Bombcat LANTIRN: pod + 4× GBU-12, 2× AIM-9M, 1× tank",
            "Strike (iron): 6× Mk-82, 2× AIM-9M, 1× tank",
        ],
        index=0,
    )

    def apply_preset(name: str):
        for sta in STATIONS:
            st.session_state[f"store_{sta}"] = "—"
            st.session_state[f"qty_{sta}"] = 0
            st.session_state[f"pylon_{sta}"] = False
        st.session_state.setdefault("ext_left_full", False)
        st.session_state.setdefault("ext_right_full", False)
        st.session_state["ext_left_full"] = False
        st.session_state["ext_right_full"] = False
        st.session_state["fuel_input_mode"] = "Percent"
        st.session_state["fuel_percent"] = 80.0
        st.session_state["fuel_total_lb"] = compute_total_fuel_lb(80.0, False, False)

        def set_sta(sta, store, qty=1):
            st.session_state[f"store_{sta}"] = store
            st.session_state[f"qty_{sta}"] = qty

        if "Fleet CAP" in name:
            for s in ("3","6"): set_sta(s, "AIM-54C")
            for s in ("4","5"): set_sta(s, "AIM-7M")
            for s in ("1A","8A"): set_sta(s, "AIM-9M")
            for s in ("2","7"): set_sta(s, "Drop Tank 267 gal")
            st.session_state["ext_left_full"] = True
            st.session_state["ext_right_full"] = True
            st.session_state["fuel_total_lb"] = compute_total_fuel_lb(80.0, True, True)
        elif "Heavy Intercept" in name:
            for s in ("3","4","5","6"): set_sta(s, "AIM-54C")
            for s in ("1A","8A"): set_sta(s, "AIM-9M")
            st.session_state["fuel_total_lb"] = compute_total_fuel_lb(80.0, False, False)
        elif "Bombcat" in name:
            set_sta("8B", "LANTIRN")
            for s in ("4","5"): set_sta(s, "GBU-12", qty=2)
            for s in ("1A","8A"): set_sta(s, "AIM-9M")
            set_sta("7", "Drop Tank 267 gal")
            st.session_state["ext_right_full"] = True
            st.session_state["fuel_total_lb"] = compute_total_fuel_lb(80.0, False, True)
        elif "Strike (iron)" in name:
            for s in ("3","4","5","6"): set_sta(s, "Mk-82")
            for s in ("1A","8A"): set_sta(s, "AIM-9M")
            set_sta("7", "Drop Tank 267 gal")
            st.session_state["ext_right_full"] = True
            st.session_state["fuel_total_lb"] = compute_total_fuel_lb(80.0, False, True)

    if preset != "—":
        apply_preset(preset)
        st.info(f"Preset applied: {preset} — Stores/Fuel only. Flaps/Thrust remain AUTO.")

    st.subheader("Flags")
auto_recompute = st.toggle("Auto-recompute", value=True)
show_debug = st.toggle("Show scenario JSON", value=False)

st.subheader("Policy")
wind_policy_choice = st.selectbox(
    "Wind credit (runway & climb calcs)",
    ["50% headwind / 150% tailwind (default)", "0% headwind / 150% tailwind (more conservative)"],
    index=0,
    help="Conservative wind credit policy:\n• 50/150 = use half of headwind benefit and 1.5× tailwind penalty.\n• 0/150 = ignore headwind benefit entirely (extra conservative), still penalize tailwind by 1.5×.\nThis affects balanced-field and climb calculations."
)

# Sticky header
st.markdown(
    """
    <div style="position:sticky;top:0;background:var(--background-color);padding:0.4rem 0;z-index:5;border-bottom:1px solid rgba(255,255,255,0.1)">
        <strong>F-14 Performance — DCS World</strong>
        <span style="opacity:0.7"> • UI-only v1.1.3 • Auto-recompute ON</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Section 1 — Aircraft
# =========================
with st.expander("1) Aircraft", expanded=True):
    ac = st.selectbox("Aircraft", ["F-14B (Tomcat)", "F-14A (future)", "F/A-18C (future)"], index=0)
    st.caption("Selecting the airframe sets defaults for flaps/thrust and W&B stations.")

# =========================
# Section 2 — Runway (GLOBAL search + manual override)
# =========================
@st.cache_data(show_spinner=False)
def load_airports_cached() -> pd.DataFrame:
    return airports

with st.expander("2) Runway", expanded=True):
    airports_cached = load_airports_cached()
    c1, c2, c3 = st.columns([1.4, 1.2, 1])

    with c1:
        # ① Pick MAP first
        maps = sorted(airports_cached["map"].dropna().unique().tolist())
        map_sel = st.selectbox("Map", maps, key="rw_map")

        # Subset to selected map
        sub_all = airports_cached[airports_cached["map"] == map_sel]
        sub_all = sub_all[sub_all["airport_name"].notna()]

        # ② Search within the selected map
        search_all = st.text_input("Search airport (selected map only)", placeholder="Type part of the airport name…")
        matches = sub_all[sub_all["airport_name"].str.contains(search_all, case=False, na=False)] if search_all else sub_all

        # ③ Airport picker (filtered by map and optional search)
        pick_names = sorted(matches["airport_name"].unique().tolist())
        apt = st.selectbox("Airport", pick_names, key="rw_airport")

        # Rows for the chosen airport on this map
        sub = sub_all[sub_all["airport_name"] == apt]


    with c2:
        rwy_rows = sub
        ends = rwy_rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        rwy_end = st.selectbox("Runway End / Intersection", sorted(ends) if ends else ["Full Length"], key="rw_end")

        # TORA: prefer specific end; else length_ft; else 0
        if "runway_end" in rwy_rows.columns:
            mask = rwy_rows["runway_end"].astype(str) == str(rwy_end)
            tora_candidate = _series_max(rwy_rows.loc[mask], "tora_ft")
        else:
            mask = None
            tora_candidate = None
        if tora_candidate is None or pd.isna(tora_candidate):
            tora_candidate = _series_max(rwy_rows, "length_ft")
        tora = _s_int(tora_candidate, 0)

        elev = _s_int(_series_max(rwy_rows, "threshold_elev_ft"), 0)
        hdg  = _s_float(_series_max(rwy_rows, "heading_deg"), 0.0)

        st.metric("Takeoff Run Available (ft)", f"{tora:,}")
        st.metric("Field Elevation (ft)", f"{elev:,}")
        st.metric("Runway Heading (°T)", f"{hdg:.0f}")

    with c3:
        st.checkbox("Manual runway entry", value=False, key="rw_manual")
        if st.session_state["rw_manual"]:
            mr_len = st.text_input("Runway length (ft or NM)", placeholder="8500 or 1.4 NM")
            len_ft, unit_label = detect_length_unit(mr_len)
            st.caption(f"Detected: {unit_label or '—'} → {f'{_s_int(len_ft):,} ft' if len_ft else ''}")
            mr_elev = st.number_input("Elevation (ft)", value=_s_int(elev, 0), step=50, min_value=0, format="%d")
            mr_hdg = st.number_input("Heading (°T)", value=_s_int(hdg, 0), step=1, min_value=0, max_value=359, format="%d")
            mr_tora = st.number_input("TORA (ft)", value=_s_int(len_ft if len_ft else tora, 0), step=100, min_value=0, format="%d")
            if len_ft:
                tora = _s_int(len_ft, tora)
            elev, hdg = _s_int(mr_elev, 0), float(_s_int(mr_hdg, 0))
            st.info("Manual values override database for calculations.")

# =========================
# Section 3 — Environment (paste parser + manual) — defaults to Manual
# =========================
with st.expander("3) Environment", expanded=True):
    mode_env = st.radio("Input mode", ["Paste from DCS briefing", "Manual"], horizontal=True, index=1)

    if mode_env == "Paste from DCS briefing":
        blob = st.text_area("Paste briefing text", height=160, placeholder="Paste the DCS weather section here…")
        if blob:
            temp_m = re.search(r"temp[^\d-]*(-?\d+)", blob, flags=re.I)
            qnh_m  = re.search(r"qnh[^\d]*(\d{3,4}|\d+\.?\d*)", blob, flags=re.I)
            wind_m = re.search(r"(\d{2,3})\s*[/@]\s*(\d+\.?\d*)\s*(kt|kts|m/s)?", blob, flags=re.I)
            temp_sl = float(temp_m.group(1)) if temp_m else None
            qnh_text = qnh_m.group(1) + (" hPa" if (qnh_m and len(qnh_m.group(1)) >= 3) else "") if qnh_m else ""
            qnh_inhg, qnh_label = detect_pressure(qnh_text)
            wind = parse_wind(wind_m.group(0)) if wind_m else {"dir_deg": None, "spd_kts": None, "unit": ""}
            st.success("Parsed tokens:")
            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("Sea-level Temperature (°C)", f"{temp_sl if temp_sl is not None else '—'}")
            with cB:
                st.metric("QNH (inHg)", f"{qnh_inhg:.2f}" if qnh_inhg else "—")
                st.caption(qnh_label)
            with cC:
                st.metric("Wind", f"{wind['dir_deg'] or '—'}/{wind['spd_kts']:.0f} kt" if wind["spd_kts"] else "—")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            temp_sl = st.number_input("Sea-level Temperature (°C)", value=15, step=1, format="%d")
            qnh_text = st.text_input("QNH (inHg or hPa)", value="29.92")
            qnh_inhg, qnh_label = detect_pressure(qnh_text)
            st.caption(f"Detected: {qnh_label}")
        with c2:
            wind_text = st.text_input("Wind (deg/speed)", value="000/00 kt", help="e.g. 090/7m/s or 270/10 kt")
            w = parse_wind(wind_text)
            st.caption(f"Detected: {w['unit']}")
        with c3:
            field_temp = temp_at_elevation(temp_sl, _s_int(locals().get("elev", 0)), ISA_LAPSE_C_PER_1000FT)
            st.metric("Estimated Temperature at Field (°C)", f"{field_temp:.1f}" if field_temp is not None else "—")
            hw, xw = hw_xw_components(w.get("dir_deg"), w.get("spd_kts"), _s_float(locals().get("hdg", 0.0)))
            st.metric("Head/Tailwind Component (kt)", f"{hw:+.0f}" if hw is not None else "—")
            st.metric("Crosswind Component (kt)", f"{xw:.0f}" if xw is not None else "—")

# =========================
# Section 4 — Weight & Balance (Simple or Detailed DCS-style)
# =========================
with st.expander("4) Weight & Balance", expanded=True):
    wb_mode = st.radio("Mode", ["Simple (enter Gross Takeoff Weight)", "Detailed (DCS-style loadout)"])

    if wb_mode.startswith("Simple"):
        gw_tow = st.number_input("Gross Takeoff Weight (lb)", value=int(DEFAULT_GTOW), step=1, min_value=0, format="%d")
        st.caption("MTOW (maximum takeoff weight) is **74,349 lb** — used as the default GTOW.")
        gw_ldg_plan = st.number_input("Planned Landing Weight (lb)", value=int(DEFAULT_LDW), step=1, min_value=0, format="%d")
        st.caption("**MLDW (field) = 60,000 lb**, **MLDW (carrier) = 54,000 lb**.")
        st.caption("Switch to Detailed mode to build weight via stations and fuel.")
    else:
        st.markdown("**Fuel** — Enter by pounds or percent. External tanks are either FULL or EMPTY.")
        cF1, cF2 = st.columns(2)
        fuel_input_mode = cF1.radio("Fuel input", ["Pounds (lb)", "Percent"], index=0, key="fuel_input_mode")
        cT = st.columns(3)
        with cT[0]:
            ext_left_full = st.checkbox("External Tank LEFT: FULL", key="ext_left_full")
        with cT[1]:
            ext_right_full = st.checkbox("External Tank RIGHT: FULL", key="ext_right_full")
        with cT[2]:
            st.caption("Landing scenarios assume external tanks are EMPTY of fuel.")

        if fuel_input_mode.startswith("Pounds"):
            fuel_total_lb = st.number_input("Total Fuel (lb)", value=int(st.session_state.get("fuel_total_lb", 12000)), step=1, min_value=0, format="%d", key="fuel_total_lb")
            computed_pct = compute_percent_from_total(float(fuel_total_lb), ext_left_full, ext_right_full)
            st.metric("Computed Internal Fuel (%)", f"{computed_pct:.0f}%" if computed_pct is not None else "—")
            st.session_state["fuel_percent"] = computed_pct
        else:
            fuel_percent = st.number_input("Total Fuel (%)", value=int(st.session_state.get("fuel_percent", 80)), min_value=0, max_value=100, step=1, format="%d", key="fuel_percent")
            computed_total = compute_total_fuel_lb(float(fuel_percent), ext_left_full, ext_right_full)
            st.metric("Computed Total Fuel (lb)", f"{_s_int(computed_total):,}" if computed_total is not None else "—")
            st.session_state["fuel_total_lb"] = computed_total

        # Import stubs
        cimp1, cimp2 = st.columns(2)
        std_choice = cimp1.selectbox("Import standard loadout (stub)",
                                     ["—","Fleet CAP","Heavy Intercept","Bombcat LANTIRN","Strike (iron)"])
        miz = cimp2.file_uploader("Import from DCS .miz (stub)", type=["miz"])
        compat_beta = st.checkbox("Compatibility Mode (beta)", value=False,
                                  help="Filters obviously impossible station/store pairs (approx).")

        st.markdown("**Loadout (F-14B)** — Click station tiles and pick a store; qty is auto-set.")
        cols = st.columns(5)
        for i, sta in enumerate(STATIONS):
            with cols[i % 5]:
                st.write(f"**STA {sta}**")
                store_key, qty_key, pylon_key = f"store_{sta}", f"qty_{sta}", f"pylon_{sta}"
                cur_store = st.session_state.get(store_key, "—")

                allowed = list(STORES_CATALOG.keys())
                if compat_beta:
                    if sta in ("1A","8A"):
                        allowed = ["—","AIM-9M"]
                    elif sta in ("1B","8B"):
                        allowed = ["—","AIM-7M","LANTIRN"]
                    elif sta in ("2","7"):
                        allowed = ["—","Drop Tank 267 gal"]
                    elif sta in ("3","4","5","6"):
                        allowed = ["—","AIM-54C","AIM-7M","Mk-82","Mk-83","GBU-12","ZUNI LAU-10"]

                st.selectbox(f"Store {sta}", allowed,
                             index=(allowed.index(cur_store) if cur_store in allowed else 0),
                             key=store_key)

                auto_qty = AUTO_QTY_BY_STORE.get(st.session_state[store_key], 0)
                st.number_input(f"Qty {sta}", value=int(auto_qty),
                                min_value=0, max_value=2, step=1, key=qty_key, disabled=True, format="%d")

                st.checkbox("Remove pylon", value=bool(st.session_state.get(pylon_key, False)), key=pylon_key)

                sym = SYMMETRY.get(sta)
                if sym and st.button(f"Apply → {sym}", key=f"symbtn_{sta}"):
                    st.session_state[f"store_{sym}"] = st.session_state[store_key]
                    st.session_state[f"qty_{sym}"] = AUTO_QTY_BY_STORE.get(st.session_state[store_key], 0)
                    st.session_state[f"pylon_{sym}"] = st.session_state[pylon_key]

        st.markdown("### Totals")

# External tanks flags
ext_left_full  = bool(st.session_state.get("ext_left_full", False))
ext_right_full = bool(st.session_state.get("ext_right_full", False))
fuel_total_lb = float(st.session_state.get("fuel_total_lb") or 0)

stations_dict = {}
for sta in STATIONS:
    stations_dict[sta] = {
        "store_weight_lb": 0.0,
        "pylon_weight_lb": 0.0,
        "qty": int(st.session_state.get(f"qty_{sta}", 0) or 0)
    }

simple_gw = gw_tow if wb_mode.startswith("Simple") else None

press_alt = core.pressure_altitude_ft(locals().get("elev", 0), locals().get("qnh_inhg", None))
sigma = core.density_ratio_sigma(press_alt, locals().get("field_temp", 15))

wb = core.build_loadout_totals(
    stations=stations_dict,
    fuel_lb=fuel_total_lb,
    ext_tanks=(ext_left_full, ext_right_full),
    mode_simple_gw=simple_gw
)

t1, t2, t3 = st.columns(3)
# GW always shown
t1.metric("Gross Weight (lb)", f"{wb['gw_tow_lb']:.0f}")

# %MAC logic: show placeholder in Simple mode or if value unavailable
cg_val = wb.get("cg_percent_mac", None)
cg_is_number = (cg_val is not None) and (not pd.isna(cg_val))
simple_mode = bool(str(wb_mode).startswith("Simple"))

if simple_mode or not cg_is_number:
    # Placeholder label only — avoid asserting a numeric when we lack detailed W&B
    t2.metric("Center of Gravity (%MAC)", "— % (std)")
    st.caption("CG shown as standardized placeholder — enter Detailed W&B for an exact %MAC.")
else:
    t2.metric("Center of Gravity (%MAC)", f"{float(cg_val):.1f}")

# Trim as before (if your core returns a numeric)
t3.metric("Stabilizer Trim (units)", f"{wb.get('stab_trim_units', 0.0):+0.1f}")


st.caption(f"PA: {int(press_alt):,} ft • Fuel TOW: {wb['fuel_tow_lb']:.0f} lb • Fuel LDG: {wb['fuel_ldg_lb']:.0f} lb")

# =========================
# Section 5 — Takeoff Configuration
# =========================
with st.expander("5) Takeoff Configuration", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        flaps = st.radio("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0, horizontal=False)
    with c2:
        thrust = st.radio("Thrust", ["Auto-Select", "MILITARY", "AFTERBURNER", "DERATE (Manual)"], index=0, horizontal=False)
    with c3:
        derate = 0
        if thrust == "DERATE (Manual)":
            derate = st.slider("Derate (RPM %)", min_value=85, max_value=100, value=95)
    st.caption("AUTO thrust will target 14 CFR 121.189 and ≥300 ft/NM AEO using the minimum required setting (to be modeled).")
req_grad = st.number_input("Required climb gradient (ft/nm)", min_value=0, max_value=1000, value=200, step=10)
st.session_state["req_climb_grad_ft_nm"] = int(req_grad)

# =========================
# (6) Takeoff Results
# =========================
st.header("Takeoff Results")


# --- Resolved thrust display for guidance table (manual DERATE honored) ---
def _resolved_thrust_display(thrust_choice: str, derate_slider_val):
    t = (thrust_choice or "").upper()
    if "DERATE" in t:
        try:
            pct = int(derate_slider_val if derate_slider_val is not None else 95)
        except Exception:
            pct = 95
        pct = max(85, min(100, pct))
        return (f"DERATE ({pct}%)", pct)
    if "AFTERBURNER" in t:
        return ("AFTERBURNER", 100)
    if "MILITARY" in t:
        return ("MILITARY", 100)
    return (thrust_choice, 100)
def build_engine_table(thrust_sel: str, derate_pct: int) -> pd.DataFrame:
    """
    Builds a simple N1 / FF(pph per engine) guidance table based on the
    *resolved* thrust selection (e.g., "MILITARY", "AFTERBURNER",
    "DERATE (92%)"). Does NOT read global `thrust` — uses `thrust_sel`.
    """
    def rows(n1_to, ff_to, n1_ic, ff_ic, n1_cl, ff_cl):
        return [
            {"Phase": "Takeoff",       "Target N1 (%)": int(round(n1_to)), "FF (pph/engine)": int(round(ff_to))},
            {"Phase": "Initial Climb", "Target N1 (%)": int(round(n1_ic)), "FF (pph/engine)": int(round(ff_ic))},
            {"Phase": "Climb Segment", "Target N1 (%)": int(round(n1_cl)), "FF (pph/engine)": int(round(ff_cl))},
        ]

    label = (thrust_sel or "").upper()

    if "AFTERBURNER" in label:
        data = rows(102, 19000, 98, 10000, 95, 7500)
    elif "DERATE" in label:
        # Use provided derate_pct if available; fallback to 95
        n1_to = max(85, min(100, int(derate_pct or 95)))
        n1_ic = max(85, n1_to - 2)
        n1_cl = max(85, n1_ic - 2)
        scale = (n1_to / 96.0)
        data = rows(n1_to, 7000*scale, n1_ic, 6500*scale, n1_cl, 6000*scale)
    elif "MILITARY" in label:
        data = rows(96, 7000, 95, 6500, 93, 6000)
    else:
        # Generic auto fallback ~ MIL-ish
        data = rows(95, 6500, 94, 6250, 92, 5750)

    return pd.DataFrame(data)


perf_takeoff_ref = getattr(core, "perf_compute_takeoff", None)

gw_lb         = float(locals().get("wb", {}).get("gw_tow_lb", DEFAULT_GTOW))
field_elev_ft = float(locals().get("elev", 0.0))
oat_c         = float(locals().get("field_temp", 15.0))
runway_slope  = 0.0
stores_list   = get_stores_drag_list()
mode_flag     = "DCS"

# Runway availability
tora_ft = _s_int(locals().get("tora", 0))
asda_ft = tora_ft  # ASDA fallback to TORA if unknown

# Apply wind policy to the headwind component BEFORE perf calls
use_50_150 = bool(wind_policy_choice.startswith("50%"))
hw_raw = float(locals().get("hw", 0.0) or 0.0)
hw_eff = apply_wind_policy(hw_raw, use_50_150)

# Scenario context for evaluations
scenario_ctx = dict(
    gw_lb=gw_lb,
    field_elev_ft=field_elev_ft,
    oat_c=oat_c,
    runway_slope=runway_slope,
    stores=stores_list,
    mode=mode_flag,
    available_tora_ft=float(tora_ft),
    available_asda_ft=float(asda_ft),
    req_grad_ft_nm=float(st.session_state.get("req_climb_grad_ft_nm", 200)),
)

# ---------- Auto-Select (DERATE→MIL, then AB feasibility) ----------
auto_mode = (flaps == "Auto-Select") or (thrust == "Auto-Select")
selection = None

# NEW (Patch 5b): capture all candidates we evaluate (for a read-only debug drawer)
candidate_logs: List[Dict[str, Any]] = []

def _ladder_derate_labels():
    # Discrete derate points; labels for UI, MIL for engine calls
    pts = [85, 88, 90, 92, 95, 97, 100]
    return [f"DERATE ({p}%)" if p < 100 else "MILITARY" for p in pts], pts

def _choose_best(cands: list[dict]):
    # Tie-breakers: min thrust → min flap → max margin → closest to balanced
    def thrust_rank(c):
        lbl = c["thrust_label"]
        if lbl.startswith("DERATE"):
            import re as _re
            m = _re.search(r"(\d+)%", lbl)
            pct = int(m.group(1)) if m else 100
            return (0, pct)  # lower pct better
        if lbl.startswith("MILITARY"):
            return (1, 100)
        if lbl.startswith("AFTERBURNER"):
            return (2, 200)
        return (9, 999)

    def flap_rank(c):
        order = {"UP": 0, "MANEUVER": 1, "FULL": 2}
        return order.get(c["flaps"], 9)

    def margin_rank(c):
        m = c["margins"]
        return -min(m.get("tora_margin_ft", -1e9), m.get("asda_margin_ft", -1e9))  # larger is better

    def balance_rank(c):
        return c["diff_ratio"]  # smaller is better

    return sorted(cands, key=lambda c: (thrust_rank(c), flap_rank(c), margin_rank(c), balance_rank(c)))[0] if cands else None

if auto_mode:
    # 1) Try DERATE→MIL ladder with flaps UP→MANEUVER→FULL
    derate_labels, derate_pcts = _ladder_derate_labels()
    flaps_order = ["UP", "MANEUVER", "FULL"]
    feasible = []

    for lbl, pct in zip(derate_labels, derate_pcts):
        for flp in flaps_order:
            cand = _evaluate_candidate(
                flaps_label=flp,
                thrust_label=lbl,
                thrust_mode="MIL",           # engine sees MIL; derate is a label decision
                hw_eff_kts=hw_eff,
                ctx=scenario_ctx
            )
            # NEW (Patch 5b): log every candidate (even failures)
            candidate_logs.append({
                "Flaps": cand["flaps"],
                "Thrust": cand["thrust_label"],
                "ThrustMode": cand["thrust_mode"],
                "TODR_ft": int(round(cand["todr_ft"])) if cand["todr_ft"] else None,
                "ASDR_ft": int(round(cand["asdr_ft"])) if cand["asdr_ft"] else None,
                "TORA_margin_ft": int(round(cand["margins"]["tora_margin_ft"])) if cand["margins"].get("tora_margin_ft") is not None else None,
                "ASDA_margin_ft": int(round(cand["margins"]["asda_margin_ft"])) if cand["margins"].get("asda_margin_ft") is not None else None,
                "AEO_grad_ft_nm": None if cand["aeo_grad_ft_nm"] is None else float(cand["aeo_grad_ft_nm"]),
                "Pass_Runway": bool(cand["pass_runway"]),
                "Pass_Climb": bool(cand["pass_climb"]),
                "Dispatchable": bool(cand["dispatchable"]),
            })
            if cand["dispatchable"]:
                feasible.append(cand)

    best_mil = _choose_best(feasible)

    if best_mil:
        selection = dict(best_mil, ab_required=False)
    else:
        # 2) Escalate to AB: evaluate feasibility but DO NOT authorize in Auto-Select
        feasible_ab = []
        for flp in flaps_order:
            cand = _evaluate_candidate(
                flaps_label=flp,
                thrust_label="AFTERBURNER (required)",
                thrust_mode="MAX",
                hw_eff_kts=hw_eff,
                ctx=scenario_ctx
            )
            # NEW (Patch 5b): also log AB candidates
            candidate_logs.append({
                "Flaps": cand["flaps"],
                "Thrust": cand["thrust_label"],
                "ThrustMode": cand["thrust_mode"],
                "TODR_ft": int(round(cand["todr_ft"])) if cand["todr_ft"] else None,
                "ASDR_ft": int(round(cand["asdr_ft"])) if cand["asdr_ft"] else None,
                "TORA_margin_ft": int(round(cand["margins"]["tora_margin_ft"])) if cand["margins"].get("tora_margin_ft") is not None else None,
                "ASDA_margin_ft": int(round(cand["margins"]["asda_margin_ft"])) if cand["margins"].get("asda_margin_ft") is not None else None,
                "AEO_grad_ft_nm": None if cand["aeo_grad_ft_nm"] is None else float(cand["aeo_grad_ft_nm"]),
                "Pass_Runway": bool(cand["pass_runway"]),
                "Pass_Climb": bool(cand["pass_climb"]),
                "Dispatchable": bool(cand["dispatchable"]),
            })
            if cand["dispatchable"]:
                feasible_ab.append(cand)

        best_ab = _choose_best(feasible_ab)
        if best_ab:
            selection = dict(best_ab, ab_required=True)
        else:
            selection = None  # not dispatchable even with AB




# ---------- Resolve UI display based on auto vs manual ----------
if auto_mode and selection:
    flap_display = selection["flaps"]
    thrust_display = selection["thrust_label"]
    t_res = selection.get("t_res") or {}
    balanced_badge = (
        "Balanced" if selection.get("diff_ratio", 1.0) <= 0.05
        else ("ASDR-limited" if selection.get("asdr_ft", 0) > selection.get("todr_ft", 0) else "TODR-limited")
    )
    governing = balanced_badge
else:
    # Manual (or no selection available) → compute single-point perf per current controls
    flap_display = flaps if flaps != "Auto-Select" else "FULL"
    thrust_display = thrust
    balanced_badge = ""
    governing = ""
    t_res = {}
    if callable(perf_takeoff_ref):
        try:
            tm = "MAX" if thrust_display == "AFTERBURNER" else "MIL"
            hw_eff_manual = apply_wind_policy(hw_raw, use_50_150)
            t_res = cached_perf_takeoff(
                gw_lb=gw_lb,
                field_elev_ft=field_elev_ft,
                oat_c=oat_c,
                headwind_kts=hw_eff_manual,
                runway_slope=runway_slope,
                thrust_mode=tm,
                mode=mode_flag,
                config=("TO_FLAPS" if flap_display in ("MANEUVER","FULL") else "CLEAN"),
                sweep_deg=20.0,
                stores=tuple(stores_list),
            )
            # Manual mode: gently override displayed V-speeds from table if available
            try:
                fl_for_tbl = flap_display if flap_display in ("UP","MANEUVER","FULL") else "UP"
                tm_for_tbl = "MAX" if tm == "MAX" else "MIL"
                vs_tbl = _vs_lookup_from_perf_table(gw_lb, fl_for_tbl, tm_for_tbl)
                if vs_tbl:
                    if not pd.isna(vs_tbl.get("Vr_kts", float("nan"))):
                        t_res["VR_kts"] = float(vs_tbl["Vr_kts"])
                    if not pd.isna(vs_tbl.get("V2_kts", float("nan"))):
                        t_res["V2_kts"] = float(vs_tbl["V2_kts"])
                    if not pd.isna(vs_tbl.get("Vs_kts", float("nan"))):
                        t_res["Vs_kts"] = float(vs_tbl["Vs_kts"])
            except Exception:
                # Safe fallback: don’t block app if lookup fails
                pass

            
        except Exception as e:
            st.warning(f"Perf engine error: {e}")
# If Auto-Select ended up with AB-only feasibility, show the policy banner
if auto_mode and selection and selection.get("ab_required", False):
    st.error("**TAKEOFF — AB REQUIRED — NOT AUTHORIZED**\n\nAuto-Select found no dispatchable MIL/DERATE option. "
             "To proceed, explicitly set **Thrust = AFTERBURNER** (manual) or adjust runway / weight / wind.",
             icon="🚫")

if auto_mode and selection:
    flap_display = selection["flaps"]
    thrust_display = selection["thrust_label"]
    t_res = selection["t_res"] or {}
    balanced_badge = (
        "Balanced" if selection["diff_ratio"] <= 0.05
        else ("ASDR-limited" if selection["asdr_ft"] > selection["todr_ft"] else "TODR-limited")
    )
    governing = balanced_badge
        # AB escalation UX: if Auto-Select required AB to be dispatchable, warn loudly
    if selection.get("ab_required", False):
        st.error("**TAKEOFF — AB REQUIRED — NOT AUTHORIZED**\n\n"
                 "Auto-Select determined that no MIL/DERATE configuration is dispatchable. "
                 "AFTERBURNER would be required to meet runway/climb gates. "
                 "Switch Thrust to **AFTERBURNER** manually to authorize and recompute.",
                 icon="🚫")

else:
    # Manual (or no selection available)
    flap_display = flaps if flaps != "Auto-Select" else "FULL"
    thrust_display = thrust
    balanced_badge = ""
    governing = ""
    t_res = {}

    if (flaps == "Auto-Select") or (thrust == "Auto-Select"):
        # Do NOT compute a manual point when user is still in Auto-Select and no candidate exists.
        # Show guidance to either authorize AB or adjust config.
        st.info("Auto-Select did not produce a dispatchable configuration. "
                "Authorize AFTERBURNER (manual Thrust = AFTERBURNER) or adjust weight / runway / wind.",
                icon="ℹ️")
    else:
        # Fully manual → compute single-point perf per current controls
        if callable(perf_takeoff_ref):
            try:
                tm = "MAX" if thrust_display == "AFTERBURNER" else "MIL"
                hw_eff_manual = apply_wind_policy(hw_raw, use_50_150)
                t_res = cached_perf_takeoff(
                    gw_lb=gw_lb,
                    field_elev_ft=field_elev_ft,
                    oat_c=oat_c,
                    headwind_kts=hw_eff_manual,
                    runway_slope=runway_slope,
                    thrust_mode=tm,
                    mode=mode_flag,
                    config=("TO_FLAPS" if flap_display in ("MANEUVER","FULL") else "CLEAN"),
                    sweep_deg=20.0,
                    stores=tuple(stores_list),
                )
            except Exception as e:
                st.warning(f"Perf engine error: {e}")



col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("V-Speeds")
    if t_res:
        # Prefer reference CSV when we are in non-AB operation
        # (CSV is MIL-based; use it for DERATE...MIL. For AB, show model.)
        use_csv = False
        if auto_mode and selection:
            use_csv = not bool(selection.get("ab_required", False))
        else:
            # manual thrust: use CSV if NOT AB
            use_csv = (str(thrust_display).upper().startswith("MILITARY") or str(thrust_display).upper().startswith("DERATE"))

        ref = get_reference_vspeeds_from_csv(perf, gw_lb, flap_display) if use_csv else None

        if ref:
            v1 = ref["V1"]; vr = ref["Vr"]; v2 = ref["V2"]
        else:
            # model fallback (note: current core returns VR as both V1/Vr placeholder)
            v1 = float(t_res.get("VR_kts", 0.0) or 0.0)
            vr = float(t_res.get("VR_kts", 0.0) or 0.0)
            v2 = float(t_res.get("V2_kts", 0.0) or 0.0)

        vfs = max(v2 * 1.1, float(t_res.get("VLOF_kts", 0.0) or 0.0) * 1.15)

        st.metric("V1 (kt)", f"{v1:.0f}" if v1 > 0 else "—")
        st.metric("Vr (kt)", f"{vr:.0f}" if vr > 0 else "—")
        st.metric("V2 (kt)", f"{v2:.0f}" if v2 > 0 else "—")
        st.metric("Vfs (kt)", f"{vfs:.0f}" if vfs > 0 else "—")
    else:
        st.metric("V1 (kt)", "—"); st.metric("Vr (kt)", "—"); st.metric("V2 (kt)", "—"); st.metric("Vfs (kt)", "—")

   
    st.caption("V1 shown as **Vr** (placeholder) until ASDR/V1 modeling is wired. "
               "All V-speeds currently come from the performance table and may need calibration against NATOPS/DCS.")

with col2:
    st.subheader("Configuration")
    st.metric("Flaps", flap_display)
    st.metric("Thrust", (thrust_display or "").replace("(Manual)",""))
    st.metric("Stabilizer Trim", f"{wb.get('stab_trim_units', 0.0):+0.1f} units")
    st.caption("N1% / FF(pph/engine) — guidance (table)")

# Derive the effective derate % from the RESOLVED thrust_display, not the radio
import re as _re

def _parse_derate_from_label(lbl: str) -> int:
    if not isinstance(lbl, str) or not lbl:
        return 100
    u = lbl.upper()
    if u.startswith("DERATE"):
        m = _re.search(r"(\d+)\s*%", u)
        if m:
            return max(85, min(100, int(m.group(1))))
        return 95
    if u.startswith("MILITARY"):
        return 100
    # For AB, the table uses its own hard-coded values; the % isn’t used there.
    if u.startswith("AFTERBURNER"):
        return 102
    return 100

# When manual DERATE slider is active, prefer that value
if thrust == "DERATE (Manual)":
    # Always honor the slider in Manual mode
    _derate_effective = int(locals().get("derate", 95))
    thrust_display = core.resolve_thrust_display("MIL", _derate_effective)
else:
    # Auto-Select or MIL/AB paths
    _derate_effective = _parse_derate_from_label(thrust_display or thrust or "MILITARY")
    thrust_display = core.resolve_thrust_display(thrust_display or thrust, _derate_effective)

engine_df = build_engine_table(thrust_display or thrust or "MILITARY", int(_derate_effective))
st.dataframe(engine_df, hide_index=True, use_container_width=True)

st.caption(
    "<span style='color:#664200;background:#FFF4CC;border:1px solid #FFD666;"
    "padding:2px 6px;border-radius:6px;font-size:12px;'>Calibration needed</span>",
    unsafe_allow_html=True,
)
with col3:
    st.subheader("Runway Distances")
    if t_res:
        st.metric("Ground roll (ft)",       f"{t_res['GroundRoll_ft']:.0f}")
        st.metric("Dist to 35 ft (ft)",     f"{t_res['DistanceTo35ft_ft']:.0f}")
        st.metric("Available (TORA)",       f"{tora_ft:,} ft")
        st.metric("Required (≤TORA)",       f"{int(round(t_res['DistanceTo35ft_ft'])):,} ft")


    else:
        st.metric("Ground roll (ft)", "—")
        st.metric("Dist to 35 ft (ft)", "—")
        st.metric("Available (TORA)", f"{tora_ft:,} ft")
        st.metric("Required (≤TORA)", "—")

with col4:
    st.subheader("Dispatchability")

    # Prefer Auto-Select candidate if present; else use current single-point t_res
    candidate = None
    if auto_mode and selection:
        candidate = selection
    elif t_res:
        # Build a minimal candidate from current point for display
        candidate = {
            "todr_ft": float(t_res.get("DistanceTo35ft_ft", 0.0) or 0.0),
            "asdr_ft": float(t_res.get("ASDR_ft", 0.0) or 0.0) or float(t_res.get("GroundRoll_ft", 0.0) or 0.0) * 1.15,
            "aeo_grad_ft_nm": None,
            "margins": {
                "tora_margin_ft": float(tora_ft) - float(t_res.get("DistanceTo35ft_ft", 0.0) or 0.0),
                "asda_margin_ft": float(asda_ft) - (float(t_res.get("ASDR_ft", 0.0) or 0.0) or float(t_res.get("GroundRoll_ft", 0.0) or 0.0) * 1.15),
            },
        }

    if candidate:
        # Gates
        todr = float(candidate["todr_ft"])
        asdr = float(candidate["asdr_ft"])
        tora = float(tora_ft)
        asda = float(asda_ft)
        grad = candidate.get("aeo_grad_ft_nm", None)
        req_grad = float(st.session_state.get("req_climb_grad_ft_nm", 200))

        pass_runway = (todr <= tora) and (asdr <= asda)
        pass_climb = (grad is None) or (grad >= req_grad)
        dispatchable = pass_runway and pass_climb

        if dispatchable:
            st.success("Dispatchable")
        else:
            st.error("NOT Dispatchable")

        # Limiter & margins
        lim = "Balanced"
        if asdr > asda and todr <= tora:
            lim = "ASDR-limited"
        elif todr > tora and asdr <= asda:
            lim = "TODR-limited"
        elif todr > tora and asdr > asda:
            lim = "TODR & ASDR limited"
        st.caption(f"Limiting: {lim}")

        tm = candidate.get("margins", {})
        st.metric("TORA Margin", f"{int(round(tm.get('tora_margin_ft', 0))):,} ft")
        st.metric("ASDA Margin", f"{int(round(tm.get('asda_margin_ft', 0))):,} ft")
        if grad is not None:
            st.metric("Expected Climb Gradient (AEO)", f"{grad:.0f} ft/NM")
        else:
            st.metric("Expected Climb Gradient (AEO)", "— ft/NM (not computed)")
    else:
        st.info("No dispatchable candidate. Adjust thrust/flaps, weight, or runway.", icon="ℹ️")
        st.metric("Expected Climb Gradient (AEO)", "— ft/NM (not computed)")


st.divider()
st.subheader("DCS Expected Performance (calculated)")
if t_res:
    p1, p2 = st.columns(2)
    p1.metric("Distance to reach Vr (ft)", f"{int(round(t_res['GroundRoll_ft']*0.67)):,}")
    p2.metric("Distance to Liftoff / 35 ft (ft)", f"{int(round(t_res['DistanceTo35ft_ft'])):,}")
else:
    st.info("Model unavailable — no values shown (placeholder).")
st.divider()
# === Auto-Select Debug (Patch 5b) — read-only, shows why candidates fail ===
if auto_mode:
    with st.expander("Auto-Select Debug (candidates)", expanded=bool(show_debug)):
        if candidate_logs:
            df_logs = pd.DataFrame(candidate_logs)
            # summary row
            total = len(df_logs)
            n_runway_pass = int(df_logs["Pass_Runway"].sum()) if "Pass_Runway" in df_logs else 0
            n_climb_pass  = int(df_logs["Pass_Climb"].sum()) if "Pass_Climb" in df_logs else 0
            n_dispatch    = int(df_logs["Dispatchable"].sum()) if "Dispatchable" in df_logs else 0

            st.caption(
                f"Tested {total} candidate(s) • Runway pass: {n_runway_pass} • "
                f"Climb pass (≥ req {int(scenario_ctx['req_grad_ft_nm'])} ft/nm): {n_climb_pass} • "
                f"Dispatchable: {n_dispatch}"
            )

            # Friendly column labels
            disp = df_logs.rename(columns={
                "Flaps": "Flaps",
                "Thrust": "Thrust (label)",
                "ThrustMode": "Power",
                "TODR_ft": "TODR (ft)",
                "ASDR_ft": "ASDR (ft)",
                "TORA_margin_ft": "TORA Margin (ft)",
                "ASDA_margin_ft": "ASDA Margin (ft)",
                "AEO_grad_ft_nm": "AEO Grad (ft/NM)",
                "Pass_Runway": "Pass Runway",
                "Pass_Climb": "Pass Climb",
                "Dispatchable": "Dispatchable",
            })

            # Sort by Dispatchable desc, then margins
            sort_cols = ["Dispatchable", "TORA Margin (ft)", "ASDA Margin (ft)"]
            disp = disp.sort_values(by=sort_cols, ascending=[False, False, False], na_position="last")

            st.dataframe(disp, hide_index=True, use_container_width=True)

            st.caption(
                "Tip: If **Pass Climb** is False while **Pass Runway** is True, your Required climb gradient gate "
                f"({int(scenario_ctx['req_grad_ft_nm'])} ft/NM) is the reason Auto-Select rejects otherwise valid runway candidates. "
                "Try a lower requirement or confirm the climb model policy."
            )
        else:
            st.info("No Auto-Select candidates were evaluated yet (manual mode or missing inputs).")

# =========================
# 5a) Intersection Margins (vectorized, cached, scoped)
# =========================
with st.expander("5) Intersection Margins", expanded=False):
    start_tm = time.time()
    if not t_res:
        st.info("Takeoff results not available yet — set runway and environment first.")
    else:
        required_ft = int(round(float(t_res.get("DistanceTo35ft_ft", 0)) or 0))
        if required_ft <= 0:
            st.info("Required distance unavailable — compute takeoff first.")
        else:
            st.caption(f"Required distance to 35 ft: **{required_ft:,} ft**")

            # Filter once (airport + map); then extract all runway_end options for this runway ID group
            apt_name = locals().get("apt")
            map_name = locals().get("map_sel")
            sel_end  = str(locals().get("rwy_end", ""))

            if not apt_name or not map_name or not sel_end:
                st.info("Pick an Airport / Map / Runway End in Section 2.")
            else:
                # Subset by airport + map, then keep same "runway" if present, else same number prefix of runway_end
                sub_all = airports[(airports["airport_name"] == apt_name) & (airports["map"] == map_name)].copy()

                # Heuristic: if "runway" column exists, use it; else use the numeric prefix of runway_end to group
                if "runway" in sub_all.columns and pd.api.types.is_string_dtype(sub_all["runway"]):
                    # Select all intersections that share the same runway id as the chosen end (if discoverable)
                    chosen_row = sub_all[sub_all["runway_end"].astype(str) == sel_end]
                    if not chosen_row.empty:
                        runway_id = chosen_row["runway"].iloc[0]
                        sub_one = sub_all[sub_all["runway"] == runway_id].copy()
                    else:
                        sub_one = sub_all.copy()
                else:
                    # Fallback: group by numeric prefix e.g., "31" from "31L @ A3"
                    import re as _re
                    prefix = "".join(_re.findall(r"^\d{2}", sel_end)) or sel_end
                    sub_one = sub_all[sub_all["runway_end"].astype(str).str.contains(prefix, na=False)].copy()

                # Vectorized available length (prefer TORA, fallback length_ft)
                if "tora_ft" in sub_one.columns:
                    sub_one["available_length_ft"] = pd.to_numeric(sub_one["tora_ft"], errors="coerce")
                else:
                    sub_one["available_length_ft"] = pd.to_numeric(sub_one.get("length_ft", 0), errors="coerce")

                # Compute margin (vectorized)
                sub_one["required_ft"] = required_ft
                sub_one["margin_ft"] = sub_one["available_length_ft"] - required_ft

                # Clean table for display
                out_cols = []
                if "runway_end" in sub_one.columns: out_cols.append("runway_end")
                out_cols += ["available_length_ft", "required_ft", "margin_ft"]
                view = sub_one[out_cols].dropna(subset=["available_length_ft"]).sort_values("available_length_ft", ascending=False)

                st.dataframe(
                    view.rename(columns={
                        "runway_end": "Intersection",
                        "available_length_ft": "Available Length (ft)",
                        "required_ft": "Required (ft)",
                        "margin_ft": "Margin (ft)"
                    }),
                    hide_index=True,
                    use_container_width=True
                )

                # Optional quick chart (Altair) — drawn only if data present
                try:
                    import altair as alt
                    chart = alt.Chart(view).mark_bar().encode(
                        x=alt.X(( "runway_end:N" if "runway_end" in view.columns else "available_length_ft:N"),
                                title="Intersection"),
                        y=alt.Y("margin_ft:Q", title="Margin (ft)"),
                        tooltip=[( "runway_end" if "runway_end" in view.columns else "available_length_ft"),
                                 "available_length_ft", "required_ft", "margin_ft"]
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    # If Altair not available, skip plotting (table is enough)
                    pass

                st.caption(f"Intersection Margins computed in {(time.time()-start_tm):.2f}s")

# =========================
# Section 6 — Climb Profile
# =========================
with st.expander("6) Climb Profile", expanded=True):
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            """
            <div style="border:1px solid rgba(255,255,255,0.15); padding:16px; border-radius:12px; background:rgba(255,255,255,0.03);">
              <h3 style="margin-top:0">Climb Schedule — Most efficient</h3>
              <ul style="line-height:1.6; font-size:1.05rem;">
                <li><strong>1,000 ft AGL</strong>: RPM/FF — / —, Target: V2 + 15 kt</li>
                <li><strong>Up to 10,000 ft</strong>: RPM/FF — / —, Target IAS — kt</li>
                <li><strong>10k → Mach</strong>: RPM/FF — / —, Target IAS — kt</li>
                <li><strong>Mach → Cruise</strong>: RPM/FF — / —, Target Mach —</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with colB:
        st.markdown(
            """
            <div style="border:1px solid rgba(255,255,255,0.15); padding:16px; border-radius:12px; background:rgba(255,255,255,0.03);">
              <h3 style="margin-top:0">Climb Schedule — Minimum time</h3>
              <ul style="line-height:1.6; font-size:1.05rem;">
                <li><strong>1,000 ft AGL</strong>: RPM/FF — / —, Target: V2 + 25 kt</li>
                <li><strong>Up to 10,000 ft</strong>: RPM/FF — / —, Target IAS — kt</li>
                <li><strong>10k → Mach</strong>: RPM/FF — / —, Target IAS — kt</li>
                <li><strong>Mach → Cruise</strong>: RPM/FF — / —, Target Mach —</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        cruise_alt = st.number_input("Cruise Altitude (MSL ft)", value=28000, step=1000, min_value=0, format="%d")
    with c2:
        climb_profile = st.selectbox("Climb Profile", ["Most efficient climb", "Minimum time to altitude"], index=0)
    with c3:
        ignore_reg = st.checkbox("Ignore regulatory speed restrictions (≤250 KIAS <10k)")

perf_climb_ref = getattr(core, "perf_compute_climb", None)


gw_lb = float(locals().get("wb", {}).get("gw_tow_lb", DEFAULT_GTOW))
alt0  = float(locals().get("elev", 0.0))
alt1  = float(locals().get("cruise_alt", 28000))
sched = "NAVY" if (locals().get("climb_profile","Most efficient climb").startswith("Most")) else "DISPATCH"

cres_10k = cres_full = None
if callable(perf_climb_ref):
    try:
        cres_10k = cached_perf_climb(
            gw_lb=gw_lb, alt_start_ft=alt0, alt_end_ft=max(10000, alt0+1),
            oat_dev_c=0.0, schedule=sched, mode="DCS", power="MIL",
            sweep_deg=20.0, config="CLEAN",
        )
        cres_full = cached_perf_climb(
            gw_lb=gw_lb, alt_start_ft=alt0, alt_end_ft=alt1,
            oat_dev_c=0.0, schedule=sched, mode="DCS", power="MIL",
            sweep_deg=20.0, config="CLEAN",
        )
    except Exception as e:
        st.warning(f"Climb model error: {e}")

r1, r2, r3, r4, r5, r6 = st.columns(6)
if cres_10k:
    r1.metric("Time to 10,000 ft", f"{cres_10k['Time_s']/60.0:02.0f}:{cres_10k['Time_s']%60:02.0f}")
else:
    r1.metric("Time to 10,000 ft", "—:—")

if cres_full:
    r2.metric("Time to Cruise Altitude", f"{cres_full['Time_s']/60.0:02.0f}:{cres_full['Time_s']%60:02.0f}")
    r3.metric("Fuel to Top of Climb", f"{cres_full['Fuel_lb']:.0f} lb")
    r4.metric("TOC Distance", f"{cres_full['Distance_nm']:.1f} NM")
else:
    r2.metric("Time to Cruise Altitude", "—:—")
    r3.metric("Fuel to Top of Climb", "—")
    r4.metric("TOC Distance", "—")

r5.metric("Time to TO + 100 NM", "—:—")
r6.metric("Fuel to TO + 100 NM", "—")

if cres_full:
    import numpy as np
    tmin = cres_full['Time_s']/60.0
    xs = np.linspace(0.0, max(1.0,tmin), 6)
    ys = np.linspace(alt0, alt1, 6)
    climb_overlay = pd.DataFrame({"Time_min": xs, "Alt_ft": ys}).set_index("Time_min")
    st.line_chart(climb_overlay)
else:
    st.caption("No climb overlay (perf model unavailable).")

st.markdown("---")

# =========================
# Section 7 — Landing Setup / Results (unchanged UI, cached compute)
# =========================
with st.expander("7) Landing Setup", expanded=True):
    def pick_destination(slot_idx: int, fixed_map: Optional[str] = None,
                         default_airport: Optional[str] = None, default_end: Optional[str] = None) -> Dict[str, Any]:
        if fixed_map:
            sel_map = fixed_map
            st.caption(f"[{slot_idx}] Theatre: **{sel_map}** (from departure)")
        else:
            maps = sorted(airports["map"].dropna().unique().tolist())
            map_key = f"ldg_map_{slot_idx}"
            sel_map = st.selectbox(f"[{slot_idx}] Map", maps, key=map_key)

        sub = airports[airports["map"] == sel_map]
        apts = sorted(sub["airport_name"].dropna().unique().tolist())
        apt_key = f"ldg_airport_{slot_idx}"
        apt_idx = apts.index(default_airport) if (default_airport in apts) else (0 if apts else 0)
        sel_apt = st.selectbox(f"[{slot_idx}] Airport", apts, index=apt_idx, key=apt_key)

        rows = sub[sub["airport_name"] == sel_apt]
        ends = sorted(rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) or ["Full Length"]
        end_key = f"ldg_end_{slot_idx}"
        end_idx = ends.index(default_end) if (default_end in ends) else 0
        sel_end = st.selectbox(f"[{slot_idx}] Runway End", ends, index=end_idx, key=end_key)

        if "runway_end" in rows.columns:
            mask = rows["runway_end"].astype(str) == str(sel_end)
            rows_masked = rows.loc[mask]
        else:
            rows_masked = rows

        tora_candidate = _series_max(rows_masked, "tora_ft")
        if tora_candidate is None or pd.isna(tora_candidate):
            tora_candidate = _series_max(rows, "length_ft")
        tora_ft = _s_int(tora_candidate, 0)

        lda_candidate = _series_max(rows_masked, "lda_ft")
        if lda_candidate is None or pd.isna(lda_candidate):
            lda_candidate = _series_max(rows, "length_ft")
        if lda_candidate is None or pd.isna(lda_candidate):
            lda_candidate = tora_ft
        lda_ft = _s_int(lda_candidate, 0)

        m1, m2 = st.columns(2)
        m1.metric("Available Takeoff (TORA)", f"{tora_ft:,} ft")
        m2.metric("Available Landing (LDA)", f"{lda_ft:,} ft")
        if "lda_ft" not in rows.columns:
            st.caption("ℹ️ LDA not in database for this runway; using runway length (or TORA) as proxy.")

        return {"map": sel_map, "airport": sel_apt, "end": sel_end, "tora_ft": tora_ft, "lda_ft": lda_ft}

    dep_map = locals().get("map_sel")
    dep_airport = locals().get("apt")
    dep_end = locals().get("rwy_end")

    st.markdown("**Destination 1 (seeded from departure selection)**")
    dest1 = pick_destination(1, fixed_map=dep_map, default_airport=dep_airport, default_end=dep_end)

    st.session_state.setdefault("alt_slots", [])
    add_alt = st.button("➕ Add Alternate")
    if add_alt and len(st.session_state["alt_slots"]) < 3:
        next_slot = 2
        while next_slot in st.session_state["alt_slots"] or next_slot == 1:
            next_slot += 1
        st.session_state["alt_slots"].append(next_slot)
        st.rerun()

    dests: List[Dict[str, Any]] = [dest1]

    for slot in sorted(st.session_state["alt_slots"]):
        st.markdown(f"**Alternate {slot - 1}**")
        alt_cols = st.columns([6, 1])
        with alt_cols[0]:
            dests.append(pick_destination(slot, fixed_map=dep_map, default_airport=None, default_end=None))
        with alt_cols[1]:
            if st.button("Remove", key=f"rm_alt_{slot}"):
                st.session_state["alt_slots"] = [s for s in st.session_state["alt_slots"] if s != slot]
                st.rerun()

    cond = st.radio("Runway condition (applies to all candidates below)", ["DRY", "WET"], horizontal=True, key="ldg_cond")
landing_scenario = st.radio(
    "Landing Scenario",
    ["3,000 lb fuel — stores retained", "3,000 lb fuel — no weapons", "Custom"],
    index=0, horizontal=False
)

def factored_distance(unfactored_ft: int, condition: str) -> int:
    factor = 1.67 if condition == "DRY" else 1.92
    return int(round(unfactored_ft * factor))

def compute_landing_weight() -> tuple[int, bool]:
    scenario_name = st.session_state.get("landing_scenario") or locals().get("landing_scenario")
    empty = _s_int(st.session_state.get("empty_weight_lb", None) or 43735, 43735)
    stores_wt = _s_int(st.session_state.get("stores_weight_lb", None) or 2500, 2500)
    weapons_only_wt = _s_int(st.session_state.get("weapons_weight_lb", None) or 1800, 1800)
    crew_misc = 400
    if isinstance(scenario_name, str):
        scenario = scenario_name.lower()
        if scenario.startswith("3,000") and "no weapons" in scenario:
            fuel = 3000
            lw = empty + max(0, stores_wt - weapons_only_wt) + fuel + crew_misc
            is_ph = ("weapons_weight_lb" not in st.session_state)
            return int(lw), bool(is_ph)
        elif scenario.startswith("3,000"):
            fuel = 3000
            lw = empty + stores_wt + fuel + crew_misc
            is_ph = ("stores_weight_lb" not in st.session_state)
            return int(lw), bool(is_ph)
        lw_custom = _s_int(st.session_state.get("gw_ldg_plan", None) or locals().get("gw_ldg_plan", None) or DEFAULT_LDW, DEFAULT_LDW)
        return int(lw_custom), ("gw_ldg_plan" not in st.session_state)
    lw_fallback = _s_int(locals().get("gw_ldg_plan", DEFAULT_LDW), DEFAULT_LDW)
    return int(lw_fallback), True

st.header("Landing Results")

perf_landing_ref = getattr(core, "perf_compute_landing", None)

if 'dests' in locals() and dests:
    gw_ldg_plan = _s_int(locals().get("gw_ldg_plan", DEFAULT_LDW), DEFAULT_LDW)

    for i, d in enumerate(dests, start=1):
        st.subheader(f"Runway {i}: {d['airport']} ({d['map']}) — RWY {d['end']}")
        lda_ft = _s_int(d.get("lda_ft", 0), 0)

        lres = None
        if callable(perf_landing_ref):
            try:
                lres = cached_perf_landing(
                    gw_lb=float(gw_ldg_plan),
                    field_elev_ft=0.0,
                    oat_c=float(locals().get("field_temp", 15.0)),
                    headwind_kts=float(locals().get("hw", 0.0) or 0.0),
                    mode="DCS",
                    config="LDG_FLAPS",
                    sweep_deg=20.0,
                )
            except Exception as e:
                st.warning(f"Landing model error: {e}")

        gw_ldg_plan, lw_placeholder = compute_landing_weight()

        if lres:
            unfact = int(round(lres["Total_ft"]))
            fact   = factored_distance(unfact, st.session_state.get("ldg_cond", "DRY"))

            w1, w2 = st.columns([1,1])
            w1.metric("Planned Landing Weight", f"{gw_ldg_plan:,} lb" + (" (placeholder)" if lw_placeholder else ""))
            w2.metric("Vref (kt)", f"{lres['Vref_kts']:.0f}")

            c1, c2, c3 = st.columns([1,1,1])
            c1.metric("Unfactored Landing Distance", f"{unfact:,} ft")
            c2.metric("Factored Landing Distance",   f"{fact:,} ft")
            c3.metric("LDA Available",               f"{lda_ft:,} ft")

            st.caption(f"Airborne: {lres['Airborne_ft']:.0f} ft • Ground Roll: {lres['GroundRoll_ft']:.0f} ft")
            st.divider()
        else:
            st.metric("Planned Landing Weight", f"{gw_ldg_plan:,} lb" + (" (placeholder)" if lw_placeholder else ""))
            st.info("Perf model not available for landing.")
            st.divider()
else:
    st.info("Set at least one destination in the Landing Setup above to see landing results.")

st.markdown("---")

# =========================
# Scenario JSON (debug)
# =========================
if 'show_debug' in locals() and show_debug:
    scenario = {
        "aircraft": locals().get("ac"),
        "runway": {
            "map": locals().get("map_sel"),
            "airport": locals().get("apt"),
            "end": locals().get("rwy_end"),
            "tora_ft": locals().get("tora"),
            "elev_ft": locals().get("elev"),
            "heading_deg": locals().get("hdg"),
            "manual_override": bool(st.session_state.get("rw_manual")),
        },
        "environment": {
            "temp_sl_c": (locals().get("temp_sl")),
            "qnh_inhg": (locals().get("qnh_inhg")),
            "wind": (locals().get("w") if "w" in locals() else {}),
            "field_temp_c": (temp_at_elevation(locals().get("temp_sl"), locals().get("elev", 0.0)) if "temp_sl" in locals() else None),
        },
        "wb": {
            "mode": locals().get("wb_mode"),
            "fuel": {
                "input_mode": st.session_state.get("fuel_input_mode"),
                "percent": st.session_state.get("fuel_percent"),
                "total_lb": st.session_state.get("fuel_total_lb"),
                "ext_left_full": st.session_state.get("ext_left_full"),
                "ext_right_full": st.session_state.get("ext_right_full"),
            },
            "stations": {
                sta: {
                    "store": st.session_state.get(f"store_{sta}", "—"),
                    "qty": AUTO_QTY_BY_STORE.get(st.session_state.get(f"store_{sta}", "—"), 0),
                    "pylon_removed": st.session_state.get(f"pylon_{sta}", False),
                } for sta in STATIONS
            },
        },
        "takeoff_config": {"flaps": locals().get("flaps"), "thrust": locals().get("thrust"), "derate_rpm": locals().get("derate")},
        "climb": {"cruise_alt_ft": locals().get("cruise_alt"), "profile": locals().get("climb_profile"), "ignore_reg": locals().get("ignore_reg")},
        "landing": {
            "condition": st.session_state.get("ldg_cond"),
            "candidates": [ {**d} for d in locals().get("dests", []) ],
            "scenarios": {
                "A_unfactored_ft": 4600,
                "B_unfactored_ft": 4200,
            },
        },
        "preset": locals().get("preset"),
    }
    st.markdown("### Scenario JSON (debug)")
    st.code(json.dumps(scenario, indent=2))
