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

AIRPORTS_PATHS = [
    "dcs_airports.csv",
    "https://raw.githubusercontent.com/Zorinco96/f14_takeoff_app.py/main/dcs_airports.csv",
]
PERF_PATHS = [
    "f14_perf.csv",
    "https://raw.githubusercontent.com/Zorinco96/f14_takeoff_app.py/main/f14_perf.csv",
]

airports = None
for p in AIRPORTS_PATHS:
    try:
        airports = load_airports(p); break
    except Exception:
        continue
if airports is None:
    st.error("Could not load dcs_airports.csv. Ensure it exists locally or in GitHub.")
    st.stop()

perf = None
for p in PERF_PATHS:
    try:
        perf = load_perf(p); break
    except Exception:
        continue

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
            wind_text = st.text_input("Wind (deg/speed)", value="270/10 kt", help="e.g. 090/7m/s or 270/10 kt")
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
t1.metric("Gross Weight (lb)", f"{wb['gw_tow_lb']:.0f}")
t2.metric("Center of Gravity (%MAC)", f"{wb['cg_percent_mac']:.1f}")
t3.metric("Stabilizer Trim (units)", f"{wb['stab_trim_units']:+.1f}")

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

def build_engine_table(thrust_sel: str, derate_pct: int) -> pd.DataFrame:
    def rows(n1_to, ff_to, n1_ic, ff_ic, n1_cl, ff_cl):
        return [
            {"Phase": "Takeoff",       "Target N1 (%)": int(round(n1_to)), "FF (pph/engine)": int(round(ff_to))},
            {"Phase": "Initial Climb", "Target N1 (%)": int(round(n1_ic)), "FF (pph/engine)": int(round(ff_ic))},
            {"Phase": "Climb Segment", "Target N1 (%)": int(round(n1_cl)), "FF (pph/engine)": int(round(ff_cl))},
        ]
    if thrust == "MILITARY":
        data = rows(96, 7000, 95, 6500, 93, 6000)
    elif thrust == "AFTERBURNER":
        data = rows(102, 19000, 98, 10000, 95, 7500)
    elif thrust == "DERATE (Manual)":
        n1_to = max(85, min(100, int(derate if 'derate' in locals() else 95)))
        n1_ic = max(85, n1_to - 2); n1_cl = max(85, n1_ic - 2)
        scale = (n1_to / 96.0)
        data = rows(n1_to, 7000*scale, n1_ic, 6500*scale, n1_cl, 6000*scale)
    else:
        data = rows(95, 6500, 94, 6250, 92, 5750)
    return pd.DataFrame(data)

perf_takeoff_ref = getattr(core, "perf_compute_takeoff", None)

gw_lb         = float(locals().get("wb", {}).get("gw_tow_lb", DEFAULT_GTOW))
field_elev_ft = float(locals().get("elev", 0.0))
oat_c         = float(locals().get("field_temp", 15.0))
headwind_kts  = float(locals().get("hw", 0.0) or 0.0)
runway_slope  = 0.0
stores_list   = get_stores_drag_list()
mode_flag     = "DCS"

tora_ft = _s_int(locals().get("tora", 0))
asda_ft = tora_ft

wp = core.WIND_POLICY_50_150 if wind_policy_choice.startswith("50%") else core.WIND_POLICY_0_150

sel_inputs = core.AutoSelectInputs(
    available_tora_ft=int(tora_ft),
    available_asda_ft=int(asda_ft),
    runway_heading_deg=float(locals().get("hdg", 0.0) or 0.0),
    headwind_kts_raw=float(locals().get("hw", 0.0) or 0.0),
    aeo_required_ft_per_nm=float(st.session_state.get("req_climb_grad_ft_nm", 200)),
    wind_policy=wp,
)

scenario_ctx = dict(
    gw_lb=gw_lb,
    field_elev_ft=field_elev_ft,
    oat_c=oat_c,
    runway_slope=runway_slope,
    stores=stores_list,
    mode=mode_flag,
    headwind_kts=float(locals().get("hw", 0.0) or 0.0),
)

def compute_candidate(*, flaps: str, thrust_pct: int, v1_kt: float | None, context: dict) -> dict:
    cfg = "TO_FLAPS" if flaps in ("MANEUVER", "FULL") else "CLEAN"
    thrust_mode = "MIL" if thrust_pct <= 100 else "MAX"
    base = {}
    try:
        if callable(perf_takeoff_ref):
            base = cached_perf_takeoff(
                gw_lb=context["gw_lb"],
                field_elev_ft=context["field_elev_ft"],
                oat_c=context["oat_c"],
                headwind_kts=context["headwind_kts"],
                runway_slope=context["runway_slope"],
                thrust_mode=("MAX" if thrust_mode == "MAX" else "MIL"),
                mode=context["mode"],
                config=cfg,
                sweep_deg=20.0,
                stores=tuple(context["stores"]),
            )
        else:
            base["__diagnostic__"] = "perf_takeoff_unavailable"
    except Exception as e:
        base["__exception__"] = str(e)

    gr = float(base.get("GroundRoll_ft", 0.0) or 0.0)
    d35 = float(base.get("DistanceTo35ft_ft", 0.0) or 0.0)
    asdr = gr * 1.15 if gr > 0 else d35 * 1.10
    todr = d35
    m = max(asdr, todr) if max(asdr, todr) > 0 else 1.0
    diff_ratio = abs(asdr - todr) / m

    return {
        "ASDR_ft": asdr,
        "TODR_OEI_35ft_ft": todr,
        "balanced_diff_ratio": diff_ratio,
        "AEO_min_grad_ft_per_nm_to_1000": None,
        "OEI_second_seg_gross_pct": None,
        "OEI_final_seg_gross_pct": None,
        "perf": base,
    }

auto_mode = (flaps == "Auto-Select") or (thrust == "Auto-Select")
selection = None
if auto_mode:
    selection = core.auto_select_flaps_thrust(
        sel=sel_inputs,
        scenario_context=scenario_ctx,
        compute_candidate=compute_candidate,
    )

if selection and selection.dispatchable:
    flap_display = selection.flaps
    thrust_display = selection.thrust_label
    t_res = selection.perf or {}
    balanced_badge = selection.balanced_label or ""
    governing = selection.governing_side
else:
    flap_display = flaps if flaps != "Auto-Select" else "FULL"
    thrust_display = thrust
    balanced_badge = ""
    governing = ""
    t_res = {}
    if callable(perf_takeoff_ref):
        try:
            t_res = cached_perf_takeoff(
                gw_lb=gw_lb,
                field_elev_ft=field_elev_ft,
                oat_c=oat_c,
                headwind_kts=headwind_kts,
                runway_slope=runway_slope,
                thrust_mode=("MAX" if thrust == "AFTERBURNER" else "MIL"),
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
        st.metric("V1 (kt)",   f"{t_res['VR_kts']:.0f}")
        st.metric("Vr (kt)",   f"{t_res['VR_kts']:.0f}")
        st.metric("V2 (kt)",   f"{t_res['V2_kts']:.0f}")
        st.metric("Vfs (kt)",  f"{max(t_res['V2_kts']*1.1, t_res['VLOF_kts']*1.15):.0f}")
    else:
        st.metric("V1 (kt)", "—"); st.metric("Vr (kt)", "—"); st.metric("V2 (kt)", "—"); st.metric("Vfs (kt)", "—")

with col2:
    st.subheader("Configuration")
    st.metric("Flaps", flap_display)
    st.metric("Thrust", (thrust_display or "").replace("(Manual)",""))
    st.metric("Stabilizer Trim", f"{wb.get('stab_trim_units', 0.0):+0.1f} units")
    st.caption("N1% / FF(pph/engine) — guidance (table)")

    _derate_pct_display = 100
    if isinstance(thrust_display, str):
        import re as _re
        m = _re.search(r"DERATE\s*\((\d+)%\)", thrust_display.upper())
        if m:
            _derate_pct_display = int(m.group(1))
        elif thrust_display.upper().startswith("MILITARY"):
            _derate_pct_display = 100
        elif thrust_display.upper().startswith("DERATE") and 'derate' in locals():
            _derate_pct_display = int(locals().get("derate", 95))

    engine_df = build_engine_table(thrust_display, int(locals().get("derate",95)))
    st.dataframe(engine_df, hide_index=True, use_container_width=True)

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
    if t_res:
        required = int(round(t_res["DistanceTo35ft_ft"]))
        available = tora_ft
        if available >= required:
            st.success("Dispatchable")
            st.caption("Limiting: None")
        else:
            st.error("NOT Dispatchable")
            st.caption("Limiting: TORA vs Dist to 35 ft")
        st.metric("Expected Climb Gradient (AEO)", "— ft/NM (placeholder)")
    else:
        st.info("Perf model not available.")
        st.metric("Expected Climb Gradient (AEO)", "— ft/NM (placeholder)")

st.divider()
st.subheader("DCS Expected Performance (calculated)")
if t_res:
    p1, p2 = st.columns(2)
    p1.metric("Distance to reach Vr (ft)", f"{int(round(t_res['GroundRoll_ft']*0.67)):,}")
    p2.metric("Distance to Liftoff / 35 ft (ft)", f"{int(round(t_res['DistanceTo35ft_ft'])):,}")
else:
    st.info("Model unavailable — no values shown (placeholder).")
st.divider()

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

# (Duplicate helper retained at end for backwards-compat anchors)
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
    if totals["AIM-9M"] >= 2: deltas.append("2xSidewinders")
    if totals["AIM-7M"] >= 2: deltas.append("2xSparrows")
    if totals["AIM-54C"] >= 2: deltas.append("2xPhoenix")
    if totals["Drop Tank 267 gal"] >= 2: deltas.append("FuelTank2x")
    seen = set(); out = []
    for d in deltas:
        if d not in seen:
            out.append(d); seen.add(d)
    return out
