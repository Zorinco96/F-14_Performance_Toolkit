# ============================================================
# F-14 Performance Calculator for DCS World — UI-first build
# File: f14_takeoff_app.py
# Version: v1.1.2 (2025-09-17)
#
# Purpose: Full UI skeleton (no performance math). Implements the approved UX:
# - Curated F-14B presets (stores/fuel only; flaps/thrust AUTO)
# - Global runway search across all maps; slope removed
# - Environment paste/manual with unit auto-detect; ISA lapse to field
# - Weight & Balance (Detailed): simple DCS-style tiles; single store picker
#   + Fuel by % or lb; external tanks FULL/EMPTY; import stubs; Compatibility Mode (beta)
# - Takeoff Config: both radios vertical; last thrust option "DERATE (Manual)"
# - Climb Profile: default Most efficient; overlay graph + schedule placeholders
# - Landing Setup: scenarios A/B/C; tanks treated EMPTY at landing
# - Results in three separate sections (plain language) incl. Stabilizer Trim
# ============================================================
# 🚨 Bogged Down Protocol (BDP) 🚨
# 1) STOP  2) REVERT to last good tag  3) RESET chat if needed  4) SCOPE small
# 5) SAVE often with clear tags
# ============================================================

from __future__ import annotations
import re
import json
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import streamlit as st

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
INTERNAL_FUEL_MAX_LB = 16200       # rough F-14B internal capacity placeholder
EXT_TANK_FUEL_LB = 1800            # ~267 gal tank full, placeholder

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

# =========================
# Data loading (CSV only)
# =========================
@st.cache_data(show_spinner=False)
def load_airports(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    for col in ("length_ft", "tora_ft", "toda_ft", "asda_ft", "threshold_elev_ft", "heading_deg"):
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
# Helpers (unit detect, parsing)
# =========================
def detect_length_unit(text: str) -> Tuple[Optional[float], str]:
    """Return (length_ft, detected_unit_str). Accept '8500', '1.2 nm', '1.2nm'.
       Heuristic: suffix nm → NM; numeric ≤5 without suffix → NM; else feet.
    """
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
    """Parse QNH in inHg or hPa. Return (inHg, label)."""
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
    """Parse '270/15 kt' or '270/7 m/s'. Returns dict(dir_deg, spd_kts, unit)."""
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

# Fuel helpers (UI-only). External tanks are FULL/EMPTY; landing assumes EMPTY.
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
    st.caption("UI skeleton • v1.1.2 (no performance math)")

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
        # Reset stations
        for sta in STATIONS:
            st.session_state[f"store_{sta}"] = "—"
            st.session_state[f"qty_{sta}"] = 0
            st.session_state[f"pylon_{sta}"] = False
        # External tanks default off
        st.session_state.setdefault("ext_left_full", False)
        st.session_state.setdefault("ext_right_full", False)
        st.session_state["ext_left_full"] = False
        st.session_state["ext_right_full"] = False
        # Fuel default 80% internal
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

# Sticky header
st.markdown(
    """
    <div style="position:sticky;top:0;background:var(--background-color);padding:0.4rem 0;z-index:5;border-bottom:1px solid rgba(255,255,255,0.1)">
        <strong>F-14 Performance — DCS World</strong>
        <span style="opacity:0.7"> • UI-only v1.1.2 • Auto-recompute ON</span>
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
with st.expander("2) Runway", expanded=True):
    c1, c2, c3 = st.columns([1.4, 1.2, 1])

    with c1:
        search_all = st.text_input("Search airport (all maps)", placeholder="Type part of the airport name…")
        all_apts = airports[airports["airport_name"].notna()]
        matches = all_apts[all_apts["airport_name"].str.contains(search_all, case=False, na=False)] if search_all else all_apts
        pick_names = sorted(matches["airport_name"].unique().tolist())
        apt = st.selectbox("Airport", pick_names, key="rw_airport")

        # Derive default map from selected airport; allow override
        default_map = None
        mdf = matches[matches["airport_name"] == apt]
        if not mdf.empty:
            default_map = mdf["map"].iloc[0]
        maps = sorted(airports["map"].dropna().unique().tolist())
        map_sel = st.selectbox("Map", maps, index=(maps.index(default_map) if default_map in maps else 0), key="rw_map")

        sub = airports[(airports["airport_name"] == apt) & (airports["map"] == map_sel)]

    with c2:
        rwy_rows = sub
        ends = rwy_rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        rwy_end = st.selectbox("Runway End / Intersection", sorted(ends) if ends else ["Full Length"], key="rw_end")
        # Available distance (TORA if present else length)
        tora_series = rwy_rows.loc[rwy_rows["runway_end"].astype(str) == str(rwy_end), "tora_ft"] if "runway_end" in rwy_rows.columns else pd.Series()
        tora = float(tora_series.max() if not tora_series.empty else rwy_rows.get("length_ft", pd.Series([0.0])).max())
        elev = float(rwy_rows.get("threshold_elev_ft", pd.Series([0.0])).max())
        hdg = float(rwy_rows.get("heading_deg", pd.Series([0.0])).max())
        st.metric("Takeoff Run Available (ft)", f"{tora:.0f}")
        st.metric("Field Elevation (ft)", f"{elev:.0f}")
        st.metric("Runway Heading (°T)", f"{hdg:.0f}")

    with c3:
        st.checkbox("Manual runway entry", value=False, key="rw_manual")
        if st.session_state["rw_manual"]:
            mr_len = st.text_input("Runway length (ft or NM)", placeholder="8500 or 1.4 NM")
            len_ft, unit_label = detect_length_unit(mr_len)
            st.caption(f"Detected: {unit_label or '—'} → {f'{len_ft:.0f} ft' if len_ft else ''}")
            mr_elev = st.number_input("Elevation (ft)", value=elev or 0.0, step=50.0)
            mr_hdg = st.number_input("Heading (°T)", value=hdg or 0.0, step=1.0)
            mr_tora = st.number_input("TORA (ft)", value=float(len_ft or tora or 0.0), step=100.0)
            if len_ft:
                tora = float(len_ft)
            elev, hdg = mr_elev, mr_hdg
            st.info("Manual values override database for calculations.")

# =========================
# Section 3 — Environment (paste parser + manual)
# =========================
with st.expander("3) Environment", expanded=True):
    mode_env = st.radio("Input mode", ["Paste from DCS briefing", "Manual"], horizontal=True)

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
            temp_sl = st.number_input("Sea-level Temperature (°C)", value=15.0, step=1.0)
            qnh_text = st.text_input("QNH (inHg or hPa)", value="29.92")
            qnh_inhg, qnh_label = detect_pressure(qnh_text)
            st.caption(f"Detected: {qnh_label}")
        with c2:
            wind_text = st.text_input("Wind (deg/speed)", value="270/10 kt", help="e.g. 090/7m/s or 270/10 kt")
            w = parse_wind(wind_text)
            st.caption(f"Detected: {w['unit']}")
        with c3:
            field_temp = temp_at_elevation(temp_sl, elev or 0.0)
            st.metric("Estimated Temperature at Field (°C)", f"{field_temp:.1f}" if field_temp is not None else "—")
            hw, xw = hw_xw_components(w.get("dir_deg"), w.get("spd_kts"), hdg)
            st.metric("Head/Tailwind Component (kt)", f"{hw:+.0f}" if hw is not None else "—")
            st.metric("Crosswind Component (kt)", f"{xw:.0f}" if xw is not None else "—")

# =========================
# Section 4 — Weight & Balance (Simple or Detailed DCS-style)
# =========================
with st.expander("4) Weight & Balance", expanded=True):
    wb_mode = st.radio("Mode", ["Simple (enter Gross Takeoff Weight)", "Detailed (DCS-style loadout)"])

    if wb_mode.startswith("Simple"):
        gw_tow = st.number_input("Gross Takeoff Weight (lb)", value=70000.0, step=500.0)
        gw_ldg_plan = st.number_input("Planned Landing Weight (lb)", value=56000.0, step=500.0)
        st.caption("Switch to Detailed mode to build weight via stations and fuel.")
    else:
        # Fuel — either percentage of internal or total pounds; ext tanks FULL/EMPTY
        st.markdown("**Fuel** — Enter by percentage or total pounds. External tanks are either FULL or EMPTY.")
        cF1, cF2 = st.columns(2)
        fuel_input_mode = cF1.radio("Fuel input", ["Percent", "Pounds (lb)"], index=0, key="fuel_input_mode")
        cT = st.columns(3)
        with cT[0]:
            ext_left_full = st.checkbox("External Tank LEFT: FULL", key="ext_left_full")
        with cT[1]:
            ext_right_full = st.checkbox("External Tank RIGHT: FULL", key="ext_right_full")
        with cT[2]:
            st.caption("Landing scenarios assume external tanks are EMPTY of fuel.")

        if fuel_input_mode == "Percent":
            fuel_percent = st.number_input("Total Fuel (%)", value=80.0, min_value=0.0, max_value=100.0, step=1.0, key="fuel_percent")
            computed_total = compute_total_fuel_lb(fuel_percent, ext_left_full, ext_right_full)
            st.metric("Computed Total Fuel (lb)", f"{computed_total:.0f}" if computed_total is not None else "—")
            st.session_state["fuel_total_lb"] = computed_total
        else:
            fuel_total_lb = st.number_input("Total Fuel (lb)", value=float(st.session_state.get("fuel_total_lb", 12000.0)), step=100.0, key="fuel_total_lb")
            computed_pct = compute_percent_from_total(fuel_total_lb, ext_left_full, ext_right_full)
            st.metric("Computed Internal Fuel (%)", f"{computed_pct:.0f}%" if computed_pct is not None else "—")
            st.session_state["fuel_percent"] = computed_pct

        # Import stubs
        cimp1, cimp2 = st.columns(2)
        std_choice = cimp1.selectbox("Import standard loadout (stub)",
                                     ["—","Fleet CAP","Heavy Intercept","Bombcat LANTIRN","Strike (iron)"])
        miz = cimp2.file_uploader("Import from DCS .miz (stub)", type=["miz"])
        compat_beta = st.checkbox("Compatibility Mode (beta)", value=False,
                                  help="Filters obviously impossible station/store pairs (approx).")

        st.markdown("**Loadout (F-14B)** — Click station tiles and pick a store; set quantity. Use symmetry to mirror.")
        cols = st.columns(5)
        for i, sta in enumerate(STATIONS):
            with cols[i % 5]:
                st.write(f"**STA {sta}**")
                store_key, qty_key, pylon_key = f"store_{sta}", f"qty_{sta}", f"pylon_{sta}"
                cur_store = st.session_state.get(store_key, "—")
                st.session_state.setdefault(qty_key, 0)

                # Allowed stores (rough filter when compatibility on)
                allowed = list(STORES_CATALOG.keys())
                if compat_beta:
                    if sta in ("1A","8A"):
                        allowed = ["—","AIM-9M"]  # glove rails A
                    elif sta in ("1B","8B"):
                        allowed = ["—","AIM-7M","LANTIRN"]  # glove rails B
                    elif sta in ("2","7"):
                        allowed = ["—","Drop Tank 267 gal"]  # tank stations
                    elif sta in ("3","4","5","6"):
                        allowed = ["—","AIM-54C","AIM-7M","Mk-82","Mk-83","GBU-12","ZUNI LAU-10"]

                st.selectbox(f"Store {sta}", allowed,
                             index=(allowed.index(cur_store) if cur_store in allowed else 0),
                             key=store_key)
                st.number_input(f"Qty {sta}", value=int(st.session_state.get(qty_key, 0)),
                                min_value=0, max_value=2, step=1, key=qty_key)
                st.checkbox("Remove pylon", value=bool(st.session_state.get(pylon_key, False)), key=pylon_key)

                # Symmetry
                sym = SYMMETRY.get(sta)
                if sym and st.button(f"Apply → {sym}", key=f"symbtn_{sta}"):
                    st.session_state[f"store_{sym}"] = st.session_state[store_key]
                    st.session_state[f"qty_{sym}"] = st.session_state[qty_key]
                    st.session_state[f"pylon_{sym}"] = st.session_state[pylon_key]

        st.markdown("### Totals (mock)")
        t1, t2, t3 = st.columns(3)
        t1.metric("Gross Weight (lb)", "70,500")
        t2.metric("Center of Gravity (%MAC)", "23.5")
        t3.metric("Stabilizer Trim (units)", "+2.0")
        st.caption("Weights/CG/trim are placeholders until the performance core is wired.")

# =========================
# Section 5 — Takeoff Configuration
# =========================
with st.expander("5) Takeoff Configuration", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        flaps = st.radio("Flaps", ["AUTO", "UP", "MANEUVER", "FULL"], horizontal=False)
    with c2:
        thrust = st.radio("Thrust", ["AUTO", "MILITARY", "AFTERBURNER", "DERATE (Manual)"], horizontal=False)
    with c3:
        derate = 0
        if thrust == "DERATE (Manual)":
            derate = st.slider("Derate (RPM %)", min_value=70, max_value=100, value=95)
        st.metric("Required climb gradient (all engines)", "≥ 300 ft/NM")
    st.caption("AUTO thrust will target 14 CFR 121.189 and ≥300 ft/NM AEO using the minimum required setting (to be modeled).")

# =========================
# Section 6 — Climb Profile (default: Most efficient)
# =========================
with st.expander("6) Climb Profile", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        cruise_alt = st.number_input("Cruise Altitude (MSL ft)", value=28000.0, step=1000.0)
    with c2:
        climb_profile = st.selectbox("Climb Profile", ["Most efficient climb", "Minimum time to altitude"], index=0)
    with c3:
        ignore_reg = st.checkbox("Ignore regulatory speed restrictions (≤250 KIAS <10k)")

    # Cards (placeholders)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Time to 10,000 ft", "02:40")
    r2.metric("Time to Cruise Altitude", "07:50")
    r3.metric("Fuel to Top of Climb", "2,100 lb")
    r4.metric("Top of Climb Distance", "37 NM")

    # Overlay climb traces (placeholder)
    climb_overlay = pd.DataFrame({
        "Time_min": [0, 2, 4, 6, 8],
        "MostEff_ft": [0, 6000, 12000, 20000, 28000],
        "MinTime_ft": [0, 7000, 13000, 20500, 28000],
    }).set_index("Time_min")
    st.line_chart(climb_overlay)

    # Schedule placeholders
    st.markdown("**Climb schedule (placeholders)**")
    s1, s2 = st.columns(2)
    with s1:
        st.write("• **1,000 ft AGL:** RPM/FF — / —, Target: V2 + 15 kt")
        st.write("• **Up to 10,000 ft:** RPM/FF — / —, Target IAS — kt")
    with s2:
        st.write("• **10k → Mach transition:** RPM/FF — / —, Target IAS — kt")
        st.write("• **Mach transition → Cruise:** RPM/FF — / —, Target Mach —")

# =========================
# Section 7 — Landing Setup (Scenarios A/B/C)
# =========================
with st.expander("7) Landing Setup", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        maps = sorted(airports["map"].dropna().unique().tolist())
        dest_map = st.selectbox("Destination Map", maps, key="ldg_map")
        dest_sub = airports[airports["map"] == dest_map]
        dest_apt = st.selectbox("Destination Airport", sorted(dest_sub["airport_name"].dropna().unique().tolist()), key="ldg_airport")
    with c2:
        dest_rows = dest_sub[dest_sub["airport_name"] == dest_apt]
        dest_end = st.selectbox("Runway End", sorted(dest_rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) or ["Full Length"], key="ldg_end")
        dest_tora_series = dest_rows.loc[dest_rows["runway_end"].astype(str) == str(dest_end), "tora_ft"] if "runway_end" in dest_rows.columns else pd.Series()
        dest_tora = float(dest_tora_series.max() if not dest_tora_series.empty else dest_rows.get("length_ft", pd.Series([0.0])).max())
        st.metric("Takeoff Run Available at Destination (ft)", f"{dest_tora:.0f}")
    with c3:
        cond = st.radio("Runway condition", ["DRY", "WET"], horizontal=True)
        st.caption("14 CFR 121.195 factors will apply (to be modeled). External tank fuel is assumed EMPTY on landing.")

    st.markdown("**Landing Scenarios**")
    st.markdown("A) 3,000 lb fuel; **all external stores from takeoff retained** (pods/tanks kept).")
    st.markdown("B) 3,000 lb fuel; **weapons expended** (missiles/bombs/rockets removed; pods/tanks kept).")
    st.markdown("C) Custom: set **remaining fuel** and **current stores** below.")

    with st.expander("C) Custom landing configuration", expanded=False):
        cF1, cF2 = st.columns(2)
        with cF1:
            ldg_fuel_mode = st.radio("Fuel input", ["Percent", "Pounds (lb)"], index=1, key="ldg_fuel_mode")
        with cF2:
            l_ext_left = st.checkbox("External Tank LEFT: FULL (landing)", key="ldg_ext_left_full")
            l_ext_right = st.checkbox("External Tank RIGHT: FULL (landing)", key="ldg_ext_right_full")
            st.caption("Note: For realism, leave landing tanks EMPTY.")
        if ldg_fuel_mode == "Percent":
            ldg_fuel_pct = st.number_input("Landing Fuel (%)", value=20.0, min_value=0.0, max_value=100.0, step=1.0, key="ldg_fuel_pct")
            ldg_total_lb = compute_total_fuel_lb(ldg_fuel_pct, l_ext_left, l_ext_right)
            st.metric("Computed Landing Fuel (lb)", f"{ldg_total_lb:.0f}" if ldg_total_lb is not None else "—")
        else:
            ldg_fuel_lb = st.number_input("Landing Fuel (lb)", value=3000.0, step=100.0, key="ldg_fuel_lb")
            st.metric("Computed Internal Fuel (%)", f"{compute_percent_from_total(ldg_fuel_lb, l_ext_left, l_ext_right):.0f}%")

        st.markdown("**Current Stores (DCS-style)**")
        colsL = st.columns(5)
        for i, sta in enumerate(STATIONS):
            with colsL[i % 5]:
                st.write(f"**STA {sta}**")
                st.selectbox(f"Store {sta} (LDG)", list(STORES_CATALOG.keys()), key=f"ldg_store_{sta}")
                st.number_input(f"Qty {sta} (LDG)", value=0, min_value=0, max_value=2, step=1, key=f"ldg_qty_{sta}")

# =========================
# RESULTS — three separate sections (plain language)
# =========================
st.markdown("---")

# Takeoff Results
st.header("Takeoff Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Takeoff Decision Speed (V1)", "145 kt")
m2.metric("Rotate Speed (Vr)", "150 kt")
m3.metric("Takeoff Safety Speed (V2)", "160 kt")
m4.metric("Final Segment Speed (Vfs)", "180 kt")

m5, m6, m7 = st.columns(3)
m5.metric("Required Flap Setting", "AUTO (placeholder)")
m6.metric("Required Thrust Setting", "AUTO (placeholder)")
m7.metric("Stabilizer Trim", "+2.0 units")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Runway Distance Required", "8,200 ft")
r2.metric("Runway Distance Available", f"{tora:.0f} ft")
r3.metric("Accelerate-go Distance", "9,100 ft")
r4.metric("Accelerate-stop Distance", "8,600 ft")

s1, s2 = st.columns(2)
s1.metric("Estimated Distance to Rotate in DCS", "3,400 ft")
s2.metric("Estimated Distance to Lift Off and Reach 35 ft in DCS", "5,100 ft")

st.warning("Most restrictive (mock): Accelerate-stop distance exceeds available runway.")

# Accessory graph: distance to rotate by flap/thrust (positive-only, mock)
st.markdown("**How configuration affects distance to rotate (mock)**")
rot_df = pd.DataFrame({
    "Configuration": ["Flaps UP / MIL", "MANEUVER / MIL", "FULL / MIL", "FULL / AB"],
    "Distance_ft": [max(0,4200), max(0,3600), max(0,3200), max(0,2800)],
})
st.line_chart(rot_df.set_index("Configuration"))

# Table: configuration breakdown (mock)
st.markdown("**Configuration breakdown (mock)**")
st.dataframe(pd.DataFrame({
    "Flaps": ["UP", "MANEUVER", "FULL", "FULL"],
    "Thrust": ["MIL", "MIL", "MIL", "AB"],
    "V1 (kt)": [150, 145, 143, 140],
    "Vr (kt)": [155, 150, 148, 145],
    "V2 (kt)": [165, 160, 158, 155],
    "Rotate Dist (ft)": [4200, 3600, 3200, 2800],
}))

st.markdown("---")

# Climb Results
st.header("Climb Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Time to 10,000 ft", "02:40")
c2.metric("Time to Cruise Altitude", "07:50")
c3.metric("Fuel to Top of Climb", "2,100 lb")
c4.metric("Top of Climb Distance", "37 NM")

# Overlay climb curves (mock)
climb_df = pd.DataFrame({
    "Time_min": [0, 2, 4, 6, 8],
    "MostEff_ft": [0, 6000, 12000, 20000, 28000],
    "MinTime_ft": [0, 7000, 13000, 20500, 28000],
}).set_index("Time_min")
st.line_chart(climb_df)

st.info("Regulatory status: Compliant (≤250 KIAS below 10,000 ft) — placeholder")

st.markdown("---")

# Landing Results
st.header("Landing Results")

st.subheader("Scenario A — 3,000 lb fuel, stores retained")
la1, la2, la3, la4, la5 = st.columns(5)
la1.metric("Stall Speed (Vs)", "121 kt")
la2.metric("Reference Speed (Vref)", "157 kt")
la3.metric("Approach Speed (Vapp)", "165 kt")
la4.metric("Go-Around Speed (Vac)", "167 kt")
la5.metric("Final Segment Speed (Vfs)", "177 kt")
st.metric("Required Landing Distance from 50 ft", "4,600 ft")

st.subheader("Scenario B — 3,000 lb fuel, weapons expended (pods/tanks kept)")
lb1, lb2, lb3, lb4, lb5 = st.columns(5)
lb1.metric("Stall Speed (Vs)", "118 kt")
lb2.metric("Reference Speed (Vref)", "153 kt")
lb3.metric("Approach Speed (Vapp)", "161 kt")
lb4.metric("Go-Around Speed (Vac)", "164 kt")
lb5.metric("Final Segment Speed (Vfs)", "174 kt")
st.metric("Required Landing Distance from 50 ft", "4,200 ft")

st.subheader("Scenario C — Custom (set above)")
lc1, lc2, lc3, lc4, lc5 = st.columns(5)
lc1.metric("Stall Speed (Vs)", "—")
lc2.metric("Reference Speed (Vref)", "—")
lc3.metric("Approach Speed (Vapp)", "—")
lc4.metric("Go-Around Speed (Vac)", "—")
lc5.metric("Final Segment Speed (Vfs)", "—")
st.metric("Required Landing Distance from 50 ft", "—")

# Accessory graph: LDR comparison (mock)
ldr_df = pd.DataFrame({"Scenario": ["A: stores kept", "B: weapons expended"], "LDR_ft": [4600, 4200]}).set_index("Scenario")
st.bar_chart(ldr_df)

# Table: runway margins (mock)
st.markdown("**Runway margins (mock)**")
st.dataframe(pd.DataFrame({
    "Runway": ["{}".format("Runway {}".format(dest_end))],
    "Available (ft)": [dest_tora],
    "Required (ft) — A": [4600],
    "Required (ft) — B": [4200],
    "Margin (ft) — A": [dest_tora - 4600],
    "Margin (ft) — B": [dest_tora - 4200],
}))

st.markdown("---")

# =========================
# Scenario JSON (debug)
# =========================
if show_debug:
    scenario = {
        "aircraft": ac,
        "runway": {
            "map": map_sel,
            "airport": apt,
            "end": rwy_end,
            "tora_ft": tora,
            "elev_ft": elev,
            "heading_deg": hdg,
            "manual_override": bool(st.session_state.get("rw_manual")),
        },
        "environment": {
            "temp_sl_c": (locals().get("temp_sl")),
            "qnh_inhg": (locals().get("qnh_inhg")),
            "wind": (locals().get("w") if "w" in locals() else {}),
            "field_temp_c": (temp_at_elevation(locals().get("temp_sl"), elev) if "temp_sl" in locals() else None),
        },
        "wb": {
            "mode": wb_mode,
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
                    "qty": st.session_state.get(f"qty_{sta}", 0),
                    "pylon_removed": st.session_state.get(f"pylon_{sta}", False),
                } for sta in STATIONS
            },
        },
        "takeoff_config": {"flaps": flaps, "thrust": thrust, "derate_rpm": (derate if 'derate' in locals() else None)},
        "climb": {"cruise_alt_ft": cruise_alt, "profile": climb_profile, "ignore_reg": ignore_reg},
        "landing": {
            "dest_map": dest_map,
            "dest_airport": dest_apt,
            "dest_end": dest_end,
            "dest_tora_ft": dest_tora,
            "condition": cond,
            "scenarios": {
                "A": {"fuel_lb": 3000, "stores": "takeoff stores kept", "ext_tanks_fuel": "EMPTY at landing"},
                "B": {"fuel_lb": 3000, "stores": "weapons expended; pods/tanks kept", "ext_tanks_fuel": "EMPTY at landing"},
                "C": {
                    "fuel_mode": st.session_state.get("ldg_fuel_mode"),
                    "fuel_pct": st.session_state.get("ldg_fuel_pct"),
                    "fuel_lb": st.session_state.get("ldg_fuel_lb"),
                    "ext_left_full": st.session_state.get("ldg_ext_left_full"),
                    "ext_right_full": st.session_state.get("ldg_ext_right_full"),
                    "stations": {sta: {"store": st.session_state.get(f"ldg_store_{sta}", "—"), "qty": st.session_state.get(f"ldg_qty_{sta}", 0)} for sta in STATIONS},
                },
            },
        },
        "preset": preset,
    }
    st.markdown("### Scenario JSON (debug)")
    st.code(json.dumps(scenario, indent=2))

st.caption("UI-only baseline v1.1.2. Once you sign off on UX, we'll move logic into f14_takeoff_core.py and wire real models.")
