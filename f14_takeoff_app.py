# ============================================================
# F-14 Performance Calculator for DCS World — UI-first build
# File: f14_takeoff_app.py
# Version: v1.1.3 (2025-09-17)
#
# Changes from v1.1.2-ui-wb-fix3:
# 1) Takeoff Results: clearer break before "DCS Expected Performance".
# 2) Landing Setup: auto-pulls selected departure airport/runway as Destination 1,
#    user can add up to 4 landing candidates total.
# 3) Landing Results: includes Factored Landing Distance, Available Landing Distance,
#    and (placeholder) Calculated Maximum Landing Weight per runway/conditions.
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

# Simple W&B defaults
DEFAULT_GTOW = 74349               # MTOW; also our default GTOW in Simple
DEFAULT_LDW  = 60000               # field MLDW (carrier 54,000)

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
# Helpers (unit detect, parsing)
# =========================
def detect_length_unit(text: str) -> Tuple[Optional[float], str]:
    """Return (length_ft, detected_unit_str). Accept '8500', '1.2 nm', '1.2nm'."""
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
    st.caption("UI skeleton • v1.1.3 (no performance math)")

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
        search_all = st.text_input("Search airport (all maps)", placeholder="Type part of the airport name…")
        all_apts = airports_cached[airports_cached["airport_name"].notna()]
        matches = all_apts[all_apts["airport_name"].str.contains(search_all, case=False, na=False)] if search_all else all_apts
        pick_names = sorted(matches["airport_name"].unique().tolist())
        apt = st.selectbox("Airport", pick_names, key="rw_airport")

        default_map = None
        mdf = matches[matches["airport_name"] == apt]
        if not mdf.empty:
            default_map = mdf["map"].iloc[0]
        maps = sorted(airports_cached["map"].dropna().unique().tolist())
        map_sel = st.selectbox("Map", maps, index=(maps.index(default_map) if default_map in maps else 0), key="rw_map")

        sub = airports_cached[(airports_cached["airport_name"] == apt) & (airports_cached["map"] == map_sel)]

    with c2:
        rwy_rows = sub
        ends = rwy_rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        rwy_end = st.selectbox("Runway End / Intersection", sorted(ends) if ends else ["Full Length"], key="rw_end")
        tora_series = rwy_rows.loc[rwy_rows["runway_end"].astype(str) == str(rwy_end), "tora_ft"] if "runway_end" in rwy_rows.columns else pd.Series()
        tora = int((tora_series.max() if not tora_series.empty else rwy_rows.get("length_ft", pd.Series([0.0])).max()) or 0)
        elev = int((rwy_rows.get("threshold_elev_ft", pd.Series([0.0])).max()) or 0)
        hdg = float((rwy_rows.get("heading_deg", pd.Series([0.0])).max()) or 0.0)
        # Also try LDA if present
        lda = int((rwy_rows.get("lda_ft", pd.Series([0.0])).max()) or tora)
        st.metric("Takeoff Run Available (ft)", f"{tora:,}")
        st.metric("Field Elevation (ft)", f"{elev:,}")
        st.metric("Runway Heading (°T)", f"{hdg:.0f}")

    with c3:
        st.checkbox("Manual runway entry", value=False, key="rw_manual")
        if st.session_state["rw_manual"]:
            mr_len = st.text_input("Runway length (ft or NM)", placeholder="8500 or 1.4 NM")
            len_ft, unit_label = detect_length_unit(mr_len)
            st.caption(f"Detected: {unit_label or '—'} → {f'{int(len_ft):,} ft' if len_ft else ''}")
            mr_elev = st.number_input("Elevation (ft)", value=elev or 0, step=50, min_value=0, format="%d")
            mr_hdg = st.number_input("Heading (°T)", value=int(hdg or 0), step=1, min_value=0, max_value=359, format="%d")
            mr_tora = st.number_input("TORA (ft)", value=int(len_ft or tora or 0), step=100, min_value=0, format="%d")
            if len_ft:
                tora = int(len_ft)
            elev, hdg = int(mr_elev), float(mr_hdg)
            st.info("Manual values override database for calculations.")

# =========================
# Section 3 — Environment (paste parser + manual)
# =========================
with st.expander("3) Environment", expanded=True):
    # Default to Manual (from previous patch)
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
            field_temp = temp_at_elevation(temp_sl, int(locals().get("elev", 0)), ISA_LAPSE_C_PER_1000FT)
            st.metric("Estimated Temperature at Field (°C)", f"{field_temp:.1f}" if field_temp is not None else "—")
            hw, xw = hw_xw_components(w.get("dir_deg"), w.get("spd_kts"), float(locals().get("hdg", 0.0)))
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
        # Fuel — Pounds default and shown first
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
            st.metric("Computed Total Fuel (lb)", f"{int(computed_total):,}" if computed_total is not None else "—")
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

                # Allowed stores (rough filter when compatibility on)
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

                # Symmetry
                sym = SYMMETRY.get(sta)
                if sym and st.button(f"Apply → {sym}", key=f"symbtn_{sta}"):
                    st.session_state[f"store_{sym}"] = st.session_state[store_key]
                    st.session_state[f"qty_{sym}"] = AUTO_QTY_BY_STORE.get(st.session_state[store_key], 0)
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
# (6) Takeoff Results — stays after Takeoff Config
# =========================
st.header("Takeoff Results")

# Vertical columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("V-Speeds")
    st.metric("V1", "145 kt")
    st.metric("Vr", "150 kt")
    st.metric("V2", "160 kt")
    st.metric("Vfs", "180 kt")

with col2:
    st.subheader("Configuration")
    st.metric("Flaps", flaps if 'flaps' in locals() else "—")
    st.metric("Thrust", thrust if 'thrust' in locals() else "—")
    st.metric("N1 / FF", "— / —")   # placeholder until engine model wired
    st.metric("Stabilizer Trim", "+2.0 units")

with col3:
    st.subheader("Runway Distances (mock)")
    st.metric("Required (RTO/TOR)", "8,200 ft")
    st.metric("Available (TORA)", f"{int(locals().get('tora', 0)):,} ft")
    st.metric("Accelerate-Stop", "8,600 ft")
    st.metric("Accelerate-Go", "9,100 ft")

with col4:
    st.subheader("Dispatchability")
    available = int(locals().get("tora", 0))
    required = 8600  # limiter (mock)
    if available >= required:
        st.success("Dispatchable")
        st.caption("Limiting: None (mock)")
    else:
        st.error("NOT Dispatchable")
        st.caption("Limiting: Accelerate-Stop distance (mock)")

# ===== Clear break for DCS Expected Performance =====
st.divider()
st.subheader("DCS Expected Performance (mock)")
p1, p2 = st.columns(2)
p1.metric("Distance to reach Vr", "3,400 ft")
p2.metric("Distance to Liftoff (35 ft)", "5,100 ft")
st.divider()

# =========================
# Section 6 — Climb Profile (default: Most efficient)
# =========================
with st.expander("6) Climb Profile", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        cruise_alt = st.number_input("Cruise Altitude (MSL ft)", value=28000, step=1000, min_value=0, format="%d")
    with c2:
        climb_profile = st.selectbox("Climb Profile", ["Most efficient climb", "Minimum time to altitude"], index=0)
    with c3:
        ignore_reg = st.checkbox("Ignore regulatory speed restrictions (≤250 KIAS <10k)")

    # Cards (placeholders)
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Time to 10,000 ft", "02:40")
    r2.metric("Time to Cruise Altitude", "07:50")
    r3.metric("Fuel to Top of Climb", "2,100 lb")
    r4.metric("TOC Distance", "37 NM")
    r5.metric("Time to TO + 100 NM", "09:45")   # placeholder
    r6.metric("Fuel to TO + 100 NM", "2,650 lb")# placeholder

    # Overlay climb traces (mock)
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

st.markdown("---")

# =========================
# Section 7 — Landing Setup
#  - Auto-seeds Destination 1 from the departure selection (airport/runway)
#  - User may add up to four candidates total
# =========================
with st.expander("7) Landing Setup", expanded=True):
    def pick_destination(slot_idx: int, default_map: Optional[str], default_airport: Optional[str], default_end: Optional[str]) -> Dict[str, Any]:
        """Render selectors for one destination slot and return info dict."""
        maps = sorted(airports["map"].dropna().unique().tolist())
        map_key = f"ldg_map_{slot_idx}"
        apt_key = f"ldg_airport_{slot_idx}"
        end_key = f"ldg_end_{slot_idx}"

        map_idx = maps.index(default_map) if (default_map in maps) else 0
        sel_map = st.selectbox(f"[{slot_idx}] Map", maps, index=map_idx, key=map_key)

        sub = airports[airports["map"] == sel_map]
        apts = sorted(sub["airport_name"].dropna().unique().tolist())
        apt_idx = apts.index(default_airport) if (default_airport in apts) else (0 if apts else 0)
        sel_apt = st.selectbox(f"[{slot_idx}] Airport", apts, index=apt_idx, key=apt_key)

        rows = sub[sub["airport_name"] == sel_apt]
        ends = sorted(rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) or ["Full Length"]
        end_idx = ends.index(default_end) if (default_end in ends) else 0
        sel_end = st.selectbox(f"[{slot_idx}] Runway End", ends, index=end_idx, key=end_key)

        # Distances
        tora_series = rows.loc[rows["runway_end"].astype(str) == str(sel_end), "tora_ft"] if "runway_end" in rows.columns else pd.Series()
        lda_series  = rows.loc[rows["runway_end"].astype(str) == str(sel_end), "lda_ft"] if "runway_end" in rows.columns else pd.Series()
        length_series = rows.get("length_ft", pd.Series([0.0]))
        tora_ft = int((tora_series.max() if not tora_series.empty else length_series.max()) or 0)
        lda_ft  = int((lda_series.max() if not lda_series.empty else length_series.max()) or 0)

        m1, m2 = st.columns(2)
        m1.metric("Available Takeoff (TORA)", f"{tora_ft:,} ft")
        m2.metric("Available Landing (LDA)", f"{lda_ft:,} ft")
        return {"map": sel_map, "airport": sel_apt, "end": sel_end, "tora_ft": tora_ft, "lda_ft": lda_ft}

    # How many candidates (1–4)
    num_dest = st.number_input("Number of landing candidates", min_value=1, max_value=4, value=1, step=1, format="%d")

    dests: List[Dict[str, Any]] = []

    # Destination 1 seeded from departure selection in Section 2
    dep_map = locals().get("map_sel")
    dep_airport = locals().get("apt")
    dep_end = locals().get("rwy_end")
    st.markdown("**Destination 1 (seeded from departure selection)**")
    dests.append(pick_destination(1, dep_map, dep_airport, dep_end))

    # Additional destinations
    for slot in range(2, num_dest + 1):
        st.markdown(f"**Destination {slot}**")
        dests.append(pick_destination(slot, None, None, None))

    # Common landing condition selector (applies to all shown results)
    cond = st.radio("Runway condition (applies to all candidates below)", ["DRY", "WET"], horizontal=True, key="ldg_cond")
    st.caption("14 CFR 121.195 factors will apply (to be modeled). External tank fuel is assumed EMPTY on landing.")

st.markdown("---")

# =========================
# Landing Results
#  - Adds Factored Landing Distance, Available Landing Distance (LDA),
#    and (placeholder) Calculated Maximum Landing Weight.
# =========================
st.header("Landing Results")

# Helper to compute factored distance (placeholder factors)
def factored_distance(unfactored_ft: int, condition: str) -> int:
    # Common planning factors (placeholder): dry ~1.67, wet ~1.92
    factor = 1.67 if condition == "DRY" else 1.92
    return int(round(unfactored_ft * factor))

# Mock base unfactored distances (A/B scenarios); choose more conservative B for table?
UNFACTORED_LDR_A = 4600
UNFACTORED_LDR_B = 4200

# Per-destination summary table
if 'dests' in locals():
    rows = []
    for i, d in enumerate(dests, start=1):
        lda_ft = int(d.get("lda_ft", 0))
        # Use A for conservative planning by default (placeholder)
        base_unfactored = UNFACTORED_LDR_A
        f_ldr = factored_distance(base_unfactored, st.session_state.get("ldg_cond", "DRY"))

        # Placeholder max landing weight calc:
        # scale field MLDW (60,000) by LDA / Factored-LDR, cap at 60,000
        mlw_est = int(min(60000, max(0, (60000 * (lda_ft / f_ldr)) if f_ldr > 0 else 0)))

        rows.append({
            "Dest": f"{i}: {d['airport']} ({d['map']}) — RWY {d['end']}",
            "Available LDA (ft)": lda_ft,
            "Factored LDR (ft)": f_ldr,
            "Calc Max Landing Wt (lb)": mlw_est,
        })
    st.dataframe(pd.DataFrame(rows))

# Existing scenario blocks (still shown for quick feel — placeholders)
st.subheader("Scenario A — 3,000 lb fuel, stores retained")
la1, la2, la3, la4, la5 = st.columns(5)
la1.metric("Stall Speed (Vs)", "121 kt")
la2.metric("Reference Speed (Vref)", "157 kt")
la3.metric("Approach Speed (Vapp)", "165 kt")
la4.metric("Go-Around Speed (Vac)", "167 kt")
la5.metric("Final Segment Speed (Vfs)", "177 kt")
st.metric("Required Landing Distance from 50 ft (unfactored)", f"{UNFACTORED_LDR_A:,} ft")

st.subheader("Scenario B — 3,000 lb fuel, weapons expended (pods/tanks kept)")
lb1, lb2, lb3, lb4, lb5 = st.columns(5)
lb1.metric("Stall Speed (Vs)", "118 kt")
lb2.metric("Reference Speed (Vref)", "153 kt")
lb3.metric("Approach Speed (Vapp)", "161 kt")
lb4.metric("Go-Around Speed (Vac)", "164 kt")
lb5.metric("Final Segment Speed (Vfs)", "174 kt")
st.metric("Required Landing Distance from 50 ft (unfactored)", f"{UNFACTORED_LDR_B:,} ft")

# Comparison chart (mock)
if 'dests' in locals() and len(dests) > 0:
    cmp_df = pd.DataFrame({
        "Destination": [f"{i}: {d['airport']} ({d['map']})" for i, d in enumerate(dests, start=1)],
        "Available LDA (ft)": [int(d.get("lda_ft", 0)) for d in dests],
        "Factored LDR A (ft)": [factored_distance(UNFACTORED_LDR_A, st.session_state.get("ldg_cond", "DRY")) for _ in dests],
        "Factored LDR B (ft)": [factored_distance(UNFACTORED_LDR_B, st.session_state.get("ldg_cond", "DRY")) for _ in dests],
    }).set_index("Destination")
    st.bar_chart(cmp_df[["Available LDA (ft)", "Factored LDR A (ft)"]])

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
            "candidates": [
                {**d} for d in locals().get("dests", [])
            ],
            "scenarios": {
                "A_unfactored_ft": UNFACTORED_LDR_A,
                "B_unfactored_ft": UNFACTORED_LDR_B,
            },
        },
        "preset": locals().get("preset"),
    }
    st.markdown("### Scenario JSON (debug)")
    st.code(json.dumps(scenario, indent=2))

st.caption("UI-only baseline v1.1.3. Next: wire f14_takeoff_core.py for W&B totals/CG/trim → takeoff → climb → landing.")
