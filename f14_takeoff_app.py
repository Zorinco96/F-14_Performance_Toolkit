# ============================================================
# F-14B Performance Calculator for DCS World — UI-first build
# File: f14_takeoff_app.py
# Version: v1.0.0 (2025-09-15)
#
# Purpose: Pure UI skeleton with realistic controls/layout and mocked outputs.
# Later we will extract logic into f14_takeoff_core.py and wire real math models.
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
import json
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import streamlit as st

# ============
# Page setup
# ============
st.set_page_config(
    page_title="F-14B Performance — DCS (UI-only)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============
# Data loading (CSV only; no core logic yet)
# ============
@st.cache_data(show_spinner=False)
def load_airports(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    # normalize numeric fields if present
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

# Try local first, then repo raw as fallback
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
        airports = load_airports(p)
        break
    except Exception:
        continue
if airports is None:
    st.stop()

perf = None
for p in PERF_PATHS:
    try:
        perf = load_perf(p)
        break
    except Exception:
        continue
if perf is None:
    st.stop()

# ============
# Sidebar — global env + quick presets
# ============
with st.sidebar:
    st.title("F-14B Perf — DCS")
    st.caption("UI-first skeleton • v1.0.0")

    st.subheader("Environment")
    colA, colB = st.columns(2)
    with colA:
        pa_ft = st.number_input("Pressure Alt (ft)", value=0.0, step=100.0)
        oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
        qnh = st.number_input("QNH (hPa)", value=1013.0, step=1.0)
    with colB:
        wind_kts = st.number_input("Wind (kts, +HW / -TW)", value=0.0, step=1.0)
        gust_kts = st.number_input("Gust (kts)", value=0.0, step=1.0)
        slope_pct = st.number_input("Runway Slope (%)", value=0.0, step=0.1)

    st.subheader("Quick Presets")
    preset = st.selectbox(
        "Load Preset",
        [
            "—",
            "Clean, 56k, 20° TO flaps",
            "Heavy, 72k, 20° TO flaps",
            "Recovery Tanker, 68k, 35° LDG flaps",
        ],
        index=0,
    )

    # Apply preset into session state (UI-only; math not applied yet)
    if preset != "—":
        st.session_state.setdefault("preset_loaded", preset)
        st.info(f"Preset loaded: {preset}")

# ============
# Header / status
# ============
st.markdown(
    """
    <div style="position:sticky;top:0;background:var(--background-color);padding:0.4rem 0;z-index:5;border-bottom:1px solid rgba(255,255,255,0.1)">
        <strong>F-14B Performance — DCS World</strong>
        <span style="opacity:0.7"> • v1.0.0 UI-first • Auto-recompute ON</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============
# Airport / runway picker
# ============
with st.expander("Airport & Runway", expanded=True):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        maps = sorted(airports["map"].dropna().unique().tolist())
        map_sel = st.selectbox("Map", maps)
    with c2:
        ap_choices = airports.loc[airports["map"] == map_sel, "airport_name"].dropna().unique().tolist()
        apt_sel = st.selectbox("Airport", sorted(ap_choices))
    with c3:
        rwy_rows = airports[(airports["map"] == map_sel) & (airports["airport_name"] == apt_sel)]
        ends = rwy_rows["runway_end"].dropna().astype(str).unique().tolist() if "runway_end" in rwy_rows.columns else []
        rwy_sel = st.selectbox("Runway End / Intersection", sorted(ends) if ends else ["Full Length"])
    with c4:
        tora = float(rwy_rows.get("tora_ft", pd.Series([rwy_rows.get("length_ft", pd.Series([0])).max()])).max()) if not rwy_rows.empty else 0.0
        elev = float(rwy_rows.get("threshold_elev_ft", pd.Series([0.0])).max()) if not rwy_rows.empty else 0.0
        st.metric("TORA (ft)", f"{tora:.0f}")
        st.metric("Elev (ft)", f"{elev:.0f}")

# ============
# Tabs: Takeoff / Climb / Landing / Tools
# ============
TAB_TK, TAB_CLIMB, TAB_LDG, TAB_TOOLS = st.tabs(["Takeoff", "Climb", "Landing", "Tools"]) 

# --- Takeoff UI
with TAB_TK:
    st.subheader("Takeoff")

    a, b, c, d = st.columns(4)
    with a:
        to_gw = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)
    with b:
        to_flap = st.number_input("TO Flaps (deg)", value=20.0, step=1.0)
    with c:
        bleed = st.selectbox("Bleed / ECS", ["ON", "OFF"], index=0)
    with d:
        epr_mode = st.selectbox("Thrust Mode", ["MIL", "AB"], index=0)

    e, f, g = st.columns(3)
    with e:
        st.toggle("Auto-calc", value=True, help="Live recompute on change (UI-only)")
    with f:
        st.button("Quick: Swap runway end", use_container_width=True)
    with g:
        st.button("Quick: Flip wind", use_container_width=True)

    st.markdown("### Results (mock)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("V1 (kt)", "145")
    m2.metric("Vr (kt)", "150")
    m3.metric("V2 (kt)", "160")
    m4.metric("TODR (ft)", "8200")

    st.caption("These are placeholder values pending core model integration.")

# --- Climb UI
with TAB_CLIMB:
    st.subheader("Climb")
    a, b, c = st.columns(3)
    with a:
        cl_gw = st.number_input("Climb GW (lb)", value=69000.0, step=500.0)
    with b:
        climb_mode = st.selectbox("Profile", ["Vy (best ROC)", "Vx (best angle)", "Noise abate"], index=0)
    with c:
        accel_alt = st.number_input("Accel Alt (ft AGL)", value=800.0, step=50.0)

    st.markdown("### Results (mock)")
    m1, m2 = st.columns(2)
    m1.metric("Vy (kt)", "220")
    m2.metric("ROC (fpm)", "6500")

# --- Landing UI
with TAB_LDG:
    st.subheader("Landing")

    a, b, c = st.columns(3)
    with a:
        ld_gw = st.number_input("Landing Weight (lb)", value=56000.0, step=500.0)
    with b:
        ld_flap = st.number_input("LDG Flaps (deg)", value=35.0, step=1.0)
    with c:
        vapp_min = st.number_input("Min Vapp Add (kts)", value=5.0, step=1.0)

    mode = st.radio("Mode", ["Required distance for weight", "Max landing weight"], horizontal=True)

    st.markdown("### Speeds (mock)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Vs (kt)", "121")
    m2.metric("Vref (kt)", "157")
    m3.metric("Vapp (kt)", "165")
    m4.metric("Va/c (kt)", "167")
    m5.metric("Vfs (kt)", "177")

    st.markdown("### Distances (mock)")
    if mode.startswith("Required"):
        st.metric("LDR 50ft (ft)", "4200")
    else:
        st.metric("Max LDW (lb)", "58400")
        st.caption("Constrained by runway TORA")

# --- Tools tab (permalink, kneeboard, CSV checks)
with TAB_TOOLS:
    st.subheader("Tools")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Copy V-speed line"):
            st.success("Copied: V1 145 / Vr 150 / V2 160 — Vref 157 / Vapp 165")
    with col2:
        st.button("Generate Kneeboard PDF", help="UI stub — disabled until models ready")
    with col3:
        st.button("Share Permalink", help="UI stub — will encode scenario in URL")

    st.markdown("### Scenario JSON (for debug)")
    scenario = {
        "env": {"pa_ft": pa_ft, "oat_c": oat_c, "qnh": qnh, "wind_kts": wind_kts, "gust_kts": gust_kts, "slope_pct": slope_pct},
        "airport": {"map": map_sel, "airport": apt_sel, "runway_end": rwy_sel, "tora_ft": tora, "elev_ft": elev},
        "takeoff": {"gw": to_gw, "flap": to_flap, "bleed": bleed, "thrust": epr_mode},
        "climb": {"gw": cl_gw, "profile": climb_mode, "accel_alt": accel_alt},
        "landing": {"gw": ld_gw, "flap": ld_flap, "vapp_min": vapp_min, "mode": mode},
    }
    st.code(json.dumps(scenario, indent=2))

st.divider()
st.caption("UI-only baseline v1.0.0. Next step: extract data classes + wire real models into f14_takeoff_core.py.")
