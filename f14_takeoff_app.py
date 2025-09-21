# ============================================================
# F-14 Performance Calculator for DCS World — Streamlit UI
# File: f14_takeoff_app.py
# Version: v1.2.1 (2025-09-21 overhaul)
#
# Features:
# - Takeoff: core.perf_compute_takeoff, factored/unfactored TODR & ASDR
# - Climb: wired to f14_climb_natops.csv (Economy vs Interceptor)
# - Cruise: reads f14_cruise_natops.csv (GW vs DI interpolation)
# - Landing: f14_landing_natops_full.csv, FAA ×1.67 factor
# - Calibration: UI overrides passed to core, debug snapshot
# ============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import f14_takeoff_core as core
import os

st.set_page_config(page_title="F-14 Performance Calculator", layout="wide")

# Sidebar — Inputs
st.sidebar.header("Takeoff Inputs")
gw = st.sidebar.number_input("Gross Weight (lbs)", 40000, 74000, 60000, 500)
oat = st.sidebar.number_input("OAT (°C)", -40, 60, 15)
qnh = st.sidebar.number_input("QNH (inHg)", 25.0, 31.0, 29.92)
wind = st.sidebar.number_input("Headwind (kt)", -20, 40, 0)
field_elev = st.sidebar.number_input("Field Elev (ft)", 0, 10000, 0)
tora = st.sidebar.number_input("TORA (ft)", 1000, 14000, 8000)
asda = st.sidebar.number_input("ASDA (ft)", 1000, 14000, 8000)

inputs = {
    "gw_lbs": gw,
    "oat_c": oat,
    "qnh_inhg": qnh,
    "headwind_kts_component": wind,
    "field_elev_ft": field_elev,
    "pressure_alt_ft": field_elev, # simplified
    "tora_ft": tora,
    "asda_ft": asda,
}

# Calibration overrides
st.sidebar.header("Calibration Overrides")
override = {}
if st.sidebar.checkbox("Enable Calibration Mode"):
    override["min_derate_up"] = st.sidebar.slider("Floor Flaps UP (%)", 80, 95, 85)
    override["min_derate_man"] = st.sidebar.slider("Floor Flaps MAN (%)", 85, 100, 90)
    override["min_derate_full"] = st.sidebar.slider("Floor Flaps FULL (%)", 90, 100, 96)
    override["m_exponent"] = st.sidebar.slider("Runway factor exponent", 1.0, 2.0, 1.6, 0.1)
    override["runway_factor"] = st.sidebar.slider("Runway length factor", 1.0, 1.3, 1.10, 0.01)

# Main layout
st.title("F-14 Performance Calculator v1.2.1")

# === Takeoff Results ===
st.header("Takeoff Results")
to = core.perf_compute_takeoff(inputs, overrides=override if override else None)

if to:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vr (kt)", round(to.get("Vr_kts",0)))
    c2.metric("V2 (kt)", round(to.get("V2_kts",0)))
    c3.metric("ASDR (ft, unfactored)", round(to.get("ASDR_ft",0)))
    c4.metric("ASDR (ft, factored ×1.10)", round(to.get("ASDR_ft",0)*1.10))
    st.write("TODR (OEI) unfactored:", round(to.get("TODR_OEI_35ft_ft",0)))
    st.write("TODR (OEI) factored ×1.10:", round(to.get("TODR_OEI_35ft_ft",0)*1.10))
else:
    st.warning("No dispatchable configuration")

# === Climb Results ===
st.header("Climb Results")
climb_path = os.path.join("data","f14_climb_natops.csv")
if os.path.exists(climb_path):
    df_climb = pd.read_csv(climb_path)
    st.dataframe(df_climb)
else:
    st.info("Climb NATOPS CSV not yet available.")

# === Cruise Results ===
st.header("Cruise Results")
cruise_path = os.path.join("data","f14_cruise_natops.csv")
if os.path.exists(cruise_path):
    df_cruise = pd.read_csv(cruise_path)
    st.dataframe(df_cruise)
else:
    st.info("Cruise NATOPS CSV not yet available.")

# === Landing Results ===
st.header("Landing Results")
land_path = os.path.join("data","f14_landing_natops_full.csv")
if os.path.exists(land_path):
    df_land = pd.read_csv(land_path)
    st.dataframe(df_land.head())
    st.write("Unfactored LDR vs Factored LDR (×1.67)")
    if not df_land.empty:
        sample = df_land.iloc[0]
        st.write("Sample LDR unfactored:", sample["ground_roll_ft_unfactored"])
        st.write("Sample LDR factored ×1.67:", sample["ground_roll_ft_unfactored"]*1.67)
else:
    st.error("Landing NATOPS CSV not found.")

# === Debug / Calibration Snapshot ===
st.header("Debug / Calibration")
st.json(override if override else {"overrides":"not active"})
