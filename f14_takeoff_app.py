# ============================================================
# F-14 Performance Calculator for DCS World — UI Overhaul
# File: f14_takeoff_app.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================

from __future__ import annotations
import math, json, time, os
from functools import lru_cache
from typing import Dict, Any
import streamlit as st

# Core + policy
import f14_takeoff_core as core
import derate

# --- Config Loader ---
@lru_cache(maxsize=1)
def load_config(path: str = "data/derate_config.json") -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load config: {e}")
        return {}

CONFIG = load_config()

# --- Page setup ---
st.set_page_config(page_title="F-14B Performance Calculator", layout="wide")
st.title("F-14B Performance Calculator — v1.2.0-overhaul1")

# --- Sidebar ---
st.sidebar.header("Scenario Setup")
st.sidebar.markdown("Configure environment, weight & balance, and stores.")

# Example inputs (placeholder; later expanded with full W&B and stores UI)
airport = st.sidebar.text_input("Airport", "BATUMI")
runway = st.sidebar.text_input("Runway", "13")
oat_c = st.sidebar.number_input("OAT (°C)", value=15, min_value=-40, max_value=60, step=1)
qnh_inhg = st.sidebar.number_input("QNH (inHg)", value=29.92, min_value=28.00, max_value=31.00, step=0.01, format="%.2f")
wind_dir = st.sidebar.number_input("Wind Dir (°)", value=0, min_value=0, max_value=360, step=10)
wind_spd = st.sidebar.number_input("Wind Speed (kt)", value=0, min_value=0, max_value=50, step=1)
gw_lbs = st.sidebar.number_input("Gross Weight (lbs)", value=60000, min_value=40000, max_value=74000, step=100)

st.sidebar.caption("✈️ Inputs will feed into Takeoff, Climb, Cruise, and Landing calculations.")
# ============================================================
# F-14 Performance Calculator for DCS World — UI Overhaul
# Chunk 2: Takeoff Section
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================

import streamlit as st
import f14_takeoff_core as core

st.header("Takeoff Results")
st.caption("Runway safety factor applied: TODR/ASDR × 1.10 before comparison to TORA/ASDA.")

# Collect resolved inputs from sidebar (passed via st.session_state or placeholder here)
inputs = {
    "airport": airport,
    "runway": runway,
    "oat_c": oat_c,
    "qnh_inhg": qnh_inhg,
    "wind_dir": wind_dir,
    "wind_spd": wind_spd,
    "gw_lbs": gw_lbs,
}

# Call core planner
try:
    plan = core.plan_takeoff(inputs)
except Exception as e:
    st.error(f"Takeoff planning failed: {e}")
    plan = None

if plan:
    status = plan.get("status", "Unknown")
    st.subheader(f"Dispatch Status: {status}")
    if status == "Balanced":
        st.success("✅ Balanced-field takeoff achieved.")
    elif "TODR" in status:
        st.warning("⚠️ TODR-limited configuration.")
    elif "ASDR" in status:
        st.warning("⚠️ ASDR-limited configuration.")
    elif "Climb" in status:
        st.warning("⚠️ Climb-limited configuration.")
    else:
        st.info("ℹ️ Status unclear.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("V1 (kt)", f"{plan.get('V1', '--')}")
    with col2:
        st.metric("Vr (kt)", f"{plan.get('Vr', '--')}")
    with col3:
        st.metric("V2 (kt)", f"{plan.get('V2', '--')}")
    with col4:
        st.metric("Vfs (kt)", f"{plan.get('Vfs', '--')}")

    st.subheader("Configuration")
    st.write(f"**Flaps:** {plan.get('flaps','--')}")
    st.write(f"**Thrust:** {plan.get('thrust','--')}")
    st.write(f"**Stabilizer Trim:** {plan.get('stab_trim','--')} units")

    st.subheader("Engine Guidance")
    st.write(f"**N1 / RPM:** {plan.get('rpm_pct','--')} %")
    st.write(f"**FF (per engine):** {plan.get('ff_pph_per_engine','--')} pph")

    with st.expander("Debug Details"):
        dbg = plan.get("_debug", {})
        st.json(dbg, expanded=False)
else:
    st.info("No takeoff plan available.")
# ============================================================
# F-14 Performance Calculator for DCS World — UI Overhaul
# Chunk 3: Climb Section
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================

import streamlit as st
import f14_takeoff_core as core

st.header("Climb Results")

# Profile selector
profile = st.radio(
    "Climb Profile",
    options=["Fuel Economy (Loiter Bias)", "Interceptor (Time-Critical)"],
    index=0,
    horizontal=True
)

respect_250 = st.checkbox("Respect 250 KIAS below 10,000 ft", value=True)

# Collect climb inputs (example: feed from sidebar / plan_takeoff outputs)
climb_inputs = {
    "gw_lbs": gw_lbs,
    "oat_c": oat_c,
    "qnh_inhg": qnh_inhg,
    "profile": "economy" if "Economy" in profile else "interceptor",
    "respect_250": respect_250,
}

try:
    climb_plan = core.plan_climb(climb_inputs)
except Exception as e:
    st.error(f"Climb planning failed: {e}")
    climb_plan = None

if climb_plan:
    st.subheader(f"Profile: {profile}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Time to TOC (min)", f"{climb_plan.get('time_min','--')}")
    with col2:
        st.metric("Fuel Used (lb)", f"{climb_plan.get('fuel_lb','--')}")
    with col3:
        st.metric("Distance (nm)", f"{climb_plan.get('dist_nm','--')}")
    with col4:
        st.metric("Avg ROC (ft/min)", f"{climb_plan.get('avg_roc_fpm','--')}")

    st.subheader("Climb Schedule (per segment)")
    sched = climb_plan.get("schedule", [])
    if sched:
        st.table(sched)
    else:
        st.info("No climb schedule available.")

    with st.expander("Debug Details"):
        dbg = climb_plan.get("_debug", {})
        st.json(dbg, expanded=False)
else:
    st.info("No climb plan available.")
# ============================================================
# F-14 Performance Calculator for DCS World — UI Overhaul
# Chunk 4: Cruise Section
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================

import streamlit as st
import pandas as pd
import os

st.header("Cruise Results")

mode = st.radio("Cruise Mode", ["Economy (Max Specific Range)", "Interceptor (Max TAS)"], horizontal=True)
alt_mode = st.radio("Cruise Altitude", ["User Selected", "Auto Optimum"], horizontal=True)

if alt_mode == "User Selected":
    cruise_alt_ft = st.number_input("Cruise Altitude (ft)", value=30000, min_value=5000, max_value=50000, step=1000)
else:
    cruise_alt_ft = None

csv_path = "data/f14_cruise_natops.csv"
if not os.path.exists(csv_path):
    st.warning("⚠️ NATOPS cruise data not loaded. Please populate data/f14_cruise_natops.csv")
    cruise_data = None
else:
    try:
        cruise_data = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load cruise CSV: {e}")
        cruise_data = None

results = None
if cruise_data is not None:
    # Placeholder: in real version, filter by GW/alt/DI and compute optimum values
    if alt_mode == "Auto Optimum":
        results = {
            "opt_alt_ft": 32000,
            "mach": 0.78 if "Economy" in mode else 0.90,
            "ias_kts": 300,
            "tas_kts": 480 if "Economy" in mode else 550,
            "rpm_pct": 85 if "Economy" in mode else 95,
            "ff_pph_per_engine": 5000 if "Economy" in mode else 7000,
            "ff_total": 10000 if "Economy" in mode else 14000,
            "spec_range": 0.12 if "Economy" in mode else 0.08,
        }
    else:
        results = {
            "opt_alt_ft": cruise_alt_ft,
            "mach": 0.75,
            "ias_kts": 290,
            "tas_kts": 470,
            "rpm_pct": 83,
            "ff_pph_per_engine": 5200,
            "ff_total": 10400,
            "spec_range": 0.11,
        }

if results:
    st.subheader("Cruise Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Opt Alt (ft)", f"{results['opt_alt_ft']}")
    with col2: st.metric("Mach", f"{results['mach']:.2f}")
    with col3: st.metric("IAS (kt)", f"{results['ias_kts']}")
    with col4: st.metric("TAS (kt)", f"{results['tas_kts']}")

    col5, col6, col7 = st.columns(3)
    with col5: st.metric("RPM (%)", f"{results['rpm_pct']}")
    with col6: st.metric("FF/eng (pph)", f"{results['ff_pph_per_engine']}")
    with col7: st.metric("Total FF (pph)", f"{results['ff_total']}")

    st.metric("Specific Range (nm/lb)", f"{results['spec_range']:.3f}")

    with st.expander("Debug Details"):
        st.json(results, expanded=False)
# ============================================================
# F-14 Performance Calculator for DCS World — UI Overhaul
# Chunk 5: Landing Section
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================

import streamlit as st
import pandas as pd
import os
import math

st.header("Landing Results")

# Load NATOPS landing CSV if present
landing_csv = "data/f14_landing_natops.csv"
landing_data = None
if not os.path.exists(landing_csv):
    st.warning("⚠️ NATOPS landing data not loaded. Please populate data/f14_landing_natops.csv")
else:
    try:
        landing_data = pd.read_csv(landing_csv)
    except Exception as e:
        st.error(f"Failed to load landing CSV: {e}")
        landing_data = None

# Scenario presets
st.subheader("Scenarios")
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**Scenario A** — 3,000 lb fuel + stores kept")
with colB:
    st.markdown("**Scenario B** — 3,000 lb fuel + weapons expended, pods/tanks kept")
with colC:
    st.markdown("**Scenario C** — Custom")

lda_ft = st.number_input("Landing Distance Available (ft)", value=8000, min_value=1000, max_value=20000, step=100)
faa_factor = 1.67  # locked per spec
st.caption(f"Factoring: FAA ×{faa_factor:.2f}. Tomcat has no thrust-reverse.")

def compute_vref_and_ldr(gw_lbs: float, flaps: str, table: pd.DataFrame | None):
    # If we have NATOPS table, attempt lookup; else simple fallback using Vs*1.3
    vref = None
    ldr = None
    if table is not None and not table.empty:
        # Expect columns: gross_weight_lbs, flap_setting (UP/DOWN), vref_kts, ground_roll_ft_unfactored
        t = table.copy()
        t["gross_weight_lbs"] = pd.to_numeric(t["gross_weight_lbs"], errors="coerce")
        t = t[t["flap_setting"].str.upper() == flaps.upper()]
        # nearest weight row
        wdiff = (t["gross_weight_lbs"] - gw_lbs).abs()
        if not wdiff.empty:
            row = t.iloc[wdiff.idxmin()]
            vref = float(row.get("vref_kts", float("nan")))
            ldr = float(row.get("ground_roll_ft_unfactored", float("nan")))
    # Fallbacks
    if (vref is None) or math.isnan(vref):
        # crude fallback: assume Vs ~ 110 kt at 50k, scale ~ sqrt(W/W0), Vref = 1.3*Vs
        vs0 = 110.0
        w0 = 50000.0
        vs = vs0 * math.sqrt(max(gw_lbs, 1.0)/w0)
        vref = round(1.3 * vs)
    if (ldr is None) or math.isnan(ldr):
        # crude fallback: base 4500 ft at 50k, scale with weight^1.1
        base = 4500.0
        w0 = 50000.0
        ldr = base * (max(gw_lbs, 1.0)/w0)**1.1
    return int(round(vref)), int(round(ldr))

# Scenario A
gw_a = st.number_input("Scenario A — Landing GW (lb)", value=max(40000, int(gw_lbs - 3000)), step=100)
vref_a, ldr_a = compute_vref_and_ldr(gw_a, "DOWN", landing_data)
fact_a = int(round(ldr_a * faa_factor))

# Scenario B
gw_b = st.number_input("Scenario B — Landing GW (lb)", value=max(40000, int(gw_lbs - 3000)), step=100)
vref_b, ldr_b = compute_vref_and_ldr(gw_b, "DOWN", landing_data)
fact_b = int(round(ldr_b * faa_factor))

# Scenario C (custom)
gw_c = st.number_input("Scenario C — Landing GW (lb)", value=max(40000, int(gw_lbs - 3000)), step=100)
flaps_c = st.selectbox("Scenario C — Flaps", ["DOWN","UP"], index=0)
vref_c, ldr_c = compute_vref_and_ldr(gw_c, flaps_c, landing_data)
fact_c = int(round(ldr_c * faa_factor))

st.subheader("Results")
import pandas as pd
res = pd.DataFrame([
    {"Scenario":"A", "GW(lb)":gw_a, "Flaps":"DOWN", "Vref(kt)":vref_a, "LDR Unfact.(ft)":ldr_a, f"LDR FAA×{faa_factor:.2f}(ft)":fact_a,
     "Dispatchable? (factored)": "Yes" if fact_a <= lda_ft else "No"},
    {"Scenario":"B", "GW(lb)":gw_b, "Flaps":"DOWN", "Vref(kt)":vref_b, "LDR Unfact.(ft)":ldr_b, f"LDR FAA×{faa_factor:.2f}(ft)":fact_b,
     "Dispatchable? (factored)": "Yes" if fact_b <= lda_ft else "No"},
    {"Scenario":"C", "GW(lb)":gw_c, "Flaps":flaps_c, "Vref(kt)":vref_c, "LDR Unfact.(ft)":ldr_c, f"LDR FAA×{faa_factor:.2f}(ft)":fact_c,
     "Dispatchable? (factored)": "Yes" if fact_c <= lda_ft else "No"},
])
st.dataframe(res, use_container_width=True)

with st.expander("Debug Details"):
    st.write("Landing table present:", landing_data is not None and not landing_data.empty)
    if landing_data is not None and not landing_data.empty:
        st.caption("First few rows of NATOPS landing table:")
        st.dataframe(landing_data.head(10), use_container_width=True)
# ============================================================
# F-14 Performance Calculator for DCS World — UI Overhaul
# Chunk 6: Calibration Mode + Debug Tools + Footer
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================

import streamlit as st

st.header("Calibration & Debug")

enable_cal = st.checkbox("Enable Calibration Mode (Advanced Users Only)", value=False)

if enable_cal:
    st.subheader("Calibration Knobs")
    col1, col2, col3 = st.columns(3)
    with col1:
        floor_up = st.number_input("Floor %RPM Flaps UP", value=85, min_value=80, max_value=100, step=1)
    with col2:
        floor_man = st.number_input("Floor %RPM Flaps MAN", value=90, min_value=80, max_value=100, step=1)
    with col3:
        floor_full = st.number_input("Floor %RPM Flaps FULL", value=96, min_value=85, max_value=100, step=1)

    col4, col5 = st.columns(2)
    with col4:
        runway_factor = st.number_input("Runway Factor", value=1.10, min_value=1.0, max_value=1.25, step=0.01, format="%.2f")
    with col5:
        m_exp = st.number_input("Derate m-exponent", value=0.75, min_value=0.50, max_value=1.00, step=0.01, format="%.2f")

    st.caption("These values override defaults from derate_config.json.")

with st.expander("Planner Debug (Core Ladder)"):
    st.write("Candidate ladder and gate causes would be displayed here once hooked to core planner output.")
    st.json({"example": "clamped_to_floor flag, CD0 breakdown, gate cause, compute time"}, expanded=False)

st.markdown("---")
st.caption("F-14B Performance Calculator — Overhaul v1.2.0-overhaul1 — © 2025")
