# ============================================================
# F-14B Performance Calculator for DCS World
# File: f14_takeoff_app.py
# Version: v0.9.0 (2025-09-14)
#
# Changelog:
# - New baseline UI with three tabs (Takeoff / Climb / Landing)
# - Landing tab wired to core.landing (required distance & max-LDW)
# - Debounced inputs, float-safe number_inputs to avoid type issues
# - Bogged Down Protocol embedded
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
import streamlit as st
import pandas as pd

from f14_takeoff_core import (
    load_perf_table,
    load_airports_table,
    LandingInputs,
    landing_required_distance,
    max_landing_weight_for_runway,
    takeoff_stub_example,
    climb_stub_example,
    TakeoffInputs,
    ClimbInputs,
)

st.set_page_config(page_title="F-14B Perf (DCS)", layout="wide")

# ---- Session caches ----
@st.cache_data(show_spinner=False)
def get_perf():
    return load_perf_table()

@st.cache_data(show_spinner=False)
def get_airports():
    return load_airports_table()

perf = get_perf()
airports = get_airports()

st.title("F-14B Performance Calculator — DCS World (v0.9.0)")

# ===============
# Sidebar — common environment
# ===============
with st.sidebar:
    st.header("Environment")
    pa_ft = st.number_input("Pressure Altitude (ft)", value=0.0, step=100.0)
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    wind = st.number_input("Steady Wind (kts, +HW / -TW)", value=0.0, step=1.0)
    gust = st.number_input("Gust Increment (kts)", value=0.0, step=1.0)
    cal = st.slider("Calibration Factor (Landing)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)

    st.caption("\nModel v0.9.0 — Landing distance uses a placeholder conservative model; calibrate with your DCS test results.")

# ===============
# Tabs
# ===============
TAB_TK, TAB_CLIMB, TAB_LDG = st.tabs(["Takeoff", "Climb", "Landing"])

# --- Takeoff (stub)
with TAB_TK:
    st.subheader("Takeoff (placeholder)")
    c1, c2, c3 = st.columns(3)
    with c1:
        gw = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)
    with c2:
        flap = st.number_input("Flaps (deg)", value=20.0, step=1.0)
    with c3:
        _ = st.selectbox("Runway Condition", ["DRY", "WET"], index=0)

    to_inputs = TakeoffInputs(gw_lbs=gw, flap_deg=flap, pa_ft=pa_ft, oat_c=oat_c)
    to_res = takeoff_stub_example(perf, to_inputs)

    st.metric("Vr (kt)", f"{to_res['Vr_kt']:.0f}")
    st.metric("V2 (kt)", f"{to_res['V2_kt']:.0f}")
    st.metric("Takeoff Distance (ft)", f"{to_res['TOD_ft']:.0f}")

# --- Climb (stub)
with TAB_CLIMB:
    st.subheader("Climb (placeholder)")
    c1, c2 = st.columns(2)
    with c1:
        gwc = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0, key="climb_gw")
    with c2:
        pass

    climb_in = ClimbInputs(gw_lbs=gwc, pa_ft=pa_ft, oat_c=oat_c)
    climb_res = climb_stub_example(climb_in)

    st.metric("Vy (kt)", f"{climb_res['Vy_kt']:.0f}")
    st.metric("ROC (fpm)", f"{climb_res['ROC_fpm']:.0f}")

# --- Landing (functional v0.9.0)
with TAB_LDG:
    st.subheader("Landing")

    # Runway/airport picker
    left, right = st.columns([2, 1])
    with left:
        map_sel = st.selectbox("Map", sorted(airports["map"].dropna().unique().tolist()))
        ap_choices = airports.loc[airports["map"] == map_sel, "airport_name"].dropna().unique().tolist()
        apt_sel = st.selectbox("Airport", sorted(ap_choices))
        rwy_rows = airports[(airports["map"] == map_sel) & (airports["airport_name"] == apt_sel)]
        if "runway_end" in rwy_rows.columns:
            rwy_choices = rwy_rows["runway_end"].dropna().astype(str).unique().tolist()
        else:
            rwy_choices = ["Full Length"]
        rwy_sel = st.selectbox("Runway End / Intersection", sorted(rwy_choices))

    with right:
        # Resolve runway available
        rwy_sel_rows = rwy_rows[rwy_rows["runway_end"].astype(str) == str(rwy_sel)] if "runway_end" in rwy_rows.columns else rwy_rows
        tora = float(rwy_sel_rows.get("tora_ft", pd.Series([rwy_sel_rows.get("length_ft", pd.Series([0])).max()])).max())
        elev = float(rwy_sel_rows.get("threshold_elev_ft", pd.Series([0.0])).max())
        st.metric("Runway TORA (ft)", f"{tora:.0f}")
        st.metric("Threshold Elev (ft)", f"{elev:.0f}")

    mode = st.radio("Landing Mode", ["Required distance for given weight", "Max landing weight for runway"], horizontal=True)

    # Common landing inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        gw_ldg = st.number_input("Landing Weight (lb)", value=54000.0, step=500.0)
    with c2:
        flap_ldg = st.number_input("Flaps (deg, landing)", value=35.0, step=1.0)
    with c3:
        vapp_min_add = st.number_input("Min Vapp Additive (kts)", value=5.0, step=1.0)

    inputs = LandingInputs(
        gw_lbs=gw_ldg,
        flap_deg=flap_ldg,
        pa_ft=float(pa_ft if pa_ft else elev),
        oat_c=float(oat_c),
        steady_wind_kts=float(wind),
        gust_increment_kts=float(gust),
        vapp_min_additive_kts=float(vapp_min_add),
        calibration_factor=float(cal),
    )

    if mode.startswith("Required"):
        res = landing_required_distance(perf, inputs)
        s = res.speeds
        st.markdown("### Speeds")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vs (kt)", f"{s.Vs_kt:.0f}")
        m2.metric("Vref (kt)", f"{s.Vref_kt:.0f}")
        m3.metric("Vapp (kt)", f"{s.Vapp_kt:.0f}")
        m4.metric("Va/c (kt)", f"{s.Vac_kt:.0f}")
        st.metric("Vfs (kt)", f"{s.Vfs_kt:.0f}")

        st.markdown("### Distances")
        st.metric("LDR (50 ft) (ft)", f"{res.ldr_50ft_ft:.0f}")
        st.caption("Conservative placeholder model — tune Calibration in sidebar.")

    else:
        w_max, res = max_landing_weight_for_runway(perf, float(tora), inputs)
        s = res.speeds
        st.metric("Max Landing Weight (lb)", f"{w_max:.0f}")
        st.caption("Computed s.t. LDR ≤ runway TORA.")
        st.markdown("### Speeds at Max-LDW")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vs (kt)", f"{s.Vs_kt:.0f}")
        m2.metric("Vref (kt)", f"{s.Vref_kt:.0f}")
        m3.metric("Vapp (kt)", f"{s.Vapp_kt:.0f}")
        m4.metric("Va/c (kt)", f"{s.Vac_kt:.0f}")
        st.metric("Vfs (kt)", f"{s.Vfs_kt:.0f}")

st.divider()
st.caption("This version is a stable baseline (v0.9.0). If development bogs down, revert to this tag and resume.")
