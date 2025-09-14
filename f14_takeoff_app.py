# f14_takeoff_app.py
# Streamlit UI for DCS F-14B Takeoff Performance App

import math
import numpy as np
import pandas as pd
import streamlit as st

from f14_climb_guidance import recommend_climb_schedule, format_climb_card

from f14_takeoff_core import (
    load_perf_csv,
    compute_takeoff,
    trim_anu,
    hpa_to_inhg,
    parse_wind_entry,
    AEO_VR_FRAC,
    AEO_VR_FRAC_FULL,
    detect_length_text_to_ft,
    detect_elev_text_to_ft,
    estimate_ab_multiplier,
)

# ------------------------------ page config ------------------------------
st.set_page_config(
    page_title="F-14B Takeoff Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .f14-card {
        background-color: #111;
        padding: 0.75em;
        margin-bottom: 1em;
        border-radius: 0.5em;
        box-shadow: 0 0 8px rgba(255,255,255,0.15);
    }
    .f14-card-ok { background-color: #113311; }
    .f14-card-bad { background-color: #331111; }
    .f14-sticky {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #222;
        padding: 0.5em;
        border-bottom: 2px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ load performance db ------------------------------
@st.cache_data
def load_db():
    return load_perf_csv("f14_perf.csv")

perfdb = load_db()

# ------------------------------ sidebar inputs ------------------------------
st.sidebar.header("Aircraft & Environment")

gw = st.sidebar.number_input("Gross Weight (lb)", 40000, 74000, 65000, step=500)
flap_mode = st.sidebar.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"])
thrust_mode = st.sidebar.selectbox("Thrust", ["Auto-Select", "MIL", "AB", "DERATE", "Manual Derate"])
target_n1 = st.sidebar.slider("Target N1 (%)", 80, 100, 95)

field_elev = st.sidebar.number_input("Field Elevation (ft)", -1000, 10000, 0)
qnh = st.sidebar.number_input("QNH (inHg)", 27.0, 32.0, 29.92, step=0.01)
oat_c = st.sidebar.number_input("OAT (°C)", -40, 60, 15)
slope_pct = st.sidebar.number_input("Runway Slope (%)", -2.0, 2.0, 0.0, step=0.1)

wind_str = st.sidebar.text_input("Wind (deg@kts, e.g. 270/10)", "")
wind_units = st.sidebar.selectbox("Wind Units", ["kts", "m/s"])
wind_policy = st.sidebar.selectbox("Wind Policy", ["None", "50/150"])

tora = st.sidebar.text_input("TORA", "10000")
toda = st.sidebar.text_input("TODA", "10000")
asda = st.sidebar.text_input("ASDA", "10000")

shorten_ft = st.sidebar.number_input("Declared Distance Reduction (ft)", 0, 5000, 0)

compliance_mode = st.sidebar.radio("Compliance Mode", ["Regulatory (OEI)", "AEO Practical"], index=0)

# ------------------------------ parse wind ------------------------------
parsed_wind = parse_wind_entry(wind_str, wind_units)
if parsed_wind:
    wind_dir, wind_speed = parsed_wind
else:
    wind_dir, wind_speed = (0.0, 0.0)

# ------------------------------ run calculation ------------------------------
res = compute_takeoff(
    perfdb,
    rwy_heading_deg=0.0,  # user can extend to select runway heading
    tora_ft=detect_length_text_to_ft(tora),
    toda_ft=detect_length_text_to_ft(toda),
    asda_ft=detect_length_text_to_ft(asda),
    field_elev_ft=detect_elev_text_to_ft(field_elev),
    slope_pct=slope_pct,
    shorten_ft=shorten_ft,
    oat_c=oat_c,
    qnh_inhg=qnh,
    wind_speed=wind_speed,
    wind_dir_deg=wind_dir,
    wind_units=wind_units,
    wind_policy=wind_policy,
    gw_lbs=gw,
    flap_mode=flap_mode,
    thrust_mode=thrust_mode,
    target_n1_pct=target_n1,
)

ok = (compliance_mode.startswith("Regulatory") and res.oei_grad_ok) or (
    compliance_mode.startswith("AEO") and res.aeo_grad_ok
)

summary_line = f"ASD={res.asd_ft:.0f} / AEO liftoff={res.agd_aeo_liftoff_ft:.0f} / OEI reg={res.agd_reg_oei_ft:.0f}"

# ------------------------------ sticky header ------------------------------
st.markdown(
    f"<div class='f14-sticky'><b>{'✅ COMPLIANT' if ok else '⛔ NOT AUTHORIZED'}</b> — {summary_line}</div>",
    unsafe_allow_html=True,
)

# ------------------------------ climb recommendation ------------------------------
sched = recommend_climb_schedule(
    gw_lbs=float(gw),
    flap_after_cleanup="UP",
    accel_alt_ft_agl=1000.0,
    policy="conservative",
)
st.markdown(format_climb_card(sched), unsafe_allow_html=True)

# ------------------------------ main output ------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader("V-Speeds")
    st.metric("V1 (kt)", f"{res.v1:.0f}")
    st.metric("Vr (kt)", f"{res.vr:.0f}")
    st.metric("V2 (kt)", f"{res.v2:.0f}")
    st.metric("Vs (kt)", f"{res.vs:.0f}")
with c2:
    st.subheader("Configuration")
    st.metric("Flaps", res.flap_text)
    st.metric("Thrust", res.thrust_text)
    st.metric("N1 (%)", f"{res.n1_pct:.0f}")
    st.metric("Trim (ANU)", f"{trim_anu(gw, 20):.1f}")
with c3:
    st.subheader("Distances")
    st.metric("ASD (ft)", f"{res.asd_ft:.0f}")
    st.metric("AEO liftoff (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")
    st.metric("OEI reg (ft)", f"{res.agd_reg_oei_ft:.0f}")
with c4:
    st.subheader("Availability")
    st.metric("Runway available (ft)", f"{max(0.0, float(detect_length_text_to_ft(tora)) - float(shorten_ft)):.0f}")
    st.metric("Limiting factor", res.limiting)
    st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
    st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
    st.caption("Tailwind > 10 kt or crosswind > 30 kt → NOT AUTHORIZED.")
    st.caption(
        "Climb gradients: "
        + ("✅ OEI 2.4%" if res.oei_grad_ok else "⛔ OEI 2.4%")
        + "  |  "
        + ("✅ AEO 200 ft/nm" if res.aeo_grad_ok else "⛔ AEO 200 ft/nm")
    )

# ------------------------------ notes ------------------------------
st.subheader("Notes")
for n in res.notes:
    st.write("•", n)
