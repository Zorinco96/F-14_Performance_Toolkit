# test_app.py — v1.3.3
# Smoke test with debug enabled for CorePlanner (UP/MANEUVER/FULL).

import streamlit as st
import f14_takeoff_core as core

st.set_page_config(page_title="F-14 Toolkit — Smoke Test", layout="wide")
st.title("F-14 Performance Toolkit — Smoke Test (v1.3.3)")

flap_mode = st.sidebar.selectbox(
    "Flaps",
    options=[("UP (0°)", 0), ("MANEUVER (20°)", 20), ("FULL (40°)", 40)],
    index=1,
    format_func=lambda x: x[0]
)
flap_deg = flap_mode[1]

gw_lbs = st.sidebar.slider("Gross Weight (lb)", 50000, 74000, 65000, 500)
field_elev_ft = st.sidebar.number_input("Field Elev (ft)", value=39.0, step=1.0)
qnh_inhg = st.sidebar.number_input("QNH (inHg)", value=29.92, step=0.01)
oat_c = st.sidebar.slider("OAT (°C)", -20, 45, 15, 1)
tora_ft = st.sidebar.number_input("TORA (ft)", value=8000, step=500)
asda_ft = st.sidebar.number_input("ASDA (ft)", value=8000, step=500)
allow_ab = st.sidebar.checkbox("Allow Afterburner escalation", value=False)

if st.button("Compute"):
    res = core.plan_takeoff_with_optional_derate(
        flap_deg=int(flap_deg),
        gw_lbs=float(gw_lbs),
        field_elev_ft=float(field_elev_ft),
        qnh_inhg=float(qnh_inhg),
        oat_c=float(oat_c),
        tora_ft=int(tora_ft),
        asda_ft=int(asda_ft),
        allow_ab=bool(allow_ab),
        debug=True,   # Always include debug diagnostics
    )
    st.subheader("Planner Output")
    st.json(res)
