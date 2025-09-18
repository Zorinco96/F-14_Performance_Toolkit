import streamlit as st
from data_loaders import load_calibration_csv
from takeoff_core_min import ground_takeoff_run

st.set_page_config(page_title="Validation Harness", layout="wide")
st.title("Validation Harness — Takeoff (Ground Roll & 35 ft)")

with st.sidebar:
    st.header("Inputs")
    weight = st.number_input("Weight (lb)", 50000, 74000, 70000, step=500)
    elev = st.number_input("Field Elevation (ft)", 0, 14000, 2410, step=10)
    oat = st.number_input("OAT (°C)", -40, 60, 15, step=1)
    headwind = st.number_input("Headwind (+) / Tailwind (-) (kt)", -30, 30, 0, step=1)
    slope = st.number_input("Runway Slope (ft/ft)", -0.02, 0.02, 0.0, step=0.001, format="%.3f")
    thrust_mode = st.selectbox("Thrust", ["MIL","MAX"], index=1)
    mode = st.selectbox("Mode", ["DCS","FAA"], index=0)
    vr_known = st.checkbox("Override VR (kt)?", value=False)
    vr_val = st.number_input("VR Override (kt)", 100, 200, 150, step=1, disabled=not vr_known)

st.markdown("—")

calib = load_calibration_csv("calibration.csv")

res = ground_takeoff_run(
    weight_lbf=weight,
    alt_ft=elev,
    oat_c=oat,
    headwind_kts=headwind,
    runway_slope=slope,
    config="TO_FLAPS",
    sweep_deg=20.0,
    power=thrust_mode,
    mode=mode,
    calib=calib,
    vr_kts=(vr_val if vr_known else None),
    dt=0.1
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("VR (kt)", f"{res['VR_kts']:.1f}")
col2.metric("Ground Roll (ft)", f"{res['GroundRoll_ft']:.0f}")
col3.metric("Distance to 35 ft (ft)", f"{res['DistanceTo35ft_ft']:.0f}")
col4.metric("Time to 35 ft (s)", f"{res['Time_s']:.1f}")

st.caption("This is a compact validation slice. Full rotation/airborne math will replace the placeholder estimates and be calibrated to your DCS tests.")
