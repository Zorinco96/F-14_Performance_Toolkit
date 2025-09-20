
# test_app.py — Interactive smoke test for F-14 Performance Toolkit core (stability patch)
import os
import streamlit as st
import f14_takeoff_core as core

st.set_page_config(page_title="F-14 Toolkit — Smoke Test", layout="wide")
st.title("F-14 Performance Toolkit — Interactive Smoke Test")
st.caption(f"Core version: {getattr(core, '__version__', '?')}")

# ---------- Sidebar inputs ----------
st.sidebar.header("Test Inputs")

flap_mode = st.sidebar.selectbox(
    "Flaps",
    options=[("TO Flaps (35°)", 35), ("Clean (0°)", 0)],
    index=0,
    format_func=lambda x: x[0],
)
flap_deg = flap_mode[1]

gw_lbs = st.sidebar.slider("Gross Weight (lb)", min_value=50000, max_value=74000, value=65000, step=500)
field_elev_ft = st.sidebar.number_input("Field Elevation (ft MSL)", value=39.0, step=1.0)
qnh_inhg = st.sidebar.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
oat_c = st.sidebar.slider("OAT (°C)", min_value=-20, max_value=45, value=15, step=1)

headwind_kts_component = st.sidebar.slider("Head/Tailwind Component (kt, + = headwind, - = tailwind)", -20, 20, 0, 1)
runway_slope = st.sidebar.number_input("Runway Slope (%)", value=0.0, step=0.1, format="%.1f")

tora_ft = st.sidebar.number_input("Available TORA (ft)", value=8000, step=500)
asda_ft = st.sidebar.number_input("Available ASDA (ft)", value=8000, step=500)

st.sidebar.markdown("---")
allow_ab = st.sidebar.checkbox("Allow Afterburner (for derate solve)", value=False)
do_derate = st.sidebar.checkbox("Compute Derate", value=True)

# ---------- Data presence check ----------
st.subheader("Data Files Status")
data_dir = os.path.join(os.path.dirname(__file__), "data")
st.write("Data directory:", f"`{data_dir}`")

required = ["f14_perf.csv"]
optional = [
    "f14_perf_calibrated_SL_overlay.csv",
    "calibration_sl_summary.csv",
    "f110_tff_model.csv",
    "f110_ff_to_rpm_knots.csv",
    "derate_config.json",
    "dcs_airports.csv",
]
cols = st.columns(2)
with cols[0]:
    st.markdown("**Required**")
    for f in required:
        p = os.path.join(data_dir, f)
        st.write(("✅" if os.path.isfile(p) else "❌"), f)
with cols[1]:
    st.markdown("**Optional**")
    for f in optional:
        p = os.path.join(data_dir, f)
        st.write(("✅" if os.path.isfile(p) else "⚪"), f)

st.markdown("---")

# ---------- Run the core ----------
st.subheader("Run Core: plan_takeoff_with_optional_derate()")
run_btn = st.button("Compute")
if run_btn:
    try:
        result = core.plan_takeoff_with_optional_derate(
            flap_deg=flap_deg,
            gw_lbs=float(gw_lbs),
            field_elev_ft=float(field_elev_ft),
            qnh_inhg=float(qnh_inhg),
            oat_c=float(oat_c),
            headwind_kts_component=float(headwind_kts_component),
            runway_slope=float(runway_slope),
            tora_ft=int(tora_ft),
            asda_ft=int(asda_ft),
            allow_ab=bool(allow_ab),
            do_derate=bool(do_derate),
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Inputs (resolved)**")
            st.json(result.get("inputs", {}))
            st.markdown("**Baseline MIL Performance**")
            st.json(result.get("baseline_MIL", {}))

        with right:
            st.markdown("**Derate Result**")
            der = result.get("derate")
            if der is None:
                st.warning("No derate result available (missing optional files or do_derate=False).")
            else:
                st.success("Derate calculation available.")
                st.json(der)

    except Exception as e:
        st.error(f"Smoke test failed: {e}")
