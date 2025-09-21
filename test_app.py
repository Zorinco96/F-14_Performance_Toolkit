# test_app.py — v1.4.0 (comparison view)
import streamlit as st
import pandas as pd
import f14_takeoff_core as core

st.set_page_config(page_title="F-14 Toolkit — Comparison", layout="wide")
st.title("F-14 Takeoff — DERATE vs MIL (±AB)")

with st.sidebar:
    flap_label = st.selectbox("Flaps", ["UP (0°)", "MANEUVER (20°)", "FULL (40°)"])
    flap_map = {"UP (0°)":0, "MANEUVER (20°)":20, "FULL (40°)":40}
    flap_deg = flap_map[flap_label]

    gw_lbs = st.number_input("Gross Weight (lb)", value=65000, min_value=50000, max_value=74000, step=500)
    field_elev_ft = st.number_input("Field Elevation (ft)", value=39, step=1)
    qnh_inhg = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    oat_c = st.number_input("OAT (°C)", value=15, step=1)
    tora_ft = st.number_input("TORA (ft)", value=8000, step=500)
    asda_ft = st.number_input("ASDA (ft)", value=8000, step=500)
    allow_ab = st.checkbox("Include Afterburner candidate", value=False)

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
        debug=False,
        compare_all=True,
    )

    tried = res.get("tried", [])
    if not tried:
        st.error("No candidates returned.")
    else:
        rows = []
        for c in tried:
            rows.append({
                "Mode": f"{c['thrust_mode']}{' '+str(c['derate_pct'])+'%' if c['thrust_mode']=='DERATE' else ''}",
                "Dispatch": "✅" if c["dispatchable"] else "❌",
                "Limiter": c["limiter"],
                "AEO ft/NM": round(c["aeo_grad_ft_per_nm"]),
                "ASD (ft)": round(c["asd_ft"]),
                "TODR (ft)": round(c["todr_ft"]),
                "V1/Vr/V2": f"{round(c['v1_kts'])}/{round(c['vr_kts'])}/{round(c['v2_kts'])}"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Planner Verdict")
    if res.get("best"):
        b = res["best"]
        st.success(f"Best: {b['thrust_mode']}{' '+str(b['derate_pct'])+'%' if b['thrust_mode']=='DERATE' else ''} — limiter {b['limiter']} — {round(b['aeo_grad_ft_per_nm'])} ft/NM")
    else:
        st.error(res.get("verdict", "NOT_DISPATCHABLE"))
