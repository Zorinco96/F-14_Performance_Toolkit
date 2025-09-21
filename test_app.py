# test_app.py — v1.5.0 (1% DERATE search + TSV)
import streamlit as st
import pandas as pd
import f14_takeoff_core as core

st.set_page_config(page_title="F-14 Toolkit — DERATE Search", layout="wide")
st.title("F-14 Takeoff — 1% DERATE Search with Flap Priority")

with st.sidebar:
    gw_lbs = st.number_input("Gross Weight (lb)", value=65000, min_value=50000, max_value=74000, step=500)
    field_elev_ft = st.number_input("Field Elevation (ft)", value=39, step=1)
    qnh_inhg = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    oat_c = st.number_input("OAT (°C)", value=15, step=1)
    tora_ft = st.number_input("TORA (ft)", value=8000, step=500)
    asda_ft = st.number_input("ASDA (ft)", value=8000, step=500)
    allow_ab = st.checkbox("Include Afterburner candidate", value=False)

if st.button("Compute"):
    res = core.plan_takeoff_with_optional_derate(
        flap_deg=0,  # starting flap selection no longer used; planner searches [0,20,40] internally
        gw_lbs=float(gw_lbs),
        field_elev_ft=float(field_elev_ft),
        qnh_inhg=float(qnh_inhg),
        oat_c=float(oat_c),
        tora_ft=int(tora_ft),
        asda_ft=int(asda_ft),
        allow_ab=bool(allow_ab),
        debug=False,
        compare_all=True,
        search_1pct=True,
    )

    tried = res.get("tried", [])
    rows = []
    for c in tried:
        rows.append({
            "Flaps_deg": c["flap_deg"],
            "Mode": f"{c['thrust_mode']}{' '+str(c['derate_pct'])+'%' if c['thrust_mode']=='DERATE' else ''}",
            "Dispatch": "YES" if c["dispatchable"] else "NO",
            "Limiter": c["limiter"],
            "AEO_ft_per_NM": round(c["aeo_grad_ft_per_nm"]),
            "ASD_ft": round(c["asd_ft"]),
            "TODR_ft": round(c["todr_ft"]),
            "V1_Vr_V2": f"{round(c['v1_kts'])}/{round(c['vr_kts'])}/{round(c['v2_kts'])}"
        })
    if rows:
        df = pd.DataFrame(rows, columns=["Flaps_deg","Mode","Dispatch","Limiter","AEO_ft_per_NM","ASD_ft","TODR_ft","V1_Vr_V2"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.subheader("Copy/Paste (TSV)")
        st.code(df.to_csv(sep="\t", index=False), language="text")

    st.divider()
    best = res.get("best")
    if best:
        st.success(f"BEST: Flaps {best['flap_deg']} — {best['thrust_mode']}{' '+str(best['derate_pct'])+'%' if best['thrust_mode']=='DERATE' else ''} — {round(best['aeo_grad_ft_per_nm'])} ft/NM — {best['limiter']}")
    else:
        st.error(res.get("verdict", "NOT_DISPATCHABLE"))
