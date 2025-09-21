# test_app.py — v1.6.0 (drag increments + debug toggle)
import streamlit as st
import pandas as pd
import f14_takeoff_core as core

st.set_page_config(page_title="F-14 Toolkit — Takeoff (Drag-Increment Model)", layout="wide")
st.title("F-14 Takeoff — Drag Increments & 1% Derate Search")

with st.sidebar:
    gw_lbs = st.number_input("Gross Weight (lb)", value=65000, min_value=50000, max_value=74000, step=500)
    field_elev_ft = st.number_input("Field Elevation (ft)", value=39, step=1)
    qnh_inhg = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    oat_c = st.number_input("OAT (°C)", value=15, step=1)
    tora_ft = st.number_input("TORA (ft)", value=8000, step=500)
    asda_ft = st.number_input("ASDA (ft)", value=8000, step=500)
    allow_ab = st.checkbox("Include Afterburner candidate", value=False)
    show_debug = st.checkbox("Show debug (drag increments, thrust, CL/CD)", value=True)

if st.button("Compute"):
    res = core.plan_takeoff_with_optional_derate(
        flap_deg=0,  # planner searches [0,20,40]
        gw_lbs=float(gw_lbs),
        field_elev_ft=float(field_elev_ft),
        qnh_inhg=float(qnh_inhg),
        oat_c=float(oat_c),
        tora_ft=int(tora_ft),
        asda_ft=int(asda_ft),
        allow_ab=bool(allow_ab),
        debug=bool(show_debug),
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
        if show_debug and best.get("_debug"):
            dbg = best["_debug"]
            st.subheader("Debug — Drag & Thrust")
            dbg_rows = [
                ["cd0_base", dbg.get("cd0_base")],
                ["cd0_inc_config", dbg.get("cd0_inc_config")],
                ["cd0_inc_stores", dbg.get("cd0_inc_stores")],
                ["cd0_total", dbg.get("cd0_total")],
                ["clmax", dbg.get("clmax")],
                ["CL_used", dbg.get("CL_used")],
                ["CD_used", dbg.get("CD_used")],
                ["T_per_lbf_final", dbg.get("T_per_lbf_final")],
                ["T_tot_N", dbg.get("T_tot_N")],
                ["D_N", dbg.get("D_N")],
                ["excess_N", dbg.get("excess_N")],
                ["grad_ft_per_nm", dbg.get("grad_ft_per_nm")],
            ]
            st.table(pd.DataFrame(dbg_rows, columns=["Metric", "Value"]))
    else:
        st.error(res.get("verdict", "NOT_DISPATCHABLE"))
