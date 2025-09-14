# app.py — Streamlit UI for DCS F-14B takeoff performance (two-file structure)

from __future__ import annotations
import math
from typing import Tuple

import pandas as pd
import streamlit as st

from f14_takeoff_core import (
    load_perf_csv, compute_takeoff, trim_anu,
    hpa_to_inhg, parse_wind_entry,
    AEO_VR_FRAC, AEO_VR_FRAC_FULL,
    detect_length_text_to_ft, detect_elev_text_to_ft
)

st.set_page_config(page_title="DCS F-14B Takeoff", page_icon="✈️", layout="wide")

# ------------------------------ cached loaders ------------------------------
@st.cache_data
def load_runways(path_primary="dcs_airports_expanded.csv", path_alt="data/dcs_airports_expanded.csv") -> pd.DataFrame:
    for p in (path_primary, path_alt):
        try:
            df = pd.read_csv(p)
            df["runway_label"] = df["airport_name"] + " " + df["runway_end"].astype(str) + " (" + df["runway_pair"].astype(str) + ")"
            return df
        except Exception:
            continue
    st.error("dcs_airports_expanded.csv not found.")
    st.stop()

@st.cache_data
def load_intersections(path_primary="intersections.csv", path_alt="data/intersections.csv") -> pd.DataFrame:
    for p in (path_primary, path_alt):
        try:
            df = pd.read_csv(p)
            # normalize key fields
            for c in ("map","airport_name","runway_end","intersection_id"):
                if c in df.columns:
                    df[c] = df[c].astype(str)
            # ensure numeric
            for c in ("tora_ft","toda_ft","asda_ft","distance_from_threshold_ft"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception:
            continue
    # optional file — return empty frame if missing
    return pd.DataFrame(columns=[
        "map","airport_name","runway_pair","runway_end",
        "intersection_id","tora_ft","toda_ft","asda_ft","distance_from_threshold_ft","notes"
    ])

# ------------------------------ load data ------------------------------
perfdb = load_perf_csv("f14_perf.csv")
rwy_db = load_runways()
ix_db  = load_intersections()

# ------------------------------ UI ------------------------------
st.title("DCS F-14B Takeoff — FAA-Based Model (two-file)")

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Theatre", sorted(rwy_db["map"].unique()))
    df_t = rwy_db[rwy_db["map"] == theatre]
    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]
    rwy_label = st.selectbox("Runway End", list(df_a["runway_label"]))
    rwy = df_a[df_a["runway_label"] == rwy_label].iloc[0]

    base_tora = float(rwy["tora_ft"]); base_toda = float(rwy["toda_ft"]); base_asda = float(rwy["asda_ft"])
    elev_ft = float(rwy["threshold_elev_ft"]); hdg = float(rwy["heading_deg"])
    slope_pct = float(rwy.get("slope_percent", 0.0) or 0.0)

    # Intersection selector (only if we have rows for this runway end)
    st.caption("Intersection (if available for this runway end)")
    df_ix = ix_db[(ix_db["map"] == theatre) & (ix_db["airport_name"] == airport) & (ix_db["runway_end"].astype(str) == str(rwy["runway_end"]))]
    use_ix = False
    if not df_ix.empty:
        inter_opts = ["— Full length —"] + [f'{row["intersection_id"]} (TORA {int(row["tora_ft"]):,} ft)' for _, row in df_ix.iterrows()]
        sel = st.selectbox("Intersection", inter_opts, index=0)
        if sel != "— Full length —":
            use_ix = True
            row = df_ix.iloc[inter_opts.index(sel) - 1]
            base_tora = float(row["tora_ft"])
            base_toda = float(row["toda_ft"])
            base_asda = float(row["asda_ft"])

    # Manual override block
    st.divider()
    st.caption("Manual runway override (optional)")
    manual_len = st.text_input("Runway available (e.g., 7200, 1.2nm, 7200ft)", value="")
    manual_elev = st.text_input("Field elevation (ft)", value="")
    if manual_len.strip():
        try:
            v = detect_length_text_to_ft(manual_len)
            base_tora = base_toda = base_asda = max(0.0, float(v))
        except Exception:
            st.warning("Could not parse runway length. Examples: 7200, 7200ft, 1.1nm")
    if manual_elev.strip():
        try:
            elev_ft = detect_elev_text_to_ft(manual_elev)
        except Exception:
            st.warning("Could not parse elevation (feet).")

    # Shorten available
    st.caption("Shorten available runway")
    sh_val = st.number_input("Value", min_value=0.0, value=0.0, step=50.0, key="sh_val")
    sh_unit = st.selectbox("Units", ["ft", "NM"], index=0, key="sh_unit")
    shorten_total = float(sh_val) if sh_unit == "ft" else float(sh_val) * 6076.12

    # Show current effective declared distances
    cA, cB = st.columns(2)
    with cA:
        st.metric("TORA (ft)", f"{base_tora:,.0f}")
        st.metric("TODA (ft)", f"{base_toda:,.0f}")
    with cB:
        st.metric("ASDA (ft)", f"{base_asda:,.0f}")
        st.metric("Elev (ft)", f"{elev_ft:,.0f}")

    st.header("Weather")
    oat_c = float(st.number_input("OAT (°C)", value=15.0, step=1.0))
    qnh_val = float(st.number_input("QNH value", value=29.92, step=0.01, format="%.2f"))
    qnh_unit = st.selectbox("QNH Units", ["inHg", "hPa"], index=0)
    qnh_inhg = float(qnh_val) if qnh_unit == "inHg" else hpa_to_inhg(float(qnh_val))

    wind_units = st.selectbox("Wind Units", ["kts", "m/s"], index=0)
    wind_entry = st.text_input("Wind (DIR@SPD)", placeholder=f"{int(hdg):03d}@00")
    parsed = parse_wind_entry(wind_entry, wind_units)
    if parsed is None and (wind_entry or "") != "":
        st.warning("Enter wind as DDD@SS, DDD/SS, or DDD SS. Example: 180@12")
        wind_dir, wind_spd = float(hdg), 0.0
    else:
        wind_dir, wind_spd = (parsed if parsed is not None else (float(hdg), 0.0))
    wind_policy = st.selectbox("Wind Policy", ["None", "50/150"], index=0,
        help="‘50/150’ = 50% headwind credit, 150% tailwind penalty (typical airline rule).")

    st.header("Weight & Config")
    mode = st.radio("Weight entry", ["Direct GW", "Fuel + Stores"], index=0)
    if mode == "Direct GW":
        gw = float(st.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=500.0))
    else:
        empty_w   = float(st.number_input("Empty weight (lb)",  min_value=38000.0, max_value=46000.0, value=41780.0, step=50.0))
        fuel_lb   = float(st.number_input("Internal fuel (lb)", min_value=0.0,     max_value=20000.0, value=8000.0,  step=100.0))
        ext_tanks = int(st.selectbox("External tanks (267 gal)", [0,1,2], index=0))
        aim9      = int(st.slider("AIM-9 count", 0, 2, 0))
        aim7      = int(st.slider("AIM-7 count", 0, 4, 0))
        aim54     = int(st.slider("AIM-54 count", 0, 6, 0))
        lantirn   = bool(st.checkbox("LANTIRN pod", value=False))
        wcalc = empty_w + fuel_lb + ext_tanks * 1900.0 + aim9 * 190.0 + aim7 * 510.0 + aim54 * 1000.0 + (440.0 if lantirn else 0.0)
        gw = float(st.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(wcalc), step=10.0))

    flap_mode = st.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    st.header("Thrust")
    thrust_mode = st.radio("Mode", ["Auto-Select", "Manual Derate", "MIL", "AB"], index=0)
    derate_n1 = 98.0
    if thrust_mode == "Manual Derate":
        # floors aligned with core rules
        if flap_mode == "UP":
            floor = 90.0
        elif flap_mode == "FULL":
            floor = 100.0
        else:
            floor = 90.0
        st.caption(f"Derate floor by flap: {floor:.0f}% N1 (MIL)")
        derate_n1 = float(st.slider("Target N1 % (MIL)", min_value=float(int(floor)), max_value=100.0, value=max(95.0, float(int(floor))), step=1.0))

    with st.expander("Advanced / Calibration", expanded=False):
        calib = st.radio("Model calibration", ["FAA-conservative", "DCS-calibrated"], index=1,
                         help=("FAA: OEI factor 1.20 (conservative). "
                               "DCS: OEI factor 1.15 (tuned to hot/high tests)."))
        if calib == "DCS-calibrated":
            st.session_state["AEO_CAL_FACTOR"] = 1.00
            st.session_state["OEI_AGD_FACTOR"] = 1.15
        else:
            st.session_state["AEO_CAL_FACTOR"] = 1.00
            st.session_state["OEI_AGD_FACTOR"] = 1.20

    st.header("Compliance Mode")
    compliance_mode = st.radio("How should limits be checked?", ["Regulatory (OEI §121.189)", "AEO Practical"], index=0,
                               help=("Regulatory: engine-out continue distance is limiting. "
                                     "AEO Practical: uses all-engines continue distance, matching typical DCS tests."))

# ------------------------------ autorun compute ------------------------------
ready = "gw" in locals() and isinstance(gw, (int, float)) and gw >= 40000.0

if ready:
    # Apply current calibration to core globals (simple module-level override)
    import f14_takeoff_core as core
    core.AEO_CAL_FACTOR = float(st.session_state.get("AEO_CAL_FACTOR", 1.00))
    core.OEI_AGD_FACTOR = float(st.session_state.get("OEI_AGD_FACTOR", 1.20))

    res = compute_takeoff(
        perfdb,
        float(hdg), float(base_tora), float(base_toda), float(base_asda),
        float(elev_ft), float(slope_pct), float(shorten_total),
        float(oat_c), float(qnh_inhg),
        float(wind_spd), float(wind_dir), str(wind_units), str(wind_policy),
        float(gw), str(flap_mode),
        ("DERATE" if thrust_mode=="Manual Derate" else thrust_mode), float(derate_n1)
    )

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.subheader("V-Speeds")
        st.metric("V1 (kt)", f"{res.v1:.0f}")
        st.metric("Vr (kt)", f"{res.vr:.0f}")
        st.metric("V2 (kt)", f"{res.v2:.0f}")
        st.metric("Vs (kt)", f"{res.vs:.0f}" if math.isfinite(res.vs) else "—")
    with c2:
        st.subheader("Settings")
        thrust_label = "AFTERBURNER" if res.thrust_text.upper().startswith("AFTERBURNER") else ("MIL" if res.n1_pct >= 100.0 else "DERATE")
        st.metric("Flaps", res.flap_text)
        st.metric("Thrust", f"{thrust_label} ({res.n1_pct:.0f}% N1)")
        flap_deg_out = 0 if res.flap_text.upper().startswith("UP") else (40 if res.flap_text.upper().startswith("FULL") else 20)
        st.metric("Trim (ANU)", f"{trim_anu(float(gw), flap_deg_out):.1f}")
    with c3:
        st.subheader("Runway distances")
        st.metric("Stop distance (ft)", f"{res.asd_ft:.0f}")
        st.metric("Continue distance (engine-out, regulatory) (ft)", f"{res.agd_reg_oei_ft:.0f}")
        st.metric("Continue distance (all engines) (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")

        # Required + margins by mode
        tora_eff = max(0.0, float(base_tora) - float(shorten_total))
        toda_eff = max(0.0, float(base_toda) - float(shorten_total))
        asda_eff_lim = max(0.0, float(base_asda) - float(shorten_total))
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        if compliance_mode.startswith("Regulatory"):
            req = max(res.asd_ft, res.agd_reg_oei_ft)
            asd_ok = res.asd_ft <= asda_eff_lim
            cont_ok = (res.agd_reg_oei_ft <= tod_limit) and (res.agd_reg_oei_ft <= toda_eff)
            limiting = "ASD (stop)" if res.asd_ft >= res.agd_reg_oei_ft else "Engine-out continue"
            asd_margin = asda_eff_lim - res.asd_ft
            cont_margin = min(tod_limit, toda_eff) - res.agd_reg_oei_ft
        else:
            req = max(res.asd_ft, res.agd_aeo_liftoff_ft)
            asd_ok = res.asd_ft <= asda_eff_lim
            cont_ok = (res.agd_aeo_liftoff_ft <= tod_limit) and (res.agd_aeo_liftoff_ft <= toda_eff)
            limiting = "ASD (stop)" if res.asd_ft >= res.agd_aeo_liftoff_ft else "All-engines continue"
            asd_margin = asda_eff_lim - res.asd_ft
            cont_margin = min(tod_limit, toda_eff) - res.agd_aeo_liftoff_ft

        ok = asd_ok and cont_ok
        st.metric("Required runway (based on mode) (ft)", f"{req:.0f}")
    with c4:
        st.subheader("Availability")
        st.metric("Runway available (ft)", f"{res.avail_ft:.0f}")
        st.metric("Limiting factor", limiting)
        st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
        st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
        st.caption("Tailwind > 10 kt or crosswind > 30 kt → NOT AUTHORIZED.")

        req_margin = min(asd_margin, cont_margin)
        if ok:
            st.success(f"COMPLIANT — Margin {req_margin:.0f} ft (ASD margin {asd_margin:.0f}, continue margin {cont_margin:.0f}).")
        else:
            st.error(f"NOT AUTHORIZED — Short by {-req_margin:.0f} ft (ASD margin {asd_margin:.0f}, continue margin {cont_margin:.0f}).")
            st.caption(f"TOD limit: {tod_limit:.0f} ft | ASDA: {asda_eff_lim:.0f} ft | Mode: {compliance_mode}")

    st.markdown("---")
    st.subheader("All-engines takeoff estimates (for DCS comparison)")
    e1, e2 = st.columns(2)
    vr_frac = AEO_VR_FRAC_FULL if res.flap_text.upper().startswith("FULL") else AEO_VR_FRAC
    with e1:
        st.metric("Vr ground roll (ft)", f"{res.agd_aeo_liftoff_ft * vr_frac:.0f}")
    with e2:
        st.metric("Liftoff distance to 35 ft (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")
    st.caption("AEO estimates shown for DCS. Regulatory checks assume an engine-out per 14 CFR 121.189.")

else:
    st.info("Select fuel/stores (or enter a valid gross weight) to compute performance.")

# ------------------------------ footer ------------------------------
st.caption("Wind Policy 50/150: apply 50% of headwind as credit, 150% of tailwind as penalty when adjusting distances. Common airline rule of thumb.")
