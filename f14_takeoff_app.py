# app.py — Streamlit UI for DCS F-14B takeoff performance (two-file structure + optimizer, what-if matrix, exports)

from __future__ import annotations
import math
from typing import Tuple, List, Dict

import pandas as pd
import streamlit as st

from f14_takeoff_core import (
    load_perf_csv, compute_takeoff, trim_anu,
    hpa_to_inhg, parse_wind_entry,
    AEO_VR_FRAC, AEO_VR_FRAC_FULL,
    detect_length_text_to_ft, detect_elev_text_to_ft,
    estimate_ab_multiplier
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
            for c in ("map","airport_name","runway_end","intersection_id"):
                if c in df.columns:
                    df[c] = df[c].astype(str)
            for c in ("tora_ft","toda_ft","asda_ft","distance_from_threshold_ft"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception:
            continue
    return pd.DataFrame(columns=[
        "map","airport_name","runway_pair","runway_end",
        "intersection_id","tora_ft","toda_ft","asda_ft","distance_from_threshold_ft","notes"
    ])

# ------------------------------ load data ------------------------------
perfdb = load_perf_csv("f14_perf.csv")
rwy_db = load_runways()
ix_db  = load_intersections()

# ------------------------------ UI ------------------------------
st.title("DCS F-14B Takeoff — FAA-Based Model")

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

    # Intersection selector (if available)
    st.caption("Intersection (if available for this runway end)")
    df_ix = ix_db[(ix_db["map"] == theatre) & (ix_db["airport_name"] == airport) & (ix_db["runway_end"].astype(str) == str(rwy["runway_end"]))]
    if not df_ix.empty:
        inter_opts = ["— Full length —"] + [f'{row["intersection_id"]} (TORA {int(row["tora_ft"]):,} ft)' for _, row in df_ix.iterrows()]
        sel = st.selectbox("Intersection", inter_opts, index=0)
        if sel != "— Full length —":
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

def _calc(perfdb, flap_mode_s, thrust_mode_s, n1pct):
    import f14_takeoff_core as core
    core.AEO_CAL_FACTOR = float(st.session_state.get("AEO_CAL_FACTOR", 1.00))
    core.OEI_AGD_FACTOR = float(st.session_state.get("OEI_AGD_FACTOR", 1.20))
    return compute_takeoff(
        perfdb,
        float(hdg), float(base_tora), float(base_toda), float(base_asda),
        float(elev_ft), float(slope_pct), float(shorten_total),
        float(oat_c), float(qnh_inhg),
        float(wind_spd), float(wind_dir), str(wind_units), str(wind_policy),
        float(gw), str(flap_mode_s),
        str("DERATE" if thrust_mode_s=="Manual Derate" else thrust_mode_s), float(n1pct)
    )

if ready:
    # current result
    res = _calc(perfdb, flap_mode, thrust_mode, derate_n1)

    # ===== Top cards =====
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

    st.markdown("---")
with st.expander("Data Checker (CSV sanity scan)", expanded=False):
    st.caption("Quick sanity checks for f14_perf.csv, dcs_airports_expanded.csv, and intersections.csv.")

    issues = []

    # --- PERF: duplicate keys
    key_cols = ["model","flap_deg","thrust","gw_lbs","press_alt_ft","oat_c"]
    if all(k in perfdb.columns for k in key_cols):
        dups = (perfdb
                .assign(_cnt=1)
                .groupby(key_cols, dropna=False)["_cnt"].sum()
                .reset_index()
                .query("_cnt > 1"))
        if not dups.empty:
            issues.append(f"Performance table: {len(dups)} duplicate key rows.")
            st.dataframe(dups, use_container_width=True)
        else:
            st.success("Performance table: no duplicate key rows.")
    else:
        issues.append("Performance table: missing expected columns for duplicate check.")

    # --- PERF: missing MAN(20) synthesized? (just a heads up)
    if not (perfdb["flap_deg"] == 20).any():
        st.info("Performance: MAN(20) rows not present; app will synthesize from UP/FULL where possible.")

    # --- AIRPORTS: both runway ends present?
    rw_counts = (rwy_db.groupby(["map","airport_name","runway_pair"])["runway_end"]
                   .nunique().reset_index(name="ends"))
    missing_ends = rw_counts[rw_counts["ends"] < 2]
    if not missing_ends.empty:
        issues.append(f"Airports: {len(missing_ends)} runway pairs have only one end.")
        st.dataframe(missing_ends, use_container_width=True)
    else:
        st.success("Airports: all runway pairs have both ends.")

    # --- INTERSECTIONS: orphaned refs?
    if not ix_db.empty:
        merged = ix_db.merge(
            rwy_db[["map","airport_name","runway_pair","runway_end"]],
            on=["map","airport_name","runway_pair","runway_end"],
            how="left",
            indicator=True
        )
        orphans = merged[merged["_merge"] == "left_only"]
        if not orphans.empty:
            issues.append(f"Intersections: {len(orphans)} rows reference missing airports/runways.")
            st.dataframe(orphans.drop(columns=["_merge"]), use_container_width=True)
        else:
            st.success("Intersections: all rows reference known airports/runways.")
    else:
        st.info("Intersections: file empty or not provided — nothing to check.")

    if issues:
        st.warning("Summary: " + "  •  ".join(issues))
    else:
        st.success("CSV sanity scan: clean ✅")

    # ===== Optimizer & What-ifs =====
    st.markdown("---")
    with st.expander("Optimizer & What-ifs", expanded=True):
        st.caption("Optimizer finds minimum MIL N1 that passes Regulatory (§121.189). If MAN can’t pass, FULL/MIL fallback is shown.")

        # Helper to bisection per flap
        def min_n1_for_flap(flap_label: str) -> Dict:
            low = 90.0 if flap_label != "FULL" else 100.0
            high = 100.0
            best = None
            # Try MIL first at 100
            r_hi = _calc(perfdb, flap_label, "MIL", 100.0)
            # We validate pass under Regulatory only
            tora_eff = max(0.0, float(base_tora) - float(shorten_total))
            toda_eff = max(0.0, float(base_toda) - float(shorten_total))
            asda_eff_lim = max(0.0, float(base_asda) - float(shorten_total))
            clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
            tod_limit = tora_eff + clearway_allow
            pass_hi = (r_hi.asd_ft <= asda_eff_lim) and (r_hi.agd_reg_oei_ft <= toda_eff) and (r_hi.agd_reg_oei_ft <= tod_limit)

            if not pass_hi:
                # Not possible at this flap (escalation will be handled elsewhere)
                return {"flap": flap_label, "possible": False, "n1": None, "res": r_hi}

            # Bisection from floor to 100
            lo, hi = low, 100.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                r_mid = _calc(perfdb, flap_label, "Manual Derate", mid)
                pass_mid = (r_mid.asd_ft <= asda_eff_lim) and (r_mid.agd_reg_oei_ft <= toda_eff) and (r_mid.agd_reg_oei_ft <= tod_limit)
                if pass_mid:
                    hi = mid
                    best = (mid, r_mid)
                else:
                    lo = mid
            if best is None:
                best = (100.0, r_hi)
            return {"flap": flap_label, "possible": True, "n1": float(int(math.ceil(best[0]))), "res": best[1]}

        man = min_n1_for_flap("MANEUVER")
        full = min_n1_for_flap("FULL")
        up   = min_n1_for_flap("UP")

        cols = st.columns(3)
        for i, pack in enumerate([up, man, full]):
            with cols[i]:
                title = f"{pack['flap']} — {'PASS' if pack['possible'] else 'FAIL @100%'}"
                st.markdown(f"**{title}**")
                if pack["possible"]:
                    st.write(f"Min N1 (MIL): **{pack['n1']:.0f}%**")
                r = pack["res"]
                st.write(f"V1/Vr/V2: {r.v1:.0f}/{r.vr:.0f}/{r.v2:.0f} kt")
                st.write(f"Stop / OEI cont / AEO cont: {r.asd_ft:.0f} / {r.agd_reg_oei_ft:.0f} / {r.agd_aeo_liftoff_ft:.0f} ft")

        # Fallback guidance
        if not man["possible"] and full["possible"]:
            st.info("Auto escalation: MAN fails Regulatory; **FULL / MIL** is available.")
        elif not man["possible"] and not full["possible"]:
            st.error("Neither MAN nor FULL meet Regulatory at declared distances (check runway, wind, slope, or weight).")

        # ===== What-if Matrix =====
        st.markdown("---")
        st.caption("What-if matrix: common configs side-by-side (Regulatory check). FULL cannot derate; AB row appears if data exists or safe to approximate.")
        # Precompute controls
        tora_eff = max(0.0, float(base_tora) - float(shorten_total))
        toda_eff = max(0.0, float(base_toda) - float(shorten_total))
        asda_eff_lim = max(0.0, float(base_asda) - float(shorten_total))
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        rows: List[Dict] = []
        flap_list = ["UP", "MANEUVER", "FULL"]
        derate_candidates = [100, 98, 96, 94, 92, 90]

        # Detect AB feasibility per flap (either real rows or via ratio)
        def ab_available_for(flap_deg_label: str) -> bool:
            flap_deg = 0 if flap_deg_label=="UP" else (40 if flap_deg_label=="FULL" else 20)
            sub_ab = perfdb[(perfdb["flap_deg"] == (20 if flap_deg==0 else flap_deg)) & (perfdb["thrust"] == "AFTERBURNER")]
            if not sub_ab.empty:
                return True
            # If no table, we still allow approximation using learned ratio
            try:
                _ = estimate_ab_multiplier(perfdb, (20 if flap_deg==0 else flap_deg))
                return True
            except Exception:
                return False

        for flap in flap_list:
            # MIL (100)
            r_mil = _calc(perfdb, flap, "MIL", 100.0)
            req_reg = max(r_mil.asd_ft, r_mil.agd_reg_oei_ft)
            pass_reg = (r_mil.asd_ft <= asda_eff_lim) and (r_mil.agd_reg_oei_ft <= toda_eff) and (r_mil.agd_reg_oei_ft <= tod_limit)
            rows.append({
                "Flap": flap, "Thrust/N1": "MIL 100%", "V1": f"{r_mil.v1:.0f}", "Vr": f"{r_mil.vr:.0f}", "V2": f"{r_mil.v2:.0f}",
                "Stop ft": int(r_mil.asd_ft), "AEO cont ft": int(r_mil.agd_aeo_liftoff_ft), "OEI cont ft": int(r_mil.agd_reg_oei_ft),
                "Required ft": int(req_reg), "Limiting": ("ASD" if r_mil.asd_ft >= r_mil.agd_reg_oei_ft else "OEI cont"),
                "Regulatory OK": "YES" if pass_reg else "NO"
            })

            # Derate ladder (not for FULL)
            if flap != "FULL":
                for n1 in derate_candidates:
                    if n1 == 100:  # already added as MIL 100
                        continue
                    r_d = _calc(perfdb, flap, "Manual Derate", n1)
                    req_reg = max(r_d.asd_ft, r_d.agd_reg_oei_ft)
                    pass_reg = (r_d.asd_ft <= asda_eff_lim) and (r_d.agd_reg_oei_ft <= toda_eff) and (r_d.agd_reg_oei_ft <= tod_limit)
                    rows.append({
                        "Flap": flap, "Thrust/N1": f"Derate {int(n1)}%", "V1": f"{r_d.v1:.0f}", "Vr": f"{r_d.vr:.0f}", "V2": f"{r_d.v2:.0f}",
                        "Stop ft": int(r_d.asd_ft), "AEO cont ft": int(r_d.agd_aeo_liftoff_ft), "OEI cont ft": int(r_d.agd_reg_oei_ft),
                        "Required ft": int(req_reg), "Limiting": ("ASD" if r_d.asd_ft >= r_d.agd_reg_oei_ft else "OEI cont"),
                        "Regulatory OK": "YES" if pass_reg else "NO"
                    })

            # AB row (if feasible)
            if ab_available_for(flap):
                r_ab = _calc(perfdb, flap, "AB", 100.0)
                req_reg = max(r_ab.asd_ft, r_ab.agd_reg_oei_ft)
                pass_reg = (r_ab.asd_ft <= asda_eff_lim) and (r_ab.agd_reg_oei_ft <= toda_eff) and (r_ab.agd_reg_oei_ft <= tod_limit)
                rows.append({
                    "Flap": flap, "Thrust/N1": "AFTERBURNER", "V1": f"{r_ab.v1:.0f}", "Vr": f"{r_ab.vr:.0f}", "V2": f"{r_ab.v2:.0f}",
                    "Stop ft": int(r_ab.asd_ft), "AEO cont ft": int(r_ab.agd_aeo_liftoff_ft), "OEI cont ft": int(r_ab.agd_reg_oei_ft),
                    "Required ft": int(req_reg), "Limiting": ("ASD" if r_ab.asd_ft >= r_ab.agd_reg_oei_ft else "OEI cont"),
                    "Regulatory OK": "YES" if pass_reg else "NO"
                })

        df_matrix = pd.DataFrame(rows)
        st.dataframe(
            df_matrix.assign(**{
                "Stop ft": df_matrix["Stop ft"].map(lambda x: f"{x:,}"),
                "AEO cont ft": df_matrix["AEO cont ft"].map(lambda x: f"{x:,}"),
                "OEI cont ft": df_matrix["OEI cont ft"].map(lambda x: f"{x:,}"),
                "Required ft": df_matrix["Required ft"].map(lambda x: f"{x:,}")
            }),
            use_container_width=True
        )

        # ===== Exports =====
        st.markdown("---")
        st.caption("Export current result & comparison matrix")
        # Current result (JSON-ish text)
        current_payload = {
            "map": theatre, "airport": airport, "runway_end": str(rwy["runway_end"]),
            "intersection": (sel if not df_ix.empty else "Full length"),
            "declared": {"TORA_ft": base_tora, "TODA_ft": base_toda, "ASDA_ft": base_asda, "elev_ft": elev_ft, "slope_pct": slope_pct},
            "weather": {"OAT_C": oat_c, "QNH_inHg": qnh_inhg, "wind_dir": wind_dir, "wind_spd": wind_spd, "wind_policy": wind_policy},
            "weight_lb": gw,
            "config": {"flaps": res.flap_text, "thrust": res.thrust_text, "N1_pct": res.n1_pct},
            "v_speeds": {"V1": res.v1, "Vr": res.vr, "V2": res.v2, "Vs": res.vs},
            "distances_ft": {"ASD": res.asd_ft, "AEO_liftoff": res.agd_aeo_liftoff_ft, "OEI_reg": res.agd_reg_oei_ft},
            "compliance_mode": compliance_mode
        }
        st.download_button("Download current result (JSON)", data=pd.Series(current_payload).to_json(indent=2),
                           file_name="f14_takeoff_result.json", mime="application/json")

                # ===== Print-friendly Report (HTML) =====
        st.markdown("---")
        st.caption("Print-friendly report (HTML) — includes your current result and the what-if matrix.")
        try:
            matrix_html = df_matrix.to_html(index=False, border=0)
        except Exception:
            matrix_html = "<p>(Matrix unavailable)</p>"

        # simple, clean HTML (inline styles so it prints nicely)
        report_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>F-14B Takeoff Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }}
  h1, h2, h3 {{ margin: 0 0 12px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 12px 24px; }}
  .card {{ padding: 12px 16px; border: 1px solid #e5e7eb; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
  th {{ background: #f9fafb; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
  <h1>F-14B Takeoff Report</h1>
  <p class="mono">{theatre} — {airport} — RWY {str(rwy["runway_end"])} {sel if (not df_ix.empty and sel != "— Full length —") else "(Full length)"} </p>

  <div class="grid">
    <div class="card">
      <h3>Declared / Field</h3>
      <div>TORA: <b>{int(base_tora):,}</b> ft</div>
      <div>TODA: <b>{int(base_toda):,}</b> ft</div>
      <div>ASDA: <b>{int(base_asda):,}</b> ft</div>
      <div>Elevation: <b>{int(elev_ft):,}</b> ft</div>
      <div>Slope: <b>{slope_pct:.2f}%</b></div>
      <div>Shorten: <b>{int(shorten_total):,}</b> ft</div>
    </div>
    <div class="card">
      <h3>Weather</h3>
      <div>OAT: <b>{oat_c:.1f} °C</b> | QNH: <b>{qnh_inhg:.2f} inHg</b></div>
      <div>Wind: <b>{int(wind_dir):03d}@{int(wind_spd)}</b> {wind_units}, policy: <b>{wind_policy}</b></div>
    </div>
    <div class="card">
      <h3>Weight & Config</h3>
      <div>GW: <b>{int(gw):,}</b> lb</div>
      <div>Flaps: <b>{res.flap_text}</b></div>
      <div>Thrust: <b>{res.thrust_text}</b> ({res.n1_pct:.0f}% N1)</div>
      <div>Trim: <b>{trim_anu(float(gw), flap_deg_out):.1f} ANU</b></div>
    </div>
    <div class="card">
      <h3>V-Speeds</h3>
      <div>V1/Vr/V2: <b>{res.v1:.0f}</b> / <b>{res.vr:.0f}</b> / <b>{res.v2:.0f}</b> kt</div>
      <div>Vs: <b>{res.vs:.0f}</b> kt</div>
    </div>
    <div class="card">
      <h3>Distances</h3>
      <div>Stop (ASD): <b>{int(res.asd_ft):,}</b> ft</div>
      <div>Cont (OEI reg): <b>{int(res.agd_reg_oei_ft):,}</b> ft</div>
      <div>Cont (AEO): <b>{int(res.agd_aeo_liftoff_ft):,}</b> ft</div>
    </div>
    <div class="card">
      <h3>All-engines (for DCS)</h3>
      <div>Vr ground roll: <b>{int(res.agd_aeo_liftoff_ft * vr_frac):,}</b> ft</div>
      <div>Liftoff to 35 ft: <b>{int(res.agd_aeo_liftoff_ft):,}</b> ft</div>
    </div>
  </div>

  <h2 style="margin-top: 18px;">What-if Matrix</h2>
  {matrix_html}
</body>
</html>
        """.strip()

        st.download_button(
            "Download print report (HTML)",
            data=report_html,
            file_name="f14_takeoff_report.html",
            mime="text/html"
        )

                # ===== Permalink / Share =====
        st.markdown("---")
        st.caption("Permalink: update the URL with your current inputs so you can share or bookmark.")

        def _qp_bool(x: bool) -> str:
            return "1" if bool(x) else "0"

        permalink_params = {
            "map": theatre,
            "apt": airport,
            "rwy": str(rwy["runway_end"]),
            "ix": (sel if (not df_ix.empty and sel != "— Full length —") else ""),
            "tora": f"{base_tora:.0f}",
            "toda": f"{base_toda:.0f}",
            "asda": f"{base_asda:.0f}",
            "elev": f"{elev_ft:.0f}",
            "slope": f"{slope_pct:.2f}",
            "shorten": f"{shorten_total:.0f}",
            "oat": f"{oat_c:.1f}",
            "qnh": f"{qnh_inhg:.2f}",
            "wunits": wind_units,
            "wdir": f"{wind_dir:.0f}",
            "wspd": f"{wind_spd:.0f}",
            "wpol": wind_policy,
            "gw": f"{gw:.0f}",
            "flaps": flap_mode,
            "thrust": thrust_mode,
            "n1": f"{derate_n1:.0f}",
            "mode": ("REG" if compliance_mode.startswith("Regulatory") else "AEO"),
            # calibration flags
            "cal_aeo": f'{st.session_state.get("AEO_CAL_FACTOR", 1.00):.2f}',
            "cal_oei": f'{st.session_state.get("OEI_AGD_FACTOR", 1.20):.2f}',
        }

        col_pl_a, col_pl_b = st.columns([1,1])
        with col_pl_a:
            if st.button("Update URL with current settings"):
                try:
                    st.query_params.update(permalink_params)
                    st.success("URL updated — copy it from the address bar.")
                except Exception as e:
                    st.warning(f"Could not update URL: {e}")
        with col_pl_b:
            # Handy read-only view
            st.text_input("Quick-copy query", value="&".join([f"{k}={v}" for k,v in permalink_params.items()]), label_visibility="collapsed")

        # Matrix CSV
        st.download_button("Download what-if matrix (CSV)", data=df_matrix.to_csv(index=False),
                           file_name="f14_takeoff_matrix.csv", mime="text/csv")

else:
    st.info("Select fuel/stores (or enter a valid gross weight) to compute performance.")

# ------------------------------ footer ------------------------------
st.caption("Wind Policy 50/150: apply 50% of headwind as credit, 150% of tailwind as penalty when adjusting distances. Common airline rule of thumb.")
