# ============================================================
# f14_takeoff_app.py — Streamlit UI for F-14 Toolkit
# Locked UI spec + wired to perf_core_v2.py
# ============================================================

from __future__ import annotations
import os
import math
import streamlit as st
import pandas as pd

import perf_core_v2 as core

APP_VERSION = "v1.2.1-locked"


# ===============================
# Helpers
# ===============================
def _ensure_float(x, default=0.0):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _data_dir():
    return os.path.join(os.getcwd(), "data")


def _load_dcs_airports():
    path = os.path.join(_data_dir(), "dcs_airports.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = cols.get("airport") or cols.get("name") or list(df.columns)[0]
    rwy_col = cols.get("runway") or cols.get("rwy")
    tora_col = cols.get("tora_ft") or cols.get("tora")
    asda_col = cols.get("asda_ft") or cols.get("asda")
    elev_col = cols.get("elevation_ft") or cols.get("elev_ft")
    out = []
    for _, row in df.iterrows():
        out.append({
            "airport": str(row.get(name_col, "")),
            "runway": str(row.get(rwy_col, "")) if rwy_col else "RWY",
            "elev_ft": _ensure_float(row.get(elev_col, 0.0)),
            "tora_ft": _ensure_float(row.get(tora_col, row.get(asda_col, 0.0))),
            "asda_ft": _ensure_float(row.get(asda_col, row.get(tora_col, 0.0))),
        })
    return pd.DataFrame(out)


def _divider(label=None):
    st.markdown("---")
    if label:
        st.caption(label)


# ===============================
# Session state init
# ===============================
def _init_state():
    ss = st.session_state
    ss.setdefault("last_takeoff", None)
    ss.setdefault("last_landing", None)
    ss.setdefault("last_cruise", None)
    ss.setdefault("last_climb", None)
    ss.setdefault("overrides", {"climb_floor_ftpnm": 300.0})
    ss.setdefault("airport_df_cache", None)


# ===============================
# Tabs
# ===============================
def tab_takeoff():
    st.header("Takeoff")
    df = st.session_state.get("airport_df_cache")
    if df is None:
        df = _load_dcs_airports()
        st.session_state["airport_df_cache"] = df

    elev_preset, tora_preset, asda_preset = 0.0, 8000.0, 8000.0
    if df is not None and not df.empty:
        airport = st.selectbox("Airport", sorted(df["airport"].unique()))
        runways = df[df["airport"] == airport]["runway"].tolist()
        runway = st.selectbox("Runway", runways)
        row = df[(df["airport"] == airport) & (df["runway"] == runway)].iloc[0]
        elev_preset, tora_preset, asda_preset = row["elev_ft"], row["tora_ft"], row["asda_ft"]
        st.caption("Loaded from dcs_airports.csv")
    else:
        st.info("dcs_airports.csv not found — enter values manually.")

    col1, col2, col3 = st.columns(3)
    with col1:
        inter_sub = st.number_input("Intersection subtract (ft)", value=0, step=100, min_value=0)
    with col2:
        tora = st.number_input("TORA (ft)", value=int(tora_preset), step=100, min_value=0)
    with col3:
        asda = st.number_input("ASDA (ft)", value=int(asda_preset), step=100, min_value=0)

    tora_final = max(0, tora - inter_sub)
    asda_final = max(0, asda - inter_sub)

    col1, col2, col3 = st.columns(3)
    with col1:
        gw = st.number_input("Gross Weight (lbs)", value=60000, step=500)
        oat = st.number_input("OAT (°C)", value=15, step=1)
        qnh = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    with col2:
        elev = st.number_input("Field Elev (ft MSL)", value=int(elev_preset), step=50)
        head = st.number_input("Head/Tailwind (kt, +HW / -TW)", value=0, step=1)
        flap_mode = st.selectbox("Flaps", ["UP", "MAN", "FULL"], index=1)
    with col3:
        thrust = st.selectbox("Thrust Pref", ["Auto-Select", "Manual MIL", "Manual AB", "Manual DERATE"], index=0)
        derate_pct = None
        if thrust == "Manual DERATE":
            derate_pct = st.slider("Manual Derate (%)", min_value=85, max_value=100, step=1, value=95)

    if st.button("Compute Takeoff", type="primary"):
        try:
            out = core.compute_takeoff({
                "gw_lbs": gw, "oat_c": oat, "qnh_inhg": qnh, "field_elev_ft": elev,
                "headwind_kts_component": head, "tora_ft": tora_final, "asda_ft": asda_final,
                "flap_mode": flap_mode, "thrust_pref": thrust, "manual_derate_pct": derate_pct
            }, overrides=st.session_state["overrides"])
            st.session_state["last_takeoff"] = out
        except Exception as e:
            st.error(f"Takeoff compute failed: {e}")
            st.stop()

    out = st.session_state.get("last_takeoff")
    if out:
        st.subheader("Results (Unfactored vs Factored ×1.10)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Vs", f"{out['Vs_kts']:.0f} kt" if out["Vs_kts"] else "—")
        col2.metric("Vr", f"{out['Vr_kts']:.0f} kt" if out["Vr_kts"] else "—")
        col3.metric("V2", f"{out['V2_kts']:.0f} kt" if out["V2_kts"] else "—")
        col4.metric("Dispatchable", "Yes" if out["Dispatchable"] else "No")
        _divider()
        col1, col2 = st.columns(2)
        col1.metric("ASDR Unfactored", f"{int(out['ASDR_ft']):,} ft")
        col1.metric("ASDR Factored (×1.10)", f"{int(out['ASDR_ft']*1.10):,} ft")
        col2.metric("TODR Unfactored", f"{int(out['TODR_OEI_35ft_ft']):,} ft")
        col2.metric("TODR Factored (×1.10)", f"{int(out['TODR_OEI_35ft_ft']*1.10):,} ft")
        st.caption(f"Thrust: {out['ResolvedThrustLabel']} | Flaps: {out['Flaps']} | GW: {int(out['GW_lbs']):,} lb")


def tab_landing():
    st.header("Landing")
    flap = st.selectbox("Flaps", ["FULL", "MAN", "UP"], index=0)
    gw = st.number_input("Gross Weight (lbs)", value=52000, step=500)
    pa = st.number_input("Pressure Altitude (ft)", value=0, step=100)
    oat = st.number_input("OAT (°C)", value=15, step=1)
    hw = st.number_input("Headwind (kt)", value=0, step=1)

    if st.button("Compute Landing", type="primary"):
        try:
            out = core.compute_landing({
                "flap_setting": flap, "gross_weight_lbs": gw,
                "pressure_alt_ft": pa, "oat_c": oat, "headwind_kt": hw
            })
            st.session_state["last_landing"] = out
        except Exception as e:
            st.error(f"Landing compute failed: {e}")
            st.stop()

    out = st.session_state.get("last_landing")
    if out:
        st.subheader("Results (Unfactored vs Factored ×1.67)")
        col1, col2 = st.columns(2)
        col1.metric("Ground Roll (Unfactored)", f"{int(out['ldr_unfactored_ft']):,} ft")
        col2.metric("Ground Roll (FAA ×1.67)", f"{int(out['ldr_unfactored_ft']*1.67):,} ft")
        if out.get("Vref_kts"):
            st.metric("Vref", f"{out['Vref_kts']:.0f} kt")


def tab_cruise():
    st.header("Cruise")
    gw = st.number_input("Gross Weight (lbs)", value=55000, step=500)
    di = st.number_input("Drag Index (DI)", value=50, step=1, min_value=0)

    if st.button("Compute Cruise", type="primary"):
        try:
            out = core.compute_cruise({"gross_weight_lbs": gw, "drag_index": di})
            st.session_state["last_cruise"] = out
        except Exception as e:
            st.error(f"Cruise compute failed: {e}")
            st.stop()

    out = st.session_state.get("last_cruise")
    if out:
        st.subheader("Optimum Cruise")
        col1, col2 = st.columns(2)
        col1.metric("Optimum Altitude", f"{int(out['optimum_alt_ft']):,} ft")
        col2.metric("Optimum Mach", f"{out['optimum_mach']:.2f}")


def tab_climb():
    st.header("Climb")
    profile = st.selectbox("Profile", ["Economy", "Interceptor"], index=0)
    respect_250 = st.checkbox("Respect 250 KIAS <10,000 ft", value=True)
    gw = st.number_input("Gross Weight (lbs)", value=55000, step=500)

    if st.button("Compute Climb", type="primary"):
        try:
            out = core.compute_climb({"profile": profile, "respect_250": respect_250, "gross_weight_lbs": gw})
            st.session_state["last_climb"] = out
        except Exception as e:
            st.error(f"Climb compute failed: {e}")
            st.stop()

    out = st.session_state.get("last_climb")
    if out:
        st.subheader(f"Climb Schedule — {out['profile']}")
        st.write(out.get("schedule"))


def tab_diagnostics():
    st.header("Diagnostics")
    st.json({
        "app_version": APP_VERSION,
        "core_version": core.__version__,
    })


# ===============================
# Main
# ===============================
def main():
    st.set_page_config(page_title="F-14 Performance Toolkit", layout="wide")
    _init_state()
    st.title("F-14 Performance Toolkit")
    tabs = st.tabs(["Takeoff", "Landing", "Cruise", "Climb", "Diagnostics"])
    with tabs[0]:
        tab_takeoff()
    with tabs[1]:
        tab_landing()
    with tabs[2]:
        tab_cruise()
    with tabs[3]:
        tab_climb()
    with tabs[4]:
        tab_diagnostics()


if __name__ == "__main__":
    main()
