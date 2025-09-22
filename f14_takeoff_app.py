
# ============================================================
# f14_takeoff_app.py — UI per LOCK SPEC (v1.2.1-UIlock)
# ============================================================
from __future__ import annotations
import os, sys, json, math
from typing import Dict, Any

import streamlit as st
import perf_core_v2 as core_v2

APP_VERSION = "v1.2.1-UIlock"

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
    """Load airport/runway db with graceful schema detection."""
    import pandas as pd
    path = os.path.join(_data_dir(), "dcs_airports.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = cols.get("airport") or cols.get("name") or cols.get("icao") or cols.get("field") or list(df.columns)[0]
    rwy_col  = cols.get("runway") or cols.get("rwy") or cols.get("runway_id") or None
    len_col  = cols.get("length_ft") or cols.get("runway_length_ft") or cols.get("length") or cols.get("len_ft") or None
    tora_col = cols.get("tora_ft") or cols.get("tora") or None
    asda_col = cols.get("asda_ft") or cols.get("asda") or None
    elev_col = cols.get("elevation_ft") or cols.get("elev_ft") or cols.get("elevation") or None

    out = []
    for _, row in df.iterrows():
        name = row.get(name_col, "")
        rwy = row.get(rwy_col, "") if rwy_col else ""
        elev = _ensure_float(row.get(elev_col, 0.0)) if elev_col else 0.0
        tora = row.get(tora_col, None)
        asda = row.get(asda_col, None)
        length = row.get(len_col, None)
        tora_ft = _ensure_float(tora if tora is not None else length, 0.0)
        asda_ft = _ensure_float(asda if asda is not None else length, 0.0)
        if tora_ft <= 0 and asda_ft <= 0:
            continue
        out.append({
            "airport": str(name),
            "runway": str(rwy) if rwy else "RWY",
            "elev_ft": elev,
            "tora_ft": tora_ft,
            "asda_ft": asda_ft
        })
    if not out:
        return None
    import pandas as pd
    return pd.DataFrame(out)

def _kv(label, value, suffix=""):
    st.metric(label, f"{value}{suffix}" if value is not None else "-")

def _divider(label=None):
    st.markdown("---")
    if label: st.caption(label)

def _init_state():
    ss = st.session_state
    ss.setdefault("last_takeoff", None)
    ss.setdefault("last_landing", None)
    ss.setdefault("last_cruise", None)
    ss.setdefault("last_climb", None)
    ss.setdefault("overrides", {"climb_floor_ftpnm": 300.0})
    ss.setdefault("airport_df_cache", None)

def tab_takeoff():
    st.header("Takeoff")
    with st.expander("Airport / Runway", expanded=True):
        df = st.session_state.get("airport_df_cache")
        if df is None:
            df = _load_dcs_airports()
            st.session_state["airport_df_cache"] = df

        if df is not None:
            airports = sorted(df["airport"].unique().tolist())
            airport = st.selectbox("Airport", airports, index=0 if airports else 0)
            df_rwy = df[df["airport"] == airport]
            runways = df_rwy["runway"].tolist()
            runway = st.selectbox("Runway", runways, index=0 if runways else 0)
            row = df_rwy[df_rwy["runway"] == runway].iloc[0] if len(df_rwy) else None
            elev_preset = float(row["elev_ft"]) if row is not None else 0.0
            tora_preset = float(row["tora_ft"]) if row is not None else 8000.0
            asda_preset = float(row["asda_ft"]) if row is not None else tora_preset
            st.caption("Loaded from dcs_airports.csv")
        else:
            st.info("`data/dcs_airports.csv` not found or unreadable — you can still enter TORA/ASDA manually.")
            elev_preset = 0.0; tora_preset = 8000.0; asda_preset = 8000.0

        col1, col2, col3 = st.columns(3)
        with col1:
            intersection_sub_ft = st.number_input("Intersection subtract (ft)", value=0, step=100, min_value=0)
        with col2:
            manual_tora = st.number_input("TORA (manual override, ft)", value=int(tora_preset), step=100, min_value=0)
        with col3:
            manual_asda = st.number_input("ASDA (manual override, ft)", value=int(asda_preset), step=100, min_value=0)

        tora_final = max(0, int(manual_tora) - int(intersection_sub_ft))
        asda_final = max(0, int(manual_asda) - int(intersection_sub_ft))

    col1, col2, col3 = st.columns(3)
    with col1:
        gw = st.number_input("Gross Weight (lbs)", value=60000, step=500, min_value=30000)
        oat = st.number_input("OAT (°C)", value=15, step=1)
        qnh = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    with col2:
        elev = st.number_input("Field Elev (ft MSL)", value=int(elev_preset), step=50)
        headwind = st.number_input("Head/Tailwind (kt, +HW / -TW)", value=0, step=1)
        tora = st.number_input("TORA (ft)", value=int(tora_final), step=100, min_value=0)
    with col3:
        asda = st.number_input("ASDA (ft)", value=int(asda_final), step=100, min_value=0)
        flap_mode = st.selectbox("Flaps", ["UP","MAN","FULL"], index=1)
        thrust_pref = st.selectbox("Thrust Pref", ["Auto-Select","Manual MIL","Manual AB","Manual DERATE"], index=0)
        manual_derate_pct = None
        if thrust_pref == "Manual DERATE":
            manual_derate_pct = st.slider("Manual Derate (%)", min_value=85, max_value=100, step=1, value=95)

    with st.expander("Calibration / Overrides", expanded=False):
        climb_floor = st.number_input("AEO Climb Floor (ft/NM)", value=int(st.session_state["overrides"].get("climb_floor_ftpnm", 300.0)), step=10, min_value=200)
        st.session_state["overrides"]["climb_floor_ftpnm"] = float(climb_floor)
        st.caption("Takeoff distances are UNFACTORED here. UI displays factored = ×1.10. Floors: UP=85%, MAN=90%, FULL=96%; absolute min 85%.")

    if st.button("Compute Takeoff", type="primary"):
        inputs = {
            "gw_lbs": float(gw),
            "oat_c": float(oat),
            "qnh_inhg": float(qnh),
            "field_elev_ft": float(elev),
            "headwind_kts_component": float(headwind),
            "tora_ft": float(tora),
            "asda_ft": float(asda),
            "flap_mode": flap_mode,
            "thrust_pref": thrust_pref,
            "manual_derate_pct": manual_derate_pct,
        }
        try:
            out = core_v2.compute_takeoff(inputs, overrides=st.session_state["overrides"])
            st.session_state["last_takeoff"] = out
        except Exception as e:
            st.error(f"Takeoff compute failed: {e}")
            st.stop()

    out = st.session_state.get("last_takeoff")
    if out:
        st.subheader("Results (Unfactored vs Factored +10%)")
        colA, colB, colC, colD = st.columns(4)
        _kv("Vs", round(out.get("Vs_kts", 0),1), " kt")
        _kv("Vr", round(out.get("Vr_kts", 0),1), " kt")
        _kv("V2", round(out.get("V2_kts", 0),1), " kt")
        _kv("Dispatchable", "Yes" if out.get("Dispatchable") else "No", "")
        _divider()
        asdr = float(out.get("ASDR_ft", 0.0))
        todr = float(out.get("TODR_OEI_35ft_ft", 0.0))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ASDR Unfactored", f"{int(asdr):,} ft")
            st.metric("ASDR Factored (×1.10)", f"{int(asdr * 1.10):,} ft")
        with col2:
            st.metric("TODR Unfactored", f"{int(todr):,} ft")
            st.metric("TODR Factored (×1.10)", f"{int(todr * 1.10):,} ft")
        st.caption(f"Resolved Thrust: **{out.get('ResolvedThrustLabel','-')}**  |  Flaps: **{out.get('Flaps','-')}**  |  GW: **{int(out.get('GW_lbs',0)):,} lb**  |  Source: {out.get('source','?')} ({out.get('version','?')})")
        with st.expander("Raw Output", expanded=False):
            st.json(out)

def tab_landing():
    st.header("Landing")
    col1, col2, col3 = st.columns(3)
    with col1:
        flap_setting = st.selectbox("Flap Setting", ["FULL","MAN","UP"], index=0)
        gw = st.number_input("Gross Weight (lbs)", value=52000, step=500, min_value=30000)
    with col2:
        pa = st.number_input("Pressure Altitude (ft)", value=0, step=100)
        tempF = st.number_input("Temperature (°F)", value=59, step=1)
    with col3:
        hw = st.number_input("Headwind (kt)", value=0, step=1)
    if st.button("Compute Landing", type="primary"):
        try:
            out = core_v2.compute_landing({
                "flap_setting": flap_setting,
                "gross_weight_lbs": float(gw),
                "pressure_alt_ft": float(pa),
                "temp_F": float(tempF),
                "headwind_kt": float(hw),
            })
            st.session_state["last_landing"] = out
        except Exception as e:
            st.error(f"Landing compute failed: {e}")
            st.stop()
    out = st.session_state.get("last_landing")
    if out:
        st.subheader("Results (Unfactored vs Factored ×1.67)")
        ldr = float(out.get("ldr_unfactored_ft", 0.0))
        col1, col2 = st.columns(2)
        col1.metric("Ground Roll (Unfactored)", f"{int(ldr):,} ft")
        col2.metric("Ground Roll (FAA ×1.67)", f"{int(ldr * 1.67):,} ft")
        if out.get("vref_kts"):
            st.metric("Vref (if available)", f"{round(out.get('vref_kts'),1)} kt")
        st.caption(f"Source: {out.get('source','?')} ({out.get('version','?')})")
        if "warning" in out:
            st.warning(out["warning"])
        with st.expander("Raw Output", expanded=False):
            st.json(out)

def tab_cruise():
    st.header("Cruise (Optimum Alt / Mach)")
    col1, col2 = st.columns(2)
    with col1:
        gw = st.number_input("Gross Weight (lbs)", value=55000, step=500, min_value=30000)
    with col2:
        di = st.number_input("Drag Index (DI)", value=50, step=1, min_value=0)
    if st.button("Compute Cruise", type="primary"):
        try:
            out = core_v2.compute_cruise({"gross_weight_lbs": float(gw), "drag_index": float(di)})
            st.session_state["last_cruise"] = out
        except Exception as e:
            st.error(f"Cruise compute failed: {e}")
            st.stop()
    out = st.session_state.get("last_cruise")
    if out:
        col1, col2 = st.columns(2)
        col1.metric("Optimum Altitude", f"{int(out.get('optimum_alt_ft',0)):,} ft")
        col2.metric("Optimum Mach", f"{round(out.get('optimum_mach',0),3)} M")
        st.caption(f"Source: {out.get('source','?')} ({out.get('version','?')})")
        if "warning" in out:
            st.warning(out["warning"])
        with st.expander("Raw Output", expanded=False):
            st.json(out)

def tab_climb():
    st.header("Climb Profile")
    col1, col2 = st.columns(2)
    with col1:
        profile = st.selectbox("Profile", ["Economy","Interceptor"], index=0)
    with col2:
        respect_250 = st.checkbox("Respect 250 KIAS below 10,000 ft", value=True)
    if st.button("Compute Climb", type="primary"):
        try:
            out = core_v2.compute_climb({"profile": profile, "respect_250": bool(respect_250)})
            st.session_state["last_climb"] = out
        except Exception as e:
            st.error(f"Climb compute failed: {e}")
            st.stop()
    out = st.session_state.get("last_climb")
    if out:
        st.caption(f"Source: {out.get('source','?')} ({out.get('version','?')})")
        if "warning" in out:
            st.warning(out["warning"])
        sched = out.get("schedule")
        if isinstance(sched, list) and len(sched) > 0:
            st.write("Schedule Preview:")
            st.dataframe(sched, use_container_width=True)
        else:
            st.info("No climb schedule available (CSV missing).")
        with st.expander("Raw Output", expanded=False):
            st.json(out)

def tab_compare():
    st.header("A/B Compare — Takeoff Scenarios")
    st.caption("Compare two configurations including intersection subtraction.")
    def scenario(label: str):
        st.subheader(label)
        with st.expander(f"Airport / Runway — {label}"):
            df = st.session_state.get("airport_df_cache")
            if df is None:
                df = _load_dcs_airports()
                st.session_state["airport_df_cache"] = df
            if df is not None:
                airports = sorted(df["airport"].unique().tolist())
                airport = st.selectbox(f"Airport ({label})", airports, index=0, key=f"{label}_apt")
                df_rwy = df[df["airport"] == airport]
                runways = df_rwy["runway"].tolist()
                runway = st.selectbox(f"Runway ({label})", runways, index=0, key=f"{label}_rwy")
                row = df_rwy[df_rwy["runway"] == runway].iloc[0] if len(df_rwy) else None
                elev_preset = float(row["elev_ft"]) if row is not None else 0.0
                tora_preset = float(row["tora_ft"]) if row is not None else 8000.0
                asda_preset = float(row["asda_ft"]) if row is not None else tora_preset
            else:
                elev_preset, tora_preset, asda_preset = 0.0, 8000.0, 8000.0
            col1, col2, col3 = st.columns(3)
            with col1:
                inter_sub = st.number_input(f"Intersection subtract (ft) — {label}", value=0, step=100, min_value=0, key=f"{label}_sub")
            with col2:
                tora = st.number_input(f"TORA (manual ft) — {label}", value=int(tora_preset), step=100, min_value=0, key=f"{label}_tora")
            with col3:
                asda = st.number_input(f"ASDA (manual ft) — {label}", value=int(asda_preset), step=100, min_value=0, key=f"{label}_asda")
            tora_final = max(0, int(tora) - int(inter_sub))
            asda_final = max(0, int(asda) - int(inter_sub))
        col1, col2, col3 = st.columns(3)
        with col1:
            gw = st.number_input(f"GW (lb) — {label}", value=60000, step=500, min_value=30000, key=f"{label}_gw")
            oat = st.number_input(f"OAT (°C) — {label}", value=15, step=1, key=f"{label}_oat")
            qnh = st.number_input(f"QNH (inHg) — {label}", value=29.92, step=0.01, format="%.2f", key=f"{label}_qnh")
        with col2:
            elev = st.number_input(f"Field Elev (ft) — {label}", value=int(elev_preset), step=50, key=f"{label}_elev")
            head = st.number_input(f"Head/Tailwind (kt) — {label}", value=0, step=1, key=f"{label}_wind")
            tora_f = st.number_input(f"TORA (ft) — {label}", value=int(tora_final), step=100, min_value=0, key=f"{label}_tora_fin")
        with col3:
            asda_f = st.number_input(f"ASDA (ft) — {label}", value=int(asda_final), step=100, min_value=0, key=f"{label}_asda_fin")
            flaps = st.selectbox(f"Flaps — {label}", ["UP","MAN","FULL"], index=1, key=f"{label}_flaps")
            thrust = st.selectbox(f"Thrust — {label}", ["Auto-Select","Manual MIL","Manual AB","Manual DERATE"], index=0, key=f"{label}_thrust")
            derate = None
            if thrust == "Manual DERATE":
                derate = st.slider(f"Derate (%) — {label}", min_value=85, max_value=100, step=1, value=95, key=f"{label}_derate")
        return {
            "gw_lbs": float(gw), "oat_c": float(oat), "qnh_inhg": float(qnh),
            "field_elev_ft": float(elev), "headwind_kts_component": float(head),
            "tora_ft": float(tora_f), "asda_ft": float(asda_f),
            "flap_mode": flaps, "thrust_pref": thrust, "manual_derate_pct": derate,
        }
    A = scenario("A"); B = scenario("B")
    if st.button("Compare A vs B", type="primary"):
        try:
            outA = core_v2.compute_takeoff(A, overrides=st.session_state["overrides"])
            outB = core_v2.compute_takeoff(B, overrides=st.session_state["overrides"])
            st.subheader("Results")
            c1, c2 = st.columns(2)
            for (lab, out, col) in [("A", outA, c1), ("B", outB, c2)]:
                with col:
                    st.markdown(f"### Scenario {lab}")
                    st.metric("Vr", f"{out['Vr_kts']:.1f} kt")
                    st.metric("ASDR Unfact.", f"{int(out['ASDR_ft']):,} ft")
                    st.metric("ASDR Fact.", f"{int(out['ASDR_ft']*1.10):,} ft")
                    st.metric("TODR Unfact.", f"{int(out['TODR_OEI_35ft_ft']):,} ft")
                    st.metric("TODR Fact.", f"{int(out['TODR_OEI_35ft_ft']*1.10):,} ft")
                    st.caption(f"{out['ResolvedThrustLabel']} | Flaps {out['Flaps']} | GW {int(out['GW_lbs']):,} lb")
        except Exception as e:
            st.error(f"Compare failed: {e}")

def tab_calibration():
    st.header("Calibration")
    st.caption("Locked policy: floors UP=85%, MAN=90%, FULL=96%; abs min 85%. AEO climb floor default 300 ft/NM.")
    climb_floor = st.number_input("AEO Climb Floor (ft/NM)", value=int(st.session_state["overrides"].get("climb_floor_ftpnm", 300.0)), step=10, min_value=200)
    st.session_state["overrides"]["climb_floor_ftpnm"] = float(climb_floor)
    st.success("Overrides updated (runtime only).")

def tab_diagnostics():
    st.header("Diagnostics")
    st.write(f"App version: **{APP_VERSION}**")
    try:
        st.write(f"perf_core_v2 version: **{core_v2.__version__}**")
    except Exception:
        st.write("perf_core_v2 version: unknown")
    data_dir = _data_dir()
    try:
        files = sorted(os.listdir(data_dir))
    except Exception:
        files = []
    st.write("**data/**:", files if files else "_(not found or empty)_")
    paths = [
        os.path.join(data_dir, "f14_landing_natops_full.csv"),
        os.path.join(data_dir, "f14_cruise_natops.csv"),
        os.path.join(data_dir, "f14_climb_natops.csv"),
        os.path.join(data_dir, "dcs_airports.csv"),
    ]
    status = {os.path.basename(p): os.path.exists(p) for p in paths}
    st.json({"csv_presence": status})

def tab_export():
    st.header("Export (Kneeboard)")
    lines = []
    to = st.session_state.get("last_takeoff")
    ld = st.session_state.get("last_landing")
    cr = st.session_state.get("last_cruise")
    cl = st.session_state.get("last_climb")
    lines.append(f"F-14 Toolkit — v{APP_VERSION}")
    lines.append("")
    if to:
        lines.append("[TAKEOFF]")
        lines.append(f"  Thrust: {to.get('ResolvedThrustLabel','-')}  Flaps: {to.get('Flaps','-')}  GW: {int(to.get('GW_lbs',0)):,} lb")
        lines.append(f"  Vs/Vr/V2: {to.get('Vs_kts',0):.0f}/{to.get('Vr_kts',0):.0f}/{to.get('V2_kts',0):.0f} kt")
        lines.append(f"  ASDR: {int(to.get('ASDR_ft',0)):,} (UF)  {int(to.get('ASDR_ft',0)*1.10):,} (×1.10) ft")
        lines.append(f"  TODR: {int(to.get('TODR_OEI_35ft_ft',0)):,} (UF)  {int(to.get('TODR_OEI_35ft_ft',0)*1.10):,} (×1.10) ft")
        lines.append("")
    if ld:
        lines.append("[LANDING]")
        lines.append(f"  LDR: {int(ld.get('ldr_unfactored_ft',0)):,} (UF)  {int(ld.get('ldr_unfactored_ft',0)*1.67):,} (×1.67) ft")
        if ld.get("vref_kts"):
            lines.append(f"  Vref: {ld.get('vref_kts'):.0f} kt")
        lines.append("")
    if cr:
        lines.append("[CRUISE]")
        lines.append(f"  Opt Alt: {int(cr.get('optimum_alt_ft',0)):,} ft  Opt Mach: {cr.get('optimum_mach',0):.2f}")
        lines.append("")
    if cl:
        lines.append("[CLIMB]")
        lines.append(f"  Profile: {cl.get('profile','-')}  Respect 250: {cl.get('respect_250',True)}")
        lines.append("")
    txt = "\n".join(lines)
    st.code(txt, language="text")
    st.download_button("Download kneeboard.txt", txt, file_name="kneeboard.txt")

def main():
    st.set_page_config(page_title="F-14 Performance Toolkit", layout="wide")
    _init_state()
    st.title("F-14 Performance Toolkit")
    st.caption("UI rebuilt to Locked Spec — using perf_core_v2")
    tabs = st.tabs(["Takeoff","Landing","Cruise","Climb","A/B Compare","Calibration","Diagnostics","Export"])
    with tabs[0]: tab_takeoff()
    with tabs[1]: tab_landing()
    with tabs[2]: tab_cruise()
    with tabs[3]: tab_climb()
    with tabs[4]: tab_compare()
    with tabs[5]: tab_calibration()
    with tabs[6]: tab_diagnostics()
    with tabs[7]: tab_export()

if __name__ == "__main__":
    main()
