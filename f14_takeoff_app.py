
# ============================================================
# F-14 Performance Toolkit — Main Streamlit App
# File: f14_takeoff_app.py
# Version: v1.2.1 (2025-09-21) — Rebuild from scratch
# ============================================================
from __future__ import annotations
import os, math, json, io
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st

# Local modules
import f14_takeoff_core as core
import data_loaders

APP_VERSION = "v1.2.1"

# ------------------------------
# Utilities
# ------------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")

def _load_csv_safe(name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Failed to read {name}: {e}")
        return None

def _interpolate_table(df: pd.DataFrame, xcol: str, ycol: str, zcol: str, x: float, y: float) -> Optional[float]:
    try:
        if df is None or df.empty: return None
        # Pivot to grid
        grid = df.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc='mean')
        xs = np.array(grid.columns, dtype=float)
        ys = np.array(grid.index, dtype=float)
        # Clamp into bounds
        x = float(np.clip(x, xs.min(), xs.max()))
        y = float(np.clip(y, ys.min(), ys.max()))
        # Bilinear interpolation
        x1 = xs[xs<=x].max(); x2 = xs[xs>=x].min()
        y1 = ys[ys<=y].max(); y2 = ys[ys>=y].min()
        z11 = grid.loc[y1, x1]; z12 = grid.loc[y2, x1]
        z21 = grid.loc[y1, x2]; z22 = grid.loc[y2, x2]
        if x1==x2 and y1==y2: return float(z11)
        if x1==x2:
            t = (y - y1) / (y2 - y1 + 1e-9)
            return float(z11*(1-t) + z12*t)
        if y1==y2:
            t = (x - x1) / (x2 - x1 + 1e-9)
            return float(z11*(1-t) + z21*t)
        tx = (x - x1) / (x2 - x1 + 1e-9)
        ty = (y - y1) / (y2 - y1 + 1e-9)
        z1 = z11*(1-tx) + z21*tx
        z2 = z12*(1-tx) + z22*tx
        return float(z1*(1-ty) + z2*ty)
    except Exception:
        return None

def _as_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "_no data_"
    out = io.StringIO()
    out.write("| " + " | ".join(df.columns) + " |\n")
    out.write("|" + "|".join(["---"]*len(df.columns)) + "|\n")
    for _, r in df.iterrows():
        out.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    return out.getvalue()

def _kneeboard_text_block(takeoff: Dict[str,Any], landing: Dict[str,Any], cruise: Dict[str,Any], climb: Dict[str,Any]) -> str:
    def fmt(v, d=0):
        try: return f"{float(v):.{d}f}"
        except: return "-"
    lines = []
    lines += [f"F-14 Toolkit Kneeboard — {APP_VERSION}"]
    lines += ["", "— TAKEOFF —"]
    lines += [f"  Thrust: {takeoff.get('ResolvedThrustLabel','-')}   Flaps:{takeoff.get('Flaps','-')}   GW:{fmt(takeoff.get('GW_lbs'),0)} lb"]
    lines += [f"  Vr:{fmt(takeoff.get('Vr_kts'))}  V2:{fmt(takeoff.get('V2_kts'))}  ASDR:{fmt(takeoff.get('ASDR_ft'))} ft  (×1.10 => {fmt(takeoff.get('ASDR_ft',0)*1.10)})"]
    lines += [f"  TODR OEI 35ft:{fmt(takeoff.get('TODR_OEI_35ft_ft'))} ft  (×1.10 => {fmt(takeoff.get('TODR_OEI_35ft_ft',0)*1.10)})"]
    lines += ["", "— CRUISE —"]
    lines += [f"  Opt Alt:{fmt(cruise.get('optimum_alt_ft'))} ft   Opt Mach:{fmt(cruise.get('optimum_mach'),3)}   DI:{fmt(cruise.get('drag_index'),0)}"]
    lines += ["", "— CLIMB —"]
    lines += [f"  Profile:{climb.get('profile','-')}   Respect 250 KIAS:{climb.get('respect_250','-')}"]
    lines += ["", "— LANDING —"]
    lines += [f"  LDR (unfactored): {fmt(landing.get('ldr_unfactored_ft'))} ft   LDR ×1.67: {fmt(landing.get('ldr_factored_ft'))} ft"]
    return "\n".join(lines)

def _call_core_perf(inputs: Dict[str,Any], overrides: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    # graceful wrapper for mixed core versions
    fn = getattr(core, "perf_compute_takeoff", None)
    if fn is None:
        legacy = getattr(core, "plan_takeoff", None) or getattr(core, "plan_takeoff_with_optional_derate", None)
        if legacy is None:
            raise RuntimeError("No compatible takeoff function in core.")
        res = legacy(inputs)
        # Map legacy fields if needed
        res.setdefault("ASDR_ft", res.get("ASD_ft"))
        res.setdefault("TODR_OEI_35ft_ft", res.get("TODR_ft"))
        return res
    else:
        try:
            return fn(inputs, overrides=overrides)
        except TypeError:
            return fn(inputs)

# ------------------------------
# UI Helpers
# ------------------------------
def sidebar_inputs() -> Dict[str,Any]:
    st.sidebar.caption("Enter conditions & runway. Intersection reduction optional.")
    col1,col2 = st.sidebar.columns(2)
    airport = col1.text_input("Airport", "BATUMI")
    runway = col2.text_input("Runway", "13")
    oat_c = st.sidebar.number_input("OAT (°C)", -40, 60, 15, step=1)
    qnh_inhg = st.sidebar.number_input("QNH (inHg)", 25.00, 31.00, 29.92, step=0.01, format="%.2f")
    wind_dir = st.sidebar.number_input("Wind Dir (°)", 0, 360, 0, step=10)
    wind_spd = st.sidebar.number_input("Wind Speed (kt)", 0, 100, 0, step=1)
    gw_lbs = st.sidebar.number_input("Gross Weight (lb)", 40000, 74000, 60000, step=100)
    tora_ft = st.sidebar.number_input("TORA (ft)", 1000, 16000, 8000, step=100)
    asda_ft = st.sidebar.number_input("ASDA (ft)", 1000, 16000, 8000, step=100)
    reduce = st.sidebar.number_input("Intersection reduction (ft)", 0, 8000, 0, step=50)
    tora_ft = max(0, tora_ft - reduce)
    asda_ft = max(0, asda_ft - reduce)
    flap_mode = st.sidebar.selectbox("Flaps", ["UP","MAN","FULL"], index=1)
    thrust_pref = st.sidebar.selectbox("Thrust Mode", ["Auto-Select","Manual MIL","Manual AB","Manual DERATE"])
    manual_derate = None
    if thrust_pref == "Manual DERATE":
        manual_derate = st.sidebar.slider("DERATE (%)", 85, 100, 90, step=1)
    respect_250 = st.sidebar.checkbox("Respect 250 KIAS below 10k", value=True)
    mission_profile = st.sidebar.selectbox("Climb Profile", ["Economy","Interceptor"])
    di = st.sidebar.number_input("Drag Index (DI)", 0, 300, 50, step=5)
    return {
        "airport": airport, "runway": runway,
        "oat_c": oat_c, "qnh_inhg": qnh_inhg,
        "wind_dir": wind_dir, "wind_spd": wind_spd,
        "gw_lbs": gw_lbs, "tora_ft": tora_ft, "asda_ft": asda_ft,
        "flap_mode": flap_mode, "thrust_pref": thrust_pref, "manual_derate": manual_derate,
        "respect_250": respect_250, "mission_profile": mission_profile, "drag_index": di
    }

def calibration_panel() -> Optional[Dict[str,Any]]:
    st.subheader("Calibration & Policy Overrides")
    enable = st.checkbox("Enable Calibration Mode (runtime overrides)", value=False)
    if not enable:
        return None
    c1,c2,c3 = st.columns(3)
    min_up = c1.slider("Floor Flaps UP (%)", 80, 100, 85, step=1)
    min_man = c2.slider("Floor Flaps MAN (%)", 85, 100, 90, step=1)
    min_full = c3.slider("Floor Flaps FULL (%)", 90, 100, 96, step=1)
    c4,c5,c6 = st.columns(3)
    mexp = c4.slider("Runway scaling exponent (m)", 1.0, 2.5, 1.6, step=0.05)
    rfac = c5.slider("ASDR/TODR safety factor (×)", 1.00, 1.25, 1.10, step=0.01)
    climb_floor = c6.slider("AEO climb floor (ft/NM)", 200, 500, 300, step=10)
    st.caption("Floors enforced: UP=85%, MAN=90%, FULL=96% (minimums). AEO climb floor default 300 ft/NM.")
    return {
        "min_derate_up": min_up, "min_derate_man": min_man, "min_derate_full": min_full,
        "m_exponent": mexp, "runway_factor": rfac, "climb_floor_ftpnm": climb_floor
    }

# ------------------------------
# Main App
# ------------------------------
st.set_page_config(page_title=f"F-14 Performance Toolkit {APP_VERSION}", layout="wide")
st.title(f"F-14 Performance Toolkit — {APP_VERSION}")
st.caption("Restored full app surface with v1.2.1 corrections.")

# Tabs
tabs = st.tabs(["Takeoff","Climb","Cruise","Landing","A/B Compare","Calibration","Diagnostics","Export"])

# Sidebar Inputs
S = sidebar_inputs()

# Build core input dict
core_inputs = {
    "gw_lbs": S["gw_lbs"],
    "oat_c": S["oat_c"],
    "qnh_inhg": S["qnh_inhg"],
    "field_elev_ft": 0,  # user can add later
    "pressure_alt_ft": 0,
    "headwind_kts_component": S["wind_spd"],
    "tora_ft": S["tora_ft"],
    "asda_ft": S["asda_ft"],
    "flap_mode": S["flap_mode"],
    "thrust_pref": S["thrust_pref"],
    "manual_derate_pct": S["manual_derate"]
}

overrides = None
with tabs[5]:
    overrides = calibration_panel()
    st.divider()
    st.subheader("Current Policy Snapshot")
    st.json(overrides or {"overrides":"not active"})

# === TAKEOFF ===
with tabs[0]:
    st.subheader("Takeoff Results")
    try:
        to_res = _call_core_perf(core_inputs, overrides=overrides)
        if not to_res:
            st.error("No dispatchable config returned by core.")
        else:
            colA,colB,colC,colD = st.columns(4)
            colA.metric("Vr (kt)", round(float(to_res.get("Vr_kts",0))))
            colB.metric("V2 (kt)", round(float(to_res.get("V2_kts",0))))
            asdr = float(to_res.get("ASDR_ft",0) or 0.0)
            todr = float(to_res.get("TODR_OEI_35ft_ft",0) or 0.0)
            colC.metric("ASDR (ft, unfactored)", round(asdr))
            colD.metric("ASDR (ft, ×1.10 factored)", round(asdr*(overrides.get("runway_factor",1.10) if overrides else 1.10)))
            st.caption("Factored ASDR uses +10% by default (FAA-style), adjustable in Calibration.")
            c1,c2 = st.columns(2)
            c1.metric("TODR OEI 35ft (ft, unfactored)", round(todr))
            c2.metric("TODR OEI 35ft (ft, ×1.10 factored)", round(todr*(overrides.get("runway_factor",1.10) if overrides else 1.10)))
            st.markdown("**Resolved Thrust**: " + str(to_res.get("ResolvedThrustLabel","-")))
            st.markdown("**Flaps**: " + str(to_res.get("Flaps","-")))
            with st.expander("Raw core result"):
                st.json(to_res)
    except Exception as e:
        st.exception(e)

# === CLIMB ===
with tabs[1]:
    st.subheader("Climb (Economy vs Interceptor)")
    df_climb = _load_csv_safe("f14_climb_natops.csv")
    if df_climb is None or df_climb.empty:
        st.info("NATOPS climb CSV not available yet — showing placeholder guidance.")
        st.markdown("- Economy: 250 KIAS to 10,000 ft, then Mach 0.70–0.75 to cruise.\n- Interceptor: 300–350 KIAS (or as permitted) to 10,000 ft, then max rate speed to intercept.")
    else:
        st.dataframe(df_climb)
    st.caption(f"Respect 250 KIAS: {S['respect_250']}  |  Selected profile: {S['mission_profile']}")

# === CRUISE ===
with tabs[2]:
    st.subheader("Cruise (Optimum Altitude / Mach)")
    df_cruise = _load_csv_safe("f14_cruise_natops.csv")
    if df_cruise is None or df_cruise.empty:
        st.info("Cruise NATOPS CSV not found. Place f14_cruise_natops.csv in /data.")
    else:
        # Expect columns: gross_weight_lbs, drag_index, optimum_alt_ft, optimum_mach
        opt_alt = _interpolate_table(df_cruise, "drag_index", "gross_weight_lbs", "optimum_alt_ft", S["drag_index"], S["gw_lbs"])
        opt_mach = _interpolate_table(df_cruise, "drag_index", "gross_weight_lbs", "optimum_mach", S["drag_index"], S["gw_lbs"])
        c1,c2 = st.columns(2)
        c1.metric("Optimum Altitude (ft)", round(opt_alt) if opt_alt else "-")
        c2.metric("Optimum Mach", f"{opt_mach:.3f}" if opt_mach else "-")
        with st.expander("Cruise grid (NATOPS-derived)"):
            st.dataframe(df_cruise)

# === LANDING ===
with tabs[3]:
    st.subheader("Landing Distance (NATOPS)")
    df_land = _load_csv_safe("f14_landing_natops_full.csv")
    if df_land is None or df_land.empty:
        st.error("Landing NATOPS CSV not found: data/f14_landing_natops_full.csv")
    else:
        st.markdown("Showing NATOPS **ground roll (unfactored)**; FAA factored = ×1.67.")
        # Simple filter UI
        c1,c2,c3 = st.columns(3)
        flap_sel = c1.selectbox("Flaps", sorted(df_land["flap_setting"].dropna().unique().tolist()))
        gw_sel = c2.selectbox("Gross Weight (lb)", sorted(df_land["gross_weight_lbs"].dropna().unique().tolist()))
        alt_sel = c3.selectbox("Pressure Alt (ft)", sorted(df_land["pressure_alt_ft"].dropna().unique().tolist()))
        c4,c5 = st.columns(2)
        temp_sel = c4.selectbox("Temp (°F)", sorted(df_land["temp_F"].dropna().unique().tolist()))
        hw_sel = c5.selectbox("Headwind (kt)", sorted(df_land["headwind_kt"].dropna().unique().tolist()))
        mask = (
            (df_land["flap_setting"]==flap_sel) &
            (df_land["gross_weight_lbs"]==gw_sel) &
            (df_land["pressure_alt_ft"]==alt_sel) &
            (df_land["temp_F"]==temp_sel) &
            (df_land["headwind_kt"]==hw_sel)
        )
        rows = df_land[mask]
        if rows.empty:
            st.warning("No matching NATOPS row for that combo. Try different values.")
        else:
            val = float(rows.iloc[0]["ground_roll_ft_unfactored"])
            st.metric("LDR (unfactored ground roll, ft)", round(val))
            st.metric("LDR (FAA ×1.67, ft)", round(val*1.67))
        with st.expander("NATOPS landing table (subset)"):
            st.dataframe(df_land.head(50))

# === A/B COMPARE ===
with tabs[4]:
    st.subheader("A/B Compare — Intersection / Flaps / Thrust")
    st.caption("Quickly compare two scenarios side-by-side.")
    def _scenario_box(label: str) -> Tuple[Dict[str,Any], Optional[Dict[str,Any]]]:
        st.markdown(f"**Scenario {label}**")
        c1,c2,c3 = st.columns(3)
        tora = c1.number_input(f"[{label}] TORA (ft)", 1000, 16000, S["tora_ft"], key=f"tora_{label}")
        asda = c2.number_input(f"[{label}] ASDA (ft)", 1000, 16000, S["asda_ft"], key=f"asda_{label}")
        reduce = c3.number_input(f"[{label}] Intersection reduction (ft)", 0, 8000, 0, key=f"reduce_{label}")
        tora = max(0, tora - reduce); asda = max(0, asda - reduce)
        c4,c5 = st.columns(2)
        flap = c4.selectbox(f"[{label}] Flaps", ["UP","MAN","FULL"], index=["UP","MAN","FULL"].index(S["flap_mode"]), key=f"flap_{label}")
        thrust = c5.selectbox(f"[{label}] Thrust", ["Auto-Select","Manual MIL","Manual AB","Manual DERATE"], index=["Auto-Select","Manual MIL","Manual AB","Manual DERATE"].index(S["thrust_pref"]), key=f"thr_{label}")
        der = None
        if thrust == "Manual DERATE":
            der = st.slider(f"[{label}] DERATE (%)", 85, 100, S["manual_derate"] or 90, key=f"der_{label}")
        inputs = dict(core_inputs)
        inputs.update({"tora_ft":tora,"asda_ft":asda,"flap_mode":flap,"thrust_pref":thrust,"manual_derate_pct":der})
        try:
            res = _call_core_perf(inputs, overrides=overrides)
        except Exception as e:
            st.warning(f"[{label}] core error: {e}")
            res = None
        return inputs, res

    left = st.container()
    right = st.container()
    with left: A_in, A_res = _scenario_box("A")
    with right: B_in, B_res = _scenario_box("B")
    st.divider()
    col1,col2 = st.columns(2)
    def _summ(label, res):
        if not res:
            col1.error(f"{label}: no result")
            return
        col1.metric(f"{label} Vr (kt)", round(float(res.get("Vr_kts",0))))
        col2.metric(f"{label} V2 (kt)", round(float(res.get("V2_kts",0))))
        col1.metric(f"{label} ASDR (ft)", round(float(res.get("ASDR_ft",0))))
        col2.metric(f"{label} TODR OEI 35ft (ft)", round(float(res.get("TODR_OEI_35ft_ft",0))))
    _summ("A", A_res); _summ("B", B_res)

# === DIAGNOSTICS ===
with tabs[6]:
    st.subheader("Diagnostics")
    st.markdown(f"**App version**: {APP_VERSION}")
    st.markdown(f"**Working dir**: {os.getcwd()}")
    st.markdown(f"**Data dir**: {DATA_DIR}")
    try:
        st.markdown("**Data contents:**")
        st.write(os.listdir(DATA_DIR))
    except Exception as e:
        st.write(f"(no data dir / error: {e})")
    with st.expander("Loaded CSVs (head)"):
        for fn in ["f14_landing_natops_full.csv","f14_cruise_natops.csv","f14_climb_natops.csv","f14_approach_speeds_natops.csv","f14_approach_corrections_natops_sheet1.csv","f14_approach_corrections_natops_sheet2.csv"]:
            df = _load_csv_safe(fn)
            st.markdown(f"**{fn}**")
            if df is None: st.write("_missing_")
            else: st.dataframe(df.head())

# === EXPORT ===
with tabs[7]:
    st.subheader("Export — Kneeboard (TXT)")
    # Build a lightweight kneeboard text from available results
    try:
        takeoff_block = to_res if 'to_res' in locals() else {}
    except:
        takeoff_block = {}
    # Landing sample (from last selection if present)
    landing_block = {}
    try:
        landing_block = {"ldr_unfactored_ft": float(rows.iloc[0]["ground_roll_ft_unfactored"]) if 'rows' in locals() and not rows.empty else None,
                         "ldr_factored_ft": float(rows.iloc[0]["ground_roll_ft_unfactored"])*1.67 if 'rows' in locals() and not rows.empty else None}
    except Exception:
        pass
    cruise_block = {"optimum_alt_ft": opt_alt if 'opt_alt' in locals() else None,
                    "optimum_mach": opt_mach if 'opt_mach' in locals() else None,
                    "drag_index": S["drag_index"]}
    climb_block = {"profile": S["mission_profile"], "respect_250": S["respect_250"]}
    text = _kneeboard_text_block(takeoff_block, landing_block, cruise_block, climb_block)
    st.code(text)
    st.download_button("Download kneeboard.txt", data=text, file_name="kneeboard.txt", mime="text/plain")
