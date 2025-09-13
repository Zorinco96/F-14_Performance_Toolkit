# app.py — DCS F‑14B/D Takeoff Performance (Streamlit) — v0.2 wired to your CSVs
#
# CHANGELOG v0.2
# • Ingests your exact runway file: data/DCS_Airports.csv (feet, headings, slopes, TORA/TODA/ASDA, elevations)
# • Ingests your perf grids: data/perf_f14b.csv and data/perf_f14d.csv (single‑V1 rows)
# • Maps UI flaps to 0/20/40 deg (UP/MAN/FULL). Your CSVs currently include 20 and 40 only.
# • Uses TORA as the default “available distance”, with manual shortening. Also displays TODA/ASDA for info.
# • Balanced‑field note: because your perf rows include only one V1 per condition, we currently set
#   Required Distance = max(ASD, AGD) (after wind/slope). True V1 balancing requires multiple rows per
#   (weight, flap, PA, OAT) with different V1_kt; you can add those later and we’ll switch logic automatically.
#
# IMPORTANT
# This is a sim planning tool based on public/manual data. Not for real‑world aviation use.

import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

################################################################################
# Atmosphere & helpers
################################################################################

def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def ft_to_m(ft: float) -> float:
    return ft * 0.3048

def m_to_ft(m: float) -> float:
    return m / 0.3048

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return field_elev_ft + (29.92 - qnh_inhg) * 1000.0

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    ISA_SEA_LEVEL_T_C = 15.0
    LAPSE_RATE_C_PER_M = 0.0065
    isa_c = ISA_SEA_LEVEL_T_C - (LAPSE_RATE_C_PER_M * ft_to_m(pa_ft))
    return pa_ft + 120.0 * (oat_c - isa_c)

def headwind_component(knots_wind: float, wind_dir_deg: float, rwy_heading_deg: float) -> float:
    delta = math.radians((wind_dir_deg - rwy_heading_deg) % 360)
    return knots_wind * math.cos(delta)

################################################################################
# Data IO: EXACTLY your files & schema
################################################################################

DATA_DIR = pathlib.Path(__file__).parent / "data"

# Safer readers that return None instead of crashing if files are missing

def read_csv_if_exists(path: pathlib.Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed reading {path.name}: {e}")
    return None

def find_path_ignore_case(filename: str) -> Optional[pathlib.Path]:
    """Return a Path in DATA_DIR matching filename regardless of case."""
    direct = DATA_DIR / filename
    if direct.exists():
        return direct
    # Try common case variations
    for cand in [filename.lower(), filename.upper(), filename.title()]:
        p = DATA_DIR / cand
        if p.exists():
            return p
    # Full scan fallback
    target = filename.lower()
    for p in DATA_DIR.glob("*.csv"):
        if p.name.lower() == target:
            return p
    return None

# Runway DB: DCS_Airports.csv
# map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,threshold_elev_ft,opp_threshold_elev_ft,slope_percent,notes

@st.cache_data
def load_runways() -> Optional[pd.DataFrame]:
    path = find_path_ignore_case("DCS_Airports.csv") or find_path_ignore_case("dcs_airports.csv")
    df = read_csv_if_exists(path) if path else None
    if df is None:
        return None
    df["rw_key"] = df["airport_name"] + " — " + df["runway_pair"].astype(str) + "/" + df["runway_end"].astype(str)
    df["runway_label"] = df["airport_name"] + " " + df["runway_end"].astype(str) + " (" + df["runway_pair"].astype(str) + ")"
    return df

# Performance: perf_f14b.csv and perf_f14d.csv
# model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft

@st.cache_data
def load_perf() -> Optional[pd.DataFrame]:
    p_b_path = find_path_ignore_case("perf_f14b.csv")
    p_d_path = find_path_ignore_case("perf_f14d.csv")
    p_b = read_csv_if_exists(p_b_path) if p_b_path else None
    p_d = read_csv_if_exists(p_d_path) if p_d_path else None
    if p_b is None and p_d is None:
        return None
    parts = [df for df in [p_b, p_d] if df is not None]
    perf = pd.concat(parts, ignore_index=True)
    perf["model"] = perf["model"].str.upper()
    perf["thrust"] = perf["thrust"].str.upper()
    return perf

################################################################################
# Core calculation using your current single‑V1 rows
################################################################################

@dataclass
class Inputs:
    model: str  # F-14B or F-14D
    theatre: str
    airport: str
    runway_label: str
    heading_deg: float
    tora_ft: float
    toda_ft: float
    asda_ft: float
    elev_ft: float
    slope_pct: float
    shorten_ft: float

    oat_c: float
    qnh_inhg: float
    wind_knots: float
    wind_dir_deg: float

    weight_mode: str  # GW or Fuel+Stores (GW only for now)
    gw_lbs: float
    cg_percent_mac: Optional[float]

    flap_sel: str  # Auto/UP/MAN/FULL
    thrust_sel: str  # Auto/REDUCED/MIL/AB

@dataclass
class PerfResult:
    v1: float
    vr: float
    v2: float
    vs: float
    flap_setting: str
    thrust_setting: str
    req_distance_ft: float
    avail_distance_ft: float
    limiting: str


def auto_flaps(gw_lbs: float, pa_ft: float, oat_c: float) -> str:
    # Heuristic: heavy/hot/high → FULL, mid → MAN, light/cool/low → UP (but your perf tables only cover 20/40)
    if gw_lbs >= 70000 or pa_ft > 3000 or oat_c > 30:
        return "FULL"
    elif gw_lbs >= 62000:
        return "MAN"
    return "UP"


def auto_thrust(oat_c: float, pa_ft: float) -> str:
    if pa_ft > 3000 or oat_c > 30:
        return "MIL"
    return "MIL"  # default MIL unless you populate reduced/AB policy


def flap_to_deg(flap_sel: str) -> Optional[int]:
    if flap_sel.upper() == "UP":
        return 0
    if flap_sel.upper() in ("MAN", "MANEUVERING", "20"):
        return 20
    if flap_sel.upper() in ("FULL", "40"):
        return 40
    return None


def thrust_label_to_table_mode(label: str) -> str:
    lab = label.upper()
    if lab in ("MIL", "MILITARY"):
        return "MILITARY"
    if lab in ("AB", "AFTERBURNER"):
        return "AFTERBURNER"
    if lab in ("REDUCED", "FLEX"):
        return "REDUCED"  # not present in current CSVs; will fall back
    return lab


def apply_wind_slope(distance_ft: float, headwind_kn: float, slope_pct: float) -> float:
    # Simple placeholder corrections; refine if you add manual corrections from NATOPS
    adj = distance_ft
    adj *= 1.0 - 0.01 * min(0.5, max(-0.5, headwind_kn / 2.0))
    adj *= 1.0 + 0.20 * max(0.0, slope_pct)  # uphill penalty
    return adj


def nearest_perf_row(perf: pd.DataFrame, model: str, flap_deg: int, thrust_mode: str,
                     gw_lbs: float, pa_ft: float, oat_c: float) -> pd.Series:
    sub = perf[(perf["model"] == model.upper()) &
               (perf["flap_deg"] == flap_deg) &
               (perf["thrust"].str.upper() == thrust_mode.upper())]
    # If that’s empty (e.g., REDUCED), fall back to MILITARY
    if sub.empty:
        sub = perf[(perf["model"] == model.upper()) &
                   (perf["flap_deg"] == flap_deg) &
                   (perf["thrust"].str.upper() == "MILITARY")]
    if sub.empty:
        raise ValueError("No performance rows match model/flap/thrust.")
    # Distance in param‑space
    sub = sub.assign(
        d_w = (sub["gw_lbs"] - gw_lbs).abs(),
        d_pa = (sub["press_alt_ft"] - pa_ft).abs(),
        d_t  = (sub["oat_c"] - oat_c).abs(),
    )
    row = sub.sort_values(["d_w", "d_pa", "d_t"]).iloc[0]
    return row


def compute_performance(inp: Inputs, perfdb: pd.DataFrame) -> PerfResult:
    pa_ft = pressure_altitude_ft(inp.elev_ft, inp.qnh_inhg)
    headwind = headwind_component(inp.wind_knots, inp.wind_dir_deg, inp.heading_deg)

    flap = inp.flap_sel if inp.flap_sel != "Auto" else auto_flaps(inp.gw_lbs, pa_ft, inp.oat_c)
    thrust = inp.thrust_sel if inp.thrust_sel != "Auto" else auto_thrust(inp.oat_c, pa_ft)

    flap_deg = flap_to_deg(flap)
    if flap_deg is None:
        # If user chose UP but tables don’t have 0°, promote to 20°
        flap_deg = 20
        flap = "MAN"

    thrust_mode = thrust_label_to_table_mode(thrust)

    row = nearest_perf_row(perfdb, inp.model, flap_deg, thrust_mode, inp.gw_lbs, pa_ft, inp.oat_c)

    V1 = float(row["V1_kt"])
    Vr = float(row["Vr_kt"])
    V2 = float(row["V2_kt"])
    Vs = float(row["Vs_kt"]) if not pd.isna(row.get("Vs_kt", np.nan)) else np.nan

    ASD = apply_wind_slope(float(row["ASD_ft"]), headwind, inp.slope_pct)
    AGD = apply_wind_slope(float(row["AGD_ft"]), headwind, inp.slope_pct)

    # Required distance: since we only have one V1 per row, use max(ASD, AGD) as the limiting field length.
    req_ft = max(ASD, AGD)

    # Available distance: use TORA minus user shortening
    avail_ft = max(0.0, float(inp.tora_ft) - float(inp.shorten_ft))

    # Determine limiting label
    limiting = "ASD" if ASD >= AGD else "AGD"

    return PerfResult(
        v1=V1, vr=Vr, v2=V2, vs=Vs,
        flap_setting=flap, thrust_setting=thrust,
        req_distance_ft=req_ft, avail_distance_ft=avail_ft,
        limiting=limiting,
    )

################################################################################
# UI
################################################################################

st.set_page_config(page_title="DCS F‑14 Takeoff Performance", page_icon="✈️", layout="wide")

st.title("DCS F‑14B/D Takeoff Performance")

rwy_df = load_runways()
perfdb = load_perf()

# Upload fallbacks
if rwy_df is None:
    st.warning("data/DCS_Airports.csv not found. Upload it below or add it to the repo.")
    up = st.file_uploader("Upload DCS_Airports.csv", type=["csv"], key="rwy_csv")
    if up is not None:
        rwy_df = pd.read_csv(up)
        rwy_df["rw_key"] = rwy_df["airport_name"] + " — " + rwy_df["runway_pair"].astype(str) + "/" + rwy_df["runway_end"].astype(str)
        rwy_df["runway_label"] = rwy_df["airport_name"] + " " + rwy_df["runway_end"].astype(str) + " (" + rwy_df["runway_pair"].astype(str) + ")"

if perfdb is None:
    st.warning("Performance CSVs not found. Upload perf_f14b.csv / perf_f14d.csv (one or both).")
    up_b = st.file_uploader("Upload perf_f14b.csv", type=["csv"], key="perf_b")
    up_d = st.file_uploader("Upload perf_f14d.csv", type=["csv"], key="perf_d")
    frames = []
    if up_b is not None:
        frames.append(pd.read_csv(up_b))
    if up_d is not None:
        frames.append(pd.read_csv(up_d))
    if frames:
        perfdb = pd.concat(frames, ignore_index=True)
        perfdb["model"] = perfdb["model"].str.upper()
        perfdb["thrust"] = perfdb["thrust"].str.upper()

if rwy_df is None or perfdb is None:
    st.stop()

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Map", sorted(rwy_df["map"].unique()))
    df_t = rwy_df[rwy_df["map"] == theatre]

    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]

    rw = st.selectbox("Runway End", list(df_a["runway_label"]))
    row_rwy = df_a[df_a["runway_label"] == rw].iloc[0]

    st.caption("Available distances are per TORA/TODA/ASDA from your CSV; app uses TORA for ‘available’.")
    st.metric("TORA (ft)", f"{int(row_rwy['tora_ft']):,}")
    st.metric("TODA (ft)", f"{int(row_rwy['toda_ft']):,}")
    st.metric("ASDA (ft)", f"{int(row_rwy['asda_ft']):,}")

    shorten_ft = st.number_input("Manually shorten available runway (ft)", min_value=0.0, value=0.0, step=100.0)

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    qnh_mode = st.selectbox("QNH Input", ["inHg", "hPa"], index=0)
    if qnh_mode == "inHg":
        qnh_inhg = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    else:
        qnh_hpa = st.number_input("QNH (hPa)", value=1013.0, step=1.0)
        qnh_inhg = hpa_to_inhg(qnh_hpa)
    wind_knots = st.number_input("Surface wind (kts)", value=0.0, step=1.0)
    wind_dir_deg = st.number_input("Wind direction (deg true)", value=float(row_rwy["heading_deg"]), step=1.0, min_value=0.0, max_value=359.9)

    st.header("Weight & Balance")
    weight_mode = st.radio("Weight entry", ["GW"], horizontal=True)
    gw_lbs = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)
    cg_percent_mac = st.number_input("CG (%MAC) — optional", value=25.0, step=0.5)

    st.header("Configuration")
    model = st.selectbox("Model", ["F-14B", "F-14D"])
    flap_sel = st.selectbox("Takeoff Flaps", ["Auto", "UP", "MAN", "FULL"], index=0)
    thrust_sel = st.selectbox("Takeoff Thrust", ["Auto", "MIL", "AB"], index=0)

if st.button("Compute Takeoff Performance", type="primary"):
    inp = Inputs(
        model=model,
        theatre=theatre,
        airport=airport,
        runway_label=rw,
        heading_deg=float(row_rwy["heading_deg"]),
        tora_ft=float(row_rwy["tora_ft"]),
        toda_ft=float(row_rwy["toda_ft"]),
        asda_ft=float(row_rwy["asda_ft"]),
        elev_ft=float(row_rwy["threshold_elev_ft"]),
        slope_pct=float(row_rwy.get("slope_percent", 0.0) or 0.0),
        shorten_ft=float(shorten_ft),
        oat_c=float(oat_c), qnh_inhg=float(qnh_inhg),
        wind_knots=float(wind_knots), wind_dir_deg=float(wind_dir_deg),
        weight_mode=weight_mode, gw_lbs=float(gw_lbs), cg_percent_mac=float(cg_percent_mac),
        flap_sel=flap_sel, thrust_sel=thrust_sel,
    )
    try:
        perf = compute_performance(inp, perfdb)
        ok = perf.req_distance_ft <= perf.avail_distance_ft

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("V‑Speeds")
            st.metric("V1 (kt)", f"{perf.v1:.0f}")
            st.metric("Vr (kt)", f"{perf.vr:.0f}")
            st.metric("V2 (kt)", f"{perf.v2:.0f}")
            if not math.isnan(perf.vs):
                st.metric("Vs (kt)", f"{perf.vs:.0f}")
        with col2:
            st.subheader("Settings")
            st.metric("Flaps", perf.flap_setting)
            st.metric("Thrust", perf.thrust_setting)
            st.caption("Trim lookup can be added later via a trim table.")
        with col3:
            st.subheader("Runway")
            st.metric("Req. Distance (ft)", f"{perf.req_distance_ft:.0f}", help="max(ASD, AGD) with wind/slope adjustments")
            st.metric("Avail. Distance (ft)", f"{perf.avail_distance_ft:.0f}")
            st.metric("Limiting", perf.limiting)

        st.markdown("✅ **" + ("TAKEOFF POSSIBLE" if ok else "TAKEOFF NOT PERMITTED") + "** — with TORA as available distance.")
        if not ok:
            st.warning("Try AB, more flap, lower weight, different runway, or cooler conditions.")

        with st.expander("Debug details"):
            st.json(vars(inp))

    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.stop()
else:
    st.info("Load your CSVs into data/ and click **Compute Takeoff Performance**.")

################################################################################
# NOTES for true balanced‑field V1 search (future)
################################################################################
# Your perf CSVs currently carry a single V1_kt and the associated ASD/AGD per condition.
# To enable a real balanced‑field search, add multiple rows per (model, flap_deg, thrust, gw, PA, OAT)
# where V1_kt varies (e.g., V1‑5, V1, V1+5, …) and ASD_ft/AGD_ft reflect that V1.
# Then we can sweep V1 and pick the row where |ASD‑AGD| is minimized (or take smallest max).
