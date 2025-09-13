# app.py — DCS F‑14B Takeoff Performance (Streamlit)
# External files expected in repo:
#   • dcs_airports.csv  (runway DB)
#   • f14_stations.csv  (optional: stores stations, arms, unit weights)
#   • f14_airframe.json (optional: {"le_mac_in": 0.0, "mac_in": 0.0, "empty_weight_lb": 41780})
#
# Key features:
# • Flaps: UP / MANEUVER / FULL / Auto‑Select. Flaps impact distances.
# • Thrust: MIL / AB / DERATE (90–100% N1). “Find Min N1” solver targets 14 CFR 121.189 field‑length checks.
# • Weather: manual entry; wind units kts or m/s (optional 50/150 wind credit toggle).
# • Runway shortening: feet and/or nautical miles.
# • Model: F‑14B only; F‑14D table rows blended into perf grid for denser interpolation.
# • CG & Trim (experimental): If stations + airframe geometry provided, compute %MAC and an estimated stab‑trim.
#
# DISCLAIMER: Training aid for DCS only. Do NOT use for real‑world flight planning.

import json
import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F‑14B Takeoff Performance", page_icon="✈️", layout="wide")

# ------------------------------
# Embedded performance grid (F‑14B + selected F‑14D rows relabeled as B)
# ------------------------------
PERF_F14B = """model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note
F-14B,20,Military,60000,0,0,120,132,141,154,4300,5200,B_est
F-14B,20,Military,60000,0,30,123,135,144,157,4700,5600,B_est
F-14B,20,Military,60000,5000,0,123,136,146,158,5200,6100,B_est
F-14B,20,Military,60000,5000,30,126,139,149,162,5600,6600,B_est
F-14B,20,Military,65000,0,0,123,136,146,159,4800,5700,B_est
F-14B,20,Military,65000,0,30,126,139,150,162,5200,6200,B_est
F-14B,20,Military,65000,5000,0,126,139,150,162,5600,6600,B_est
F-14B,20,Military,65000,5000,30,129,143,154,167,6000,7100,B_est
F-14B,20,Military,70000,0,0,126,140,151,164,5300,6300,B_est
F-14B,20,Military,70000,0,30,129,143,155,168,5800,6900,B_est
F-14B,20,Military,70000,5000,0,130,144,156,169,6200,7400,B_est
F-14B,20,Military,70000,5000,30,133,147,160,173,6700,7900,B_est
F-14B,20,Military,74300,0,0,128,142,154,167,5600,6700,B_est
F-14B,20,Military,74300,0,30,131,145,158,171,6100,7300,B_est
F-14B,20,Military,74300,5000,0,133,147,160,173,6600,7900,B_est
F-14B,20,Military,74300,5000,30,136,151,164,178,7100,8500,B_est
F-14B,20,Afterburner,60000,0,0,118,130,139,152,3900,4200,B_est
F-14B,20,Afterburner,60000,0,30,120,132,141,154,4200,4500,B_est
F-14B,20,Afterburner,60000,5000,0,120,133,143,156,4400,4700,B_est
F-14B,20,Afterburner,60000,5000,30,122,135,145,158,4700,5000,B_est
F-14B,20,Afterburner,65000,0,0,121,133,143,156,4200,4500,B_est
F-14B,20,Afterburner,65000,0,30,123,136,146,159,4500,4900,B_est
F-14B,20,Afterburner,65000,5000,0,123,136,147,160,4800,5200,B_est
F-14B,20,Afterburner,65000,5000,30,125,138,149,162,5200,5600,B_est
F-14B,20,Afterburner,70000,0,0,124,137,148,161,4500,4900,B_est
F-14B,20,Afterburner,70000,0,30,126,139,151,164,4900,5400,B_est
F-14B,20,Afterburner,70000,5000,0,127,140,152,165,5300,5900,B_est
F-14B,20,Afterburner,70000,5000,30,129,143,155,168,5800,6500,B_est
F-14B,20,Afterburner,74300,0,0,126,139,151,164,4800,5300,B_est
F-14B,20,Afterburner,74300,0,30,128,141,153,167,5200,5800,B_est
F-14B,20,Afterburner,74300,5000,0,130,143,156,169,5700,6400,B_est
F-14B,20,Afterburner,74300,5000,30,132,145,158,172,6200,7000,B_est
F-14B,40,Military,60000,0,0,116,128,137,149,4200,5100,B_est
F-14B,40,Military,60000,0,30,118,130,139,152,4500,5500,B_est
F-14B,40,Military,60000,5000,0,118,131,141,153,5000,6000,B_est
F-14B,40,Military,60000,5000,30,120,133,143,156,5400,6500,B_est
F-14B,40,Military,70000,0,0,123,136,146,158,5200,6200,B_est
F-14B,40,Military,70000,0,30,126,139,149,162,5700,6900,B_est
F-14B,40,Military,70000,5000,0,128,141,152,165,6200,7600,B_est
F-14B,40,Military,70000,5000,30,131,144,156,169,6800,8300,B_est
F-14B,40,Afterburner,60000,0,0,114,126,135,148,3800,4100,B_est
F-14B,40,Afterburner,60000,0,30,116,128,137,150,4000,4300,B_est
F-14B,40,Afterburner,60000,5000,0,116,129,139,151,4300,4700,B_est
F-14B,40,Afterburner,60000,5000,30,118,131,141,154,4500,5000,B_est
F-14B,40,Afterburner,70000,0,0,121,134,144,156,4500,5400,B_est
F-14B,40,Afterburner,70000,0,30,123,136,147,159,5000,6100,B_est
F-14B,40,Afterburner,70000,5000,0,124,137,148,160,5400,6800,B_est
F-14B,40,Afterburner,70000,5000,30,126,139,151,163,6000,7500,B_est
"""

PERF_F14D_AS_B = """model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note
F-14B,20,Military,60000,0,0,121,133,142,155,4200,5100,D_est
F-14B,20,Military,60000,0,30,123,135,145,157,4600,5500,D_est
F-14B,20,Military,60000,5000,0,123,136,146,158,5100,6000,D_est
F-14B,20,Military,60000,5000,30,126,139,149,162,5500,6500,D_est
F-14B,20,Military,65000,0,0,124,137,147,160,4700,5600,D_est
F-14B,20,Military,65000,0,30,127,140,151,163,5200,6100,D_est
F-14B,20,Military,65000,5000,0,127,140,151,163,5600,6600,D_est
F-14B,20,Military,65000,5000,30,130,144,155,168,6000,7100,D_est
F-14B,20,Military,70000,0,0,127,141,152,165,5200,6100,D_est
F-14B,20,Military,70000,0,30,130,144,156,169,5700,6800,D_est
F-14B,20,Military,70000,5000,0,131,145,157,170,6200,7400,D_est
F-14B,20,Military,70000,5000,30,134,148,161,174,6700,8000,D_est
F-14B,20,Military,74300,0,0,129,143,155,168,5500,6500,D_est
F-14B,20,Military,74300,0,30,132,146,159,172,6000,7100,D_est
F-14B,20,Military,74300,5000,0,134,148,161,174,6500,7700,D_est
F-14B,20,Military,74300,5000,30,137,151,165,178,7000,8300,D_est
F-14B,20,Afterburner,60000,0,0,119,131,140,153,3800,4000,D_est
F-14B,20,Afterburner,60000,0,30,121,133,142,155,4100,4300,D_est
F-14B,20,Afterburner,60000,5000,0,121,134,144,156,4300,4600,D_est
F-14B,20,Afterburner,60000,5000,30,123,136,146,159,4600,4900,D_est
F-14B,20,Afterburner,65000,0,0,122,134,144,157,4100,4300,D_est
F-14B,20,Afterburner,65000,0,30,124,136,147,160,4400,4700,D_est
F-14B,20,Afterburner,65000,5000,0,124,137,148,161,4700,5100,D_est
F-14B,20,Afterburner,65000,5000,30,126,139,150,163,5100,5600,D_est
F-14B,40,Military,60000,0,0,116,128,137,149,4100,5000,D_est
F-14B,40,Military,60000,0,30,118,130,139,152,4400,5400,D_est
F-14B,40,Military,60000,5000,0,118,131,141,153,4900,5900,D_est
F-14B,40,Military,60000,5000,30,120,133,143,156,5300,6400,D_est
F-14B,40,Military,70000,0,0,123,136,146,158,5100,6100,D_est
F-14B,40,Military,70000,0,30,126,139,149,162,5600,6800,D_est
F-14B,40,Military,70000,5000,0,127,140,151,164,6100,7500,D_est
F-14B,40,Military,70000,5000,30,130,143,155,168,6700,8200,D_est
F-14B,40,Afterburner,60000,0,0,114,126,135,148,3700,4000,D_est
F-14B,40,Afterburner,60000,0,30,116,128,137,150,4000,4300,D_est
F-14B,40,Afterburner,60000,5000,0,116,129,139,151,4200,4600,D_est
F-14B,40,Afterburner,60000,5000,30,118,131,141,154,4500,5000,D_est
F-14B,40,Afterburner,70000,0,0,121,134,144,156,4400,5200,D_est
F-14B,40,Afterburner,70000,0,30,123,136,147,159,4900,5900,D_est
F-14B,40,Afterburner,70000,5000,0,123,136,147,160,5300,6600,D_est
F-14B,40,Afterburner,70000,5000,30,125,138,149,162,5900,7300,D_est
"""

# constants (approximate)
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}

# ---------- helpers ----------

def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return field_elev_ft + (29.92 - qnh_inhg) * 1000.0

def headwind_component(knots_wind: float, wind_dir_deg: float, rwy_heading_deg: float) -> float:
    delta = math.radians((wind_dir_deg - rwy_heading_deg) % 360)
    return knots_wind * math.cos(delta)

def wind_factored_component(wind_comp_kn: float) -> float:
    if wind_comp_kn >= 0:
        return 0.5 * wind_comp_kn
    else:
        return 1.5 * wind_comp_kn

def apply_slope(distance_ft: float, slope_pct: float) -> float:
    return distance_ft * (1.0 + max(0.0, slope_pct) * 0.20)

def apply_wind(distance_ft: float, factored_wind_kn: float) -> float:
    return distance_ft * (1.0 - 0.005 * factored_wind_kn)

# ---------- loaders ----------
@st.cache_data
def load_runways() -> pd.DataFrame:
    for path in ["dcs_airports.csv", "data/dcs_airports.csv"]:
        try:
            df = pd.read_csv(path)
            df["runway_label"] = df["airport_name"] + " " + df["runway_end"].astype(str) + " (" + df["runway_pair"].astype(str) + ")"
            return df
        except Exception:
            continue
    st.error("dcs_airports.csv not found. Add it to the repo (exact case).")
    st.stop()

@st.cache_data
def load_perf() -> pd.DataFrame:
    a = pd.read_csv(StringIO(PERF_F14B))
    b = pd.read_csv(StringIO(PERF_F14D_AS_B))
    perf = pd.concat([a, b], ignore_index=True)
    perf["model"] = "F-14B"
    perf["thrust"] = perf["thrust"].str.upper()
    return perf

@st.cache_data
def load_stations() -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv("f14_stations.csv")
    except Exception:
        return None

@st.cache_data
def load_airframe_defaults() -> Dict[str, float]:
    try:
        with open("f14_airframe.json", "r") as f:
            data = json.load(f)
            return {
                "le_mac_in": float(data.get("le_mac_in", float("nan"))),
                "mac_in": float(data.get("mac_in", float("nan"))),
                "empty_weight_lb": float(data.get("empty_weight_lb", 41780.0)),
            }
    except Exception:
        return {"le_mac_in": float("nan"), "mac_in": float("nan"), "empty_weight_lb": 41780.0}

# ---------- core models ----------
@dataclass
class Inputs:
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
    use_wind_credit: bool
    gw_lbs: float
    flap_sel: str  # UP / MANEUVER / FULL / Auto-Select
    thrust_mode: str  # MIL / AB / DERATE
    derate_n1_pct: float
    cg_percent_mac: Optional[float]

@dataclass
class PerfResult:
    v1: float
    vr: float
    v2: float
    vs: float
    flap_setting: str
    thrust_label: str
    req_distance_ft: float
    avail_distance_ft: float
    limiting: str
    n1_used_pct: float
    cg_percent_mac: Optional[float]
    trim_units_anu: Optional[float]


def flap_to_deg(sel: str) -> int:
    s = sel.upper()
    if s.startswith("UP"):
        return 0
    if s.startswith("MAN"):
        return 20
    return 40


def auto_flaps(gw_lbs: float, pa_ft: float, oat_c: float) -> str:
    if gw_lbs >= 70000 or pa_ft > 3000 or oat_c > 30:
        return "FULL"
    elif gw_lbs >= 62000:
        return "MANEUVER"
    return "UP"


def nearest_perf_row(perf: pd.DataFrame, flap_deg: int, thrust_mode: str,
                     gw_lbs: float, pa_ft: float, oat_c: float) -> pd.Series:
    sub = perf[(perf["flap_deg"] == flap_deg) & (perf["thrust"] == thrust_mode)]
    if sub.empty:
        sub = perf[(perf["flap_deg"] == flap_deg)]
    sub = sub.assign(d_w=(sub["gw_lbs"] - gw_lbs).abs(), d_pa=(sub["press_alt_ft"] - pa_ft).abs(), d_t=(sub["oat_c"] - oat_c).abs())
    return sub.sort_values(["d_w", "d_pa", "d_t"]).iloc[0]


def dist_with_adjustments(base_ft: float, slope_pct: float, headwind_kn: float, use_wind_credit: bool) -> float:
    d = float(base_ft)
    d = apply_slope(d, slope_pct)
    if use_wind_credit:
        d = apply_wind(d, wind_factored_component(headwind_kn))
    return max(d, 0.0)


def compute_trim_units_anu(cg_pct_mac: Optional[float], flap_deg: int) -> Optional[float]:
    if cg_pct_mac is None or math.isnan(cg_pct_mac):
        return None
    base = 3.5 + 0.10 * (cg_pct_mac - 20.0)
    if flap_deg == 0:
        base -= 0.8
    elif flap_deg == 40:
        base += 0.8
    return max(0.0, round(base, 1))


def compute_oei_second_segment_ok(gw_lbs: float, n1_pct: float, flap_deg: int) -> bool:
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1_pct / 100.0)
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024


def solve_min_derate_for_121(row: pd.Series, tora_ft: float, toda_ft: float, asda_ft: float,
                             slope_pct: float, headwind_kn: float, use_wind_credit: bool,
                             flap_deg: int, gw_lbs: float) -> float:
    base_asd = float(row["ASD_ft"])
    base_agd = float(row["AGD_ft"])

    def ok(n1pct: float) -> bool:
        eff = max(0.9, min(1.0, n1pct / 100.0))
        asd = dist_with_adjustments(base_asd / eff, slope_pct, headwind_kn, use_wind_credit)
        agd = dist_with_adjustments(base_agd / eff, slope_pct, headwind_kn, use_wind_credit)
        clearway_allow = min(tora_ft * 0.5, max(0.0, toda_ft - tora_ft))
        tod_limit = tora_ft + clearway_allow
        field_ok = (asd <= asda_ft) and (agd <= tod_limit) and (agd <= toda_ft)
        climb_ok = compute_oei_second_segment_ok(gw_lbs, n1pct, flap_deg)
        return field_ok and climb_ok

    if not ok(100.0):
        return 101.0
    lo, hi = 90.0, 100.0
    for _ in range(16):
        mid = (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return round(hi, 1)


def compute_performance(perfdb: pd.DataFrame, heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                        elev_ft: float, slope_pct: float, shorten_ft: float, oat_c: float, qnh_inhg: float,
                        wind_knots: float, wind_dir_deg: float, use_wind_credit: bool, gw_lbs: float,
                        flap_sel: str, thrust_mode: str, derate_n1_pct: float,
                        cg_percent_mac: Optional[float]) -> PerfResult:
    pa_ft = pressure_altitude_ft(elev_ft, qnh_inhg)
    headwind = headwind_component(wind_knots, wind_dir_deg, heading_deg)

    flap = flap_sel if flap_sel != "Auto-Select" else auto_flaps(gw_lbs, pa_ft, oat_c)
    flap_deg = flap_to_deg(flap)

    table_thrust = "MILITARY" if thrust_mode in ("MIL", "DERATE") else "AFTERBURNER"
    row = nearest_perf_row(perfdb, flap_deg, table_thrust, gw_lbs, pa_ft, oat_c)

    v1 = float(row["V1_kt"]) ; vr = float(row["Vr_kt"]) ; v2 = float(row["V2_kt"]) ; vs = float(row.get("Vs_kt", np.nan))

    base_asd = float(row["ASD_ft"]) ; base_agd = float(row["AGD_ft"]) ; eff = 1.0
    if thrust_mode == "DERATE":
        eff = max(0.90, min(1.0, derate_n1_pct / 100.0))
    asd_eff = dist_with_adjustments(base_asd / eff, slope_pct, headwind, use_wind_credit)
    agd_eff = dist_with_adjustments(base_agd / eff, slope_pct, headwind, use_wind_credit)

    clearway_allow = min(tora_ft * 0.5, max(0.0, toda_ft - tora_ft))
    tod_limit = tora_ft + clearway_allow

    req_ft = max(asd_eff, agd_eff)
    avail_ft = max(0.0, tora_ft - shorten_ft)
    limiting = "ASD" if asd_eff >= agd_eff else "AGD"

    trim_units = compute_trim_units_anu(cg_percent_mac, flap_deg)

    return PerfResult(v1, vr, v2, vs, flap, thrust_mode, req_ft, avail_ft, limiting, eff * 100.0, cg_percent_mac, trim_units)

# ------------------------------
# UI
# ------------------------------
st.title("DCS F‑14B Takeoff Performance")

rwy_df = load_runways()
perfdb = load_perf()
stations_df = load_stations()
airframe = load_airframe_defaults()

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Map", sorted(rwy_df["map"].unique()))
    df_t = rwy_df[rwy_df["map"] == theatre]
    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]
    rw = st.selectbox("Runway End", list(df_a["runway_label"]))
    row_rwy = df_a[df_a["runway_label"] == rw].iloc[0]

    st.metric("TORA (ft)", f"{int(row_rwy['tora_ft']):,}")
    st.metric("TODA (ft)", f"{int(row_rwy['toda_ft']):,}")
    st.metric("ASDA (ft)", f"{int(row_rwy['asda_ft']):,}")

    shorten_ft = st.number_input("Shorten available runway (ft)", min_value=0.0, value=0.0, step=100.0)
    shorten_nm = st.number_input("Shorten also by (NM)", min_value=0.0, value=0.0, step=0.1)
    shorten_total_ft = float(shorten_ft) + float(shorten_nm) * 6076.12

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    qnh_unit = st.selectbox("QNH Units", ["inHg", "hPa"], index=0)
    if qnh_unit == "inHg":
        qnh_inhg = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    else:
        qnh_hpa = st.number_input("QNH (hPa)", value=1013.0, step=1.0)
        qnh_inhg = hpa_to_inhg(qnh_hpa)
    wind_units = st.selectbox("Wind Units", ["kts", "m/s"], index=0)
    wind_dir_deg = st.number_input("Wind direction (deg true)", value=float(row_rwy["heading_deg"]), step=1.0, min_value=0.0, max_value=359.9)
    wind_speed = st.number_input(f"Wind speed ({wind_units})", value=0.0, step=1.0)
    wind_knots = wind_speed if wind_units == "kts" else wind_speed * 1.943844
    use_wind_credit = st.checkbox("Apply 50% headwind / 150% tailwind credit (approx)", value=False)

    st.header("Weight & Config")
    model = "F-14B"

    weight_mode = st.radio("How to set weight?", ["Enter Gross Weight", "Fuel + Stores (compute %MAC)"])
    cg_pct_mac = None

    if weight_mode == "Enter Gross Weight":
        gw_lbs = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)
    else:
        empty_weight = st.number_input("Empty Weight (lb)", value=float(airframe.get("empty_weight_lb", 41780.0)), step=10.0)
        fuel_weight = st.number_input("Internal Fuel (lb)", value=10000.0, step=100.0)
        stores_weight = 0.0
        if stations_df is None:
            st.warning("Add f14_stations.csv to enable stores list and CG. Expected columns: station_id,name,arm_in,unit_weight_lb,max_qty")
            stores_weight = st.number_input("Stores total weight (lb)", value=0.0, step=10.0)
        else:
            st.subheader("Stores selection")
            chosen = []
            for _, r in stations_df.iterrows():
                maxq = int(r.get("max_qty", 2) or 2)
                q = st.number_input(f"{r['name']} (station {r['station_id']}) qty", min_value=0, max_value=maxq, value=0, step=1)
                if q > 0:
                    chosen.append({"station_id": r["station_id"], "qty": q})
            if chosen:
                sel = pd.DataFrame(chosen)
                tmp = stations_df.merge(sel, on="station_id", how="inner")
                tmp["w"] = tmp["unit_weight_lb"].fillna(0) * tmp["qty"].fillna(0)
                stores_weight = tmp["w"].sum()
                # Simple %MAC using placeholder empty arm 300 in and fuel arm 300 in unless geometry provided
                try:
                    le_mac = float(airframe.get("le_mac_in", float("nan")))
                    mac_in = float(airframe.get("mac_in", float("nan")))
                    arm_empty = 300.0 ; arm_fuel = 300.0
                    w_empty = float(empty_weight) ; w_fuel = float(fuel_weight)
                    m_empty = w_empty * arm_empty ; m_fuel = w_fuel * arm_fuel
                    tmp["m"] = tmp["w"] * tmp["arm_in"].fillna(0)
                    w_tot = w_empty + w_fuel + tmp["w"].sum()
                    m_tot = m_empty + m_fuel + tmp["m"].sum()
                    if w_tot > 0 and not math.isnan(le_mac) and not math.isnan(mac_in) and mac_in > 0:
                        x_cg_in = m_tot / w_tot
                        cg_pct_mac = (x_cg_in - le_mac) / mac_in * 100.0
                        st.metric("Computed CG (%MAC)", f"{cg_pct_mac:.1f}")
                except Exception:
                    pass
        gw_lbs = empty_weight + fuel_weight + stores_weight

    flap_sel = st.selectbox("Takeoff Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    st.header("Thrust")
    thrust_mode = st.radio("Mode", ["MIL", "DERATE", "AB"], index=0)
    derate_n1_pct = 100.0
    if thrust_mode == "DERATE":
        derate_n1_pct = st.slider("Target N1 % (MIL)", min_value=90.0, max_value=100.0, value=98.0, step=0.5)
        if st.button("Find minimum N1% (approx) to meet 121.189"):
            pa_ft = pressure_altitude_ft(float(row_rwy["threshold_elev_ft"]), float(qnh_inhg))
            flap_deg = flap_to_deg(flap_sel if flap_sel != "Auto-Select" else auto_flaps(gw_lbs, pa_ft, oat_c))
            row = nearest_perf_row(perfdb, flap_deg, "MILITARY", gw_lbs, pa_ft, oat_c)
            m = solve_min_derate_for_121(row, float(row_rwy["tora_ft"]), float(row_rwy["toda_ft"]), float(row_rwy["asda_ft"]),
                                         float(row_rwy.get("slope_percent", 0.0) or 0.0),
                                         headwind_component(wind_knots, wind_dir_deg, float(row_rwy["heading_deg"])),
                                         bool(use_wind_credit), flap_deg, float(gw_lbs))
            if m > 100.0:
                st.error("Cannot meet 121.189 with derate in this model. Increase thrust or change config.")
            else:
                st.success(f"Minimum N1%: {m:.1f}")

# compute
if st.button("Compute Takeoff Performance", type="primary"):
    try:
        perf = compute_performance(perfdb,
                                   float(row_rwy["heading_deg"]), float(row_rwy["tora_ft"]), float(row_rwy["toda_ft"]), float(row_rwy["asda_ft"]),
                                   float(row_rwy["threshold_elev_ft"]), float(row_rwy.get("slope_percent", 0.0) or 0.0), float(shorten_total_ft),
                                   float(oat_c), float(qnh_inhg), float(wind_knots), float(wind_dir_deg), bool(use_wind_credit),
                                   float(gw_lbs), flap_sel, thrust_mode, float(derate_n1_pct), cg_pct_mac)

        ok = (perf.req_distance_ft <= perf.avail_distance_ft) and (perf.req_distance_ft <= float(row_rwy["toda_ft"])) and (perf.req_distance_ft <= float(row_rwy["asda_ft"]))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("V‑Speeds")
            st.metric("V1 (kt)", f"{perf.v1:.0f}")
            st.metric("Vr (kt)", f"{perf.vr:.0f}")
            st.metric("V2 (kt)", f"{perf.v2:.0f}")
            if not np.isnan(perf.vs):
                st.metric("Vs (kt)", f"{perf.vs:.0f}")
        with c2:
            st.subheader("Settings")
            st.metric("Flaps", perf.flap_setting)
            st.metric("Thrust", f"{perf.thrust_label} ({perf.n1_used_pct:.1f}% N1)")
            if perf.trim_units_anu is not None:
                st.metric("Stab Trim (ANU)", f"{perf.trim_units_anu:.1f}")
            else:
                st.caption("Trim estimate requires %MAC from stations; experimental.")
        with c3:
            st.subheader("Runway")
            st.metric("Required (ft)", f"{perf.req_distance_ft:.0f}")
            st.metric("Available (ft)", f"{perf.avail_distance_ft:.0f}")
            st.metric("Limiting", perf.limiting)
        with c4:
            st.subheader("Compliance")
            clearway_allow = min(float(row_rwy["tora_ft"]) * 0.5, max(0.0, float(row_rwy["toda_ft"]) - float(row_rwy["tora_ft"])) )
            st.metric("TOD Limit (ft)", f"{(float(row_rwy['tora_ft']) + clearway_allow):.0f}")
            st.caption("Clearway credit capped at 50% of runway length.")

        st.markdown("✅ " + ("DISPATCH OK (approx)" if ok else "NOT PERMITTED by 121.189 field limits"))

        with st.expander("Debug inputs"):
            st.json({
                "GW_lb": gw_lbs,
                "Flap": flap_sel,
                "Thrust": thrust_mode,
                "Derate_N1%": derate_n1_pct,
                "CG_%MAC": cg_pct_mac,
                "UseWindCredit": use_wind_credit,
            })
    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.stop()
else:
    st.info("Set inputs and click Compute Takeoff Performance.")
