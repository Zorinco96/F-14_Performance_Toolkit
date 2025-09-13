# app.py — DCS F‑14B Takeoff Performance (Streamlit)
# External files expected in repo:
#   • dcs_airports.csv  (runway DB)
#   • f14_stations.csv  (optional: stores list to compute GW from fuel & stores)
#
# Key features (V6):
# • Clean UI; no manual runway shortening.
# • Weather: Simple (OAT only) or Advanced (QNH + wind with 50/150 policy).
# • Weight: enter Gross Weight OR Fuel & Stores (uses f14_stations.csv; no CG).
# • Flaps: Auto‑Select defaults to MANEUVER; if limits not met → escalate to FULL.
# • Thrust (preferred interface): MIL / DERATE / AB.  
#   – DERATE range: 90–100% MIL, flap‑dependent floor enforced (UP=90, MAN=90, FULL=95).  
#   – Rule: **No reduced thrust with FULL flaps**.  
#   – "Find minimum N1%" button solves to meet 121.189 (field limits + OEI guardrail).
# • Estimated **stab trim (ANU)** from weight & flaps (training aid).
# • Model: F‑14B only. Do NOT use for real‑world ops.

import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F‑14B Takeoff Performance", page_icon="✈️", layout="wide")

# ------------------------------
# Embedded performance grid (F‑14B only)
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

# constants (approximate)
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 0.95}  # min eff (N1 fraction) by flap (UP/MAN=90, FULL=95)
ALPHA_N1_DIST = 1.12  # non-linear exponent for distance scaling vs N1 (tunable)
UP_FLAP_DISTANCE_FACTOR = 1.12  # UP (0°) tends to require more runway vs 20° baseline
EMPTY_WEIGHT_LB = 41780
INTERNAL_FUEL_MAX_LB = 16200  # approx F-14B internal fuel

# ---------- helpers ----------

def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return field_elev_ft + (29.92 - qnh_inhg) * 1000.0

def headwind_component(knots_wind: float, wind_dir_deg: float, rwy_heading_deg: float) -> float:
    delta = math.radians((wind_dir_deg - rwy_heading_deg) % 360)
    return knots_wind * math.cos(delta)

def wind_factored_component(wind_comp_kn: float, policy: str) -> float:
    if policy == "50/150":
        return 0.5 * wind_comp_kn if wind_comp_kn >= 0 else 1.5 * wind_comp_kn
    return 0.0

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
    perf = a.copy()
    perf["model"] = "F-14B"
    perf["thrust"] = perf["thrust"].str.upper()
    return perf

@st.cache_data
def load_stations() -> Optional[pd.DataFrame]:
    for path in ["f14_stations.csv", "data/f14_stations.csv"]:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            req = {"station_id", "name", "arm_in", "unit_weight_lb", "max_qty"}
            if not req.issubset(set(df.columns)):
                return None
            for c in ["unit_weight_lb", "max_qty"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception:
            continue
    return None

# ---------- core models ----------
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
    trim_units_anu: float


def flap_to_deg(sel: str) -> int:
    s = sel.upper()
    if s.startswith("UP"):
        return 0
    if s.startswith("MAN"):
        return 20
    return 40


def nearest_perf_row(perf: pd.DataFrame, flap_deg: int, thrust_mode: str,
                     gw_lbs: float, pa_ft: float, oat_c: float) -> pd.Series:
    sub = perf[(perf["flap_deg"] == flap_deg) & (perf["thrust"] == thrust_mode)]
    if sub.empty and flap_deg == 0:
        sub = perf[(perf["flap_deg"] == 20) & (perf["thrust"] == thrust_mode)]
    if sub.empty:
        sub = perf[(perf["flap_deg"] == (20 if flap_deg == 0 else flap_deg))]
    sub = sub.assign(d_w=(sub["gw_lbs"] - gw_lbs).abs(), d_pa=(sub["press_alt_ft"] - pa_ft).abs(), d_t=(sub["oat_c"] - oat_c).abs())
    return sub.sort_values(["d_w", "d_pa", "d_t"]).iloc[0]


def dist_with_adjustments(base_ft: float, slope_pct: float, headwind_kn: float, wind_policy: str) -> float:
    d = float(base_ft)
    d = apply_slope(d, slope_pct)
    wfac = wind_factored_component(headwind_kn, wind_policy)
    if wfac != 0.0:
        d = apply_wind(d, wfac)
    return max(d, 0.0)


def enforce_derate_floor(n1pct: float, flap_deg: int) -> float:
    floor = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90) * 100.0
    return max(n1pct, floor)


def distance_scale_factor_from_n1(n1pct: float, flap_deg: int) -> float:
    n1pct = enforce_derate_floor(n1pct, flap_deg)
    eff = max(0.90, min(1.0, n1pct / 100.0))
    return 1.0 / (eff ** ALPHA_N1_DIST)


def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    n1pct = enforce_derate_floor(n1pct, flap_deg)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct / 100.0)
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024


def solve_min_derate_for_121(row: pd.Series, tora_ft: float, toda_ft: float, asda_ft: float,
                             slope_pct: float, headwind_kn: float, wind_policy: str,
                             flap_deg: int, gw_lbs: float) -> float:
    base_asd = float(row["ASD_ft"]) ; base_agd = float(row["AGD_ft"])

    def ok(n1pct: float) -> bool:
        mult = distance_scale_factor_from_n1(n1pct, flap_deg)
        asd = dist_with_adjustments(base_asd * mult, slope_pct, headwind_kn, wind_policy)
        agd = dist_with_adjustments(base_agd * mult, slope_pct, headwind_kn, wind_policy)
        if flap_deg == 0:
            asd *= UP_FLAP_DISTANCE_FACTOR
            agd *= UP_FLAP_DISTANCE_FACTOR
        clearway_allow = min(tora_ft * 0.5, max(0.0, toda_ft - tora_ft))
        tod_limit = tora_ft + clearway_allow
        field_ok = (asd <= asda_ft) and (agd <= tod_limit) and (agd <= toda_ft)
        climb_ok = compute_oei_second_segment_ok(gw_lbs, n1pct, flap_deg)
        return field_ok and climb_ok

    floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90) * 100.0
    if not ok(100.0):
        return 101.0
    lo, hi = floor_pct, 100.0
    for _ in range(16):
        mid = (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return round(hi, 1)


def compute_trim_units_anu(gw_lbs: float, flap_deg: int) -> float:
    """Very simple training‑aid trim estimator (ANU) from GW & flaps.
    Baseline MAN(20°)=4.5 ANU @ 60,000 lb; ±0.8 per 10k change; UP −1.0; FULL +1.0; clamp to 2–8.
    """
    base = 4.5 + (gw_lbs - 60000.0) / 10000.0 * 0.8
    if flap_deg == 0:
        base -= 1.0
    elif flap_deg == 40:
        base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))


def compute_performance(perfdb: pd.DataFrame, heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                        elev_ft: float, slope_pct: float, oat_c: float, qnh_inhg: float,
                        wind_knots: float, wind_dir_deg: float, wind_policy: str, gw_lbs: float,
                        flap_sel: str, thrust_mode: str, derate_n1_pct: float) -> PerfResult:
    pa_ft = pressure_altitude_ft(elev_ft, qnh_inhg)
    headwind = headwind_component(wind_knots, wind_dir_deg, heading_deg)

    flap_deg = flap_to_deg(flap_sel)

    table_thrust = "MILITARY" if thrust_mode in ("MIL", "DERATE") else "AFTERBURNER"
    row = nearest_perf_row(perfdb, flap_deg, table_thrust, gw_lbs, pa_ft, oat_c)

    v1 = float(row["V1_kt"]) ; vr = float(row["Vr_kt"]) ; v2 = float(row["V2_kt"]) ; vs = float(row.get("Vs_kt", np.nan))

    base_asd = float(row["ASD_ft"]) ; base_agd = float(row["AGD_ft"]) ; mult = 1.0
    n1_used_pct = 100.0
    if thrust_mode == "DERATE":
        n1_used_pct = enforce_derate_floor(derate_n1_pct, flap_deg)
        mult = distance_scale_factor_from_n1(n1_used_pct, flap_deg)
    asd_eff = dist_with_adjustments(base_asd * mult, slope_pct, headwind, wind_policy)
    agd_eff = dist_with_adjustments(base_agd * mult, slope_pct, headwind, wind_policy)

    if flap_deg == 0:
        asd_eff *= UP_FLAP_DISTANCE_FACTOR
        agd_eff *= UP_FLAP_DISTANCE_FACTOR

    clearway_allow = min(tora_ft * 0.5, max(0.0, toda_ft - tora_ft))
    tod_limit = tora_ft + clearway_allow

    req_ft = max(asd_eff, agd_eff)
    avail_ft = float(tora_ft)
    limiting = "ASD" if asd_eff >= agd_eff else "AGD"

    trim = compute_trim_units_anu(gw_lbs, flap_deg)

    return PerfResult(v1, vr, v2, vs, flap_sel, thrust_mode, req_ft, avail_ft, limiting, n1_used_pct, trim)

# ---------- selection helpers ----------

def meets_limits(pr: PerfResult, rwy: dict, gw_lbs: float) -> bool:
    field_ok = (pr.req_distance_ft <= pr.avail_distance_ft) and (pr.req_distance_ft <= rwy['toda']) and (pr.req_distance_ft <= rwy['asda'])
    if pr.thrust_label == "DERATE":
        flap_deg = flap_to_deg(pr.flap_setting)
        field_ok = field_ok and compute_oei_second_segment_ok(gw_lbs, pr.n1_used_pct, flap_deg)
    return field_ok


def choose_flap_for_limits(perfdb: pd.DataFrame, rwy: dict, env: dict, gw_lbs: float,
                           flap_sel: str, thrust_mode: str, derate_n1_pct: float) -> PerfResult:
    # If user specified flaps explicitly, honor it (with rule: no derate on FULL)
    if flap_sel != "Auto-Select":
        if flap_sel == "FULL" and thrust_mode == "DERATE":
            # Enforce no-reduced-thrust-with-FULL
            return compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                       env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                       gw_lbs, flap_sel, "MIL", 100.0)
        return compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                   env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                   gw_lbs, flap_sel, thrust_mode, derate_n1_pct)

    # Auto-Select: default to MANEUVER; only escalate to FULL if MAN cannot meet limits.
    # Try MAN
    pr_man = compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                 env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                 gw_lbs, "MANEUVER", ("MIL" if thrust_mode == "DERATE" and False else thrust_mode), derate_n1_pct)
    if meets_limits(pr_man, rwy, gw_lbs):
        return pr_man

    # Escalate to FULL. If user set DERATE, switch to MIL by rule.
    thrust_for_full = "MIL" if thrust_mode == "DERATE" else thrust_mode
    pr_full = compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                  env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                  gw_lbs, "FULL", thrust_for_full, derate_n1_pct)
    return pr_full

# ------------------------------
# Input parsing helpers (UI)
# ------------------------------
import re
WIND_PAT = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*(?:@|/|\s)\s*(\d{1,3}(?:\.\d+)?)\s*$")

def parse_wind_entry(entry: str, unit: str) -> Optional[tuple]:
    m = WIND_PAT.match(entry or "")
    if not m:
        return None
    d = float(m.group(1)) % 360.0
    s = float(m.group(2))
    if unit == "m/s":
        s = s * 1.943844
    return (d, s)

# ------------------------------
# UI
# ------------------------------
st.title("DCS F‑14B Takeoff Performance")

rwy_df = load_runways()
perfdb = load_perf()
stations_df = load_stations()

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

    st.header("Weather")
    wx_mode = st.radio("Input mode", ["Simple", "Advanced"], horizontal=True, index=0)
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)

    if wx_mode == "Advanced":
        qnh_val = st.number_input("QNH value", value=29.92, step=0.01, format="%.2f")
        qnh_unit = st.selectbox("QNH Units", ["inHg", "hPa"], index=0)
        qnh_inhg = float(qnh_val) if qnh_unit == "inHg" else hpa_to_inhg(float(qnh_val))
        wind_unit = st.selectbox("Wind Units", ["kts", "m/s"], index=0)
        wind_entry = st.text_input("Wind (DIR@SPD)", placeholder="e.g., 180@12")
        wind_parsed = parse_wind_entry(wind_entry, wind_unit)
        if wind_parsed is None and (wind_entry or "") != "":
            st.warning("Enter wind as DDD@SS, DDD/SS, or DDD SS. Example: 180@12")
            wind_dir_deg, wind_knots = float(row_rwy["heading_deg"]), 0.0
        else:
            wind_dir_deg, wind_knots = (wind_parsed if wind_parsed is not None else (float(row_rwy["heading_deg"]), 0.0))
        use_50150 = st.checkbox("Conservative wind policy (50/150)", value=False)
        wind_policy = "50/150" if use_50150 else "None"
    else:
        qnh_inhg = 29.92
        wind_dir_deg, wind_knots = float(row_rwy["heading_deg"]), 0.0
        wind_policy = "None"

    st.header("Weight & Config")
    weight_mode = st.radio("Set weight by", ["Gross Weight", "Fuel & Stores"], horizontal=True)

    if weight_mode == "Gross Weight":
        gw_lbs = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)
        computed_breakdown = None
    else:
        st.caption("Internal fuel + selected stores (no CG)")
        fuel_input_mode = st.radio("Fuel input", ["Percent", "Pounds"], horizontal=True)
        if fuel_input_mode == "Percent":
            fuel_pct = st.slider("Internal fuel %", 0, 100, 80, 1)
            internal_fuel_lb = INTERNAL_FUEL_MAX_LB * (fuel_pct / 100.0)
        else:
            internal_fuel_lb = st.number_input("Internal fuel (lb)", min_value=0.0, max_value=float(INTERNAL_FUEL_MAX_LB), value=float(0.8*INTERNAL_FUEL_MAX_LB), step=100.0)
        stores_weight = 0.0
        chosen = {}
        if stations_df is not None:
            with st.expander("Select Stores"):
                for _, r in stations_df.iterrows():
                    qty = st.number_input(f"{r['name']} (max {int(r['max_qty'])})", min_value=0, max_value=int(r['max_qty']), value=0, step=1, key=f"st_{r['station_id']}")
                    if qty:
                        chosen[r['station_id']] = qty
                        stores_weight += qty * float(r['unit_weight_lb'])
        else:
            st.info("Add f14_stations.csv to enable store selection.")
        gw_lbs = EMPTY_WEIGHT_LB + internal_fuel_lb + stores_weight
        computed_breakdown = {
            "Empty": EMPTY_WEIGHT_LB,
            "InternalFuel": round(internal_fuel_lb, 0),
            "Stores": round(stores_weight, 0),
            "GW": round(gw_lbs, 0),
        }

    st.subheader("Configuration")
    col_f, col_t = st.columns(2)
    with col_f:
        flap_sel = st.radio("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], horizontal=True, index=0)
    with col_t:
        thrust_mode = st.radio("Thrust", ["MIL", "DERATE", "AB"], horizontal=True, index=0)
        derate_n1_pct = 100.0
        if thrust_mode == "DERATE":
            flap_deg_hint = {0: "≥90%", 20: "≥90%", 40: "≥95%"}[flap_to_deg(flap_sel if flap_sel != "Auto-Select" else "MANEUVER")]
            st.caption(f"Derate floor by flap (approx): {flap_deg_hint}")
            derate_n1_pct = st.slider("Target N1 % (MIL)", min_value=90.0, max_value=100.0, value=98.0, step=0.5, key="derate_n1_slider")
            if st.button("Find minimum N1% to meet 121.189"):
                pa_ft = pressure_altitude_ft(float(row_rwy["threshold_elev_ft"]), float(qnh_inhg))
                chosen_flap = flap_sel if flap_sel != "Auto-Select" else "MANEUVER"
                flap_deg = flap_to_deg(chosen_flap)
                rowp = nearest_perf_row(perfdb, flap_deg, "MILITARY", gw_lbs, pa_ft, oat_c)
                min_n1 = solve_min_derate_for_121(rowp, float(row_rwy["tora_ft"]), float(row_rwy["toda_ft"]), float(row_rwy["asda_ft"]),
                                                  float(row_rwy.get("slope_percent", 0.0) or 0.0),
                                                  headwind_component(wind_knots, wind_dir_deg, float(row_rwy["heading_deg"])),
                                                  str(wind_policy), flap_deg, float(gw_lbs))
                if min_n1 > 100.0:
                    st.error("Cannot meet 121.189 with derate in this model. Increase thrust or change config.")
                else:
                    st.success(f"Minimum N1%: {min_n1:.1f}")

# compute
if st.button("Compute Takeoff Performance", type="primary"):
    try:
        rwy = {
            'hdg': float(row_rwy["heading_deg"]),
            'tora': float(row_rwy["tora_ft"]),
            'toda': float(row_rwy["toda_ft"]),
            'asda': float(row_rwy["asda_ft"]),
            'elev': float(row_rwy["threshold_elev_ft"]),
            'slope': float(row_rwy.get("slope_percent", 0.0) or 0.0),
        }
        env = {
            'oat': float(oat_c),
            'qnh': float(qnh_inhg),
            'wind_kn': float(wind_knots),
            'wind_dir': float(wind_dir_deg),
            'wind_policy': str(wind_policy),
        }

        # Selection logic: Auto flaps only; thrust is manual per your preference
        def meets_limits(pr: PerfResult, rwy: dict, gw_lbs: float) -> bool:
            field_ok = (pr.req_distance_ft <= pr.avail_distance_ft) and (pr.req_distance_ft <= rwy['toda']) and (pr.req_distance_ft <= rwy['asda'])
            if pr.thrust_label == "DERATE":
                flap_deg = flap_to_deg(pr.flap_setting)
                field_ok = field_ok and compute_oei_second_segment_ok(gw_lbs, pr.n1_used_pct, flap_deg)
            return field_ok

        def choose_flap_for_limits(perfdb: pd.DataFrame, rwy: dict, env: dict, gw_lbs: float,
                                   flap_sel: str, thrust_mode: str, derate_n1_pct: float) -> PerfResult:
            # If explicit flap selected, honor it (enforce no-derate-with-FULL)
            if flap_sel != "Auto-Select":
                if flap_sel == "FULL" and thrust_mode == "DERATE":
                    return compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                               env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                               gw_lbs, flap_sel, "MIL", 100.0)
                return compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                           env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                           gw_lbs, flap_sel, thrust_mode, derate_n1_pct)
            # Auto → MAN first
            pr_man = compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                         env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                         gw_lbs, "MANEUVER", thrust_mode, derate_n1_pct)
            if meets_limits(pr_man, rwy, gw_lbs):
                return pr_man
            # Escalate to FULL (and drop DERATE → MIL by rule)
            t_for_full = "MIL" if thrust_mode == "DERATE" else thrust_mode
            pr_full = compute_performance(perfdb, rwy['hdg'], rwy['tora'], rwy['toda'], rwy['asda'], rwy['elev'], rwy['slope'],
                                          env['oat'], env['qnh'], env['wind_kn'], env['wind_dir'], env['wind_policy'],
                                          gw_lbs, "FULL", t_for_full, derate_n1_pct)
            return pr_full

        perf = choose_flap_for_limits(perfdb, rwy, env, float(gw_lbs), flap_sel, thrust_mode, float(derate_n1_pct))
        ok = meets_limits(perf, rwy, float(gw_lbs))

        c1, c2, c3, c4, c5 = st.columns(5)
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
            thrust_label = perf.thrust_label + (f" ({perf.n1_used_pct:.1f}% N1)" if perf.thrust_label=="DERATE" else "")
            st.metric("Thrust", thrust_label)
            st.metric("Trim (ANU)", f"{perf.trim_units_anu:.1f}")
        with c3:
            st.subheader("Runway")
            st.metric("Required (ft)", f"{perf.req_distance_ft:.0f}")
            st.metric("Available (ft)", f"{perf.avail_distance_ft:.0f}")
            st.metric("Limiting", perf.limiting)
        with c4:
            st.subheader("Compliance")
            clearway_allow = min(rwy['tora'] * 0.5, max(0.0, rwy['toda'] - rwy['tora']))
            st.metric("TOD Limit (ft)", f"{(rwy['tora'] + clearway_allow):.0f}")
            st.caption("Clearway credit capped at 50% of runway length.")
        with c5:
            st.subheader("Status")
            st.markdown("✅ **DISPATCH OK (approx)**" if ok else "❌ **NOT PERMITTED** (121.189 field limits)")

        if 'computed_breakdown' in locals() and computed_breakdown is not None:
            with st.expander("Weight breakdown"):
                st.json(computed_breakdown)

    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.stop()
else:
    st.info("Set inputs and click Compute Takeoff Performance. You can use DERATE + 'Find minimum N1%' to test floors at hot/high.")
