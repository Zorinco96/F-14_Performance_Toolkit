# app.py — DCS F‑14B Takeoff Performance (Streamlit)
# External file required in repo root (exact case):
#   • dcs_airports.csv  (runway DB)
#
# V11 = V8 feature set + V10 calculation fixes
# • Tri‑linear interpolation (with safe linear extrapolation) across GW / PA / OAT
# • Auto‑flap policy: Auto tries MANEUVER first; escalates to FULL only if needed to meet 121.189
# • Rule: No reduced‑thrust (DERATE) with FULL flaps (calc auto‑uses MIL in that case)
# • Derate solver uses interpolated bases (not nearest row)
# • DCS AEO estimates: Vr & Liftoff (as fraction of Regulatory Required), auto‑cal from N1% or manual sliders
# • Trim (ANU) estimate from GW & flap shown alongside settings
# • Weather UI: OAT, QNH (value + units), single wind entry (DIR@SPD) w/ kts or m/s, 50/150 option
# • Runway: optional "Shorten Available" (ft or NM)
# • Thrust UI: MIL / DERATE / AB + "Find minimum N1%" button
# • Model: F‑14B only (no CG/stores, no F‑14D)
#
# DISCLAIMER: Training aid for DCS only. Do NOT use for real‑world flight planning.

import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional

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
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 0.95}  # min eff (N1 fraction) by flap
ALPHA_N1_DIST = 1.12  # non-linear exponent for distance scaling vs N1 (tunable)
UP_FLAP_DISTANCE_FACTOR = 1.12  # UP (0°) tends to require more runway vs 20° baseline

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
    # ~0.5% per knot of *factored* wind component
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

# ---------- interpolation (with extrapolation) ----------

def _bounds(vals, x):
    vals = sorted(set(map(float, vals)))
    lo = max([v for v in vals if v <= x], default=vals[0])
    hi = min([v for v in vals if v >= x], default=vals[-1])
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return lo, hi, w


def _interp_weight_at(sub: pd.DataFrame, pa: float, oat: float, field: str, gw_x: float) -> float:
    s = sub[(sub["press_alt_ft"] == pa) & (sub["oat_c"] == oat)].sort_values("gw_lbs")
    if s.empty:
        s = sub.sort_values(["press_alt_ft", "oat_c", "gw_lbs"])  # broad fallback
    xs = s["gw_lbs"].values.astype(float)
    ys = s[field].values.astype(float)

    if len(xs) < 2:
        return float(ys[0])

    if gw_x <= xs[0]:
        x0, x1 = xs[0], xs[1]
        y0, y1 = ys[0], ys[1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (gw_x - x0))
    if gw_x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]
        y0, y1 = ys[-2], ys[-1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (gw_x - x0))
    return float(np.interp(gw_x, xs, ys))


def interp_perf_values(perf: pd.DataFrame, flap_deg: int, thrust_mode: str, gw_x: float, pa_x: float, oat_x: float):
    # Fallback: use MAN(20°) table for UP(0°); apply UP factor to distances later
    use_flap = 20 if (flap_deg == 0) else flap_deg
    sub = perf[(perf["flap_deg"] == use_flap) & (perf["thrust"] == thrust_mode)]
    if sub.empty:
        sub = perf[perf["flap_deg"] == use_flap]

    pa0, pa1, wp = _bounds(sub["press_alt_ft"].unique(), pa_x)
    t0,  t1,  wt = _bounds(sub["oat_c"].unique(),        oat_x)

    fields = ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]
    out = {}

    for f in fields:
        v00 = _interp_weight_at(sub, pa0, t0, f, gw_x)
        v01 = _interp_weight_at(sub, pa0, t1, f, gw_x)
        v10 = _interp_weight_at(sub, pa1, t0, f, gw_x)
        v11 = _interp_weight_at(sub, pa1, t1, f, gw_x)
        v0  = v00*(1-wt) + v01*wt
        v1  = v10*(1-wt) + v11*wt
        out[f] = v0*(1-wp) + v1*wp

    return out

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
    dcs_vr_est_ft: float
    dcs_lo_est_ft: float


def flap_to_deg(sel: str) -> int:
    s = sel.upper()
    if s.startswith("UP"):
        return 0
    if s.startswith("MAN"):
        return 20
    return 40


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
    # Returns multiplier to apply to baseline distances
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


def solve_min_derate_for_121_from_bases(base_asd: float, base_agd: float, tora_ft: float, toda_ft: float, asda_ft: float,
                                        slope_pct: float, headwind_kn: float, wind_policy: str,
                                        flap_deg: int, gw_lbs: float) -> float:
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
    base = 4.5 + (gw_lbs - 60000.0) / 10000.0 * 0.8
    if flap_deg == 0:
        base -= 1.0
    elif flap_deg == 40:
        base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))


def compute_performance(perfdb: pd.DataFrame, heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                        elev_ft: float, slope_pct: float, shorten_ft: float, oat_c: float, qnh_inhg: float,
                        wind_knots: float, wind_dir_deg: float, wind_policy: str, gw_lbs: float,
                        flap_sel: str, thrust_mode: str, derate_n1_pct: float,
                        dcs_vr_factor: float, dcs_lo_factor: float) -> PerfResult:
    pa_ft = pressure_altitude_ft(elev_ft, qnh_inhg)
    headwind = headwind_component(wind_knots, wind_dir_deg, heading_deg)

    # Resolve flap & thrust policy (enforce: no derate with FULL)
    chosen_flap = flap_sel if flap_sel != "Auto-Select" else "MANEUVER"
    flap_deg = flap_to_deg(chosen_flap)
    effective_thrust = thrust_mode
    if chosen_flap == "FULL" and thrust_mode == "DERATE":
        effective_thrust = "MIL"

    table_thrust = "MILITARY" if effective_thrust in ("MIL", "DERATE") else "AFTERBURNER"
    vals = interp_perf_values(perfdb, flap_deg, table_thrust, gw_lbs, pa_ft, oat_c)

    v1 = float(vals["V1_kt"]) ; vr = float(vals["Vr_kt"]) ; v2 = float(vals["V2_kt"]) ; vs = float(vals.get("Vs_kt", np.nan))

    base_asd = float(vals["ASD_ft"]) ; base_agd = float(vals["AGD_ft"]) ; mult = 1.0
    n1_used_pct = 100.0
    if effective_thrust == "DERATE":
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
    avail_ft = max(0.0, tora_ft - shorten_ft)
    limiting = "ASD" if asd_eff >= agd_eff else "AGD"

    trim = compute_trim_units_anu(gw_lbs, flap_deg)

    # DCS AEO informational estimates vs regulatory required
    dcs_vr_est_ft = req_ft * float(dcs_vr_factor)
    dcs_lo_est_ft = req_ft * float(dcs_lo_factor)

    return PerfResult(v1, vr, v2, vs, chosen_flap, effective_thrust, req_ft, avail_ft, limiting, n1_used_pct, trim, dcs_vr_est_ft, dcs_lo_est_ft)

# ---------- selection helpers ----------

def meets_limits(pr: PerfResult, tora_ft: float, toda_ft: float, asda_ft: float, gw_lbs: float) -> bool:
    clearway_allow = min(tora_ft * 0.5, max(0.0, toda_ft - tora_ft))
    tod_limit = tora_ft + clearway_allow
    field_ok = (pr.req_distance_ft <= asda_ft) and (pr.req_distance_ft <= tod_limit) and (pr.req_distance_ft <= toda_ft)
    if pr.thrust_label == "DERATE":
        flap_deg = flap_to_deg(pr.flap_setting)
        field_ok = field_ok and compute_oei_second_segment_ok(gw_lbs, pr.n1_used_pct, flap_deg)
    return field_ok


def choose_flap_for_limits(perfdb: pd.DataFrame, heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                           elev_ft: float, slope_pct: float, shorten_ft: float, oat_c: float, qnh_inhg: float,
                           wind_knots: float, wind_dir_deg: float, wind_policy: str, gw_lbs: float,
                           flap_sel: str, thrust_mode: str, derate_n1_pct: float,
                           dcs_vr_factor: float, dcs_lo_factor: float) -> PerfResult:
    # If explicit flap, compute (enforce rule: no derate with FULL)
    if flap_sel != "Auto-Select":
        pr = compute_performance(perfdb, heading_deg, tora_ft, toda_ft, asda_ft, elev_ft, slope_pct, shorten_ft,
                                 oat_c, qnh_inhg, wind_knots, wind_dir_deg, wind_policy, gw_lbs,
                                 flap_sel, thrust_mode, derate_n1_pct, dcs_vr_factor, dcs_lo_factor)
        return pr

    # Auto → try MAN first
    pr_man = compute_performance(perfdb, heading_deg, tora_ft, toda_ft, asda_ft, elev_ft, slope_pct, shorten_ft,
                                 oat_c, qnh_inhg, wind_knots, wind_dir_deg, wind_policy, gw_lbs,
                                 "MANEUVER", thrust_mode, derate_n1_pct, dcs_vr_factor, dcs_lo_factor)
    if meets_limits(pr_man, tora_ft, toda_ft, asda_ft, gw_lbs):
        return pr_man

    # Escalate to FULL (and if DERATE was selected, switch to MIL by rule within compute_performance)
    pr_full = compute_performance(perfdb, heading_deg, tora_ft, toda_ft, asda_ft, elev_ft, slope_pct, shorten_ft,
                                  oat_c, qnh_inhg, wind_knots, wind_dir_deg, wind_policy, gw_lbs,
                                  "FULL", thrust_mode, derate_n1_pct, dcs_vr_factor, dcs_lo_factor)
    return pr_full

# ------------------------------
# DCS factor fits (from your testing) — Auto‑cal from N1%
# ------------------------------

def dcs_vr_factor_from_n1(n1_pct: float) -> float:
    # ~0.44 @ 96.5%, ~0.56 @ 90%
    return float(max(0.40, min(0.70, 0.38 + 0.0185 * (100.0 - n1_pct))))

def dcs_lo_factor_from_n1(n1_pct: float) -> float:
    # ~0.65 @ 96.5%, ~0.74–0.78 @ 90%
    return float(max(0.55, min(0.85, 0.607 + 0.0135 * (100.0 - n1_pct))))

# ------------------------------
# Input parsing helpers (UI)
# ------------------------------
import re
WIND_PAT = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*(?:@|/|\s)\s*(\d{1,3}(?:\.\d+)?)\s*$")

def parse_wind_entry(entry: str, unit: str) -> Optional[tuple]:
    """Parses 'DDD@SS' or 'DDD/SS' or 'DDD SS'. Returns (dir_deg, speed_knots)."""
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

    # Optional shortening input (as in V8)
    st.caption("Shorten Available Runway")
    sh_val = st.number_input("Value", min_value=0.0, value=0.0, step=100.0, key="sh_val")
    sh_unit = st.selectbox("Units", ["ft", "NM"], index=0, key="sh_unit")
    shorten_total_ft = float(sh_val) if sh_unit == "ft" else float(sh_val) * 6076.12

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)

    # QNH numeric + unit selector
    qnh_val = st.number_input("QNH value", value=29.92, step=0.01, format="%.2f")
    qnh_unit = st.selectbox("QNH Units", ["inHg", "hPa"], index=0)
    qnh_inhg = float(qnh_val) if qnh_unit == "inHg" else hpa_to_inhg(float(qnh_val))

    # Single wind entry + unit type
    wind_unit = st.selectbox("Wind Units", ["kts", "m/s"], index=0)
    wind_entry = st.text_input("Wind (DIR@SPD)", placeholder="e.g., 180@12")
    wind_parsed = parse_wind_entry(wind_entry, wind_unit)
    if wind_parsed is None and (wind_entry or "") != "":
        st.warning("Enter wind as DDD@SS, DDD/SS, or DDD SS. Example: 180@12")
        wind_dir_deg, wind_knots = float(row_rwy["heading_deg"]), 0.0
    else:
        wind_dir_deg, wind_knots = (wind_parsed if wind_parsed is not None else (float(row_rwy["heading_deg"]), 0.0))

    wind_policy = st.selectbox("Wind Policy", ["None", "50/150"], index=0, help="'50/150' = 50% headwind credit, 150% tailwind penalty")

    st.header("Weight & Config")
    model = "F-14B"
    gw_lbs = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)

    flap_sel = st.selectbox("Takeoff Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    st.header("Thrust")
    thrust_mode = st.radio("Mode", ["MIL", "DERATE", "AB"], index=0)
    derate_n1_pct = 100.0
    if thrust_mode == "DERATE":
        flap_hint = {0: "≥90%", 20: "≥90%", 40: "≥95%"}[flap_to_deg(flap_sel if flap_sel != "Auto-Select" else "MANEUVER")]
        st.caption(f"Derate floor by flap (approx): {flap_hint}")
        derate_n1_pct = st.slider("Target N1 % (MIL)", min_value=90.0, max_value=100.0, value=98.0, step=0.5)
        if st.button("Find minimum N1% (approx) to meet 121.189"):
            pa_ft = pressure_altitude_ft(float(row_rwy["threshold_elev_ft"]), float(qnh_inhg))
            chosen_flap_for_solver = (flap_sel if flap_sel != "Auto-Select" else "MANEUVER")
            flap_deg_solver = flap_to_deg(chosen_flap_for_solver)
            table_thrust = "MILITARY"  # bases from MIL
            vals = interp_perf_values(perfdb, flap_deg_solver, table_thrust, float(gw_lbs), float(pa_ft), float(oat_c))
            m = solve_min_derate_for_121_from_bases(
                base_asd=float(vals["ASD_ft"]), base_agd=float(vals["AGD_ft"]),
                tora_ft=float(row_rwy["tora_ft"]), toda_ft=float(row_rwy["toda_ft"]), asda_ft=float(row_rwy["asda_ft"]),
                slope_pct=float(row_rwy.get("slope_percent", 0.0) or 0.0),
                headwind_kn=headwind_component(float(wind_knots), float(wind_dir_deg), float(row_rwy["heading_deg"])),
                wind_policy=str(wind_policy), flap_deg=int(flap_deg_solver), gw_lbs=float(gw_lbs))
            if m > 100.0:
                st.error("Cannot meet 121.189 with derate in this model. Increase thrust or change config.")
            else:
                st.success(f"Minimum N1%: {m:.1f}")

    # DCS Estimates (informational)
    st.subheader("DCS Estimates (AEO)")
    auto_dcs = st.checkbox("Auto‑calibrate from N1%", value=True, help="Use empirical fit from derate to estimate AEO Vr/Liftoff distances.")
    if auto_dcs:
        n1_for_est = derate_n1_pct if thrust_mode == "DERATE" else 100.0
        dcs_vr_factor = dcs_vr_factor_from_n1(n1_for_est)
        dcs_lo_factor = dcs_lo_factor_from_n1(n1_for_est)
        st.caption(f"Auto Vr factor ≈ {dcs_vr_factor:.2f}, Liftoff factor ≈ {dcs_lo_factor:.2f}")
    else:
        dcs_vr_factor = st.slider("Vr factor vs Required", 0.40, 0.70, 0.52, 0.01)
        dcs_lo_factor = st.slider("Liftoff factor vs Required", 0.55, 0.85, 0.71, 0.01)

# compute
if st.button("Compute Takeoff Performance", type="primary"):
    try:
        perf = choose_flap_for_limits(perfdb,
                                      float(row_rwy["heading_deg"]), float(row_rwy["tora_ft"]), float(row_rwy["toda_ft"]), float(row_rwy["asda_ft"]),
                                      float(row_rwy["threshold_elev_ft"]), float(row_rwy.get("slope_percent", 0.0) or 0.0), float(shorten_total_ft),
                                      float(oat_c), float(qnh_inhg), float(wind_knots), float(wind_dir_deg), str(wind_policy),
                                      float(gw_lbs), flap_sel, thrust_mode, float(derate_n1_pct),
                                      float(dcs_vr_factor), float(dcs_lo_factor))

        ok = (perf.req_distance_ft <= perf.avail_distance_ft) and (perf.req_distance_ft <= float(row_rwy["toda_ft"])) and (perf.req_distance_ft <= float(row_rwy["asda_ft"]))

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.subheader("V‑Speeds")
            st.metric("V1 (kt)", f"{perf.v1:.0f}")
            st.metric("Vr (kt)", f"{perf.vr:.0f}")
            st.metric("V2 (kt)", f"{perf.v2:.0f}")
        with c2:
            st.subheader("Settings")
            st.metric("Flaps", perf.flap_setting)
            thrust_label = perf.thrust_label + (f" ({perf.n1_used_pct:.1f}% N1)" if perf.thrust_label=="DERATE" else "")
            st.metric("Thrust", thrust_label)
            st.metric("Trim (ANU)", f"{perf.trim_units_anu:.1f}")
        with c3:
            st.subheader("Runway (Regulatory)")
            st.metric("Required (ft)", f"{perf.req_distance_ft:.0f}")
            st.metric("Available (ft)", f"{perf.avail_distance_ft:.0f}")
            st.metric("Limiting", perf.limiting)
        with c4:
            st.subheader("Compliance")
            clearway_allow = min(float(row_rwy["tora_ft"]) * 0.5, max(0.0, float(row_rwy["toda_ft"]) - float(row_rwy["tora_ft"])) )
            st.metric("TOD Limit (ft)", f"{(float(row_rwy['tora_ft']) + clearway_allow):.0f}")
            st.caption("Clearway credit capped at 50% of runway length.")
        with c5:
            st.subheader("DCS Estimates (AEO)")
            st.metric("Vr distance est (ft)", f"{perf.dcs_vr_est_ft:.0f}")
            st.metric("Liftoff dist est (ft)", f"{perf.dcs_lo_est_ft:.0f}")

        st.markdown("✅ " + ("DISPATCH OK (approx)" if ok else "NOT PERMITTED by 121.189 field limits"))

        with st.expander("Debug inputs"):
            st.json({
                "GW_lb": float(gw_lbs),
                "Flap": perf.flap_setting,
                "Thrust": perf.thrust_label,
                "Derate_N1%": perf.n1_used_pct if perf.thrust_label == "DERATE" else None,
                "PA_ft": pressure_altitude_ft(float(row_rwy["threshold_elev_ft"]), float(qnh_inhg)),
                "WindPolicy": wind_policy,
            })
    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.stop()
else:
    st.info("Set inputs and click Compute Takeoff Performance. Auto‑flaps tries MAN first; FULL only if needed. No reduced thrust with FULL.")
