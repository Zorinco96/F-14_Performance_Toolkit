# app.py — DCS F‑14B Takeoff Performance (FAA‑based model)
#
# External file expected in repo root (exact case):
#   • dcs_airports.csv  (runway DB with: map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,threshold_elev_ft,...)
#
# Model overview (for DCS training only — NOT for real‑world use):
# • Distances start from MIL baseline (AGD/ASD) via tri‑linear interpolation over GW / Pressure Alt / OAT.
# • FAA‑style constraints per 14 CFR 121.189: ASD ≤ RWY+Stopway (ASDA); Takeoff distance ≤ RWY+≤50% Clearway (TORA + min(CWY, 0.5*TORA)); Takeoff run ≤ RWY. 
# • Derate solver finds the MIN N1% (MIL scale) that satisfies the limits, using thrust scaling on baseline distances.
# • Physics‑informed scalings:
#     – Thrust effect: distance ∝ 1 / (N1_frac^ALPHA_THRUST), ALPHA_THRUST ≈ 1.35 (tunable).
#     – Density effect: handled inside the baseline interpolation; additional wind/slope corrections applied per policy.
#     – Flaps: UP / MAN (20°) / FULL (40°). UP modeled by CLmax ratio vs MAN; FULL uses own rows if available.
# • V‑speeds come from the same tri‑linear interpolation grid (keeps consistency with available F‑14B data).
# • No derated thrust with FULL flaps (operational guardrail). AB is shown as last resort with warning.
# • Tailwind >10 kt or crosswind >30 kt → NOT AUTHORIZED per your ops policy (UI enforces/flags).
#
# Citations inside comments: 14 CFR 121.189 clearway cap at 50% runway; 25.111/25.113 takeoff path/distance; FAA PHAK density‑alt formula (DA ≈ PA + 120*(OAT−ISA)).

import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F‑14B Takeoff (FAA Model)", page_icon="✈️", layout="wide")

# ------------------------------
# Embedded performance grid (subset from your F‑14B CSV)
# ------------------------------
PERF_F14B = """model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note
F-14B,20,Military,60000,0,0,120,132,141,154,4300,5200,est
F-14B,20,Military,60000,0,30,123,135,144,157,4700,5600,est
F-14B,20,Military,60000,5000,0,123,136,146,158,5200,6100,est
F-14B,20,Military,60000,5000,30,126,139,149,162,5600,6600,est
F-14B,20,Military,65000,0,0,123,136,146,159,4800,5700,est
F-14B,20,Military,65000,0,30,126,139,150,162,5200,6200,est
F-14B,20,Military,65000,5000,0,126,139,150,162,5600,6600,est
F-14B,20,Military,65000,5000,30,129,143,154,167,6000,7100,est
F-14B,20,Military,70000,0,0,126,140,151,164,5300,6300,est
F-14B,20,Military,70000,0,30,129,143,155,168,5800,6900,est
F-14B,20,Military,70000,5000,0,130,144,156,169,6200,7400,est
F-14B,20,Military,70000,5000,30,133,147,160,173,6700,7900,est
F-14B,40,Military,60000,0,0,116,128,137,149,4200,5100,est
F-14B,40,Military,60000,0,30,118,130,139,152,4500,5500,est
F-14B,40,Military,70000,5000,30,131,144,156,169,6800,8300,est
F-14B,40,Afterburner,70000,5000,30,126,139,151,163,6000,7500,est
"""

# Aircraft constants (Heatblur manual / public sources)
WING_AREA_FT2 = 565.0   # Heatblur manual
ENGINE_MIL_THRUST_LBF = 16333.0  # F110‑GE‑400 (per‑engine, uninstalled)
ENGINES = 2

# Flap CLmax (calibrated so Vs(20°,60k,SL)≈120 kt)
CLMAX = {
    0: 1.80,   # UP (approx)
    20: 2.18,  # MAN (calibrated)
    40: 2.45,  # FULL (approx)
}

# Derate scaling exponent (distance ∝ 1/N1^ALPHA_THRUST)
ALPHA_THRUST = 1.35

# UP additional drag/rotation penalty on distances (beyond CL effect)
UP_EXTRA_DIST_FACTOR = 1.06

# No derate floors by flap
DERATE_FLOOR = {0: 0.90, 20: 0.90, 40: 1.00}  # FULL → force ≥100% MIL equivalent

# Wind policy helper
WIND_FACTORS = {
    "None": (1.0, 1.0),      # (headwind credit factor, tailwind penalty factor)
    "50/150": (0.5, 1.5),    # common operator standard
}

# ------------------------------ helpers ------------------------------

def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

@st.cache_data
def load_runways() -> pd.DataFrame:
    for path in ["dcs_airports.csv", "data/dcs_airports.csv"]:
        try:
            df = pd.read_csv(path)
            df["runway_label"] = df["airport_name"] + " " + df["runway_end"].astype(str) + " (" + df["runway_pair"].astype(str) + ")"
            return df
        except Exception:
            continue
    st.error("dcs_airports.csv not found in repo.")
    st.stop()

@st.cache_data
def load_perf() -> pd.DataFrame:
    df = pd.read_csv(StringIO(PERF_F14B))
    df["thrust"] = df["thrust"].str.upper()
    return df

# Atmosphere — FAA PHAK style DA approximation, plus standard‑atmos conversion

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - qnh_inhg) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (h_ft / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    isa = isa_temp_c_at_ft(pa_ft)
    return float(pa_ft + 120.0 * (oat_c - isa))

def sigma_from_da(da_ft: float) -> float:
    # Convert DA to sigma via standard atmosphere (troposphere model)
    h_m = da_ft * 0.3048
    T0 = 288.15; L = 0.0065; g = 9.80665; R = 287.05
    p0 = 101325.0; rho0 = 1.225
    T = T0 - L*h_m
    p = p0 * (T/T0)**(g/(R*L))
    rho = p/(R*T)
    return float(rho/rho0)

# Wind components

def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float,float]:
    # returns (headwind +/-, crosswind +/- right)
    delta = math.radians((dir_deg - rwy_heading_deg) % 360)
    hw = speed_kn * math.cos(delta)
    cw = speed_kn * math.sin(delta)
    return hw, cw

# Interpolation across grid

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
        x0, x1 = xs[0], xs[1]; y0, y1 = ys[0], ys[1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (gw_x - x0))
    if gw_x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]; y0, y1 = ys[-2], ys[-1]
        slope = (y1 - y0) / (x1 - x0)
        return float(y0 + slope * (gw_x - x0))
    return float(np.interp(gw_x, xs, ys))


def interp_perf(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float):
    # If UP, fall back to MAN table for V‑speeds/distances and apply UP penalties later
    use_flap = 20 if flap_deg == 0 else flap_deg
    sub = perf[(perf["flap_deg"] == use_flap) & (perf["thrust"] == thrust)]
    if sub.empty:
        sub = perf[perf["flap_deg"] == use_flap]
    pa0, pa1, wp = _bounds(sub["press_alt_ft"].unique(), pa)
    t0,  t1,  wt = _bounds(sub["oat_c"].unique(),        oat)

    fields = ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]
    out = {}
    for f in fields:
        v00 = _interp_weight_at(sub, pa0, t0, f, gw)
        v01 = _interp_weight_at(sub, pa0, t1, f, gw)
        v10 = _interp_weight_at(sub, pa1, t0, f, gw)
        v11 = _interp_weight_at(sub, pa1, t1, f, gw)
        v0  = v00*(1-wt) + v01*wt
        v1  = v10*(1-wt) + v11*wt
        out[f] = v0*(1-wp) + v1*wp
    return out

# Scaling & solver

def enforce_derate_floor(n1pct: float, flap_deg: int) -> float:
    floor = DERATE_FLOOR.get(flap_deg, 0.90) * 100.0
    return max(n1pct, floor)


def scale_dist_for_derate(base_ft: float, n1pct: float, flap_deg: int) -> float:
    n1pct = enforce_derate_floor(n1pct, flap_deg)
    eff = max(0.90, min(1.0, n1pct/100.0))
    return float(base_ft) / (eff ** ALPHA_THRUST)


def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    # slope: +20% per +1% uphill (conservative)
    if slope_pct > 0:
        d *= (1.0 + 0.20 * slope_pct)
    # wind: ~0.5% per knot of factored head/tail component
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)  # headwind_kn negative → increases distance
    return max(d, 0.0)


def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0) / 10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_ft: float; req_ft: float; avail_ft: float; limiting: str
    notes: list


def compute_takeoff(perfdb: pd.DataFrame, 
                    rwy_heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                    field_elev_ft: float, slope_pct: float,
                    user_override_len_ft: Optional[float], user_override_elev_ft: Optional[float],
                    shorten_ft: float,
                    oat_c: float, qnh_inhg: float,
                    wind_speed: float, wind_dir_deg: float, wind_units: str, wind_policy: str,
                    gw_lbs: float,
                    flap_mode: str,
                    thrust_mode: str,
                    target_n1_pct: float) -> Result:

    # Use overrides if provided
    elev_ft = float(user_override_elev_ft if user_override_elev_ft is not None else field_elev_ft)
    tora0 = float(user_override_len_ft if user_override_len_ft is not None else tora_ft)
    toda0 = float(user_override_len_ft if user_override_len_ft is not None else toda_ft)
    asda0 = float(user_override_len_ft if user_override_len_ft is not None else asda_ft)

    # Apply shortening (manual or intersection)
    tora = max(0.0, tora0 - shorten_ft)
    toda = max(0.0, toda0 - shorten_ft)
    asda = max(0.0, asda0 - shorten_ft)

    # Atmosphere & wind
    pa = pressure_altitude_ft(elev_ft, qnh_inhg)
    da = density_altitude_ft(pa, oat_c)
    sigma = sigma_from_da(da)
    spd_kn = wind_speed if wind_units == "kts" else wind_speed * 1.943844
    hw, cw = wind_components(spd_kn, wind_dir_deg, rwy_heading_deg)

    notes = []
    if hw < -10.0:
        notes.append("Tailwind component exceeds 10 kt — NOT AUTHORIZED.")
    if abs(cw) > 30.0:
        notes.append("Crosswind component exceeds 30 kt — NOT AUTHORIZED.")

    # Flap resolution (auto favors MAN)
    flap_text = flap_mode
    if flap_mode == "Auto-Select":
        flap_text = "MANEUVER"
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)

    # Thrust resolution
    if thrust_mode == "AB":
        notes.append("Afterburner selected — NOT AUTHORIZED for F‑14B except as last resort.")
    if flap_deg == 40 and thrust_mode == "DERATE":
        notes.append("Derate with FULL flaps not allowed — switching to MIL for calc.")
        thrust_mode = "MIL"

    # Interpolate baseline at MIL for the chosen flap (or MAN for UP)
    base_flap_for_interp = 20 if flap_deg == 0 else flap_deg
    base = interp_perf(perfdb, base_flap_for_interp, "MILITARY", float(gw_lbs), float(pa), float(oat_c))

    vs = float(base["Vs_kt"]) ; v1_tab = float(base["V1_kt"]) ; vr = float(base["Vr_kt"]) ; v2 = float(base["V2_kt"]) 
    asd_base = float(base["ASD_ft"]) ; agd_base = float(base["AGD_ft"]) 

    # UP: adjust distances by CLmax ratio + small extra drag factor
    if flap_deg == 0:
        cl_ratio = CLMAX[20] / CLMAX[0]
        asd_base *= cl_ratio * UP_EXTRA_DIST_FACTOR
        agd_base *= cl_ratio * UP_EXTRA_DIST_FACTOR

    # FULL: use baseline as is (already flap‑specific where available)

    # Thrust/N1 handling
    n1 = 100.0
    thrust_text = thrust_mode

    def field_ok(asd_eff, agd_eff):
        # Clearway credit capped at 50% runway length (121.189)
        clearway_allow = min(tora * 0.5, max(0.0, toda - tora))
        tod_limit = tora + clearway_allow
        req = max(asd_eff, agd_eff)
        return (asd_eff <= asda) and (agd_eff <= tod_limit) and (agd_eff <= toda), req, ("ASD" if asd_eff >= agd_eff else "AGD")

    def distances_for(n1pct: float) -> Tuple[float,float]:
        asd = scale_dist_for_derate(asd_base, n1pct, flap_deg)
        agd = scale_dist_for_derate(agd_base, n1pct, flap_deg)
        # Apply wind/slope
        asd = apply_wind_slope(asd, slope_pct, hw, wind_policy)
        agd = apply_wind_slope(agd, slope_pct, hw, wind_policy)
        return asd, agd

    # Auto‑select thrust: find minimum N1 that passes 121.189
    if thrust_mode == "Auto-Select" or thrust_mode == "DERATE":
        start_floor = DERATE_FLOOR.get(flap_deg, 0.90) * 100.0 if thrust_mode != "Auto-Select" else DERATE_FLOOR.get(flap_deg, 0.90)*100.0
        req_n1 = max(start_floor, (target_n1_pct if thrust_mode == "DERATE" else start_floor))
        lo, hi = req_n1, 100.0
        ok_any = False
        for _ in range(18):
            mid = (lo + hi) / 2.0
            asd_m, agd_m = distances_for(mid)
            ok, req, _ = field_ok(asd_m, agd_m)
            if ok:
                hi = mid; ok_any = True
            else:
                lo = mid
        n1 = round(hi, 1)
        if not ok_any:
            # Could not satisfy with derate — try MIL as fallback and mark note
            asd_m, agd_m = distances_for(100.0)
            ok, req, limiting = field_ok(asd_m, agd_m)
            n1 = 100.0
            if not ok:
                notes.append("Even MIL power fails field limits at this config — consider FULL flaps or different runway.")
            thrust_text = "MIL" if thrust_mode != "AB" else "AB"
        else:
            thrust_text = "DERATE"
    elif thrust_mode == "MIL":
        n1 = 100.0
    else:  # AB (shown but discouraged)
        n1 = 100.0

    # Final distances at selected n1
    asd_fin, agd_fin = distances_for(n1)
    ok, req, limiting = field_ok(asd_fin, agd_fin)

    # If Auto‑flap and NOT OK at MAN, escalate to FULL (single pass)
    if flap_mode == "Auto-Select" and not ok:
        # Re‑compute at FULL (force MIL if thrust was DERATE)
        return compute_takeoff(perfdb, rwy_heading_deg, tora, toda, asda, elev_ft, slope_pct,
                               user_override_len_ft, user_override_elev_ft,
                               0.0,  # already applied shorten_ft above
                               oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                               gw_lbs, "FULL", ("MIL" if thrust_text=="DERATE" else thrust_text), 100.0)

    avail = tora
    # Authorization notes already populated for tailwind/crosswind

    return Result(
        v1=v1_tab, vr=vr, v2=v2, vs=vs,
        flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
        asd_ft=asd_fin, agd_ft=agd_fin, req_ft=max(asd_fin, agd_fin), avail_ft=avail, limiting=limiting,
        notes=notes,
    )

# ------------------------------ UI ------------------------------

rwy_db = load_runways()
perfdb = load_perf()

st.title("DCS F‑14B Takeoff — FAA‑Based Model (v1)")

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Theatre", sorted(rwy_db["map"].unique()))
    df_t = rwy_db[rwy_db["map"] == theatre]
    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]
    rwy_label = st.selectbox("Runway End", list(df_a["runway_label"]))
    rwy = df_a[df_a["runway_label"] == rwy_label].iloc[0]

    tora_ft = float(rwy["tora_ft"]) ; toda_ft = float(rwy["toda_ft"]) ; asda_ft = float(rwy["asda_ft"]) ; elev_ft = float(rwy["threshold_elev_ft"]) ; hdg = float(rwy["heading_deg"]) ; slope = float(rwy.get("slope_percent", 0.0) or 0.0)

    st.metric("TORA (ft)", f"{tora_ft:,.0f}")
    st.metric("TODA (ft)", f"{toda_ft:,.0f}")
    st.metric("ASDA (ft)", f"{asda_ft:,.0f}")
    st.metric("Elev (ft)", f"{elev_ft:,.0f}")

    st.subheader("Overrides / Intersection")
    use_override = st.checkbox("Manually enter runway length & elevation")
    user_len_ft = st.number_input("Manual Length (ft)", min_value=0.0, value=tora_ft, step=50.0) if use_override else None
    user_elev_ft = st.number_input("Manual Elevation (ft)", min_value=-1000.0, value=elev_ft, step=10.0) if use_override else None

    use_intersection = st.checkbox("Intersection departure")
    inter_dist = st.number_input("Distance from threshold to intersection (ft)", min_value=0.0, value=0.0, step=50.0) if use_intersection else 0.0

    manual_shorten = st.number_input("Additional manual reduction (ft)", min_value=0.0, value=0.0, step=50.0)
    shorten_total = float(inter_dist) + float(manual_shorten)

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    qnh_val = st.number_input("QNH value", value=29.92, step=0.01, format="%.2f")
    qnh_unit = st.selectbox("QNH Units", ["inHg", "hPa"], index=0)
    qnh_inhg = float(qnh_val) if qnh_unit == "inHg" else hpa_to_inhg(float(qnh_val))

    wind_units = st.selectbox("Wind Units", ["kts", "m/s"], index=0)
    wind_dir = st.number_input("Wind direction (deg)", min_value=0.0, max_value=359.9, value=float(hdg))
    wind_spd = st.number_input("Wind speed", min_value=0.0, value=0.0, step=1.0)
    wind_policy = st.selectbox("Wind Policy", ["None", "50/150"], index=0)

    st.header("Weight & Config")
    entry_mode = st.radio("Weight entry", ["Direct GW", "Fuel + Stores"], index=0)
    if entry_mode == "Direct GW":
        gw = st.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=500.0)
    else:
        empty_w = st.number_input("Empty weight (lb)", min_value=38000.0, max_value=46000.0, value=41780.0, step=50.0)
        fuel_lb = st.number_input("Internal fuel (lb)", min_value=0.0, max_value=20000.0, value=8000.0, step=100.0)
        ext_tanks = st.selectbox("External tanks (267 gal)", [0,1,2], index=0)
        aim9 = st.slider("AIM‑9 count", 0, 2, 0)
        aim7 = st.slider("AIM‑7 count", 0, 4, 0)
        aim54 = st.slider("AIM‑54 count", 0, 6, 0)
        lantirn = st.checkbox("LANTIRN pod")
        w = empty_w + fuel_lb + ext_tanks*1900 + aim9*190 + aim7*510 + aim54*1000 + (440 if lantirn else 0)
        gw = st.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(w), step=10.0)

    flap_mode = st.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    thrust_mode = st.selectbox("Thrust", ["Auto-Select", "Manual Derate", "MIL", "AB"], index=0)
    tgt_n1 = 98.0
    if thrust_mode == "Manual Derate":
        floor = DERATE_FLOOR.get(0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20), 0.90)*100.0
        st.caption(f"Derate floor (by flap): {floor:.0f}% N1 (MIL)")
        tgt_n1 = st.slider("Target N1 % (MIL)", min_value=floor, max_value=100.0, value=max(floor, 95.0), step=0.5)

colL, colR = st.columns([1,2])
with colL:
    if st.button("Compute Takeoff Performance", type="primary"):
        res = compute_takeoff(perfdb,
                              float(hdg), float(tora_ft), float(toda_ft), float(asda_ft),
                              float(elev_ft), float(slope),
                              user_len_ft, user_elev_ft,
                              float(shorten_total),
                              float(oat_c), float(qnh_inhg),
                              float(wind_spd), float(wind_dir), str(wind_units), str(wind_policy),
                              float(gw),
                              str(flap_mode),
                              ("Auto-Select" if thrust_mode=="Auto-Select" else ("DERATE" if thrust_mode=="Manual Derate" else thrust_mode)),
                              float(tgt_n1))
        st.session_state["last_result"] = res

with colR:
    res = st.session_state.get("last_result")
    if res is None:
        st.info("Set inputs and click Compute.")
    else:
        vcol, scol, rcol, ccol = st.columns(4)
        with vcol:
            st.subheader("V‑Speeds")
            st.metric("V1 (kt)", f"{res.v1:.0f}")
            st.metric("Vr (kt)", f"{res.vr:.0f}")
            st.metric("V2 (kt)", f"{res.v2:.0f}")
        with scol:
            st.subheader("Settings")
            st.metric("Flaps", res.flap_text)
            st.metric("Thrust", f"{res.thrust_text} ({res.n1_pct:.1f}% N1)")if False else  float(0) or 0 + 0 + 0):s}")
        with rcol:
            st.subheader("Runway Distances")
            st.metric("Accelerate‑Stop (ft)", f"{res.asd_ft:.0f}")
            st.metric("Accelerate‑Go (ft)", f"{res.agd_ft:.0f}")
            st.metric("Required (ft)", f"{res.req_ft:.0f}")
        with ccol:
            st.subheader("Availability")
            st.metric("Available (ft)", f"{res.avail_ft:.0f}")
            st.metric("Limiting", res.limiting)
            hw, cw = wind_components((st.sidebar.session_state.get('wind_spd',0.0) if False else 0.0) or 0.0, (st.sidebar.session_state.get('wind_dir',0.0) if False else 0.0) or 0.0, float(hdg))
            st.caption("Tailwind >10 kt or crosswind >30 kt: NOT AUTHORIZED.")
        if res.notes:
            st.warning("\n".join(res.notes))
        
        # Compute Trim with actual GW and flap
        st.subheader("Trim")
        flap_deg = 0 if res.flap_text.upper().startswith("UP") else (40 if res.flap_text.upper().startswith("FULL") else 20)
        st.metric("Takeoff Trim (ANU)", f"{trim_anu(float(gw), flap_deg):.1f}")

st.markdown("---")
st.caption("This tool estimates F‑14B takeoff performance for DCS per FAA‑style rules (14 CFR 121.189). Not for real‑world use.")
