# app.py — DCS F-14B Takeoff Performance (FAA-based model v1.1)
# External file expected in repo root (exact case):
#   • dcs_airports.csv  (map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,threshold_elev_ft,...)
#
# DCS training aid ONLY — not for real-world planning.

import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F-14B Takeoff (FAA Model)", page_icon="✈️", layout="wide")

# ------------------------------
# Minimal F-14B performance grid (from your CSV; MIL/AB at MAN/FULL)
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

# Constants / parameters (plausible public values; tunable for DCS fidelity)
WING_AREA_FT2 = 565.0
ENGINE_MIL_THRUST_LBF = 16333.0  # per engine, uninstalled
ALPHA_THRUST = 1.35              # distance ∝ 1/(N1^alpha)
CLMAX = {0: 1.80, 20: 2.18, 40: 2.45}   # calibrated so Vs at MAN(20) matches grid
UP_EXTRA_DIST_FACTOR = 1.06              # extra rotation/drag penalty beyond CL difference
DERATE_FLOOR = {0: 0.90, 20: 0.90, 40: 1.00}  # FULL => force 100% MIL equivalent
WIND_FACTORS = {"None": (1.0, 1.0), "50/150": (0.5, 1.5)}

# ------------------------------ loaders ------------------------------
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

# ------------------------------ atmosphere & wind ------------------------------
def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - qnh_inhg) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (h_ft / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    # FAA approx
    return float(pa_ft + 120.0 * (oat_c - isa_temp_c_at_ft(pa_ft)))

def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float, float]:
    delta = math.radians((dir_deg - rwy_heading_deg) % 360)
    hw = speed_kn * math.cos(delta)
    cw = speed_kn * math.sin(delta)
    return hw, cw

# ------------------------------ interpolation ------------------------------
def _bounds(vals, x):
    vals = sorted(set(map(float, vals)))
    lo = max([v for v in vals if v <= x], default=vals[0])
    hi = min([v for v in vals if v >= x], default=vals[-1])
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return lo, hi, w

def _interp_weight_at(sub: pd.DataFrame, pa: float, oat: float, field: str, gw_x: float) -> float:
    s = sub[(sub["press_alt_ft"] == pa) & (sub["oat_c"] == oat)].sort_values("gw_lbs")
    if s.empty:
        s = sub.sort_values(["press_alt_ft", "oat_c", "gw_lbs"])
    xs = s["gw_lbs"].values.astype(float)
    ys = s[field].values.astype(float)
    if len(xs) < 2: return float(ys[0])
    if gw_x <= xs[0]:
        x0, x1 = xs[0], xs[1]; y0, y1 = ys[0], ys[1]
        return float(y0 + (y1-y0)/(x1-x0)*(gw_x-x0))
    if gw_x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]; y0, y1 = ys[-2], ys[-1]
        return float(y0 + (y1-y0)/(x1-x0)*(gw_x-x0))
    return float(np.interp(gw_x, xs, ys))

def interp_perf(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float):
    use_flap = 20 if flap_deg == 0 else flap_deg  # UP falls back to MAN table
    sub = perf[(perf["flap_deg"] == use_flap) & (perf["thrust"] == thrust)]
    if sub.empty: sub = perf[perf["flap_deg"] == use_flap]
    pa0, pa1, wp = _bounds(sub["press_alt_ft"].unique(), pa)
    t0,  t1,  wt = _bounds(sub["oat_c"].unique(),        oat)
    out = {}
    for f in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
        v00 = _interp_weight_at(sub, pa0, t0, f, gw)
        v01 = _interp_weight_at(sub, pa0, t1, f, gw)
        v10 = _interp_weight_at(sub, pa1, t0, f, gw)
        v11 = _interp_weight_at(sub, pa1, t1, f, gw)
        v0  = v00*(1-wt) + v01*wt
        v1  = v10*(1-wt) + v11*wt
        out[f] = v0*(1-wp) + v1*wp
    return out

# ------------------------------ scaling & solver ------------------------------
def enforce_derate_floor(n1pct: float, flap_deg: int) -> float:
    floor = DERATE_FLOOR.get(flap_deg, 0.90) * 100.0
    return max(n1pct, floor)

def scale_dist_for_derate(base_ft: float, n1pct: float, flap_deg: int) -> float:
    n1pct = enforce_derate_floor(n1pct, flap_deg)
    eff = max(0.90, min(1.0, n1pct/100.0))
    return float(base_ft) / (eff ** ALPHA_THRUST)

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    if slope_pct > 0: d *= (1.0 + 0.20 * slope_pct)  # +20% per +1% uphill (conservative)
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0: d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:                 d *= (1.0 - 0.005 * tail_fac * headwind_kn)  # negative headwind = tailwind
    return max(d, 0.0)

def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# ------------------------------ core compute ------------------------------
@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_ft: float; req_ft: float; avail_ft: float; limiting: str
    hw_kn: float; cw_kn: float; notes: list

def compute_takeoff(perfdb: pd.DataFrame,
                    rwy_heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                    field_elev_ft: float, slope_pct: float,
                    user_len_ft: Optional[float], user_elev_ft: Optional[float],
                    shorten_ft: float,
                    oat_c: float, qnh_inhg: float,
                    wind_speed: float, wind_dir_deg: float, wind_units: str, wind_policy: str,
                    gw_lbs: float,
                    flap_mode: str,
                    thrust_mode: str,
                    target_n1_pct: float) -> Result:

    elev_ft = float(user_elev_ft if user_elev_ft is not None else field_elev_ft)
    base_len = float(user_len_ft if user_len_ft is not None else tora_ft)
    tora = max(0.0, base_len - shorten_ft)
    toda = max(0.0, (user_len_ft if user_len_ft is not None else toda_ft) - shorten_ft)
    asda = max(0.0, (user_len_ft if user_len_ft is not None else asda_ft) - shorten_ft)

    pa = pressure_altitude_ft(elev_ft, qnh_inhg)
    spd_kn = wind_speed if wind_units == "kts" else wind_speed * 1.943844
    hw, cw = wind_components(spd_kn, wind_dir_deg, rwy_heading_deg)

    notes = []
    if hw < -10.0: notes.append("Tailwind component exceeds 10 kt — NOT AUTHORIZED.")
    if abs(cw) > 30.0: notes.append("Crosswind component exceeds 30 kt — NOT AUTHORIZED.")

    # Flaps
    flap_text = "MANEUVER" if flap_mode == "Auto-Select" else flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)

    # Thrust rule: no derate with FULL → force MIL in calc
    thrust_text = thrust_mode
    if flap_deg == 40 and thrust_mode in ("Auto-Select","DERATE","Manual Derate"):
        thrust_text = "MIL"

    # Baseline from MIL table (or AB if explicitly chosen)
    table_thrust = "AFTERBURNER" if thrust_mode == "AB" else "MILITARY"
    base = interp_perf(perfdb, (20 if flap_deg==0 else flap_deg), table_thrust if table_thrust=="MILITARY" else table_thrust,
                       float(gw_lbs), float(pa), float(oat_c))
    vs = float(base["Vs_kt"]); v1 = float(base["V1_kt"]); vr = float(base["Vr_kt"]); v2 = float(base["V2_kt"])
    asd_base = float(base["ASD_ft"]); agd_base = float(base["AGD_ft"])

    # UP adjustment vs MAN table
    if flap_deg == 0:
        cl_ratio = CLMAX[20] / CLMAX[0]
        asd_base *= cl_ratio * UP_EXTRA_DIST_FACTOR
        agd_base *= cl_ratio * UP_EXTRA_DIST_FACTOR

    def distances_for(n1pct: float) -> Tuple[float,float]:
        asd = scale_dist_for_derate(asd_base, n1pct, flap_deg)
        agd = scale_dist_for_derate(agd_base, n1pct, flap_deg)
        asd = apply_wind_slope(asd, slope_pct, hw, wind_policy)
        agd = apply_wind_slope(agd, slope_pct, hw, wind_policy)
        return asd, agd

    def field_ok(asd_eff: float, agd_eff: float) -> Tuple[bool,float,str]:
        clearway_allow = min(tora * 0.5, max(0.0, toda - tora))
        tod_limit = tora + clearway_allow
        req = max(asd_eff, agd_eff)
        return ((asd_eff <= asda) and (agd_eff <= tod_limit) and (agd_eff <= toda)), req, ("ASD" if asd_eff >= agd_eff else "AGD")

    # Choose thrust
    n1 = 100.0
    if thrust_mode in ("Auto-Select", "Manual Derate", "DERATE"):
        floor = DERATE_FLOOR.get(flap_deg, 0.90) * 100.0
        lo = max(floor, target_n1_pct if thrust_mode != "Auto-Select" else floor)
        hi = 100.0
        ok_any = False
        for _ in range(18):
            mid = (lo + hi) / 2.0
            asd_m, agd_m = distances_for(mid)
            ok, _, _ = field_ok(asd_m, agd_m)
            if ok:
                hi = mid; ok_any = True
            else:
                lo = mid
        n1 = round(hi, 1)
        if not ok_any:
            # even 100% fails — keep 100 and warn
            asd_m, agd_m = distances_for(100.0)
            ok, _, _ = field_ok(asd_m, agd_m)
            if not ok: notes.append("Even MIL power fails field limits — consider FULL flaps or a different runway.")
            thrust_text = "MIL" if thrust_mode != "AB" else "AB"
        else:
            thrust_text = "DERATE"
    elif thrust_mode == "MIL":
        n1 = 100.0
    else:  # AB
        n1 = 100.0
        notes.append("Afterburner selected — NOT AUTHORIZED for F-14B except as last resort.")

    # Final distances
    asd_fin, agd_fin = distances_for(n1)
    ok, req, limiting = field_ok(asd_fin, agd_fin)

    # Auto-flap escalation if MAN not sufficient
    if flap_mode == "Auto-Select" and not ok:
        return compute_takeoff(perfdb, rwy_heading_deg, tora, toda, asda, elev_ft, slope_pct,
                               None, None, 0.0,
                               oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                               gw_lbs, "FULL", ("MIL" if thrust_text=="DERATE" else thrust_text), 100.0)

    return Result(v1=v1, vr=vr, v2=v2, vs=vs,
                  flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
                  asd_ft=asd_fin, agd_ft=agd_fin, req_ft=req, avail_ft=tora, limiting=limiting,
                  hw_kn=hw, cw_kn=cw, notes=notes)

# ------------------------------ UI ------------------------------
rwy_db = load_runways()
perfdb = load_perf()

st.title("DCS F-14B Takeoff — FAA-Based Model (v1.1)")

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Theatre", sorted(rwy_db["map"].unique()))
    df_t = rwy_db[rwy_db["map"] == theatre]
    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]
    rwy_label = st.selectbox("Runway End", list(df_a["runway_label"]))
    rwy = df_a[df_a["runway_label"] == rwy_label].iloc[0]

    tora_ft = float(rwy["tora_ft"]); toda_ft = float(rwy["toda_ft"]); asda_ft = float(rwy["asda_ft"])
    elev_ft = float(rwy["threshold_elev_ft"]); hdg = float(rwy["heading_deg"]); slope = float(rwy.get("slope_percent", 0.0) or 0.0)

    cA, cB = st.columns(2)
    with cA:
        st.metric("TORA (ft)", f"{tora_ft:,.0f}")
        st.metric("TODA (ft)", f"{toda_ft:,.0f}")
    with cB:
        st.metric("ASDA (ft)", f"{asda_ft:,.0f}")
        st.metric("Elev (ft)", f"{elev_ft:,.0f}")

    st.subheader("Overrides / Intersection")
    use_override = st.checkbox("Manually enter runway length & elevation")
    user_len_ft = st.number_input("Manual Length (ft)", min_value=0.0, value=tora_ft, step=50.0) if use_override else None
    user_elev_ft = st.number_input("Manual Elevation (ft)", min_value=-1000.0, value=elev_ft, step=10.0) if use_override else None

    use_intersection = st.checkbox("Intersection departure")
    inter_dist = st.number_input("Distance to intersection (ft)", min_value=0.0, value=0.0, step=50.0) if use_intersection else 0.0

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
        aim9 = st.slider("AIM-9 count", 0, 2, 0)
        aim7 = st.slider("AIM-7 count", 0, 4, 0)
        aim54 = st.slider("AIM-54 count", 0, 6, 0)
        lantirn = st.checkbox("LANTIRN pod")
        w = empty_w + fuel_lb + ext_tanks*1900 + aim9*190 + aim7*510 + aim54*1000 + (440 if lantirn else 0)
        gw = st.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(w), step=10.0)

    flap_mode = st.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    thrust_pick = st.selectbox("Thrust", ["Auto-Select", "Manual Derate", "MIL", "AB"], index=0)
    tgt_n1 = 98.0
    if thrust_pick == "Manual Derate":
        # show floor hint by flap
        flap_for_floor = 0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20)
        floor = DERATE_FLOOR.get(flap_for_floor, 0.90)*100.0
        st.caption(f"Derate floor (by flap): {floor:.0f}% N1 (MIL)")
        tgt_n1 = st.slider("Target N1 % (MIL)", min_value=floor, max_value=100.0, value=max(floor, 95.0), step=0.5)

run = st.button("Compute Takeoff Performance", type="primary")

if run:
    res = compute_takeoff(perfdb,
                          float(hdg), float(tora_ft), float(toda_ft), float(asda_ft),
                          float(elev_ft), float(slope),
                          user_len_ft, user_elev_ft,
                          float(shorten_total),
                          float(oat_c), float(qnh_inhg),
                          float(wind_spd), float(wind_dir), str(wind_units), str(wind_policy),
                          float(gw),
                          str(flap_mode),
                          ("DERATE" if thrust_pick=="Manual Derate" else thrust_pick),
                          float(tgt_n1))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("V-Speeds")
        st.metric("V1 (kt)", f"{res.v1:.0f}")
        st.metric("Vr (kt)", f"{res.vr:.0f}")
        st.metric("V2 (kt)", f"{res.v2:.0f}")
        st.metric("Vs (kt)", f"{res.vs:.0f}")
    with c2:
        st.subheader("Settings")
        st.metric("Flaps", res.flap_text)
        st.metric("Thrust", f"{res.thrust_text} ({res.n1_pct:.1f}% N1)")
        flap_deg = 0 if res.flap_text.upper().startswith("UP") else (40 if res.flap_text.upper().startswith("FULL") else 20)
        st.metric("Trim (ANU)", f"{trim_anu(float(gw), flap_deg):.1f}")
    with c3:
        st.subheader("Runway Distances")
        st.metric("Accelerate-Stop (ft)", f"{res.asd_ft:.0f}")
        st.metric("Accelerate-Go (ft)", f"{res.agd_ft:.0f}")
        st.metric("Required (ft)", f"{res.req_ft:.0f}")
    with c4:
        st.subheader("Availability")
        st.metric("Available (ft)", f"{res.avail_ft:.0f}")
        st.metric("Limiting", res.limiting)
        st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
        st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
        st.caption("Tailwind >10 kt or crosswind >30 kt → NOT AUTHORIZED.")

    if res.notes:
        st.warning("\n".join(res.notes))

else:
    st.info("Set inputs and click Compute.")
