# f14_takeoff_app.py — DCS F-14B Takeoff Performance (FAA-style model, MIL-anchored derates)
#
# External data files expected in repo root (exact case):
#   • dcs_airports.csv  (runway DB with: map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,threshold_elev_ft,slope_percent(optional))
#   • f14_perf.csv      (performance grid with columns: model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note)
#
# Highlights
# • 14 CFR 121.189 checks: ASD ≤ ASDA, OEI continue ≤ TORA+(≤50% TORA clearway), and ≤ TODA.
# • MIL-anchored derate solver (bisection) finds MIN N1% that passes regulatory limits + OEI climb guardrail.
# • Treat CSV AGD_ft as AEO ground roll; convert to AEO liftoff-to-35ft via LIFTOFF_FACTOR, then apply OEI penalty for regulatory checks.
# • Density-altitude scaling outside the grid (PA>5k ft or OAT>30 °C).
# • Flaps: Auto-Select (defaults to MAN), UP, MANEUVER (20°), FULL (40°). FULL may NOT be derated.
# • Thrust: Auto-Select, Manual Derate (whole-% N1), MIL, AB (AB not authorized—shows warning).
# • Weather: OAT (°C), QNH (inHg or hPa), Wind (DDD@SS in kts or m/s) with policy (None or 50/150). Tailwind>10 kt / Xwind>30 kt → NOT AUTHORIZED note.
# • Weight: Direct GW or Fuel + Stores builder; editable computed GW.
# • Outputs: V1/Vr/V2/Vs, flaps, thrust w/ N1, trim (ANU), Stop distance, Continue distance (engine-out, regulatory), Required runway, Available runway, limiting factor.
# • All-engines takeoff estimates (Vr ground roll & liftoff distance) shown separately for DCS comparison.
#
# DISCLAIMER: Training aid for DCS only — NOT for real-world flight planning.

from __future__ import annotations
import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F-14B Takeoff (FAA Model)", page_icon="✈️", layout="wide")

# ------------------------------ tuning constants ------------------------------
# Per-engine approximate thrust (not used directly in distances; used in OEI guardrail heuristic)
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}

# FULL flaps cannot be derated; UP/MAN may derate to 90% N1 floor
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}

# Distance sensitivity to N1 derate (stronger than before)
ALPHA_N1_DIST = 2.0

# Treat CSV AGD_ft as AEO ground roll; convert to AEO liftoff-to-35ft by this factor
LIFTOFF_FACTOR = 2.15

# Regulatory engine-out continue penalty vs AEO liftoff distance (toggle in Advanced)
OEI_AGD_FACTOR = 1.20  # set to 1.15 for DCS-calibrated behavior

# Optional global AEO calibration (keep 1.00 when using LIFTOFF_FACTOR)
AEO_CAL_FACTOR = 1.00

# AEO Vr ground-roll fraction relative to AEO liftoff distance (for display only)
AEO_VR_FRAC = 0.78

# Wind policy factors (headwind credit, tailwind penalty)
WIND_FACTORS = {"None": (1.0, 1.0), "50/150": (0.5, 1.5)}

# ------------------------------ helpers: atmosphere / wind ------------------------------
def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - qnh_inhg) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (h_ft / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    # FAA approximation: DA ≈ PA + 120 × (OAT − ISA)
    return float(pa_ft + 120.0 * (oat_c - isa_temp_c_at_ft(pa_ft)))

def sigma_from_da(da_ft: float) -> float:
    # ISA troposphere density ratio
    h_m = da_ft * 0.3048
    T0 = 288.15; L = 0.0065; g = 9.80665; R = 287.05
    p0 = 101325.0; rho0 = 1.225
    T = T0 - L*h_m
    p = p0 * (T/T0)**(g/(R*L))
    rho = p/(R*T)
    return float(rho/rho0)

def da_out_of_grid_scale(pa_ft: float, oat_c: float) -> float:
    """Outside the grid (PA>5000 or OAT>30), upscale distances by density ratio vs a clamped ref (≤5000 ft, ≤30 °C)."""
    da_act = density_altitude_ft(pa_ft, oat_c)
    da_ref = density_altitude_ft(min(pa_ft, 5000.0), min(oat_c, 30.0))
    sig_act = sigma_from_da(da_act)
    sig_ref = sigma_from_da(da_ref)
    BETA = 0.85  # tuning knob: 0.6–0.9 reasonable
    return (sig_ref / max(1e-6, sig_act)) ** BETA

def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float, float]:
    # dir is FROM direction
    delta = math.radians((dir_deg - rwy_heading_deg) % 360.0)
    hw = speed_kn * math.cos(delta)   # headwind (+) / tailwind (−)
    cw = speed_kn * math.sin(delta)   # crosswind component (signed)
    return hw, cw

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    # conservative slope: +20% per +1% uphill (ignore downhill credit)
    if slope_pct > 0:
        d *= (1.0 + 0.20 * slope_pct)
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)
    return max(d, 0.0)

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
    st.error("dcs_airports.csv not found in repo root.")
    st.stop()

@st.cache_data
def load_perf() -> pd.DataFrame:
    try:
        df = pd.read_csv("f14_perf.csv", comment="#")
    except Exception as e:
        st.error("f14_perf.csv not found in repo root.")
        st.stop()
    # Normalize types & labels
    df["thrust"] = df["thrust"].astype(str).str.upper().replace({"MIL": "MILITARY", "AB": "AFTERBURNER"})
    num_cols = ["flap_deg","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["flap_deg","gw_lbs","press_alt_ft","oat_c","Vr_kt","ASD_ft","AGD_ft"])
    return df

# ------------------------------ interpolation (tri-linear over GW/PA/OAT) ------------------------------
def _bounds(vals, x):
    vals = sorted(set(map(float, vals)))
    lo = max([v for v in vals if v <= x], default=vals[0])
    hi = min([v for v in vals if v >= x], default=vals[-1])
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return lo, hi, w

def _interp_weight_at(sub: pd.DataFrame, pa: float, oat: float, field: str, gw_x: float) -> float:
    s = sub[(sub["press_alt_ft"] == pa) & (sub["oat_c"] == oat)].sort_values("gw_lbs")
    if s.empty:
        s = sub.sort_values(["press_alt_ft", "oat_c", "gw_lbs"])  # fallback
    xs = s["gw_lbs"].values.astype(float)
    ys = s[field].values.astype(float)
    if len(xs) < 2:
        return float(ys[0])
    if gw_x <= xs[0]:
        x0, x1 = xs[0], xs[1]; y0, y1 = ys[0], ys[1]
        return float(y0 + (y1 - y0)/(x1 - x0) * (gw_x - x0))
    if gw_x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]; y0, y1 = ys[-2], ys[-1]
        return float(y0 + (y1 - y0)/(x1 - x0) * (gw_x - x0))
    return float(np.interp(gw_x, xs, ys))

def interp_perf(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float):
    # For UP (0°), use MAN (20°) table as base
    use_flap = 20 if flap_deg == 0 else flap_deg
    sub = perf[(perf["flap_deg"] == use_flap) & (perf["thrust"] == thrust)]
    if sub.empty:
        # fallback: same flap, any thrust (rare), or same thrust, any flap
        sub = perf[(perf["flap_deg"] == use_flap)]
        if sub.empty:
            sub = perf[(perf["thrust"] == thrust)]
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

# ------------------------------ OEI climb guardrail ------------------------------
def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    # Very simple guardrail to avoid unrealistically deep derates
    # gradient ≈ T/W − D/W ; demand ~ 2.4% (121.189(b))
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct/100.0)  # MIL-scaled per-engine
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024

# ------------------------------ trim model ------------------------------
def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# ------------------------------ wind entry parsing ------------------------------
def parse_wind_entry(entry: str, unit: str) -> Optional[Tuple[float,float]]:
    text = (entry or "").strip().replace('/', ' ').replace('@', ' ')
    parts = [p for p in text.split(' ') if p]
    if len(parts) >= 2:
        try:
            d = float(parts[0]) % 360.0
            s = float(parts[1])
            if unit == "m/s":
                s *= 1.943844
            return (d, s)
        except Exception:
            return None
    return None

# ------------------------------ core computation ------------------------------
@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_aeo_liftoff_ft: float; agd_reg_oei_ft: float
    req_ft: float; avail_ft: float; limiting: str
    hw_kn: float; cw_kn: float; notes: list

def compute_takeoff(perfdb: pd.DataFrame,
                    rwy_heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                    field_elev_ft: float, slope_pct: float, shorten_ft: float,
                    oat_c: float, qnh_inhg: float,
                    wind_speed: float, wind_dir_deg: float, wind_units: str, wind_policy: str,
                    gw_lbs: float,
                    flap_mode: str,
                    thrust_mode: str,
                    target_n1_pct: float) -> Result:
    # Atmosphere & wind
    pa = pressure_altitude_ft(field_elev_ft, qnh_inhg)
    spd_kn = wind_speed if wind_units == "kts" else wind_speed * 1.943844
    hw, cw = wind_components(spd_kn, wind_dir_deg, rwy_heading_deg)

    notes: list[str] = []
    if hw < -10.0:
        notes.append("Tailwind component exceeds 10 kt — NOT AUTHORIZED.")
    if abs(cw) > 30.0:
        notes.append("Crosswind component exceeds 30 kt — NOT AUTHORIZED.")

    # Flaps (auto favors MAN)
    flap_text = "MANEUVER" if flap_mode == "Auto-Select" else flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)

    # Baseline table: always MIL for derate math; AB only if explicitly selected
    use_flap_for_table = 20 if flap_deg == 0 else flap_deg
    table_thrust = "AFTERBURNER" if thrust_mode == "AB" else "MILITARY"
    base = interp_perf(perfdb, use_flap_for_table, table_thrust, float(gw_lbs), float(pa), float(oat_c))

    vs = float(base["Vs_kt"]); v1 = float(base["V1_kt"])
    vr = float(base["Vr_kt"]); v2 = float(base["V2_kt"])

    # CSV: ASD_ft ≈ AEO accelerate-stop; AGD_ft ≈ AEO ground roll
    asd_base = float(base["ASD_ft"])
    agd_aeo_liftoff_base = float(base["AGD_ft"]) * LIFTOFF_FACTOR

    # UP penalty vs MAN baseline
    if flap_deg == 0:
        asd_base *= 1.06
        agd_aeo_liftoff_base *= 1.06

    # FULL cannot derate
    if flap_deg == 40 and thrust_mode in ("Auto-Select", "DERATE", "Manual Derate"):
        notes.append("Derate with FULL flaps not allowed — using MIL for calculation.")
        thrust_mode = "MIL"

    def n1_mult(n1pct: float) -> float:
        # MIL-anchored distance scaling
        eff = max(0.90, min(1.0, n1pct/100.0))
        return 1.0 / (eff ** ALPHA_N1_DIST)

    def maybe_da_scale(d_ft: float) -> float:
        d = d_ft
        if pa > 5000.0 or oat_c > 30.0:
            d *= da_out_of_grid_scale(pa, oat_c)
        return d

    def wind_slope(d_ft: float) -> float:
        return apply_wind_slope(d_ft, slope_pct, hw, wind_policy)

    def distances_for(n1pct: float) -> Tuple[float, float]:
        mult = n1_mult(n1pct)
        asd = wind_slope(maybe_da_scale(asd_base * mult))
        agd_aeo = wind_slope(maybe_da_scale(agd_aeo_liftoff_base * mult))
        agd_aeo *= AEO_CAL_FACTOR
        return asd, agd_aeo

    def field_ok(asd_eff: float, agd_aeo_eff: float) -> Tuple[bool, float, str]:
        tora_eff = max(0.0, tora_ft - shorten_ft)
        toda_eff = max(0.0, toda_ft - shorten_ft)
        asda_eff_lim = max(0.0, asda_ft - shorten_ft)
        # Clearway credit capped at 50% of available runway length
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow
        # Regulatory OEI continue distance
        agd_reg = agd_aeo_eff * OEI_AGD_FACTOR
        req = max(asd_eff, agd_reg)
        ok = (asd_eff <= asda_eff_lim) and (agd_reg <= tod_limit) and (agd_reg <= toda_eff)
        limiting = "ASD (stop)" if asd_eff >= agd_reg else "Engine-out continue"
        return ok, req, limiting

    # Choose thrust / N1
    n1 = 100.0
    thrust_text = thrust_mode

    if thrust_mode == "DERATE":
        floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
        n1 = max(floor_pct, float(target_n1_pct))
        n1 = float(int(math.ceil(n1)))  # round up to whole percent for gauge
        thrust_text = "DERATE" if n1 < 100.0 else "MIL"
    elif thrust_mode == "Auto-Select":
        # Check MIL first
        asd_mil, agd_aeo_mil = distances_for(100.0)
        ok_mil, _, _ = field_ok(asd_mil, agd_aeo_mil)
        if ok_mil:
            # Find minimal N1 that passes
            floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
            lo, hi = floor_pct, 100.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                asd_m, agd_aeo_m = distances_for(mid)
                ok_m, _, _ = field_ok(asd_m, agd_aeo_m)
                ok_m = ok_m and compute_oei_second_segment_ok(gw_lbs, mid, flap_deg)
                if ok_m:
                    hi = mid
                else:
                    lo = mid
            n1 = float(int(math.ceil(hi)))
            thrust_text = "DERATE" if n1 < 100.0 else "MIL"
        else:
            # Escalate once: FULL + MIL (derate not allowed)
            if flap_deg != 40:
                notes.append("Auto: MAN @ MIL fails §121.189; escalating to FULL @ MIL.")
            return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                                   field_elev_ft, slope_pct, shorten_ft,
                                   oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                                   gw_lbs, "FULL", "MIL", 100.0)
    elif thrust_mode == "MIL":
        n1 = 100.0
        thrust_text = "MIL"
    else:  # AB
        n1 = 100.0
        thrust_text = "AFTERBURNER"
        notes.append("Afterburner selected — NOT AUTHORIZED for F-14B except as last resort.")

    # Final distances
    asd_fin, agd_aeo_fin = distances_for(n1)
    agd_reg_fin = agd_aeo_fin * OEI_AGD_FACTOR

    ok, req, limiting = field_ok(asd_fin, agd_aeo_fin)

    # If Auto-flap and not OK at MAN, escalate to FULL+MIL
    if flap_mode == "Auto-Select" and not ok and flap_deg != 40:
        return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                               field_elev_ft, slope_pct, shorten_ft,
                               oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                               gw_lbs, "FULL", "MIL", 100.0)

    avail = max(0.0, tora_ft - shorten_ft)

    return Result(
        v1=v1, vr=vr, v2=v2, vs=vs,
        flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
        asd_ft=asd_fin, agd_aeo_liftoff_ft=agd_aeo_fin, agd_reg_oei_ft=agd_reg_fin,
        req_ft=req, avail_ft=avail, limiting=limiting,
        hw_kn=hw, cw_kn=cw, notes=notes
    )

# ------------------------------ UI ------------------------------
st.title("DCS F-14B Takeoff — FAA-Based Model (MIL-anchored)")

rwy_db = load_runways()
perfdb = load_perf()

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Theatre", sorted(rwy_db["map"].unique()))
    df_t = rwy_db[rwy_db["map"] == theatre]
    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]
    rwy_label = st.selectbox("Runway End", list(df_a["runway_label"]))
    rwy = df_a[df_a["runway_label"] == rwy_label].iloc[0]

    tora_ft = float(rwy["tora_ft"])
    toda_ft = float(rwy["toda_ft"])
    asda_ft = float(rwy["asda_ft"])
    elev_ft = float(rwy["threshold_elev_ft"])
    hdg = float(rwy["heading_deg"])
    slope = float(rwy.get("slope_percent", 0.0) or 0.0)

    cA, cB = st.columns(2)
    with cA:
        st.metric("TORA (ft)", f"{tora_ft:,.0f}")
        st.metric("TODA (ft)", f"{toda_ft:,.0f}")
    with cB:
        st.metric("ASDA (ft)", f"{asda_ft:,.0f}")
        st.metric("Elev (ft)", f"{elev_ft:,.0f}")

    st.caption("Shorten Available Runway")
    sh_val = st.number_input("Value", min_value=0.0, value=0.0, step=50.0, key="sh_val")
    sh_unit = st.selectbox("Units", ["ft", "NM"], index=0, key="sh_unit")
    shorten_total = float(sh_val) if sh_unit == "ft" else float(sh_val) * 6076.12

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    qnh_val = st.number_input("QNH value", value=29.92, step=0.01, format="%.2f")
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
    wind_policy = st.selectbox("Wind Policy", ["None", "50/150"], index=0)

    st.header("Weight & Config")
    mode = st.radio("Weight entry", ["Direct GW", "Fuel + Stores"], index=0)
    if mode == "Direct GW":
        gw = st.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=500.0)
    else:
        empty_w = st.number_input("Empty weight (lb)", min_value=38000.0, max_value=46000.0, value=41780.0, step=50.0)
        fuel_lb = st.number_input("Internal fuel (lb)", min_value=0.0, max_value=20000.0, value=8000.0, step=100.0)
        ext_tanks = st.selectbox("External tanks (267 gal)", [0,1,2], index=0)
        aim9 = st.slider("AIM-9 count", 0, 2, 0)
        aim7 = st.slider("AIM-7 count", 0, 4, 0)
        aim54 = st.slider("AIM-54 count", 0, 6, 0)
        lantirn = st.checkbox("LANTIRN pod")
        wcalc = empty_w + fuel_lb + ext_tanks*1900 + aim9*190 + aim7*510 + aim54*1000 + (440 if lantirn else 0)
        gw = st.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(wcalc), step=10.0)

    flap_mode = st.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    st.header("Thrust")
    thrust_mode = st.radio("Mode", ["Auto-Select", "Manual Derate", "MIL", "AB"], index=0)
    derate_n1 = 98.0
    if thrust_mode == "Manual Derate":
        flap_for_floor = 0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20)
        floor = DERATE_FLOOR_BY_FLAP.get(flap_for_floor, 0.90)*100.0
        st.caption(f"Derate floor by flap: {floor:.0f}% N1 (MIL)")
        derate_n1 = st.slider("Target N1 % (MIL)", min_value=float(int(floor)), max_value=100.0, value=max(95.0, float(int(floor))), step=1.0)

    with st.expander("Advanced / Calibration", expanded=False):
        calib = st.radio("Model calibration", ["FAA-conservative", "DCS-calibrated"], index=1,
                         help=("FAA: AEO liftoff uncalibrated; engine-out factor 1.20 (conservative). "
                               "DCS: engine-out factor 1.15 (matches your hot/high tests)."))
        if calib == "DCS-calibrated":
            AEO_CAL = 1.00
            OEI_FAC = 1.15
        else:
            AEO_CAL = 1.00
            OEI_FAC = 1.20
        globals()["AEO_CAL_FACTOR"] = AEO_CAL
        globals()["OEI_AGD_FACTOR"] = OEI_FAC

run = st.button("Compute Takeoff Performance", type="primary")

if run:
    res = compute_takeoff(perfdb,
                          float(hdg), float(tora_ft), float(toda_ft), float(asda_ft),
                          float(elev_ft), float(slope), float(shorten_total),
                          float(oat_c), float(qnh_inhg),
                          float(wind_spd), float(wind_dir), str(wind_units), str(wind_policy),
                          float(gw), str(flap_mode),
                          ("DERATE" if thrust_mode=="Manual Derate" else thrust_mode), float(derate_n1))

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.subheader("V-Speeds")
        st.metric("V1 (kt)", f"{res.v1:.0f}")
        st.metric("Vr (kt)", f"{res.vr:.0f}")
        st.metric("V2 (kt)", f"{res.v2:.0f}")
        st.metric("Vs (kt)", f"{res.vs:.0f}")
    with c2:
        st.subheader("Settings")
        st.metric("Flaps", res.flap_text)
        thrust_label = "MIL" if (res.thrust_text.upper().startswith("MIL") or res.n1_pct >= 100.0) else "DERATE"
        st.metric("Thrust", f"{thrust_label} ({res.n1_pct:.0f}% N1)")
        flap_deg = 0 if res.flap_text.upper().startswith("UP") else (40 if res.flap_text.upper().startswith("FULL") else 20)
        st.metric("Trim (ANU)", f"{trim_anu(float(gw), flap_deg):.1f}")
    with c3:
        st.subheader("Runway distances")
        st.metric("Stop distance (ft)", f"{res.asd_ft:.0f}")
        st.metric("Continue distance (engine-out, regulatory) (ft)", f"{res.agd_reg_oei_ft:.0f}")
        st.metric("Required runway (regulatory) (ft)", f"{res.req_ft:.0f}")
    with c4:
        st.subheader("Availability")
        st.metric("Runway available (ft)", f"{res.avail_ft:.0f}")
        st.metric("Limiting factor", res.limiting)
        st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
        st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
        st.caption("Tailwind > 10 kt or crosswind > 30 kt → NOT AUTHORIZED.")

        # Compliance banner & margins
        tora_eff = max(0.0, float(tora_ft) - float(shorten_total))
        toda_eff = max(0.0, float(toda_ft) - float(shorten_total))
        asda_eff_lim = max(0.0, float(asda_ft) - float(shorten_total))
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        agd_reg = res.agd_reg_oei_ft
        asd_ok = res.asd_ft <= asda_eff_lim
        agd_ok = (agd_reg <= tod_limit) and (agd_reg <= toda_eff)
        ok = asd_ok and agd_ok

        asd_margin = asda_eff_lim - res.asd_ft
        agd_margin = tod_limit - agd_reg
        req_margin = min(asd_margin, agd_margin)

        if ok:
            st.success(f"COMPLIANT — Margin {req_margin:.0f} ft (ASD margin {asd_margin:.0f}, OEI-continue margin {agd_margin:.0f}).")
        else:
            st.error(f"NOT AUTHORIZED — Short by {-req_margin:.0f} ft (ASD margin {asd_margin:.0f}, OEI-continue margin {agd_margin:.0f}).")
            st.caption(f"TOD limit: {tod_limit:.0f} ft | ASDA: {asda_eff_lim:.0f} ft | OEI-continue: {agd_reg:.0f} ft")

    st.markdown("---")
    st.subheader("All-engines takeoff estimates (for DCS comparison)")
    e1, e2 = st.columns(2)
    with e1:
        st.metric("Vr ground roll (ft)", f"{res.agd_aeo_liftoff_ft * AEO_VR_FRAC:.0f}")
    with e2:
        st.metric("Liftoff distance to 35 ft (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")
    st.caption("These are all-engines estimates. Regulatory checks assume an engine-out at V1 per 14 CFR 121.189.")

    for n in res.notes:
        st.warning(n)

else:
    st.info("Set inputs and click Compute.")
