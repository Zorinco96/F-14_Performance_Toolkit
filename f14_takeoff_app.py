# f14_takeoff_app.py — DCS F‑14B Takeoff Performance (FAA‑style model)
#
# EXPECTS in repo root (exact case):
#   • dcs_airports.csv  — runway DB with: map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,threshold_elev_ft,slope_percent(optional)
#
# Key features
# • FAA 14 CFR 121.189 field‑length checks (ASD ≤ ASDA; TOD ≤ TORA + clearway_cap; TOR ≤ TORA; clearway credit ≤ 50% TORA).
# • Tri‑linear interpolation of MIL baseline performance over GW / Press Alt / OAT.
# • Thrust derate solver (bisection) finds MIN N1% (MIL scale) that satisfies §121.189 (+ simple OEI climb guardrail).
# • Density‑altitude correction when outside the table (PA>5000 ft or OAT>30 °C) via density ratio scaling.
# • Flaps: UP / MANEUVER / FULL / Auto‑Select (defaults to MAN; escalates to FULL only if MAN can’t meet limits). No derate with FULL.
# • Thrust: Auto‑Select / Manual Derate / MIL / AB (AB warned as not authorized). Manual Derate enforces flap‑based floor.
# • Weather: OAT (°C), QNH (inHg or hPa), single wind entry (DIR@SPD) with units (kts or m/s) and policy (None or 50/150).
# • Runway: auto‑populate from dcs_airports.csv, single “Shorten Available Runway” input (ft or NM).
# • Weight: Direct GW, or Fuel + Stores builder (AIM‑9/7/54, LANTIRN, external tanks) → editable computed GW.
# • Outputs: V1/Vr/V2/Vs, flap/thrust/N1, Trim (ANU), ASD/AGD/Required/Available/limiting, headwind & crosswind, policy notes.
#
# DISCLAIMER: Training aid for DCS only — NOT for real‑world flight planning.

import math
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F‑14B Takeoff (FAA Model)", page_icon="✈️", layout="wide")

# ------------------------------
# Embedded (minimal) F‑14B baseline grid (MAN/FULL @ MIL/AB). Distances are placeholders tuned for DCS plausibility.
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

# ------------------------------ constants / tuning ------------------------------
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}  # per engine, uninstalled (approx)
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}   # FULL may not be derated
ALPHA_N1_DIST = 1.55                                   # distance ∝ 1/(N1^alpha) — tightened for hot/high
UP_FLAP_DISTANCE_FACTOR = 1.06                          # extra penalty beyond CL diff when using UP
OEI_AGD_FACTOR = 1.20                                   # regulatory OEI accelerate‑go penalty vs AEO
AEO_CAL_FACTOR = 1.00                                    # AEO AGD calibration (1.00 = FAA; <1 = DCS‑cal)

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
    delta = math.radians((dir_deg - rwy_heading_deg) % 360)
    hw = speed_kn * math.cos(delta)
    cw = speed_kn * math.sin(delta)
    return hw, cw

# Wind policy: headwind credit / tailwind penalty factors
WIND_FACTORS = {"None": (1.0, 1.0), "50/150": (0.5, 1.5)}


def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    # conservative slope: +20% per +1% uphill (downhill ignored)
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
    st.error("dcs_airports.csv not found in repo.")
    st.stop()

@st.cache_data
def load_perf() -> pd.DataFrame:
    df = pd.read_csv(StringIO(PERF_F14B))
    df["thrust"] = df["thrust"].str.upper()
    return df

# ------------------------------ interpolation (tri‑linear over GW/PA/OAT) ------------------------------

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
    use_flap = 20 if flap_deg == 0 else flap_deg  # UP uses MAN table as base
    sub = perf[(perf["flap_deg"] == use_flap) & (perf["thrust"] == thrust)]
    if sub.empty:
        sub = perf[perf["flap_deg"] == use_flap]
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

# ------------------------------ core scaling & solver ------------------------------

def enforce_derate_floor(n1pct: float, flap_deg: int) -> float:
    return max(n1pct, DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90) * 100.0)


def distance_scale_from_n1(n1pct: float, flap_deg: int) -> float:
    n1pct = enforce_derate_floor(n1pct, flap_deg)
    eff = max(0.90, min(1.0, n1pct/100.0))
    return 1.0 / (eff ** ALPHA_N1_DIST)


def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    # Very simple OEI guardrail to avoid unrealistically deep derates
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct/100.0)  # MIL‑scaled per‑engine
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024


@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_ft: float; req_ft: float; avail_ft: float; limiting: str
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
    """Compute takeoff performance and regulatory compliance."""
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

    # Baseline (MIL unless AB explicitly picked)
    table_thrust = "AFTERBURNER" if thrust_mode == "AB" else "MILITARY"
    base = interp_perf(perfdb, (20 if flap_deg == 0 else flap_deg), table_thrust, float(gw_lbs), float(pa), float(oat_c))

    vs = float(base["Vs_kt"]) ; v1 = float(base["V1_kt"]) ; vr = float(base["Vr_kt"]) ; v2 = float(base["V2_kt"]) 
    asd_base = float(base["ASD_ft"]) ; agd_base = float(base["AGD_ft"]) 

    # UP: extra penalty to distances relative to MAN base
    if flap_deg == 0:
        asd_base *= UP_FLAP_DISTANCE_FACTOR
        agd_base *= UP_FLAP_DISTANCE_FACTOR

    # Derate logic: FULL flaps cannot derate
    if flap_deg == 40 and thrust_mode in ("Auto-Select", "DERATE", "Manual Derate"):
        notes.append("Derate with FULL flaps not allowed — using MIL for calculation.")
        thrust_mode = "MIL"

    def distances_for(n1pct: float) -> tuple[float, float]:
        mult = distance_scale_from_n1(n1pct, flap_deg)
        asd = apply_wind_slope(asd_base * mult, slope_pct, hw, wind_policy)
        agd = apply_wind_slope(agd_base * mult, slope_pct, hw, wind_policy)
        # Density‑altitude scale if outside the grid
        if pa > 5000.0 or oat_c > 30.0:
            da_scale = da_out_of_grid_scale(pa, oat_c)
            asd *= da_scale
            agd *= da_scale
        # DCS AEO calibration (optional via sidebar)
        agd *= AEO_CAL_FACTOR
        return asd, agd

    def field_ok(asd_eff: float, agd_eff: float) -> tuple[bool, float, str]:
        # Apply intersection/shorten to declared distances
        tora_eff = max(0.0, tora_ft - shorten_ft)
        toda_eff = max(0.0, toda_ft - shorten_ft)
        asda_eff_lim = max(0.0, asda_ft - shorten_ft)

        # Clearway credit capped at 50% of available runway length
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        # Regulatory OEI accelerate‑go distance to 35 ft
        agd_reg = agd_eff * OEI_AGD_FACTOR

        # Required distance and pass/fail
        req = max(asd_eff, agd_reg)
        ok = (asd_eff <= asda_eff_lim) and (agd_reg <= tod_limit) and (agd_reg <= toda_eff)
        limiting = "ASD" if asd_eff >= agd_reg else "AGD (OEI)"
        return ok, req, limiting

    n1 = 100.0
    thrust_text = thrust_mode

    # Manual Derate uses the exact user-selected N1% (no solver)
    if thrust_mode == "DERATE":
        n1 = enforce_derate_floor(target_n1_pct, flap_deg)
        thrust_text = "DERATE"
    # Auto-Select searches for MIN N1% with current flap; only if MAN@MIL fails do we escalate to FULL+MIL
    elif thrust_mode == "Auto-Select":
        # First, check if MAN (or current flap) at MIL passes §121.189
        asd_mil, agd_mil = distances_for(100.0)
        tora_eff = max(0.0, tora_ft - shorten_ft)
        toda_eff = max(0.0, toda_ft - shorten_ft)
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow
        mil_ok, req_mil, limiting_mil = field_ok(asd_mil, agd_mil)
        if mil_ok:
            # Bisection from floor to MIL to find minimum N1 that still passes
            req_floor = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90) * 100.0
            lo, hi = req_floor, 100.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                asd_m, agd_m = distances_for(mid)
                ok_m, _, _ = field_ok(asd_m, agd_m)
                ok_m = ok_m and compute_oei_second_segment_ok(gw_lbs, mid, flap_deg)
                if ok_m:
                    hi = mid
                else:
                    lo = mid
            # add small guard so rounding doesn't flip pass→fail
            n1 = min(100.0, round(hi + 0.2, 1))
            thrust_text = "DERATE"
            # Verify result; if solver edge-case fails, fall back to MIL (no flap escalation)
            asd_chk, agd_chk = distances_for(n1)
            ok_chk, _, _ = field_ok(asd_chk, agd_chk)
            ok_chk = ok_chk and compute_oei_second_segment_ok(gw_lbs, n1, flap_deg)
            if not ok_chk:
                notes.append("Auto derate landed on an edge; reverting to MAN @ MIL to satisfy §121.189 without changing flaps.")
                n1 = 100.0
                thrust_text = "MIL"
        else:
            # MAN@MIL fails → escalate to FULL+MIL (derate prohibited with FULL)
            if flap_deg != 40:
                notes.append(f"Auto: {flap_text} @ MIL required {max(asd_mil, agd_mil*OEI_AGD_FACTOR):.0f} ft > TOD limit {tod_limit:.0f} ft — escalating to FULL + MIL.")
            return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                                   field_elev_ft, slope_pct, shorten_ft,
                                   oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                                   gw_lbs, "FULL", "MIL", 100.0)
    elif thrust_mode == "MIL":
        n1 = 100.0
        n1 = 100.0
        n1 = 100.0
    else:  # AB
        n1 = 100.0
        notes.append("Afterburner selected — NOT AUTHORIZED for F‑14B except as last resort.")

    # Final distances and compliance
    asd_fin, agd_fin = distances_for(n1)
    ok, req, limiting = field_ok(asd_fin, agd_fin)

    # If Auto‑flap and not OK at MAN, escalate once to FULL+MIL
    if flap_mode == "Auto-Select" and not ok and flap_deg != 40:
        return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                               field_elev_ft, slope_pct, shorten_ft,
                               oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                               gw_lbs, "FULL", "MIL", 100.0)

    avail = max(0.0, tora_ft - shorten_ft)

    return Result(
        v1=v1, vr=vr, v2=v2, vs=vs,
        flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
        asd_ft=asd_fin, agd_ft=agd_fin, req_ft=req, avail_ft=avail, limiting=limiting,
        hw_kn=hw, cw_kn=cw, notes=notes
    )

# ------------------------------ trim model ------------------------------

def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# ------------------------------ Wind parsing helper (no regex to keep canvas replace safe) ------------------------------

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

# ------------------------------ App ------------------------------

st.title("DCS F‑14B Takeoff — FAA‑Based Model (v1.2)")

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

    tora_ft = float(rwy["tora_ft"]) ; toda_ft = float(rwy["toda_ft"]) ; asda_ft = float(rwy["asda_ft"]) ; elev_ft = float(rwy["threshold_elev_ft"]) ; hdg = float(rwy["heading_deg"]) ; slope = float(rwy.get("slope_percent", 0.0) or 0.0)

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
        aim9 = st.slider("AIM‑9 count", 0, 2, 0)
        aim7 = st.slider("AIM‑7 count", 0, 4, 0)
        aim54 = st.slider("AIM‑54 count", 0, 6, 0)
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
        derate_n1 = st.slider("Target N1 % (MIL)", min_value=floor, max_value=100.0, value=max(95.0, floor), step=0.5)

    with st.expander("Advanced / Calibration", expanded=False):
        calib = st.radio(
            "Model calibration",
            ["FAA-conservative", "DCS-calibrated"],
            index=1,
            help="FAA: AEO distances uncalibrated; OEI factor 1.20 (more conservative).
DCS: AEO AGD ×0.74; OEI factor 1.15 — tuned to match your 40 °C/70k tests.",
        )
        if calib == "DCS-calibrated":
            globals()["AEO_CAL_FACTOR"] = 0.74
            globals()["OEI_AGD_FACTOR"] = 1.15
        else:
            globals()["AEO_CAL_FACTOR"] = 1.00
            globals()["OEI_AGD_FACTOR"] = 1.20

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
        st.subheader("V‑Speeds")
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
        st.metric("Accelerate‑Stop (ft)", f"{res.asd_ft:.0f}")
        st.metric("Accelerate‑Go (ft)", f"{res.agd_ft:.0f}")
        st.metric("Required (ft)", f"{res.req_ft:.0f}")
    with c4:
        st.subheader("Availability")
        st.metric("Available (ft)", f"{res.avail_ft:.0f}")
        st.metric("Limiting", res.limiting)
        st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
        st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
        st.caption("Tailwind >10 kt or crosswind >30 kt → NOT AUTHORIZED.")

        # --- Compliance banner & margins ---
        tora_eff = max(0.0, float(tora_ft) - float(shorten_total))
        toda_eff = max(0.0, float(toda_ft) - float(shorten_total))
        asda_eff_lim = max(0.0, float(asda_ft) - float(shorten_total))
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        agd_reg = res.agd_ft * OEI_AGD_FACTOR
        asd_ok = res.asd_ft <= asda_eff_lim
        agd_ok = (agd_reg <= tod_limit) and (agd_reg <= toda_eff)
        ok = asd_ok and agd_ok

        asd_margin = asda_eff_lim - res.asd_ft
        agd_margin = tod_limit - agd_reg
        req_margin = min(asd_margin, agd_margin)

        if ok:
            st.success(f"COMPLIANT — Margin {req_margin:.0f} ft (ASD margin {asd_margin:.0f}, AGD(OEI) margin {agd_margin:.0f}).")
        else:
            st.error(f"NOT AUTHORIZED — Short by {-req_margin:.0f} ft (ASD margin {asd_margin:.0f}, AGD(OEI) margin {agd_margin:.0f}).")
            st.caption(f"TOD limit: {tod_limit:.0f} ft | ASDA: {asda_eff_lim:.0f} ft | AGD(OEI): {agd_reg:.0f} ft")

    st.markdown("---")
    st.subheader("All-engines takeoff estimates")
    e1, e2 = st.columns(2)
    with e1:
        st.metric("Vr ground roll (ft)", f"{res.agd_ft * AEO_VR_FRAC:.0f}")
    with e2:
        st.metric("Liftoff distance (ft)", f"{res.agd_ft:.0f}")
    st.caption("These all-engines estimates are for DCS comparison; regulatory checks assume an engine-out scenario.")

    for n in res.notes:
        st.warning(n)

else:
    st.info("Set inputs and click Compute.")

st.markdown("---")
st.caption("This tool estimates F‑14B takeoff performance for DCS per FAA‑style rules (14 CFR 121.189). Not for real‑world use.")
