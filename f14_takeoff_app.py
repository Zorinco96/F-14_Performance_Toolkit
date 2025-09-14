# f14_takeoff_app.py — DCS F-14B Takeoff Performance
# Reads:
#   - dcs_airports.csv
#   - f14_perf.csv    (model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note)
# NOT FOR REAL-WORLD USE.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F-14B Takeoff (FAA Model)", page_icon="✈️", layout="wide")

# ---------------- constants / tuning ----------------
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}  # per engine (approx) for guardrail
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}  # FULL may not derate
ALPHA_N1_DIST = 2.0                                   # distance ∝ 1/(N1^alpha)
OEI_AGD_FACTOR = 1.20                                  # default (UI can set 1.15)
AEO_CAL_FACTOR = 1.00                                  # keep 1.0 (DA handled separately)
AEO_VR_FRAC = 0.88
AEO_VR_FRAC_FULL = 0.82
WIND_FACTORS = {"None": (1.0, 1.0), "50/150": (0.5, 1.5)}  # head credit / tail penalty
# Speed floors (reasonable, to avoid ultra-low light-weight results)
VMCG_FLOOR = {40: 110.0, 20: 120.0, 0: 130.0}

# -------------- atmosphere / wind --------------
def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - qnh_inhg) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (h_ft / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    return float(pa_ft + 120.0 * (oat_c - isa_temp_c_at_ft(pa_ft)))

def sigma_from_da(da_ft: float) -> float:
    h_m = da_ft * 0.3048
    T0 = 288.15; L = 0.0065; g = 9.80665; R = 287.05
    p0 = 101325.0; rho0 = 1.225
    T = T0 - L*h_m
    p = p0 * (T/T0)**(g/(R*L))
    rho = p/(R*T)
    return float(rho/rho0)

def da_out_of_grid_scale(pa_ft: float, oat_c: float) -> float:
    da_act = density_altitude_ft(pa_ft, oat_c)
    da_ref = density_altitude_ft(min(pa_ft, 5000.0), min(oat_c, 30.0))
    sig_act = sigma_from_da(da_act); sig_ref = sigma_from_da(da_ref)
    BETA = 0.85
    return (sig_ref / max(1e-6, sig_act)) ** BETA

def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float, float]:
    delta = math.radians((dir_deg - rwy_heading_deg) % 360.0)
    hw = speed_kn * math.cos(delta)
    cw = speed_kn * math.sin(delta)
    return hw, cw

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    if slope_pct > 0:
        d *= (1.0 + 0.20 * slope_pct)  # +20% per +1% up-slope
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)
    return max(d, 0.0)

# -------------- loaders --------------
@st.cache_data
def load_runways() -> pd.DataFrame:
    for path in ["dcs_airports.csv", "data/dcs_airports.csv"]:
        try:
            df = pd.read_csv(path)
            df["runway_label"] = df["airport_name"] + " " + df["runway_end"].astype(str) + " (" + df["runway_pair"].astype(str) + ")"
            return df
        except Exception:
            continue
    st.error("dcs_airports.csv not found.")
    st.stop()

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_flap20(df: pd.DataFrame) -> pd.DataFrame:
    if (df["flap_deg"] == 20).any():
        return df
    keys = ["thrust","gw_lbs","press_alt_ft","oat_c"]
    up = df[df["flap_deg"] == 0]
    full = df[df["flap_deg"] == 40]
    if up.empty or full.empty:
        return df
    m = pd.merge(up, full, on=keys, suffixes=("_up","_full"))
    if m.empty:
        return df
    def blend(a,b,w=0.4): return w*a + (1.0-w)*b
    new = pd.DataFrame({
        "model": "F-14B",
        "flap_deg": 20,
        "thrust": m["thrust"],
        "gw_lbs": m["gw_lbs"],
        "press_alt_ft": m["press_alt_ft"],
        "oat_c": m["oat_c"],
        "Vs_kt":  blend(m["Vs_kt_up"],  m["Vs_kt_full"]),
        "V1_kt":  blend(m["V1_kt_up"],  m["V1_kt_full"]),
        "Vr_kt":  blend(m["Vr_kt_up"],  m["Vr_kt_full"]),
        "V2_kt":  blend(m["V2_kt_up"],  m["V2_kt_full"]),
        "ASD_ft": blend(m["ASD_ft_up"], m["ASD_ft_full"]),
        "AGD_ft": blend(m["AGD_ft_up"], m["AGD_ft_full"]),
        "note": "synth-MAN(20)=0.4*UP+0.6*FULL"
    })
    return pd.concat([df, new], ignore_index=True)

@st.cache_data
def load_perf() -> pd.DataFrame:
    try:
        df = pd.read_csv("f14_perf.csv", comment="#")
    except Exception:
        st.error("f14_perf.csv not found.")
        st.stop()
    df["thrust"] = df["thrust"].astype(str).str.upper().replace({"MIL":"MILITARY","AB":"AFTERBURNER"})
    df = _ensure_numeric(df, ["flap_deg","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"])
    df = df.dropna(subset=["flap_deg","gw_lbs","press_alt_ft","oat_c","Vr_kt","ASD_ft","AGD_ft"])
    df = ensure_flap20(df)
    return df

def estimate_ab_multiplier(perfdb: pd.DataFrame, flap_deg: int) -> float:
    def median_ratio(df_m: pd.DataFrame, df_a: pd.DataFrame) -> Optional[float]:
        keys = ["gw_lbs","press_alt_ft","oat_c"]
        M = df_m[keys+["ASD_ft","AGD_ft"]]; A = df_a[keys+["ASD_ft","AGD_ft"]]
        merged = pd.merge(M,A,on=keys,suffixes=("_MIL","_AB"))
        if merged.empty: return None
        ratios = []
        for col in ("ASD_ft","AGD_ft"):
            num = merged[f"{col}_AB"].to_numpy(float); den = merged[f"{col}_MIL"].to_numpy(float)
            valid = den > 1.0
            if valid.any():
                ratios.append(float(np.median(np.clip(num[valid]/den[valid],0.65,0.95))))
        return min(ratios) if ratios else None
    mil_f = perfdb[(perfdb["flap_deg"]==flap_deg)&(perfdb["thrust"]=="MILITARY")]
    ab_f  = perfdb[(perfdb["flap_deg"]==flap_deg)&(perfdb["thrust"]=="AFTERBURNER")]
    r = median_ratio(mil_f, ab_f)
    if r is not None: return r
    mil_any = perfdb[perfdb["thrust"]=="MILITARY"]; ab_any = perfdb[perfdb["thrust"]=="AFTERBURNER"]
    r2 = median_ratio(mil_any, ab_any)
    return r2 if r2 is not None else 0.82

# -------------- interpolation (tri-linear) --------------
def _bounds(vals, x):
    vals = sorted(set(map(float, vals)))
    lo = max([v for v in vals if v <= x], default=vals[0])
    hi = min([v for v in vals if v >= x], default=vals[-1])
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return lo, hi, w

def _interp_weight_at(sub: pd.DataFrame, pa: float, oat: float, field: str, gw_x: float) -> float:
    s = sub[(sub["press_alt_ft"]==pa)&(sub["oat_c"]==oat)].sort_values("gw_lbs")
    if s.empty: s = sub.sort_values(["press_alt_ft","oat_c","gw_lbs"])
    xs = s["gw_lbs"].values.astype(float); ys = s[field].values.astype(float)
    if len(xs) < 2: return float(ys[0])
    if gw_x <= xs[0]:
        x0,x1 = xs[0],xs[1]; y0,y1 = ys[0],ys[1]
        return float(y0 + (y1-y0)/(x1-x0) * (gw_x-x0))
    if gw_x >= xs[-1]:
        x0,x1 = xs[-2],xs[-1]; y0,y1 = ys[-2],ys[-1]
        return float(y0 + (y1-y0)/(x1-x0) * (gw_x-x0))
    return float(np.interp(gw_x, xs, ys))

def interp_perf(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float):
    use_flap = 20 if flap_deg == 0 else flap_deg
    sub = perf[(perf["flap_deg"]==use_flap)&(perf["thrust"]==thrust)]
    if sub.empty: sub = perf[(perf["flap_deg"]==use_flap)]
    if sub.empty: sub = perf[(perf["thrust"]==thrust)]
    if sub.empty: sub = perf
    pa0,pa1,wp = _bounds(sub["press_alt_ft"].unique(), pa)
    t0,t1,wt   = _bounds(sub["oat_c"].unique(),        oat)
    out = {}
    for field in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
        v00 = _interp_weight_at(sub, pa0,t0, field, gw)
        v01 = _interp_weight_at(sub, pa0,t1, field, gw)
        v10 = _interp_weight_at(sub, pa1,t0, field, gw)
        v11 = _interp_weight_at(sub, pa1,t1, field, gw)  # fixed t1 here
        v0 = v00*(1.0-wt) + v01*wt
        v1 = v10*(1.0-wt) + v11*wt
        out[field] = v0*(1.0-wp) + v1*wp
    return out

# -------------- NATOPS detection --------------
def agd_is_liftoff_mode(perfdb: pd.DataFrame, flap_deg: int, thrust: str) -> bool:
    sub = perfdb[(perfdb["flap_deg"]==flap_deg)&(perfdb["thrust"]==thrust)]
    if sub.empty: return False
    mask = sub["note"].astype(str).str.contains("NATOPS", case=False, na=False)
    return bool(mask.mean() >= 0.5)

# -------------- OEI guardrail --------------
def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct/100.0)
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024  # 2.4%

# -------------- trim --------------
def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# -------------- wind text parsing --------------
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

# -------------- speed floors (robust) --------------
def enforce_speed_floors(vs: float, v1: float, vr: float, v2: float, flap_deg: int) -> tuple[float, float, float]:
    """Apply sane floors and monotonic relationships to V1/Vr/V2.
    Robust against NaN/Inf by falling back to reasonable values derived from Vs."""
    def _safe(x, fallback):
        try:
            x = float(x)
            if math.isfinite(x):
                return x
        except Exception:
            pass
        return float(fallback)

    # Base fallbacks if the interpolator ever yields junk
    vs  = _safe(vs, 120.0)
    v1  = _safe(v1, vs + 10.0)
    vr  = _safe(vr, vs + 20.0)
    v2  = _safe(v2, vs + 30.0)

    # Vmcg-ish floor by flap + monotonic relationships
    vmcg = VMCG_FLOOR.get(flap_deg, 120.0)
    v1f = max(v1, vmcg, vs + 5.0)
    vrf = max(vr, v1f + 5.0, vs + 10.0)
    v2f = max(v2, vrf + 10.0, 1.2 * vs)

    # Return floats; UI already formats with .0f
    return v1f, vrf, v2f


# -------------- result struct --------------
@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_aeo_liftoff_ft: float; agd_reg_oei_ft: float
    req_ft: float; avail_ft: float; limiting: str
    hw_kn: float; cw_kn: float; notes: list

# -------------- core compute --------------
def compute_takeoff(perfdb: pd.DataFrame,
                    rwy_heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                    field_elev_ft: float, slope_pct: float, shorten_ft: float,
                    oat_c: float, qnh_inhg: float,
                    wind_speed: float, wind_dir_deg: float, wind_units: str, wind_policy: str,
                    gw_lbs: float,
                    flap_mode: str,
                    thrust_mode: str,
                    target_n1_pct: float) -> Result:

    pa = pressure_altitude_ft(field_elev_ft, qnh_inhg)
    da = density_altitude_ft(pa, oat_c)
    spd_kn = wind_speed if wind_units == "kts" else wind_speed * 1.943844
    hw, cw = wind_components(spd_kn, wind_dir_deg, rwy_heading_deg)

    notes: list[str] = []
    if hw < -10.0: notes.append("Tailwind component exceeds 10 kt — NOT AUTHORIZED.")
    if abs(cw) > 30.0: notes.append("Crosswind component exceeds 30 kt — NOT AUTHORIZED.")

    # Auto flaps default to MAN; escalate to FULL+MIL only if MAN@MIL fails regulation
    flap_text = "MANEUVER" if flap_mode == "Auto-Select" else flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)
    use_flap_for_table = 20 if flap_deg == 0 else flap_deg

    # Decide thrust table to interpolate (AB only if explicitly selected and slice exists)
    has_ab_slice = not perfdb[(perfdb["flap_deg"]==use_flap_for_table)&(perfdb["thrust"]=="AFTERBURNER")].empty
    if thrust_mode == "AB" and has_ab_slice:
        table_thrust = "AFTERBURNER"
    else:
        table_thrust = "MILITARY"

    # Bounds → DA top-up only outside grid
    sub_bounds = perfdb[(perfdb["flap_deg"]==use_flap_for_table)&(perfdb["thrust"]==table_thrust)]
    if sub_bounds.empty: sub_bounds = perfdb[(perfdb["flap_deg"]==use_flap_for_table)]
    if sub_bounds.empty: sub_bounds = perfdb[(perfdb["thrust"]==table_thrust)]
    if sub_bounds.empty: sub_bounds = perfdb
    pa_min, pa_max = float(sub_bounds["press_alt_ft"].min()), float(sub_bounds["press_alt_ft"].max())
    t_min,  t_max  = float(sub_bounds["oat_c"].min()),           float(sub_bounds["oat_c"].max())
    outside_grid = (pa < pa_min or pa > pa_max or oat_c < t_min or oat_c > t_max)

    # Interpolate
    base = interp_perf(perfdb, use_flap_for_table, table_thrust, float(gw_lbs), float(pa), float(oat_c))
    vs  = float(base["Vs_kt"]); v1 = float(base["V1_kt"]); vr = float(base["Vr_kt"]); v2 = float(base["V2_kt"])
    # Apply floors & monotonicity
    v1, vr, v2 = enforce_speed_floors(vs, v1, vr, v2, flap_deg)

    asd_base = float(base["ASD_ft"])
    agd_csv  = float(base["AGD_ft"])

    natops_liftoff = agd_is_liftoff_mode(perfdb, use_flap_for_table, table_thrust)
    if natops_liftoff:
        agd_aeo_liftoff_base = agd_csv
    else:
        liftoff_factor = 1.42 + 0.15 * max(0.0, min(da/8000.0, 1.25))
        agd_aeo_liftoff_base = agd_csv * liftoff_factor

    if flap_deg == 0:
        asd_base *= 1.06
        agd_aeo_liftoff_base *= 1.06

    if flap_deg == 40 and thrust_mode in ("Auto-Select","DERATE","Manual Derate"):
        notes.append("Derate with FULL flaps not allowed — using MIL.")
        thrust_mode = "MIL"

    if thrust_mode == "AB" and not has_ab_slice:
        ab_mult = estimate_ab_multiplier(perfdb, use_flap_for_table)
        asd_base *= ab_mult
        agd_aeo_liftoff_base *= ab_mult
        notes.append(f"AFTERBURNER table missing; approximated AB/MIL ratio ≈ {ab_mult:.2f}.")
        # AB is always full rated; N1 stays 100 later

    def mult_from_n1(n1pct: float) -> float:
        eff = max(0.90, min(1.0, n1pct/100.0))
        return 1.0 / (eff ** ALPHA_N1_DIST)

    def maybe_da_scale(d_ft: float) -> float:
        return d_ft * da_out_of_grid_scale(pa, oat_c) if outside_grid else d_ft

    def wind_slope(d_ft: float) -> float:
        return apply_wind_slope(d_ft, slope_pct, hw, wind_policy)

    def distances_for(n1pct: float) -> Tuple[float,float]:
        m = mult_from_n1(n1pct)
        asd = wind_slope(maybe_da_scale(asd_base * m))
        agd_aeo = wind_slope(maybe_da_scale(agd_aeo_liftoff_base * m))
        agd_aeo *= AEO_CAL_FACTOR
        return asd, agd_aeo

    def field_ok(asd_eff: float, agd_cont_ft: float, engine_out: bool) -> Tuple[bool,float,str,float,float]:
        tora_eff = max(0.0, tora_ft - shorten_ft)
        toda_eff = max(0.0, toda_ft - shorten_ft)
        asda_eff = max(0.0, asda_ft - shorten_ft)
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow
        if engine_out:
            cont = agd_cont_ft * OEI_AGD_FACTOR
            limiting = "ASD (stop)" if asd_eff >= cont else "Engine-out continue"
        else:
            cont = agd_cont_ft
            limiting = "ASD (stop)" if asd_eff >= cont else "All-engines continue"
        req = max(asd_eff, cont)
        ok = (asd_eff <= asda_eff) and (cont <= tod_limit) and (cont <= toda_eff)
        return ok, req, limiting, asda_eff, tod_limit

    # Choose thrust/N1
    n1 = 100.0
    thrust_text = thrust_mode

    if thrust_mode == "DERATE":
        floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
        n1 = float(int(math.ceil(max(floor_pct, target_n1_pct))))
        thrust_text = "DERATE" if n1 < 100.0 else "MIL"

    elif thrust_mode == "Auto-Select":
        # Always try MAN@MIL first (no flap change yet)
        asd_mil, agd_aeo_mil = distances_for(100.0)
        ok_reg_mil, _, _, _, _ = field_ok(asd_mil, agd_aeo_mil, engine_out=True)

        if ok_reg_mil:
            # Find minimum N1 that still passes regulatory limits at current flaps
            floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
            lo, hi = floor_pct, 100.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                asd_m, agd_aeo_m = distances_for(mid)
                ok_m, _, _, _, _ = field_ok(asd_m, agd_aeo_m, engine_out=True)
                ok_m = ok_m and compute_oei_second_segment_ok(gw_lbs, mid, flap_deg)
                if ok_m: hi = mid
                else:    lo = mid
            n1 = float(int(math.ceil(hi)))
            thrust_text = "DERATE" if n1 < 100.0 else "MIL"
        else:
            # Only now escalate to FULL+MIL (single step); derate not permitted on FULL
            if flap_mode == "Auto-Select" and flap_deg != 40:
                notes.append("Auto: MAN @ MIL fails §121.189 → escalating to FULL @ MIL.")
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
    ok_reg, req_reg, limiting_reg, asda_eff_lim, tod_limit = field_ok(asd_fin, agd_aeo_fin, engine_out=True)
    avail = max(0.0, tora_ft - shorten_ft)
    agd_reg_fin = agd_aeo_fin * OEI_AGD_FACTOR

    return Result(
        v1=v1, vr=vr, v2=v2, vs=vs,
        flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
        asd_ft=asd_fin, agd_aeo_liftoff_ft=agd_aeo_fin, agd_reg_oei_ft=agd_reg_fin,
        req_ft=req_reg, avail_ft=avail, limiting=limiting_reg,
        hw_kn=hw, cw_kn=cw, notes=notes
    )

# ---------------- UI ----------------
st.title("DCS F-14B Takeoff — FAA-Based Model (NATOPS-aware)")

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

    tora_ft = float(rwy["tora_ft"]); toda_ft = float(rwy["toda_ft"]); asda_ft = float(rwy["asda_ft"])
    elev_ft = float(rwy["threshold_elev_ft"]); hdg = float(rwy["heading_deg"])
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
    sh_unit = st.selectbox("Units", ["ft","NM"], index=0, key="sh_unit")
    shorten_total = float(sh_val) if sh_unit=="ft" else float(sh_val)*6076.12

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    qnh_val = st.number_input("QNH value", value=29.92, step=0.01, format="%.2f")
    qnh_unit = st.selectbox("QNH Units", ["inHg","hPa"], index=0)
    qnh_inhg = float(qnh_val) if qnh_unit=="inHg" else hpa_to_inhg(float(qnh_val))

    wind_units = st.selectbox("Wind Units", ["kts","m/s"], index=0)
    wind_entry = st.text_input("Wind (DIR@SPD)", placeholder=f"{int(hdg):03d}@00")
    parsed = parse_wind_entry(wind_entry, wind_units)
    if parsed is None and (wind_entry or "") != "":
        st.warning("Enter wind as DDD@SS, DDD/SS, or DDD SS. Example: 180@12")
        wind_dir, wind_spd = float(hdg), 0.0
    else:
        wind_dir, wind_spd = (parsed if parsed is not None else (float(hdg), 0.0))
    wind_policy = st.selectbox("Wind Policy", ["None","50/150"], index=0)

    st.header("Weight & Config")
    mode = st.radio("Weight entry", ["Direct GW","Fuel + Stores"], index=0)
    if mode == "Direct GW":
        gw = st.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=500.0)
    else:
        empty_w = st.number_input("Empty weight (lb)", min_value=38000.0, max_value=46000.0, value=41780.0, step=50.0)
        fuel_lb = st.number_input("Internal fuel (lb)", min_value=0.0, max_value=20000.0, value=8000.0, step=100.0)
        ext_tanks = st.selectbox("External tanks (267 gal)", [0,1,2], index=0)
        aim9 = st.slider("AIM-9 count", 0, 2, 0); aim7 = st.slider("AIM-7 count", 0, 4, 0)
        aim54 = st.slider("AIM-54 count", 0, 6, 0); lantirn = st.checkbox("LANTIRN pod")
        wcalc = empty_w + fuel_lb + ext_tanks*1900 + aim9*190 + aim7*510 + aim54*1000 + (440 if lantirn else 0)
        gw = st.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(wcalc), step=10.0)

    flap_mode = st.selectbox("Flaps", ["Auto-Select","UP","MANEUVER","FULL"], index=0)

    st.header("Thrust")
    thrust_mode = st.radio("Mode", ["Auto-Select","Manual Derate","MIL","AB"], index=0)
    derate_n1 = 98.0
    if thrust_mode == "Manual Derate":
        flap_for_floor = 0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20)
        floor = DERATE_FLOOR_BY_FLAP.get(flap_for_floor, 0.90)*100.0
        st.caption(f"Derate floor by flap: {floor:.0f}% N1 (MIL)")
        derate_n1 = st.slider("Target N1 % (MIL)", min_value=float(int(floor)), max_value=100.0,
                              value=float(max(95.0, int(floor))), step=1.0)

    with st.expander("Advanced / Calibration", expanded=False):
        calib = st.radio("Model calibration", ["FAA-conservative","DCS-calibrated"], index=1,
                         help=("FAA: OEI factor 1.20 (conservative). "
                               "DCS: OEI factor 1.15 (tuned to your hot/high tests)."))
        if calib == "DCS-calibrated":
            AEO_CAL = 1.00; OEI_FAC = 1.15
        else:
            AEO_CAL = 1.00; OEI_FAC = 1.20
        globals()["AEO_CAL_FACTOR"] = AEO_CAL
        globals()["OEI_AGD_FACTOR"]  = OEI_FAC

    st.header("Compliance Mode")
    compliance_mode = st.radio("How should limits be checked?",
                               ["Regulatory (OEI §121.189)","AEO Practical"], index=0)

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
        thrust_label = "AFTERBURNER" if res.thrust_text.upper().startswith("AFTERBURNER") else ("MIL" if res.n1_pct >= 100.0 else "DERATE")
        st.metric("Flaps", res.flap_text)
        st.metric("Thrust", f"{thrust_label} ({res.n1_pct:.0f}% N1)")
        flap_deg_out = 0 if res.flap_text.upper().startswith("UP") else (40 if res.flap_text.upper().startswith("FULL") else 20)
        st.metric("Trim (ANU)", f"{trim_anu(float(gw), flap_deg_out):.1f}")
    with c3:
        st.subheader("Runway distances")
        st.metric("Stop distance (ft)", f"{res.asd_ft:.0f}")
        st.metric("Continue distance (engine-out, regulatory) (ft)", f"{res.agd_reg_oei_ft:.0f}")
        st.metric("Continue distance (all engines) (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")

        # Required by mode
        tora_eff = max(0.0, float(tora_ft) - float(shorten_total))
        toda_eff = max(0.0, float(toda_ft) - float(shorten_total))
        asda_eff_lim = max(0.0, float(asda_ft) - float(shorten_total))
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        if compliance_mode.startswith("Regulatory"):
            req = max(res.asd_ft, res.agd_reg_oei_ft)
            asd_ok = res.asd_ft <= asda_eff_lim
            cont_ok = (res.agd_reg_oei_ft <= tod_limit) and (res.agd_reg_oei_ft <= toda_eff)
            limiting = "ASD (stop)" if res.asd_ft >= res.agd_reg_oei_ft else "Engine-out continue"
            asd_margin = asda_eff_lim - res.asd_ft
            cont_margin = min(tod_limit, toda_eff) - res.agd_reg_oei_ft
        else:
            req = max(res.asd_ft, res.agd_aeo_liftoff_ft)
            asd_ok = res.asd_ft <= asda_eff_lim
            cont_ok = (res.agd_aeo_liftoff_ft <= tod_limit) and (res.agd_aeo_liftoff_ft <= toda_eff)
            limiting = "ASD (stop)" if res.asd_ft >= res.agd_aeo_liftoff_ft else "All-engines continue"
            asd_margin = asda_eff_lim - res.asd_ft
            cont_margin = min(tod_limit, toda_eff) - res.agd_aeo_liftoff_ft

        ok = asd_ok and cont_ok
        st.metric("Required runway (based on mode) (ft)", f"{req:.0f}")
    with c4:
        st.subheader("Availability")
        st.metric("Runway available (ft)", f"{res.avail_ft:.0f}")
        st.metric("Limiting factor", limiting)
        st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
        st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
        st.caption("Tailwind > 10 kt or crosswind > 30 kt → NOT AUTHORIZED.")

        req_margin = min(asd_margin, cont_margin)
        if ok:
            st.success(f"COMPLIANT — Margin {req_margin:.0f} ft (ASD margin {asd_margin:.0f}, continue margin {cont_margin:.0f}).")
        else:
            st.error(f"NOT AUTHORIZED — Short by {-req_margin:.0f} ft (ASD margin {asd_margin:.0f}, continue margin {cont_margin:.0f}).")
            st.caption(f"TOD limit: {tod_limit:.0f} ft | ASDA: {asda_eff_lim:.0f} ft | Mode: {compliance_mode}")

    st.markdown("---")
    st.subheader("All-engines takeoff estimates (for DCS comparison)")
    vr_frac = AEO_VR_FRAC_FULL if res.flap_text.upper().startswith("FULL") else AEO_VR_FRAC
    e1,e2 = st.columns(2)
    with e1: st.metric("Vr ground roll (ft)", f"{res.agd_aeo_liftoff_ft * vr_frac:.0f}")
    with e2: st.metric("Liftoff distance to 35 ft (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")
    st.caption("AEO estimates shown for DCS. Regulatory checks assume an engine-out per 14 CFR 121.189.")

    for n in res.notes:
        st.warning(n)
else:
    st.info("Set inputs and click Compute.")
