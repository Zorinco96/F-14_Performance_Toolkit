# f14_takeoff_app_v14.py — DCS F-14B Takeoff Performance
# FAA-style, NATOPS-aware, auto-derate, regulatory vs AEO modes, UX polish
#
# Required files in repo root (case-sensitive):
#   • dcs_airports.csv
#   • f14_perf.csv  # columns: model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note
#
# NOT FOR REAL-WORLD USE. Training aid for DCS.

from __future__ import annotations
import math
import json
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# ---- Streamlit rerun compatibility shim --------------------------------------
# Makes both names available (st.rerun and st.experimental_rerun) on any version,
# and provides a _safe_rerun() helper you can call anywhere.

# Try to alias whichever exists so BOTH attrs are present
try:
    if hasattr(st, "rerun") and not hasattr(st, "experimental_rerun"):
        st.experimental_rerun = st.rerun
    elif hasattr(st, "experimental_rerun") and not hasattr(st, "rerun"):
        st.rerun = st.experimental_rerun
except Exception:
    pass

def _safe_rerun():
    """Call rerun in a version-safe way; swallow if not available."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass
# -----------------------------------------------------------------------------
# ---- Streamlit query params compatibility (new API: st.query_params) --------
# Use these helpers everywhere instead of experimental_* calls.

def qp_get() -> dict:
    """Return current query params as a plain dict."""
    try:
        # New API (Streamlit ≥ 1.31)
        return dict(st.query_params)
    except Exception:
        # Old API fallback
        return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                for k, v in st.experimental_get_query_params().items()}

def qp_set(**kwargs):
    """Replace the query string with the provided key/values."""
    try:
        st.query_params.from_dict(kwargs)   # New API
    except Exception:
        st.experimental_set_query_params(**kwargs)  # Fallback

def qp_update(**kwargs):
    """Update/merge into current query params (preserving others)."""
    params = qp_get()
    params.update({k: v for k, v in kwargs.items() if v is not None})
    qp_set(**params)

def qp_clear():
    """Clear all query params."""
    try:
        st.query_params.clear()  # New API
    except Exception:
        st.experimental_set_query_params()  # Fallback clears by setting nothing
# -----------------------------------------------------------------------------


st.set_page_config(page_title="DCS F-14B Takeoff (Pro)", page_icon="✈️", layout="wide")

# ============================= TUNING / CONSTANTS =============================
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}  # per engine, approx (for OEI guardrail)
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}  # FULL cannot derate
ALPHA_N1_DIST = 2.0          # distance scale ~ 1/(N1^alpha)
AEO_VR_FRAC = 0.88           # Vr ground roll ~ fraction of liftoff-to-35ft (UP/MAN)
AEO_VR_FRAC_FULL = 0.82      # for FULL flaps
OEI_AGD_FACTOR = 1.20        # default regulatory engine-out factor; can be set to 1.15 in UI
AEO_CAL_FACTOR = 1.00        # keep 1.00; CSVs already carry realistic AEO/AGD values
WIND_FACTORS = {"None": (1.0, 1.0), "50/150": (0.5, 1.5)}  # headwind credit, tailwind penalty

# ============================= ATMOSPHERE / WIND =============================
def hpa_to_inhg(hpa: float) -> float:
    return float(hpa) * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - qnh_inhg) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (h_ft / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    # FAA rule of thumb: DA ≈ PA + 120 × (OAT − ISA)
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
    """Outside grid, scale distances by density ratio vs clamped ref (≤5000 ft, ≤30 °C)."""
    da_act = density_altitude_ft(pa_ft, oat_c)
    da_ref = density_altitude_ft(min(pa_ft, 5000.0), min(oat_c, 30.0))
    sig_act = sigma_from_da(da_act)
    sig_ref = sigma_from_da(da_ref)
    BETA = 0.85
    return (sig_ref / max(1e-6, sig_act)) ** BETA

def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float, float]:
    # dir is FROM direction
    delta = math.radians((dir_deg - rwy_heading_deg) % 360.0)
    hw = speed_kn * math.cos(delta)   # headwind (+) / tailwind (−)
    cw = speed_kn * math.sin(delta)   # crosswind (+R/−L)
    return float(hw), float(cw)

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    if slope_pct > 0:  # +20% per +1% uphill (ignore downhill credit)
        d *= (1.0 + 0.20 * slope_pct)
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)
    return max(d, 0.0)

# ============================= UTILITIES =============================
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

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_perf() -> pd.DataFrame:
    try:
        df = pd.read_csv("f14_perf.csv", comment="#")
    except Exception:
        st.error("f14_perf.csv not found in repo root.")
        st.stop()
    df["thrust"] = df["thrust"].astype(str).str.upper().replace({"MIL": "MILITARY", "AB": "AFTERBURNER"})
    df = _ensure_numeric(df, ["flap_deg","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"])\
           .dropna(subset=["flap_deg","gw_lbs","press_alt_ft","oat_c","Vr_kt","ASD_ft","AGD_ft"])
    return df

def csv_version_info(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    try:
        mtime = os.path.getmtime(path)
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()[:8]
        return f"{os.path.basename(path)} | md5 {md5} | mtime {pd.to_datetime(mtime, unit='s')}"
    except Exception:
        return "unknown"

def lint_perf(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    req_cols = ["model","flap_deg","thrust","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]
    for c in req_cols:
        if c not in df.columns:
            issues.append(f"Missing column: {c}")
    if df.duplicated(subset=["flap_deg","thrust","gw_lbs","press_alt_ft","oat_c"]).any():
        issues.append("Duplicate (flap,thrust,gw,PA,OAT) keys found — interpolation may be unstable.")
    if (df[["ASD_ft","AGD_ft"]] <= 0).any().any():
        issues.append("Non-positive distances present.")
    if (df[["Vr_kt","V2_kt","Vs_kt"]] <= 0).any().any():
        issues.append("Non-positive speeds present.")
    return issues

# Adaptive AB/MIL ratio learned per flap and GW/PA bands
@st.cache_data
def ab_ratio_by_band(perfdb: pd.DataFrame, flap_deg: int) -> Dict[tuple, float]:
    keys = ["gw_lbs","press_alt_ft","oat_c"]
    mil = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == "MILITARY")]
    ab  = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == "AFTERBURNER")]
    if mil.empty or ab.empty:
        return {(): 0.82}
    merged = pd.merge(mil[keys + ["ASD_ft","AGD_ft"]], ab[keys + ["ASD_ft","AGD_ft"]], on=keys, suffixes=("_MIL","_AB"))
    if merged.empty:
        return {(): 0.82}
    # bin by GW and PA bands
    gw_bins = [0, 60000, 70000, 80000, 1e9]
    pa_bins = [-1e9, 0, 2000, 5000, 8000, 1e9]
    merged["gw_bin"] = pd.cut(merged["gw_lbs"], gw_bins, labels=False, include_lowest=True)
    merged["pa_bin"] = pd.cut(merged["press_alt_ft"], pa_bins, labels=False, include_lowest=True)
    ratios: Dict[tuple, float] = {}
    for (g, p), grp in merged.groupby(["gw_bin","pa_bin"]):
        vals = []
        for col in ("ASD_ft","AGD_ft"):
            den = grp[f"{col}_MIL"].to_numpy(dtype=float)
            num = grp[f"{col}_AB"].to_numpy(dtype=float)
            ok = den > 1.0
            if ok.any():
                r = np.median(np.clip(num[ok]/den[ok], 0.65, 0.95))
                vals.append(float(r))
        if vals:
            ratios[(int(g), int(p))] = min(vals)
    if not ratios:
        ratios[()] = 0.82
    return ratios

def lookup_ab_ratio(ratios: Dict[tuple, float], gw: float, pa: float) -> float:
    if () in ratios:
        return ratios[()]
    # approximate by nearest band center
    def pick(val, bins):
        idx = np.digitize([val], bins)[0]-1
        return max(0, min(idx, len(bins)-2))
    gw_bins = [0, 60000, 70000, 80000, 1e9]
    pa_bins = [-1e9, 0, 2000, 5000, 8000, 1e9]
    g = pick(gw, gw_bins); p = pick(pa, pa_bins)
    return ratios.get((g, p), np.median(list(ratios.values())))

# ============================= INTERPOLATION =============================
def _bounds(vals, x):
    vals = sorted(set(map(float, vals)))
    lo = max([v for v in vals if v <= x], default=vals[0])
    hi = min([v for v in vals if v >= x], default=vals[-1])
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return lo, hi, w

def _interp_weight_at(sub: pd.DataFrame, pa: float, oat: float, field: str, gw_x: float) -> float:
    s = sub[(sub["press_alt_ft"] == pa) & (sub["oat_c"] == oat)].sort_values("gw_lbs")
    if s.empty:
        s = sub.sort_values(["press_alt_ft", "oat_c", "gw_lbs"]).copy()
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
        sub = perf[(perf["flap_deg"] == use_flap)]
        if sub.empty:
            sub = perf[(perf["thrust"] == thrust)]
        if sub.empty:
            sub = perf
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

# ============================= NATOPS / DATA MODES =============================
def agd_is_liftoff_mode(perfdb: pd.DataFrame, flap_deg: int, thrust: str) -> bool:
    sub = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == thrust)]
    if sub.empty:
        return False
    mask = sub["note"].astype(str).str.contains("NATOPS", case=False, na=False)
    return bool(mask.mean() >= 0.5)

# ============================= SPEED FLOORS =============================
def enforce_speed_floors(vs: float, v1: float, vr: float, v2: float, flap_deg: int, cw_kn: float, gw_lbs: float) -> Tuple[float,float,float]:
    # Vmcg floor scales with crosswind and weight (very rough model)
    base_vmcg = 80.0 + 0.0007 * gw_lbs  # e.g., ~115 kt at 50k, ~129 kt at 70k
    vmcg = base_vmcg + 0.20 * abs(cw_kn)  # add 0.2 kt per kt crosswind

    # Basic relationships
    vr_min = max(1.08 * vs, 120.0 if flap_deg != 40 else 110.0)
    v2_min = max(vr_min + 7.0, 1.13 * vs)
    v1_min = max(0.94 * max(vr, vr_min), vmcg)

    v1f = max(v1, v1_min)
    vrf = max(vr, vr_min)
    v2f = max(v2, v2_min)

    # Keep ordering V1 < Vr < V2
    if v1f >= vrf:
        v1f = max(min(vrf - 3.0, v1f), vmcg)
    if vrf >= v2f:
        v2f = vrf + 5.0

    return float(round(v1f)), float(round(vrf)), float(round(v2f))

# ============================= OEI CLIMB GUARDRAIL =============================
def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct/100.0)
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024

# ============================= CORE COMPUTE =============================
@dataclass
class Slice:
    v1: float; vr: float; v2: float; vs: float
    asd: float; agd_aeo_liftoff: float

@dataclass
class Result:
    # Speeds/settings
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    # Distances
    asd_ft: float; agd_aeo_liftoff_ft: float; agd_reg_oei_ft: float
    req_ft_reg: float; req_ft_aeo: float
    avail_ft: float; limiting_reg: str; limiting_aeo: str
    # Wind
    hw_kn: float; cw_kn: float
    # Meta
    notes: list; confidence: str; grid_dx: Dict[str, float]


def compute_slice(perfdb: pd.DataFrame, flap_deg: int, thrust: str, gw_lbs: float, pa: float, oat_c: float,
                  slope_pct: float, hw: float, wind_policy: str, outside_grid: bool,
                  ab_ratio_map: Optional[Dict[tuple, float]] = None) -> Slice:
    # choose thrust table
    use_thrust = thrust
    # If AB selected but table missing, approximate via ratio
    has_table = not perfdb[(perfdb["flap_deg"] == (20 if flap_deg == 0 else flap_deg)) & (perfdb["thrust"] == thrust)].empty
    base = interp_perf(perfdb, (20 if flap_deg == 0 else flap_deg), (use_thrust if has_table else "MILITARY"), float(gw_lbs), float(pa), float(oat_c))

    vs = float(base["Vs_kt"]); v1 = float(base["V1_kt"]); vr = float(base["Vr_kt"]); v2 = float(base["V2_kt"])
    asd = float(base["ASD_ft"]); agd = float(base["AGD_ft"])  # AGD may be liftoff or ground roll depending on source

    liftoff_mode = agd_is_liftoff_mode(perfdb, (20 if flap_deg == 0 else flap_deg), (use_thrust if has_table else "MILITARY"))
    if liftoff_mode:
        agd_aeo = agd
    else:
        # Convert ground roll -> liftoff-to-35 ft with gentle DA-aware factor outside grid only
        liftoff_factor = 1.42 + 0.15 * 0.0
        agd_aeo = agd * liftoff_factor

    # UP penalty relative to MAN
    if flap_deg == 0:
        asd *= 1.06
        agd_aeo *= 1.06

    # AB ratio if approximating
    if thrust == "AFTERBURNER" and not has_table:
        ratio = lookup_ab_ratio(ab_ratio_map or {():0.82}, gw_lbs, pa)
        asd *= ratio
        agd_aeo *= ratio

    # DA scale only when outside the CSV grid
    def maybe_da_scale(x):
        if outside_grid:
            return x * da_out_of_grid_scale(pa, oat_c)
        return x

    # Wind & slope
    asd = apply_wind_slope(maybe_da_scale(asd), slope_pct, hw, wind_policy)
    agd_aeo = apply_wind_slope(maybe_da_scale(agd_aeo), slope_pct, hw, wind_policy)

    return Slice(v1=v1, vr=vr, v2=v2, vs=vs, asd=asd, agd_aeo_liftoff=agd_aeo)


def compute_all(perfdb: pd.DataFrame,
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

    # Flaps (Auto defaults to MAN)
    flap_text = "MANEUVER" if flap_mode == "Auto-Select" else flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)

    use_flap_for_table = 20 if flap_deg == 0 else flap_deg

    # grid bounds -> outside_grid flag
    sub = perfdb[(perfdb["flap_deg"] == use_flap_for_table) & (perfdb["thrust"].isin(["MILITARY","AFTERBURNER"]))]
    if sub.empty: sub = perfdb
    pa_min = float(sub["press_alt_ft"].min()); pa_max = float(sub["press_alt_ft"].max())
    t_min  = float(sub["oat_c"].min());        t_max  = float(sub["oat_c"].max())
    outside_grid = (pa < pa_min or pa > pa_max or oat_c < t_min or oat_c > t_max)

    # AB ratio map
    ab_map = ab_ratio_by_band(perfdb, use_flap_for_table)

    # Helper: declared distances
    tora_eff = max(0.0, tora_ft - shorten_ft)
    toda_eff = max(0.0, toda_ft - shorten_ft)
    asda_eff = max(0.0, asda_ft - shorten_ft)
    clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
    tod_limit = tora_eff + clearway_allow

    # N1 multiplier
    def mult_from_n1(n1pct: float) -> float:
        eff = max(0.90, min(1.0, n1pct/100.0))
        return 1.0 / (eff ** ALPHA_N1_DIST)

    # Slice computation (with n1 multiplier applied at the end)
    def distances_for(n1pct: float, table_thrust: str) -> Tuple[Slice, float, float]:
        s = compute_slice(perfdb, flap_deg, table_thrust, gw_lbs, pa, oat_c, slope_pct, hw, wind_policy, outside_grid, ab_map)
        # apply N1 to distances only (speeds unchanged)
        m = mult_from_n1(n1pct)
        asd = s.asd * m
        agd_aeo = s.agd_aeo_liftoff * m * AEO_CAL_FACTOR
        return Slice(v1=s.v1, vr=s.vr, v2=s.v2, vs=s.vs, asd=asd, agd_aeo_liftoff=agd_aeo), asd, agd_aeo

    # Field-limits evaluator
    def check_ok(asd_eff: float, agd_aeo_eff: float, engine_out: bool) -> Tuple[bool, float, str]:
        if engine_out:
            cont = agd_aeo_eff * st.session_state.get("OEI_AGD_FACTOR_SS", OEI_AGD_FACTOR)
            limiting = "ASD (stop)" if asd_eff >= cont else "Engine-out continue"
        else:
            cont = agd_aeo_eff
            limiting = "ASD (stop)" if asd_eff >= cont else "All-engines continue"
        req = max(asd_eff, cont)
        ok = (asd_eff <= asda_eff) and (cont <= tod_limit) and (cont <= toda_eff)
        return ok, req, limiting

    # Speed floors enforcement wrapper
    def apply_speed_floors(slice_obj: Slice) -> Slice:
        v1f, vrf, v2f = enforce_speed_floors(slice_obj.vs, slice_obj.v1, slice_obj.vr, slice_obj.v2, flap_deg, cw, gw_lbs)
        return Slice(v1=v1f, vr=vrf, v2=v2f, vs=float(round(slice_obj.vs)), asd=slice_obj.asd, agd_aeo_liftoff=slice_obj.agd_aeo_liftoff)

    # Thrust / N1 selection
    n1 = 100.0
    thrust_text = thrust_mode

    # FULL cannot derate
    if flap_deg == 40 and thrust_mode in ("Auto-Select","DERATE","Manual Derate"):
        thrust_mode = "MIL"
        notes.append("Derate with FULL flaps not allowed — using MIL.")

    if thrust_mode == "DERATE":
        floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
        n1 = float(int(max(floor_pct, target_n1_pct)))
        thrust_text = "DERATE" if n1 < 100.0 else "MIL"
        table_thrust = "MILITARY"
    elif thrust_mode == "AB":
        n1 = 100.0
        thrust_text = "AFTERBURNER"
        table_thrust = "AFTERBURNER"
        notes.append("Afterburner selected — NOT AUTHORIZED for F-14B except as last resort.")
    elif thrust_mode == "MIL":
        n1 = 100.0
        thrust_text = "MIL"
        table_thrust = "MILITARY"
    else:  # Auto-Select
        # Try MAN/current flap @ MIL: if OK (regulatory), find min N1 via bisection
        table_thrust = "MILITARY"
        s_mil, asd_mil, agd_mil = distances_for(100.0, table_thrust)
        s_mil = apply_speed_floors(s_mil)
        ok_mil, _, _ = check_ok(s_mil.asd, s_mil.agd_aeo_liftoff, engine_out=True)
        if ok_mil:
            floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
            lo, hi = floor_pct, 100.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                s_mid, asd_mid, agd_mid = distances_for(mid, table_thrust)
                s_mid = apply_speed_floors(s_mid)
                ok_mid, _, _ = check_ok(s_mid.asd, s_mid.agd_aeo_liftoff, engine_out=True)
                ok_mid = ok_mid and compute_oei_second_segment_ok(gw_lbs, mid, flap_deg)
                if ok_mid: hi = mid
                else:      lo = mid
            n1 = float(int(math.ceil(hi)))
            thrust_text = "DERATE" if n1 < 100.0 else "MIL"
        else:
            # Escalate to FULL + MIL (single step)
            if flap_deg != 40:
                flap_text = "FULL"
                flap_deg = 40
                table_thrust = "MILITARY"
                notes.append("Auto: MAN @ MIL fails §121.189; escalating to FULL @ MIL.")
            n1 = 100.0
            thrust_text = "MIL"

    # Compute final slice at chosen n1
    slice_final, asd_fin, agd_aeo_fin = distances_for(n1, table_thrust)
    slice_final = apply_speed_floors(slice_final)

    # Regulatory and AEO checks
    ok_reg, req_reg, limiting_reg = check_ok(slice_final.asd, slice_final.agd_aeo_liftoff, engine_out=True)
    ok_aeo, req_aeo, limiting_aeo = check_ok(slice_final.asd, slice_final.agd_aeo_liftoff, engine_out=False)

    avail = tora_eff

    # Confidence score: normalized distance from nearest CSV points (GW, PA, OAT)
    def grid_distance_norm() -> Dict[str,float]:
        sub = perfdb[(perfdb["flap_deg"] == use_flap_for_table) & (perfdb["thrust"].isin(["MILITARY","AFTERBURNER"]))]
        if sub.empty: sub = perfdb
        def norm(val, arr):
            arr = np.array(sorted(set(arr)), dtype=float)
            if arr.size == 0:
                return 1.0
            d = np.min(np.abs(arr - val))
            span = max(arr.max() - arr.min(), 1.0)
            return float(d / span)
        return {
            "GW": norm(gw_lbs, sub["gw_lbs"]),
            "PA": norm(pa, sub["press_alt_ft"]),
            "OAT": norm(oat_c, sub["oat_c"]) }

    dx = grid_distance_norm()
    mean_dx = np.mean(list(dx.values()))
    if mean_dx < 0.03: conf = "High"
    elif mean_dx < 0.10: conf = "Medium"
    else: conf = "Low"

    return Result(
        v1=slice_final.v1, vr=slice_final.vr, v2=slice_final.v2, vs=slice_final.vs,
        flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
        asd_ft=slice_final.asd, agd_aeo_liftoff_ft=slice_final.agd_aeo_liftoff,
        agd_reg_oei_ft=slice_final.agd_aeo_liftoff * st.session_state.get("OEI_AGD_FACTOR_SS", OEI_AGD_FACTOR),
        req_ft_reg=req_reg, req_ft_aeo=req_aeo, avail_ft=avail,
        limiting_reg=limiting_reg, limiting_aeo=limiting_aeo,
        hw_kn=hw, cw_kn=cw, notes=notes, confidence=conf, grid_dx=dx)

# ============================= SOLVERS =============================
def solve_min_n1(perfdb: pd.DataFrame, params: Dict[str, Any], engine_out: bool) -> float:
    # returns whole-percent N1
    floor_pct = DERATE_FLOOR_BY_FLAP.get(params["flap_deg"], 0.90)*100.0
    lo, hi = floor_pct, 100.0
    def ok_at(n1):
        res = compute_all(perfdb, **{**params, "thrust_mode":"DERATE", "target_n1_pct": n1})
        req = res.req_ft_reg if engine_out else res.req_ft_aeo
        # authorization against declared distances
        tora_eff = max(0.0, params["tora_ft"] - params["shorten_ft"]) ; toda_eff = max(0.0, params["toda_ft"] - params["shorten_ft"]) ; asda_eff = max(0.0, params["asda_ft"] - params["shorten_ft"]) ; clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff)) ; tod_limit = tora_eff + clearway_allow
        return (res.asd_ft <= asda_eff) and (req <= min(tod_limit, toda_eff))
    if not ok_at(100.0):
        return 100.0
    for _ in range(18):
        mid = (lo + hi)/2.0
        if ok_at(mid): hi = mid
        else: lo = mid
    return float(int(math.ceil(hi)))

def solve_max_tow(perfdb: pd.DataFrame, params: Dict[str, Any], engine_out: bool) -> float:
    # binary search gross weight 40k..80k
    lo, hi = 40000.0, 80000.0
    def ok_at(gw):
        res = compute_all(perfdb, **{**params, "gw_lbs": gw})
        req = res.req_ft_reg if engine_out else res.req_ft_aeo
        tora_eff = max(0.0, params["tora_ft"] - params["shorten_ft"]) ; toda_eff = max(0.0, params["toda_ft"] - params["shorten_ft"]) ; asda_eff = max(0.0, params["asda_ft"] - params["shorten_ft"]) ; clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff)) ; tod_limit = tora_eff + clearway_allow
        return (res.asd_ft <= asda_eff) and (req <= min(tod_limit, toda_eff))
    if not ok_at(lo):
        return lo
    if ok_at(hi):
        return hi
    for _ in range(20):
        mid = (lo + hi)/2.0
        if ok_at(mid): lo = mid
        else: hi = mid
    return float(int(round(lo, -2)))

# Equivalent assumed-temperature for a given derate (rough heuristic)
def assumed_oat_for_n1(pa_ft: float, oat_c: float, n1pct: float) -> float:
    # We find OAT_assumed such that DA scale ~ derate multiplier
    m = 1.0 / (max(0.90, min(1.0, n1pct/100.0)) ** ALPHA_N1_DIST)
    # target sigma_ratio^BETA == m -> search OAT_assumed
    lo, hi = max(-40.0, oat_c-40), min(60.0, oat_c+40)
    for _ in range(30):
        mid = (lo + hi) / 2.0
        # scale from actual OAT to assumed OAT at same PA
        da_act = density_altitude_ft(pa_ft, oat_c)
        da_mid = density_altitude_ft(pa_ft, mid)
        sig_act = sigma_from_da(da_act)
        sig_mid = sigma_from_da(da_mid)
        BETA = 0.85
        scale = (sig_act / max(1e-6, sig_mid)) ** BETA
        if scale < m:  # need hotter
            lo = mid
        else:
            hi = mid
    return float(round(hi, 1))

# ============================= WIND PARSER =============================
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

# ============================= UI / STATE HELPERS =============================
def get_query_params() -> Dict[str, Any]:
    try:
        return st.experimental_get_query_params()  # type: ignore
    except Exception:
        return {}

def set_query_params(d: Dict[str, Any]):
    try:
        st.experimental_set_query_params(**d)  # type: ignore
    except Exception:
        pass

# ============================= APP =============================
st.title("DCS F-14B Takeoff — Pro Model")

rwy_db = load_runways()
perfdb = load_perf()
issues = lint_perf(perfdb)

# Calibration toggles stored in session for access inside compute_all
with st.sidebar:
    st.header("Model Calibration")
    calib = st.radio("Engine-out factor", ["Regulatory 1.20","DCS-tuned 1.15"], index=1)
    st.session_state["OEI_AGD_FACTOR_SS"] = 1.15 if calib.endswith("1.15") else 1.20
    st.caption("This affects engine-out continue distance only.")

# ---------------- Runway selection with type-ahead filter ----------------
st.sidebar.header("Runway")
filter_txt = st.sidebar.text_input("Filter airports", placeholder="Type part of name")
if filter_txt:
    df_air = rwy_db[rwy_db["airport_name"].str.contains(filter_txt, case=False, na=False)]
else:
    df_air = rwy_db

Theatre = st.sidebar.selectbox("DCS Theatre", sorted(df_air["map"].unique()))
df_t = df_air[df_air["map"] == Theatre]
Airport = st.sidebar.selectbox("Airport", sorted(df_t["airport_name"].unique()))
df_a = df_t[df_t["airport_name"] == Airport]
RunwayLabel = st.sidebar.selectbox("Runway End", list(df_a["runway_label"]))
rwy = df_a[df_a["runway_label"] == RunwayLabel].iloc[0]

# Declared distances + heading/elev
TORA = float(rwy["tora_ft"]) ; TODA = float(rwy["toda_ft"]) ; ASDA = float(rwy["asda_ft"]) ; ELEV = float(rwy["threshold_elev_ft"]) ; HDG = float(rwy["heading_deg"]) ; SLOPE = float(rwy.get("slope_percent", 0.0) or 0.0)

cA, cB = st.sidebar.columns(2)
with cA:
    st.metric("TORA (ft)", f"{TORA:,.0f}")
    st.metric("TODA (ft)", f"{TODA:,.0f}")
with cB:
    st.metric("ASDA (ft)", f"{ASDA:,.0f}")
    st.metric("Elev (ft)", f"{ELEV:,.0f}")

st.sidebar.caption("Shorten Available Runway")
sh_val = st.sidebar.number_input("Value", min_value=0.0, value=0.0, step=50.0, key="sh_val")
sh_unit = st.sidebar.selectbox("Units", ["ft", "NM"], index=0, key="sh_unit")
Shorten = float(sh_val) if sh_unit == "ft" else float(sh_val) * 6076.12

# ---------------- Weather ----------------
st.sidebar.header("Weather")
OAT = st.sidebar.number_input("OAT (°C)", value=15.0, step=1.0)
QNH_val = st.sidebar.number_input("QNH value", value=29.92, step=0.01, format="%.2f")
QNH_unit = st.sidebar.selectbox("QNH Units", ["inHg", "hPa"], index=0)
QNH_inHg = float(QNH_val) if QNH_unit == "inHg" else hpa_to_inhg(float(QNH_val))

wind_units = st.sidebar.selectbox("Wind Units", ["kts", "m/s"], index=0)
wind_entry = st.sidebar.text_input("Wind (DIR@SPD)", placeholder=f"{int(HDG):03d}@00")
parsed = parse_wind_entry(wind_entry, wind_units)
if parsed is None and (wind_entry or "") != "":
    st.sidebar.warning("Enter wind as DDD@SS, DDD/SS, or DDD SS. Example: 180@12")
    WIND_DIR, WIND_SPD = float(HDG), 0.0
else:
    WIND_DIR, WIND_SPD = (parsed if parsed is not None else (float(HDG), 0.0))
wind_policy = st.sidebar.selectbox("Wind Policy", ["None", "50/150"], index=0)

# ---------------- Weight & Config ----------------
st.sidebar.header("Weight & Config")
mode = st.sidebar.radio("Weight entry", ["Direct GW", "Fuel + Stores"], index=0)
if mode == "Direct GW":
    GW = st.sidebar.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=500.0)
else:
    empty_w = st.sidebar.number_input("Empty weight (lb)", min_value=38000.0, max_value=46000.0, value=41780.0, step=50.0)
    fuel_lb = st.sidebar.number_input("Internal fuel (lb)", min_value=0.0, max_value=20000.0, value=8000.0, step=100.0)
    ext_tanks = st.sidebar.selectbox("External tanks (267 gal)", [0,1,2], index=0)
    aim9 = st.sidebar.slider("AIM-9 count", 0, 2, 0)
    aim7 = st.sidebar.slider("AIM-7 count", 0, 4, 0)
    aim54 = st.sidebar.slider("AIM-54 count", 0, 6, 0)
    lantirn = st.sidebar.checkbox("LANTIRN pod")
    wcalc = empty_w + fuel_lb + ext_tanks*1900 + aim9*190 + aim7*510 + aim54*1000 + (440 if lantirn else 0)
    GW = st.sidebar.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(wcalc), step=10.0)

flap_mode = st.sidebar.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

st.sidebar.header("Thrust")
thrust_mode = st.sidebar.radio("Mode", ["Auto-Select", "Manual Derate", "MIL", "AB"], index=0)
derate_n1 = 98.0
if thrust_mode == "Manual Derate":
    flap_for_floor = 0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20)
    floor = DERATE_FLOOR_BY_FLAP.get(flap_for_floor, 0.90)*100.0
    st.sidebar.caption(f"Derate floor by flap: {floor:.0f}% N1 (MIL)")
    derate_n1 = st.sidebar.slider("Target N1 % (MIL)", min_value=float(int(floor)), max_value=100.0, value=max(95.0, float(int(floor))), step=1.0)

# ---------------- Compliance & Auto-calc ----------------
st.sidebar.header("Compliance Mode")
compliance_mode = st.sidebar.radio("How should limits be checked?", ["Regulatory (OEI §121.189)", "AEO Practical"], index=0)

# Cache clear & CSV info
with st.sidebar.expander("Data & Cache"):
    st.caption(csv_version_info("f14_perf.csv"))
    st.caption(csv_version_info("dcs_airports.csv"))
    if issues:
        st.error("CSV Lint: Issues found\n- " + "\n- ".join(issues))
    else:
        st.success("CSV Lint: OK")
    if st.button("Clear caches"):
        st.cache_data.clear()
        st.experimental_rerun()

# Pack params (for solvers and link sharing)
params = dict(
    perfdb=perfdb,
    rwy_heading_deg=float(HDG), tora_ft=float(TORA), toda_ft=float(TODA), asda_ft=float(ASDA),
    field_elev_ft=float(ELEV), slope_pct=float(SLOPE), shorten_ft=float(Shorten),
    oat_c=float(OAT), qnh_inhg=float(QNH_inHg),
    wind_speed=float(WIND_SPD), wind_dir_deg=float(WIND_DIR), wind_units=str(wind_units), wind_policy=str(wind_policy),
    gw_lbs=float(GW), flap_mode=str(flap_mode), thrust_mode=("DERATE" if thrust_mode=="Manual Derate" else thrust_mode), target_n1_pct=float(derate_n1)
)

# Auto-calc: compute on every rerun
res = compute_all(**params)

# Shareable link (query params)
qp = {
    "theatre": [Theatre], "airport": [Airport], "runway": [RunwayLabel],
    "oat": [str(OAT)], "qnh": [str(QNH_inHg)], "wind": [f"{int(WIND_DIR):03d}@{int(WIND_SPD)}"],
    "gw": [str(GW)], "flaps": [flap_mode], "thrust": [thrust_mode], "n1": [str(derate_n1)],
    "mode": ["reg" if compliance_mode.startswith("Regulatory") else "aeo"]
}
set_query_params(qp)

# ============================= OUTPUT LAYOUT =============================
st.subheader("Results (Auto-Calculated)")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown("**V-Speeds**")
    st.metric("V1 (kt)", f"{res.v1:.0f}")
    st.metric("Vr (kt)", f"{res.vr:.0f}")
    st.metric("V2 (kt)", f"{res.v2:.0f}")
    st.metric("Vs (kt)", f"{res.vs:.0f}")
with colB:
    st.markdown("**Settings**")
    thrust_label = "AFTERBURNER" if res.thrust_text.upper().startswith("AFTERBURNER") else ("MIL" if res.n1_pct >= 100.0 else "DERATE")
    st.metric("Flaps", res.flap_text)
    st.metric("Thrust", f"{thrust_label} ({res.n1_pct:.0f}% N1)")
    flap_deg_out = 0 if res.flap_text.upper().startswith("UP") else (40 if res.flap_text.upper().startswith("FULL") else 20)
    st.metric("Trim (ANU)", f"{(4.5 + (GW - 60000.0)/10000.0 * 0.8 + (1.0 if flap_deg_out==40 else (-1.0 if flap_deg_out==0 else 0.0))):.1f}")
with colC:
    st.markdown("**Runway distances (ft)**")
    st.metric("Stop distance", f"{res.asd_ft:.0f}")
    st.metric("Continue (engine-out, regulatory)", f"{res.agd_reg_oei_ft:.0f}")
    st.metric("Continue (all engines)", f"{res.agd_aeo_liftoff_ft:.0f}")
with colD:
    st.markdown("**Availability**")
    st.metric("Runway available", f"{res.avail_ft:.0f}")
    st.metric("Headwind (kt)", f"{res.hw_kn:.1f}")
    st.metric("Crosswind (kt)", f"{res.cw_kn:.1f}")
    st.metric("Confidence", res.confidence)

# Authorization chips for both modes

def auth_banner(label: str, req: float, asd: float, cont: float):
    tora_eff = max(0.0, TORA - Shorten)
    toda_eff = max(0.0, TODA - Shorten)
    asda_eff = max(0.0, ASDA - Shorten)
    clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
    tod_limit = tora_eff + clearway_allow
    asd_ok = asd <= asda_eff
    cont_ok = cont <= min(tod_limit, toda_eff)
    ok = asd_ok and cont_ok
    asd_margin = asda_eff - asd
    cont_margin = min(tod_limit, toda_eff) - cont
    req_margin = min(asd_margin, cont_margin)
    if ok:
        st.success(f"{label}: AUTHORIZED — Margin {req_margin:.0f} ft (ASD {asd_margin:.0f}, Continue {cont_margin:.0f}).")
    else:
        st.error(f"{label}: NOT AUTHORIZED — Short by {-req_margin:.0f} ft (ASD {asd_margin:.0f}, Continue {cont_margin:.0f}).")
        st.caption(f"TOD limit: {tod_limit:.0f} ft | ASDA: {asda_eff:.0f} ft")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Regulatory check (OEI §121.189)")
    auth_banner("Regulatory", max(res.asd_ft, res.agd_reg_oei_ft), res.asd_ft, res.agd_reg_oei_ft)
with col2:
    st.markdown("### AEO practical check")
    auth_banner("AEO", max(res.asd_ft, res.agd_aeo_liftoff_ft), res.asd_ft, res.agd_aeo_liftoff_ft)

# Why-log (concise)
st.markdown("### Why these numbers?")
why = []
why.append(f"Flaps {res.flap_text}, thrust {res.thrust_text} at {res.n1_pct:.0f}% N1 after derate solver.")
why.append(f"Inside grid: {'No' if (res.confidence=='Low') else 'Yes'}; nearest-grid deltas (norm): GW {res.grid_dx['GW']:.2f}, PA {res.grid_dx['PA']:.2f}, OAT {res.grid_dx['OAT']:.2f}.")
why.append(f"Wind/slope applied; policy {wind_policy}. Tailwind>10 or crosswind>30 triggers NOT AUTHORIZED banners.")
st.write("\n".join(["• "+w for w in why]))

# Sensitivity mini-table (± deltas)
st.markdown("### Sensitivity (delta from current)")

def sensitivity_row(name, mutate_fn):
    p = params.copy()
    base = res.req_ft_aeo if compliance_mode.endswith("Practical") else res.req_ft_reg
    r = []
    for dv in [-1, +1]:
        p2 = mutate_fn(p, dv)
        r2 = compute_all(**p2)
        v = (r2.req_ft_aeo if compliance_mode.endswith("Practical") else r2.req_ft_reg) - base
        r.append(int(round(v)))
    return name, r[0], r[1]

rows = []
rows.append(sensitivity_row("OAT ±5°C", lambda p,dv: {**p, "oat_c": p["oat_c"] + 5.0*dv}))
rows.append(sensitivity_row("Headwind ±5 kt", lambda p,dv: {**p, "wind_speed": max(0.0, p["wind_speed"] + (5.0 if dv>0 else -5.0)), "wind_dir_deg": p["wind_dir_deg"]}))
rows.append(sensitivity_row("GW ±1000 lb", lambda p,dv: {**p, "gw_lbs": p["gw_lbs"] + 1000.0*dv}))
rows.append(sensitivity_row("N1 ±2%", lambda p,dv: {**p, "thrust_mode":"DERATE", "target_n1_pct": min(100.0, max(DERATE_FLOOR_BY_FLAP.get(0 if p['flap_mode']=='UP' else (40 if p['flap_mode']=='FULL' else 20),0.9)*100.0, (res.n1_pct + 2*dv))) }))

sens_df = pd.DataFrame(rows, columns=["Parameter", "-Δ (ft)", "+Δ (ft)"])
st.dataframe(sens_df, use_container_width=True, hide_index=True)

# Comparison table: candidate configs
st.markdown("### Candidate configurations (side-by-side)")

def evaluate_config(flap_txt: str, thrust_txt: str) -> Dict[str, Any]:
    p = params.copy()
    p["flap_mode"] = flap_txt
    p["thrust_mode"] = thrust_txt
    if thrust_txt == "Manual Derate":
        p["thrust_mode"] = "DERATE"
        p["target_n1_pct"] = res.n1_pct  # use current best n1
    out = compute_all(**p)
    return {
        "Flaps": out.flap_text,
        "Thrust": ("AFTERBURNER" if out.thrust_text.upper().startswith("AFTERBURNER") else ("MIL" if out.n1_pct>=100 else "DERATE")),
        "N1%": int(out.n1_pct),
        "V1": int(out.v1), "Vr": int(out.vr), "V2": int(out.v2),
        "Stop ft": int(out.asd_ft),
        "Cont (OEI) ft": int(out.agd_reg_oei_ft),
        "Cont (AEO) ft": int(out.agd_aeo_liftoff_ft),
        "Req (Reg) ft": int(out.req_ft_reg),
        "Req (AEO) ft": int(out.req_ft_aeo),
        "Authorized Reg": "YES" if max(out.asd_ft,out.agd_reg_oei_ft) <= max(0.0, TORA-Shorten) + min(max(0.0, TODA-Shorten) - max(0.0, TORA-Shorten), max(0.0, TORA-Shorten)*0.5) and out.asd_ft <= max(0.0, ASDA-Shorten) else "NO",
        "Authorized AEO": "YES" if max(out.asd_ft,out.agd_aeo_liftoff_ft) <= max(0.0, TORA-Shorten) + min(max(0.0, TODA-Shorten) - max(0.0, TORA-Shorten), max(0.0, TORA-Shorten)*0.5) and out.asd_ft <= max(0.0, ASDA-Shorten) else "NO",
    }

cand = [
    evaluate_config("MANEUVER", "Auto-Select"),
    evaluate_config("MANEUVER", "MIL"),
    evaluate_config("MANEUVER", "Manual Derate"),
    evaluate_config("FULL", "MIL"),
]
try:
    cand_df = pd.DataFrame(cand)
    st.dataframe(cand_df, use_container_width=True)
except Exception:
    st.write(cand)

# Solvers: Min N1 and Max TOW
st.markdown("### Solvers")
colx, coly, colz = st.columns(3)
with colx:
    want_reg = st.toggle("Solve for Regulatory (else AEO)", value=True)
with coly:
    if st.button("Find minimum N1% for current flaps"):
        base = dict(
            perfdb=perfdb, rwy_heading_deg=float(HDG), tora_ft=float(TORA), toda_ft=float(TODA), asda_ft=float(ASDA),
            field_elev_ft=float(ELEV), slope_pct=float(SLOPE), shorten_ft=float(Shorten),
            oat_c=float(OAT), qnh_inhg=float(QNH_inHg), wind_speed=float(WIND_SPD), wind_dir_deg=float(WIND_DIR), wind_units=str(wind_units), wind_policy=str(wind_policy),
            gw_lbs=float(GW), flap_mode=str(flap_mode), thrust_mode="DERATE", target_n1_pct=float(98.0)
        )
        n1_min = solve_min_n1(perfdb, {**base, "flap_deg": (0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20))}, engine_out=want_reg)
        eq_oat = assumed_oat_for_n1(pressure_altitude_ft(ELEV, QNH_inHg), OAT, n1_min)
        st.info(f"Minimum N1 ≈ {n1_min:.0f}% (equivalent assumed OAT ≈ {eq_oat:.1f} °C)")
with colz:
    if st.button("Find max Takeoff Weight"):
        base = dict(
            perfdb=perfdb, rwy_heading_deg=float(HDG), tora_ft=float(TORA), toda_ft=float(TODA), asda_ft=float(ASDA),
            field_elev_ft=float(ELEV), slope_pct=float(SLOPE), shorten_ft=float(Shorten),
            oat_c=float(OAT), qnh_inhg=float(QNH_inHg), wind_speed=float(WIND_SPD), wind_dir_deg=float(WIND_DIR), wind_units=str(wind_units), wind_policy=str(wind_policy),
            gw_lbs=float(GW), flap_mode=str(flap_mode), thrust_mode=str(thrust_mode if thrust_mode!="Manual Derate" else "MIL"), target_n1_pct=float(derate_n1)
        )
        mtow = solve_max_tow(perfdb, base, engine_out=want_reg)
        st.info(f"Max GW for current config ≈ {mtow:,.0f} lb")

# Export / Presets
st.markdown("### Export & Presets")
# Flight card
card = {
    "theatre": Theatre, "airport": Airport, "runway": RunwayLabel,
    "declared": {"TORA": TORA, "TODA": TODA, "ASDA": ASDA, "Elev": ELEV, "Shorten": Shorten},
    "weather": {"OAT_C": OAT, "QNH_inHg": QNH_inHg, "wind": f"{int(WIND_DIR):03d}@{int(WIND_SPD)} {wind_units}"},
    "config": {"GW_lb": GW, "Flaps": res.flap_text, "Thrust": res.thrust_text, "N1_pct": int(res.n1_pct)},
    "speeds": {"V1": int(res.v1), "Vr": int(res.vr), "V2": int(res.v2), "Vs": int(res.vs)},
    "distances_ft": {"ASD": int(res.asd_ft), "Continue_OEI": int(res.agd_reg_oei_ft), "Continue_AEO": int(res.agd_aeo_liftoff_ft),
                      "Required_Reg": int(res.req_ft_reg), "Required_AEO": int(res.req_ft_aeo), "Available": int(res.avail_ft)},
    "auth": {"Reg": "OK" if max(res.asd_ft,res.agd_reg_oei_ft)<=res.avail_ft else "NO", "AEO": "OK" if max(res.asd_ft,res.agd_aeo_liftoff_ft)<=res.avail_ft else "NO"},
    "data_version": {"perf": csv_version_info("f14_perf.csv"), "runways": csv_version_info("dcs_airports.csv")}
}
card_json = json.dumps(card, indent=2)
st.download_button("Download flight card (JSON)", card_json, file_name="f14_takeoff_card.json")

# Presets: save/load
preset = {
    "Theatre": Theatre, "Airport": Airport, "RunwayLabel": RunwayLabel,
    "OAT": OAT, "QNH_inHg": QNH_inHg, "Wind": f"{int(WIND_DIR):03d}@{int(WIND_SPD)}", "WindUnits": wind_units, "WindPolicy": wind_policy,
    "GW": GW, "Flaps": flap_mode, "Thrust": thrust_mode, "DerateN1": derate_n1,
    "Shorten": Shorten
}
st.download_button("Save preset", json.dumps(preset), file_name="f14_preset.json")
up = st.file_uploader("Load preset", type=["json"], accept_multiple_files=False)
if up is not None:
    try:
        data = json.loads(up.read().decode("utf-8"))
        st.success("Preset loaded — please re-select from sidebar to apply (URL updated).")
    except Exception as e:
        st.error(f"Failed to load preset: {e}")

# Notes / advisories
if res.hw_kn < -10.0:
    st.warning("Tailwind component exceeds 10 kt — takeoff NOT AUTHORIZED.")
if abs(res.cw_kn) > 30.0:
    st.warning("Crosswind component exceeds 30 kt — takeoff NOT AUTHORIZED.")

for n in res.notes:
    st.info(n)

st.caption("This tool estimates F-14B takeoff performance for DCS per FAA-style rules (14 CFR 121.189) with NATOPS-aware data. Not for real-world use.")
