# f14_takeoff_app.py — DCS F-14B Takeoff Performance (autorun + MAN synthesis + manual runway override)
#
# Requires (exact case) in repo root:
#   • dcs_airports.csv
#   • f14_perf.csv
#
# Autoruns the calculation as soon as inputs are valid. Hardened against
# UI transients (Fuel+Stores), guards v-speed floors, and prevents NaNs.

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F-14B Takeoff", page_icon="✈️", layout="wide")

# ------------------------------ tuning constants ------------------------------
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}  # per engine (approx) for OEI guardrail

DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}  # FULL cannot derate
ALPHA_N1_DIST = 2.0           # distance ∝ 1/(N1^alpha)
AEO_VR_FRAC   = 0.88          # Vr ground roll ≈ 0.88 × AEO liftoff-to-35ft (MAN/UP)
AEO_VR_FRAC_FULL = 0.82       # crisper Vr fraction for FULL flaps

OEI_AGD_FACTOR = 1.20         # default; UI can set to 1.15 for DCS-calibrated
AEO_CAL_FACTOR = 1.00

WIND_FACTORS = {"None": (1.0, 1.0), "50/150": (0.5, 1.5)}  # headwind credit, tailwind penalty

# ------------------------------ atmosphere & wind ------------------------------
def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

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
    """Outside grid, scale distances by density ratio vs a clamped ref (≤5000 ft, ≤30 °C)."""
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
    return hw, cw

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    # +20% per +1% uphill (ignore downhill credit)
    if slope_pct > 0:
        d *= (1.0 + 0.20 * slope_pct)
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)
    return max(d, 0.0)

# ------------------------------ data loaders ------------------------------
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

def ensure_flap20(df: pd.DataFrame) -> pd.DataFrame:
    """If MAN(20°) rows are missing, synthesize by blending UP(0°)/FULL(40°) on matching keys.
       Bias toward FULL: MAN ≈ 0.4*UP + 0.6*FULL."""
    if (df["flap_deg"] == 20).any():
        return df
    keys = ["thrust","gw_lbs","press_alt_ft","oat_c"]
    up   = df[df["flap_deg"] == 0]
    full = df[df["flap_deg"] == 40]
    if up.empty or full.empty:
        return df
    m = pd.merge(up, full, on=keys, suffixes=("_up","_full"))
    if m.empty:
        return df

    def blend(a, b, w=0.4):  # w = weight on UP
        return w*a + (1.0 - w)*b

    new = pd.DataFrame({
        "model": "F-14B",
        "flap_deg": 20,
        "thrust": m["thrust"],
        "gw_lbs": m["gw_lbs"],
        "press_alt_ft": m["press_alt_ft"],
        "oat_c": m["oat_c"],
        "Vs_kt": blend(m["Vs_kt_up"], m["Vs_kt_full"]),
        "V1_kt": blend(m["V1_kt_up"], m["V1_kt_full"]),
        "Vr_kt": blend(m["Vr_kt_up"], m["Vr_kt_full"]),
        "V2_kt": blend(m["V2_kt_up"], m["V2_kt_full"]),
        "ASD_ft": blend(m["ASD_ft_up"], m["ASD_ft_full"]),
        "AGD_ft": blend(m["AGD_ft_up"], m["AGD_ft_full"]),
        "note": "synth-MAN(20) 0.4*UP + 0.6*FULL"
    })
    return pd.concat([df, new], ignore_index=True)

@st.cache_data
def load_perf() -> pd.DataFrame:
    try:
        df = pd.read_csv("f14_perf.csv", comment="#")
    except Exception:
        st.error("f14_perf.csv not found in repo root.")
        st.stop()
    df["thrust"] = df["thrust"].astype(str).str.upper().replace({"MIL": "MILITARY", "AB": "AFTERBURNER"})
    df = _ensure_numeric(df, ["flap_deg","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"])
    df = df.dropna(subset=["flap_deg","gw_lbs","press_alt_ft","oat_c","Vr_kt","ASD_ft","AGD_ft"])
    df = ensure_flap20(df)
    return df

# Learn AB vs MIL distance ratio when AB table is missing for a flap
def estimate_ab_multiplier(perfdb: pd.DataFrame, flap_deg: int) -> float:
    """
    Estimate AB vs MIL distance ratio for the given flap by comparing rows that
    share (gw_lbs, press_alt_ft, oat_c). Falls back to any flap, else default.
    Returns a factor <1.0 (multiply distances by this when AB table is missing).
    """
    def median_ratio(df_m: pd.DataFrame, df_a: pd.DataFrame) -> Optional[float]:
        keys = ["gw_lbs", "press_alt_ft", "oat_c"]
        M = df_m[keys + ["ASD_ft", "AGD_ft"]]
        A = df_a[keys + ["ASD_ft", "AGD_ft"]]
        merged = pd.merge(M, A, on=keys, suffixes=("_MIL", "_AB"))
        if merged.empty:
            return None
        ratios = []
        for col in ("ASD_ft", "AGD_ft"):
            num = merged[f"{col}_AB"].to_numpy(dtype=float)
            den = merged[f"{col}_MIL"].to_numpy(dtype=float)
            valid = (den > 1.0)
            if valid.any():
                r = np.median(np.clip(num[valid] / den[valid], 0.65, 0.95))
                ratios.append(r)
        if ratios:
            return float(min(ratios))
        return None

    # Try same flap first
    mil_f = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == "MILITARY")]
    ab_f  = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == "AFTERBURNER")]
    r = median_ratio(mil_f, ab_f)
    if r is not None:
        return r

    # Any flap
    mil_any = perfdb[perfdb["thrust"] == "MILITARY"]
    ab_any  = perfdb[perfdb["thrust"] == "AFTERBURNER"]
    r2 = median_ratio(mil_any, ab_any)
    if r2 is not None:
        return r2

    # Default if no AB rows at all
    return 0.82

# ------------------------------ interpolation helpers ------------------------------
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
    if len(xs) < 2:
        return float(ys[0])
    if gw_x <= xs[0]:
        x0, x1 = xs[0], xs[1]; y0, y1 = ys[0], ys[1]
        return float(y0 + (y1 - y0)/(x1 - x0) * (gw_x - x0))
    if gw_x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]; y0, y1 = ys[-2], ys[-1]
        return float(y0 + (y1 - y0)/(x1 - x0) * (gw_x - x0))
    return float(np.interp(gw_x, xs, ys))

def _interp_slice(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float) -> dict:
    """Interpolate a single flap/thrust slice at (gw, pa, oat).
       Returns dict with Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft."""
    sub = perf[(perf["flap_deg"] == flap_deg) & (perf["thrust"] == thrust)]
    if sub.empty:
        # fallbacks: any thrust for that flap, else any flap with thrust, else whole table
        sub = perf[(perf["flap_deg"] == flap_deg)]
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

def interp_perf(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float):
    """
    Interpolate the requested flap/thrust. If MAN(20°) is missing at this condition,
    synthesize it from UP(0°) and FULL(40°) by blending their *interpolated* values.
    UP requests still use MAN as the base to avoid unrealistically short distances.
    """
    # UP uses MAN table as base
    flap_req = 20 if flap_deg == 0 else flap_deg

    # If we need MAN and the table is sparse/absent, synthesize from UP & FULL
    if flap_req == 20 and (perf[(perf["flap_deg"] == 20) & (perf["thrust"] == thrust)].empty):
        up   = _interp_slice(perf, 0,  thrust, gw, pa, oat)
        full = _interp_slice(perf, 40, thrust, gw, pa, oat)
        w_up, w_full = 0.45, 0.55
        out = {}
        for f in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
            out[f] = w_up*up[f] + w_full*full[f]
        return out

    # Normal slice
    return _interp_slice(perf, flap_req, thrust, gw, pa, oat)

# ------------------------------ NATOPS detection ------------------------------
def agd_is_liftoff_mode(perfdb: pd.DataFrame, flap_deg: int, thrust: str) -> bool:
    """
    Heuristic: if most rows for this (flap, thrust) carry a NATOPS note,
    assume AGD_ft in the CSV is already 'liftoff to 35 ft' (not ground roll).
    """
    sub = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == thrust)]
    if sub.empty:
        return False
    mask = sub["note"].astype(str).str.contains("NATOPS", case=False, na=False)
    return bool(mask.mean() >= 0.5)

# ------------------------------ OEI guardrail ------------------------------
def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct/100.0)  # MIL-scaled per engine
    t_over_w = t_oei / max(gw_lbs, 1.0)
    gradient_net = t_over_w - drag_over_w
    return gradient_net >= 0.024  # 2.4%

# ------------------------------ trim model ------------------------------
def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# ------------------------------ wind text parsing ------------------------------
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

# ------------------------------ ENFORCE SPEED FLOORS (hardened) ------------------------------
def enforce_speed_floors(vs, v1, vr, v2, flap_deg: int):
    import math

    def num(x, default=float("nan")):
        try:
            y = float(x)
            return y if math.isfinite(y) else default
        except Exception:
            return default

    # Coerce to floats; allow NaNs for now
    vs = num(vs); v1 = num(v1); vr = num(vr); v2 = num(v2)

    # If the table returns NaNs (e.g., during UI transient), seed sensible defaults
    if not math.isfinite(vs) or vs <= 0:
        vs = 110.0 if flap_deg == 40 else 120.0

    # Seed missing v-speeds from Vs if needed
    if not math.isfinite(v1): v1 = vs * 1.10
    if not math.isfinite(vr): vr = vs * 1.20
    if not math.isfinite(v2): v2 = vs * 1.30

    # Vmcg floor (simple guard). You can refine by GW/DA later.
    vmcg = 112.0 if flap_deg == 40 else 118.0

    # Ordered, monotonic minimums and gaps
    v1_min = max(vmcg + 3.0, 0.95 * vr)      # V1 not far below VR
    vr_min = max(vmcg + 8.0, v1 + 3.0, 1.05 * vs)
    v2_min = max(vr + 10.0, 1.18 * vs)

    v1f = max(v1, v1_min)
    vrf = max(vr, vr_min)
    v2f = max(v2, v2_min)

    # Round to integer knots
    return float(int(round(v1f))), float(int(round(vrf))), float(int(round(v2f)))

# ------------------------------ core data structures ------------------------------
@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_aeo_liftoff_ft: float; agd_reg_oei_ft: float
    req_ft: float; avail_ft: float; limiting: str
    hw_kn: float; cw_kn: float; notes: list

# ------------------------------ main compute ------------------------------
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
    da = density_altitude_ft(pa, oat_c)
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

    # Which performance slice (for interpolation *and* grid bounds)
    use_flap_for_table = 20 if flap_deg == 0 else flap_deg

    # Check if AB slice exists for this flap
    has_ab_slice = not perfdb[(perfdb["flap_deg"] == use_flap_for_table) & (perfdb["thrust"] == "AFTERBURNER")].empty

    # Choose table thrust:
    #  • AB only if explicitly selected AND slice exists; otherwise use MIL table
    #  • Derates always MIL-anchored
    if thrust_mode == "AB" and has_ab_slice:
        table_thrust = "AFTERBURNER"
    else:
        table_thrust = "MILITARY"

    # Bounds used to decide if we’re outside grid (disable DA top-up inside the grid)
    sub_bounds = perfdb[(perfdb["flap_deg"] == use_flap_for_table) & (perfdb["thrust"] == table_thrust)]
    if sub_bounds.empty:
        sub_bounds = perfdb[(perfdb["flap_deg"] == use_flap_for_table)]
    if sub_bounds.empty:
        sub_bounds = perfdb[(perfdb["thrust"] == table_thrust)]
    if sub_bounds.empty:
        sub_bounds = perfdb

    pa_min = float(sub_bounds["press_alt_ft"].min()); pa_max = float(sub_bounds["press_alt_ft"].max())
    t_min  = float(sub_bounds["oat_c"].min());        t_max  = float(sub_bounds["oat_c"].max())
    outside_grid = (pa < pa_min or pa > pa_max or oat_c < t_min or oat_c > t_max)

    # Interpolate speeds & base distances (now MAN synthesis is built-in)
    base = interp_perf(perfdb, use_flap_for_table, table_thrust, float(gw_lbs), float(pa), float(oat_c))

    # Pull raw speeds, then enforce floors IMMEDIATELY (prevents NaNs on first render)
    vs = float(base.get("Vs_kt", np.nan))
    v1 = float(base.get("V1_kt", np.nan))
    vr = float(base.get("Vr_kt", np.nan))
    v2 = float(base.get("V2_kt", np.nan))
    v1, vr, v2 = enforce_speed_floors(vs, v1, vr, v2, flap_deg)

    # CSV distances
    asd_base = float(base["ASD_ft"])
    agd_csv  = float(base["AGD_ft"])

    # If the source block is NATOPS-based for this flap/thrust, treat AGD as already 'liftoff to 35 ft'
    agd_already_liftoff = agd_is_liftoff_mode(perfdb, use_flap_for_table, table_thrust)

    if agd_already_liftoff:
        agd_aeo_liftoff_base = agd_csv
    else:
        # Convert ground roll → liftoff-to-35 ft (gentle DA-aware factor; no extra scaling inside grid)
        liftoff_factor = 1.42 + 0.15 * max(0.0, min(da/8000.0, 1.25))
        agd_aeo_liftoff_base = agd_csv * liftoff_factor

    # UP penalty vs MAN baseline
    if flap_deg == 0:
        asd_base *= 1.06
        agd_aeo_liftoff_base *= 1.06

    # FULL cannot derate
    if flap_deg == 40 and thrust_mode in ("Auto-Select", "DERATE", "Manual Derate"):
        notes.append("Derate with FULL flaps not allowed — using MIL for calculation.")
        thrust_mode = "MIL"

    # If AB requested but slice doesn’t exist, approximate AB by reducing MIL distances
    if thrust_mode == "AB" and not has_ab_slice:
        ab_mult = estimate_ab_multiplier(perfdb, use_flap_for_table)
        asd_base *= ab_mult
        agd_aeo_liftoff_base *= ab_mult
        notes.append(f"AFTERBURNER table missing for this flap; approximated using AB/MIL ratio ≈ {ab_mult:.2f}.")

    # MIL-anchored N1 multiplier for derates (AB uses full power, so N1 stays 100%)
    def mult_from_n1(n1pct: float) -> float:
        eff = max(0.90, min(1.0, n1pct/100.0))
        return 1.0 / (eff ** ALPHA_N1_DIST)

    # DA “top-up” ONLY if outside CSV grid
    def maybe_da_scale(d_ft: float) -> float:
        if outside_grid:
            return d_ft * da_out_of_grid_scale(pa, oat_c)
        return d_ft

    def wind_slope(d_ft: float) -> float:
        return apply_wind_slope(d_ft, slope_pct, hw, wind_policy)

    def distances_for(n1pct: float) -> Tuple[float, float]:
        m = mult_from_n1(n1pct)
        asd = wind_slope(maybe_da_scale(asd_base * m))
        agd_aeo = wind_slope(maybe_da_scale(agd_aeo_liftoff_base * m))
        agd_aeo *= AEO_CAL_FACTOR
        # sanity: non-negative
        return max(asd, 0.0), max(agd_aeo, 0.0)

    def field_ok(asd_eff: float, agd_cont_ft: float, engine_out: bool) -> Tuple[bool, float, str]:
        """Return (ok, required, limiting) for either OEI (engine_out=True) or AEO (False)."""
        tora_eff = max(0.0, tora_ft - shorten_ft)
        toda_eff = max(0.0, toda_ft - shorten_ft)
        asda_eff_lim = max(0.0, asda_ft - shorten_ft)
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        if engine_out:
            cont = agd_cont_ft * OEI_AGD_FACTOR  # OEI regulatory
            limiting = "ASD (stop)" if asd_eff >= cont else "Engine-out continue"
        else:
            cont = agd_cont_ft                 # AEO practical
            limiting = "ASD (stop)" if asd_eff >= cont else "All-engines continue"

        req = max(asd_eff, cont)
        ok = (asd_eff <= asda_eff_lim) and (cont <= tod_limit) and (cont <= toda_eff)
        return ok, req, limiting

    # Choose thrust / N1
    n1 = 100.0
    thrust_text = thrust_mode
    if thrust_mode == "DERATE":
        floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
        n1 = float(int(math.ceil(max(floor_pct, target_n1_pct))))  # whole %
        thrust_text = "DERATE" if n1 < 100.0 else "MIL"
    elif thrust_mode == "Auto-Select":
        # Evaluate MAN @ MIL first (now robust even if MAN is sparse in CSV)
        asd_mil, agd_aeo_mil = distances_for(100.0)
        ok_mil, _, _ = field_ok(asd_mil, agd_aeo_mil, engine_out=True)  # regulatory check for escalation
        if ok_mil:
            # Find lowest compliant derate
            floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
            lo, hi = floor_pct, 100.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                asd_m, agd_aeo_m = distances_for(mid)
                ok_m, _, _ = field_ok(asd_m, agd_aeo_m, engine_out=True)
                ok_m = ok_m and compute_oei_second_segment_ok(gw_lbs, mid, flap_deg)
                if ok_m: hi = mid
                else:    lo = mid
            n1 = float(int(math.ceil(hi)))
            thrust_text = "DERATE" if n1 < 100.0 else "MIL"
        else:
            if flap_deg != 40:
                notes.append("Auto: MAN @ MIL fails §121.189; escalating to FULL @ MIL.")
            return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                                   field_elev_ft, slope_pct, shorten_ft,
                                   oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                                   gw_lbs, "FULL", "MIL", 100.0)
    elif thrust_mode == "MIL":
        n1 = 100.0
        thrust_text = "MIL"
    else:  # AB (always 100% N1)
        n1 = 100.0
        thrust_text = "AFTERBURNER"
        notes.append("Afterburner selected — NOT AUTHORIZED for F-14B except as last resort.")

    # Final distances (AEO liftoff & ASD)
    asd_fin, agd_aeo_fin = distances_for(n1)
    # Regulatory OEI continue distance is derived in field_ok when engine_out=True

    # Default to regulatory limiting for auto-flap escalate logic
    ok_reg, req_reg, limiting_reg = field_ok(asd_fin, agd_aeo_fin, engine_out=True)

    # Auto-flap escalation if not OK at MAN
    if flap_mode == "Auto-Select" and not ok_reg and flap_deg != 40:
        return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                               field_elev_ft, slope_pct, shorten_ft,
                               oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                               gw_lbs, "FULL", "MIL", 100.0)

    avail = max(0.0, tora_ft - shorten_ft)

    # Package result (regulatory numbers for req/limiting by default; UI may recompute for AEO mode)
    agd_reg_fin = agd_aeo_fin * OEI_AGD_FACTOR
    return Result(
        v1=v1, vr=vr, v2=v2, vs=float(base.get("Vs_kt", np.nan)),
        flap_text=flap_text, thrust_text=thrust_text, n1_pct=n1,
        asd_ft=asd_fin, agd_aeo_liftoff_ft=agd_aeo_fin, agd_reg_oei_ft=agd_reg_fin,
        req_ft=req_reg, avail_ft=avail, limiting=limiting_reg,
        hw_kn=hw, cw_kn=cw, notes=notes
    )

# ------------------------------ UI ------------------------------
st.title("DCS F-14B Takeoff — FAA-Based Model (autorun)")

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

    # Base values from the database
    tora_ft = float(rwy["tora_ft"])
    toda_ft = float(rwy["toda_ft"])
    asda_ft = float(rwy["asda_ft"])
    elev_ft = float(rwy["threshold_elev_ft"])
    hdg = float(rwy["heading_deg"])
    slope = float(rwy.get("slope_percent", 0.0) or 0.0)

    # Manual runway override (length in ft or NM, elevation in ft)
    st.checkbox("Override runway data", value=False, key="rw_override")
    if st.session_state.rw_override:
        st.caption("Manual runway override")
        manual_len_txt = st.text_input("Runway Length (ft or NM)", value="")
        def _parse_len(txt: str) -> float:
            t = (txt or "").strip().lower()
            if not t:
                return float(rwy["tora_ft"])
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", t)
            if not m:
                return float(rwy["tora_ft"])
            val = float(m.group(1))
            is_nm = ("nm" in t) or (" nmi" in t)
            feet = val * 6076.12 if is_nm else val
            return max(0.0, feet)

        manual_elev_ft = st.number_input("Field Elevation (ft)", value=float(elev_ft), step=1.0)
        length_ft = _parse_len(manual_len_txt)

        # Apply to TORA/TODA/ASDA uniformly (simple & conservative). Shorten will still apply below.
        tora_ft = float(length_ft)
        toda_ft = float(length_ft)
        asda_ft = float(length_ft)
        elev_ft = float(manual_elev_ft)
        st.info(f"Override active → TORA/TODA/ASDA set to {tora_ft:.0f} ft, Elev {elev_ft:.0f} ft")

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
    oat_c = float(st.number_input("OAT (°C)", value=15.0, step=1.0))
    qnh_val = float(st.number_input("QNH value", value=29.92, step=0.01, format="%.2f"))
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
        gw = float(st.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=500.0))
    else:
        # Hardened numeric inputs that always yield a valid GW
        empty_w   = float(st.number_input("Empty weight (lb)",  min_value=38000.0, max_value=46000.0, value=41780.0, step=50.0))
        fuel_lb   = float(st.number_input("Internal fuel (lb)", min_value=0.0,     max_value=20000.0, value=8000.0,  step=100.0))
        ext_tanks = int(st.selectbox("External tanks (267 gal)", [0,1,2], index=0))
        aim9      = int(st.slider("AIM-9 count", 0, 2, 0))
        aim7      = int(st.slider("AIM-7 count", 0, 4, 0))
        aim54     = int(st.slider("AIM-54 count", 0, 6, 0))
        lantirn   = bool(st.checkbox("LANTIRN pod", value=False))

        wcalc = (
            empty_w
            + fuel_lb
            + ext_tanks * 1900.0
            + aim9 * 190.0
            + aim7 * 510.0
            + aim54 * 1000.0
            + (440.0 if lantirn else 0.0)
        )
        # Always produce a numeric GW; user can edit if desired
        gw = float(st.number_input("Computed GW (editable)", min_value=40000.0, max_value=80000.0, value=float(wcalc), step=10.0))

    flap_mode = st.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=0)

    st.header("Thrust")
    thrust_mode = st.radio("Mode", ["Auto-Select", "Manual Derate", "MIL", "AB"], index=0)
    derate_n1 = 98.0
    if thrust_mode == "Manual Derate":
        flap_for_floor = 0 if flap_mode=="UP" else (40 if flap_mode=="FULL" else 20)
        floor = DERATE_FLOOR_BY_FLAP.get(flap_for_floor, 0.90)*100.0
        st.caption(f"Derate floor by flap: {floor:.0f}% N1 (MIL)")
        derate_n1 = float(st.slider("Target N1 % (MIL)", min_value=float(int(floor)), max_value=100.0, value=max(95.0, float(int(floor))), step=1.0))

    with st.expander("Advanced / Calibration", expanded=False):
        calib = st.radio("Model calibration", ["FAA-conservative", "DCS-calibrated"], index=1,
                         help=("FAA: OEI factor 1.20 (conservative). "
                               "DCS: OEI factor 1.15 (tuned to your hot/high tests)."))
        if calib == "DCS-calibrated":
            AEO_CAL = 1.00
            OEI_FAC = 1.15
        else:
            AEO_CAL = 1.00
            OEI_FAC = 1.20
        globals()["AEO_CAL_FACTOR"] = AEO_CAL
        globals()["OEI_AGD_FACTOR"]  = OEI_FAC

    st.header("Compliance Mode")
    compliance_mode = st.radio("How should limits be checked?", ["Regulatory (OEI §121.189)", "AEO Practical"], index=0,
                               help=("Regulatory: engine-out continue distance is limiting. "
                                     "AEO Practical: uses all-engines continue distance, matching typical DCS tests."))

# ------------------------------ autorun compute ------------------------------
ready = "gw" in locals() and isinstance(gw, (int, float)) and gw >= 40000.0

if ready:
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
        st.metric("Vs (kt)", f"{res.vs:.0f}" if math.isfinite(res.vs) else "—")
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

        # Required + margins by mode
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

        # Compliance banner
        req_margin = min(asd_margin, cont_margin)
        if ok:
            st.success(f"COMPLIANT — Margin {req_margin:.0f} ft (ASD margin {asd_margin:.0f}, continue margin {cont_margin:.0f}).")
        else:
            st.error(f"NOT AUTHORIZED — Short by {-req_margin:.0f} ft (ASD margin {asd_margin:.0f}, continue margin {cont_margin:.0f}).")
            st.caption(f"TOD limit: {tod_limit:.0f} ft | ASDA: {asda_eff_lim:.0f} ft | Mode: {compliance_mode}")

    st.markdown("---")
    st.subheader("All-engines takeoff estimates (for DCS comparison)")
    e1, e2 = st.columns(2)
    vr_frac = AEO_VR_FRAC_FULL if res.flap_text.upper().startswith("FULL") else AEO_VR_FRAC
    with e1:
        st.metric("Vr ground roll (ft)", f"{res.agd_aeo_liftoff_ft * vr_frac:.0f}")
    with e2:
        st.metric("Liftoff distance to 35 ft (ft)", f"{res.agd_aeo_liftoff_ft:.0f}")
    st.caption("AEO estimates shown for DCS. Regulatory checks assume an engine-out per 14 CFR 121.189.")

    for n in res.notes:
        st.warning(n)
else:
    st.info("Select fuel/stores (or enter a valid gross weight) to compute performance.")
