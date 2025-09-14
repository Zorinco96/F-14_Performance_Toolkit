# f14_takeoff_core.py
# Core logic for DCS F-14B takeoff performance (UI-agnostic)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ------------------------------ tuning constants ------------------------------
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}   # per engine (approx) for OEI guardrail
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}  # FULL cannot derate

ALPHA_N1_DIST = 2.0           # distance ∝ 1/(N1^alpha)
AEO_VR_FRAC   = 0.88          # Vr ground roll ≈ 0.88 × AEO liftoff-to-35ft (MAN/UP)
AEO_VR_FRAC_FULL = 0.82       # crisper Vr fraction for FULL flaps

OEI_AGD_FACTOR = 1.20         # default; can be set to 1.15 (DCS tuned) by UI
AEO_CAL_FACTOR = 1.00         # reserved overall AEO calibration if needed

WIND_FACTORS: Dict[str, Tuple[float, float]] = {
    "None": (1.0, 1.0),    # headwind credit 1.0×, tailwind penalty 1.0×
    "50/150": (0.5, 1.5),  # 50% headwind credit, 150% tailwind penalty (airline policy)
}

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
    BETA = 0.80
    return (sig_ref / max(1e-6, sig_act)) ** BETA

def _hot_high_relief(oat_c: float, da_ft: float) -> float:
    """Empirical relief to reduce AEO liftoff distances at very hot/high conditions.
    Targets ~10–12% reduction near 47 °C with high DA, tapering to ~0 near ISA."""
    if oat_c <= 30.0 or da_ft <= 3000.0:
        return 1.0
    dt  = max(0.0, oat_c - 30.0)     # degrees above 30 °C
    dda = max(0.0, da_ft - 3000.0)   # DA above 3000 ft
    relief = 0.0025 * dt + 0.0000075 * dda  # caps at ~0.12
    return float(max(0.88, 1.0 - min(relief, 0.12)))

def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float, float]:
    # direction is FROM
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

# ------------------------------ data helpers (UI can call) ------------------------------
def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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

    def blend(a, b, w=0.4):
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

def load_perf_csv(path: str = "f14_perf.csv") -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["thrust"] = df["thrust"].astype(str).str.upper().replace({"MIL": "MILITARY", "AB": "AFTERBURNER"})
    df = ensure_numeric(df, ["flap_deg","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"])
    df = df.dropna(subset=["flap_deg","gw_lbs","press_alt_ft","oat_c","Vr_kt","ASD_ft","AGD_ft"])
    df = ensure_flap20(df)
    return df

def estimate_ab_multiplier(perfdb: pd.DataFrame, flap_deg: int) -> float:
    """Estimate AB vs MIL distance ratio for the given flap; fallback to any flap; else 0.82."""
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

    mil_f = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == "MILITARY")]
    ab_f  = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == "AFTERBURNER")]
    r = median_ratio(mil_f, ab_f)
    if r is not None:
        return r

    mil_any = perfdb[perfdb["thrust"] == "MILITARY"]
    ab_any  = perfdb[perfdb["thrust"] == "AFTERBURNER"]
    r2 = median_ratio(mil_any, ab_any)
    if r2 is not None:
        return r2

    return 0.82

# ------------------------------ interpolation (tri-linear) ------------------------------
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

# ------------------------------ NATOPS detection ------------------------------
def agd_is_liftoff_mode(perfdb: pd.DataFrame, flap_deg: int, thrust: str) -> bool:
    """If most rows for this (flap, thrust) say NATOPS, assume AGD_ft already liftoff-to-35 ft."""
    sub = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == thrust)]
    if sub.empty:
        return False
    mask = sub["note"].astype(str).str.contains("NATOPS", case=False, na=False)
    return bool(mask.mean() >= 0.5)

# ------------------------------ OEI guardrail ------------------------------
def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct/100.0)  # MIL-scaled per engine (first segment proxy)
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

    vs = num(vs); v1 = num(v1); vr = num(vr); v2 = num(v2)
    if not math.isfinite(vs) or vs <= 0:
        vs = 110.0 if flap_deg == 40 else 120.0

    if not math.isfinite(v1): v1 = vs * 1.10
    if not math.isfinite(vr): vr = vs * 1.20
    if not math.isfinite(v2): v2 = vs * 1.30

    vmcg = 112.0 if flap_deg == 40 else 118.0

    v1_min = max(vmcg + 3.0, 0.95 * vr)
    vr_min = max(vmcg + 8.0, v1 + 3.0, 1.05 * vs)
    v2_min = max(vr + 10.0, 1.18 * vs)

    v1f = max(v1, v1_min)
    vrf = max(vr, vr_min)
    v2f = max(v2, v2_min)

    return float(int(round(v1f))), float(int(round(vrf))), float(int(round(v2f)))

# ------------------------------ core result ------------------------------
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

    # Flaps (Auto defaults to MAN when evaluating)
    flap_text = "MANEUVER" if flap_mode == "Auto-Select" else flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)
    use_flap_for_table = 20 if flap_deg == 0 else flap_deg

    # Thrust table presence
    has_ab_slice = not perfdb[(perfdb["flap_deg"] == use_flap_for_table) & (perfdb["thrust"] == "AFTERBURNER")].empty

    # Base table thrust
    if thrust_mode == "AB" and has_ab_slice:
        table_thrust = "AFTERBURNER"
    else:
        table_thrust = "MILITARY"

    # Bounds for DA “top-up” (only if outside grid)
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

    # Interpolate base
    base = interp_perf(perfdb, use_flap_for_table, table_thrust, float(gw_lbs), float(pa), float(oat_c))

    # Pull raw speeds, enforce floors
    vs = float(base.get("Vs_kt", np.nan))
    v1 = float(base.get("V1_kt", np.nan))
    vr = float(base.get("Vr_kt", np.nan))
    v2 = float(base.get("V2_kt", np.nan))
    v1, vr, v2 = enforce_speed_floors(vs, v1, vr, v2, flap_deg)

    # Base distances
    asd_base = float(base["ASD_ft"])
    agd_csv  = float(base["AGD_ft"])
    agd_already_liftoff = agd_is_liftoff_mode(perfdb, use_flap_for_table, table_thrust)

    if agd_already_liftoff:
        agd_aeo_liftoff_base = agd_csv
    else:
        # Gentler high-DA slope and hot/high relief
        liftoff_factor = 1.38 + 0.12 * max(0.0, min(da/8000.0, 1.25))
        agd_aeo_liftoff_base = agd_csv * liftoff_factor
        agd_aeo_liftoff_base *= _hot_high_relief(oat_c, da)

    # UP penalty vs MAN baseline (slightly softened)
    if flap_deg == 0:
        asd_base *= 1.04
        agd_aeo_liftoff_base *= 1.04

    # FULL cannot derate
    if flap_deg == 40 and thrust_mode in ("Auto-Select", "DERATE", "Manual Derate"):
        notes.append("Derate with FULL flaps not allowed — using MIL for calculation.")
        thrust_mode = "MIL"

    # If AB requested but slice missing, approximate
    if thrust_mode == "AB" and not has_ab_slice:
        ab_mult = estimate_ab_multiplier(perfdb, use_flap_for_table)
        asd_base *= ab_mult
        agd_aeo_liftoff_base *= ab_mult
        notes.append(f"AFTERBURNER table missing for this flap; approximated using AB/MIL ratio ≈ {ab_mult:.2f}.")

    # MIL-anchored N1 multiplier for derates (AB always 100%)
    def mult_from_n1(n1pct: float) -> float:
        eff = max(0.90, min(1.0, n1pct/100.0))
        return 1.0 / (eff ** ALPHA_N1_DIST)

    # DA top-up only if outside CSV grid
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
        return max(asd, 0.0), max(agd_aeo, 0.0)

    def field_ok(asd_eff: float, agd_cont_ft: float, engine_out: bool) -> Tuple[bool, float, str]:
        """Return (ok, required, limiting) for OEI (engine_out=True) or AEO (False)."""
        tora_eff = max(0.0, tora_ft - shorten_ft)
        toda_eff = max(0.0, toda_ft - shorten_ft)
        asda_eff_lim = max(0.0, asda_ft - shorten_ft)
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        if engine_out:
            cont = agd_cont_ft * OEI_AGD_FACTOR  # OEI regulatory
            limiting = "ASD (stop)" if asd_eff >= cont else "Engine-out continue"
        else:
            cont = agd_cont_ft
            limiting = "ASD (stop)" if asd_eff >= cont else "All-engines continue"

        req = max(asd_eff, cont)
        ok = (asd_eff <= asda_eff_lim) and (cont <= tod_limit) and (cont <= toda_eff)
        return ok, req, limiting

    # Choose thrust / N1
    n1 = 100.0
    thrust_text = thrust_mode
    if thrust_mode == "DERATE":
        floor_pct = DERATE_FLOOR_BY_FLAP.get(flap_deg, 0.90)*100.0
        n1 = float(int(math.ceil(max(floor_pct, target_n1_pct))))
        thrust_text = "DERATE" if n1 < 100.0 else "MIL"
    elif thrust_mode == "Auto-Select":
        asd_mil, agd_aeo_mil = distances_for(100.0)
        ok_mil, _, _ = field_ok(asd_mil, agd_aeo_mil, engine_out=True)
        if ok_mil:
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
                # Lightweight anti-escalation: below ~58k, keep MAN/MIL @100 unless failure is clearly >300 ft.
                asd_100, agd_100 = distances_for(100.0)
                tora_eff = max(0.0, tora_ft - shorten_ft)
                toda_eff = max(0.0, toda_ft - shorten_ft)
                asda_eff_lim = max(0.0, asda_ft - shorten_ft)
                clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
                tod_limit = tora_eff + clearway_allow
                over_asd  = max(0.0, asd_100 - asda_eff_lim)
                over_cont = max(0.0, (agd_100 * OEI_AGD_FACTOR) - min(tod_limit, toda_eff))
                significant_fail = (over_asd > 300.0) or (over_cont > 300.0)
                if (gw_lbs < 58000.0) and (not significant_fail):
                    n1 = 100.0
                    thrust_text = "MIL"
                else:
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

    # Final distances (AEO liftoff & ASD)
    asd_fin, agd_aeo_fin = distances_for(n1)

    # Regulatory check (default)
    ok_reg, req_reg, limiting_reg = field_ok(asd_fin, agd_aeo_fin, engine_out=True)

    # Auto-flap escalation if not OK at MAN and we didn't already escalate
    if flap_mode == "Auto-Select" and not ok_reg and flap_deg != 40:
        return compute_takeoff(perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                               field_elev_ft, slope_pct, shorten_ft,
                               oat_c, qnh_inhg, wind_speed, wind_dir_deg, wind_units, wind_policy,
                               gw_lbs, "FULL", "MIL", 100.0)

    avail = max(0.0, tora_ft - shorten_ft)
    agd_reg_fin = agd_aeo_fin * OEI_AGD_FACTOR

    return Result(
        v1=float(v1), vr=float(vr), v2=float(v2), vs=float(base.get("Vs_kt", np.nan)),
        flap_text=("MANEUVER" if flap_mode == "Auto-Select" else flap_mode),
        thrust_text=thrust_text, n1_pct=float(n1),
        asd_ft=float(asd_fin), agd_aeo_liftoff_ft=float(agd_aeo_fin), agd_reg_oei_ft=float(agd_reg_fin),
        req_ft=float(req_reg), avail_ft=float(avail), limiting=str(limiting_reg),
        hw_kn=float(hw), cw_kn=float(cw), notes=[]
    )

# ------------------------------ small helpers exposed for UI ------------------------------
def detect_length_text_to_ft(text: str) -> float:
    """Parse user entry for runway length (supports ft or NM). If numeric < 10, assume NM."""
    s = str(text).strip().lower()
    if s.endswith("nm"):
        val = float(s[:-2].strip())
        return val * 6076.12
    if s.endswith("ft"):
        val = float(s[:-2].strip())
        return val
    # unitless: try float
    val = float(s)
    if val < 10.0:
        return val * 6076.12
    return val

def detect_elev_text_to_ft(text: str) -> float:
    return float(str(text).strip())
