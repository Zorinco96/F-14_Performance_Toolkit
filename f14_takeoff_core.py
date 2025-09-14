# f14_takeoff_core.py
# Core performance model for DCS F-14B Takeoff Performance App
# - Auto-Select derate (90% floor, snap-to-floor, 0.5% tolerance)
# - Light-weight anti-escalation below 58k
# - OEI (2.4%) and AEO (200 ft/nm) climb-gradient flags
# - Wind/slope & hot/high shaping
# - load_perf_csv returns a pandas DataFrame (required by the app)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math
import csv
import pandas as pd

# ---------------------------------------------------------------------------
# Constants / Tunables
# ---------------------------------------------------------------------------

ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}  # per engine (approx)

# Proxy drag/weight by flap
DRAG_OVER_W = {0: 0.06, 20: 0.08, 40: 0.10}

# Vr fraction vs AEO liftoff-to-35ft
AEO_VR_FRAC = 0.88
AEO_VR_FRAC_FULL = 0.82

# OEI regulatory continue factor (AEO liftoff → OEI continue)
OEI_AGD_FACTOR = 1.20   # app can overwrite to 1.15 via "DCS-calibrated"

# Overall AEO calibration (app can overwrite at runtime)
AEO_CAL_FACTOR = 1.00

# Distance ∝ 1 / N1^alpha (derate shaping)
ALPHA_N1_DIST = 2.0

# Out-of-grid DA shaping (softer extrapolation)
DA_BETA = 0.80

# Derate floors by flap (FULL cannot derate)
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}

# Wind policy scaling (distance change per kt ≈ 0.5%, scaled)
WIND_FACTORS: Dict[str, Tuple[float, float]] = {
    "None": (1.0, 1.0),
    "50/150": (0.5, 1.5),
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Result:
    v1: float
    vr: float
    v2: float
    vs: float
    flap_text: str
    thrust_text: str
    n1_pct: float
    asd_ft: float
    agd_aeo_liftoff_ft: float
    agd_reg_oei_ft: float
    req_ft: float
    avail_ft: float
    limiting: str
    hw_kn: float
    cw_kn: float
    oei_grad_ok: bool
    aeo_grad_ok: bool
    notes: List[str]

# ---------------------------------------------------------------------------
# CSV loader — returns DataFrame (the app expects this!)
# ---------------------------------------------------------------------------

def load_perf_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        # graceful fallback if needed
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            return pd.DataFrame(rows)
        except Exception:
            # empty DF with expected columns (prevents crashes in Matrix tab)
            return pd.DataFrame(columns=[
                "model","flap_deg","thrust","gw_lbs","press_alt_ft","oat_c",
                "asd_ft","agd_aeo_ft","agd_oei_ft","v1","vr","v2","vs"
            ])

# ---------------------------------------------------------------------------
# Atmosphere / wind helpers
# ---------------------------------------------------------------------------

def hpa_to_inhg(hpa: float) -> float:
    return float(hpa) * 0.029529983

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    # PA ≈ field elev + (29.92 - QNH) * 1000
    return float(field_elev_ft + (29.92 - float(qnh_inhg)) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (float(h_ft) / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    # FAA rule: DA ≈ PA + 120 × (OAT − ISA)
    return float(pa_ft + 120.0 * (float(oat_c) - isa_temp_c_at_ft(pa_ft)))

def _sigma_from_da(da_ft: float) -> float:
    # simple ISA density ratio
    h_m = float(da_ft) * 0.3048
    T0 = 288.15; L = 0.0065; g = 9.80665; R = 287.05
    p0 = 101325.0; rho0 = 1.225
    T = max(150.0, T0 - L*h_m)
    p = p0 * (T/T0)**(g/(R*L))
    rho = p/(R*T)
    return float(max(0.2, min(1.5, rho/rho0)))

def _da_out_of_grid_scale(pa_ft: float, oat_c: float) -> float:
    da_act = density_altitude_ft(pa_ft, oat_c)
    da_ref = density_altitude_ft(min(pa_ft, 5000.0), min(oat_c, 30.0))
    sig_act = _sigma_from_da(da_act)
    sig_ref = _sigma_from_da(da_ref)
    return (sig_ref / max(1e-6, sig_act)) ** DA_BETA

def wind_components(speed_kn: float, dir_deg_from: float, rwy_heading_deg: float) -> Tuple[float, float]:
    # dir is FROM; return (headwind + / tailwind -, crosswind +R/-L)
    delta = math.radians((float(dir_deg_from) - float(rwy_heading_deg)) % 360.0)
    hw = float(speed_kn) * math.cos(delta)
    cw = float(speed_kn) * math.sin(delta)
    return hw, cw

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    # Uphill penalty (+20% per +1%)
    if slope_pct > 0:
        d *= (1.0 + 0.20 * float(slope_pct))
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)
    return max(d, 0.0)

def parse_wind_entry(entry: str, unit: str) -> Optional[Tuple[float, float]]:
    text = (entry or "").strip().replace("/", " ").replace("@", " ")
    parts = [p for p in text.split() if p]
    if len(parts) >= 2:
        try:
            d = float(parts[0]) % 360.0
            s = float(parts[1])
            if (unit or "kts").lower().startswith("m"):
                s *= 1.943844
            return (d, s)
        except Exception:
            return None
    return None

# ---------------------------------------------------------------------------
# Hot/high relief (empirical trim for extreme heat/DA)
# ---------------------------------------------------------------------------

def _hot_high_relief(oat_c: float, da_ft: float) -> float:
    # Reduce AEO liftoff ~10–12% near 47C and high DA; negligible near ISA
    if oat_c <= 30.0 or da_ft <= 3000.0:
        return 1.0
    dt = max(0.0, float(oat_c) - 30.0)
    dda = max(0.0, float(da_ft) - 3000.0)
    relief = 0.0025 * dt + 0.0000075 * dda
    return float(max(0.88, 1.0 - min(relief, 0.12)))

# ---------------------------------------------------------------------------
# Convenience: unit parsers used by app
# ---------------------------------------------------------------------------

def detect_length_text_to_ft(text: str) -> float:
    s = str(text).strip().lower()
    try:
        if s.endswith("nm"):
            return float(s[:-2].strip()) * 6076.12
        if s.endswith("ft"):
            return float(s[:-2].strip())
        val = float(s)
        if val < 10.0:  # ambiguous small number => assume NM
            return val * 6076.12
        return val
    except Exception:
        return 0.0

def detect_elev_text_to_ft(text) -> float:
    try:
        return float(text)
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Speeds / trim helpers
# ---------------------------------------------------------------------------

def _enforce_speed_floors(vs, v1, vr, v2, flap_deg: int) -> Tuple[float, float, float]:
    # Basic floors to avoid nonsense; tuned for F-14B UI
    try: vs = float(vs)
    except Exception: vs = 120.0 if flap_deg != 40 else 110.0
    try: v1 = float(v1)
    except Exception: v1 = vs * 1.10
    try: vr = float(vr)
    except Exception: vr = vs * 1.20
    try: v2 = float(v2)
    except Exception: v2 = vs * 1.30

    vmcg = 118.0 if flap_deg != 40 else 112.0
    v1 = max(v1, vmcg + 3.0, 0.95 * vr)
    vr = max(vr, v1 + 3.0, 1.05 * vs)
    v2 = max(v2, vr + 10.0, 1.18 * vs)
    return float(int(round(v1))), float(int(round(vr))), float(int(round(v2)))

def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (float(gw_lbs) - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------

def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    # OEI net 2.4%: (T/W - D/W) >= 0.024, with one engine at MIL*N1
    drag_over_w = DRAG_OVER_W.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (float(n1pct)/100.0)
    t_over_w = t_oei / max(float(gw_lbs), 1.0)
    gradient = t_over_w - drag_over_w
    return gradient >= 0.024

def compute_aeo_normal_climb_ok(gw_lbs: float, n1pct: float, flap_deg: int, gradient_req: float = 0.033) -> bool:
    # AEO (200 ft/nm ≈ 3.3%): (T/W - D/W) >= gradient_req with both engines at selected N1
    drag_over_w = DRAG_OVER_W.get(flap_deg, 0.08)
    t_aeo = 2.0 * ENGINE_THRUST_LBF["MIL"] * (float(n1pct)/100.0)
    t_over_w = t_aeo / max(float(gw_lbs), 1.0)
    gradient = t_over_w - drag_over_w
    return gradient >= float(gradient_req)

# ---------------------------------------------------------------------------
# AB estimator (used by app matrix if needed)
# ---------------------------------------------------------------------------

def estimate_ab_multiplier(perfdb: pd.DataFrame, flap_deg: int) -> float:
    """Return an approximate AB/MIL thrust ratio used for distance scaling."""
    try:
        return float(ENGINE_THRUST_LBF["AB"]) / float(ENGINE_THRUST_LBF["MIL"])
    except Exception:
        return 1.60

# ---------------------------------------------------------------------------
# Core compute (simplified physics + tunable shaping; can be swapped to tables later)
# ---------------------------------------------------------------------------

def compute_takeoff(
    perfdb: pd.DataFrame,             # kept for signature compatibility
    rwy_heading_deg: float,
    tora_ft: float,
    toda_ft: float,
    asda_ft: float,
    field_elev_ft: float,
    slope_pct: float,
    shorten_ft: float,
    oat_c: float,
    qnh_inhg: float,
    wind_speed: float,
    wind_dir_deg: float,
    wind_units: str,
    wind_policy: str,
    gw_lbs: float,
    flap_mode: str,
    thrust_mode: str,
    target_n1_pct: float,
) -> Result:

    # Atmosphere and wind
    pa = pressure_altitude_ft(field_elev_ft, qnh_inhg)
    da = density_altitude_ft(pa, oat_c)
    spd_kn = float(wind_speed) if (wind_units or "kts") == "kts" else float(wind_speed) * 1.943844
    hw, cw = wind_components(spd_kn, wind_dir_deg, rwy_heading_deg)

    notes: List[str] = []
    if hw < -10.0:
        notes.append("Tailwind component exceeds 10 kt — NOT AUTHORIZED.")
    if abs(cw) > 30.0:
        notes.append("Crosswind component exceeds 30 kt — NOT AUTHORIZED.")

    # Flaps
    flap_text = flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)
    use_flap = flap_deg

    # --- Base distances (calibrated simple law; table interpolation can replace this later) ---
    base_man = 0.08 * float(gw_lbs) + 1400.0      # e.g., 70k → ~7000 ft before scaling
    if flap_deg == 0:     base = base_man * 1.06
    elif flap_deg == 40:  base = base_man * 0.92
    else:                 base = base_man

    liftoff_factor = 1.38 + 0.12 * max(0.0, min(da/8000.0, 1.25))   # DA shaping
    aeo_liftoff_nom = base * liftoff_factor
    aeo_liftoff_nom *= _hot_high_relief(oat_c, da)                   # hot/high relief
    aeo_liftoff_nom *= _da_out_of_grid_scale(pa, oat_c)              # soft extrapolation
    asd_nom = aeo_liftoff_nom * 0.92                                 # crude ASD proxy

    # N1 multiplier (derate)
    def _mult_from_n1(n1pct: float) -> float:
        eff = max(0.90, min(1.00, float(n1pct)/100.0))
        return 1.0 / (eff ** ALPHA_N1_DIST)

    # Compose wind/slope
    def _wind_slope(d_ft: float) -> float:
        return apply_wind_slope(d_ft, slope_pct, hw, wind_policy)

    def distances_for(n1pct: float) -> Tuple[float, float]:
        m = _mult_from_n1(n1pct)
        asd = _wind_slope(asd_nom * m)
        agd_aeo = _wind_slope(aeo_liftoff_nom * m) * AEO_CAL_FACTOR
        return max(asd, 0.0), max(agd_aeo, 0.0)

    # Field checker
    def field_ok(asd_eff: float, agd_aeo_cont: float, engine_out: bool) -> Tuple[bool, float, str]:
        tora_eff = max(0.0, float(tora_ft) - float(shorten_ft))
        toda_eff = max(0.0, float(toda_ft) - float(shorten_ft))
        asda_eff = max(0.0, float(asda_ft) - float(shorten_ft))
        clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
        tod_limit = tora_eff + clearway_allow

        if engine_out:
            cont = agd_aeo_cont * OEI_AGD_FACTOR
            limiting = "ASD (stop)" if asd_eff >= cont else "Engine-out continue"
        else:
            cont = agd_aeo_cont
            limiting = "ASD (stop)" if asd_eff >= cont else "All-engines continue"

        req = max(asd_eff, cont)
        ok = (asd_eff <= asda_eff) and (cont <= tod_limit) and (cont <= toda_eff)
        return ok, req, limiting

    # Choose thrust / N1
    thrust_text = thrust_mode
    n1 = float(target_n1_pct)

    if thrust_mode in ("DERATE", "Manual Derate"):
        floor_pct = DERATE_FLOOR_BY_FLAP.get(use_flap, 0.90) * 100.0
        n1 = float(int(math.ceil(max(floor_pct, n1))))
        thrust_text = "DERATE" if n1 < 100.0 else "MIL"

    elif thrust_mode == "Auto-Select":
        # Can MIL (100%) pass Regulatory?
        asd_mil, agd_aeo_mil = distances_for(100.0)
        ok_mil, _, _ = field_ok(asd_mil, agd_aeo_mil, engine_out=True)

        if ok_mil:
            floor_pct = DERATE_FLOOR_BY_FLAP.get(use_flap, 0.90) * 100.0
            # Try floor directly (also check OEI gradient)
            asd_floor, agd_floor = distances_for(floor_pct)
            ok_floor, _, _ = field_ok(asd_floor, agd_floor, engine_out=True)
            ok_floor = ok_floor and compute_oei_second_segment_ok(gw_lbs, floor_pct, use_flap)

            if ok_floor:
                n1 = floor_pct
            else:
                lo, hi = floor_pct, 100.0
                for _ in range(18):
                    mid = (lo + hi) / 2.0
                    asd_m, agd_m = distances_for(mid)
                    ok_m, _, _ = field_ok(asd_m, agd_m, engine_out=True)
                    ok_m = ok_m and compute_oei_second_segment_ok(gw_lbs, mid, use_flap)
                    if ok_m: hi = mid
                    else:    lo = mid
                EPS = 0.5
                n1 = floor_pct if (hi - floor_pct) <= EPS else float(int(math.ceil(hi)))
            thrust_text = "DERATE" if n1 < 100.0 else "MIL"

        else:
            # MIL fails at current flap → consider FULL/MIL unless ≤58k and miss is tiny
            if use_flap != 40:
                asd_100, agd_100 = distances_for(100.0)
                tora_eff = max(0.0, float(tora_ft) - float(shorten_ft))
                toda_eff = max(0.0, float(toda_ft) - float(shorten_ft))
                asda_eff = max(0.0, float(asda_ft) - float(shorten_ft))
                clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
                tod_limit = tora_eff + clearway_allow
                over_asd  = max(0.0, asd_100 * 1.0 - asda_eff)
                over_cont = max(0.0, (agd_100 * OEI_AGD_FACTOR) - min(tod_limit, toda_eff))
                significant_fail = (over_asd > 300.0) or (over_cont > 300.0)
                if float(gw_lbs) < 58000.0 and not significant_fail:
                    n1 = 100.0
                    thrust_text = "MIL"
                else:
                    # escalate to FULL/MIL
                    return compute_takeoff(
                        perfdb, rwy_heading_deg, tora_ft, toda_ft, asda_ft,
                        field_elev_ft, slope_pct, shorten_ft, oat_c, qnh_inhg,
                        wind_speed, wind_dir_deg, wind_units, wind_policy,
                        gw_lbs, "FULL", "MIL", 100.0
                    )
            else:
                n1 = 100.0
                thrust_text = "MIL"

    elif thrust_mode == "MIL":
        n1 = 100.0
        thrust_text = "MIL"
    elif thrust_mode == "AB":
        n1 = 100.0
        thrust_text = "AFTERBURNER"

    # Final distances at chosen N1
    asd_fin, agd_aeo_fin = distances_for(n1)
    agd_reg_fin = agd_aeo_fin * OEI_AGD_FACTOR

    # Availability (TORA minus declared reduction)
    avail = max(0.0, float(tora_ft) - float(shorten_ft))

    # Gradient flags
    oei_grad_ok = compute_oei_second_segment_ok(gw_lbs, n1, use_flap)
    aeo_grad_ok = compute_aeo_normal_climb_ok(gw_lbs, n1, use_flap, 0.033)

    if thrust_mode in ("Manual Derate", "MIL", "AB") and not aeo_grad_ok:
        notes.append("Selected N1 fails AEO 200 ft/nm climb gradient. Increase N1 or reduce weight.")

    # Speeds (simple schedule + floors)
    vs_guess = 95.0 + (float(gw_lbs) - 50000.0) / 2500.0
    v2_guess = vs_guess * 1.20
    vr_guess = v2_guess - 10.0
    v1_guess = vr_guess - 5.0
    v1, vr, v2 = _enforce_speed_floors(vs_guess, v1_guess, vr_guess, v2_guess, use_flap)

    # Limiting (Regulatory default; app recomputes mode-specific margins)
    asd_eff = asd_fin
    cont_reg = agd_reg_fin
    limiting_reg = "ASD (stop)" if asd_eff >= cont_reg else "Engine-out continue"
    req_reg = max(asd_eff, cont_reg)

    return Result(
        v1=float(v1),
        vr=float(vr),
        v2=float(v2),
        vs=float(vs_guess),
        flap_text=("MANEUVER" if flap_mode == "Auto-Select" and use_flap == 20 else
                   ("UP" if use_flap == 0 else ("FULL" if use_flap == 40 else "MANEUVER"))),
        thrust_text=str(thrust_text),
        n1_pct=float(n1),
        asd_ft=float(asd_fin),
        agd_aeo_liftoff_ft=float(agd_aeo_fin),
        agd_reg_oei_ft=float(agd_reg_fin),
        req_ft=float(req_reg),
        avail_ft=float(avail),
        limiting=str(limiting_reg),
        hw_kn=float(hw),
        cw_kn=float(cw),
        oei_grad_ok=bool(oei_grad_ok),
        aeo_grad_ok=bool(aeo_grad_ok),
        notes=notes,
    )
