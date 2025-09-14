# f14_takeoff_core.py
# Core math & utilities for the DCS F-14B takeoff performance app.
# Safe for Python 3.9+ (Streamlit Cloud default).

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import re

import numpy as np
import pandas as pd

# ============================== Public constants used by the app ==============================
# Fraction of the AEO liftoff distance that approximates Vr ground roll (empirical proxy)
AEO_VR_FRAC = 0.86        # UP / MAN flaps
AEO_VR_FRAC_FULL = 0.82   # FULL flaps

# Calibration factors (set by the app before compute_takeoff via session_state)
AEO_CAL_FACTOR: float = 1.00     # scale for AEO liftoff distance
OEI_AGD_FACTOR: float = 1.20     # scale to go from AEO liftoff to OEI regulatory continue distance

# Thrust (per engine) proxies (approximate)
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}

# ============================== Data model returned to the app ==============================
@dataclass
class Result:
    v1: float
    vr: float
    v2: float
    vs: float

    flap_text: str
    thrust_text: str
    n1_pct: float

    # distances (feet)
    asd_ft: float                # accelerate-stop
    agd_aeo_liftoff_ft: float    # all-engines liftoff distance to 35 ft
    agd_reg_oei_ft: float        # regulatory engine-out continue distance

    # convenience
    req_ft: float
    avail_ft: float
    limiting: str

    # wind components (kts)
    hw_kn: float
    cw_kn: float

    # notes / flags
    notes: Tuple[str, ...]
    aeo_grad_ok: bool

# ============================== CSV loader ==============================
def load_perf_csv(path: str) -> pd.DataFrame:
    """
    Load the performance CSV. We don't rely on specific columns here,
    but we keep the DataFrame for future model tuning and AB ratio estimates.
    """
    df = pd.read_csv(path)
    # best-effort dtype cleanups if present
    for c in ("flap_deg", "gw_lbs", "press_alt_ft", "oat_c"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "thrust" in df.columns:
        df["thrust"] = df["thrust"].astype(str)
    return df

# ============================== Small helpers exposed to UI ==============================
def hpa_to_inhg(hpa: float) -> float:
    # 1 inHg = 33.8639 hPa
    return float(hpa) / 33.8639

_wind_pat = re.compile(r"^\s*(\d{1,3})\s*[@/ ]\s*(\d{1,3})\s*$")
def parse_wind_entry(s: str, units: str = "kts") -> Optional[Tuple[float, float]]:
    """
    Parse 'DDD@SS', 'DDD/SS', or 'DDD SS'. Returns (dir_deg, speed) with speed in input units.
    """
    if not s:
        return None
    m = _wind_pat.match(str(s))
    if not m:
        return None
    d = float(int(m.group(1)) % 360)
    spd = float(int(m.group(2)))
    if units not in ("kts", "m/s"):
        units = "kts"
    if units == "m/s":
        # store in m/s, app passes units separately
        return (d, spd)
    return (d, spd)

_len_pat = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")
def detect_length_text_to_ft(text: str) -> float:
    """
    Accepts values like: '7200', '7200ft', '1.2nm', '1.2 NM'
    """
    m = _len_pat.match(str(text))
    if not m:
        raise ValueError("unparsable length")
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("", "ft", "feet"):
        return val
    if unit in ("nm", "nmi", "nauticalmile", "nauticalmiles"):
        return val * 6076.12
    if unit in ("m", "meter", "meters"):
        return val * 3.28084
    raise ValueError("unknown length unit")

def detect_elev_text_to_ft(text: str) -> float:
    m = _len_pat.match(str(text))
    if not m:
        raise ValueError("unparsable elevation")
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("", "ft", "feet"):
        return val
    if unit in ("m", "meter", "meters"):
        return val * 3.28084
    raise ValueError("unknown elevation unit")

def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    """
    Very simple trim proxy: around ~6.5 ANU near 60k with MAN, bias by flap & weight.
    """
    base = 6.5
    base += (gw_lbs - 60000.0) / 20000.0 * 1.2  # +1.2 ANU per +20k
    if flap_deg == 0:
        base -= 0.6
    elif flap_deg == 40:
        base += 0.6
    return float(round(base, 1))

# ============================== Wind, density, and internal physics proxies ==============================
def _wind_components(runway_hdg_deg: float, wind_dir_deg: float, wind_spd: float) -> Tuple[float, float]:
    """
    Returns (headwind_kn, crosswind_kn). Positive headwind, positive crosswind from the right.
    """
    # angle from runway TO wind
    ang = math.radians((wind_dir_deg - runway_hdg_deg + 360.0) % 360.0)
    hw = wind_spd * math.cos(ang)
    cw = wind_spd * math.sin(ang)
    return (hw, cw)

def _isa_density_ratio(oat_c: float, qnh_inhg: float, elev_ft: float) -> float:
    """
    Crude density ratio sigma = rho/rho0, combining pressure influence and temperature.
    """
    # Pressure altitude approx from QNH and field elev
    # PA ≈ elev + 1000 * (29.92 - QNH)
    pa_ft = float(elev_ft) + 1000.0 * (29.92 - float(qnh_inhg))
    # ISA temp at PA
    isa_t_c = 15.0 - 1.98 * (pa_ft / 1000.0)
    t_ratio = (273.15 + float(oat_c)) / (273.15 + isa_t_c)
    # Simple sigma proxy (temp dominates; pressure baked into ISA lapse)
    sigma = 1.0 / t_ratio
    return max(0.6, min(1.3, sigma))

# ============================== Performance core (stable proxies) ==============================
def _flap_label_to_deg(label: str) -> int:
    s = (label or "").strip().upper()
    if s.startswith("UP"):
        return 0
    if s.startswith("FULL"):
        return 40
    return 20  # MANEUVER

def _n_engines_from_thrust_mode(thrust_mode: str) -> int:
    # we model both engines for AEO, 1 for OEI continue checks
    return 2

def _total_thrust_lbf(n1pct: float, thrust_mode: str) -> float:
    """
    Two engines available (AEO) thrust proxy.
    - For "AB": use AB static
    - For "MIL" or "DERATE": scale MIL linearly by N1%
    """
    if thrust_mode == "AB":
        per_eng = ENGINE_THRUST_LBF["AB"]
        return 2.0 * per_eng
    # Manual Derate / MIL share MIL basis; N1 at or below 100
    per_eng = ENGINE_THRUST_LBF["MIL"] * max(0.0, float(n1pct)) / 100.0
    return 2.0 * per_eng

def _oei_thrust_lbf(n1pct: float, thrust_mode: str) -> float:
    """One-engine-remaining thrust proxy for OEI continue segment."""
    if thrust_mode == "AB":
        return ENGINE_THRUST_LBF["AB"]
    return ENGINE_THRUST_LBF["MIL"] * max(0.0, float(n1pct)) / 100.0

def _stall_speed_kt(gw_lbs: float, flap_deg: int) -> float:
    """
    Very simple stall speed model: Vs ∝ sqrt(W). Anchor MAN(20) at ~125 kt @ 60k GW.
    Flap effects: UP +8%, FULL -8% vs MAN.
    """
    base_vs_man_60k = 125.0
    vs_man = base_vs_man_60k * math.sqrt(max(gw_lbs, 1.0) / 60000.0)
    if flap_deg == 0:
        return vs_man * 1.08
    if flap_deg == 40:
        return vs_man * 0.92
    return vs_man

def _v_speeds(gw_lbs: float, flap_deg: int) -> Tuple[float, float, float, float]:
    """
    Return (V1, Vr, V2, Vs) in knots.
    V2 ≈ 1.20*Vs; Vr ≈ V2-10; V1 ≈ Vr-5 with floor protections.
    """
    vs = _stall_speed_kt(gw_lbs, flap_deg)
    v2 = max(vs * 1.20, vs + 10.0)
    vr = max(v2 - 10.0, vs + 5.0)
    v1 = max(vr - 5.0, vs + 1.0)
    return (v1, vr, v2, vs)

def _ground_dist_aeo_ft(v_liftoff_kt: float, total_thrust_lbf: float, gw_lbs: float, sigma: float) -> float:
    """
    Extremely stable proxy for AEO liftoff distance, increasing with V^2 and weight,
    decreasing with thrust and density. Tuned to return plausible values (2-8 kft).
    """
    # Convert knots to ft/s
    v_fps = v_liftoff_kt * 1.68781
    # "Acceleration" proxy ~ T/W scaled by density
    t_over_w = (total_thrust_lbf / max(gw_lbs, 1.0))
    # Base K chosen to make outputs realistic
    K = 0.065
    dist = K * (gw_lbs / max(total_thrust_lbf, 1.0)) * (v_fps ** 2) / max(0.35 * sigma + 0.15, 0.05)
    # add a small floor based on weight to prevent unrealistically tiny values
    dist += (gw_lbs - 40000.0) / 200.0  # +50 ft per +10k
    return max(500.0, dist)

def _accelerate_stop_ft(agd_aeo_ft: float, v1_kt: float, sigma: float) -> float:
    """
    ASD proxy ≈ accelerate-to-V1 + reject + braking.
    Use AEO ground distance as a base + braking term that grows with V1^2 and density.
    """
    v_fps = v1_kt * 1.68781
    braking = (v_fps ** 2) / max(50.0 * sigma, 25.0)  # simple kinetic energy dump
    return agd_aeo_ft * 0.85 + braking * 1.25

def _apply_wind_policy(dist_ft: float, hw_kn: float, policy: str) -> float:
    """
    Apply '50/150' head/tailwind adjustments to distances.
    - Headwind credit: 0.5% per kt
    - Tailwind penalty: 1.5% per kt
    """
    if str(policy).upper().startswith("50"):
        if hw_kn >= 0:
            return dist_ft * max(0.6, 1.0 - 0.005 * min(hw_kn, 40.0))
        else:
            return dist_ft * (1.0 + 0.015 * min(abs(hw_kn), 20.0))
    return dist_ft

def compute_aeo_normal_climb_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    """
    Quick AEO gradient guardrail: require ~200 ft/nm (~3.3%) net gradient at cleanup.
    T/W - D/W >= 0.033; D/W proxy varies by flap.
    """
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_aeo = ENGINE_THRUST_LBF["MIL"] * (n1pct / 100.0) * 2.0
    grad = (t_aeo / max(gw_lbs, 1.0)) - drag_over_w
    return grad >= 0.033

# ------------------------------ OEI guardrail (added to fix import errors) ------------------------------
def compute_oei_second_segment_ok(gw_lbs: float, n1pct: float, flap_deg: int) -> bool:
    """Check OEI second-segment climb meets 2.4% net gradient."""
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    t_oei = ENGINE_THRUST_LBF["MIL"] * (n1pct / 100.0)  # one engine remaining
    gradient_net = (t_oei / max(gw_lbs, 1.0)) - drag_over_w
    return gradient_net >= 0.024

# ============================== AB estimator from table (optional) ==============================
def estimate_ab_multiplier(perfdb: pd.DataFrame, flap_deg: int) -> float:
    """
    Estimate the ratio of AEO liftoff distance in AB vs MIL for a given flap setting,
    using rows in the perf table if available. Fallback: 0.78 (AB ~22% shorter).
    """
    try:
        if perfdb is None or perfdb.empty:
            return 0.78
        sub_mil = perfdb[(perfdb.get("flap_deg") == flap_deg) & (perfdb.get("thrust") == "MIL")]
        sub_ab  = perfdb[(perfdb.get("flap_deg") == flap_deg) & (perfdb.get("thrust") == "AFTERBURNER")]
        if sub_mil.empty or sub_ab.empty:
            return 0.78
        # Look for any distance-like columns
        cand_cols = [c for c in perfdb.columns if c.lower().endswith("_ft") or "distance" in c.lower()]
        if not cand_cols:
            return 0.78
        col = cand_cols[0]
        # approximate ratio using medians
        r = (sub_ab[col].median() / max(sub_mil[col].median(), 1.0))
        if 0.5 < r < 1.0:
            return float(r)
        return 0.78
    except Exception:
        return 0.78

# ============================== Climb recommendation (wrapper used by app) ==============================
def recommend_climb(gw_lbs: float,
                    flap_after_cleanup: str = "UP",
                    accel_alt_ft_agl: float = 1000.0,
                    policy: str = "conservative") -> Tuple[float, float]:
    """
    Minimal wrapper that returns (N1%, climb IAS) for the merged app.
    """
    flap_deg = _flap_label_to_deg(flap_after_cleanup)
    # target N1 to clear AEO gradient with margin
    # 200 ft/nm (3.3%); add 1% or 2% depending on policy
    margin = 2.0 if (policy or "conservative").lower().startswith("conserv") else 1.0
    # invert T/W - D/W >= 0.033 for N1
    drag_over_w = {0: 0.06, 20: 0.08, 40: 0.10}.get(flap_deg, 0.08)
    rhs = 0.033 + drag_over_w
    n1 = (rhs * max(gw_lbs, 1.0)) / (2.0 * ENGINE_THRUST_LBF["MIL"]) * 100.0
    n1_cmd = max(90.0, min(100.0, n1 + margin))
    # climb IAS: 250 below 10k, otherwise 300 as app shows; we return the immediate target
    climb_ias = 250.0
    return (float(int(round(n1_cmd))), float(int(round(climb_ias))))

# ============================== Main entrypoint: compute_takeoff ==============================
def compute_takeoff(perfdb: pd.DataFrame,
                    runway_hdg_deg: float,
                    tora_ft: float, toda_ft: float, asda_ft: float,
                    elev_ft: float, slope_pct: float, shorten_ft: float,
                    oat_c: float, qnh_inhg: float,
                    wind_spd: float, wind_dir: float, wind_units: str, wind_policy: str,
                    gw_lbs: float, flap_mode: str,
                    thrust_mode: str, n1pct: float) -> Result:
    """
    Core performance routine. Uses stable proxies so the UI is responsive while you
    continue tuning the CSV-based model. Applies wind credits/penalties per 50/150.
    """

    # Normalize/guard inputs
    flap_mode = str(flap_mode or "MANEUVER").upper()
    thrust_mode = str(thrust_mode or "MIL").upper()
    wind_units = (wind_units or "kts").lower()
    wind_spd_kn = float(wind_spd if wind_units == "kts" else float(wind_spd) * 1.94384)

    # FULL cannot be derated in this model
    if flap_mode.startswith("FULL") and thrust_mode == "DERATE":
        thrust_mode = "MIL"
        n1pct = 100.0

    flap_deg = _flap_label_to_deg(flap_mode)
    sigma = _isa_density_ratio(oat_c, qnh_inhg, elev_ft)

    # Wind components (kts)
    hw_kn, cw_kn = _wind_components(float(runway_hdg_deg), float(wind_dir), float(wind_spd_kn))

    # V-speeds
    v1, vr, v2, vs = _v_speeds(float(gw_lbs), flap_deg)

    # Thrust models
    tot_thrust = _total_thrust_lbf(float(n1pct), thrust_mode)
    oei_thrust = _oei_thrust_lbf(float(n1pct), thrust_mode)

    # AEO liftoff distance proxy (to 35 ft)
    agd_aeo = _ground_dist_aeo_ft(v2, tot_thrust, float(gw_lbs), sigma) * float(AEO_CAL_FACTOR)

    # Optional AB multiplier if thrust_mode == AB and we have a table to learn from
    if thrust_mode == "AB":
        try:
            mult = estimate_ab_multiplier(perfdb, (20 if flap_deg == 0 else flap_deg))
            agd_aeo *= max(0.55, min(1.0, mult))
        except Exception:
            pass

    # Adjust for runway slope (mild effect): uphill increases distances, downhill decreases
    slope_adj = 1.0 + float(slope_pct) * 0.01 * 0.10  # 1% slope -> ~10% change
    agd_aeo *= max(0.7, min(1.3, slope_adj))

    # Apply wind policy (headwind credit / tailwind penalty)
    agd_aeo = _apply_wind_policy(agd_aeo, hw_kn, wind_policy)

    # OEI regulatory continue distance (conservative factor vs AEO)
    agd_reg = float(OEI_AGD_FACTOR) * agd_aeo

    # Accelerate-stop distance
    asd = _accelerate_stop_ft(agd_aeo, v1, sigma)
    asd = _apply_wind_policy(asd, hw_kn, wind_policy)

    # Declared distances after shorten
    tora_eff = max(0.0, float(tora_ft) - float(shorten_ft))
    toda_eff = max(0.0, float(toda_ft) - float(shorten_ft))
    asda_eff = max(0.0, float(asda_ft) - float(shorten_ft))

    # Clearway allowance (typical rule: max 50% of TORA and cannot exceed TODA-TORA)
    clearway_allow = min(tora_eff * 0.5, max(0.0, toda_eff - tora_eff))
    tod_limit = tora_eff + clearway_allow

    # Limiting (Regulatory mode assumed here; the app will also compute AEO-practical)
    req_reg = max(asd, agd_reg)
    limiting = "ASD (stop)" if asd >= agd_reg else "Engine-out continue"

    # Label for thrust
    if thrust_mode == "AB":
        thrust_text = "AFTERBURNER"
    elif thrust_mode == "DERATE":
        thrust_text = f"DERATE {int(round(n1pct))}%"
    else:
        # MIL
        thrust_text = "MIL"

    # Gradient sanity (AEO 200 ft/nm proxy)
    aeo_grad_ok = compute_aeo_normal_climb_ok(float(gw_lbs), float(n1pct if thrust_mode != "AB" else 100.0), flap_deg)

    # Build result (final clean return — replaces any mangled fragments)
    notes: Tuple[str, ...] = tuple()

    return Result(
        v1=float(v1), vr=float(vr), v2=float(v2), vs=float(vs),
        flap_text=("MANEUVER" if flap_mode.startswith("AUTO") else ("UP" if flap_deg == 0 else ("FULL" if flap_deg == 40 else "MANEUVER"))),
        thrust_text=thrust_text, n1_pct=float(100.0 if thrust_mode == "AB" else n1pct),
        asd_ft=float(asd), agd_aeo_liftoff_ft=float(agd_aeo), agd_reg_oei_ft=float(agd_reg),
        req_ft=float(req_reg), avail_ft=float(tora_eff), limiting=str(limiting),
        hw_kn=float(hw_kn), cw_kn=float(cw_kn), notes=notes, aeo_grad_ok=bool(aeo_grad_ok)
    )
