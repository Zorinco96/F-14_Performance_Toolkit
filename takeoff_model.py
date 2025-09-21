# ============================================================
# F-14 Performance Calculator for DCS World — Takeoff Model
# File: takeoff_model.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Callable
import math

# Optional external solver (if present)
try:
    from bfv1 import solve_bfv1 as _solve_bfv1_ext  # (asd_fn, todr_fn, v1_lo, v1_hi, tol)
except Exception:
    _solve_bfv1_ext = None

# ------------------------------------------------------------
# Inputs and simple atmosphere helpers
# ------------------------------------------------------------

@dataclass
class TOInputs:
    gw_lbs: float
    flap_deg: float
    thrust_pct: float            # commanded %RPM (already clamped to floor in core)
    oat_c: float
    qnh_inhg: float
    headwind_kt: float = 0.0
    runway_mu: float = 0.45      # dry concrete ~0.45 effective decel
    slope_percent: float = 0.0
    field_elevation_ft: float = 0.0

def _isa_sigma(oat_c: float, elev_ft: float) -> float:
    """Crude density ratio (sigma) for standard performance scaling."""
    # Simple linearized approximation near sea level
    # For overhaul: replace with your isa.py if desired
    T0 = 288.15
    a = -0.0065  # K/m
    p0 = 101325.0
    R = 287.05
    g = 9.80665
    z = elev_ft * 0.3048
    T = T0 + a*z
    p = p0 * (T/T0) ** (-g/(a*R))
    rho = p / (R * T)
    rho0 = p0 / (R * T0)
    return float(rho / rho0)

# ------------------------------------------------------------
# Distance models vs V1
# ------------------------------------------------------------

def asdr_at_v1(v1: float, I: TOInputs) -> float:
    """Accelerate-Stop Distance (raw, unfactored) as a function of V1.
    This is a simplified continuous model suitable for BF V1 solving.
    Notes:
      - No thrust reverse (Tomcat), decel via brakes + aero drag.
      - Headwind reduces ground speed; we treat it as a first-order reduction.
    """
    v_gs = max(v1 - I.headwind_kt, 60.0)  # kt ground speed at decision
    sigma = _isa_sigma(I.oat_c, I.field_elevation_ft)
    mass = I.gw_lbs * 0.453592 / 9.81  # slug-ish to kg
    # Accel phase proxy to reach V1: scale with mass, inversely with thrust
    accel_term = 0.015 * mass / max(I.thrust_pct, 1.0) * (v_gs ** 1.15)
    # Brake phase proxy: distance ~ v^2 / (2 * mu * g) with sigma drag assist
    mu = max(I.runway_mu, 0.25)
    brake_term = (v_gs ** 2) / max(2.0 * mu * 3.28, 1.0)  # feet; rough
    # Add aero drag braking benefit with higher density
    aero_bonus = 0.10 * brake_term * sigma
    s = accel_term + brake_term * (1.0 - 0.10 * sigma) - aero_bonus
    # Slope penalty (uphill -> longer stop)
    s *= (1.0 + 0.005 * I.slope_percent)
    return float(max(s, 0.0))

def todr_at_v1(v1: float, I: TOInputs) -> float:
    """Takeoff Distance Required (raw, unfactored) accelerate-go as a function of V1.
    Includes: accel to V1, continue to Vr/Vlof, and 35 ft screen height.
    This is a smooth proxy for BF solving; exact NATOPS replication is future work.
    """
    # Assume Vr ~ V1 + delta; delta shrinks as V1 increases
    delta = max(10.0, 20.0 - 0.2 * (v1 - 120.0))
    vr = v1 + delta
    v_lof = vr + 5.0
    sigma = _isa_sigma(I.oat_c, I.field_elevation_ft)
    headwind = max(I.headwind_kt, 0.0)
    # Accel terms scale inversely with thrust and directly with mass and (speed)^1.1
    mass = I.gw_lbs * 0.453592 / 9.81
    accel1 = 0.012 * mass / max(I.thrust_pct, 1.0) * (v1 ** 1.10)
    accel2 = 0.010 * mass / max(I.thrust_pct, 1.0) * (max(v_lof, v1+1.0) ** 1.08)
    rotate = 250.0  # ft
    # Airborne to 35ft: scale with (weight/thrust) and density
    airborne = 900.0 * (mass / max(I.thrust_pct, 1.0)) / max(sigma, 0.6)
    # Headwind reduces ground run components modestly
    hw_factor = max(0.85, 1.0 - headwind / 60.0)
    s = (accel1 + accel2 + rotate) * hw_factor + airborne
    # Slope penalty for uphill
    s *= (1.0 + 0.004 * I.slope_percent)
    return float(max(s, 0.0))

# ------------------------------------------------------------
# Balanced-field V1 solver wrapper
# ------------------------------------------------------------

def solve_balanced_v1(I: TOInputs,
                      v1_lo: float = 100.0,
                      v1_hi: float = 170.0,
                      tol: float = 0.25) -> Tuple[float, float, float, bool]:
    """Find V1 such that ASDR(V1) ~= TODR(V1).
    Returns (v1*, asd_ft_raw, todr_ft_raw, converged).
    """
    asd_fn = lambda v: asdr_at_v1(v, I)
    tod_fn = lambda v: todr_at_v1(v, I)

    if _solve_bfv1_ext is not None:
        return _solve_bfv1_ext(asd_fn, tod_fn, v1_lo, v1_hi, tol)

    # Internal bisection fallback
    lo, hi = float(v1_lo), float(v1_hi)
    f_lo = asd_fn(lo) - tod_fn(lo)
    f_hi = asd_fn(hi) - tod_fn(hi)
    if f_lo == 0:
        return lo, asd_fn(lo), tod_fn(lo), True
    if f_hi == 0:
        return hi, asd_fn(hi), tod_fn(hi), True
    # If no sign change, pick the better end
    if f_lo * f_hi > 0:
        v_best = lo if abs(f_lo) < abs(f_hi) else hi
        return v_best, asd_fn(v_best), tod_fn(v_best), False
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = asd_fn(mid) - tod_fn(mid)
        if abs(hi - lo) < tol:
            return mid, asd_fn(mid), tod_fn(mid), True
        if f_mid == 0:
            return mid, asd_fn(mid), tod_fn(mid), True
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    mid = 0.5 * (lo + hi)
    return mid, asd_fn(mid), tod_fn(mid), False

# ------------------------------------------------------------
# Convenience facade used by core
# ------------------------------------------------------------

def distances_vs_v1(inputs: Dict[str, Any],
                    v1: float) -> Tuple[float, float]:
    """Return (ASDR_raw_ft, TODR_raw_ft) for a given V1 and inputs dict."""
    I = TOInputs(
        gw_lbs=float(inputs.get("gw_lbs", 60000)),
        flap_deg=float(inputs.get("flap_deg", 0)),
        thrust_pct=float(inputs.get("thrust_pct", 100)),
        oat_c=float(inputs.get("oat_c", 15)),
        qnh_inhg=float(inputs.get("qnh_inhg", 29.92)),
        headwind_kt=float(inputs.get("headwind_kt", 0.0)),
        runway_mu=float(inputs.get("runway_mu", 0.45)),
        slope_percent=float(inputs.get("slope_percent", 0.0)),
        field_elevation_ft=float(inputs.get("field_elevation_ft", 0.0)),
    )
    return asdr_at_v1(v1, I), todr_at_v1(v1, I)

def solve_bfv1(inputs: Dict[str, Any],
               v1_bounds: Tuple[float, float] = (100.0, 170.0),
               tol: float = 0.25) -> Dict[str, Any]:
    """Wrapper that solves for balanced V1 and returns a small dict payload.    Keys: v1, asd_ft_raw, todr_ft_raw, converged"""
    I = TOInputs(
        gw_lbs=float(inputs.get("gw_lbs", 60000)),
        flap_deg=float(inputs.get("flap_deg", 0)),
        thrust_pct=float(inputs.get("thrust_pct", 100)),
        oat_c=float(inputs.get("oat_c", 15)),
        qnh_inhg=float(inputs.get("qnh_inhg", 29.92)),
        headwind_kt=float(inputs.get("headwind_kt", 0.0)),
        runway_mu=float(inputs.get("runway_mu", 0.45)),
        slope_percent=float(inputs.get("slope_percent", 0.0)),
        field_elevation_ft=float(inputs.get("field_elevation_ft", 0.0)),
    )
    v1, asd, tod, ok = solve_balanced_v1(I, v1_lo=v1_bounds[0], v1_hi=v1_bounds[1], tol=tol)
    return {"v1": round(v1,1), "asd_ft_raw": float(asd), "todr_ft_raw": float(tod), "converged": bool(ok)}
