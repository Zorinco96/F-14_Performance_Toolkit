# ============================================================
# Balanced-Field V1 Solver (Bisection Utility)
# File: bfv1.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
from typing import Callable, Tuple

def solve_bfv1(asd_fn: Callable[[float], float],
               todr_fn: Callable[[float], float],
               v1_lo: float, v1_hi: float, tol: float = 0.25) -> Tuple[float, float, float, bool]:
    """Solve for V1 where asd_fn(V1) ~= todr_fn(V1) using bisection.
    Returns (v1*, asd, todr, converged).
    If no sign change is present, returns best endpoint.
    """
    lo, hi = float(v1_lo), float(v1_hi)
    f_lo = asd_fn(lo) - todr_fn(lo)
    f_hi = asd_fn(hi) - todr_fn(hi)
    if f_lo == 0:
        return lo, asd_fn(lo), todr_fn(lo), True
    if f_hi == 0:
        return hi, asd_fn(hi), todr_fn(hi), True
    if f_lo * f_hi > 0:
        v_best = lo if abs(f_lo) < abs(f_hi) else hi
        return v_best, asd_fn(v_best), todr_fn(v_best), False
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = asd_fn(mid) - todr_fn(mid)
        if abs(hi - lo) < tol:
            return mid, asd_fn(mid), todr_fn(mid), True
        if f_mid == 0:
            return mid, asd_fn(mid), todr_fn(mid), True
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    mid = 0.5 * (lo + hi)
    return mid, asd_fn(mid), todr_fn(mid), False
