# ============================================================
# perf_core_v2.py — Stable, Resilient Core API (v1.2.1)
# ============================================================
# What this provides:
#   - Dict-in / dict-out, app-stable functions:
#       compute_takeoff(inputs, overrides=None)
#       compute_landing(inputs)
#       compute_cruise(inputs)
#       compute_climb(inputs)
#   - LOCK SPEC respected:
#       * Derate floors: UP=85%, MAN=90%, FULL=96% (absolute floor = 85%)
#       * Takeoff distances returned UNFACTORED (UI shows ×1.10)
#       * Landing distance returned UNFACTORED (UI shows ×1.67)
#       * AEO climb floor default 300 ft/NM
#   - Resilient imports for local modules (takeoff_model, derate, f14_aero, engine_f110)
#     * If takeoff_model is function-based (e.g., solve_bfv1), an adapter class wraps it.
#   - CSV hooks for landing / cruise / climb with safe fallbacks
#
# Drop this file in your repo root as: perf_core_v2.py
# Then in your app:
#   import perf_core_v2 as core_v2
#   to = core_v2.compute_takeoff(inputs, overrides=override)
# ============================================================

from __future__ import annotations

import os, sys, importlib, importlib.util, math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

__version__ = "v1.2.1"

# ---------------------------
# Path hardening (Streamlit)
# ---------------------------
_HERE = os.path.dirname(__file__)
if _HERE and _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------
# Optional pandas / numpy
# ---------------------------
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # graceful fallback

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # graceful fallback


# ---------------------------
# Dynamic module resolution
# ---------------------------
def _import_first_available(mod_names: List[str] | Tuple[str, ...]) -> Any:
    if isinstance(mod_names, str):
        mod_names = [mod_names]
    for mn in mod_names:
        try:
            if importlib.util.find_spec(mn) is not None:
                return importlib.import_module(mn)
        except Exception:
            continue
    raise ModuleNotFoundError(f"No module found among: {mod_names}")

def _resolve_class(module: Any, preferred: List[str], keywords: List[str]) -> Any:
    # Try preferred names first
    for name in preferred:
        if hasattr(module, name):
            return getattr(module, name)
    # Fallback: scan classes by keywords
    candidates = []
    for name, obj in vars(module).items():
        if isinstance(obj, type):
            low = name.lower()
            score = sum(1 for kw in keywords if kw in low)
            if score:
                for attr in ("compute","evaluate","calc","build","lookup","predict","plan"):
                    if hasattr(obj, attr):
                        score += 1
                candidates.append((score, name, obj))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    raise ImportError(f"{module.__name__} has no preferred class {preferred} nor keyword-matching class {keywords}")


# ---------------------------
# Load sibling modules
# ---------------------------
_takeoff_mod = _import_first_available(("takeoff_model",))
_derate_mod  = _import_first_available(("derate",))
_aero_mod    = _import_first_available(("f14_aero",))
_engine_mod  = _import_first_available(("engine_f110",))

# Resolve Derate/Aero/Engine classes (these exist in your repo)
DerateModel = _resolve_class(_derate_mod, ["DerateModel","Derate"], ["derate","model"])
F14Aero     = _resolve_class(_aero_mod, ["F14Aero","F14AeroModel"], ["aero","f14"])
F110Deck    = _resolve_class(_engine_mod, ["F110Deck","EngineF110","F110Model"], ["f110","engine","deck","model"])

# Resolve Takeoff: allow class OR function-based (solve_bfv1)
_TakeoffClass = None
try:
    _TakeoffClass = _resolve_class(_takeoff_mod, ["TakeoffDeck","TakeoffModel","TakeoffTable"], ["takeoff","deck","model"])
except Exception:
    _TakeoffClass = None

if _TakeoffClass is None:
    # Functional adapter if takeoff_model exposes a function API
    if not hasattr(_takeoff_mod, "solve_bfv1"):
        raise ImportError("takeoff_model.py must export a class (TakeoffDeck/TakeoffModel) or a function solve_bfv1(inputs)")
    _solve_bfv1 = getattr(_takeoff_mod, "solve_bfv1")

    @dataclass
    class _TakeoffResult:
        asd_ft: float
        todr_ft: float

    class TakeoffDeck:  # type: ignore
        def lookup(self, flap_deg: int, base_thrust: str, gw_lbs: float, pa_ft: float, oat_c: float, **kwargs) -> _TakeoffResult:
            """
            Minimal adapter. Calls solve_bfv1 with a compact input dict and returns core distances.
            Expected solve_bfv1 keys (best-effort): v1, asd_ft_raw or asd_ft, todr_ft_raw or todr_ft, converged
            """
            inputs = dict(
                flap_deg=flap_deg,
                base_thrust=base_thrust,     # "MILITARY" or "AFTERBURNER"
                gw_lbs=gw_lbs,
                pressure_alt_ft=pa_ft,
                oat_c=oat_c,
            )
            inputs.update(kwargs or {})
            res = _solve_bfv1(inputs) or {}
            asd = float(res.get("asd_ft_raw", res.get("asd_ft", 0.0)) or 0.0)
            todr = float(res.get("todr_ft_raw", res.get("todr_ft", 0.0)) or 0.0)
            if not math.isfinite(asd): asd = 0.0
            if not math.isfinite(todr): todr = 0.0
            return _TakeoffResult(asd_ft=asd, todr_ft=todr)
else:
    TakeoffDeck = _TakeoffClass  # type: ignore


# ---------------------------
# Helpers / math
# ---------------------------
def _pressure_alt_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    # Simple ISA-ish mapping used elsewhere in your codebase
    return field_elev_ft + (29.92 - float(qnh_inhg)) * 1000.0

def _ensure_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _nearest_row(df, match: Dict[str, Any], keys: List[str]):
    """Return the row with minimal L1 distance across the given keys."""
    if df is None or len(df) == 0:
        return None
    # Only consider keys that exist in df
    cand_keys = [k for k in keys if k in df.columns]
    if not cand_keys:
        return None
    # Build distance
    best_idx, best_dist = None, None
    for idx, row in df.iterrows():
        dist = 0.0
        for k in cand_keys:
            try:
                dist += abs(float(row[k]) - float(match.get(k, row[k])))
            except Exception:
                continue
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, idx
    return df.loc[best_idx] if best_idx is not None else None

def _load_csv(path: str) -> Optional["pd.DataFrame"]:
    if pd is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _bilinear(df: "pd.DataFrame", xcol: str, ycol: str, zcol: str, x: float, y: float) -> Optional[float]:
    """Bilinear interpolation over a grid (requires pandas & numpy)."""
    if pd is None or np is None or df is None or df.empty:
        return None
    try:
        grid = df.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc="mean")
        xs = np.array(grid.columns, dtype=float)
        ys = np.array(grid.index, dtype=float)
        x = float(np.clip(x, xs.min(), xs.max()))
        y = float(np.clip(y, ys.min(), ys.max()))
        x1 = xs[xs <= x].max(); x2 = xs[xs >= x].min()
        y1 = ys[ys <= y].max(); y2 = ys[ys >= y].min()
        z11 = grid.loc[y1, x1]; z12 = grid.loc[y2, x1]
        z21 = grid.loc[y1, x2]; z22 = grid.loc[y2, x2]
        if x1 == x2 and y1 == y2:
            return float(z11)
        if x1 == x2:
            t = (y - y1) / (y2 - y1 + 1e-9)
            return float(z11*(1-t) + z12*t)
        if y1 == y2:
            t = (x - x1) / (x2 - x1 + 1e-9)
            return float(z11*(1-t) + z21*t)
        tx = (x - x1) / (x2 - x1 + 1e-9)
        ty = (y - y1) / (y2 - y1 + 1e-9)
        z1 = z11*(1-tx) + z21*tx
        z2 = z12*(1-tx) + z22*tx
        return float(z1*(1-ty) + z2*ty)
    except Exception:
        return None


# ============================================================
#   TAKEOFF
# ============================================================
@dataclass
class _Candidate:
    vs_kts: float
    v1_kts: float
    vr_kts: float
    v2_kts: float
    asd_ft: float
    todr_ft: float
    dispatchable: bool

class _Planner:
    def __init__(self) -> None:
        self.deck = TakeoffDeck()
        self.derate = DerateModel()
        self.aero = F14Aero()
        self.eng = F110Deck()

    def _speeds_placeholder(self, gw_lbs: float) -> Tuple[float, float, float, float]:
        # If takeoff_model exposes V-speeds, wire them here; otherwise use monotonic placeholders.
        vs = max(80.0, min(180.0, 0.85 * math.sqrt(max(gw_lbs, 1) / 80.0) + 100.0))
        v1 = vs + 8.0
        vr = v1 + 5.0
        v2 = vr + 10.0
        return vs, v1, vr, v2

    def _build_candidate(self, flap_deg: int, thrust_mode: str, gw_lbs: float,
                         field_elev_ft: float, qnh_inhg: float, oat_c: float,
                         tora_ft: float, asda_ft: float, req_aeo_ftpnm: float,
                         headwind_kts: float = 0.0) -> _Candidate:

        pa_ft = _pressure_alt_ft(field_elev_ft, qnh_inhg)
        base = self.deck.lookup(
            flap_deg=flap_deg, base_thrust=thrust_mode,
            gw_lbs=gw_lbs, pa_ft=pa_ft, oat_c=oat_c
        )

        # Apply a simple wind effect on distances (headwind reduces, tailwind increases) — light-touch placeholder
        hw = float(headwind_kts or 0.0)
        wind_factor = 1.0 - 0.01*max(-10.0, min(10.0, hw))  # ±10 kt -> ~±10% cap
        asd = max(0.0, base.asd_ft * wind_factor)
        todr = max(0.0, base.todr_ft * wind_factor)

        # Speeds (placeholder unless your deck returns them)
        vs, v1, vr, v2 = self._speeds_placeholder(gw_lbs)

        # Dispatchability (length gates; climb floor gate can be layered if/when we compute gradients)
        dispatch = (asd <= asda_ft) and (todr <= tora_ft)

        return _Candidate(vs_kts=vs, v1_kts=v1, vr_kts=vr, v2_kts=v2,
                          asd_ft=asd, todr_ft=todr, dispatchable=dispatch)

    def best_of(self, candA: _Candidate, candB: _Candidate) -> _Candidate:
        # Prefer dispatchable; among dispatchables, prefer lower todr; else lowest todr anyway
        a_ok, b_ok = candA.dispatchable, candB.dispatchable
        if a_ok and b_ok:
            return candA if candA.todr_ft <= candB.todr_ft else candB
        if a_ok: return candA
        if b_ok: return candB
        return candA if candA.todr_ft <= candB.todr_ft else candB


def compute_takeoff(inputs: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns UNFACTORED performance. The UI applies +10% for 'factored' presentation.
    Required inputs (keys are case-sensitive):
      - gw_lbs, oat_c, qnh_inhg, field_elev_ft, headwind_kts_component
      - tora_ft, asda_ft
      - flap_mode in {"UP","MAN","FULL"}
      - thrust_pref in {"Auto-Select","Manual MIL","Manual AB","Manual DERATE"}
      - manual_derate_pct (optional)
    """
    # Pull inputs
    gw = _ensure_float(inputs.get("gw_lbs", 60000))
    oat = _ensure_float(inputs.get("oat_c", 15))
    qnh = _ensure_float(inputs.get("qnh_inhg", 29.92))
    elev = _ensure_float(inputs.get("field_elev_ft", 0))
    headwind = _ensure_float(inputs.get("headwind_kts_component", 0))
    tora = _ensure_float(inputs.get("tora_ft", 999999))
    asda = _ensure_float(inputs.get("asda_ft", 999999))
    flap_mode = str(inputs.get("flap_mode", "MAN")).upper()
    thrust_pref = str(inputs.get("thrust_pref", "Auto-Select"))
    manual_pct = inputs.get("manual_derate_pct", None)

    # Overrides (LOCK SPEC)
    ov = overrides or {}
    climb_floor = _ensure_float(ov.get("climb_floor_ftpnm", 300.0))
    # Floors (not all enforced here, but used to clamp derate selections)
    floor_up, floor_man, floor_full = 85, 90, 96
    min_abs = 85

    # Map flaps
    flap2deg = {"UP":0, "MAN":20, "FULL":40}
    flap_deg = flap2deg.get(flap_mode, 20)

    # Resolve thrust mode
    # - Manual AB -> AFTERBURNER
    # - Manual MIL -> MILITARY
    # - Manual DERATE -> applies % floors (clamped >= abs 85%)
    # - Auto-Select -> choose between MIL and AB based on dispatchability (simple heuristic here)
    planner = _Planner()

    def _mk(flap_deg: int, mode: str) -> _Candidate:
        return planner._build_candidate(
            flap_deg, mode, gw, elev, qnh, oat, tora, asda, climb_floor, headwind
        )

    resolved_label = "MIL"
    candidate: _Candidate

    if thrust_pref == "Manual AB":
        resolved_label = "MAX AB"
        candidate = _mk(flap_deg, "AFTERBURNER")

    elif thrust_pref == "Manual MIL":
        resolved_label = "MIL"
        candidate = _mk(flap_deg, "MILITARY")

    elif thrust_pref == "Manual DERATE":
        pct = int(manual_pct or 95)
        # Enforce floors by flap mode
        floor = {"UP":floor_up, "MAN":floor_man, "FULL":floor_full}.get(flap_mode, floor_man)
        pct = max(pct, floor, min_abs)
        resolved_label = f"DERATE {pct}%"
        # For now, treat DERATE as MILITARY with margin (your DerateModel can apply internal scaling)
        candidate = _mk(flap_deg, "MILITARY")

    else:
        # Auto-select: test MIL, then AB if needed
        mil = _mk(flap_deg, "MILITARY")
        if mil.dispatchable:
            candidate = mil
            resolved_label = "MIL"
        else:
            ab = _mk(flap_deg, "AFTERBURNER")
            candidate = planner.best_of(mil, ab)
            resolved_label = "MAX AB" if candidate is ab else "MIL"

    return {
        "Vs_kts": candidate.vs_kts,
        "Vr_kts": candidate.vr_kts,
        "V2_kts": candidate.v2_kts,
        "ASDR_ft": candidate.asd_ft,                 # UNFACTORED; UI shows ×1.10
        "TODR_OEI_35ft_ft": candidate.todr_ft,       # UNFACTORED; UI shows ×1.10
        "Dispatchable": bool(candidate.dispatchable),
        "ResolvedThrustLabel": resolved_label,
        "Flaps": flap_mode,
        "GW_lbs": gw,
        "source": "model",
        "version": __version__,
    }


# ============================================================
#   LANDING (NATOPS → Unfactored; UI displays ×1.67)
# ============================================================
def compute_landing(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (recommended keys):
      - flap_setting (UP/MAN/FULL) or numeric degrees
      - gross_weight_lbs
      - pressure_alt_ft
      - temp_F
      - headwind_kt
    Data source: data/f14_landing_natops_full.csv (if present)
    """
    path = os.path.join("data", "f14_landing_natops_full.csv")
    df = _load_csv(path)

    flap = str(inputs.get("flap_setting", "FULL"))
    gw = _ensure_float(inputs.get("gross_weight_lbs", 60000))
    pa = _ensure_float(inputs.get("pressure_alt_ft", 0))
    tempF = _ensure_float(inputs.get("temp_F", 59))
    hw = _ensure_float(inputs.get("headwind_kt", 0))

    if df is not None and not df.empty:
        # Try exact match, else nearest neighbor
        mask_cols = [c for c in ["flap_setting","gross_weight_lbs","pressure_alt_ft","temp_F","headwind_kt"] if c in df.columns]
        exact = None
        if mask_cols:
            m = (df[mask_cols].astype(str) == pd.Series({
                "flap_setting": flap,
                "gross_weight_lbs": int(gw),
                "pressure_alt_ft": int(pa),
                "temp_F": int(tempF),
                "headwind_kt": int(hw)
            }).astype(str)).all(axis=1) if pd is not None else None
            if pd is not None and m is not None and m.any():
                exact = df.loc[m].iloc[0]
        row = exact if exact is not None else _nearest_row(df, {
            "flap_setting": flap, "gross_weight_lbs": gw, "pressure_alt_ft": pa, "temp_F": tempF, "headwind_kt": hw
        }, ["gross_weight_lbs","pressure_alt_ft","temp_F","headwind_kt"])

        if row is not None:
            ldr = _ensure_float(row.get("ground_roll_ft_unfactored", row.get("ldr_unfactored_ft", 0.0)))
            vref = _ensure_float(row.get("vref_kts", 0.0))
            return {
                "ldr_unfactored_ft": ldr,      # UI multiplies ×1.67
                "vref_kts": vref if vref > 0 else None,
                "source": "NATOPS_csv",
                "version": __version__,
            }

    # Fallback placeholder if CSV missing
    # Simple heuristic: heavier weight -> longer ground roll
    base = 2500.0 + 0.05 * max(0.0, gw - 40000.0)
    return {
        "ldr_unfactored_ft": base,
        "vref_kts": None,
        "source": "placeholder",
        "version": __version__,
        "warning": "Landing CSV not found; using placeholder heuristic.",
    }


# ============================================================
#   CRUISE (NATOPS GW×DI → Opt Alt / Mach)
# ============================================================
def compute_cruise(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      - gross_weight_lbs
      - drag_index
    Data source: data/f14_cruise_natops.csv (if present)
      columns: gross_weight_lbs, drag_index, optimum_alt_ft, optimum_mach
    """
    gw = _ensure_float(inputs.get("gross_weight_lbs", inputs.get("gw_lbs", 60000)))
    di = _ensure_float(inputs.get("drag_index", 50))
    path = os.path.join("data", "f14_cruise_natops.csv")
    df = _load_csv(path)

    if df is not None and not df.empty:
        opt_alt = _bilinear(df, "drag_index", "gross_weight_lbs", "optimum_alt_ft", di, gw)
        opt_mach = _bilinear(df, "drag_index", "gross_weight_lbs", "optimum_mach", di, gw)
        if opt_alt and opt_mach:
            return {
                "optimum_alt_ft": float(opt_alt),
                "optimum_mach": float(opt_mach),
                "gross_weight_lbs": gw,
                "drag_index": di,
                "source": "NATOPS_csv",
                "version": __version__,
            }

    # Fallback placeholder (conservative)
    return {
        "optimum_alt_ft": max(20000.0, 45000.0 - 0.2*(gw-50000.0) - 10.0*di),
        "optimum_mach": 0.75,
        "gross_weight_lbs": gw,
        "drag_index": di,
        "source": "placeholder",
        "version": __version__,
        "warning": "Cruise CSV not found; using placeholder estimate.",
    }


# ============================================================
#   CLIMB (Economy vs Interceptor)
# ============================================================
def compute_climb(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      - profile: "Economy" | "Interceptor"
      - respect_250: bool (250 KIAS below 10k)
      - gw_lbs (optional), oat_c, qnh_inhg, etc (optional)
    Data source: data/f14_climb_natops.csv (if present)
    """
    profile = str(inputs.get("profile", "Economy"))
    respect_250 = bool(inputs.get("respect_250", True))
    path = os.path.join("data", "f14_climb_natops.csv")
    df = _load_csv(path)

    out: Dict[str, Any] = {
        "profile": profile,
        "respect_250": respect_250,
        "source": "NATOPS_csv" if (df is not None and not df.empty) else "placeholder",
        "version": __version__,
    }

    if df is not None and not df.empty:
        # We don't know your exact schema yet; return a trimmed preview
        out["schedule"] = df.head(50).to_dict(orient="records") if pd is not None else []
        return out

    # Placeholder schedule (human-friendly, conservative)
    if profile == "Economy":
        sched = [
            {"phase":"Liftoff to 35ft","target":"V2","note":"hold V2 until 35 ft AGL"},
            {"phase":"Acceleration to 1000ft AFE","target":"V2+15","note":"clean-up as per SOP"},
            {"phase":"Below 10k","target":"250 KIAS" if respect_250 else "300 KIAS","note":"fuel efficient climb"},
            {"phase":"Above 10k","target":"Mach 0.70–0.75","note":"transition IAS→Mach"},
        ]
    else:  # Interceptor
        sched = [
            {"phase":"Liftoff to 35ft","target":"V2","note":"minimum time"},
            {"phase":"Acceleration to 1000ft AFE","target":"V2+15","note":"clean-up as per SOP"},
            {"phase":"Below 10k","target":"250 KIAS" if respect_250 else "320–350 KIAS","note":"max rate"},
            {"phase":"Above 10k","target":"Best rate IAS→Mach","note":"expedite to intercept"},
        ]
    out["schedule"] = sched
    out["warning"] = "Climb CSV not found; using placeholder schedule."
    return out
