
# ============================================================
# F-14 Performance Calculator for DCS World — Derate Policy
# File: derate.py
# Version: v1.2.0-overhaul1 (2025-09-21)
# ============================================================
from __future__ import annotations
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Tuple

DEFAULT_CFG_PATH = "data/derate_config.json"

# --- Helpers ---

def _nearest_flap_key(deg: float) -> str:
    """Return the nearest flap key among {'0','20','35'} for a given flap angle in degrees."""
    targets = [0.0, 20.0, 35.0]
    nearest = min(targets, key=lambda t: abs((deg or 0.0) - t))
    return str(int(nearest))

@lru_cache(maxsize=1)
def _load_cfg(path: str = DEFAULT_CFG_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # Minimal safe fallback
        return {
            "policy": {"allow_ab": False, "min_takeoff_rpm_pct": 85},
            "floors": {"min_pct_by_flap_deg": {"0": 85, "20": 90, "35": 96}},
            "thrust_model": {"thrust_exponent_m": {"0": 0.75, "20": 0.75, "35": 0.75}, "min_idle_ff_pph": 1200},
            "safety": {"runway_factor": 1.10},
            # legacy mirrors
            "allow_ab": False,
            "min_idle_ff_pph": 1200,
            "thrust_exponent_m": {"0": 0.75, "20": 0.75, "35": 0.75},
            "min_pct_by_flap_deg": {"0": 85, "20": 90, "35": 96},
        }

def _get_section(cfg: Dict[str, Any], path_keys, legacy_key=None, default=None):
    node = cfg
    for k in path_keys:
        if not isinstance(node, dict) or k not in node:
            node = None
            break
        node = node[k]
    if node is not None:
        return node
    # fallback to legacy flat key if provided
    if legacy_key and isinstance(cfg, dict):
        return cfg.get(legacy_key, default)
    return default

# --- Policy Accessors ---

def allow_ab(path: str = DEFAULT_CFG_PATH) -> bool:
    cfg = _load_cfg(path)
    val = _get_section(cfg, ["policy", "allow_ab"], legacy_key="allow_ab", default=False)
    return bool(val)

def min_takeoff_rpm_pct(path: str = DEFAULT_CFG_PATH) -> int:
    cfg = _load_cfg(path)
    val = _get_section(cfg, ["policy", "min_takeoff_rpm_pct"], default=85)
    return int(val)

def runway_factor(path: str = DEFAULT_CFG_PATH) -> float:
    cfg = _load_cfg(path)
    val = _get_section(cfg, ["safety", "runway_factor"], default=1.10)
    return float(val)

def floor_pct_for_flaps(flap_deg: float, path: str = DEFAULT_CFG_PATH) -> int:
    cfg = _load_cfg(path)
    key = _nearest_flap_key(flap_deg)
    floors = _get_section(cfg, ["floors", "min_pct_by_flap_deg"], legacy_key="min_pct_by_flap_deg", default={})
    floor = floors.get(key, 85)
    # Enforce absolute minimum RPM for takeoff
    floor = max(int(floor), min_takeoff_rpm_pct(path))
    return int(floor)

def m_exponent_for_flaps(flap_deg: float, path: str = DEFAULT_CFG_PATH) -> float:
    cfg = _load_cfg(path)
    key = _nearest_flap_key(flap_deg)
    m_map = _get_section(cfg, ["thrust_model", "thrust_exponent_m"], legacy_key="thrust_exponent_m", default={})
    return float(m_map.get(key, 0.75))

def min_idle_ff_pph(path: str = DEFAULT_CFG_PATH) -> int:
    cfg = _load_cfg(path)
    val = _get_section(cfg, ["thrust_model", "min_idle_ff_pph"], legacy_key="min_idle_ff_pph", default=1200)
    return int(val)

# --- Clamp Utility ---

@dataclass
class ClampResult:
    requested_pct: float
    applied_pct: float
    floor_pct: int
    clamped_to_floor: bool

def clamp_derate_pct(requested_pct: float, flap_deg: float, path: str = DEFAULT_CFG_PATH) -> ClampResult:
    """Clamp a requested %RPM to the flap-specific floor and absolute minimum takeoff RPM.
    Returns the applied value and whether we clamped to a floor.
    """
    floor = floor_pct_for_flaps(flap_deg, path)
    applied = max(float(requested_pct or 0.0), float(floor))
    return ClampResult(
        requested_pct=float(requested_pct or 0.0),
        applied_pct=applied,
        floor_pct=int(floor),
        clamped_to_floor=(applied > (requested_pct or 0.0))
    )

# --- Convenience API for UI/Core ---

def get_policy_snapshot(path: str = DEFAULT_CFG_PATH) -> Dict[str, Any]:
    """Return a single dict snapshot the UI can display in debug/calibration."""
    cfg = _load_cfg(path)
    return {
        "allow_ab": allow_ab(path),
        "min_takeoff_rpm_pct": min_takeoff_rpm_pct(path),
        "floors": _get_section(cfg, ["floors", "min_pct_by_flap_deg"], legacy_key="min_pct_by_flap_deg", default={}),
        "thrust_exponent_m": _get_section(cfg, ["thrust_model", "thrust_exponent_m"], legacy_key="thrust_exponent_m", default={}),
        "min_idle_ff_pph": min_idle_ff_pph(path),
        "runway_factor": runway_factor(path),
    }
