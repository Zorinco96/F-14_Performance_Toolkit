# data_loaders.py — v1.2.2-data
# Path-hardening: all CSV loads default to ./data relative to this file.
# New: smart fallback — if a caller passes a bare filename or a non-existent path,
#      we transparently try ./data/<name> before failing.

from __future__ import annotations
import os
from functools import lru_cache
import pandas as pd
from typing import Optional

# Resolve ./data relative to this file (works in Streamlit Cloud, local, etc.)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _csv_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)

def _resolve_user_or_data(path: Optional[str], default_name: str) -> str:
    """
    Resolution rules:
      1) If path is None → use ./data/<default_name>.
      2) If path is absolute or has a directory and exists → use as-is.
      3) If path is a bare filename (no dir) AND exists in CWD → use as-is.
      4) Otherwise → try ./data/<basename(path or default_name)>.
      5) If that still doesn't exist → raise FileNotFoundError (with both tried paths).
    """
    tried = []
    if path is None:
        p = _csv_path(default_name)
        if os.path.isfile(p):
            return p
        tried.append(p)
        raise FileNotFoundError(f"Missing required CSV: {p}")

    # Normalize
    base = os.path.basename(path)
    has_dir = (os.path.dirname(path) != "")
    # Case A: given path exists → use it
    if os.path.isfile(path):
        return path
    tried.append(path)

    # Case B: caller passed a dir path but it doesn't exist; try ./data/<base>
    candidate = _csv_path(base)
    if os.path.isfile(candidate):
        return candidate
    tried.append(candidate)

    # Case C: if caller passed a bare filename that also isn't in CWD, we already checked ./data/<base>.
    # Nothing left to try:
    raise FileNotFoundError(f"No such file. Tried: {tried}")

@lru_cache(maxsize=None)
def load_engine_csv(path: str | None = None) -> pd.DataFrame:
    """
    F110 engine deck CSV loader.
    Columns required: Altitude_ft, Mach, PowerSetting, Thrust_lbf, FuelFlow_pph
    Default location: ./data/f110_engine.csv
    Smart fallback: if a caller passes 'f110_engine.csv' or some bad relative path,
                    we transparently try ./data/f110_engine.csv.
    """
    resolved = _resolve_user_or_data(path, "f110_engine.csv")
    df = pd.read_csv(resolved)
    req = {"Altitude_ft", "Mach", "PowerSetting", "Thrust_lbf", "FuelFlow_pph"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Engine CSV missing columns: {missing}")
    return df

@lru_cache(maxsize=None)
def load_aero_csv(path: str | None = None) -> pd.DataFrame:
    """
    F-14 aero polar CSV loader.
    Columns required: Config, WingSweep_deg, CLmax, CD0, k
    Default location: ./data/f14_aero.csv
    Smart fallback: as above.
    """
    resolved = _resolve_user_or_data(path, "f14_aero.csv")
    df = pd.read_csv(resolved)
    req = {"Config", "WingSweep_deg", "CLmax", "CD0", "k"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Aero CSV missing columns: {missing}")
    return df

@lru_cache(maxsize=None)
def load_calibration_csv(path: str | None = None) -> dict:
    """
    Generic calibration table to dict.
    Columns required: Parameter, FAA_Default, DCS_Default
    Default location: ./data/calibration.csv  (optional in some builds)
    Smart fallback: as above.
    """
    resolved = _resolve_user_or_data(path, "calibration.csv")
    df = pd.read_csv(resolved)
    req = {"Parameter", "FAA_Default", "DCS_Default"}
    if not set(req).issubset(df.columns):
        raise ValueError("Calibration CSV missing required columns")
    return {
        r.Parameter: {"FAA": r.FAA_Default, "DCS": r.DCS_Default}
        for _, r in df.iterrows()
    }
