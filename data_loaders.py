# data_loaders.py — v1.2.1-data
# Path-hardening: all CSV loads default to ./data relative to this file.

from __future__ import annotations
import os
from functools import lru_cache
import pandas as pd

# Resolve ./data relative to this file (module location works in Streamlit Cloud, local, etc.)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _csv_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)

@lru_cache(maxsize=None)
def load_engine_csv(path: str | None = None) -> pd.DataFrame:
    """
    F110 engine deck CSV loader.
    Columns required: Altitude_ft, Mach, PowerSetting, Thrust_lbf, FuelFlow_pph
    Default location: ./data/f110_engine.csv
    """
    if path is None:
        path = _csv_path("f110_engine.csv")
    df = pd.read_csv(path)
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
    """
    if path is None:
        path = _csv_path("f14_aero.csv")
    df = pd.read_csv(path)
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
    Default location: ./data/calibration.csv  (keep optional)
    """
    if path is None:
        path = _csv_path("calibration.csv")
    df = pd.read_csv(path)
    req = {"Parameter", "FAA_Default", "DCS_Default"}
    if not set(req).issubset(df.columns):
        raise ValueError("Calibration CSV missing required columns")
    return {
        r.Parameter: {"FAA": r.FAA_Default, "DCS": r.DCS_Default}
        for _, r in df.iterrows()
    }
