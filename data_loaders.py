
from __future__ import annotations
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=None)
def load_engine_csv(path: str = "f110_engine.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # sanity
    req = {"Altitude_ft","Mach","PowerSetting","Thrust_lbf","FuelFlow_pph"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Engine CSV missing columns: {missing}")
    return df

@lru_cache(maxsize=None)
def load_aero_csv(path: str = "f14_aero.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"Config","WingSweep_deg","CLmax","CD0","k"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Aero CSV missing columns: {missing}")
    return df

@lru_cache(maxsize=None)
def load_calibration_csv(path: str = "calibration.csv") -> dict:
    df = pd.read_csv(path)
    req = {"Parameter","FAA_Default","DCS_Default"}
    if not set(req).issubset(df.columns):
        raise ValueError("Calibration CSV missing required columns")
    return {r.Parameter: {"FAA": r.FAA_Default, "DCS": r.DCS_Default} for _, r in df.iterrows()}
