# data_loaders.py — v1.3.0-data
# Path-hardening for CSV/JSON loads. Default root is ./data relative to this file.
# New:
#  - resolve_data_path(name): resolve a filename reliably to ./data/<name> (or existing absolute/relative path).
#  - load_json_config(name): JSON loader with same resolution rules.
#  - (Kept) CSV loaders for aero/calibration; removed outdated f110_engine.csv shim.

from __future__ import annotations
import os
import json
from functools import lru_cache
from typing import Optional
import pandas as pd

# Resolve ./data relative to this file (works in Streamlit Cloud, local, etc.)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _csv_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)

def resolve_data_path(path_or_name: Optional[str], default_name: Optional[str] = None) -> str:
    """
    Resolution rules:
      1) If path_or_name is None -> use ./data/<default_name> (if provided) or raise.
      2) If path_or_name is an existing file (absolute or relative) -> use it.
      3) Else, try ./data/<basename(path_or_name or default_name)>.
      4) Else, raise FileNotFoundError listing attempted locations.
    """
    tried = []
    # Case 1: None -> must have default_name
    if path_or_name is None:
        if not default_name:
            raise FileNotFoundError("resolve_data_path: default_name is required when path_or_name is None")
        p = _csv_path(default_name)
        if os.path.isfile(p):
            return p
        tried.append(p)
        raise FileNotFoundError(f"Missing required data file. Tried: {tried}")

    # Normalize
    # If user provided a path and it exists, take it
    if os.path.isfile(path_or_name):
        return path_or_name
    tried.append(path_or_name)

    base = os.path.basename(path_or_name if path_or_name else (default_name or ""))
    candidate = _csv_path(base)
    if os.path.isfile(candidate):
        return candidate
    tried.append(candidate)

    raise FileNotFoundError(f"No such file. Tried: {tried}")

@lru_cache(maxsize=None)
def load_json_config(path_or_name: Optional[str]) -> dict:
    """
    Load JSON config using resolve_data_path rules.
    Typical: load_json_config('derate_config.json')
    """
    p = resolve_data_path(path_or_name, "derate_config.json")
    with open(p, "r") as f:
        return json.load(f)

@lru_cache(maxsize=None)
def load_aero_csv(path: str | None = None) -> pd.DataFrame:
    """
    F-14 aero polar CSV loader.
    Columns required: Config, WingSweep_deg, CLmax, CD0, k
    Default location: ./data/f14_aero.csv
    """
    resolved = resolve_data_path(path, "f14_aero.csv")
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
    Default location: ./data/calibration.csv
    """
    resolved = resolve_data_path(path, "calibration.csv")
    df = pd.read_csv(resolved)
    req = {"Parameter", "FAA_Default", "DCS_Default"}
    if not set(req).issubset(df.columns):
        raise ValueError("Calibration CSV missing required columns")
    return {
        r.Parameter: {"FAA": r.FAA_Default, "DCS": r.DCS_Default}
        for _, r in df.iterrows()
    }
