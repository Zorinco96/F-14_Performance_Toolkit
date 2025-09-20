# f14_aero.py — v1.2.1-data
# Change: default CSV path now resolves to ./data/f14_aero.csv via data_loaders.

from __future__ import annotations
import math
import pandas as pd
from data_loaders import load_aero_csv

S_WING_FT2 = 565.0  # ref area (approx)
S_WING_M2 = S_WING_FT2 * 0.09290304

class F14Aero:
    def __init__(self, path: str | None = None):
        # When path is None, load_aero_csv() will use ./data/f14_aero.csv
        self.df = load_aero_csv(path)

    def polar(self, config: str, sweep_deg: int | float = 20):
        # Find nearest entry for this config & sweep
        d = self.df[self.df["Config"] == config]
        if d.empty:
            raise ValueError(f"Config not found: {config}")
        if "WingSweep_deg" in d.columns and d["WingSweep_deg"].nunique() > 1:
            # choose nearest sweep value
            diffs = (d["WingSweep_deg"] - float(sweep_deg)).abs()
            row = d.iloc[diffs.idxmin()]
        else:
            row = d.iloc[0]
        return float(row["CLmax"]), float(row["CD0"]), float(row["k"])

    def lift_coeff(self, weight_lbf: float, rho: float, V_mps: float) -> float:
        q = 0.5 * rho * V_mps**2
        L = weight_lbf * 4.4482216153  # N
        return L / (q * S_WING_M2)

    def drag_coeff(self, CL: float, CD0: float, k: float, cd_misc: float = 0.0) -> float:
        return CD0 + k * CL**2 + cd_misc
