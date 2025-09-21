# engine_f110.py — v1.2
# Corrected thrust scaling: use T_MIL_lbf from CSV with Mach falloff law, AB ratio applied.

import pandas as pd
from data_loaders import resolve_data_path

class F110Deck:
    def __init__(self, csv_file: str = "f110_tff_model.csv"):
        self.df = pd.read_csv(resolve_data_path(csv_file, "f110_tff_model.csv"))
        self.df = self.df.sort_values("alt_ft")
        self.ab_ratio = 1.95  # MAX/AB relative to MIL, approx from NATOPS
        self.k_mach = 0.35    # MIL thrust falloff with Mach

    def _interp_on_alt(self, column: str, alt_ft: float) -> float:
        """Simple 1D interpolation on altitude for given column."""
        d = self.df
        if alt_ft <= d["alt_ft"].min():
            return float(d[column].iloc[0])
        if alt_ft >= d["alt_ft"].max():
            return float(d[column].iloc[-1])
        lower = d[d["alt_ft"] <= alt_ft].iloc[-1]
        upper = d[d["alt_ft"] >= alt_ft].iloc[0]
        frac = (alt_ft - lower["alt_ft"]) / (upper["alt_ft"] - lower["alt_ft"] + 1e-6)
        return float(lower[column] + frac * (upper[column] - lower[column]))

    def thrust_lbf(self, alt_ft: float, mach: float, power: str) -> float:
        """Return per-engine thrust in pounds at given altitude, Mach, and power setting."""
        base_T = self._interp_on_alt("T_MIL_lbf", alt_ft)
        # Apply Mach falloff (linear, clamp at 70% minimum)
        base_T = base_T * max(0.7, (1.0 - self.k_mach * float(mach)))

        pu = str(power).upper()
        if pu.startswith("MAX") or "AFTERBURNER" in pu:
            return base_T * self.ab_ratio
        if pu.startswith("IDLE"):
            return max(0.15 * base_T, 500.0)
        return base_T
