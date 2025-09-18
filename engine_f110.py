
from __future__ import annotations
import numpy as np
from data_loaders import load_engine_csv

class F110Deck:
    def __init__(self, path: str = "f110_engine.csv"):
        self.df = load_engine_csv(path)

    def _interp3(self, alt_ft: float, mach: float, power: str, col: str) -> float:
        d = self.df[self.df["PowerSetting"] == power].copy()
        # Bound inputs
        alt_grid = np.sort(d["Altitude_ft"].unique())
        mach_grid = np.sort(d["Mach"].unique())
        alt_ft = float(np.clip(alt_ft, alt_grid.min(), alt_grid.max()))
        mach = float(np.clip(mach, mach_grid.min(), mach_grid.max()))

        # Surrounding points
        a1 = alt_grid[alt_grid <= alt_ft].max()
        a2 = alt_grid[alt_grid >= alt_ft].min()
        m1 = mach_grid[mach_grid <= mach].max()
        m2 = mach_grid[mach_grid >= mach].min()

        def val(A, M):
            sub = d[(d["Altitude_ft"] == A) & (d["Mach"] == M)]
            return float(sub[col].values[0])

        if a1 == a2 and m1 == m2:
            return val(a1, m1)
        elif a1 == a2:
            v1, v2 = val(a1, m1), val(a2, m2)  # same alt actually
            # linear in Mach
            return v1 + (v2 - v1) * ((mach - m1) / max(1e-9, m2 - m1))
        elif m1 == m2:
            v1, v2 = val(a1, m1), val(a2, m2)
            return v1 + (v2 - v1) * ((alt_ft - a1) / max(1e-9, a2 - a1))
        else:
            Q11 = val(a1, m1)
            Q21 = val(a2, m1)
            Q12 = val(a1, m2)
            Q22 = val(a2, m2)
            # bilinear interpolation
            fa = (alt_ft - a1) / max(1e-9, a2 - a1)
            fm = (mach - m1) / max(1e-9, m2 - m1)
            R1 = Q11 + (Q21 - Q11) * fa
            R2 = Q12 + (Q22 - Q12) * fa
            return R1 + (R2 - R1) * fm

    def thrust_lbf(self, alt_ft: float, mach: float, power: str) -> float:
        return self._interp3(alt_ft, mach, power, "Thrust_lbf")

    def fuel_flow_pph(self, alt_ft: float, mach: float, power: str) -> float:
        return self._interp3(alt_ft, mach, power, "FuelFlow_pph")
