
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

class F110Deck:
    """
    Lightweight F110-GE-400/401 MIL/MAX thrust model backed by two CSVs:
      - data/f110_tff_model.csv          (columns: alt_ft, C, alpha, T_MIL_lbf, FF_MIL_pph, ...)
      - data/f110_ff_to_rpm_knots.csv    (columns: FF_pph, RPM_pct)
    This replaces the old monolithic f110_engine.csv deck.
    """
    def __init__(self, data_dir: str | None = None, ab_ratio: float | None = None):
        here = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else (here / "data")
        # Load models
        self.tff = pd.read_csv(self.data_path("f110_tff_model.csv"))
        self.ff2rpm = pd.read_csv(self.data_path("f110_ff_to_rpm_knots.csv"))
        # Ensure sorted by altitude
        self.tff = self.tff.sort_values("alt_ft").reset_index(drop=True)
        # AB thrust ratio relative to MIL. If not provided, use 1.695 taken from MIL/MAX at SL in legacy deck.
        # (MAX should be refined later via cockpit sweeps; keeping here to avoid breaking MAX paths.)
        self.ab_ratio = float(ab_ratio if ab_ratio is not None else 1.695)

    def data_path(self, name: str) -> str:
        p = self.data_dir / name
        if not p.exists():
            # also try CWD/data for Streamlit working-dir runs
            alt = Path.cwd() / "data" / name
            if alt.exists():
                return str(alt)
        return str(p)

    # ---- Helpers ------------------------------------------------------------
    def _interp_on_alt(self, col: str, alt_ft: float) -> float:
        x = self.tff["alt_ft"].to_numpy(dtype=float)
        y = self.tff[col].to_numpy(dtype=float)
        a = float(np.clip(alt_ft, x.min(), x.max()))
        return float(np.interp(a, x, y))

    # ---- Public API ---------------------------------------------------------
    def thrust_lbf(self, alt_ft: float, mach: float, power: str) -> float:
        """Return per-engine thrust in pounds at given alt/mach and power."""
        C = self._interp_on_alt("C", alt_ft)
        alpha = self._interp_on_alt("alpha", alt_ft)
        # simple model: T_mil(alt, M) = C(alt) * (1 + M)^alpha
        T_mil = C * (1.0 + float(mach)) ** alpha
        if str(power).upper().startswith("MAX") or "AFTERBURNER" in str(power).upper():
            return T_mil * self.ab_ratio
        # include IDLE in same family: return a small fraction of MIL if ever asked
        if str(power).upper().startswith("IDLE"):
            return max(0.15 * T_mil, 500.0)
        return T_mil

    def fuel_flow_pph(self, alt_ft: float, mach: float, power: str) -> float:
        """MIL fuel flow scaled with thrust; MAX uses a simple multiplier (placeholder)."""
        base_ff = self._interp_on_alt("FF_MIL_pph", alt_ft)
        base_T = self._interp_on_alt("T_MIL_lbf", alt_ft)
        T = self.thrust_lbf(alt_ft, mach, power if str(power).upper().startswith("MIL") else "MIL")
        ff = 0.0
        if base_T > 1.0:
            ff = base_ff * (T / base_T)
        else:
            ff = base_ff
        if str(power).upper().startswith("MAX") or "AFTERBURNER" in str(power).upper():
            # crude MAX uplift; will be replaced by calibrated curve when available
            ff *= 1.85
        return float(ff)

    def rpm_from_ff(self, ff_pph: float) -> float:
        """Map fuel flow to RPM (%) using cockpit-swept table (piecewise linear)."""
        df = self.ff2rpm.sort_values("FF_pph")
        x = df["FF_pph"].to_numpy(dtype=float)
        y = df["RPM_pct"].to_numpy(dtype=float)
        ff = float(np.clip(ff_pph, x.min(), x.max()))
        return float(np.interp(ff, x, y))
