
"""
derate.py — F‑14B derate helper
--------------------------------
Chain: required thrust  → required FF → required RPM%
Data sources (CSV artifacts):
- f110_tff_model.csv           : per-altitude power law fits T = C * FF^alpha (M≈0)
- f110_ff_to_rpm_knots.csv     : FF→RPM% monotonic piecewise mapping (Sea Level sweep)
- calibration_sl_summary.csv   : SL flap-specific bias and delta-to-35ft (optional)

Typical use:
    from derate import DerateModel

    dm = DerateModel(
        tff_model_csv="f110_tff_model.csv",
        ff_rpm_knots_csv="f110_ff_to_rpm_knots.csv",
        calibration_sl_csv="calibration_sl_summary.csv",  # optional
        config_json="derate_config.json"                  # optional
    )

    # Ground-roll driven derate example:
    out = dm.compute_derate_from_groundroll(
        flap_deg=0,
        pa_ft=0,
        mil_ground_roll_ft=3200,    # baseline at MIL for the scenario (already bias-corrected)
        runway_available_ft=4500,   # available distance minus safety margin
        allow_ab=False              # cap at MIL (default)
    )
    print(out)  # dict with T_required_lbf, FF_required_pph, RPM_required_pct, thrust_multiplier

Notes:
- This scaffold does not compute "mil_ground_roll_ft" from scratch; feed it from your calibrated
  performance model (NATOPS + SL bias + delta-to-35 ft, etc.).
- If you have AB spot checks, set flap-specific 'm' in derate_config.json for better accuracy.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import bisect
import math
import csv

@dataclass
class TFFPoint:
    alt_ft: int
    C: float
    alpha: float
    T_MIL_lbf: float
    FF_MIL_pph: float

class DerateModel:
    def __init__(
        self,
        tff_model_csv: str = "f110_tff_model.csv",
        ff_rpm_knots_csv: str = "f110_ff_to_rpm_knots.csv",
        calibration_sl_csv: Optional[str] = "calibration_sl_summary.csv",
        config_json: Optional[str] = "derate_config.json",
    ):
        self.tff_by_alt = self._load_tff(tff_model_csv)
        self.knots_ff, self.knots_rpm = self._load_knots(ff_rpm_knots_csv)
        self.cal_sl = self._load_calibration(calibration_sl_csv) if calibration_sl_csv else {}
        self.cfg = self._load_config(config_json) if config_json else self._default_config()

    # ---------- Loading ----------
    def _load_tff(self, path: str) -> Dict[int, TFFPoint]:
        by_alt: Dict[int, TFFPoint] = {}
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                alt = int(float(row["alt_ft"]))
                by_alt[alt] = TFFPoint(
                    alt_ft = alt,
                    C = float(row["C"]),
                    alpha = float(row["alpha"]),
                    T_MIL_lbf = float(row.get("T_MIL_lbf", "0") or 0),
                    FF_MIL_pph = float(row.get("FF_MIL_pph", "0") or 0),
                )
        return dict(sorted(by_alt.items()))

    def _load_knots(self, path: str) -> Tuple[List[float], List[float]]:
        ff: List[float] = []
        rpm: List[float] = []
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                ff.append(float(row["FF_pph"]))
                rpm.append(float(row["RPM_pct"]))
        # ensure sorted by FF
        pairs = sorted(zip(ff, rpm), key=lambda x: x[0])
        ff = [p[0] for p in pairs]
        rpm = [p[1] for p in pairs]
        return ff, rpm

    def _load_calibration(self, path: str) -> Dict[int, Dict[str, float]]:
        # Returns { flap_deg: {"bias_factor": ..., "delta_to35_ft": ...} }
        out: Dict[int, Dict[str, float]] = {}
        try:
            with open(path, newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    flap = int(float(row["Flaps_deg"]))
                    out[flap] = {
                        "bias_factor": float(row.get("bias_factor", "1.0") or 1.0),
                        "delta_to35_ft": float(row.get("delta_to35_ft", "0.0") or 0.0),
                    }
        except FileNotFoundError:
            pass
        return out

    def _default_config(self) -> Dict:
        return {
            "allow_ab": False,
            "min_idle_ff_pph": 1200,   # from SL sweep
            "thrust_exponent_m": { "0": 0.75, "35": 0.75 },  # placeholder; replace with AB/MIL spot-check fit
            "safety": { "runway_margin_ft": 0.0 },
        }

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()

    # ---------- Core transforms ----------
    def _nearest_alt_point(self, pa_ft: float) -> TFFPoint:
        # Nearest neighbor in alt grid
        alts = list(self.tff_by_alt.keys())
        idx = bisect.bisect_left(alts, pa_ft)
        if idx == 0: return self.tff_by_alt[alts[0]]
        if idx == len(alts): return self.tff_by_alt[alts[-1]]
        before = alts[idx-1]; after = alts[idx]
        return self.tff_by_alt[before] if (pa_ft - before) <= (after - pa_ft) else self.tff_by_alt[after]

    def thrust_to_ff(self, T_req_lbf: float, pa_ft: float) -> float:
        """Invert T = C * FF^alpha at nearest altitude."""
        p = self._nearest_alt_point(pa_ft)
        # Guard
        T_req_lbf = max(T_req_lbf, 0.0)
        # Solve FF = (T/C)^(1/alpha)
        FF = (T_req_lbf / max(p.C, 1e-6)) ** (1.0 / max(p.alpha, 1e-6))
        # Never below idle
        FF = max(FF, float(self.cfg.get("min_idle_ff_pph", 1000)))
        return FF

    def ff_to_rpm(self, ff_pph: float) -> float:
        """Piecewise linear interpolation through SL knots; clamp to ends."""
        x = self.knots_ff; y = self.knots_rpm
        if ff_pph <= x[0]: return y[0]
        if ff_pph >= x[-1]: return y[-1]
        i = bisect.bisect_left(x, ff_pph)
        x0, x1 = x[i-1], x[i]
        y0, y1 = y[i-1], y[i]
        t = (ff_pph - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # ---------- Ground-roll driven derate ----------
    def solve_thrust_multiplier_from_groundroll(self, mil_ground_roll_ft: float, target_ground_roll_ft: float, flap_deg: int) -> float:
        """
        Local sensitivity model: Ground Roll ∝ T^(−m).
        multiplier = (GR_MIL / GR_target)^(1/m).
        """
        m = float(self.cfg.get("thrust_exponent_m", {}).get(str(int(flap_deg)), 0.75))
        m = max(min(m, 2.0), 0.2)  # sanity
        gr_mil = max(mil_ground_roll_ft, 1.0)
        gr_target = max(target_ground_roll_ft, 1.0)
        mult = (gr_mil / gr_target) ** (1.0 / m)
        return max(mult, 0.0)

    def compute_derate_from_groundroll(
        self,
        flap_deg: int,
        pa_ft: float,
        mil_ground_roll_ft: float,
        runway_available_ft: float,
        allow_ab: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Solve for the minimum thrust so that GroundRoll <= runway_available_ft.
        Requires baseline MIL ground roll for the scenario (already bias-corrected in your perf model).
        """
        if allow_ab is None: allow_ab = bool(self.cfg.get("allow_ab", False))
        margin = float(self.cfg.get("safety", {}).get("runway_margin_ft", 0.0))
        target = max(runway_available_ft - margin, 1.0)

        # Thrust multiplier needed
        mult = self.solve_thrust_multiplier_from_groundroll(mil_ground_roll_ft, target, flap_deg)

        # Map to absolute thrust using nearest-alt MIL reference
        p = self._nearest_alt_point(pa_ft)
        T_required = mult * p.T_MIL_lbf
        # If AB not allowed, cap at MIL
        if not allow_ab:
            T_required = min(T_required, p.T_MIL_lbf)
            mult = T_required / max(p.T_MIL_lbf, 1e-6)

        # Convert to FF and RPM
        FF_required = self.thrust_to_ff(T_required, pa_ft)
        RPM_required = self.ff_to_rpm(FF_required)

        return {
            "thrust_multiplier": float(mult),
            "T_required_lbf": float(T_required),
            "FF_required_pph": float(FF_required),
            "RPM_required_pct": float(RPM_required),
            "T_MIL_lbf": float(p.T_MIL_lbf),
            "FF_MIL_pph": float(p.FF_MIL_pph),
            "alpha_used": float(p.alpha),
            "alt_used_ft": int(p.alt_ft),
        }
