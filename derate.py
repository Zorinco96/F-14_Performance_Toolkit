"""
derate.py — F-14B derate helper  (v1.2.2-derate)
Changes:
- Adds flap-band derate floors via config (defaults: 0°→85, 35°→90).
- Returns 'derate_pct' and 'clamped_to_floor' in output.
- If computed RPM%/FF would imply < floor, we snap to the floor (and recompute).
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
                alt = int(float(row["alt_ft\]))
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
        pairs = sorted(zip(ff, rpm), key=lambda x: x[0])
        ff = [p[0] for p in pairs]
        rpm = [p[1] for p in pairs]
        return ff, rpm

    def _load_calibration(self, path: str) -> Dict[int, Dict[str, float]]:
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
            "min_idle_ff_pph": 1200,
            "thrust_exponent_m": { "0": 0.75, "35": 0.75 },
            "min_pct_by_flap_deg": { "0": 85, "35": 90 },
            "safety": { "runway_margin_ft": 0.0 },
        }

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = self._default_config()
        cfg.setdefault("min_idle_ff_pph", 1200)
        cfg.setdefault("thrust_exponent_m", { "0": 0.75, "35": 0.75 })
        cfg.setdefault("min_pct_by_flap_deg", { "0": 85, "35": 90 })
        cfg.setdefault("safety", { "runway_margin_ft": 0.0 })
        return cfg

    # ---------- Core transforms ----------
    def _nearest_alt_point(self, pa_ft: float) -> TFFPoint:
        alts = list(self.tff_by_alt.keys())
        idx = bisect.bisect_left(alts, pa_ft)
        if idx == 0: return self.tff_by_alt[alts[0]]
        if idx == len(alts): return self.tff_by_alt[alts[-1]]
        before = alts[idx-1]; after = alts[idx]
        return self.tff_by_alt[before] if (pa_ft - before) <= (after - pa_ft) else self.tff_by_alt[after]

    def thrust_to_ff(self, T_req_lbf: float, pa_ft: float) -> float:
        p = self._nearest_alt_point(pa_ft)
        T_req_lbf = max(T_req_lbf, 0.0)
        FF = (T_req_lbf / max(p.C, 1e-6)) ** (1.0 / max(p.alpha, 1e-6))
        FF = max(FF, float(self.cfg.get("min_idle_ff_pph", 1000)))
        return FF

    def ff_to_rpm(self, ff_pph: float) -> float:
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
        m_map = self.cfg.get("thrust_exponent_m", {})
        m = float(m_map.get(str(int(flap_deg)), m_map.get("0", 0.75)))
        m = max(min(m, 2.0), 0.2)
        gr_mil = max(mil_ground_roll_ft, 1.0)
        gr_target = max(target_ground_roll_ft, 1.0)
        mult = (gr_mil / gr_target) ** (1.0 / m)
        return max(mult, 0.0)

    def _min_pct_for_flaps(self, flap_deg: int) -> int:
        table = self.cfg.get("min_pct_by_flap_deg", {"0": 85, "35": 90})
        return int(table.get(str(int(flap_deg)), 85))

    def compute_derate_from_groundroll(
        self,
        flap_deg: int,
        pa_ft: float,
        mil_ground_roll_ft: float,
        runway_available_ft: float,
        allow_ab: Optional[bool] = None
    ) -> Dict[str, float]:
        if allow_ab is None: allow_ab = bool(self.cfg.get("allow_ab", False))
        margin = float(self.cfg.get("safety", {}).get("runway_margin_ft", 0.0))
        target = max(runway_available_ft - margin, 1.0)

        mult = self.solve_thrust_multiplier_from_groundroll(mil_ground_roll_ft, target, flap_deg)

        p = self._nearest_alt_point(pa_ft)
        floor_pct = self._min_pct_for_flaps(flap_deg)

        pct = int(round(mult * 100.0))
        clamped = False

        if not allow_ab:
            pct = min(pct, 100)

        if pct < floor_pct:
            pct = floor_pct
            clamped = True

        mult = pct / 100.0
        T_required = mult * p.T_MIL_lbf

        FF_required = self.thrust_to_ff(T_required, pa_ft)
        RPM_required = self.ff_to_rpm(FF_required)

        return {
            "thrust_multiplier": float(mult),
            "derate_pct": int(pct),
            "clamped_to_floor": bool(clamped),
            "T_required_lbf": float(T_required),
            "FF_required_pph": float(FF_required),
            "RPM_required_pct": float(RPM_required),
            "T_MIL_lbf": float(p.T_MIL_lbf),
            "FF_MIL_pph": float(p.FF_MIL_pph),
            "alpha_used": float(p.alpha),
            "alt_used_ft": int(p.alt_ft),
        }
