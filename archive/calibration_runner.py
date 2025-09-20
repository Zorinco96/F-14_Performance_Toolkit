
from __future__ import annotations
import pandas as pd, numpy as np, json, sys
from data_loaders import load_calibration_csv
from takeoff_model import takeoff_run

FIT_PARAMS = ["RollingResistance","RotationLag_sec","GearDragDelta_CD0","ThrustScaleLowAlt"]

def run_calibration(test_csv_path: str, mode="DCS"):
    data = pd.read_csv(test_csv_path)
    calib = load_calibration_csv("calibration.csv")
    x = np.array([calib["RollingResistance"][mode], calib["RotationLag_sec"][mode], calib["GearDragDelta_CD0"][mode], calib["ThrustScaleLowAlt"][mode]], dtype=float)

    def clamp(x):
        mins = np.array([0.005, 0.5, 0.004, 0.85])
        maxs = np.array([0.060, 4.0, 0.030, 1.10])
        return np.minimum(maxs, np.maximum(mins, x))

    def error_vector(x):
        local = {k: {mode: v} for k,v in zip(FIT_PARAMS, x)}
        errs = []
        for _, r in data.iterrows():
            res = takeoff_run(
                weight_lbf=r.Weight_lb, alt_ft=r.Elev_ft, oat_c=r.OAT_C,
                headwind_kts=r.Wind_kts, runway_slope=r.Slope,
                config="TO_FLAPS", sweep_deg=20.0, power=r.Power,
                mode=mode, calib=local
            )
            e_vr = (res["VR_kts"] - r.Measured_VR_kts)
            e_35 = (res["DistanceTo35ft_ft"] - r.Measured_35ft_ft)/100.0
            errs.extend([e_vr, e_35])
        return np.array(errs)

    for _ in range(10):
        x = clamp(x)
        f0 = error_vector(x)
        J = []
        eps = 1e-4
        for i in range(len(x)):
            x2 = x.copy(); x2[i] += eps
            fi = error_vector(x2)
            J.append((fi - f0)/eps)
        J = np.vstack(J).T
        dx, *_ = np.linalg.lstsq(J, -f0, rcond=None)
        x += dx
        if np.linalg.norm(dx) < 1e-4: break

    return {k: float(v) for k,v in zip(FIT_PARAMS, x)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibration_runner.py Test_Results.csv")
        sys.exit(1)
    fitted = run_calibration(sys.argv[1], mode="DCS")
    print(json.dumps(fitted, indent=2))
