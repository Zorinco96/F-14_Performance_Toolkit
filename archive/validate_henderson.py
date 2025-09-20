
from __future__ import annotations
from data_loaders import load_calibration_csv
from takeoff_core_min import ground_takeoff_run

def main():
    calib = load_calibration_csv("calibration.csv")
    # Henderson 35L example (user mentioned 70,000 lb, 15 C)
    result = ground_takeoff_run(
        weight_lbf=70000,
        alt_ft=2410,          # KHND field elev ~2410 ft
        oat_c=15.0,
        headwind_kts=0.0,
        runway_slope=0.0,
        config="TO_FLAPS",
        sweep_deg=20.0,
        power="MAX",
        mode="DCS",
        calib=calib,
        vr_kts=None,          # let model estimate; can override with test data
        dt=0.1
    )
    for k,v in result.items():
        print(f"{k}: {v:.2f}" if isinstance(v,(int,float)) else f"{k}: {v}")

if __name__ == "__main__":
    main()
