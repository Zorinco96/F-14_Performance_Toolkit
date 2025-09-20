# Minimal smoke test for F-14 Performance Toolkit core
# Assumes repo structure:
#   - f14_takeoff_core.py  (this module)
#   - data/
#       f14_perf.csv
#       (optional) f14_perf_calibrated_SL_overlay.csv
#       (optional) calibration_sl_summary.csv
#       (optional) f110_tff_model.csv, f110_ff_to_rpm_knots.csv, derate_config.json

import os
import sys

# Allow running from repo root or from tests/ directory
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import f14_takeoff_core as core  # noqa: E402

def test_import_and_data_loading():
    # Core imports OK?
    assert hasattr(core, "__version__")
    print("core version:", core.__version__)

    # Data dir exists
    data_dir = os.path.join(ROOT, "data")
    assert os.path.isdir(data_dir), "Expected ./data folder"

    # Baseline perf CSV should be present
    perf_csv = os.path.join(data_dir, "f14_perf.csv")
    assert os.path.isfile(perf_csv), "Expected data/f14_perf.csv"

    # Call the top-level planning function with conservative, generic inputs
    result = core.plan_takeoff_with_optional_derate(
        flap_deg=35,
        gw_lbs=65000.0,
        field_elev_ft=39.0,       # Batumi elevation example
        qnh_inhg=29.92,
        oat_c=15.0,
        headwind_kts_component=0.0,
        runway_slope=0.0,
        tora_ft=8000,
        asda_ft=8000,
        allow_ab=False,
        do_derate=True,
    )

    assert isinstance(result, dict) and "baseline_MIL" in result
    # Not all repos will have derate assets—this is optional
    if result.get("derate") is not None:
        d = result["derate"]
        assert "RPM_required_pct" in d or "FF_required_pph" in d

    print("Smoke test OK. Keys:", list(result.keys()))
