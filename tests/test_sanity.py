# tests/test_sanity.py — minimal smoke test
import importlib, types, pathlib

def test_imports_and_symbols():
    m = importlib.import_module("f14_takeoff_core")
    assert isinstance(m, types.ModuleType)
    for name in [
        "perf_compute_takeoff",
        "auto_select_flaps_thrust",
        "plan_takeoff_with_optional_derate",
        "compute_derate_for_run",
    ]:
        assert hasattr(m, name), f"missing symbol: {name}"

def test_data_folder_presence():
    m = importlib.import_module("f14_takeoff_core")
    data_dir = pathlib.Path(getattr(m, "DATA_DIR", pathlib.Path(__file__).parents[1] / "data"))
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"
    # Soft-check (informational): list any common files if present
    common = [
        "f14_perf.csv",
        "f14_perf_calibrated_SL_overlay.csv",
        "calibration_sl_summary.csv",
        "f110_engine.csv",
        "f110_tff_model.csv",
        "f110_ff_to_rpm_knots.csv",
        "dcs_airports.csv",
        "dcs_airports_expanded.csv",
        "intersections.csv",
    ]
    found = [p for p in common if (data_dir / p).exists()]
    # Print to pytest output (doesn't fail if some are missing)
    print("Existing data files:", found)
