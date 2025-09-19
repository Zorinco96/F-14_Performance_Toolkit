# F-14B Performance Calculator (DCS World)

A Streamlit-based EFB-style application to compute **takeoff, climb, and landing performance** for the F-14B Tomcat in DCS World.  
Supports all maps, full airport/runway/intersection data, and both **manual** and **auto-select** configuration modes.

---

## Features
- **Global runway search** across all DCS maps (with intersections, margins, alternates).
- **W&B panel**: simple and detailed loadouts, preset stores, compatibility filter, CG/trim output.
- **Environment panel**: manual input or paste directly from DCS briefing.
- **Takeoff results**:
  - V-speeds (Vr, Vlof, V2, Vfs)
  - Ground roll / Distance to 35 ft
  - Stabilizer trim
  - N1% / FF (pph/engine) guidance
  - Dispatchability with AEO/OEI climb gates
- **Climb profile** overlay (AEO vs OEI schedules).
- **Landing results** per destination/alternate, with unfactored/factored distances.
- **Performance engine** calibrated against in-sim test runs (`f14_perf.csv`, `Test Results.csv`).
- **Modes**: DCS vs FAA calibration.
- **Caching** and performance optimizations keep recomputes < 2.5 s.

---

## App Scope Definition

The **auto-select algorithm** (frozen in `v1.1.3-auto-select-slow`) enforces the following rules and policies:

### Thrust & Flaps Policy
- **MILITARY** = 100% RPM, no afterburner.
- **DERATE** = 85–99% RPM (band depends on flap setting):
  - UP: ≥85%
  - MANEUVER: ≥90%
  - FULL: ≥98%
- **Afterburner**: evaluated only for “would pass” annotation, **prohibited** for dispatch.
- **Preference order**: UP → MAN → FULL.
- **Tie-breaker**: prefer **MAN DERATE** (≥90%) over **UP MIL** if both pass.

### Balanced Field Check
- V1 sweep (two-phase):
  - Coarse: 100,110,120,130,140,150 kt
  - Refine: ±5 kt around best, 2-kt steps
- **Balanced** if |ASDR−TODR| / max ≤ 1%.
- Governing side shown otherwise.
- **ASDR ≤ ASDA**, **TODR ≤ TORA** required.

> v1.1.3 used proxies if ASDR/TODR not exposed:  
> ASDR ≈ 1.10×GroundRoll (fallback 1.05×Dist35), TODR ≈ Dist35.

### Climb Gates
- **AEO**: ≥200 ft/NM to 1000 ft AFE (at selected thrust).
- **OEI** (always MIL thrust):
  - Second segment: ≥2.4% gross (gear up, TO flaps, V2)
  - Final segment: ≥1.2% gross (gear up, flaps UP, Vfs)
- Net = gross − 0.8% (displayed for reference).

### Wind Policy
- Selectable in sidebar:
  - **50/150**: 50% headwind credit, 150% tailwind penalty (default).
  - **0/150**: ignore headwind, 150% tailwind penalty (conservative).
- Policy applied consistently to both runway distances and climb gradients.

### Display Rules
- **Configuration**: always shows resolved flaps (UP/MAN/FULL) and thrust (DERATE % or MIL).
- **Balanced/Governing badge** shown in Field Length card.
- **Dispatchability card** shows AEO and OEI checks with OK/FAIL badges.

---

## Performance Optimizations
- Cached wrappers for `perf_compute_takeoff` and `perf_compute_climb`.
- In-memory memoization of V1 sweep results.
- Coarse→refine V1 sweep (~10–12 perf calls per candidate).
- Core integrator step `dt=0.1` (vs 0.05).
- Typical AUTO recompute: <2.5 s.

---

## Validation Scenarios
- **Scenario A (easy field)**: UP + min derate.
- **Scenario B (tie-break)**: UP@100 MIL vs MAN@≥90 derate → select MAN DERATE.
- **Scenario C (short field)**: FULL + min derate (≥98%).
- **Scenario D (derates fail, MIL passes)**: lowest flap MIL that passes.
- **Scenario E (only AB passes)**: Not Dispatchable + note (AB prohibited).

---

## Repository Layout
- `f14_takeoff_app.py` — Streamlit UI (main app).
- `f14_takeoff_core.py` — performance math & selection logic.
- `f14_perf.csv` — baseline performance data.
- `Test Results.csv` — in-sim test calibration results.
- `dcs_airports_expanded.csv` — full DCS airport/runway database.
- `intersections.csv` — verified runway intersection data.
- `requirements.txt` — Python dependencies.

---

## Notes
- When you say **“App Scope Definition”** in future dev sessions, this exact spec is referenced.
- Future versions (≥v1.1.4) should replace proxy ASDR/TODR with true values from the core and wire N1/FF guidance tables to actual perf data.
