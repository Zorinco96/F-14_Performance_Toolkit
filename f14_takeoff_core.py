Below are the drop‑in files for **M1 — core skeleton + tests**. Paste each into your GitHub repo via **Add file → Create new file**. The stubs return deterministic placeholders so the UI keeps working. Replace math later in M2–M4.

---

## `f14_takeoff_core.py`

```python
# f14_takeoff_core.py — v1.2.0-core-skel
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

# ---- Constants (placeholders; refine later) ----
F14B_BEW_LB = 43735.0  # basic empty weight placeholder
EXT_TANK_EMPTY_LB = 1100.0  # each, structural weight only
ISA_LAPSE_C_PER_FT = 1.98 / 1000.0

# -------- Atmosphere / conversions --------
def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: Optional[float]) -> float:
    """Compute pressure altitude from field elevation and QNH inHg.
    PA(ft) ≈ elev + (29.92 - QNH)*1000. If QNH is None, assume 29.92 (=> PA=elev).
    """
    if qnh_inhg is None:
        return float(field_elev_ft)
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0


def density_ratio_sigma(press_alt_ft: float, oat_c: float) -> float:
    """ISA-based density ratio σ = δ/θ (troposphere approx).
    Uses ft-based approximation: δ ≈ (1 - 6.87535e-6 * h)^4.2561 ;
    θ = (T/T0) with T0=288.15 K and T = (ISA at PA) adjusted to OAT.
    This is a pragmatic placeholder adequate for v1.2 scaffolding.
    """
    h = max(0.0, float(press_alt_ft))
    delta = (1.0 - 6.87535e-6 * h) ** 4.2561
    # ISA temp at altitude (°C)
    isa_c = 15.0 - ISA_LAPSE_C_PER_FT * h
    theta = (float(oat_c) + 273.15) / (isa_c + 273.15)
    sigma = max(0.2, min(1.2, delta / theta))
    return sigma


def tas_from_ias_kt(ias_kt: float, sigma: float) -> float:
    """Approx TAS from IAS and density ratio: TAS ≈ IAS / sqrt(σ)."""
    s = max(0.1, float(sigma))
    return float(ias_kt) / math.sqrt(s)


# -------- Weight & Balance --------
def build_loadout_totals(
    stations: Dict[str, Dict[str, object]],
    fuel_lb: float,
    ext_tanks: Tuple[bool, bool],
    mode_simple_gw: Optional[float] = None,
) -> Dict[str, float]:
    """
    Aggregate gross weights. Placeholder CG/trim mapping.

    stations: { name: { 'store_weight_lb': float, 'pylon_weight_lb': float, 'qty': int } }
    ext_tanks: (left_has_tank, right_has_tank) — structural weight only here.

    Returns keys:
        'gw_tow_lb','gw_ldg_lb','zf_weight_lb','fuel_tow_lb','fuel_ldg_lb',
        'cg_percent_mac','stab_trim_units'
    """
    if mode_simple_gw is not None:
        gw_tow = max(0.0, float(mode_simple_gw))
        # Assume simple-mode fuel is as provided; landing fuel placeholder 3000 lb
        fuel_tow = max(0.0, float(fuel_lb))
        fuel_ldg = min(fuel_tow, 3000.0)
        zfw = gw_tow - fuel_tow
    else:
        # Sum detailed station weights
        stores = 0.0
        pylons = 0.0
        for _, d in stations.items():
            qty = int(d.get('qty', 0) or 0)
            stores += qty * float(d.get('store_weight_lb', 0.0) or 0.0)
            pylons += (qty > 0) * float(d.get('pylon_weight_lb', 0.0) or 0.0)
        tank_struct = (EXT_TANK_EMPTY_LB if ext_tanks[0] else 0.0) + (EXT_TANK_EMPTY_LB if ext_tanks[1] else 0.0)
        zfw = F14B_BEW_LB + stores + pylons + tank_struct
        fuel_tow = max(0.0, float(fuel_lb))
        fuel_ldg = min(fuel_tow, 3000.0)  # project rule: ext tanks empty at landing, reserve ~3k placeholder
        gw_tow = zfw + fuel_tow
    gw_ldg = zfw + fuel_ldg

    # Placeholder CG%MAC and stab trim as a function of GW (to be replaced by station-moment model)
    cg = 20.0 + (gw_tow - 54000.0) * 0.0005  # gentle slope
    cg = max(15.0, min(25.0, cg))
    trim = 10.0 + (cg - 20.0) * 0.6

    return {
        'gw_tow_lb': gw_tow,
        'gw_ldg_lb': gw_ldg,
        'zf_weight_lb': zfw,
        'fuel_tow_lb': fuel_tow,
        'fuel_ldg_lb': fuel_ldg,
        'cg_percent_mac': cg,
        'stab_trim_units': trim,
    }


# -------- Performance speeds (placeholder scaling) --------
def _vs_from_weight(gw_lb: float, flaps_mode: str) -> float:
    base = 110.0  # Vs at ~60k for ref config
    k = math.sqrt(max(30000.0, gw_lb) / 60000.0)
    flap_bias = {
        'AUTO': -2.0,
        'FULL': -6.0,
        '5°': -3.0,
        '10°': -5.0,
        '20°': -7.5,
    }.get(str(flaps_mode).upper(), -2.0)
    return max(90.0, base * k + flap_bias)


def takeoff_speeds(gw_lb: float, flaps_mode: str, press_alt_ft: float, oat_c: float) -> Dict[str, float]:
    """Return plausible, ordered speeds until real interpolation is wired (M2)."""
    vs = _vs_from_weight(gw_lb, flaps_mode)
    v1 = vs + 10.0
    vr = vs + 15.0
    v2 = vs + 25.0
    vfs = vs + 35.0
    # enforce ordering
    v1 = min(v1, vr)
    return {'V1': v1, 'Vr': vr, 'V2': v2, 'Vfs': vfs, 'Vs': vs}


# -------- Field length (placeholder) --------
def takeoff_distances(
    gw_lb: float,
    flaps_mode: str,
    thrust_mode: str,
    derate_pct: int,
    press_alt_ft: float,
    oat_c: float,
    headwind_kt: float,
    runway_condition: str = "DRY",
) -> Dict[str, float]:
    """Return deterministic placeholder distances showing sensible trends.
    Trends: heavier/hotter/higher -> longer; more headwind -> shorter; WET > DRY; DERATE -> longer; AB -> shorter.
    """
    w = max(40000.0, min(74000.0, float(gw_lb)))
    pa = max(0.0, float(press_alt_ft))
    temp = float(oat_c)
    # Base model ~ quadratic in weight with altitude/temp penalties
    base = 6.0e-4 * (w - 40000.0) ** 2 + 0.15 * pa + 12.0 * max(0.0, temp - 15.0) + 2500.0

    # Flaps effect (shorter with more flap within reason)
    flap_f = {
        'AUTO': 1.00,
        'FULL': 0.90,
        '5°': 0.97,
        '10°': 0.94,
        '20°': 0.92,
    }.get(str(flaps_mode).upper(), 1.00)

    # Thrust / derate effect
    thrust = str(thrust_mode).upper()
    if thrust == 'AB':
        thrust_f = 0.85
    elif thrust == 'MIL':
        thrust_f = 1.00
    elif thrust == 'AUTO':
        thrust_f = 0.97
    else:  # 'DERATE (MANUAL)' path
        thrust_f = 1.00 * (100.0 / max(85.0, min(100.0, float(derate_pct))))

    # Runway condition
    cond = str(runway_condition).upper()
    wet_f = 1.00 if cond == 'DRY' else 1.15

    # Headwind credit (cap tailwind to zero credit)
    hw = float(headwind_kt)
    hw_effect = max(-5.0, min(25.0, hw))  # cap credit
    hw_f = 1.0 - 0.01 * hw_effect  # ~1% per kt up to 25 kt headwind

    tor = max(500.0, base * flap_f * thrust_f * wet_f * hw_f)
    asdr = tor * 1.15
    agd = tor * 1.05
    rdr = tor * 0.95

    rotate_ft = tor * 0.85
    liftoff_35 = tor * 1.15
    return {
        'asdr_ft': asdr,
        'agd_ft': agd,
        'rdr_ft': rdr,
        'tor_ft': tor,
        'rotate_ft': rotate_ft,
        'liftoff_35ft_ft': liftoff_35,
    }


# -------- Climb model (placeholder) --------
def climb_profile_metrics(
    gw_lb: float,
    profile: str,
    press_alt_ft: float,
    oat_c: float,
    ignore_250kt: bool,
    cruise_alt_ft: float,
) -> Dict[str, float]:
    """Return simple, self-consistent placeholder climb metrics."""
    w = max(40000.0, min(74000.0, float(gw_lb)))
    pa = max(0.0, float(press_alt_ft))
    crz = max(10000.0, float(cruise_alt_ft))

    # Profile knobs
    p = str(profile).lower()
    speed_bias = 0.9 if 'minimum time' in p else 1.0

    base_rate_fpm = 6000.0 * (60000.0 / w) ** 0.5 * (1.0 - 0.00002 * pa)
    time_to_10k = (10000.0 / max(1500.0, base_rate_fpm)) * speed_bias

    # Integrate to cruise (very rough)
    avg_rate_to_crz = max(1200.0, base_rate_fpm * 0.6)
    time_to_crz = ((crz - max(0.0, pa)) / avg_rate_to_crz) * speed_bias
    fuel_to_toc = 800.0 + 0.02 * (w - 50000.0) + 0.03 * (crz - pa)  # lb
    toc_distance = max(8.0, (time_to_crz * 160.0 / 60.0))  # nm, assuming ~160 kt avg climb GS

    # To TO + 100nm
    time_to_to_plus_100nm = (100.0 / 280.0) * 60.0 + time_to_crz  # assume 280 kt GS in climb/cruise blend
    fuel_to_to_plus_100nm = fuel_to_toc + 600.0

    # Initial AEO gradient (ft/nm), scales with weight inversely
    climb_grad = 400.0 * (60000.0 / w)  # placeholder

    return {
        'time_to_10k_min': time_to_10k,
        'time_to_crz_min': time_to_crz,
        'fuel_to_toc_lb': fuel_to_toc,
        'toc_distance_nm': toc_distance,
        'time_to_to_plus_100nm_min': time_to_to_plus_100nm,
        'fuel_to_to_plus_100nm_lb': fuel_to_to_plus_100nm,
        'climb_gradient_aeo_ft_per_nm': climb_grad,
    }


# -------- Landing performance (placeholder) --------
def landing_metrics(
    gw_lb: float,
    lda_ft: float,
    condition: str,
    retained_stores: bool,
) -> Dict[str, float]:
    """Return simple landing speeds and distances with factored LDR.
    121.195 factors placeholder: 1.67 dry / 1.92 wet.
    """
    w = max(38000.0, min(66000.0, float(gw_lb)))
    vs = 110.0 * math.sqrt(w / 60000.0)
    vref = 1.3 * vs
    vapp = vref + 5.0
    vac = max(120.0, vref + 20.0)
    vfs = vref + 35.0

    # Unfactored LDR baseline (scales with W^1.7)
    ldr_unfact = 1800.0 + 0.000015 * (w ** 1.7)
    cond = str(condition).upper()
    factor = 1.67 if cond == 'DRY' else 1.92
    ldr_fact = ldr_unfact * factor

    # Compute MLW allowed by LDA (inverse)
    # Solve w from ldr_unfact * factor <= lda -> (w^1.7) <= (lda - 1800)/k
    k = 0.000015
    rhs = max(1.0, (float(lda_ft) / factor) - 1800.0)
    mlw = min(60000.0, (rhs / k) ** (1.0 / 1.7))

    return {
        'vref_kt': vref,
        'vapp_kt': vapp,
        'vac_kt': vac,
        'vfs_kt': vfs,
        'vs_kt': vs,
        'ldr_unfactored_ft': ldr_unfact,
        'ldr_factored_ft': ldr_fact,
        'lda_ft': float(lda_ft),
        'calc_mlw_lb': mlw,
    }


# -------- Validation / safety --------
def dispatchability(tora_ft: float, asdr_ft: float, agd_ft: float, rdr_ft: float) -> Dict[str, str]:
    """Compare required vs available and return limiter with simple logic."""
    tora = float(tora_ft)
    reqs = {'ASD': float(asdr_ft), 'AGD': float(agd_ft), 'RDR': float(rdr_ft)}
    limiting = max(reqs.items(), key=lambda kv: kv[1])[0]
    ok = all(req <= tora for req in reqs.values())
    note = 'All requirements within TORA' if ok else f'Limited by {limiting}: requires {reqs[limiting]:.0f} ft vs TORA {tora:.0f} ft'
    return {'dispatchable': 'YES' if ok else 'NO', 'limiter': limiting, 'note': note}
```

---

## `tests/test_core.py`

```python
# tests/test_core.py — v1.2.0-core-skel
import math
import f14_takeoff_core as core


def test_pressure_altitude():
    assert core.pressure_altitude_ft(1000, 29.92) == 1000
    assert core.pressure_altitude_ft(0, 28.92) == 1000


def test_density_ratio_sigma_bounds():
    s1 = core.density_ratio_sigma(0, 15)
    s2 = core.density_ratio_sigma(8000, 35)
    assert 0.2 <= s1 <= 1.2
    assert 0.2 <= s2 <= 1.2


def test_tas_from_ias():
    tas = core.tas_from_ias_kt(200, 0.8)
    assert tas > 200


def test_build_loadout_totals_shape():
    stations = {
        'STA1': {'store_weight_lb': 500, 'pylon_weight_lb': 80, 'qty': 2},
        'STA2': {'store_weight_lb': 0, 'pylon_weight_lb': 60, 'qty': 0},
    }
    out = core.build_loadout_totals(stations, fuel_lb=9000, ext_tanks=(True, False))
    for k in ['gw_tow_lb','gw_ldg_lb','zf_weight_lb','fuel_tow_lb','fuel_ldg_lb','cg_percent_mac','stab_trim_units']:
        assert k in out
    assert out['gw_tow_lb'] >= out['gw_ldg_lb']


def test_takeoff_speeds_ordering():
    s = core.takeoff_speeds(70000, 'AUTO', 1000, 20)
    assert set(s.keys()) == {'V1','Vr','V2','Vfs','Vs'}
    assert s['V1'] <= s['Vr'] < s['V2'] < s['Vfs']
    assert s['Vs'] < s['Vr']


def test_takeoff_distances_sensible():
    d = core.takeoff_distances(70000, 'AUTO', 'MIL', 100, 0, 15, headwind_kt=10, runway_condition='DRY')
    for k in ['asdr_ft','agd_ft','rdr_ft','tor_ft','rotate_ft','liftoff_35ft_ft']:
        assert k in d and d[k] > 0


def test_climb_profile_metrics_shape():
    c = core.climb_profile_metrics(70000, 'Most efficient', 0, 15, False, 28000)
    keys = ['time_to_10k_min','time_to_crz_min','fuel_to_toc_lb','toc_distance_nm',
            'time_to_to_plus_100nm_min','fuel_to_to_plus_100nm_lb','climb_gradient_aeo_ft_per_nm']
    for k in keys:
        assert k in c
        assert c[k] >= 0


def test_landing_metrics_shape():
    L = core.landing_metrics(60000, 7000, 'DRY', True)
    assert L['ldr_factored_ft'] >= L['ldr_unfactored_ft']
    assert L['vref_kt'] > 0 and L['vs_kt'] > 0


def test_dispatchability_flag():
    d = core.dispatchability(8000, 7500, 5000, 4000)
    assert d['dispatchable'] in ('YES','NO')
    assert d['limiter'] in ('ASD','AGD','RDR')
```

