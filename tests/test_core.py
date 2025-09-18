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
