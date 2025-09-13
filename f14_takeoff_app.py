# f14_takeoff_app_dcs.py
import os, math
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "F-14 Takeoff Performance (DCS • Part 121 • Dynamic Min Thrust)"

# ---------------- Core aircraft / tuning ----------------
EMPTY_WEIGHT_LBS = 40000
MAX_GW = 74349
MIN_GW = 30000

# Engine %RPM anchors
RPM_MIN = 88.0     # hard lower bound for reduced thrust search
RPM_MIL = 96.0
RPM_AB  = 102.0

# Thrust label thresholds
AB_THRESH = 101.0
MIL_MIN   = 96.0
def classify_thrust(rpm_value: float) -> str:
    if rpm_value is None: return "Military"
    if rpm_value >= AB_THRESH: return "Afterburner"
    if rpm_value >= MIL_MIN:   return "Military"
    return "Reduced"

# Flaps
FLAP_OPTIONS = {"Flaps Up": 0, "Maneuvering Flaps": 20, "Flaps Full": 40}
TAILWIND_LIMIT_KT = 10

def ceil_kn(x): return int(math.ceil(float(x)))

# -------- Below-MIL extrapolation knobs (more conservative) --------
BELOW_MIL_DISTANCE_BETA = 2.2   # steeper distance growth below MIL
BELOW_MIL_SPEED_BETA    = 0.30  # gentler speed decrease
MAX_SPEED_DROP_FRAC     = 0.05  # ≤5% below MIL

# ---------------- NATOPS tables (starter) ----------------
STARTER_PERF_CSV = """model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft
F-14D,20,Military,60000,0,15,122,135,144,156,4200,4700
F-14D,20,Afterburner,60000,0,15,120,132,141,154,3800,3900
F-14D,20,Military,70000,0,15,128,142,152,166,5200,6100
F-14D,20,Afterburner,70000,0,15,126,139,148,163,4600,4700
F-14D,40,Military,70000,0,15,118,131,140,152,4800,5600
F-14D,40,Afterburner,70000,0,15,116,128,137,150,4300,4400
F-14D,20,Military,70000,2000,30,132,146,157,172,6400,7300
F-14D,20,Afterburner,70000,2000,30,130,143,154,169,5600,5900
"""

@st.cache_data(show_spinner=False)
def ensure_starter_perf():
    if not os.path.exists("perf_f14b.csv") and not os.path.exists("perf_f14d.csv"):
        with open("perf_f14d.csv", "w", encoding="utf-8") as f:
            f.write(STARTER_PERF_CSV)

@st.cache_data(show_spinner=False)
def load_perf_tables():
    for fname in ["perf_f14b.csv", "perf_f14d.csv"]:
        try:
            df = pd.read_csv(fname)
            df["flap_deg"] = pd.to_numeric(df["flap_deg"], errors="coerce")
            df["gw_lbs"]   = pd.to_numeric(df["gw_lbs"], errors="coerce")
            df["press_alt_ft"] = pd.to_numeric(df["press_alt_ft"], errors="coerce")
            df["oat_c"]    = pd.to_numeric(df["oat_c"], errors="coerce")
            for c in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["thrust"] = df["thrust"].map({
                "Minimum Required":"Minimum Required","minimum":"Minimum Required","min":"Minimum Required",
                "Military":"Military","military":"Military","mil":"Military",
                "Afterburner":"Afterburner","afterburner":"Afterburner","ab":"Afterburner"
            })
            return df, fname
        except Exception:
            continue
    return None, None

def nearest_row(df, flap_deg, thrust, gw_lbs, pa_ft, oat_c):
    sub = df[(df["flap_deg"]==flap_deg) & (df["thrust"]==thrust)].copy()
    if sub.empty:
        flaps = df[df["thrust"]==thrust]["flap_deg"].dropna().unique()
        if len(flaps)==0: return None
        flap_deg = int(sorted(flaps, key=lambda x:abs(x-flap_deg))[0])
        sub = df[(df["flap_deg"]==flap_deg) & (df["thrust"]==thrust)].copy()
        if sub.empty: return None
    sub["dist"] = (abs(sub["gw_lbs"]-gw_lbs)
                   + abs(sub["press_alt_ft"]-pa_ft)
                   + abs(sub["oat_c"]-oat_c))
    row = sub.sort_values("dist").head(1)
    return None if row.empty else row.iloc[0].to_dict()

# -------- Interp/Extrap (MIL↔AB and below-MIL) --------
def blend_metric(mil_val, ab_val, alpha, kind="distance"):
    if mil_val is None or ab_val is None or (isinstance(mil_val,float) and math.isnan(mil_val)) or (isinstance(ab_val,float) and math.isnan(ab_val)):
        return mil_val
    if alpha >= 0.0:
        return mil_val + alpha*(ab_val - mil_val)
    a = -alpha
    if kind == "distance":
        if ab_val > 0 and mil_val > 0 and mil_val > ab_val:
            k = math.log(mil_val/ab_val)
            return mil_val * math.exp(k * a * BELOW_MIL_DISTANCE_BETA)
        return mil_val * (1.0 + a * 2.0 * BELOW_MIL_DISTANCE_BETA)
    else:  # speed
        delta = (mil_val - ab_val)
        raw = mil_val + (-a) * BELOW_MIL_SPEED_BETA * delta
        min_allowed = mil_val * (1.0 - MAX_SPEED_DROP_FRAC)
        return max(raw, min_allowed)

def rpm_from_alpha(alpha):
    if alpha >= 0:
        return RPM_MIL + alpha*(RPM_AB - RPM_MIL)
    return RPM_MIL + alpha*(RPM_MIL - RPM_MIN)

def alpha_from_rpm(rpm):
    rpm = max(RPM_MIN, min(RPM_AB, rpm))
    if rpm >= RPM_MIL:
        return (rpm - RPM_MIL) / (RPM_AB - RPM_MIL)
    return (rpm - RPM_MIL) / (RPM_MIL - RPM_MIN)

# ---------------- Part 121 gating ----------------
def part121_pass(runway_ft, asd_ft, agd_ft, condition, stopway_ft=0, clearway_ft=0, extra_margin_ft=0):
    tora = float(runway_ft)
    asda = tora + float(stopway_ft)
    toda = tora + float(min(clearway_ft, tora/2.0))
    # Core checks
    asd_ok = float(asd_ft) + extra_margin_ft <= asda            # include user margin
    oei_tod_ok = float(agd_ft) + extra_margin_ft <= toda        # include user margin
    aeo_tr_proxy = (0.85 if condition=="Dry" else 0.90) * float(agd_ft)
    tor_ok = aeo_tr_proxy + extra_margin_ft <= tora             # include user margin
    return asd_ok and oei_tod_ok and tor_ok

def find_min_feasible_alpha(check_fn, alpha_hi, alpha_min_bound):
    if not check_fn(alpha_hi): return None
    step = 0.05
    a_pass = alpha_hi
    a = alpha_hi
    while a > alpha_min_bound:
        a_next = max(alpha_min_bound, a - step)
        if check_fn(a_next):
            a_pass = a_next; a = a_next
        else:
            break
        if a == alpha_min_bound: break
    left = max(alpha_min_bound, a_pass - step)
    right = a_pass
    if check_fn(left): left = max(alpha_min_bound - 1e-6, left - 0.001)
    for _ in range(32):
        mid = 0.5*(left+right)
        if check_fn(mid): right = mid
        else: left = mid
    return right

def evaluate_combo_for_121(perf_df, flap_deg, gw, pa_ft, oat_c, runway_ft, condition, stopway_ft, clearway_ft, extra_margin_ft=0):
    mil = nearest_row(perf_df, flap_deg, "Military", gw, pa_ft, oat_c)
    ab  = nearest_row(perf_df, flap_deg, "Afterburner", gw, pa_ft, oat_c)
    if not mil or not ab: return None

    def metrics_at(alpha):
        asd = blend_metric(mil["ASD_ft"], ab["ASD_ft"], alpha, "distance") * 1.15
        agd = blend_metric(mil["AGD_ft"], ab["AGD_ft"], alpha, "distance") * 1.15
        vs  = ceil_kn(blend_metric(mil["Vs_kt"], ab["Vs_kt"], alpha, "speed"))
        v1  = ceil_kn(blend_metric(mil["V1_kt"], ab["V1_kt"], alpha, "speed"))
        vr  = ceil_kn(blend_metric(mil["Vr_kt"], ab["Vr_kt"], alpha, "speed"))
        v2  = ceil_kn(blend_metric(mil["V2_kt"], ab["V2_kt"], alpha, "speed"))
        rpm = round(max(RPM_MIN, min(RPM_AB, rpm_from_alpha(alpha))), 1)
        return dict(Vs=vs,V1=v1,Vr=vr,V2=v2,
                    stop_ft=int(round(asd)), go_ft=int(round(agd)),
                    bfl_ft=int(round(max(asd, agd))), rpm=rpm, alpha=alpha)

    def pass_121(alpha):
        m = metrics_at(alpha)
        return part121_pass(runway_ft, m["stop_ft"], m["go_ft"], condition, stopway_ft, clearway_ft, extra_margin_ft)

    alpha_min = (RPM_MIN - RPM_MIL) / (RPM_AB - RPM_MIL)
    seeds = [0.0, 0.5, 1.0, -0.1, -0.2]
    a_pass = None
    for s in seeds:
        if s >= alpha_min and pass_121(s):
            a_pass = s; break
    if a_pass is None: return None

    a_min = find_min_feasible_alpha(pass_121, a_pass, alpha_min)
    if a_min is None: a_min = a_pass
    return metrics_at(a_min)

def solve_min_required_dynamic(perf_df, flap_deg, gw_lbs, pa_ft, oat_c, runway_available_ft, extra_margin_ft=0):
    # This solver enforces Vr by 60% of runway (after margin), with +15% safety baked in.
    mil = nearest_row(perf_df, flap_deg, "Military", int(gw_lbs), int(pa_ft), int(oat_c))
    ab  = nearest_row(perf_df, flap_deg, "Afterburner", int(gw_lbs), int(pa_ft), int(oat_c))
    if not mil or not ab: return None
    usable = max(0.0, float(runway_available_ft) - float(extra_margin_ft))
    target_agd = 0.60 * usable
    def agd_of(alpha):
        return blend_metric(mil["AGD_ft"], ab["AGD_ft"], alpha, "distance") * 1.15
    if agd_of(1.0) > target_agd:
        return None
    alpha_min = (RPM_MIN - RPM_MIL) / (RPM_AB - RPM_MIL)
    lo, hi = (alpha_min, 0.0) if agd_of(0.0) <= target_agd else (0.0, 1.0)
    for _ in range(28):
        mid = 0.5*(lo+hi)
        if agd_of(mid) <= target_agd: hi = mid
        else: lo = mid
    alpha = hi
    rpm   = round(max(RPM_MIN, min(RPM_AB, rpm_from_alpha(alpha))), 1)
    def vblend(k): return ceil_kn(blend_metric(mil[k], ab[k], alpha, "speed"))
    asd = int(round(blend_metric(mil["ASD_ft"], ab["ASD_ft"], alpha, "distance") * 1.15))
    agd = int(round(blend_metric(mil["AGD_ft"], ab["AGD_ft"], alpha, "distance") * 1.15))
    return {"Vs": vblend("Vs_kt"), "V1": vblend("V1_kt"), "Vr": vblend("Vr_kt"), "V2": vblend("V2_kt"),
            "stop_ft": asd, "go_ft": agd, "bfl_ft": max(asd, agd), "rpm": rpm, "alpha": alpha}

# ---------------- DCS airports ----------------
@st.cache_data(show_spinner=False)
def load_dcs_airports():
    df = pd.read_csv(
        "dcs_airports.csv",
        dtype={"map":"string","airport_name":"string","runway_pair":"string","runway_end":"string","notes":"string"},
        keep_default_na=True, na_values=["","NA","N/A","null","None"]
    )
    for col in ["heading_deg","length_ft","tora_ft","toda_ft","asda_ft","threshold_elev_ft","opp_threshold_elev_ft","slope_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------------- Trim model & stab units ----------------
def compute_takeoff_trim_deg(gross_lbs: int, flap_deg: int, bias_nu: float = 0.0) -> float:
    base = 7.5  # at ~50k, flaps 20
    wt_term = 0.9 * max(0.0, (gross_lbs - 50000) / 5000.0)
    if flap_deg >= 35: flap_term = 0.0
    elif flap_deg >= 10: flap_term = 1.0
    else: flap_term = 2.0
    trim_nu = base + wt_term + flap_term + float(bias_nu)
    return max(5.0, min(16.0, round(trim_nu, 1)))

def deg_to_stab_units(trim_deg: float, deg_per_unit: float) -> float:
    if deg_per_unit <= 0: deg_per_unit = 1.0
    return round(trim_deg / deg_per_unit, 1)

# ---------------- Streamlit App ----------------
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

ensure_starter_perf()
perf_df, perf_source = load_perf_tables()
dcs = load_dcs_airports()

# 1) Theatre / Airport / Runway
st.header("1) DCS Theatre / Airport / Runway")
theatres = sorted(dcs["map"].dropna().unique().tolist())
theatre = st.selectbox("Theatre", theatres)
df_map = dcs[dcs["map"]==theatre].copy()

airports = sorted(df_map["airport_name"].dropna().unique().tolist())
airport = st.selectbox("Airport", airports)
df_ap = df_map[df_map["airport_name"]==airport].copy()

pairs = sorted(df_ap["runway_pair"].dropna().unique().tolist())
runway_pair = st.selectbox("Runway pair", pairs)

df_pair = df_ap[df_ap["runway_pair"]==runway_pair].copy()
ends_display = [f"{row['runway_end']}  (hdg {int(row['heading_deg'])}°)" for _,row in df_pair.sort_values("runway_end").iterrows()]
end_label = st.radio("Runway end (departure)", ends_display, horizontal=True)
sel_end = end_label.split()[0]
row = df_pair[df_pair["runway_end"]==sel_end].iloc[0]

dep_hdg   = int(row["heading_deg"]) if not np.isnan(row["heading_deg"]) else 0
length_ft = int(row["length_ft"]) if not np.isnan(row["length_ft"]) else 0
tora_ft   = int(row["tora_ft"]) if not np.isnan(row["tora_ft"]) else length_ft
toda_ft   = int(row["toda_ft"]) if not np.isnan(row["toda_ft"]) else tora_ft
asda_ft   = int(row["asda_ft"]) if not np.isnan(row["asda_ft"]) else tora_ft

thre = row.get("threshold_elev_ft", np.nan); oppe = row.get("opp_threshold_elev_ft", np.nan)
if length_ft>0 and not np.isnan(thre) and not np.isnan(oppe):
    slope_dir = (oppe - thre)/length_ft*100.0
    field_elev_auto = int(thre)
else:
    slope_dir = float(row["slope_percent"]) if not np.isnan(row["slope_percent"]) else 0.0
    field_elev_auto = int(thre if not np.isnan(thre) else (oppe if not np.isnan(oppe) else 0))

# Allow corrections + new manual reduction
if length_ft == 0:
    length_ft = st.number_input("Runway length (ft)", min_value=1000, max_value=20000, value=8000, step=50)
tora_ft = st.number_input("TORA (ft) — available runway", min_value=1000, max_value=20000, value=int(tora_ft), step=50)
manual_reduce_ft = st.number_input("Reduce available runway (ft) — e.g., back-taxi not available", min_value=0, max_value=int(tora_ft), value=0, step=50)
field_elev = st.number_input("Field Elevation (ft) at departure threshold", min_value=0, max_value=12000, value=int(field_elev_auto), step=1)

st.caption(
    f"Runway {runway_pair} end **{sel_end}**  |  Heading **{dep_hdg}°**  |  Length **{length_ft:,} ft**  "
    f"|  TORA **{int(tora_ft):,} ft**  |  TODA **{int(toda_ft):,} ft**  |  ASDA **{int(asda_ft):,} ft**  "
    f"|  Slope **{slope_dir:.1f}%**  |  Elev **{int(field_elev)} ft**"
)

# 2) Weather
st.header("2) Weather")
oat = st.number_input("OAT (°C)", min_value=-40, max_value=55, value=15, step=1)
wind_mode = st.radio("Headwind input", ["Manual (head/tailwind in kt)","From DCS Mission Briefing"], index=0)
if wind_mode == "Manual (head/tailwind in kt)":
    headwind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", min_value=-40.0, max_value=40.0, value=0.0, step=1.0)
    crosswind_kt = 0.0
else:
    c1,c2 = st.columns(2)
    with c1: wind_dir = st.number_input("Wind Direction (° FROM, briefing)", min_value=0, max_value=359, value=0, step=1)
    with c2: wind_spd = st.number_input("Wind Speed (kt)", min_value=0, max_value=100, value=0, step=1)
    rel = (wind_dir - dep_hdg) % 360
    th  = math.radians(rel)
    headwind_kt = round(wind_spd*math.cos(th),1)
    crosswind_kt = round(abs(wind_spd*math.sin(th)),1)
    if headwind_kt >= 0:
        st.info(f"Headwind: **+{headwind_kt} kt**  |  Crosswind: **{crosswind_kt} kt**")
    else:
        st.warning(f"Tailwind: **{headwind_kt} kt**  |  Crosswind: **{crosswind_kt} kt**")

max_cx = st.number_input("Max crosswind advisory (kt)", min_value=0, max_value=50, value=20, step=1)
if wind_mode!="Manual (head/tailwind in kt)" and crosswind_kt>max_cx:
    st.error(f"⚠ MAX CROSSWIND EXCEEDED: {crosswind_kt} kt (limit {max_cx} kt)")
if headwind_kt<0 and abs(headwind_kt)>TAILWIND_LIMIT_KT:
    st.error(f"⚠ MAX TAILWIND EXCEEDED: {abs(headwind_kt)} kt (limit {TAILWIND_LIMIT_KT} kt)")

# 2b) Part 121
st.subheader("Part 121—Takeoff Performance (Regulatory Checks)")
apply_part121 = st.checkbox("Apply 14 CFR Part 121 takeoff limits", value=True)
col121a, col121b, col121c = st.columns(3)
with col121a: runway_cond = st.selectbox("Runway condition", ["Dry","Wet"], index=0)
with col121b: stopway_ft = st.number_input("Stopway credited (ft)", min_value=0, max_value=5000, value=0, step=50)
with col121c: clearway_ft = st.number_input("Clearway credited (ft)", min_value=0, max_value=5000, value=0, step=50)
clearway_cap_ft = min(int(clearway_ft), int(tora_ft)//2)
st.caption("121 checks include your extra safety margin below.")

# 3) Weight & Balance
st.header("3) Weight & Balance")
use_gw = st.checkbox("Enter Gross Weight directly?", value=False)
if use_gw:
    gross = st.number_input("Gross Weight (lbs)", min_value=MIN_GW, max_value=MAX_GW, value=min(72000, MAX_GW), step=50)
else:
    fuel = st.number_input("Fuel Load (lbs)", min_value=0, max_value=MAX_GW, value=5000, step=50)
    ordn = st.number_input("Ordnance Load (lbs)", min_value=0, max_value=MAX_GW, value=2000, step=50)
    gross = EMPTY_WEIGHT_LBS + fuel + ordn
    if gross > MAX_GW:
        st.warning(f"Gross exceeds {MAX_GW:,} lb — limiting to max."); gross = MAX_GW
st.caption(f"Computed Gross Weight: **{int(gross):,} lbs** (max {MAX_GW:,})")

# 4) Configuration (auto-flaps default = Maneuvering)
st.header("4) Configuration")
auto_flaps = st.checkbox("Auto-select flap setting? (prefers Maneuvering)", value=True)
if auto_flaps:
    if int(tora_ft) < 5000 or gross > 70000: flap_name = "Flaps Full"
    elif int(tora_ft) > 9000 and gross < 60000: flap_name = "Flaps Up"
    else: flap_name = "Maneuvering Flaps"
else:
    flap_name = st.selectbox("Flap Setting", list(FLAP_OPTIONS.keys()), index=1)
flap_deg = FLAP_OPTIONS[flap_name]

# Trim bias + stab units
trim_bias = st.slider("Trim bias (NU)", min_value=-2.0, max_value=+2.0, value=+0.5, step=0.1,
                      help="Nudge if the jet feels nose-heavy (increase) or light (decrease).")
deg_per_stab_unit = st.number_input("Degrees per stab unit (for display)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Thrust mode
thrust_mode = st.radio("Thrust Mode", ["Auto-select most efficient thrust", "Set thrust manually"], index=0)
custom_rpm = None
if thrust_mode == "Set thrust manually":
    choice = st.selectbox("Manual Thrust", ["Military", "Afterburner", "Custom RPM (%)"], index=0)
    if choice == "Custom RPM (%)":
        custom_rpm = st.number_input("Custom Target RPM (%)", min_value=RPM_MIN, max_value=RPM_AB, value=RPM_MIL, step=0.1)
    thrust_choice = choice
else:
    thrust_choice = "Auto"

# 5) Intersection feasibility (optional) + Extra margin applies to solver
st.header("5) Intersection / Margins")
run_intersection = st.checkbox("Run intersection feasibility check", value=True)
extra_margin = st.number_input("Extra safety margin beyond BFL (ft)", min_value=0, max_value=4000, value=0, step=50)

# Effective available runway for ALL solvers (auto thrust will aim to pass with this + margin)
effective_tora = max(0, int(tora_ft) - int(manual_reduce_ft))
pa_ft = int(field_elev)

# ---------- Perf helpers ----------
def with_15pct(asd_ft, agd_ft):
    return int(round(asd_ft*1.15)), int(round(agd_ft*1.15))

def compute_static(thrust_mode):
    if perf_df is not None and thrust_mode in ("Military","Afterburner"):
        row = nearest_row(perf_df, flap_deg, thrust_mode, int(gross), pa_ft, int(oat))
        if row:
            asd, agd = with_15pct(row["ASD_ft"], row["AGD_ft"])
            rpm = round(RPM_AB if thrust_mode=="Afterburner" else RPM_MIL, 1)
            return {
                "Vs": ceil_kn(row["Vs_kt"]), "V1": ceil_kn(row["V1_kt"]),
                "Vr": ceil_kn(row["Vr_kt"]), "V2": ceil_kn(row["V2_kt"]),
                "stop_ft": asd, "go_ft": agd, "bfl_ft": max(asd,agd),
                "rpm": rpm, "alpha": alpha_from_rpm(rpm),
                "source": f"NATOPS ({perf_source})"
            }
    # fallback if no perf table
    Vs = 130 if flap_deg==20 else (120 if flap_deg==40 else 140)
    V1 = int(Vs*1.03); Vr = int(Vs*1.12); V2 = int(Vs*1.20)
    asd = 6000; agd = 6000
    return {"Vs":Vs,"V1":V1,"Vr":Vr,"V2":V2,"stop_ft":asd,"go_ft":agd,"bfl_ft":max(asd,agd),
            "rpm":RPM_MIL,"alpha":alpha_from_rpm(RPM_MIL),"source":"FALLBACK"}

# ---------- Selection logic ----------
selected_flap_name = flap_name
selected_flap_deg = flap_deg

if apply_part121 and perf_df is not None:
    if thrust_mode == "Auto-select most efficient thrust":
        # Try each flap to find minimal RPM that passes Part 121 INCLUDING extra margin and effective TORA
        flap_candidates = [("Flaps Up", 0), ("Maneuvering Flaps", 20), ("Flaps Full", 40)] if auto_flaps else [(flap_name, flap_deg)]
        best = None
        for fname, fdeg in flap_candidates:
            cand = evaluate_combo_for_121(perf_df, fdeg, int(gross), pa_ft, int(oat),
                                          int(effective_tora), runway_cond,
                                          int(stopway_ft), int(clearway_cap_ft),
                                          extra_margin_ft=int(extra_margin))
            if cand:
                key = (cand["rpm"], cand["bfl_ft"])  # minimize RPM then BFL
                if (best is None) or (key < best[0]): best = (key, cand, fname, fdeg)
        if best:
            _, base, selected_flap_name, selected_flap_deg = best
            base["source"] = f"NATOPS ({perf_source})"
            st.success(f"✅ Part 121 PASS — **{selected_flap_name}**, Target RPM ≈ **{base['rpm']}%** (incl. margin)")
        else:
            st.error("⛔ Part 121 not satisfied for any flap (even AB) with current runway/margin. Increase thrust or reduce WT/OAT.")
            base = compute_static("Afterburner")
    else:
        # Manual thrust under Part 121
        if thrust_choice == "Custom RPM (%)" and custom_rpm is not None and perf_df is not None:
            alpha = alpha_from_rpm(custom_rpm)
            mil = nearest_row(perf_df, selected_flap_deg, "Military", int(gross), pa_ft, int(oat))
            ab  = nearest_row(perf_df, selected_flap_deg, "Afterburner", int(gross), pa_ft, int(oat))
            if mil and ab:
                asd = blend_metric(mil["ASD_ft"], ab["ASD_ft"], alpha, "distance") * 1.15
                agd = blend_metric(mil["AGD_ft"], ab["AGD_ft"], alpha, "distance") * 1.15
                vs  = ceil_kn(blend_metric(mil["Vs_kt"], ab["Vs_kt"], alpha, "speed"))
                v1  = ceil_kn(blend_metric(mil["V1_kt"], ab["V1_kt"], alpha, "speed"))
                vr  = ceil_kn(blend_metric(mil["Vr_kt"], ab["Vr_kt"], alpha, "speed"))
                v2  = ceil_kn(blend_metric(mil["V2_kt"], ab["V2_kt"], alpha, "speed"))
                base = dict(Vs=vs,V1=v1,Vr=vr,V2=v2,
                            stop_ft=int(round(asd)), go_ft=int(round(agd)),
                            bfl_ft=int(round(max(asd,agd))),
                            rpm=round(max(RPM_MIN,min(RPM_AB,custom_rpm)),1), alpha=alpha,
                            source=f"NATOPS ({perf_source})")
                ok = part121_pass(int(effective_tora), base["stop_ft"], base["go_ft"], runway_cond,
                                  int(stopway_ft), int(clearway_cap_ft), extra_margin_ft=int(extra_margin))
                if ok: st.success("✅ Part 121 PASS at your custom RPM (incl. margin).")
                else:  st.error("⛔ Part 121 FAIL at your custom RPM with margin/runway.")
            else:
                st.warning("Perf tables missing for custom interpolation; falling back to MIL.")
                base = compute_static("Military")
        else:
            fixed = "Military" if thrust_choice=="Military" else "Afterburner"
            base = compute_static(fixed)
            ok = part121_pass(int(effective_tora), base["stop_ft"], base["go_ft"], runway_cond,
                              int(stopway_ft), int(clearway_cap_ft), extra_margin_ft=int(extra_margin))
            if ok: st.success(f"✅ Part 121 PASS at fixed thrust: {fixed} (incl. margin).")
            else:  st.error(f"⛔ Part 121 FAIL at fixed thrust: {fixed} with margin/runway.")
else:
    # No Part 121: Auto uses 60% rule vs effective_tora and your extra margin (harder target)
    if thrust_mode == "Auto-select most efficient thrust" and perf_df is not None:
        dyn = solve_min_required_dynamic(perf_df, selected_flap_deg, int(gross), pa_ft, int(oat),
                                         int(effective_tora), extra_margin_ft=int(extra_margin))
        if dyn is None:
            st.error("⛔ Not feasible to reach Vr by 60% runway (incl. margin) even at AB (+15%).")
            base = compute_static("Afterburner")
        else:
            base = dyn; base["source"] = f"NATOPS ({perf_source})"
            st.success(f"✅ Minimum Required thrust solved (incl. margin): ~**{base['rpm']}%** (alpha={base['alpha']:+.3f}).")
    else:
        if thrust_mode == "Set thrust manually" and thrust_choice == "Custom RPM (%)" and custom_rpm is not None and perf_df is not None:
            alpha = alpha_from_rpm(custom_rpm)
            mil = nearest_row(perf_df, selected_flap_deg, "Military", int(gross), pa_ft, int(oat))
            ab  = nearest_row(perf_df, selected_flap_deg, "Afterburner", int(gross), pa_ft, int(oat))
            if mil and ab:
                asd = blend_metric(mil["ASD_ft"], ab["ASD_ft"], alpha, "distance") * 1.15
                agd = blend_metric(mil["AGD_ft"], ab["AGD_ft"], alpha, "distance") * 1.15
                vs  = ceil_kn(blend_metric(mil["Vs_kt"], ab["Vs_kt"], alpha, "speed"))
                v1  = ceil_kn(blend_metric(mil["V1_kt"], ab["V1_kt"], alpha, "speed"))
                vr  = ceil_kn(blend_metric(mil["Vr_kt"], ab["Vr_kt"], alpha, "speed"))
                v2  = ceil_kn(blend_metric(mil["V2_kt"], ab["V2_kt"], alpha, "speed"))
                base = dict(Vs=vs,V1=v1,Vr=vr,V2=v2,
                            stop_ft=int(round(asd)), go_ft=int(round(agd)),
                            bfl_ft=int(round(max(asd,agd))),
                            rpm=round(max(RPM_MIN,min(RPM_AB,custom_rpm)),1), alpha=alpha,
                            source=f"NATOPS ({perf_source})")
            else:
                st.warning("Perf tables missing for custom interpolation; falling back to MIL.")
                base = compute_static("Military")
        else:
            base = compute_static("Military" if (thrust_mode=="Set thrust manually" and thrust_choice=="Military") else "Afterburner")

# ---- Optional intersection feasibility readout (purely informational) ----
required_len = base["bfl_ft"] + int(extra_margin)
available_len = int(effective_tora)
if st.checkbox("Show intersection feasibility result", value=True) and run_intersection:
    if required_len <= available_len:
        max_offset = available_len - required_len
        st.success(f"Intersection feasible. Max allowable offset: **{int(max_offset):,} ft**  (Effective {int(required_len):,} ft).")
    else:
        st.error(f"Intersection NOT feasible: need {int(required_len):,} ft; available {int(available_len):,} ft.")

# ---------------- Final Numbers ----------------
st.subheader("Final Numbers")

# Trim (deg & stab units)
trim_deg = compute_takeoff_trim_deg(int(gross), int(selected_flap_deg), bias_nu=trim_bias)
trim_units = deg_to_stab_units(trim_deg, deg_per_unit=deg_per_stab_unit)

if "alpha" in base:
    st.caption(f"Debug: alpha={base['alpha']:+.3f}, rpm={base.get('rpm','--')}%")

rpm_val = base.get("rpm", None)
thrust_label = classify_thrust(rpm_val)
rpm_str = f"{rpm_val:.1f}%" if rpm_val is not None else "--"

st.write(f"**Selected Flap:** {selected_flap_name} ({selected_flap_deg}°)")
st.write(
    f"**Thrust:** {thrust_label} ({rpm_str})  "
    f"|  **Runway Required (BFL + margin):** {required_len:,} ft  |  **Available (TORA eff.):** {available_len:,} ft"
)
st.write(f"**Vs:** {base['Vs']} kt   **V1:** {base['V1']} kt   **Vr:** {base['Vr']} kt   **V2:** {base['V2']} kt")
st.write(f"**Balanced Field Length:** {base['bfl_ft']:,} ft")
st.write(f"**Accel-Go:** {base['go_ft']:,} ft   **Accel-Stop:** {base['stop_ft']:,} ft")
st.write(f"**Trim:** {trim_deg:.1f}° NU  (**{trim_units:.1f} stab units**, {deg_per_stab_unit:.2f}°/unit)")
st.caption(
    f"Source: **{base.get('source','')}**  |  121 Mode: {'ON' if apply_part121 else 'OFF'}  "
    f"|  RWY: {runway_cond}, Stopway {int(stopway_ft):,} ft, Clearway {int(clearway_cap_ft):,} ft"
)

# ---- Comparison table: minimum passing RPM per flap (Part 121) ----
if apply_part121 and perf_df is not None:
    rows_cmp = []
    for fname, fdeg in [("Flaps Up",0),("Maneuvering Flaps",20),("Flaps Full",40)]:
        res = evaluate_combo_for_121(perf_df, fdeg, int(gross), int(field_elev), int(oat),
                                     int(effective_tora), runway_cond,
                                     int(stopway_ft), int(clearway_cap_ft), extra_margin_ft=int(extra_margin))
        if res:
            label = classify_thrust(res["rpm"])
            rows_cmp.append({
                "Flap": f"{fname} ({fdeg}°)",
                "Min RPM (121 + margin)": f'{res["rpm"]:.1f}',
                "Thrust Label": label,
                "Vs": res["Vs"], "V1": res["V1"], "Vr": res["Vr"], "V2": res["V2"],
                "BFL (ft)": res["bfl_ft"], "AG (ft)": res["go_ft"], "AS (ft)": res["stop_ft"]
            })
        else:
            rows_cmp.append({
                "Flap": f"{fname} ({fdeg}°)",
                "Min RPM (121 + margin)": "—",
                "Thrust Label": "No pass",
                "Vs": "—", "V1": "—", "Vr": "—", "V2": "—",
                "BFL (ft)": "—", "AG (ft)": "—", "AS (ft)": "—"
            })
    st.subheader("Part 121: Minimum Passing RPM by Flap (with margin)")
    st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True)
