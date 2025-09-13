# f14_takeoff_app_dcs.py
import os, math
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# App meta
# =========================
APP_TITLE = "F-14 Takeoff Performance (DCS • Part 121 • Dynamic Min Thrust)"

# =========================
# Core aircraft constants / tuning
# =========================
EMPTY_WEIGHT_LBS = 40000
MAX_GW = 74349
MIN_GW = 30000

# Engine %RPM anchors for thrust blending
RPM_MIN = 88.0     # practical lower bound for below-MIL search (reduced thrust)
RPM_MIL = 96.0     # nominal Military
RPM_AB  = 102.0    # nominal full afterburner

# Config options
FLAP_OPTIONS = {"Flaps Up": 0, "Maneuvering Flaps": 20, "Flaps Full": 40}
TAILWIND_LIMIT_KT = 10  # “Max tailwind” advisory

def ceil_kn(x): return int(math.ceil(float(x)))

# =========================
# Performance-table loading (NATOPS digitized CSVs)
# =========================
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
    """
    Load perf_f14b.csv or perf_f14d.csv (whichever exists). Expects columns:
    model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft
    """
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
                "Minimum Required":"Minimum Required",
                "minimum":"Minimum Required","min":"Minimum Required",
                "Military":"Military","military":"Military","mil":"Military",
                "Afterburner":"Afterburner","afterburner":"Afterburner","ab":"Afterburner"
            })
            return df, fname
        except Exception:
            continue
    return None, None

def nearest_row(df, flap_deg, thrust, gw_lbs, pa_ft, oat_c):
    """Pick the single closest table row to requested state for a given flap/thrust."""
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

def blend_linear(mil_val, ab_val, alpha):
    """
    alpha: 0 → MIL, 1 → AB, <0 → below-MIL with steeper penalty on distances.
    """
    if alpha >= 0:
        return mil_val + alpha*(ab_val - mil_val)
    # below-MIL: distances worsen faster; speeds slightly lower than MIL
    slope = (ab_val - mil_val)
    penalty = 1.8
    return mil_val + alpha * penalty * slope

def rpm_from_alpha(alpha):
    """Map alpha to nominal %RPM across MIN–MIL–AB anchors."""
    if alpha >= 0:
        return RPM_MIL + alpha*(RPM_AB - RPM_MIL)
    return RPM_MIL + alpha*(RPM_MIL - RPM_MIN)

def solve_min_required_dynamic(perf_df, flap_deg, gw_lbs, pa_ft, oat_c, runway_available_ft):
    """
    Solve for Minimum Required thrust (may be below MIL) such that:
    AGD(α)*1.15 <= 0.60 * runway_available_ft
    """
    mil = nearest_row(perf_df, flap_deg, "Military", int(gw_lbs), int(pa_ft), int(oat_c))
    ab  = nearest_row(perf_df, flap_deg, "Afterburner", int(gw_lbs), int(pa_ft), int(oat_c))
    if not mil or not ab:
        return None
    target_agd = 0.60 * float(runway_available_ft)
    def agd_of(alpha): return blend_linear(mil["AGD_ft"], ab["AGD_ft"], alpha) * 1.15
    # If even AB can't make 60% rule → infeasible
    if agd_of(1.0) > target_agd:
        return None
    # Bracket search region
    alpha_min = (RPM_MIN - RPM_MIL) / (RPM_AB - RPM_MIL)  # negative
    if agd_of(0.0) <= target_agd:
        lo, hi = alpha_min, 0.0  # search below-MIL
    else:
        lo, hi = 0.0, 1.0        # search MIL→AB
    # Binary search for minimal alpha meeting target
    for _ in range(28):
        mid = 0.5*(lo+hi)
        if agd_of(mid) <= target_agd: hi = mid
        else:                          lo = mid
    alpha = hi
    rpm   = int(round(rpm_from_alpha(alpha)))
    def vblend(k): return ceil_kn(blend_linear(mil[k], ab[k], alpha))
    asd = int(round(blend_linear(mil["ASD_ft"], ab["ASD_ft"], alpha) * 1.15))
    agd = int(round(blend_linear(mil["AGD_ft"], ab["AGD_ft"], alpha) * 1.15))
    return {"Vs": vblend("Vs_kt"), "V1": vblend("V1_kt"), "Vr": vblend("Vr_kt"), "V2": vblend("V2_kt"),
            "stop_ft": asd, "go_ft": agd, "bfl_ft": max(asd, agd), "rpm": rpm, "alpha": alpha}

# =========================
# Part 121 gating
# =========================
def part121_pass(runway_ft, asd_ft, agd_ft, condition, stopway_ft=0, clearway_ft=0):
    """
    121.189 / 25.113 checks:
      • ASD ≤ runway + stopway  (ASDA)
      • OEI takeoff distance (proxy: AGD to 35 ft) ≤ runway + min(clearway, ½ runway) (TODA)
      • AEO takeoff run ≤ runway (TORA) — proxy 0.85×AGD for Dry, 0.90×AGD for Wet
    """
    tora = float(runway_ft)
    asda = tora + float(stopway_ft)
    toda = tora + float(min(clearway_ft, tora/2.0))
    asd_ok = float(asd_ft) <= asda
    oei_tod_ok = float(agd_ft) <= toda
    aeo_tr_proxy = (0.85 if condition=="Dry" else 0.90) * float(agd_ft)
    tor_ok = aeo_tr_proxy <= tora
    return asd_ok and oei_tod_ok and tor_ok

def evaluate_combo_for_121(perf_df, flap_deg, gw, pa_ft, oat_c, runway_ft, condition, stopway_ft, clearway_ft):
    """
    For a given flap, find minimal thrust alpha (may be below MIL) satisfying Part 121 gates
    using NATOPS MIL/AB rows blended with +15% safety.
    """
    dyn = solve_min_required_dynamic(perf_df, flap_deg, gw, pa_ft, oat_c, runway_ft)
    trial_alphas = []
    if dyn: trial_alphas.append(dyn["alpha"])
    alpha_min = (RPM_MIN - RPM_MIL) / (RPM_AB - RPM_MIL)
    trial_alphas += [alpha_min, 0.0, 0.5, 1.0]
    mil = nearest_row(perf_df, flap_deg, "Military", gw, pa_ft, oat_c)
    ab  = nearest_row(perf_df, flap_deg, "Afterburner", gw, pa_ft, oat_c)
    if not mil or not ab: return None

    def results_at(alpha):
        asd = blend_linear(mil["ASD_ft"], ab["ASD_ft"], alpha) * 1.15
        agd = blend_linear(mil["AGD_ft"], ab["AGD_ft"], alpha) * 1.15
        vs  = ceil_kn(blend_linear(mil["Vs_kt"], ab["Vs_kt"], alpha))
        v1  = ceil_kn(blend_linear(mil["V1_kt"], ab["V1_kt"], alpha))
        vr  = ceil_kn(blend_linear(mil["Vr_kt"], ab["Vr_kt"], alpha))
        v2  = ceil_kn(blend_linear(mil["V2_kt"], ab["V2_kt"], alpha))
        return dict(Vs=vs, V1=v1, Vr=vr, V2=v2,
                    stop_ft=int(round(asd)), go_ft=int(round(agd)),
                    bfl_ft=int(round(max(asd, agd))),
                    rpm=int(round(rpm_from_alpha(alpha))), alpha=alpha)

    feasible = []
    for a in trial_alphas:
        r = results_at(a)
        if part121_pass(runway_ft, r["stop_ft"], r["go_ft"], condition, stopway_ft, clearway_ft):
            feasible.append((a, r))
    if not feasible: return None

    lo = min(a for a,_ in feasible) - 0.05
    hi = min(a for a,_ in feasible)
    best = results_at(hi)
    for _ in range(30):
        mid = 0.5*(lo+hi)
        r = results_at(mid)
        if part121_pass(runway_ft, r["stop_ft"], r["go_ft"], condition, stopway_ft, clearway_ft):
            hi = mid; best = r
        else:
            lo = mid
    return best

# =========================
# DCS airport data
# =========================
@st.cache_data(show_spinner=False)
def load_dcs_airports():
    """
    Expects dcs_airports.csv with per-runway-end rows:
    map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,
    threshold_elev_ft,opp_threshold_elev_ft,slope_percent,notes
    """
    df = pd.read_csv(
        "dcs_airports.csv",
        dtype={"map":"string","airport_name":"string","runway_pair":"string","runway_end":"string","notes":"string"},
        keep_default_na=True, na_values=["","NA","N/A","null","None"]
    )
    for col in ["heading_deg","length_ft","tora_ft","toda_ft","asda_ft","threshold_elev_ft","opp_threshold_elev_ft","slope_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

ensure_starter_perf()
perf_df, perf_source = load_perf_tables()
dcs = load_dcs_airports()

# 1) Theatre / Airport / Runway (END-SPECIFIC)
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

# Allow corrections
if length_ft == 0:
    length_ft = st.number_input("Runway length (ft)", min_value=1000, max_value=20000, value=8000, step=50)
tora_ft = st.number_input("TORA (ft) — available runway", min_value=1000, max_value=20000, value=int(tora_ft), step=50)
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
    # compute components
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

# 2b) Part 121 options
st.subheader("Part 121—Takeoff Performance (Regulatory Checks)")
apply_part121 = st.checkbox("Apply 14 CFR Part 121 takeoff limits", value=True)
col121a, col121b, col121c = st.columns(3)
with col121a:
    runway_cond = st.selectbox("Runway condition", ["Dry","Wet"], index=0)
with col121b:
    stopway_ft = st.number_input("Stopway credited (ft)", min_value=0, max_value=5000, value=0, step=50)
with col121c:
    clearway_ft = st.number_input("Clearway credited (ft)", min_value=0, max_value=5000, value=0, step=50)
clearway_cap_ft = min(int(clearway_ft), int(tora_ft)//2)
st.caption("121 checks: ASD ≤ RWY+Stopway; OEI distance ≤ RWY+min(Clearway, ½RWY); AEO TOR ≤ RWY (proxy).")

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
        st.warning(f"Gross exceeds {MAX_GW:,} lb — limiting to max.")
        gross = MAX_GW
st.caption(f"Computed Gross Weight: **{int(gross):,} lbs** (max {MAX_GW:,})")

# 4) Configuration
st.header("4) Configuration")
auto_flaps = st.checkbox("Auto-select flap setting?", value=True)
if auto_flaps:
    # simple heuristic: shorter runways or heavier weights → more flap
    if int(tora_ft) < 5000 or gross > 70000: flap_name = "Flaps Full"
    elif int(tora_ft) < 8000 or gross > 60000: flap_name = "Maneuvering Flaps"
    else: flap_name = "Flaps Up"
else:
    flap_name = st.selectbox("Flap Setting", list(FLAP_OPTIONS.keys()), index=1)
flap_deg = FLAP_OPTIONS[flap_name]

thrust_choice = st.selectbox(
    "Thrust Rating (manual mode only)",
    ["Minimum Required (dynamic)", "Military", "Afterburner"],
    index=0
)

# 5) Intersection feasibility – we use TORA as available
st.header("5) Intersection Feasibility (auto)")
margin = st.number_input("Extra safety margin beyond BFL (ft)", min_value=0, max_value=2000, value=0, step=50)
runway_available_ft = int(tora_ft)
pa_ft = int(field_elev)  # pressure altitude proxy

# ======== PERFORMANCE / SELECTION ========
def with_15pct(asd_ft, agd_ft):
    return int(round(asd_ft*1.15)), int(round(agd_ft*1.15))

def compute_static(thrust_mode):
    """Use nearest-row NATOPS values (+15%) for a given thrust setting; fallback if no tables."""
    if perf_df is not None and thrust_mode in ("Military","Afterburner"):
        row = nearest_row(perf_df, flap_deg, thrust_mode, int(gross), pa_ft, int(oat))
        if row:
            asd, agd = with_15pct(row["ASD_ft"], row["AGD_ft"])
            return {
                "Vs": ceil_kn(row["Vs_kt"]), "V1": ceil_kn(row["V1_kt"]),
                "Vr": ceil_kn(row["Vr_kt"]), "V2": ceil_kn(row["V2_kt"]),
                "stop_ft": asd, "go_ft": agd, "bfl_ft": max(asd,agd),
                "trim": max(5.0, min(12.0, round(8.0 + (int(gross)-50000)/10000.0,1))),
                "rpm": int(RPM_AB if thrust_mode=="Afterburner" else RPM_MIL),
                "source": f"NATOPS ({perf_source})"
            }
    # Minimal fallback if no tables; strongly recommend providing perf_f14b.csv
    Vs = 130 if flap_deg==20 else (120 if flap_deg==40 else 140)
    V1 = int(Vs*1.03); Vr = int(Vs*1.12); V2 = int(Vs*1.20)
    asd = 6000; agd = 6000
    return {"Vs":Vs,"V1":V1,"Vr":Vr,"V2":V2,"stop_ft":asd,"go_ft":agd,"bfl_ft":max(asd,agd),
            "trim":8.0,"rpm":(RPM_AB if thrust_mode=="Afterburner" else RPM_MIL),"source":"FALLBACK"}

selected = None
selected_flap_name = None
selected_flap_deg = None

if apply_part121 and perf_df is not None:
    # Try each flap to find the minimal RPM that passes Part 121
    flap_candidates = [("Flaps Up", 0), ("Maneuvering Flaps", 20), ("Flaps Full", 40)] if auto_flaps else [(flap_name, flap_deg)]
    best = None
    for fname, fdeg in flap_candidates:
        cand = evaluate_combo_for_121(perf_df, fdeg, int(gross), pa_ft, int(oat),
                                      int(runway_available_ft), runway_cond,
                                      int(stopway_ft), int(clearway_cap_ft))
        if cand:
            key = (cand["rpm"], cand["bfl_ft"])  # minimize RPM, then BFL
            if (best is None) or (key < best[0]):
                best = (key, cand, fname, fdeg)
    if best:
        _, selected, selected_flap_name, selected_flap_deg = best
        base = selected.copy()
        base["trim"] = max(5.0, min(12.0, round(8.0 + (int(gross)-50000)/10000.0,1)))
        base["source"] = f"NATOPS ({perf_source})"
        st.success(f"✅ Part 121 PASS — Selected **{selected_flap_name}**, Target RPM ≈ **{base['rpm']}%**")
    else:
        st.error("⛔ Part 121 not satisfied for any flap with available thrust (even AB). Adjust weight/temp/runway/wind.")
        base = compute_static("Afterburner")
        selected_flap_name, selected_flap_deg = flap_name, flap_deg
else:
    # Manual / no-121 mode
    if thrust_choice == "Minimum Required (dynamic)":
        if perf_df is not None:
            dyn = solve_min_required_dynamic(perf_df, flap_deg, int(gross), pa_ft, int(oat), int(runway_available_ft))
            if dyn is None:
                st.error("⛔ Not feasible to reach Vr by 60% runway even at AB (+15% safety).")
                base = compute_static("Afterburner")
            else:
                base = dyn
                base["trim"] = max(5.0, min(12.0, round(8.0 + (int(gross)-50000)/10000.0,1)))
                base["source"] = f"NATOPS ({perf_source})"
                st.success(f"✅ Minimum Required thrust solved: ~**{base['rpm']}% RPM** (alpha={base['alpha']:+.3f}).")
        else:
            st.warning("No performance CSVs found — using rough fallback values.")
            base = compute_static("Military")
    else:
        base = compute_static("Military" if thrust_choice=="Military" else "Afterburner")
    selected_flap_name, selected_flap_deg = flap_name, flap_deg

# Required vs Available & Intersection Feasibility
required_len = max(base["bfl_ft"], 0) + int(margin)
available_len = int(runway_available_ft)
if required_len <= available_len:
    max_offset = available_len - required_len
    st.success(f"Intersection feasible. Max allowable offset: **{int(max_offset):,} ft**  (Effective {int(required_len):,} ft).")
else:
    st.error(f"Intersection NOT feasible: need {int(required_len):,} ft; available {int(available_len):,} ft.")

# =========================
# Final Numbers (text only)
# =========================
st.subheader("Final Numbers")

thrust_label = (
    "Reduced" if (apply_part121 and perf_df is not None and selected)
    or thrust_choice.startswith("Minimum")
    else ("Military" if base.get("rpm", RPM_MIL) <= RPM_MIL+0.5 else "Afterburner")
)

st.write(f"**Selected Flap:** {selected_flap_name} ({selected_flap_deg}°)")
st.write(
    f"**Thrust:** {thrust_label}  |  **Target RPM:** {base.get('rpm','--')}%  "
    f"|  **Runway Required (BFL + margin):** {required_len:,} ft  |  **Available (TORA):** {available_len:,} ft"
)
st.write(f"**Vs:** {base['Vs']} kt   **V1:** {base['V1']} kt   **Vr:** {base['Vr']} kt   **V2:** {base['V2']} kt")
st.write(f"**Balanced Field Length:** {base['bfl_ft']:,} ft")
st.write(f"**Accel-Go:** {base['go_ft']:,} ft   **Accel-Stop:** {base['stop_ft']:,} ft")
st.write(f"**Trim:** {base.get('trim','--')}° NU")
st.caption(
    f"Source: **{base.get('source','')}**  |  121 Mode: {'ON' if apply_part121 else 'OFF'}  "
    f"|  RWY: {runway_cond}, Stopway {int(stopway_ft):,} ft, Clearway {int(clearway_cap_ft):,} ft"
)
