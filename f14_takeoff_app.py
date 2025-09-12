# f14_takeoff_app_dcs.py
import io, os, math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader

# =========================
# Constants & visuals
# =========================
APP_TITLE = "F-14B Takeoff Calculator (DCS VR • Part 121 • Dynamic Min Thrust)"
TAILWIND_LIMIT_KT = 10
THRUST_LEVELS_UI = ["Minimum Required (dynamic)", "Military", "Afterburner"]
FLAP_OPTIONS = {"Flaps Up": 0, "Maneuvering Flaps": 20, "Flaps Full": 40}
EMPTY_WEIGHT_LBS = 40000
MAX_GW = 74349
MIN_GW = 30000

# Engine/RPM anchors (tunable)
RPM_MIN = 88.0     # practical lower bound for below-MIL search
RPM_MIL = 96.0     # nominal Military
RPM_AB  = 102.0    # nominal full afterburner

# Toy physics (fallback only)
g = 9.80665
ft2m = 0.3048
lb2kg = 0.45359237
m2kt = 1.94384
m2ft = 3.28084
WING_AREA_M2 = 52.5
CLMAX = {0: 1.2, 20: 1.7, 40: 2.1}
brake_decel_g = 0.35
rotation_time = 2.0
THRUST_FACTOR = {"subMIL": 1.30, "Military": 1.00, "Afterburner": 0.80}  # toy scaling (subMIL → longer AGD)

def ceil_kn(x): return int(math.ceil(float(x)))

# =========================
# Utility helpers
# =========================
def air_density(alt_ft, oat_c):
    return 1.225 * math.exp(-(alt_ft*ft2m)/8000)

def v_stall(weight_kg, rho, S, clmax):
    return math.sqrt((2*weight_kg*g)/(rho*S*clmax))

def accel_stop_m(V1_mps):   return (V1_mps**2)/(2*brake_decel_g*g) + 150
def accel_go_m(Vr_mps,V2_mps): return Vr_mps*rotation_time + V2_mps*2.5

def carrier_adjust(Vr_mps, trim_deg, carrier):
    if carrier: Vr_mps -= 40/m2kt; trim_deg += 2
    return Vr_mps, trim_deg

def runway_heads_from_id(runway_id):
    try:
        a,b = [s.strip() for s in runway_id.split("/")]
        parse = lambda s:int("".join(ch for ch in s if ch.isdigit()) or "0")%36*10
        return (a,b, parse(a), parse(b))
    except: return ("XX","YY",0,180)

def wind_components(runway_heading_deg, wind_dir_deg, wind_speed_kt):
    rel = (wind_dir_deg - runway_heading_deg) % 360
    th = math.radians(rel)
    head = wind_speed_kt*math.cos(th)
    cross = wind_speed_kt*math.sin(th)
    return head, abs(cross), ("right" if cross>0 else "left")

def runway_safety_color(available_ft, bfl_ft):
    margin = available_ft - bfl_ft
    return "green" if margin>500 else ("yellow" if margin>0 else "red")

def auto_flap(weight_lbs, runway_ft):
    if runway_ft < 5000 or weight_lbs > 70000: return "Flaps Full"
    if runway_ft < 8000 or weight_lbs > 60000: return "Maneuvering Flaps"
    return "Flaps Up"

# =========================
# NATOPS data layer
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
        with open("perf_f14d.csv","w",encoding="utf-8") as f:
            f.write(STARTER_PERF_CSV)

@st.cache_data(show_spinner=False)
def load_perf_tables():
    for fname in ["perf_f14b.csv", "perf_f14d.csv"]:
        try:
            df = pd.read_csv(fname)
            df["flap_deg"] = pd.to_numeric(df["flap_deg"], errors="coerce")
            df["gw_lbs"] = pd.to_numeric(df["gw_lbs"], errors="coerce")
            df["press_alt_ft"] = pd.to_numeric(df["press_alt_ft"], errors="coerce")
            df["oat_c"] = pd.to_numeric(df["oat_c"], errors="coerce")
            for col in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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
    # alpha: 0 → MIL, 1 → AB, <0 → below-MIL with steeper penalty
    if alpha >= 0:
        return mil_val + alpha*(ab_val - mil_val)
    else:
        slope = (ab_val - mil_val)
        penalty = 1.8
        return mil_val + alpha * penalty * slope

def rpm_from_alpha(alpha):
    if alpha >= 0:
        return RPM_MIL + alpha*(RPM_AB - RPM_MIL)
    else:
        return RPM_MIL + alpha*(RPM_MIL - RPM_MIN)

def solve_min_required_dynamic(perf_df, flap_deg, gw_lbs, pa_ft, oat_c, runway_available_ft):
    """
    Minimum Required thrust solves AGD(α)*1.15 <= 0.60*runway.
    α < 0 → below MIL; α=0 → MIL; α=1 → AB.
    """
    mil = nearest_row(perf_df, flap_deg, "Military", int(gw_lbs), int(pa_ft), int(oat_c))
    ab  = nearest_row(perf_df, flap_deg, "Afterburner", int(gw_lbs), int(pa_ft), int(oat_c))
    if not mil or not ab:
        return None
    target_agd = 0.60 * float(runway_available_ft)
    def agd_of(alpha): return blend_linear(mil["AGD_ft"], ab["AGD_ft"], alpha) * 1.15
    if agd_of(1.0) > target_agd: return None
    alpha_min = (RPM_MIN - RPM_MIL) / (RPM_AB - RPM_MIL)
    if agd_of(0.0) <= target_agd: lo, hi = alpha_min, 0.0
    else:                           lo, hi = 0.0, 1.0
    for _ in range(28):
        mid = 0.5*(lo+hi)
        if agd_of(mid) <= target_agd: hi = mid
        else:                         lo = mid
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
      • ASD ≤ runway + stopway
      • OEI takeoff distance (proxy: AGD to 35 ft) ≤ runway + min(clearway, ½ runway)
      • AEO takeoff run ≤ runway (proxy from AGD: 0.85× for Dry, 0.90× for Wet)
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
# Toy perf (fallback when no CSVs)
# =========================
def toy_perf(gw_lbs, oat_c, pa_ft, flap_deg, thrust_mode, headwind_kt, slope_pct, carrier, rpm=None):
    rho = air_density(pa_ft, oat_c)
    Vs = v_stall(gw_lbs*lb2kg, rho, WING_AREA_M2, CLMAX.get(flap_deg,1.7))
    Vs = max(0.0, Vs - 0.5*(headwind_kt/m2kt))
    Vr = Vs*1.05; V2 = Vs*1.20; V1 = Vs*1.04
    trim = 8.0 + (gw_lbs-50000)/10000.0
    Vr, trim = carrier_adjust(Vr, trim, carrier); trim = max(5.0, min(12.0, trim))
    if thrust_mode == "Afterburner":
        factor = THRUST_FACTOR["Afterburner"]; rpm_out = RPM_AB
    elif thrust_mode == "Military":
        factor = THRUST_FACTOR["Military"]; rpm_out = RPM_MIL
    else:
        rpm_out = rpm if rpm is not None else RPM_MIL
        if rpm_out < RPM_MIL:
            t = (rpm_out - RPM_MIN) / (RPM_MIL - RPM_MIN + 1e-6)
            factor = THRUST_FACTOR["subMIL"]*(1-t) + THRUST_FACTOR["Military"]*t
        else:
            t = (rpm_out - RPM_MIL) / (RPM_AB - RPM_MIL + 1e-6)
            factor = THRUST_FACTOR["Military"]*(1-t) + THRUST_FACTOR["Afterburner"]*t
    stop_m = accel_stop_m(V1) * 1.15
    go_m   = accel_go_m(Vr,V2)*factor * 1.15
    slope_factor = 1 + float(slope_pct)/100.0
    stop_ft = stop_m*m2ft*slope_factor
    go_ft   = go_m*m2ft*slope_factor
    bfl_ft  = max(stop_ft, go_ft)
    return dict(Vs=ceil_kn(Vs*m2kt), V1=ceil_kn(V1*m2kt), Vr=ceil_kn(Vr*m2kt), V2=ceil_kn(V2*m2kt),
                stop_ft=int(round(stop_ft)), go_ft=int(round(go_ft)), bfl_ft=int(round(bfl_ft)),
                trim=round(trim,1), rpm=int(round(rpm_out)))

# =========================
# VR Kneeboard
# =========================
def draw_flap_indicator(draw, name, pos, size=100):
    x,y = pos
    draw.rectangle([x,y,x+size,y+size//4], outline="black", width=5)
    fill = {"Flaps Up":0.33,"Maneuvering Flaps":0.66,"Flaps Full":1.0}[name]*size
    draw.rectangle([x,y,x+fill,y+size//4], fill="blue")
    draw.text((x+size/2,y+size//4+5), name, fill="black", anchor="ms")

def kneeboard_png(results, weight, temp, alt, flap_name, thrust_label, carrier, template="Day"):
    scale=6; W,H=512*scale,768*scale
    theme={"Day":("white","black"),"Night":("#111","#FFF"),"High Contrast":("#FF0","#000")}
    bg,fg=theme.get(template,("white","black"))
    img=Image.new("RGB",(W,H),bg); d=ImageDraw.Draw(img)
    try:
        fb=ImageFont.truetype("arialbd.ttf",150); fn=ImageFont.truetype("arial.ttf",120)
    except: fb=fn=ImageFont.load_default()
    m=30
    d.text((W//2,50),"F-14B TAKEOFF CARD",font=fb,fill=fg,anchor="mm")
    d.rectangle([m,100,W-m,220],outline=fg,width=6)
    d.text((m+10,110),f"GW: {int(weight):,} lbs",font=fn,fill=fg)
    d.text((200,110),f"OAT: {int(temp)} °C",font=fn,fill=fg)
    d.text((380,110),f"Elev: {int(alt)} ft",font=fn,fill=fg)
    d.text((m+10,170),f"Flaps: {flap_name}",font=fn,fill=fg)
    d.text((300,170),f"Thrust: {thrust_label}",font=fn,fill=fg)
    y=250; draw_flap_indicator(d, flap_name,(50,y))
    d.text((50,y+120),f"Vs {results['Vs']}  V1 {results['V1']}  Vr {results['Vr']}  V2 {results['V2']}",font=fb,fill=fg)
    d.text((50,400),f"BFL: {results['bfl_ft']:,} ft",font=fb,fill=fg)
    d.text((50,460),f"Accel-Go: {results['go_ft']:,}   Accel-Stop: {results['stop_ft']:,}",font=fb,fill=fg)
    d.text((50,540),f"Trim: {results.get('trim','--')}°  RPM: {results.get('rpm','--')}%",font=fb,fill=fg)
    if carrier: d.text((50,600),"CARRIER MODE",font=fb,fill="red")
    img=img.resize((512,768),resample=Image.LANCZOS)
    buf=io.BytesIO(); img.save(buf,"PNG"); buf.seek(0); return buf

def kneeboard_pdf(results, weight, temp, alt, flap_name, thrust_label, carrier, template="Day"):
    img=Image.open(kneeboard_png(results, weight, temp, alt, flap_name, thrust_label, carrier, template))
    buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=portrait((img.width,img.height)))
    c.drawImage(ImageReader(img),0,0,width=img.width,height=img.height); c.showPage(); c.save(); buf.seek(0); return buf

# =========================
# App
# =========================
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

ensure_starter_perf()
perf_df, perf_source = load_perf_tables()

# ---- DCS airports CSV ----
# Expected columns: map,airport_name,runway_id,length_ft,slope_percent,le_elev_ft,he_elev_ft
dcs = pd.read_csv("dcs_airports.csv",
                  dtype={"map":"string","airport_name":"string","runway_id":"string"},
                  keep_default_na=True, na_values=["","NA","N/A","null","None"])
for col in ["length_ft","slope_percent","le_elev_ft","he_elev_ft"]:
    if col in dcs.columns: dcs[col]=pd.to_numeric(dcs[col],errors="coerce")
    else: dcs[col]=np.nan

# 1) Theatre/Airport/Runway  (Field Elevation here)
st.header("1) DCS Theatre / Airport / Runway")
theatres = sorted([t for t in dcs["map"].dropna().unique()])
theatre = st.selectbox("Theatre", theatres)
df_map = dcs[dcs["map"]==theatre]
airports = sorted([a for a in df_map["airport_name"].dropna().unique()])
airport = st.selectbox("Airport", airports)
df_ap = df_map[df_map["airport_name"]==airport]
rwy_pair = st.selectbox("Runway (pair)", [r for r in df_ap["runway_id"].dropna().unique()])

endA,endB,hdgA,hdgB = runway_heads_from_id(rwy_pair)
end = st.radio("Runway End (departure)", [f"{endA} ({hdgA}°)", f"{endB} ({hdgB}°)"], horizontal=True)
dep_end = endA if end.startswith(endA) else endB
dep_hdg = hdgA if dep_end==endA else hdgB

row = df_ap[df_ap["runway_id"]==rwy_pair].iloc[0]
length_ft = int(row["length_ft"]) if not np.isnan(row["length_ft"]) else 0
le_elev = row.get("le_elev_ft",np.nan); he_elev=row.get("he_elev_ft",np.nan)
if length_ft>0 and not np.isnan(le_elev) and not np.isnan(he_elev):
    slope_dir = ((he_elev - le_elev)/length_ft*100.0) if dep_end==endA else ((le_elev - he_elev)/length_ft*100.0)
    field_elev_auto = int(le_elev if dep_end==endA else he_elev)
else:
    slope_dir = float(row["slope_percent"]) if not np.isnan(row["slope_percent"]) else 0.0
    field_elev_auto = int(le_elev if not np.isnan(le_elev) else (he_elev if not np.isnan(he_elev) else 0))
if length_ft == 0:
    length_ft = st.number_input("Runway Length missing — enter value (ft)", min_value=1000, max_value=20000, value=8000, step=50)
field_elev = st.number_input("Field Elevation (ft)", min_value=0, max_value=12000, value=int(field_elev_auto), step=1)
st.caption(f"Runway length: **{length_ft:,} ft**  |  Slope (takeoff dir): **{slope_dir:.1f}%**  |  Elevation: **{field_elev} ft**")

# 2) Weather
st.header("2) Weather")
oat = st.number_input("OAT (°C)", min_value=-40, max_value=55, value=15, step=1)
wind_mode = st.radio("Headwind input", ["Manual (head/tailwind in kt)","From DCS Mission Briefing"], index=0)
if wind_mode == "Manual (head/tailwind in kt)":
    headwind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", min_value=-40.0, max_value=40.0, value=0.0, step=1.0)
    crosswind_kt = 0.0; cross_from = "n/a"
else:
    c1,c2 = st.columns(2)
    with c1: wind_dir = st.number_input("Wind Direction (° FROM, briefing)", min_value=0, max_value=359, value=0, step=1)
    with c2: wind_spd = st.number_input("Wind Speed (kt)", min_value=0, max_value=100, value=0, step=1)
    headwind_kt, crosswind_kt, cross_from = wind_components(dep_hdg, wind_dir, wind_spd)
    headwind_kt = round(headwind_kt,1); crosswind_kt = round(crosswind_kt,1)
    if headwind_kt >= 0:
        st.info(f"Headwind: **+{headwind_kt} kt**  |  Crosswind: **{crosswind_kt} kt from {cross_from}**")
    else:
        st.warning(f"Tailwind: **{headwind_kt} kt**  |  Crosswind: **{crosswind_kt} kt from {cross_from}**")
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
clearway_cap_ft = min(int(clearway_ft), int(length_ft)//2)
st.caption("121 checks: ASD ≤ RWY+Stopway; OEI Takeoff Distance ≤ RWY+min(Clearway, ½RWY); AEO TOR ≤ RWY (proxy).")

# 3) Weight & Balance (GW capped)
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
carrier = st.checkbox("Carrier Ops Mode (Catapult)", value=False)
auto_flaps = st.checkbox("Auto-select flap setting?", value=True)
if auto_flaps: flap_name = auto_flap(gross, length_ft)
else:          flap_name = st.selectbox("Flap Setting", list(FLAP_OPTIONS.keys()), index=1)
flap_deg = FLAP_OPTIONS[flap_name]
thrust_choice = st.selectbox("Thrust Rating (manual mode only)", THRUST_LEVELS_UI, index=0)
template = st.selectbox("Kneeboard Template", ["Day","Night","High Contrast"])

# 5) Intersection feasibility (simple; full-length used here)
st.header("5) Intersection Feasibility (auto)")
margin = st.number_input("Extra safety margin beyond BFL (ft)", min_value=0, max_value=2000, value=0, step=50)
runway_available_ft = int(length_ft)

# ===== PERFORMANCE / SELECTION =====
pa_ft = int(field_elev)

def with_15pct(asd_ft, agd_ft):
    return int(round(asd_ft*1.15)), int(round(agd_ft*1.15))

def compute_static(thrust_mode):
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
    toy = toy_perf(int(gross), int(oat), pa_ft, flap_deg,
                   "Afterburner" if thrust_mode=="Afterburner" else "Military",
                   float(headwind_kt), float(slope_dir), carrier, rpm=None)
    toy["source"] = "TOY"
    return toy

selected = None
selected_flap_name = None
selected_flap_deg = None

if apply_part121 and perf_df is not None:
    flap_candidates = [("Flaps Up", 0), ("Maneuvering Flaps", 20), ("Flaps Full", 40)] if auto_flaps else [(flap_name, flap_deg)]
    best = None
    for fname, fdeg in flap_candidates:
        cand = evaluate_combo_for_121(perf_df, fdeg, int(gross), pa_ft, int(oat),
                                      int(runway_available_ft), runway_cond, int(stopway_ft), int(clearway_cap_ft))
        if cand:
            key = (cand["rpm"], cand["bfl_ft"])
            if (best is None) or (key < best[0]):
                best = (key, cand, fname, fdeg)
    if best:
        _, selected, selected_flap_name, selected_flap_deg = best
        base = selected.copy()
        base["trim"] = max(5.0, min(12.0, round(8.0 + (int(gross)-50000)/10000.0,1)))
        base["source"] = f"NATOPS ({perf_source})"
        st.success(f"✅ Part 121 PASS — Selected **{selected_flap_name}**, Target RPM ≈ **{base['rpm']}%**")
    else:
        st.error("⛔ Part 121 not satisfied for any flap with available thrust (even AB). Try lighter weight, cooler temps, longer runway, or different runway.")
        base = compute_static("Afterburner")
        selected_flap_name, selected_flap_deg = flap_name, flap_deg
else:
    # Manual mode (no 121 gating): use user's thrust selection or dynamic 60% solver
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
            # toy fallback bisection on rpm
            def meets(rpm):
                r = toy_perf(int(gross), int(oat), pa_ft, flap_deg, "Minimum", float(headwind_kt), float(slope_dir), carrier, rpm=float(rpm))
                return r["go_ft"] <= int(0.60*runway_available_ft), r
            ok_ab, r_ab = meets(RPM_AB)
            if not ok_ab:
                st.error("⛔ Not feasible to reach Vr by 60% runway even at AB (toy).")
                base = r_ab; base["source"]="TOY"
            else:
                ok_mil, r_mil = meets(RPM_MIL)
                if ok_mil: lo, hi = RPM_MIN, RPM_MIL; r_best = r_mil
                else:      lo, hi = RPM_MIL, RPM_AB;   r_best = r_ab
                for _ in range(25):
                    mid = 0.5*(lo+hi)
                    ok, r_mid = meets(mid)
                    if ok: hi = mid; r_best = r_mid
                    else:  lo = mid
                base = r_best; base["source"]="TOY"
                st.success(f"✅ Minimum Required thrust solved (toy): ~**{base['rpm']}% RPM** meets AGD ≤ 60% RWY.")
    else:
        base = compute_static("Military" if thrust_choice=="Military" else "Afterburner")
    selected_flap_name, selected_flap_deg = flap_name, flap_deg

# BFL + intersection feasibility
required_len = max(base["bfl_ft"], 0) + int(margin)
if required_len <= runway_available_ft:
    max_offset = runway_available_ft - required_len
    st.success(f"Intersection feasible. Max allowable offset: **{int(max_offset):,} ft**  (Effective {int(required_len):,} ft).")
else:
    st.error(f"Intersection NOT feasible: need {int(required_len):,} ft; available {int(runway_available_ft):,} ft.")

safety_color = runway_safety_color(runway_available_ft, base["bfl_ft"])
st.markdown(f"**Runway Safety:** <span style='color:{safety_color}'>{safety_color.upper()}</span>", unsafe_allow_html=True)

# Final output
st.subheader("Final Numbers")
st.write(f"**Selected Flap:** {selected_flap_name} ({selected_flap_deg}°)")
st.write(f"**Vs:** {base['Vs']} kt   **V1:** {base['V1']} kt   **Vr:** {base['Vr']} kt   **V2:** {base['V2']} kt")
st.write(f"**Balanced Field Length:** {base['bfl_ft']:,} ft")
st.write(f"**Accel-Go:** {base['go_ft']:,} ft   **Accel-Stop:** {base['stop_ft']:,} ft")
st.write(f"**Trim:** {base.get('trim','--')}° NU   **Target RPM:** {base.get('rpm','--')}%")
st.caption(f"Source: **{base.get('source','')}**  |  121 Mode: {'ON' if apply_part121 else 'OFF'}  |  RWY: {runway_cond}, Stopway {int(stopway_ft):,} ft, Clearway {int(clearway_cap_ft):,} ft")

# Kneeboard export
thrust_label = "Minimum Required" if (apply_part121 and perf_df is not None and selected) or thrust_choice.startswith("Minimum") else thrust_choice
png = kneeboard_png(base, int(gross), int(oat), int(field_elev), selected_flap_name, thrust_label, carrier, template)
st.image(png, caption="VR-Ready Kneeboard (512×768)", use_column_width=True)
st.download_button("🖼️ Export Takeoff Card (PNG, VR-ready)",
                   data=png, file_name=f"F14_Takeoff_{theatre}_{airport}_{rwy_pair}_{dep_end}.png",
                   mime="image/png")
pdf = kneeboard_pdf(base, int(gross), int(oat), int(field_elev), selected_flap_name, thrust_label, carrier, template)
st.download_button("📄 Export Takeoff Card (PDF, VR-ready)",
                   data=pdf, file_name=f"F14_Takeoff_{theatre}_{airport}_{rwy_pair}_{dep_end}.pdf",
                   mime="application/pdf")
