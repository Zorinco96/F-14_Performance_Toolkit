# f14_takeoff_app_dcs.py
import io, math, json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader

# =========================
# Constants
# =========================
g = 9.80665
ft2m = 0.3048
lb2kg = 0.45359237
m2kt = 1.94384
m2ft = 3.28084

# Visual & UI
APP_TITLE = "F-14B Takeoff Calculator (DCS VR, NATOPS-ready)"
TAILWIND_LIMIT_KT = 10
THRUST_LEVELS = ["Minimum Required", "Military", "Afterburner"]
FLAP_OPTIONS = {"Flaps Up": 0, "Maneuvering Flaps": 20, "Flaps Full": 40}

# Small physics scaffold (only used if no NATOPS tables available)
WING_AREA_M2 = 52.5
CLMAX = {0: 1.2, 20: 1.7, 40: 2.1}
brake_decel_g = 0.35
rotation_time = 2.0
empty_weight_lbs = 40000
THRUST_FACTOR = {"Minimum Required": 1.10, "Military": 1.00, "Afterburner": 0.80}

# =========================
# Helpers
# =========================
def ceil_kn(x): return int(math.ceil(x))

def auto_flap(weight_lbs, runway_ft):
    if runway_ft < 5000 or weight_lbs > 70000: return "Flaps Full"
    if runway_ft < 8000 or weight_lbs > 60000: return "Maneuvering Flaps"
    return "Flaps Up"

def air_density(alt_ft, oat_c):  # simple ISA proxy
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

# =========================
# NATOPS/Heatblur data layer
# =========================
"""
We support two optional CSVs (put them next to this app):
- perf_f14b.csv  (preferred if present)
- perf_f14d.csv  (fallback if B missing; from public F-14D NATOPS)

Expected tidy columns (one row per grid point):
 model, flap_deg, thrust, gw_lbs, press_alt_ft, oat_c, Vs_kt, V1_kt, Vr_kt, V2_kt, ASD_ft, AGD_ft
Where:
 - thrust in {"Minimum Required","Military","Afterburner"} (map NATOPS nomenclature as needed)
 - press_alt_ft: pressure altitude for the table row (use field elev if you don’t compute PA)
 - ASD_ft, AGD_ft are *raw* (the app will apply the +15% safety factor)

If no CSVs are found, we fall back to a (clearly-marked) toy model.
"""
@st.cache_data(show_spinner=False)
def load_perf_tables():
    for fname in ["perf_f14b.csv", "perf_f14d.csv"]:
        try:
            df = pd.read_csv(fname)
            # normalize
            df["model"] = df.get("model","").astype(str)
            df["flap_deg"] = pd.to_numeric(df["flap_deg"], errors="coerce")
            df["gw_lbs"] = pd.to_numeric(df["gw_lbs"], errors="coerce")
            df["press_alt_ft"] = pd.to_numeric(df["press_alt_ft"], errors="coerce")
            df["oat_c"] = pd.to_numeric(df["oat_c"], errors="coerce")
            for col in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["thrust"] = df["thrust"].map({
                "min":"Minimum Required","minimum":"Minimum Required","Minimum Required":"Minimum Required",
                "mil":"Military","military":"Military","Military":"Military",
                "ab":"Afterburner","afterburner":"Afterburner","Afterburner":"Afterburner"
            })
            return df, fname
        except Exception:
            continue
    return None, None

def trilinear_interp(df, flap_deg, thrust, gw_lbs, pa_ft, oat_c):
    """
    Simple multi-axis nearest-grid interpolation over (GW, PA, OAT) within a single (flap, thrust).
    We do piecewise linear along each axis using the 8-corner cube if possible; else nearest.
    """
    sub = df[(df["flap_deg"]==flap_deg) & (df["thrust"]==thrust)].copy()
    if sub.empty:
        # Try closest available flap in table
        candidates = df[df["thrust"]==thrust]["flap_deg"].dropna().unique()
        if len(candidates)==0: return None
        flap_deg = int(sorted(candidates, key=lambda x:abs(x-flap_deg))[0])
        sub = df[(df["flap_deg"]==flap_deg) & (df["thrust"]==thrust)].copy()
        if sub.empty: return None

    # axes
    GWs = np.sort(sub["gw_lbs"].dropna().unique())
    PAs = np.sort(sub["press_alt_ft"].dropna().unique())
    OATs= np.sort(sub["oat_c"].dropna().unique())
    if len(GWs)==0 or len(PAs)==0 or len(OATs)==0:
        return None

    # clamp to bounds
    gw1 = GWs[GWs<=gw_lbs].max(initial=GWs.min()); gw2 = GWs[GWs>=gw_lbs].min(initial=GWs.max())
    pa1 = PAs[PAs<=pa_ft].max(initial=PAs.min());   pa2 = PAs[PAs>=pa_ft].min(initial=PAs.max())
    t1  = OATs[OATs<=oat_c].max(initial=OATs.min());t2  = OATs[OATs>=oat_c].min(initial=OATs.max())

    # edge cases: exact or nearest
    if (gw1==gw2) and (pa1==pa2) and (t1==t2):
        row = sub[(sub["gw_lbs"]==gw1)&(sub["press_alt_ft"]==pa1)&(sub["oat_c"]==t1)].head(1)
        return row.iloc[0].to_dict() if not row.empty else None

    # get 8 corners
    def corner(gw,pa,t):
        r = sub[(sub["gw_lbs"]==gw)&(sub["press_alt_ft"]==pa)&(sub["oat_c"]==t)]
        return r.iloc[0] if not r.empty else None

    corners = {
        (gw1,pa1,t1): corner(gw1,pa1,t1),
        (gw1,pa1,t2): corner(gw1,pa1,t2),
        (gw1,pa2,t1): corner(gw1,pa2,t1),
        (gw1,pa2,t2): corner(gw1,pa2,t2),
        (gw2,pa1,t1): corner(gw2,pa1,t1),
        (gw2,pa1,t2): corner(gw2,pa1,t2),
        (gw2,pa2,t1): corner(gw2,pa2,t1),
        (gw2,pa2,t2): corner(gw2,pa2,t2),
    }
    if any(v is None for v in corners.values()):
        # fall back to nearest neighbor
        sub["dist"] = (abs(sub["gw_lbs"]-gw_lbs)
                       + abs(sub["press_alt_ft"]-pa_ft)
                       + abs(sub["oat_c"]-oat_c))
        row = sub.sort_values("dist").head(1)
        return row.iloc[0].to_dict() if not row.empty else None

    # weights
    def w(a,a1,a2):
        return 0.0 if a2==a1 else (a-a1)/(a2-a1)
    wg = w(gw_lbs, gw1, gw2); wp = w(pa_ft, pa1, pa2); wt = w(oat_c, t1, t2)

    # interpolate for each numeric perf key
    out = {"flap_deg":flap_deg,"thrust":thrust}
    keys = ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]
    for k in keys:
        # trilinear
        def val(c): return corners[c][k]
        v = (
            val((gw1,pa1,t1))*(1-wg)*(1-wp)*(1-wt) +
            val((gw2,pa1,t1))*   wg *(1-wp)*(1-wt) +
            val((gw1,pa2,t1))*(1-wg)*   wp *(1-wt) +
            val((gw2,pa2,t1))*   wg *   wp *(1-wt) +
            val((gw1,pa1,t2))*(1-wg)*(1-wp)*   wt  +
            val((gw2,pa1,t2))*   wg *(1-wp)*   wt  +
            val((gw1,pa2,t2))*(1-wg)*   wp *   wt  +
            val((gw2,pa2,t2))*   wg *   wp *   wt
        )
        out[k] = float(v)
    return out

def natops_calc(df, flap_deg, thrust, gw_lbs, field_elev_ft, oat_c):
    # If you don’t compute Pressure Altitude, field elevation is a decent proxy for DCS
    pa_ft = field_elev_ft
    row = trilinear_interp(df, flap_deg, thrust, gw_lbs, pa_ft, oat_c)
    if row is None:
        return None
    # +15% safety factor to AG/AS, then take BFL=max
    ASD = row["ASD_ft"]*1.15
    AGD = row["AGD_ft"]*1.15
    Vs  = ceil_kn(row["Vs_kt"])
    V1  = ceil_kn(row["V1_kt"])
    Vr  = ceil_kn(row["Vr_kt"])
    V2  = ceil_kn(row["V2_kt"])
    rpm = {"Afterburner":102,"Military":96,"Minimum Required":92}[thrust]
    # Trim: simple GW-based heuristic (no NATOPS trim table published)
    trim = max(5.0, min(12.0, 8.0 + (gw_lbs-50000)/10000.0))
    return dict(Vs=Vs, V1=V1, Vr=Vr, V2=V2,
                stop_ft=int(round(ASD)), go_ft=int(round(AGD)),
                bfl_ft=int(round(max(ASD,AGD))),
                trim=round(trim,1), rpm=rpm)

def toy_calc(gw_lbs, oat_c, field_elev_ft, flap_deg, thrust, headwind_kt, slope_pct, carrier):
    # Only used if no NATOPS tables exist; includes +15% safety factor and rounding
    rho = air_density(field_elev_ft, oat_c)
    Vs = v_stall(gw_lbs*lb2kg, rho, WING_AREA_M2, CLMAX.get(flap_deg,1.7))
    Vs = max(0.0, Vs - 0.5*(headwind_kt/m2kt))
    Vr = Vs*1.05; V2 = Vs*1.20; V1 = Vs*1.04
    trim = 8.0 + (gw_lbs-50000)/10000.0
    Vr, trim = carrier_adjust(Vr, trim, carrier)
    trim = max(5.0, min(12.0, trim))
    stop_m = accel_stop_m(V1)*1.15
    go_m   = accel_go_m(Vr,V2)*THRUST_FACTOR[thrust]*1.15
    slope_factor = 1 + slope_pct/100.0
    stop_ft = stop_m*m2ft*slope_factor
    go_ft   = go_m*m2ft*slope_factor
    bfl_ft  = max(stop_ft, go_ft)
    return dict(
        Vs=ceil_kn(Vs*m2kt), V1=ceil_kn(V1*m2kt), Vr=ceil_kn(Vr*m2kt), V2=ceil_kn(V2*m2kt),
        stop_ft=int(round(stop_ft)), go_ft=int(round(go_ft)), bfl_ft=int(round(bfl_ft)),
        trim=round(trim,1), rpm={"Afterburner":102,"Military":96,"Minimum Required":92}[thrust]
    )

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
    d.text((200,110),f"OAT: {temp} °C",font=fn,fill=fg)
    d.text((380,110),f"Elev: {int(alt)} ft",font=fn,fill=fg)
    d.text((m+10,170),f"Flaps: {flap_name}",font=fn,fill=fg)
    d.text((300,170),f"Thrust: {thrust_label}",font=fn,fill=fg)
    y=250; draw_flap_indicator(d, flap_name,(50,y))
    d.text((50,y+120),f"Vs {results['Vs']}  V1 {results['V1']}  Vr {results['Vr']}  V2 {results['V2']}",font=fb,fill=fg)
    d.text((50,400),f"BFL: {results['bfl_ft']:,} ft",font=fb,fill=fg)
    d.text((50,460),f"Accel-Go: {results['go_ft']:,}   Accel-Stop: {results['stop_ft']:,}",font=fb,fill=fg)
    d.text((50,540),f"Trim: {results['trim']}°  RPM: {results['rpm']}%",font=fb,fill=fg)
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

# Airports CSV (built from OurAirports via the script below)
dcs = pd.read_csv(
    "dcs_airports.csv",
    dtype={"map":"string","airport_name":"string","runway_id":"string"},
    keep_default_na=True, na_values=["","NA","N/A","null","None"]
)
for col in ["length_ft","slope_percent","le_elev_ft","he_elev_ft"]:
    if col in dcs.columns: dcs[col]=pd.to_numeric(dcs[col],errors="coerce")
    else: dcs[col]=np.nan

perf_df, perf_source = load_perf_tables()

# 1) THEATRE / AIRPORT / RUNWAY  (Field Elevation is here)
st.header("1) DCS Theatre / Airport / Runway")
theatres = sorted(dcs["map"].dropna().unique())
theatre = st.selectbox("Theatre", theatres)
df_map = dcs[dcs["map"]==theatre]
airports = sorted(df_map["airport_name"].dropna().unique())
airport = st.selectbox("Airport", airports)
df_ap = df_map[df_map["airport_name"]==airport]
rwy_pair = st.selectbox("Runway (pair)", df_ap["runway_id"].dropna().unique())

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
    length_ft = st.number_input("Runway Length missing — enter value (ft)", 1000, 20000, 8000, step=50)
field_elev = st.number_input("Field Elevation (ft)", 0, 12000, int(field_elev_auto))
st.caption(f"Runway length: **{length_ft:,} ft**  |  Slope (takeoff dir): **{slope_dir:.1f}%**  |  Elevation: **{field_elev} ft**")

# 2) WEATHER
st.header("2) Weather")
oat = st.number_input("OAT (°C)", -40, 55, 15)
wind_mode = st.radio("Headwind input", ["Manual (head/tailwind in kt)","From DCS Mission Briefing"], index=0)
if wind_mode == "Manual (head/tailwind in kt)":
    headwind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", -40, 40, 0.0)
    crosswind_kt = 0.0; cross_from = "n/a"
else:
    c1,c2 = st.columns(2)
    with c1: wind_dir = st.number_input("Wind Direction (° FROM, briefing)", 0, 359, 0)
    with c2: wind_spd = st.number_input("Wind Speed (kt)", 0, 100, 0)
    headwind_kt, crosswind_kt, cross_from = wind_components(dep_hdg, wind_dir, wind_spd)
    headwind_kt = round(headwind_kt,1); crosswind_kt = round(crosswind_kt,1)
    if headwind_kt >= 0:
        st.info(f"Headwind: **+{headwind_kt} kt**  |  Crosswind: **{crosswind_kt} kt from {cross_from}**")
    else:
        st.warning(f"Tailwind: **{headwind_kt} kt**  |  Crosswind: **{crosswind_kt} kt from {cross_from}**")
max_cx = st.number_input("Max crosswind advisory (kt)", 0, 50, 20)
if wind_mode!="Manual (head/tailwind in kt)" and crosswind_kt>max_cx:
    st.error(f"⚠ MAX CROSSWIND EXCEEDED: {crosswind_kt} kt (limit {max_cx} kt)")
if headwind_kt<0 and abs(headwind_kt)>TAILWIND_LIMIT_KT:
    st.error(f"⚠ MAX TAILWIND EXCEEDED: {abs(headwind_kt)} kt (limit {TAILWIND_LIMIT_KT} kt)")

# 3) WEIGHT & BALANCE
st.header("3) Weight & Balance")
use_gw = st.checkbox("Enter Gross Weight directly?", value=False)
if use_gw:
    gross = st.number_input("Gross Weight (lbs)", 30000, 80000, 72000, step=100)
else:
    fuel = st.number_input("Fuel Load (lbs)", 0, 20000, 5000, step=100)
    ordn = st.number_input("Ordnance Load (lbs)", 0, 20000, 2000, step=100)
    gross = empty_weight_lbs + fuel + ordn
st.caption(f"Computed Gross Weight: **{int(gross):,} lbs**")

# 4) CONFIGURATION
st.header("4) Configuration")
carrier = st.checkbox("Carrier Ops Mode (Catapult)", value=False)
auto_flaps = st.checkbox("Auto-select flap setting?", value=True)
if auto_flaps:
    flap_name = auto_flap(gross, length_ft)
else:
    flap_name = st.selectbox("Flap Setting", list(FLAP_OPTIONS.keys()), index=1)
flap_deg = FLAP_OPTIONS[flap_name]
thrust_user = st.selectbox("Thrust Rating (your selection)", THRUST_LEVELS, index=1)
template = st.selectbox("Kneeboard Template", ["Day","Night","High Contrast"])

# Baseline perf using NATOPS tables if present; otherwise toy
def compute(thrust):
    if perf_df is not None:
        r = natops_calc(perf_df, flap_deg, thrust, gross, field_elev, oat)
        if r is not None:
            return r, "NATOPS"
    # Fallback
    r = toy_calc(gross, oat, field_elev, flap_deg, thrust, headwind_kt, slope_dir, carrier)
    return r, "TOY"

base, source_used = compute(thrust_user)
bfl_needed = base["bfl_ft"]

# Thrust auto-upgrade if needed (for full runway)
thrust_final = thrust_user; overridden=False
ok = base["bfl_ft"] <= length_ft
if not ok:
    for lvl in THRUST_LEVELS[THRUST_LEVELS.index(thrust_user):] + THRUST_LEVELS:
        r_try, src = compute(lvl)
        if r_try["bfl_ft"] <= length_ft:
            if lvl != thrust_user: overridden=True
            base, source_used = r_try, src
            thrust_final = lvl
            break

st.markdown("---")

# 5) AUTO INTERSECTION CHECK
st.header("5) Intersection Feasibility (auto)")
margin = st.number_input("Extra safety margin beyond BFL (ft)", 0, 2000, 0, step=50)
required_len = max(0, base["bfl_ft"] + margin)
if required_len <= length_ft:
    max_offset = length_ft - required_len
    st.success(f"✅ Feasible. **Max allowable offset from {dep_end}: {int(max_offset):,} ft** "
               f"(Effective runway: {int(required_len):,} ft; BFL {base['bfl_ft']:,} ft; source: {source_used}{' ('+perf_source+')' if perf_source else ''}).")
else:
    max_offset = 0
    st.error(f"⛔ Not feasible: Need {int(required_len):,} ft; available {length_ft:,} ft. (source: {source_used}{' ('+perf_source+')' if perf_source else ''})")

if overridden:
    st.markdown(
        f"**Thrust (your selection):** <span style='color:#B00020;'>{thrust_user}</span> → "
        f"**OVERRIDDEN to:** <span style='color:#00B050;'>{thrust_final}</span>",
        unsafe_allow_html=True
    )

apply = st.checkbox("Apply suggested offset automatically")
offset_used = int(max_offset if apply else 0)
effective_runway = max(0, length_ft - offset_used)
safety_color = runway_safety_color(effective_runway, base["bfl_ft"])
st.markdown(f"**Effective runway:** {effective_runway:,} ft — "
            f"**Runway Safety:** <span style='color:{safety_color}'>{safety_color.upper()}</span>",
            unsafe_allow_html=True)

# Final numbers & export
st.subheader("Final Numbers")
st.write(f"**Vs:** {base['Vs']} kt   **V1:** {base['V1']} kt   **Vr:** {base['Vr']} kt   **V2:** {base['V2']} kt")
st.write(f"**Balanced Field Length:** {base['bfl_ft']:,} ft")
st.write(f"**Accel-Go:** {base['go_ft']:,} ft   **Accel-Stop:** {base['stop_ft']:,} ft")
st.write(f"**Trim:** {base['trim']}° NU   **Target RPM:** {base['rpm']}%")
st.caption(f"Performance source: **{source_used}**{' — '+perf_source if perf_source else ''}")

png = kneeboard_png(base, gross, oat, field_elev, flap_name, thrust_final, carrier, template)
st.image(png, caption="VR-Ready Kneeboard (512×768)", use_column_width=True)
st.download_button("🖼️ Export Takeoff Card (PNG, VR-ready)",
                   data=png, file_name=f"F14_Takeoff_{theatre}_{airport}_{rwy_pair}_{dep_end}_off{offset_used}.png",
                   mime="image/png")
pdf = kneeboard_pdf(base, gross, oat, field_elev, flap_name, thrust_final, carrier, template)
st.download_button("📄 Export Takeoff Card (PDF, VR-ready)",
                   data=pdf, file_name=f"F14_Takeoff_{theatre}_{airport}_{rwy_pair}_{dep_end}_off{offset_used}.pdf",
                   mime="application/pdf")
