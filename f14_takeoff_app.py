import io, math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader

# ===== Constants & simple (toy) perf model =====
g = 9.80665
ft2m = 0.3048
lb2kg = 0.45359237
m2kt = 1.94384
m2ft = 3.28084

WING_AREA_M2 = 52.5
CLMAX = {0: 1.2, 20: 1.7, 40: 2.1}
brake_decel_g = 0.35
rotation_time = 2.0
empty_weight_lbs = 40000

TAILWIND_LIMIT_KT = 10
THRUST_LEVELS = ["Minimum Required", "Military", "Afterburner"]
THRUST_FACTOR = {"Minimum Required": 1.10, "Military": 1.00, "Afterburner": 0.80}

FLAP_OPTIONS = {"Flaps Up": 0, "Maneuvering Flaps": 20, "Flaps Full": 40}

def auto_flap(weight_lbs, runway_ft):
    if runway_ft < 5000 or weight_lbs > 70000: return "Flaps Full"
    if runway_ft < 8000 or weight_lbs > 60000: return "Maneuvering Flaps"
    return "Flaps Up"

def air_density(alt_ft, oat_c):
    return 1.225 * math.exp(-(alt_ft*ft2m)/8000)

def v_stall(weight_kg, rho, S, clmax):
    return math.sqrt((2*weight_kg*g)/(rho*S*clmax))

def accel_stop(V1_mps):    return (V1_mps**2)/(2*brake_decel_g*g) + 150
def accel_go(Vr_mps, V2_mps): return Vr_mps*rotation_time + V2_mps*2.5

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

def perf_calc(weight_lbs, oat_c, field_alt_ft, flap_deg, thrust_mode, headwind_kt, slope_pct, carrier):
    rho = air_density(field_alt_ft, oat_c)
    Vs = v_stall(weight_lbs*lb2kg, rho, WING_AREA_M2, CLMAX.get(flap_deg,1.7))
    Vs = max(0.0, Vs - 0.5*(headwind_kt/m2kt))   # crude wind effect

    Vr = Vs*1.05; V2 = Vs*1.20; V1 = Vs*1.04
    trim = 8.0 + (weight_lbs-50000)/10000.0
    Vr, trim = carrier_adjust(Vr, trim, carrier)
    trim = max(5.0, min(12.0, trim))

    stop_m = accel_stop(V1)
    go_m = accel_go(Vr, V2)*THRUST_FACTOR[thrust_mode]
    bfl_m = max(stop_m, go_m)

    slope_factor = 1 + slope_pct/100.0
    stop_ft = stop_m*m2ft*slope_factor
    go_ft   = go_m*m2ft*slope_factor
    bfl_ft  = bfl_m*m2ft*slope_factor

    rpm = {"Afterburner":102,"Military":96,"Minimum Required":92}[thrust_mode]
    return dict(
        Vs=Vs*m2kt, V1=V1*m2kt, Vr=Vr*m2kt, V2=V2*m2kt,
        stop_ft=int(round(stop_ft)),
        go_ft=int(round(go_ft)),
        bfl_ft=int(round(bfl_ft)),
        trim=round(trim,1),
        rpm=rpm
    )

def runway_safety_color(available_ft, bfl_ft):
    margin = available_ft - bfl_ft
    return "green" if margin>500 else ("yellow" if margin>0 else "red")

# ===== VR card =====
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
    d.text((50,y+120),f"Vs {results['Vs']:.1f}  V1 {results['V1']:.1f}  Vr {results['Vr']:.1f}  V2 {results['V2']:.1f}",font=fb,fill=fg)
    d.text((50,400),f"BFL: {results['bfl_ft']:,} ft",font=fb,fill=fg)
    d.text((50,460),f"Accel-Go: {results['go_ft']:,}   Accel-Stop: {results['stop_ft']:,}",font=fb,fill=fg)
    d.text((50,540),f"Trim: {results['trim']}°  RPM: {results['rpm']}%",font=fb,fill=fg)
    if carrier: d.text((50,600),"CARRIER MODE",font=fb,fill="red")
    img=img.resize((512,768),resample=Image.LANCZOS)
    buf=io.BytesIO(); img.save(buf,"PNG"); buf.seek(0); return buf

def kneeboard_pdf(results, weight, temp, alt, flap_name, thrust_label, carrier, template="Day"):
    png=kneeboard_png(results, weight, temp, alt, flap_name, thrust_label, carrier, template)
    img=Image.open(png)
    buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=portrait((img.width,img.height)))
    c.drawImage(ImageReader(img),0,0,width=img.width,height=img.height); c.showPage(); c.save(); buf.seek(0); return buf

# ===== App =====
st.set_page_config(page_title="F-14B Takeoff (DCS VR)", layout="centered")
st.title("F-14B Takeoff Calculator (DCS VR)")

# Load airports
dcs = pd.read_csv(
    "dcs_airports.csv",
    dtype={"map":"string","airport_name":"string","runway_id":"string"},
    keep_default_na=True, na_values=["","NA","N/A","null","None"]
)
for col in ["length_ft","slope_percent","le_elev_ft","he_elev_ft"]:
    if col in dcs.columns: dcs[col]=pd.to_numeric(dcs[col],errors="coerce")
    else: dcs[col]=np.nan

# --- 1) Theatre / Airport / Runway (auto-pop populate length/slope/alt) ---
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
length_ft = int(row["length_ft"]) if not np.isnan(row["length_ft"]) else 8000
le_elev = row.get("le_elev_ft",np.nan); he_elev=row.get("he_elev_ft",np.nan)
# directional slope & field elevation
if not np.isnan(le_elev) and not np.isnan(he_elev) and length_ft>0:
    slope_dir = ((he_elev - le_elev)/length_ft*100.0) if dep_end==endA else ((le_elev - he_elev)/length_ft*100.0)
    field_elev_auto = int(le_elev if dep_end==endA else he_elev)
else:
    slope_dir = float(row["slope_percent"]) if not np.isnan(row["slope_percent"]) else 0.0
    field_elev_auto = int(le_elev if not np.isnan(le_elev) else (he_elev if not np.isnan(he_elev) else 0))
st.caption(f"Auto runway length: **{length_ft:,} ft**  |  Auto slope (takeoff dir): **{slope_dir:.1f}%**  |  Auto field elev: **{field_elev_auto} ft**")

# --- 2) Weather Data ---
st.header("2) Weather Data")
wind_mode = st.radio("Headwind input", ["Manual (head/tailwind in kt)","From DCS Mission Briefing"], index=0)
if wind_mode == "Manual (head/tailwind in kt)":
    headwind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", -40, 40, 0)
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

# --- 3) Weight & Balance ---
st.header("3) Weight & Balance")
use_gw = st.checkbox("Enter Gross Weight directly?", value=False)
if use_gw:
    gross = st.number_input("Gross Weight (lbs)", 30000, 80000, 72000, step=100)
else:
    fuel = st.number_input("Fuel Load (lbs)", 0, 20000, 5000, step=100)
    ordn = st.number_input("Ordnance Load (lbs)", 0, 20000, 2000, step=100)
    gross = empty_weight_lbs + fuel + ordn
st.caption(f"Computed Gross Weight: **{int(gross):,} lbs**")

# Atmosphere inputs (field elevation auto-filled above; you can override)
c1,c2 = st.columns(2)
with c1: oat = st.number_input("OAT (°C)", -40, 55, 15)
with c2: field_elev = st.number_input("Field Elevation (ft)", 0, 12000, int(field_elev_auto))
carrier = st.checkbox("Carrier Ops Mode (Catapult)", value=False)

# --- 4) Configuration ---
st.header("4) Configuration")
auto_flaps = st.checkbox("Auto-select flap setting?", value=True)
if auto_flaps:
    flap_name = auto_flap(gross, length_ft)
else:
    flap_name = st.selectbox("Flap Setting", list(FLAP_OPTIONS.keys()), index=1)
flap_deg = FLAP_OPTIONS[flap_name]
thrust_user = st.selectbox("Thrust Rating (your selection)", THRUST_LEVELS, index=1)
template = st.selectbox("Kneeboard Template", ["Day","Night","High Contrast"])

# Compute baseline perf with selected config, full runway (no intersection yet)
base = perf_calc(gross, oat, field_elev, flap_deg, thrust_user, headwind_kt, slope_dir, carrier)
bfl_needed = base["bfl_ft"]

# Auto thrust override if needed for full runway (rare, but consistent behavior)
thrust_final = thrust_user; overridden=False
def meets_bfl(thrust):
    r = perf_calc(gross, oat, field_elev, flap_deg, thrust, headwind_kt, slope_dir, carrier)
    return r["bfl_ft"] <= length_ft, r
ok, r_try = meets_bfl(thrust_final)
if not ok:
    for lvl in THRUST_LEVELS[THRUST_LEVELS.index(thrust_user):] + THRUST_LEVELS:
        ok2, r2 = meets_bfl(lvl)
        if ok2:
            if lvl!=thrust_user: overridden=True
            thrust_final = lvl; r_try = r2; break
base = r_try; bfl_needed = base["bfl_ft"]

st.markdown("---")
# --- 5) Automatic intersection feasibility check ---
st.header("5) Intersection Feasibility (auto)")
# Tightest effective runway that still meets BFL (+ optional margin the user can adjust)
margin = st.number_input("Extra safety margin beyond BFL (ft)", 0, 2000, 0, step=50)
required_len = max(0, bfl_needed + margin)
if required_len <= length_ft:
    max_offset = length_ft - required_len
    st.success(f"✅ Intersection takeoff is feasible. **Max allowable offset** from **{dep_end}**: **{int(max_offset):,} ft** "
               f"(Effective runway: {int(required_len):,} ft, BFL {bfl_needed:,} ft).")
else:
    max_offset = 0
    st.error(f"⛔ Not feasible: Required {int(required_len):,} ft, available {length_ft:,} ft. "
             f"Try opposite end, reduce weight, increase thrust, or cooler temps.")

# Highlight thrust override
if overridden:
    st.markdown(
        f"**Thrust (your selection):** <span style='color:#B00020;'>{thrust_user}</span> → "
        f"**OVERRIDDEN to:** <span style='color:#00B050;'>{thrust_final}</span>",
        unsafe_allow_html=True
    )

# Option to apply suggested offset and preview final numbers
apply = st.checkbox("Apply suggested offset automatically")
offset_used = int(max_offset if apply else 0)
effective_runway = max(0, length_ft - offset_used)

# Recompute with effective runway by ensuring BFL ≤ effective_runway (thrust already chosen above)
# (Performance numbers themselves don't change with offset in this toy model; feasibility does.)
safety_color = runway_safety_color(effective_runway, bfl_needed)
st.markdown(f"**Effective runway for takeoff:** {effective_runway:,} ft — "
            f"**Runway Safety:** <span style='color:{safety_color}'>{safety_color.upper()}</span>", unsafe_allow_html=True)

# Show final numeric results (from base calc with thrust_final)
st.subheader("Final Numbers")
st.write(f"**Vs:** {base['Vs']:.1f} kt   **V1:** {base['V1']:.1f} kt   **Vr:** {base['Vr']:.1f} kt   **V2:** {base['V2']:.1f} kt")
st.write(f"**Balanced Field Length:** {base['bfl_ft']:,} ft")
st.write(f"**Accel-Go:** {base['go_ft']:,} ft   **Accel-Stop:** {base['stop_ft']:,} ft")
st.write(f"**Trim:** {base['trim']}° NU   **Target RPM:** {base['rpm']}%")
if headwind_kt >= 0:
    st.caption(f"Headwind used: +{headwind_kt:.1f} kt")
else:
    st.caption(f"Tailwind used: {headwind_kt:.1f} kt (limit {TAILWIND_LIMIT_KT} kt)")
if wind_mode != "Manual (head/tailwind in kt)":
    st.caption(f"Crosswind: {crosswind_kt:.1f} kt from {cross_from} (limit {max_cx} kt)")

# Export VR kneeboard (labels reflect thrust_final, flap_name)
png = kneeboard_png(
    {"Vs":base["Vs"],"V1":base["V1"],"Vr":base["Vr"],"V2":base["V2"],
     "bfl_ft":base["bfl_ft"],"go_ft":base["go_ft"],"stop_ft":base["stop_ft"],
     "trim":base["trim"],"rpm":base["rpm"]},
    gross, oat, field_elev, flap_name, thrust_final, carrier, template
)
st.image(png, caption="VR-Ready Kneeboard (512×768)", use_column_width=True)
st.download_button("🖼️ Export Takeoff Card (PNG, VR-ready)",
                   data=png, file_name=f"F14_Takeoff_{theatre}_{airport}_{rwy_pair}_{dep_end}_offset{offset_used}.png",
                   mime="image/png")
pdf = kneeboard_pdf(
    {"Vs":base["Vs"],"V1":base["V1"],"Vr":base["Vr"],"V2":base["V2"],
     "bfl_ft":base["bfl_ft"],"go_ft":base["go_ft"],"stop_ft":base["stop_ft"],
     "trim":base["trim"],"rpm":base["rpm"]},
    gross, oat, field_elev, flap_name, thrust_final, carrier, template
)
st.download_button("📄 Export Takeoff Card (PDF, VR-ready)",
                   data=pdf, file_name=f"F14_Takeoff_{theatre}_{airport}_{rwy_pair}_{dep_end}_offset{offset_used}.pdf",
                   mime="application/pdf")
