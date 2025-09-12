import streamlit as st
import math, io, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader

# -----------------------------
# Constants & Aircraft Data
# -----------------------------
g = 9.80665
ft2m = 0.3048
lb2kg = 0.45359237
m2kt = 1.94384
m2ft = 3.28084

WING_AREA_M2 = 52.5
CLmax_by_flap = {0:1.2, 20:1.7, 40:2.1}
brake_decel_g = 0.35
rotation_time = 2.0
empty_weight_lbs = 40000

flap_options = {"Flaps Full":40,"Maneuvering Flaps":20,"Flaps Up":0}
def auto_flap(weight_lbs, runway_length_ft):
    if runway_length_ft < 5000 or weight_lbs > 70000: return "Flaps Full"
    elif runway_length_ft < 8000 or weight_lbs > 60000: return "Maneuvering Flaps"
    else: return "Flaps Up"

# -----------------------------
# Physics & Calculations
# -----------------------------
def air_density(alt_ft, oat_c):
    alt_m = alt_ft * ft2m
    rho0 = 1.225
    return rho0 * math.exp(-alt_m/8000)

def v_stall(weight_kg, rho, S, CLmax):
    W = weight_kg * g
    return math.sqrt(2*W/(rho*S*CLmax))

def accelerate_stop_distance(V1): return (V1**2)/(2*brake_decel_g*g) + 150
def accelerate_go_distance(Vr, V2): return Vr*rotation_time + V2*2.5

def runway_safety_color(runway_length_ft, balanced_field_ft):
    margin = runway_length_ft - balanced_field_ft
    if margin > 500: return "green"
    elif margin > 0: return "yellow"
    else: return "red"

def carrier_takeoff(Vr, trim, carrier_mode):
    if carrier_mode: Vr -= 40; trim += 2
    return Vr, trim

def recommend_thrust(weight_lbs, flap_deg, runway_length_ft):
    if runway_length_ft < 5000: return "Afterburner"
    elif weight_lbs > 65000 or flap_deg < 20: return "Military"
    else: return "Minimum Required"

def find_takeoff_speeds(weight_lbs, oat_c, alt_ft, flap_deg, thrust_mode, wind_kt=0, slope_percent=0, carrier_mode=False):
    weight_kg = weight_lbs * lb2kg
    rho = air_density(alt_ft, oat_c)
    CLmax = CLmax_by_flap.get(flap_deg,1.7)

    Vs = v_stall(weight_kg, rho, WING_AREA_M2, CLmax)
    Vs -= 0.5*wind_kt*m2kt
    Vr = Vs*1.05
    V2 = Vs*1.2
    V1 = Vs*1.04

    Vr, trim = carrier_takeoff(Vr, 8.0+(weight_lbs-50000)/10000, carrier_mode)

    stop_dist_m = accelerate_stop_distance(V1)
    go_dist_m = accelerate_go_distance(Vr, V2)
    balanced_field_m = max(stop_dist_m, go_dist_m)

    stop_dist_ft = stop_dist_m*m2ft*(1+slope_percent/100)
    go_dist_ft = go_dist_m*m2ft*(1+slope_percent/100)
    balanced_field_ft = balanced_field_m*m2ft*(1+slope_percent/100)

    trim = max(5.0,min(12.0,trim))
    rpm_target = {"Afterburner":102,"Military":96,"Minimum Required":92}[thrust_mode]

    return {
        "Vs (kt)": round(Vs*m2kt,1),
        "V1 (kt)": round(V1*m2kt,1),
        "Vr (kt)": round(Vr*m2kt,1),
        "V2 (kt)": round(V2*m2kt,1),
        "Balanced Field Length (ft)": round(balanced_field_ft),
        "Accel-Go Distance (ft)": round(go_dist_ft),
        "Accel-Stop Distance (ft)": round(stop_dist_ft),
        "Takeoff Trim (° NU)": round(trim,1),
        "Target Engine RPM (%)": rpm_target,
        "Thrust Mode": thrust_mode
    }

# -----------------------------
# Load DCS Airport/Runway Data
# -----------------------------
dcs_df = pd.read_csv("dcs_airports.csv")

# -----------------------------
# VR Graphics Functions
# -----------------------------
def draw_flap_indicator(draw, flap_name, position, size=100):
    x, y = position
    draw.rectangle([x,y,x+size,y+size//4],outline="black",width=5)
    fill_width = {"Flaps Up":0.33,"Maneuvering Flaps":0.66,"Flaps Full":1.0}[flap_name]*size
    draw.rectangle([x,y,x+fill_width,y+size//4],fill="blue")
    draw.text((x+size/2, y+size//4+5), flap_name, fill="black", anchor="ms")

def generate_png_image_vr(results, weight, temp, alt, flap_name, carrier_mode, template="Day"):
    scale = 6
    width, height = 512*scale, 768*scale
    colors = {"Day":("white","black"), "Night":("#111111","#FFFFFF"), "High Contrast":("#FFFF00","#000000")}
    bg_color, font_color = colors.get(template,("white","black"))
    img = Image.new("RGB",(width,height),bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font_bold = ImageFont.truetype("arialbd.ttf", 150)
        font_normal = ImageFont.truetype("arial.ttf", 120)
    except: font_bold=font_normal=ImageFont.load_default()

    margin = 30
    draw.text((width//2,50),"F-14B TAKEOFF CARD",font=font_bold,fill=font_color,anchor="mm")
    draw.rectangle([margin,100,width-margin,220],outline=font_color,width=6)
    draw.text((margin+10,110),f"GW: {weight} lbs",font=font_normal,fill=font_color)
    draw.text((200,110),f"OAT: {temp} °C",font=font_normal,fill=font_color)
    draw.text((380,110),f"Elev: {alt} ft",font=font_normal,fill=font_color)
    draw.text((margin+10,170),f"Flaps: {flap_name}",font=font_normal,fill=font_color)
    draw.text((300,170),f"Thrust: {results['Thrust Mode']}",font=font_normal,fill=font_color)
    y = 250
    draw_flap_indicator(draw, flap_name, (50,y))
    draw.text((50,y+120),f"Vs: {results['Vs (kt)']} kt   V1: {results['V1 (kt)']} kt   Vr: {results['Vr (kt)']} kt   V2: {results['V2 (kt)']} kt",font=font_bold,fill=font_color)
    draw.text((50,400),f"Balanced Field: {results['Balanced Field Length (ft)']} ft",font=font_bold,fill=font_color)
    draw.text((50,460),f"Accel-Go: {results['Accel-Go Distance (ft)']} ft   Accel-Stop: {results['Accel-Stop Distance (ft)']} ft",font=font_bold,fill=font_color)
    draw.text((50,540),f"Trim: {results['Takeoff Trim (° NU)']}° NU   Target RPM: {results['Target Engine RPM (%)']}%",font=font_bold,fill=font_color)
    if carrier_mode: draw.text((50,600),"CARRIER MODE",font=font_bold,fill="red")
    img = img.resize((512,768),resample=Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer,format="PNG")
    buffer.seek(0)
    return buffer

def generate_pdf_vr(results, weight, temp, alt, flap_name, carrier_mode, template="Day"):
    png_bytes = generate_png_image_vr(results, weight, temp, alt, flap_name, carrier_mode, template)
    img = Image.open(png_bytes)
    buffer_pdf = io.BytesIO()
    c = canvas.Canvas(buffer_pdf,pagesize=portrait((img.width, img.height)))
    c.drawImage(ImageReader(img), 0, 0, width=img.width, height=img.height)
    c.showPage(); c.save(); buffer_pdf.seek(0)
    return buffer_pdf

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="centered")
st.title("F-14B Takeoff Calculator (DCS VR)")

# Weight
use_gw = st.checkbox("Enter Gross Weight directly?", value=False)
if use_gw:
    weight = st.number_input("Gross Weight (lbs)", 30000, 80000, 72000)
else:
    fuel_lbs = st.number_input("Fuel Load (lbs)",0,15000,5000)
    ordnance_lbs = st.number_input("Ordnance Load (lbs)",0,10000,2000)
    weight = empty_weight_lbs + fuel_lbs + ordnance_lbs

# Takeoff Conditions
temp = st.number_input("OAT (°C)", -30, 50, 15)
alt = st.number_input("Field Elevation (ft)",0,8000,0)

# -----------------------------
# DCS Airport & Runway Selection
# -----------------------------
airport_name = st.selectbox("Select DCS Airport", sorted(dcs_df["airport_name"].unique()))
runways = dcs_df[dcs_df["airport_name"]==airport_name]
runway_id = st.selectbox("Select Runway", runways["runway_id"])
selected_runway = runways[runways["runway_id"]==runway_id].iloc[0]
runway_length_ft = selected_runway["length_ft"]
slope_percent = selected_runway["slope_percent"]

# Intersection Takeoff
intersection_offset_ft = st.number_input("Intersection Offset (ft)", 0, runway_length_ft, 0)
effective_runway_ft = runway_length_ft - intersection_offset_ft

# Flaps
auto_flap_sel = st.checkbox("Auto-select flap setting?", value=True)
if auto_flap_sel:
    flap_name = auto_flap(weight, effective_runway_ft)
else:
    flap_name = st.selectbox("Flap Setting", list(flap_options.keys()), index=1)
flap_deg = flap_options[flap_name]

# Thrust & Environment
thrust_mode = st.selectbox("Thrust Rating", ["Afterburner","Military","Minimum Required"], index=1)
carrier_mode = st.checkbox("Carrier Ops Mode (Catapult)", value=False)
template = st.selectbox("Kneeboard Template", ["Day","Night","High Contrast"])
wind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", -30,30,0)
auto_thrust = recommend_thrust(weight, flap_deg, effective_runway_ft)
st.info(f"Auto Thrust Recommendation: {auto_thrust}")

# Calculation
if st.button("Calculate"):
    res = find_takeoff_speeds(weight,temp,alt,flap_deg,thrust_mode,wind_kt,slope_percent,carrier_mode)
    st.subheader("Results")
    for k,v in res.items(): st.write(f"**{k}:** {v}")
    color = runway_safety_color(effective_runway_ft,res["Balanced Field Length (ft)"])
    st.markdown(f"**Runway Safety:** <span style='color:{color}'>{color.upper()}</span>",unsafe_allow_html=True)

    # PNG VR Export
    png_bytes = generate_png_image_vr(res,weight,temp,alt,flap_name,carrier_mode,template)
    st.image(png
