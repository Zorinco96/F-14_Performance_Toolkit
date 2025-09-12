import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader

# =========================
#  Constants & Aircraft Data
# =========================
g = 9.80665
ft2m = 0.3048
lb2kg = 0.45359237
m2kt = 1.94384
m2ft = 3.28084

WING_AREA_M2 = 52.5
# Simplified CLmax table by flap mode
CLmax_by_flap = {0: 1.2, 20: 1.7, 40: 2.1}

brake_decel_g = 0.35
rotation_time = 2.0
empty_weight_lbs = 40000

# Flap options (realistic nomenclature)
flap_options = {"Flaps Full": 40, "Maneuvering Flaps": 20, "Flaps Up": 0}

def auto_flap(weight_lbs: float, runway_length_ft: float) -> str:
    """Rule-of-thumb auto flap selection for the F-14 in DCS."""
    if runway_length_ft < 5000 or weight_lbs > 70000:
        return "Flaps Full"
    elif runway_length_ft < 8000 or weight_lbs > 60000:
        return "Maneuvering Flaps"
    else:
        return "Flaps Up"

# =========================
#  Performance Math (Toy model)
# =========================
def air_density(alt_ft: float, oat_c: float) -> float:
    alt_m = alt_ft * ft2m
    rho0 = 1.225
    return rho0 * math.exp(-alt_m / 8000)

def v_stall(weight_kg: float, rho: float, S: float, CLmax: float) -> float:
    W = weight_kg * g
    return math.sqrt(2 * W / (rho * S * CLmax))  # m/s

def accelerate_stop_distance(V1_mps: float) -> float:
    return (V1_mps**2) / (2 * brake_decel_g * g) + 150  # meters (toy)

def accelerate_go_distance(Vr_mps: float, V2_mps: float) -> float:
    return Vr_mps * rotation_time + V2_mps * 2.5  # meters (toy)

def runway_safety_color(runway_length_ft: float, balanced_field_ft: float) -> str:
    margin = runway_length_ft - balanced_field_ft
    if margin > 500:
        return "green"
    elif margin > 0:
        return "yellow"
    else:
        return "red"

def carrier_takeoff(Vr_mps: float, trim_deg: float, carrier_mode: bool):
    if carrier_mode:
        Vr_mps -= 40 / m2kt  # ~40 kt lower rotate when on the cat
        trim_deg += 2
    return Vr_mps, trim_deg

def recommend_thrust(weight_lbs: float, flap_deg: int, runway_length_ft: float) -> str:
    if runway_length_ft < 5000:
        return "Afterburner"
    elif weight_lbs > 65000 or flap_deg < 20:
        return "Military"
    else:
        return "Minimum Required"

def find_takeoff_speeds(weight_lbs: float, oat_c: float, field_alt_ft: float,
                        flap_deg: int, thrust_mode: str, wind_kt: float = 0.0,
                        slope_percent: float = 0.0, carrier_mode: bool = False):
    """Toy model for Vs/V1/Vr/V2 and distances. For DCS kneeboard convenience only."""
    weight_kg = weight_lbs * lb2kg
    rho = air_density(field_alt_ft, oat_c)
    CLmax = CLmax_by_flap.get(flap_deg, 1.7)

    Vs_mps = v_stall(weight_kg, rho, WING_AREA_M2, CLmax)
    # Headwind reduces reference speeds a bit (toy)
    Vs_mps = max(0.0, Vs_mps - 0.5 * (wind_kt / m2kt))

    Vr_mps = Vs_mps * 1.05
    V2_mps = Vs_mps * 1.20
    V1_mps = Vs_mps * 1.04

    # Trim baseline from weight; tweak for carrier
    trim = 8.0 + (weight_lbs - 50000) / 10000.0
    Vr_mps, trim = carrier_takeoff(Vr_mps, trim, carrier_mode)
    trim = max(5.0, min(12.0, trim))

    # Distances (meters) -> feet, include slope factor
    stop_m = accelerate_stop_distance(V1_mps)
    go_m = accelerate_go_distance(Vr_mps, V2_mps)
    bfl_m = max(stop_m, go_m)
    slope_factor = 1 + (slope_percent / 100.0)

    stop_ft = stop_m * m2ft * slope_factor
    go_ft = go_m * m2ft * slope_factor
    bfl_ft = bfl_m * m2ft * slope_factor

    rpm_target = {"Afterburner": 102, "Military": 96, "Minimum Required": 92}[thrust_mode]

    return {
        "Vs (kt)": round(Vs_mps * m2kt, 1),
        "V1 (kt)": round(V1_mps * m2kt, 1),
        "Vr (kt)": round(Vr_mps * m2kt, 1),
        "V2 (kt)": round(V2_mps * m2kt, 1),
        "Balanced Field Length (ft)": int(round(bfl_ft)),
        "Accel-Go Distance (ft)": int(round(go_ft)),
        "Accel-Stop Distance (ft)": int(round(stop_ft)),
        "Takeoff Trim (° NU)": round(trim, 1),
        "Target Engine RPM (%)": rpm_target,
        "Thrust Mode": thrust_mode
    }

# =========================
#  VR Kneeboard Graphics
# =========================
def draw_flap_indicator(draw: ImageDraw.ImageDraw, flap_name: str, position, size=100):
    x, y = position
    draw.rectangle([x, y, x + size, y + size // 4], outline="black", width=5)
    fill_width = {"Flaps Up": 0.33, "Maneuvering Flaps": 0.66, "Flaps Full": 1.0}[flap_name] * size
    draw.rectangle([x, y, x + fill_width, y + size // 4], fill="blue")
    draw.text((x + size / 2, y + size // 4 + 5), flap_name, fill="black", anchor="ms")

def generate_png_image_vr(results, weight, temp, alt, flap_name, carrier_mode, template="Day"):
    scale = 6  # big for VR; we downscale to 512x768 at the end (DCS kneeboard)
    width, height = 512 * scale, 768 * scale
    colors = {"Day": ("white", "black"), "Night": ("#111111", "#FFFFFF"), "High Contrast": ("#FFFF00", "#000000")}
    bg, fg = colors.get(template, ("white", "black"))

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    try:
        font_bold = ImageFont.truetype("arialbd.ttf", 150)
        font_normal = ImageFont.truetype("arial.ttf", 120)
    except:
        font_bold = font_normal = ImageFont.load_default()

    margin = 30
    draw.text((width // 2, 50), "F-14B TAKEOFF CARD", font=font_bold, fill=fg, anchor="mm")
    draw.rectangle([margin, 100, width - margin, 220], outline=fg, width=6)

    draw.text((margin + 10, 110), f"GW: {int(weight):,} lbs", font=font_normal, fill=fg)
    draw.text((200, 110), f"OAT: {temp} °C", font=font_normal, fill=fg)
    draw.text((380, 110), f"Elev: {int(alt)} ft", font=font_normal, fill=fg)
    draw.text((margin + 10, 170), f"Flaps: {flap_name}", font=font_normal, fill=fg)
    draw.text((300, 170), f"Thrust: {results['Thrust Mode']}", font=font_normal, fill=fg)

    y = 250
    draw_flap_indicator(draw, flap_name, (50, y))
    draw.text(
        (50, y + 120),
        f"Vs {results['Vs (kt)']}  V1 {results['V1 (kt)']}  Vr {results['Vr (kt)']}  V2 {results['V2 (kt)']}",
        font=font_bold,
        fill=fg,
    )

    draw.text((50, 400), f"BFL: {results['Balanced Field Length (ft)']:,} ft", font=font_bold, fill=fg)
    draw.text(
        (50, 460),
        f"Accel-Go: {results['Accel-Go Distance (ft)']:,}  Accel-Stop: {results['Accel-Stop Distance (ft)']:,}",
        font=font_bold,
        fill=fg,
    )
    draw.text(
        (50, 540),
        f"Trim: {results['Takeoff Trim (° NU)']}°  RPM: {results['Target Engine RPM (%)']}%",
        font=font_bold,
        fill=fg,
    )

    if carrier_mode:
        draw.text((50, 600), "CARRIER MODE", font=font_bold, fill="red")

    # Downscale to DCS kneeboard resolution
    img = img.resize((512, 768), resample=Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return out

def generate_pdf_vr(results, weight, temp, alt, flap_name, carrier_mode, template="Day"):
    png_bytes = generate_png_image_vr(results, weight, temp, alt, flap_name, carrier_mode, template)
    img = Image.open(png_bytes)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=portrait((img.width, img.height)))
    c.drawImage(ImageReader(img), 0, 0, width=img.width, height=img.height)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# =========================
#  UI: App
# =========================
st.set_page_config(page_title="F-14B Takeoff Calculator (DCS VR)", layout="centered")
st.title("F-14B Takeoff Calculator (DCS VR)")

# -------------------------
# Load DCS Airfields CSV (robust, blanks allowed)
# -------------------------
dcs_df = pd.read_csv(
    "dcs_airports.csv",
    dtype={"map": "string", "airport_name": "string", "runway_id": "string"},
    keep_default_na=True,
    na_values=["", "NA", "N/A", "null", "None"]
)
for col in ["length_ft", "slope_percent", "le_elev_ft", "he_elev_ft"]:
    if col in dcs_df.columns:
        dcs_df[col] = pd.to_numeric(dcs_df[col], errors="coerce")
    else:
        dcs_df[col] = np.nan

# -------------------------
# Weight entry
# -------------------------
st.subheader("Weight / Balance")
use_gw = st.checkbox("Enter Gross Weight directly?", value=False)
if use_gw:
    weight = st.number_input("Gross Weight (lbs)", 30000, 80000, 72000, step=100)
else:
    fuel_lbs = st.number_input("Fuel Load (lbs)", 0, 20000, 5000, step=100)
    ordnance_lbs = st.number_input("Ordnance Load (lbs)", 0, 20000, 2000, step=100)
    weight = empty_weight_lbs + fuel_lbs + ordnance_lbs
st.caption(f"Computed Gross Weight: **{int(weight):,} lbs**")

# -------------------------
# Atmosphere / Field
# -------------------------
st.subheader("Conditions")
temp = st.number_input("OAT (°C)", -40, 55, 15)
alt = st.number_input("Field Elevation (ft)", 0, 12000, 0)
wind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", -40, 40, 0)
carrier_mode = st.checkbox("Carrier Ops Mode (Catapult)", value=False)

# -------------------------
# DCS Airport → Runway → Intersection
# -------------------------
st.subheader("DCS Airport / Runway")
airport_name = st.selectbox("Airport", sorted(dcs_df["airport_name"].dropna().unique()))
subset = dcs_df[dcs_df["airport_name"] == airport_name].copy()

maps_for_airport = subset["map"].dropna().unique()
selected_map = st.selectbox("Map / Theater", maps_for_airport) if len(maps_for_airport) > 1 else (maps_for_airport[0] if len(maps_for_airport)==1 else "DCS")
if isinstance(selected_map, str):
    subset = subset[subset["map"].fillna("DCS") == selected_map]

runway_id = st.selectbox("Runway", subset["runway_id"].dropna().unique())
row = subset[subset["runway_id"] == runway_id].iloc[0]

stored_length = row.get("length_ft", np.nan)
stored_slope = row.get("slope_percent", np.nan)
le_elev = row.get("le_elev_ft", np.nan)
he_elev = row.get("he_elev_ft", np.nan)

st.markdown("#### Runway Data")
c1, c2 = st.columns(2)
with c1:
    effective_length_base = st.number_input(
        "Runway Length (ft)",
        min_value=1000, max_value=20000,
        value=int(stored_length) if pd.notna(stored_length) else 8000,
        step=50
    )
with c2:
    slope_percent = st.number_input(
        "Runway Slope (%)",
        min_value=-5.0, max_value=5.0, value=float(stored_slope) if pd.notna(stored_slope) else 0.0, step=0.1
    )

intersection_offset_ft = st.number_input(
    "Intersection Offset (ft)",
    min_value=0, max_value=effective_length_base, value=0, step=50
)
effective_runway_ft = max(0, effective_length_base - intersection_offset_ft)
st.caption(f"Effective runway available for takeoff: **{effective_runway_ft:,} ft**")
if effective_runway_ft < 1500:
    st.warning("Effective runway is very short. Verify your intersection offset/runway.")

# Offer a downloadable one-row CSV “fix” if user corrected fields
def _csv_escape(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val)
    if any(ch in s for ch in [",", "\"", "\n"]):
        s = "\"" + s.replace("\"", "\"\"") + "\""
    return s

updated_row = {
    "map": row.get("map", selected_map if isinstance(selected_map, str) else "DCS"),
    "airport_name": row.get("airport_name", airport_name),
    "runway_id": row.get("runway_id", runway_id),
    "length_ft": effective_length_base,
    "slope_percent": slope_percent,
    "le_elev_ft": le_elev if pd.notna(le_elev) else "",
    "he_elev_ft": he_elev if pd.notna(he_elev) else "",
}
line = (
    f"{_csv_escape(updated_row['map'])},"
    f"{_csv_escape(updated_row['airport_name'])},"
    f"{_csv_escape(updated_row['runway_id'])},"
    f"{_csv_escape(updated_row['length_ft'])},"
    f"{_csv_escape(updated_row['slope_percent'])},"
    f"{_csv_escape(updated_row['le_elev_ft'])},"
    f"{_csv_escape(updated_row['he_elev_ft'])}\n"
)
st.markdown("##### Update your CSV")
st.write("If you corrected values above, download this one-row CSV and paste it into your `dcs_airports.csv`:")
st.code("map,airport_name,runway_id,length_ft,slope_percent,le_elev_ft,he_elev_ft\n" + line, language="csv")
st.download_button(
    "⬇️ Download updated CSV row",
    data="map,airport_name,runway_id,length_ft,slope_percent,le_elev_ft,he_elev_ft\n" + line,
    file_name=f"updated_{airport_name}_{runway_id}.csv",
    mime="text/csv"
)

# -------------------------
# Flaps / Thrust
# -------------------------
st.subheader("Configuration")
auto_flap_sel = st.checkbox("Auto-select flap setting?", value=True)
if auto_flap_sel:
    flap_name = auto_flap(weight, effective_runway_ft)
else:
    flap_name = st.selectbox("Flap Setting", list(flap_options.keys()), index=1)
flap_deg = flap_options[flap_name]

thrust_mode = st.selectbox("Thrust Rating", ["Afterburner", "Military", "Minimum Required"], index=1)
template = st.selectbox("Kneeboard Template", ["Day", "Night", "High Contrast"])

auto_thrust = recommend_thrust(weight, flap_deg, effective_runway_ft)
st.info(f"Auto Thrust Recommendation: {auto_thrust}")

# =========================
#  Calculate & Export
# =========================
if st.button("Calculate"):
    res = find_takeoff_speeds(
        weight_lbs=weight,
        oat_c=temp,
        field_alt_ft=alt,
        flap_deg=flap_deg,
        thrust_mode=thrust_mode,
        wind_kt=wind_kt,
        slope_percent=slope_percent,
        carrier_mode=carrier_mode
    )

    st.subheader("Results")
    for k, v in res.items():
        st.write(f"**{k}:** {v}")

    safety_color = runway_safety_color(effective_runway_ft, res["Balanced Field Length (ft)"])
    st.markdown(f"**Runway Safety:** <span style='color:{safety_color}'>{safety_color.upper()}</span>", unsafe_allow_html=True)

    # PNG
    png_bytes = generate_png_image_vr(res, weight, temp, alt, flap_name, carrier_mode, template)
    st.image(png_bytes, caption="VR-Ready Kneeboard (512×768)", use_column_width=True)
    st.download_button(
        "🖼️ Export Takeoff Card (PNG, VR-ready)",
        data=png_bytes,
        file_name=f"F14_Takeoff_{airport_name}_{runway_id}.png",
        mime="image/png"
    )

    # PDF (same graphics as PNG)
    pdf_bytes = generate_pdf_vr(res, weight, temp, alt, flap_name, carrier_mode, template)
    st.download_button(
        "📄 Export Takeoff Card (PDF, VR-ready)",
        data=pdf_bytes,
        file_name=f"F14_Takeoff_{airport_name}_{runway_id}.pdf",
        mime="application/pdf"
    )
