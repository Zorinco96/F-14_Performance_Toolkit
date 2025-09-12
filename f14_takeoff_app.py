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
CLmax_by_flap = {0: 1.2, 20: 1.7, 40: 2.1}  # toy CLmax table

brake_decel_g = 0.35
rotation_time = 2.0
empty_weight_lbs = 40000

flap_options = {"Flaps Full": 40, "Maneuvering Flaps": 20, "Flaps Up": 0}
thrust_levels = ["Minimum Required", "Military", "Afterburner"]

# Thrust effect on accelerate-go / BFL (toy multipliers)
THRUST_FACTOR = {
    "Minimum Required": 1.10,  # longer go distance
    "Military": 1.00,          # baseline
    "Afterburner": 0.80        # shorter go distance
}

def auto_flap(weight_lbs: float, runway_length_ft: float) -> str:
    if runway_length_ft < 5000 or weight_lbs > 70000:
        return "Flaps Full"
    elif runway_length_ft < 8000 or weight_lbs > 60000:
        return "Maneuvering Flaps"
    else:
        return "Flaps Up"

# =========================
#  Performance Math (Toy model for DCS kneeboard convenience)
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
        Vr_mps -= 40 / m2kt  # lower rotate when on cat
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
    """
    Toy model for Vs/V1/Vr/V2 & distances; accelerate-go scaled by THRUST_FACTOR.
    """
    weight_kg = weight_lbs * lb2kg
    rho = air_density(field_alt_ft, oat_c)
    CLmax = CLmax_by_flap.get(flap_deg, 1.7)

    Vs_mps = v_stall(weight_kg, rho, WING_AREA_M2, CLmax)
    Vs_mps = max(0.0, Vs_mps - 0.5 * (wind_kt / m2kt))  # simple head/tailwind effect

    Vr_mps = Vs_mps * 1.05
    V2_mps = Vs_mps * 1.20
    V1_mps = Vs_mps * 1.04

    trim = 8.0 + (weight_lbs - 50000) / 10000.0
    Vr_mps, trim = carrier_takeoff(Vr_mps, trim, carrier_mode)
    trim = max(5.0, min(12.0, trim))

    stop_m = accelerate_stop_distance(V1_mps)
    go_m = accelerate_go_distance(Vr_mps, V2_mps) * THRUST_FACTOR[thrust_mode]  # thrust effect here
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
#  Helpers: Headwind from briefing + runway heading
# =========================
def runway_heads_from_id(runway_id: str):
    """
    Parse runway_id like '13/31' into headings [130, 310] deg.
    Falls back to [0, 180] if parsing fails.
    """
    try:
        parts = runway_id.replace(" ", "").split("/")
        heads = []
        for p in parts:
            num = ""
            for ch in p:
                if ch.isdigit():
                    num += ch
            if len(num) >= 2:
                heads.append((int(num[:2]) % 36) * 10)
            elif len(num) == 1:
                heads.append((int(num) % 36) * 10)
        if len(heads) == 1:
            heads.append((heads[0] + 180) % 360)
        if len(heads) >= 2:
            return [heads[0], heads[1]]
    except:
        pass
    return [0, 180]

def headwind_component(runway_heading_deg: float, wind_dir_deg: float, wind_speed_kt: float) -> float:
    """
    Positive = headwind, negative = tailwind. Angles in degrees.
    """
    # angle between wind direction (from which it blows) and runway heading
    theta = math.radians((wind_dir_deg - runway_heading_deg) % 360)
    # Headwind component (wind is FROM direction; cos positive => headwind)
    return wind_speed_kt * math.cos(theta)

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
    scale = 6  # draw large, downscale to 512x768 for DCS kneeboard
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
#  UI
# =========================
st.set_page_config(page_title="F-14B Takeoff Calculator (DCS VR)", layout="centered")
st.title("F-14B Takeoff Calculator (DCS VR)")

# Load DCS airfields CSV with blanks allowed
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
# Weight / Conditions
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

st.subheader("Conditions")
temp = st.number_input("OAT (°C)", -40, 55, 15)
alt = st.number_input("Field Elevation (ft)", 0, 12000, 0)

# -------------------------
# Theatre → Airport → Runway
# -------------------------
st.subheader("DCS Theatre / Airport / Runway")
theatres = sorted(dcs_df["map"].dropna().unique())
selected_map = st.selectbox("Theatre", theatres)

df_map = dcs_df[dcs_df["map"] == selected_map].copy()
airports = sorted(df_map["airport_name"].dropna().unique())
airport_name = st.selectbox("Airport", airports)

df_airport = df_map[df_map["airport_name"] == airport_name].copy()
runways = df_airport["runway_id"].dropna().unique()
runway_id = st.selectbox("Runway", runways)

# Choose runway end for headwind calc
rwy_heads = runway_heads_from_id(runway_id)
rwy_end = st.radio("Runway End (departure)", [f"{runway_id.split('/')[0]} ({rwy_heads[0]}°)", f"{runway_id.split('/')[1]} ({rwy_heads[1]}°)"], horizontal=True)
runway_heading = rwy_heads[0] if rwy_end.startswith(runway_id.split('/')[0]) else rwy_heads[1]

row = df_airport[df_airport["runway_id"] == runway_id].iloc[0]
stored_length = row.get("length_ft", np.nan)
stored_slope = row.get("slope_percent", np.nan)

# -------------------------
# Wind Input Mode
# -------------------------
st.subheader("Wind / Briefing")
wind_mode = st.radio("Headwind input", ["Manual (head/tailwind in kt)", "From DCS Mission Briefing"], index=0)

if wind_mode == "Manual (head/tailwind in kt)":
    wind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", -40, 40, 0)
else:
    colw1, colw2 = st.columns(2)
    with colw1:
        wind_dir_deg = st.number_input("Wind Direction (° FROM, briefing)", 0, 359, 0)
    with colw2:
        wind_speed_kt = st.number_input("Wind Speed (kt)", 0, 100, 0)
    # Compute headwind component from briefing
    wind_kt = round(headwind_component(runway_heading, wind_dir_deg, wind_speed_kt), 1)
    if wind_kt >= 0:
        st.info(f"Computed headwind component: **+{wind_kt} kt** (Runway {rwy_end.split()[0]})")
    else:
        st.warning(f"Computed tailwind component: **{wind_kt} kt** (Runway {rwy_end.split()[0]})")

carrier_mode = st.checkbox("Carrier Ops Mode (Catapult)", value=False)

# -------------------------
# Intersection Takeoff Options
# -------------------------
st.subheader("Intersection Takeoff")
# Base runway values (ask if missing)
base_length_ft = int(stored_length) if not np.isnan(stored_length) else st.number_input(
    "Runway Length missing — enter value (ft):", min_value=1000, max_value=20000, value=8000, step=50
)
slope_percent_val = float(stored_slope) if not np.isnan(stored_slope) else st.number_input(
    "Runway Slope missing — enter value (%):", min_value=-5.0, max_value=5.0, value=0.0, step=0.1
)

consider_intersection = st.checkbox("Consider intersection takeoff?", value=False)
intersection_mode = "None"
manual_offset_ft = 0
auto_margin_ft = 0

if consider_intersection:
    intersection_mode = st.radio("Intersection mode", ["Manual offset", "Auto (most restrictive)"], index=1)
    if intersection_mode == "Manual offset":
        manual_offset_ft = st.number_input("Manual intersection offset (ft)", min_value=0, max_value=base_length_ft, value=0, step=50)
    else:
        auto_margin_ft = st.number_input("Auto margin (ft) — extra safety beyond BFL", min_value=0, max_value=2000, value=0, step=50)

# -------------------------
# Flaps / Thrust
# -------------------------
st.subheader("Configuration")
auto_flap_sel = st.checkbox("Auto-select flap setting?", value=True)
thrust_mode_user = st.selectbox("Thrust Rating (your selection)", thrust_levels, index=1)
template = st.selectbox("Kneeboard Template", ["Day", "Night", "High Contrast"])

# =========================
#  Calculate & Export
# =========================
if st.button("Calculate"):
    # First pass: choose flaps based on base runway (will re-evaluate after intersection effective length)
    flap_name_preview = auto_flap(weight, base_length_ft) if auto_flap_sel else None
    flap_name = flap_name_preview if auto_flap_sel else st.session_state.get("flap_name_manual", None)
    if not flap_name:
        # If not auto and no previous manual, default Maneuvering
        flap_name = "Maneuvering Flaps"
    flap_deg = flap_options[flap_name]

    # Initial performance with user's thrust to get BFL (no intersection yet)
    res_initial = find_takeoff_speeds(
        weight_lbs=weight, oat_c=temp, field_alt_ft=alt,
        flap_deg=flap_deg, thrust_mode=thrust_mode_user,
        wind_kt=wind_kt, slope_percent=slope_percent_val, carrier_mode=carrier_mode
    )
    bfl_needed_user = res_initial["Balanced Field Length (ft)"]

    # Determine intersection
    if consider_intersection:
        if intersection_mode == "Manual offset":
            effective_runway_ft = max(0, base_length_ft - manual_offset_ft)
        else:
            # AUTO: tightest effective runway that still meets BFL (with user's thrust initially)
            required_len = max(0, bfl_needed_user + auto_margin_ft)
            effective_runway_ft = max(0, min(base_length_ft, required_len))
            manual_offset_ft = max(0, base_length_ft - effective_runway_ft)
    else:
        effective_runway_ft = base_length_ft
        manual_offset_ft = 0

    # Re-select flaps with effective runway (if auto)
    if auto_flap_sel:
        flap_name = auto_flap(weight, effective_runway_ft)
        flap_deg = flap_options[flap_name]

    # Now ensure thrust is sufficient for effective runway: try user, else auto-upgrade
    thrust_final = thrust_mode_user
    thrust_overridden = False

    # Function to see if given thrust meets BFL within available runway
    def meets_bfl(thrust_mode_chk: str) -> bool:
        res_chk = find_takeoff_speeds(
            weight_lbs=weight, oat_c=temp, field_alt_ft=alt,
            flap_deg=flap_deg, thrust_mode=thrust_mode_chk,
            wind_kt=wind_kt, slope_percent=slope_percent_val, carrier_mode=carrier_mode
        )
        return res_chk["Balanced Field Length (ft)"] <= effective_runway_ft, res_chk

    ok, res_try = meets_bfl(thrust_final)
    if not ok:
        # Attempt minimal auto bump to next levels
        for lvl in thrust_levels[thrust_levels.index(thrust_mode_user):] + thrust_levels:
            ok2, res_try2 = meets_bfl(lvl)
            if ok2:
                if lvl != thrust_mode_user:
                    thrust_overridden = True
                thrust_final = lvl
                res_try = res_try2
                break

    res = res_try  # final results with adequate thrust

    # Display summary
    st.subheader("Results")
    st.write(f"**Theatre:** {selected_map}   **Airport:** {airport_name}   **Runway:** {runway_id}   **Departure end:** {rwy_end}")
    st.write(f"**Runway length (base):** {base_length_ft:,} ft   **Effective runway:** {effective_runway_ft:,} ft")
    if consider_intersection:
        st.write(f"**Intersection mode:** {intersection_mode}   **Offset:** {manual_offset_ft:,} ft   **Margin (auto):** {auto_margin_ft if intersection_mode!='Manual offset' else 0} ft")
    st.write(f"**Slope:** {slope_percent_val:.1f}%   **Wind component used:** {wind_kt:+.1f} kt")

    # Thrust line with override highlight
    if thrust_overridden:
        st.markdown(f"**Thrust (your selection):** <span style='color:#B00020;'>{thrust_mode_user}</span>  →  "
                    f"**OVERRIDDEN to:** <span style='color:#00B050;'>{thrust_final}</span>",
                    unsafe_allow_html=True)
    else:
        st.write(f"**Thrust:** {thrust_final}")

    st.write(f"**Flaps:** {flap_name}")

    for k, v in res.items():
        st.write(f"**{k}:** {v}")

    safety_color = runway_safety_color(effective_runway_ft, res["Balanced Field Length (ft)"])
    st.markdown(f"**Runway Safety:** <span style='color:{safety_color}'>{safety_color.upper()}</span>", unsafe_allow_html=True)

    # PNG kneeboard
    png_bytes = generate_png_image_vr(res, weight, temp, alt, flap_name, carrier_mode, template)
    st.image(png_bytes, caption="VR-Ready Kneeboard (512×768)", use_column_width=True)
    st.download_button(
        "🖼️ Export Takeoff Card (PNG, VR-ready)",
        data=png_bytes,
        file_name=f"F14_Takeoff_{selected_map}_{airport_name}_{runway_id}.png",
        mime="image/png"
    )

    # PDF kneeboard
    pdf_bytes = generate_pdf_vr(res, weight, temp, alt, flap_name, carrier_mode, template)
    st.download_button(
        "📄 Export Takeoff Card (PDF, VR-ready)",
        data=pdf_bytes,
        file_name=f"F14_Takeoff_{selected_map}_{airport_name}_{runway_id}.pdf",
        mime="application/pdf"
    )
