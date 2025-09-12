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
CLmax_by_flap = {0: 1.2, 20: 1.7, 40: 2.1}  # simplified toy

brake_decel_g = 0.35
rotation_time = 2.0
empty_weight_lbs = 40000

flap_options = {"Flaps Full": 40, "Maneuvering Flaps": 20, "Flaps Up": 0}
thrust_levels = ["Minimum Required", "Military", "Afterburner"]

# Thrust effect (toy multipliers)
THRUST_FACTOR = {
    "Minimum Required": 1.10,  # longer go distance
    "Military": 1.00,          # baseline
    "Afterburner": 0.80        # shorter go distance
}

TAILWIND_LIMIT_KT = 10  # advisory limit

# =========================
#  Helpers
# =========================
def auto_flap(weight_lbs: float, runway_length_ft: float) -> str:
    if runway_length_ft < 5000 or weight_lbs > 70000:
        return "Flaps Full"
    elif runway_length_ft < 8000 or weight_lbs > 60000:
        return "Maneuvering Flaps"
    else:
        return "Flaps Up"

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
        Vr_mps -= 40 / m2kt
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
    weight_kg = weight_lbs * lb2kg
    rho = air_density(field_alt_ft, oat_c)
    CLmax = CLmax_by_flap.get(flap_deg, 1.7)

    Vs_mps = v_stall(weight_kg, rho, WING_AREA_M2, CLmax)
    Vs_mps = max(0.0, Vs_mps - 0.5 * (wind_kt / m2kt))  # crude head/tail effect

    Vr_mps = Vs_mps * 1.05
    V2_mps = Vs_mps * 1.20
    V1_mps = Vs_mps * 1.04

    trim = 8.0 + (weight_lbs - 50000) / 10000.0
    Vr_mps, trim = carrier_takeoff(Vr_mps, trim, carrier_mode)
    trim = max(5.0, min(12.0, trim))

    stop_m = accelerate_stop_distance(V1_mps)
    go_m = accelerate_go_distance(Vr_mps, V2_mps) * THRUST_FACTOR[thrust_mode]
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

# -------- Wind math --------
def runway_heads_from_id(runway_id: str):
    """Parse '01/19' into [10, 190] degrees."""
    try:
        parts = runway_id.replace(" ", "").split("/")
        heads = []
        for p in parts:
            digits = "".join(ch for ch in p if ch.isdigit())
            if len(digits) >= 2:
                heads.append((int(digits[:2]) % 36) * 10)
            elif len(digits) == 1:
                heads.append((int(digits) % 36) * 10)
        if len(heads) == 1:
            heads.append((heads[0] + 180) % 360)
        if len(heads) >= 2:
            return [heads[0], heads[1]]
    except:
        pass
    return [0, 180]

def wind_components(runway_heading_deg: float, wind_dir_deg: float, wind_speed_kt: float):
    """
    Returns (headwind_kt, crosswind_kt_abs, cross_from{'left'|'right'}).
    Positive headwind => headwind; negative => tailwind.
    """
    rel = (wind_dir_deg - runway_heading_deg) % 360
    theta = math.radians(rel)
    head = wind_speed_kt * math.cos(theta)
    cross = wind_speed_kt * math.sin(theta)
    cross_from = "right" if cross > 0 else "left"
    return head, abs(cross), cross_from

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
    scale = 6
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
        font=font_bold, fill=fg,
    )
    draw.text((50, 400), f"BFL: {results['Balanced Field Length (ft)']:,} ft", font=font_bold, fill=fg)
    draw.text((50, 460), f"Accel-Go: {results['Accel-Go Distance (ft)']:,}  Accel-Stop: {results['Accel-Stop Distance (ft)']:,}", font=font_bold, fill=fg)
    draw.text((50, 540), f"Trim: {results['Takeoff Trim (° NU)']}°  RPM: {results['Target Engine RPM (%)']}%", font=font_bold, fill=fg)
    if carrier_mode:
        draw.text((50, 600), "CARRIER MODE", font=font_bold, fill="red")

    img = img.resize((512, 768), resample=Image.LANCZOS)
    out = io.BytesIO(); img.save(out, format="PNG"); out.seek(0)
    return out

def generate_pdf_vr(results, weight, temp, alt, flap_name, carrier_mode, template="Day"):
    png_bytes = generate_png_image_vr(results, weight, temp, alt, flap_name, carrier_mode, template)
    img = Image.open(png_bytes)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=portrait((img.width, img.height)))
    c.drawImage(ImageReader(img), 0, 0, width=img.width, height=img.height)
    c.showPage(); c.save(); buf.seek(0)
    return buf

# =========================
#  UI
# =========================
st.set_page_config(page_title="F-14B Takeoff Calculator (DCS VR)", layout="centered")
st.title("F-14B Takeoff Calculator (DCS VR)")

# ---- Load DCS airfields CSV (and optional intersections CSV) ----
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

# Optional: intersection presets (map,airport_name,runway_end,label,offset_ft)
try:
    inter_df = pd.read_csv(
        "dcs_intersections.csv",
        dtype={"map":"string","airport_name":"string","runway_end":"string","label":"string"},
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "None"]
    )
    if "offset_ft" in inter_df.columns:
        inter_df["offset_ft"] = pd.to_numeric(inter_df["offset_ft"], errors="coerce")
    else:
        inter_df["offset_ft"] = np.nan
except Exception:
    inter_df = None

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

st.subheader("Atmosphere / Field")
# Altitude will auto-fill from runway end selection; user may override if desired.
alt_default = st.session_state.get("auto_field_elev_ft", 0)
temp = st.number_input("OAT (°C)", -40, 55, 15)
alt = st.number_input("Field Elevation (ft)", 0, 12000, int(alt_default))

# -------------------------
# Theatre → Airport → Runway → End (auto-populate values)
# -------------------------
st.subheader("DCS Theatre / Airport / Runway")
theatres = sorted(dcs_df["map"].dropna().unique())
selected_map = st.selectbox("Theatre", theatres)

df_map = dcs_df[dcs_df["map"] == selected_map].copy()
airports = sorted(df_map["airport_name"].dropna().unique())
airport_name = st.selectbox("Airport", airports)

df_airport = df_map[df_map["airport_name"] == airport_name].copy()
runways = df_airport["runway_id"].dropna().unique()
runway_id = st.selectbox("Runway (pair)", runways)

# Choose runway end (01 vs 19)
rwy_heads = runway_heads_from_id(runway_id)
end_a, end_b = runway_id.split("/")[0].strip(), runway_id.split("/")[1].strip()
end_choice = st.radio("Runway End (departure)", [end_a, end_b], horizontal=True)
runway_heading = rwy_heads[0] if end_choice == end_a else rwy_heads[1]

# Pull row and compute auto length/slope/elevation from end
row = df_airport[df_airport["runway_id"] == runway_id].iloc[0]
stored_length = row.get("length_ft", np.nan)
stored_slope = row.get("slope_percent", np.nan)
le_elev = row.get("le_elev_ft", np.nan)
he_elev = row.get("he_elev_ft", np.nan)

# Auto-populate: length
base_length_ft_auto = int(stored_length) if not np.isnan(stored_length) else 8000

# Auto-populate: elevation at selected end (if known); else average; else 0
def _auto_elev(end_choice, end_a, end_b, le_elev, he_elev):
    if not np.isnan(le_elev) and not np.isnan(he_elev):
        return int(le_elev) if end_choice == end_a else int(he_elev)
    if not np.isnan(le_elev):
        return int(le_elev)
    if not np.isnan(he_elev):
        return int(he_elev)
    return 0

auto_field_elev_ft = _auto_elev(end_choice, end_a, end_b, le_elev, he_elev)
st.session_state["auto_field_elev_ft"] = auto_field_elev_ft  # drives the default above

# Auto-populate: directional slope (% up in takeoff direction)
def _dir_slope_percent(end_choice, end_a, end_b, le_elev, he_elev, length_ft, fallback):
    if not np.isnan(le_elev) and not np.isnan(he_elev) and length_ft and length_ft > 0:
        # If departing end_a -> toward end_b; positive if destination end higher (uphill takeoff)
        if end_choice == end_a:
            return ((he_elev - le_elev) / length_ft) * 100.0
        else:
            return ((le_elev - he_elev) / length_ft) * 100.0
    # fall back to stored slope (directionless)
    return float(fallback) if not np.isnan(fallback) else 0.0

slope_percent_auto = _dir_slope_percent(end_choice, end_a, end_b, le_elev, he_elev, base_length_ft_auto, stored_slope)

# Show the auto-populated values (user can still override later if they wish)
st.caption(f"Auto runway length: **{base_length_ft_auto:,} ft**  |  Auto slope (takeoff dir): **{slope_percent_auto:.1f}%**  |  Auto field elev: **{auto_field_elev_ft} ft**")

# Sync the Field Elevation input default to auto-pop elevation (only when it changes selection)
if "last_end_key" not in st.session_state or st.session_state["last_end_key"] != (selected_map, airport_name, runway_id, end_choice):
    st.session_state["last_end_key"] = (selected_map, airport_name, runway_id, end_choice)
    # push auto field elevation into the current alt number_input default
    st.session_state["auto_field_elev_ft"] = auto_field_elev_ft

# -------------------------
# Wind Input (+ components)
# -------------------------
st.subheader("Wind / Briefing")
wind_mode = st.radio("Headwind input", ["Manual (head/tailwind in kt)", "From DCS Mission Briefing"], index=0)

if wind_mode == "Manual (head/tailwind in kt)":
    wind_kt = st.number_input("Headwind (+) / Tailwind (-) (kt)", -40, 40, 0)
    crosswind_kt = 0.0
    cross_from = "n/a"
else:
    c1, c2 = st.columns(2)
    with c1:
        wind_dir_deg = st.number_input("Wind Direction (° FROM, briefing)", 0, 359, 0)
    with c2:
        wind_speed_kt = st.number_input("Wind Speed (kt)", 0, 100, 0)
    head, cross, cross_from = wind_components(runway_heading, wind_dir_deg, wind_speed_kt)
    wind_kt = round(head, 1)
    crosswind_kt = round(cross, 1)

max_crosswind_limit = st.number_input("Max crosswind advisory limit (kt)", 0, 50, 20)

if wind_mode == "From DCS Mission Briefing":
    if wind_kt >= 0:
        st.info(f"Headwind component: **+{wind_kt} kt**  |  Crosswind: **{crosswind_kt} kt from {cross_from}**")
    else:
        st.warning(f"Tailwind component: **{wind_kt} kt**  |  Crosswind: **{crosswind_kt} kt from {cross_from}**")
if wind_kt < 0 and abs(wind_kt) > TAILWIND_LIMIT_KT:
    st.error(f"⚠ MAX TAILWIND EXCEEDED: {abs(wind_kt)} kt (limit {TAILWIND_LIMIT_KT} kt).")
if wind_mode == "From DCS Mission Briefing" and crosswind_kt > max_crosswind_limit:
    st.error(f"⚠ MAX CROSSWIND EXCEEDED: {crosswind_kt} kt (limit {max_crosswind_limit} kt).")

carrier_mode = st.checkbox("Carrier Ops Mode (Catapult)", value=False)

# -------------------------
# Intersection Takeoff (with presets)
# -------------------------
st.subheader("Intersection Takeoff")
base_length_ft = base_length_ft_auto
slope_percent_val = float(slope_percent_auto)

consider_intersection = st.checkbox("Consider intersection takeoff?", value=False)
intersection_mode = "None"
manual_offset_ft = 0
auto_margin_ft = 0
preset_offset_ft = 0

if consider_intersection:
    st.caption("Offset is measured from the **selected departure threshold**.")
    # Presets (optional CSV dcs_intersections.csv)
    if inter_df is not None:
        presets = inter_df[
            (inter_df["map"] == selected_map) &
            (inter_df["airport_name"] == airport_name) &
            (inter_df["runway_end"].str.strip() == end_choice)
        ].copy()
        if not presets.empty:
            labels = ["(none)"] + [f"{r['label']}  ({int(r['offset_ft'])} ft)" for _, r in presets.iterrows()]
            pick = st.selectbox("Preset intersection", labels, index=0)
            if pick != "(none)":
                # parse offset from label
                idx = labels.index(pick) - 1
                preset_offset_ft = int(presets.iloc[idx]["offset_ft"] if not np.isnan(presets.iloc[idx]["offset_ft"]) else 0)
        else:
            st.caption("No presets found for this runway end.")
    else:
        st.caption("Add optional `dcs_intersections.csv` for presets (see format below).")

    intersection_mode = st.radio("Intersection mode", ["Manual offset", "Auto (most restrictive)"], index=1)

    if intersection_mode == "Manual offset":
        manual_offset_ft = st.number_input("Manual intersection offset (ft)",
            min_value=0, max_value=base_length_ft, value=preset_offset_ft, step=50
        )
    else:
        auto_margin_ft = st.number_input("Auto margin (ft) — extra safety beyond BFL",
            min_value=0, max_value=2000, value=0, step=50
        )

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
    # First flap guess from full runway (we re-evaluate after effective length)
    flap_name = auto_flap(weight, base_length_ft) if auto_flap_sel else "Maneuvering Flaps"
    flap_deg = flap_options[flap_name]

    # Initial performance with user's thrust; no intersection yet
    res_initial = find_takeoff_speeds(
        weight_lbs=weight, oat_c=temp, field_alt_ft=alt,
        flap_deg=flap_deg, thrust_mode=thrust_mode_user,
        wind_kt=wind_kt, slope_percent=slope_percent_val, carrier_mode=carrier_mode
    )
    bfl_needed_user = res_initial["Balanced Field Length (ft)"]

    # Determine effective runway
    if consider_intersection:
        if intersection_mode == "Manual offset":
            effective_runway_ft = max(0, base_length_ft - manual_offset_ft)
        else:
            required_len = max(0, bfl_needed_user + auto_margin_ft)
            effective_runway_ft = max(0, min(base_length_ft, required_len))
            manual_offset_ft = max(0, base_length_ft - effective_runway_ft)
    else:
        effective_runway_ft = base_length_ft
        manual_offset_ft = 0

    # Recompute flaps using effective runway if auto
    if auto_flap_sel:
        flap_name = auto_flap(weight, effective_runway_ft)
        flap_deg = flap_options[flap_name]

    # Thrust override path
    thrust_final = thrust_mode_user
    thrust_overridden = False

    def meets_bfl(thrust_mode_chk: str):
        res_chk = find_takeoff_speeds(
            weight_lbs=weight, oat_c=temp, field_alt_ft=alt,
            flap_deg=flap_deg, thrust_mode=thrust_mode_chk,
            wind_kt=wind_kt, slope_percent=slope_percent_val, carrier_mode=carrier_mode
        )
        return res_chk["Balanced Field Length (ft)"] <= effective_runway_ft, res_chk

    ok, res_try = meets_bfl(thrust_final)
    if not ok:
        for lvl in thrust_levels[thrust_levels.index(thrust_mode_user):] + thrust_levels:
            ok2, res_try2 = meets_bfl(lvl)
            if ok2:
                if lvl != thrust_mode_user:
                    thrust_overridden = True
                thrust_final = lvl
                res_try = res_try2
                break

    res = res_try

    # Display
    st.subheader("Results")
    st.write(f"**Theatre:** {selected_map}   **Airport:** {airport_name}")
    st.write(f"**Runway:** {runway_id}   **Departure end:** {end_choice}   (Heading {runway_heading}°)")
    st.write(f"**Runway length (base):** {base_length_ft:,} ft   **Effective runway:** {effective_runway_ft:,} ft")
    if consider_intersection:
        st.write(f"**Intersection mode:** {intersection_mode}   **Offset:** {manual_offset_ft:,} ft   **Margin (auto):** {auto_margin_ft if intersection_mode!='Manual offset' else 0} ft")
    st.write(f"**Slope (takeoff dir):** {slope_percent_val:.1f}%   **Field elevation (auto):** {auto_field_elev_ft} ft")

    # Wind components summary
    if wind_mode == "From DCS Mission Briefing":
        head_str = f"+{wind_kt:.1f} kt headwind" if wind_kt >= 0 else f"{wind_kt:.1f} kt tailwind"
        cross_str = f"{crosswind_kt:.1f} kt crosswind from {cross_from}"
        st.write(f"**Wind components:** {head_str}  |  {cross_str}")
    else:
        head_str = f"{wind_kt:+.1f} kt head/tailwind (manual)"
        st.write(f"**Wind component used:** {head_str}")

    # Thrust with override banner
    if thrust_overridden:
        st.markdown(
            f"**Thrust (your selection):** <span style='color:#B00020;'>{thrust_mode_user}</span>  →  "
            f"**OVERRIDDEN to:** <span style='color:#00B050;'>{thrust_final}</span>",
            unsafe_allow_html=True
        )
    else:
        st.write(f"**Thrust:** {thrust_final}")

    st.write(f"**Flaps:** {flap_name}")

    for k, v in res.items():
        st.write(f"**{k}:** {v}")

    safety_color = runway_safety_color(effective_runway_ft, res["Balanced Field Length (ft)"])
    st.markdown(f"**Runway Safety:** <span style='color:{safety_color}'>{safety_color.upper()}</span>", unsafe_allow_html=True)

    # Tailwind & crosswind advisories
    if wind_kt < 0 and abs(wind_kt) > TAILWIND_LIMIT_KT:
        st.error(f"⚠ MAX TAILWIND EXCEEDED: {abs(wind_kt)} kt (limit {TAILWIND_LIMIT_KT} kt)")
    if wind_mode == "From DCS Mission Briefing" and crosswind_kt > max_crosswind_limit:
        st.error(f"⚠ MAX CROSSWIND EXCEEDED: {crosswind_kt} kt (limit {max_crosswind_limit} kt)")

    # Exports
    png_bytes = generate_png_image_vr(res, weight, temp, alt, flap_name, carrier_mode, template)
    st.image(png_bytes, caption="VR-Ready Kneeboard (512×768)", use_column_width=True)
    st.download_button(
        "🖼️ Export Takeoff Card (PNG, VR-ready)",
        data=png_bytes,
        file_name=f"F14_Takeoff_{selected_map}_{airport_name}_{runway_id}_{end_choice}.png",
        mime="image/png"
    )
    pdf_bytes = generate_pdf_vr(res, weight, temp, alt, flap_name, carrier_mode, template)
    st.download_button(
        "📄 Export Takeoff Card (PDF, VR-ready)",
        data=pdf_bytes,
        file_name=f"F14_Takeoff_{selected_map}_{airport_name}_{runway_id}_{end_choice}.pdf",
        mime="application/pdf"
    )
