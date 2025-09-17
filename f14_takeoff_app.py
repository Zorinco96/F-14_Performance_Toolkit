# f14_takeoff_app.py — Streamlit UI for DCS F-14B Performance (UI-first, math stubs)
# v1.1.3-ui-wb1
#
# This file targets the v1.1.2 UI spec and applies the requested changes:
# 1) W&B Simple defaults: GTOW=74,349 lb (MTOW), LDW=54,500 lb
# 2) W&B Detailed: Fuel unit defaults to Pounds, AND Pounds listed first
# 3) W&B Stations: Selecting a store auto-sets quantity; no qty widget
# 4) Takeoff Results: Clean presentation (cards + tidy tables + positive-only line chart)
#
# UI scope preserved from v1.1.2 target:
# - Presets (curated F-14B; flaps/thrust AUTO)
# - Global runway search (all maps); slope removed
# - Environment: paste/manual with unit auto-detect; ISA lapse helper; wind components
# - W&B Detailed: DCS-style station tiles; external tanks FULL/EMPTY; Compatibility Mode (beta);
#   import standard loadouts (dropdown) and .miz upload (stubs)
# - Takeoff Config: vertical radios; thrust last option "DERATE (Manual)"
# - Climb: "Most efficient climb" default; overlay profiles with placeholder schedules
# - Landing: Scenarios A/B/C per spec
# - Results: separate Takeoff/Climb/Landing sections; plain language; Stabilizer Trim; rotate chart is a line graph
#
# Math is NOT wired yet. Numbers shown are placeholders; wiring will land in subsequent commits.

from __future__ import annotations

import io
import json
import math
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# ------------------------------
# Constants & simple defaults
# ------------------------------
APP_VERSION = "v1.1.3-ui-wb1"

# F-14B fixed limits / UI defaults
MTOW_LB = 74349            # Requested default for GTOW (Simple W&B)
DEFAULT_LDW_LB = 54500     # Requested default for LDW  (Simple W&B)
DEFAULT_FUEL_UNIT = "lb"   # Requested Detailed W&B fuel unit default
SEA_LEVEL_STD_C = 15.0
ISA_LAPSE_C_PER_FT = 1.98 / 1000.0  # ~1.98°C per 1000 ft

# Thrust & flaps options (UI-only)
TAKEOFF_FLAP_OPTIONS = ["AUTO", "10°", "20°", "FULL"]
TAKEOFF_THRUST_OPTIONS = ["AUTO", "MIL", "MAX", "DERATE (Manual)"]  # vertical radios, DERATE last

# Simple curated presets (stores/fuel only; flaps/thrust AUTO)
PRESETS = {
    "Clean / 40% fuel": {"stores": "Clean", "fuel_pct": 40},
    "4xAIM-7 / 2xAIM-9 / 50% fuel": {"stores": "Light A2A", "fuel_pct": 50},
    "2xTanks / 6xAIM-54 / 60% fuel": {"stores": "Heavy A2A", "fuel_pct": 60},
    "TARPS / 55% fuel": {"stores": "TARPS", "fuel_pct": 55},
}

# Minimal store catalog (UI behavior only — quantities are implicit, no qty inputs)
# mass_each_lb are placeholder values; to be replaced during math wiring/calibration.
STORE_CATALOG: Dict[str, Dict] = {
    "Empty": {"qty": 0, "mass_each_lb": 0},
    "AIM-9": {"qty": 1, "mass_each_lb": 188},
    "AIM-7": {"qty": 1, "mass_each_lb": 510},
    "AIM-54": {"qty": 1, "mass_each_lb": 1000},
    "Fuel Tank 267gal (FULL)": {"qty": 1, "mass_each_lb": 2000},
    "Fuel Tank 267gal (EMPTY)": {"qty": 1, "mass_each_lb": 400},
    "TARPS": {"qty": 1, "mass_each_lb": 1000},
}

# Example per-station allowed stores (UI only). To be replaced with true compatibility rules later.
STATIONS_ALLOWED = {
    "Station 1": ["Empty", "AIM-9", "AIM-7"],
    "Station 2": ["Empty", "AIM-9", "AIM-7"],
    "Station 3": ["Empty", "AIM-54", "Fuel Tank 267gal (FULL)", "Fuel Tank 267gal (EMPTY)"],
    "Station 4": ["Empty", "AIM-54", "Fuel Tank 267gal (FULL)", "Fuel Tank 267gal (EMPTY)"],
    "Station 5": ["Empty", "TARPS"],
    "Station 6": ["Empty", "AIM-9"],
    "Station 7": ["Empty", "AIM-7"],
    "Station 8": ["Empty", "AIM-9"],
}

# Optional station-level fixed qty override (if a station mounts a dual rack etc.)
STATION_QTY_OVERRIDE: Dict[str, int] = {
    # e.g., "Station 3": 1
}

# ------------------------------
# Helpers (UI utility)
# ------------------------------
def parse_environment_paste(text: str) -> Dict:
    """
    Very forgiving parser for paste like:
    "OAT 15C, QNH 29.92, wind 180/12, elev 2200 ft, RWY 18"
    """
    if not text:
        return {}
    t = text.lower()

    # Temperature C
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*c", t)
    oat_c = float(m.group(1)) if m else None

    # QNH / Altimeter (inHg)
    m = re.search(r"(?:qnh|alt|altimeter)\s*(\d+(?:\.\d+)?)", t)
    qnh_inhg = float(m.group(1)) if m else None

    # Wind ddd/ff
    m = re.search(r"wind\s*(\d{1,3})\s*/\s*(\d+(?:\.\d+)?)", t)
    wind_dir = int(m.group(1)) if m else None
    wind_spd = float(m.group(2)) if m else None

    # Field elevation
    m = re.search(r"(?:elev|elevation)\s*(\d+(?:\.\d+)?)\s*ft", t)
    elev_ft = float(m.group(1)) if m else None

    # Runway (RWY 18)
    m = re.search(r"(?:rwy|runway)\s*(\d{1,2}[LRC]?)", t)
    rwy = m.group(1).upper() if m else None

    return {
        "oat_c": oat_c,
        "qnh_inhg": qnh_inhg,
        "wind_dir": wind_dir,
        "wind_spd": wind_spd,
        "elev_ft": elev_ft,
        "rwy": rwy,
    }


def isa_temp_at_field(elev_ft: float | None) -> float | None:
    if elev_ft is None:
        return None
    return SEA_LEVEL_STD_C - ISA_LAPSE_C_PER_FT * elev_ft


def wind_components(rwy_heading_deg: float | None, wind_dir_deg: float | None, wind_spd_kt: float | None) -> Tuple[float, float]:
    """Return (head/tailwind + = head, - = tail, crosswind magnitude) in kt."""
    if None in (rwy_heading_deg, wind_dir_deg, wind_spd_kt):
        return (0.0, 0.0)
    # Angle between runway and wind
    ang = math.radians((wind_dir_deg - rwy_heading_deg) % 360)
    head = wind_spd_kt * math.cos(ang)
    cross = abs(wind_spd_kt * math.sin(ang))
    return (round(head, 1), round(cross, 1))


@lru_cache(maxsize=1)
def load_airports_df() -> pd.DataFrame:
    """
    Try to load airports globally:
    - local repo "dcs_airports_expanded.csv"
    - bundled /mnt/data/dcs_airports.csv (if present in this environment)
    Falls back to a tiny sample.
    """
    candidates = ["dcs_airports_expanded.csv", "/mnt/data/dcs_airports.csv"]
    for path in candidates:
        try:
            df = pd.read_csv(path)
            # Expected columns (best-effort): ident, name, map, rwy, rwy_heading, elev_ft
            # Normalize common variants if found
            cols = {c.lower(): c for c in df.columns}
            # Make standard names where possible
            df = df.rename(
                columns={
                    cols.get("ident", "ident"): "ident",
                    cols.get("name", "name"): "name",
                    cols.get("map", "map"): "map",
                    cols.get("rwy", "rwy"): "rwy",
                    cols.get("runway", "rwy"): "rwy",
                    cols.get("rwy_heading", "rwy_heading"): "rwy_heading",
                    cols.get("heading", "rwy_heading"): "rwy_heading",
                    cols.get("elev_ft", "elev_ft"): "elev_ft",
                    cols.get("elevation_ft", "elev_ft"): "elev_ft",
                }
            )
            return df
        except Exception:
            continue
    # Fallback minimal dataset
    return pd.DataFrame(
        [
            {"ident": "KVUO", "name": "Pearson Field", "map": "Caucasus", "rwy": "08/26", "rwy_heading": 80, "elev_ft": 24},
            {"ident": "HENDERSON", "name": "Henderson 35L/17R", "map": "Nevada", "rwy": "35L/17R", "rwy_heading": 350, "elev_ft": 2450},
        ]
    )


def search_airports(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q:
        return df.head(50).copy()
    ql = q.lower()
    mask = (
        df["ident"].astype(str).str.lower().str.contains(ql, na=False)
        | df["name"].astype(str).str.lower().str.contains(ql, na=False)
        | df["map"].astype(str).str.lower().str.contains(ql, na=False)
        | df.get("rwy", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(ql, na=False)
    )
    return df[mask].head(100).copy()


# ------------------------------
# Page config & Header
# ------------------------------
st.set_page_config(page_title="DCS F-14B Performance (UI-first)", layout="wide")
st.title("DCS World — F-14 Performance Calculator (F-14B first)")
st.caption(f"UI-first scaffold • {APP_VERSION} • Next: math wiring (W&B totals/CG/trim → takeoff → climb → landing)")

# ------------------------------
# Sidebar: Presets, Global airport search, Environment, Calibration
# ------------------------------
with st.sidebar:
    st.subheader("Quick Presets")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    st.caption("Presets affect stores/fuel only; flaps/thrust stay AUTO per v1.1.2.")

    st.subheader("Runway (Global Search)")
    df_airports = load_airports_df()
    q = st.text_input("Search airports/maps/runways", placeholder="e.g., Henderson 35L, Nevada, Kobuleti, Syria…")
    results = search_airports(df_airports, q)
    st.dataframe(results[["ident", "name", "map", "rwy"]], use_container_width=True, height=200)

    # Select one entry
    rwy_idx = st.number_input("Select row # from results", min_value=0, max_value=max(len(results) - 1, 0), value=0, step=1)
    sel = results.iloc[int(rwy_idx)] if len(results) > 0 else None
    if sel is not None:
        st.success(f"Selected: {sel['ident']} • {sel['name']} • RWY: {sel.get('rwy','')}")

    st.subheader("Environment")
    env_mode = st.radio("Entry mode", ["Paste", "Manual"], horizontal=True)
    pasted = {}
    if env_mode == "Paste":
        txt = st.text_area("Paste (e.g. 'OAT 15C, QNH 29.92, wind 180/12, elev 2200 ft, RWY 18')", height=80)
        pasted = parse_environment_paste(txt)

    # Manual inputs with auto-detect defaults from paste
    oat_c = st.number_input("OAT (°C)", value=float(pasted.get("oat_c")) if pasted.get("oat_c") is not None else 15.0, step=1.0)
    qnh_inhg = st.number_input("Altimeter/QNH (inHg)", value=float(pasted.get("qnh_inhg")) if pasted.get("qnh_inhg") is not None else 29.92, step=0.01, format="%.2f")
    field_elev_ft = st.number_input("Field Elevation (ft)", value=float(pasted.get("elev_ft")) if pasted.get("elev_ft") is not None else (float(sel["elev_ft"]) if sel is not None and pd.notnull(sel.get("elev_ft")) else 0.0), step=10.0)
    isa_field = isa_temp_at_field(field_elev_ft)
    st.caption(f"ISA @ field ≈ {isa_field:.1f}°C" if isa_field is not None else "ISA @ field: —")

    wind_dir = st.number_input("Wind Direction (° true)", value=float(pasted.get("wind_dir")) if pasted.get("wind_dir") is not None else 0.0, step=1.0, min_value=0.0, max_value=359.0)
    wind_spd = st.number_input("Wind Speed (kt)", value=float(pasted.get("wind_spd")) if pasted.get("wind_spd") is not None else 0.0, step=1.0, min_value=0.0)

    # Runway heading estimate
    rwy_heading = float(sel["rwy_heading"]) if (sel is not None and pd.notnull(sel.get("rwy_heading"))) else None
    if pasted.get("rwy"):
        try:
            # Simple mapping: RWY 18 => 180
            rwy_heading = (int(re.sub(r"[^0-9]", "", pasted["rwy"])) % 36) * 10
        except Exception:
            pass

    headwind_kt, crosswind_kt = wind_components(rwy_heading, wind_dir, wind_spd)
    st.caption(f"Wind components ≈ Head/Tail: {headwind_kt:+.1f} kt, Crosswind: {crosswind_kt:.1f} kt")

    st.subheader("Calibration Mode")
    cal_mode = st.radio("Calibration", ["DCS", "FAA-like"], horizontal=True)

# ------------------------------
# Weight & Balance — Simple
# ------------------------------
st.header("Weight & Balance")

s1, s2, s3 = st.columns([1, 1, 2])

with s1:
    st.subheader("Simple")
    # (1) Requested defaults here:
    gtow_lb = st.number_input("Gross Takeoff Weight (lb)", min_value=0, max_value=120000, value=MTOW_LB, step=50)
    ldw_lb = st.number_input("Landing Weight (lb)", min_value=0, max_value=120000, value=DEFAULT_LDW_LB, step=50)
    st.caption("Defaults set per F-14B MTOW and typical LDW placeholder. Adjust as needed.")

with s2:
    # Quick presets reflect flaps/thrust AUTO; only stores/fuel vary
    st.subheader("Quick Presets")
    chosen = PRESETS[preset_name]
    st.write(f"Preset: **{preset_name}**")
    st.write(f"Stores: **{chosen['stores']}**, Fuel: **{chosen['fuel_pct']}%**")
    st.button("Apply Preset (UI only)", help="UI confirmation only; math wiring in later commit")

with s3:
    st.subheader("Takeoff Configuration")
    # Vertical radio groups as requested
    flaps = st.radio("Flaps", TAKEOFF_FLAP_OPTIONS, index=0, horizontal=False)
    thrust = st.radio("Thrust", TAKEOFF_THRUST_OPTIONS, index=0, horizontal=False)
    st.caption("Thrust last option is 'DERATE (Manual)'; radios are vertical per spec.")

# ------------------------------
# Weight & Balance — Detailed
# ------------------------------
st.subheader("Detailed (DCS-style stations)")

# (2) Fuel unit default/order: Pounds first and defaulted to Pounds
fuel_unit = st.radio("Fuel entry unit", options=["lb", "%"], index=0, horizontal=True)
if fuel_unit == "lb":
    fuel_lb_input = st.number_input("Fuel (lb)", min_value=0, max_value=200000, value=12000, step=100)
    fuel_pct_input = None
else:
    fuel_pct_input = st.number_input("Fuel (%)", min_value=0, max_value=100, value=40, step=1)
    fuel_lb_input = None

cmod, import_col, miz_col = st.columns([1, 1, 1])
with cmod:
    compat_mode = st.checkbox("Compatibility Mode (beta)", value=False,
                              help="Filters stations/stores to likely-valid pairs (placeholder). True rules coming later.")
with import_col:
    std_loadout = st.selectbox("Import standard loadouts (stub)", ["—", "Clean", "Light A2A", "Heavy A2A", "TARPS"])
with miz_col:
    miz = st.file_uploader(".miz file upload (stub)", type=["miz"])

st.caption("Stations auto-derive quantity from store selection; no separate qty picker per your request.")

def station_tile(station_name: str, allowed: List[str], default_store: str = "Empty") -> Dict:
    st.caption(station_name)
    # Optionally filter if Compatibility Mode is on (placeholder behavior)
    options = allowed if not compat_mode else [x for x in allowed if x != "TARPS" or station_name == "Station 5"]
    if default_store not in options:
        options = [default_store] + [x for x in options if x != default_store]
    store = st.selectbox(f"{station_name} store", options, key=f"{station_name}_store")

    # (3) Auto quantity behavior
    qty = STATION_QTY_OVERRIDE.get(station_name, STORE_CATALOG.get(store, {}).get("qty", 0))
    mass_each = STORE_CATALOG.get(store, {}).get("mass_each_lb", 0)
    total_mass = qty * mass_each

    with st.container(border=True):
        st.write(f"Selected: **{store}**")
        st.write(f"Quantity: **{qty}** (auto)")
        st.write(f"Est. mass: **{total_mass:,} lb**")

    return {"station": station_name, "store": store, "qty": qty, "mass_each_lb": mass_each, "total_mass_lb": total_mass}

g1, g2, g3, g4 = st.columns(4)
with g1:
    s1 = station_tile("Station 1", STATIONS_ALLOWED["Station 1"])
    s2 = station_tile("Station 2", STATIONS_ALLOWED["Station 2"])
with g2:
    s3 = station_tile("Station 3", STATIONS_ALLOWED["Station 3"])
    s4 = station_tile("Station 4", STATIONS_ALLOWED["Station 4"])
with g3:
    s5 = station_tile("Station 5", STATIONS_ALLOWED["Station 5"])
    s6 = station_tile("Station 6", STATIONS_ALLOWED["Station 6"])
with g4:
    s7 = station_tile("Station 7", STATIONS_ALLOWED["Station 7"])
    s8 = station_tile("Station 8", STATIONS_ALLOWED["Station 8"])

detailed_stations = [s1, s2, s3, s4, s5, s6, s7, s8]

# External tanks landing assumption: EMPTY
st.info("Landing assumes external tanks **EMPTY** per your spec.")

# ------------------------------
# Climb Profile (placeholders & overlays)
# ------------------------------
st.header("Climb Profile")

climb_mode = st.radio("Profile", ["Most efficient climb", "Minimum time climb"], index=0, horizontal=True)

# Placeholder schedules shown as cards (RPM/FF & IAS/Mach by segment)
c1, c2 = st.columns(2)
with c1:
    st.write("**Engine Schedule (placeholder)**")
    st.table(pd.DataFrame({
        "Segment": ["TO", "Climb 1", "Climb 2", "Transition"],
        "RPM%":    ["AUTO", "95", "92", "AUTO"],
        "FF pph":  ["—", "—", "—", "—"],
    }))
with c2:
    st.write("**Airspeed/Mach Schedule (placeholder)**")
    st.table(pd.DataFrame({
        "Segment": ["TO", "Climb 1", "Climb 2", "High"],
        "IAS/Mach": ["250 KIAS", "300 KIAS", "0.72 M", "0.78 M"],
    }))

# Overlay chart (placeholder curves)
x_alt = list(range(0, 32000, 2000))
eff_spd = [240 + (i * 0.5) for i in range(len(x_alt))]
min_spd = [260 + (i * 0.7) for i in range(len(x_alt))]
df_climb = pd.DataFrame({"Altitude (ft)": x_alt, "Most efficient": eff_spd, "Min time": min_spd})
df_melt = df_climb.melt("Altitude (ft)", var_name="Profile", value_name="Speed")

chart = (
    alt.Chart(df_melt)
    .mark_line()
    .encode(
        x=alt.X("Altitude (ft):Q"),
        y=alt.Y("Speed:Q"),
        color="Profile:N",
        tooltip=["Profile", "Altitude (ft)", "Speed"],
    )
    .properties(height=280)
)
st.altair_chart(chart, use_container_width=True)

# ------------------------------
# Landing Setup
# ------------------------------
st.header("Landing Setup")

st.write("Choose a scenario:")
landing_scn = st.radio(
    "Scenario",
    [
        "A: 3,000 lb fuel + stores retained",
        "B: 3,000 lb fuel + weapons expended (pods/tanks kept)",
        "C: Custom fuel & stores",
    ],
    index=0,
)

if landing_scn.startswith("C"):
    st.number_input("Custom Landing Fuel (lb)", min_value=0, max_value=200000, value=3000, step=100)
    st.caption("External tanks treated **EMPTY** at landing.")

# ------------------------------
# RESULTS
# ------------------------------

# TAKEOFF
st.header("Results — Takeoff")

# (4) Clean presentation: Cards + tidy subtables + positive-only line chart
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Vr (kts)", value="—")
with m2:
    st.metric("V1 (kts)", value="—")
with m3:
    st.metric("V2 (kts)", value="—")
with m4:
    st.metric("Stabilizer Trim (units)", value="—")

cA, cB, cC = st.columns(3)

with cA:
    st.write("**All-engines**")
    st.table(pd.DataFrame({
        "Metric": ["Ground roll", "Liftoff to 35 ft"],
        "Distance (ft)": ["—", "—"],
    }))

with cB:
    st.write("**Accelerate-go / Reject**")
    st.table(pd.DataFrame({
        "Metric": ["ASDR", "RDR"],
        "Distance (ft)": ["—", "—"],
    }))

with cC:
    st.write("**Wind Components**")
    st.table(pd.DataFrame({
        "Metric": ["Head/Tailwind", "Crosswind"],
        "Value": [f"{headwind_kt:+.1f} kt", f"{crosswind_kt:.1f} kt"],
    }))

st.markdown("### Rotate Distance Trend")
# Positive-only line chart using placeholder series
rot_knots = [120, 130, 140, 150, 160]
rot_ft = [500, 800, 1200, 1600, 1900]
rot_df = pd.DataFrame({"IAS (kts)": rot_knots, "Rotate Distance (ft)": rot_ft})
st.line_chart(rot_df.set_index("IAS (kts)"))

# CLIMB
st.header("Results — Climb")
st.table(pd.DataFrame({
    "Profile": ["Most efficient", "Minimum time"],
    "Top of Climb (ft)": ["—", "—"],
    "Time to TOC": ["—", "—"],
    "Fuel to TOC (lb)": ["—", "—"],
}))

# LANDING
st.header("Results — Landing")
st.table(pd.DataFrame({
    "Metric": ["Vref", "Vapp", "Vac", "Vfs", "LDR (ft)", "Max LDW (lb)"],
    "Value": ["—", "—", "—", "—", "—", f"{DEFAULT_LDW_LB:,} (placeholder)"],
}))

# Footer / About
st.divider()
st.caption(
    "UI-first scaffold. Math wiring (W&B totals + CG + trim → takeoff speeds/distances → climb model → landing) "
    "will be added in small, tagged commits."
)
