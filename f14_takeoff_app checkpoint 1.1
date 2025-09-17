# ============================================================
# F-14B Performance Calculator for DCS World — UI-first build
# File: f14_takeoff_app.py
# Version: v1.1.0 (2025-09-16)
#
# Purpose: Full UI skeleton (no performance math). Implements your approved
# design: Aircraft → Runway → Environment → Weight & Balance → Takeoff Config
# → Climb Profile → Landing Setup → Results. Mobile-friendly, debounced.
#
# ============================================================
# 🚨 Bogged Down Protocol 🚨
# If development chat becomes slow or confusing:
# 1. STOP — Do not keep patching endlessly.
# 2. REVERT — Roll back to last saved checkpoint (Git tag vX.Y.Z).
# 3. RESET — Start a new chat if needed, say "continue from vX.Y.Z".
# 4. SCOPE — Focus on one module/card at a time.
# 5. SAVE — Commit working versions often with clear tags.
# ============================================================

from __future__ import annotations
import re
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st

# =========================
# Page + global settings
# =========================
st.set_page_config(
    page_title="F-14B Performance — DCS (UI)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lightweight debounce for text areas/inputs
st.session_state.setdefault("_ui_seq", 0)

def _bump_seq():
    st.session_state["_ui_seq"] += 1

# =========================
# Data loading (CSV only)
# =========================
@st.cache_data(show_spinner=False)
def load_airports(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    for col in ("length_ft", "tora_ft", "toda_ft", "asda_ft", "threshold_elev_ft", "heading_deg", "slope_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_perf(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    for col in ("gw_lbs", "flap_deg", "Vs_kt", "V1_kt", "Vr_kt", "V2_kt", "press_alt_ft", "oat_c"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

AIRPORTS_PATHS = [
    "dcs_airports.csv",
    "https://raw.githubusercontent.com/Zorinco96/f14_takeoff_app.py/main/dcs_airports.csv",
]
PERF_PATHS = [
    "f14_perf.csv",
    "https://raw.githubusercontent.com/Zorinco96/f14_takeoff_app.py/main/f14_perf.csv",
]

airports = None
for p in AIRPORTS_PATHS:
    try:
        airports = load_airports(p)
        break
    except Exception:
        continue
if airports is None:
    st.error("Could not load dcs_airports.csv. Ensure it exists locally or in GitHub.")
    st.stop()

perf = None
for p in PERF_PATHS:
    try:
        perf = load_perf(p)
        break
    except Exception:
        continue
if perf is None:
    st.warning("f14_perf.csv not found. UI will still function with placeholders.")

# =========================
# Helpers (unit detect, parsing)
# =========================
FT_PER_NM = 6076.11549
ISA_LAPSE_C_PER_1000FT = 1.98


def detect_length_unit(text: str) -> Tuple[Optional[float], str]:
    """Return (length_ft, detected_unit_str). Accepts inputs like '8500', '1.2 nm', '1.2nm'.
    Heuristic: if suffix nm present → NM. If numeric ≤ 5 with no suffix → likely NM. Else → feet.
    """
    if text is None:
        return None, ""
    s = text.strip().lower()
    if not s:
        return None, ""

    nm_match = re.search(r"([0-9]*\.?[0-9]+)\s*(nm|nmi)", s)
    if nm_match:
        nm = float(nm_match.group(1))
        return nm * FT_PER_NM, "NM (auto)"

    num_match = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not num_match:
        return None, ""
    val = float(num_match.group(1))

    # If small value with no suffix, assume NM; else feet
    if val <= 5:
        return val * FT_PER_NM, "NM (heuristic)"
    return val, "ft (auto)"


def detect_pressure(qnh_text: str) -> Tuple[Optional[float], str]:
    """Parse QNH in inHg or hPa. Return (inHg, label). Default display is inHg."""
    if not qnh_text:
        return None, ""
    s = qnh_text.strip().lower()
    hpa_match = re.search(r"([0-9]{3,4})\s*(hpa|mb)", s)
    inhg_match = re.search(r"([0-9]*\.?[0-9]+)\s*(inhg|hg)", s)
    num_match = re.search(r"([0-9]*\.?[0-9]+)", s)

    if hpa_match:
        hpa = float(hpa_match.group(1))
        inhg = hpa * 0.0295299830714
        return inhg, "hPa → inHg"
    if inhg_match:
        return float(inhg_match.group(1)), "inHg"
    if num_match:
        val = float(num_match.group(1))
        # Heuristic: values > 28 and < 32 likely inHg; values ~1000/1013 likely hPa
        if 900 <= val <= 1100:
            return val * 0.0295299830714, "hPa (heuristic) → inHg"
        return val, "inHg (assumed)"
    return None, ""


def parse_wind(text: str) -> Dict[str, Any]:
    """Parse wind like '270/15', '270@7m/s', '270/20kt'. Returns degrees_true, speed_kts, unit_label."""
    if not text:
        return {"dir_deg": None, "spd_kts": None, "unit": ""}
    s = text.strip().lower()
    m = re.search(r"(\d{2,3})\s*[/@]??\s*([0-9]*\.?[0-9]+)\s*(m/s|ms|kt|kts)?", s)
    if not m:
        return {"dir_deg": None, "spd_kts": None, "unit": ""}
    deg = int(m.group(1))
    val = float(m.group(2))
    unit = (m.group(3) or "kt").replace("ms", "m/s")
    spd_kts = val * 1.94384 if unit in ("m/s",) else val
    return {"dir_deg": deg, "spd_kts": spd_kts, "unit": "m/s→kt" if unit == "m/s" else "kt"}


def temp_at_elevation(temp_sl_c: Optional[float], elev_ft: float, lapse_c_per_1000ft: float = ISA_LAPSE_C_PER_1000FT) -> Optional[float]:
    if temp_sl_c is None:
        return None
    return float(temp_sl_c - lapse_c_per_1000ft * (elev_ft / 1000.0))


def hw_xw_components(wind_dir: Optional[int], wind_kts: Optional[float], rwy_heading_deg: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (head/tailwind + = head, -, tail in kts; crosswind in kts)."""
    import math
    if None in (wind_dir, wind_kts, rwy_heading_deg):
        return None, None
    # Convert to wind relative angle (from dir blowing toward)
    angle = math.radians((wind_dir - rwy_heading_deg) % 360)
    hw = wind_kts * math.cos(angle)
    xw = wind_kts * math.sin(angle)
    return hw, abs(xw)

# =========================
# Sidebar: status + presets
# =========================
with st.sidebar:
    st.title("F-14B Perf — DCS")
    st.caption("UI skeleton • v1.1.0 (no math yet)")

    st.subheader("Quick Presets")
    preset = st.selectbox(
        "Load preset",
        ["—", "Clean • 56k • TO 20°", "Heavy • 72k • TO 20°", "Recovery • 68k • LDG 35°"],
        index=0,
    )
    if preset != "—":
        st.info(f"Preset loaded: {preset}")

    st.subheader("Flags")
    auto_recompute = st.toggle("Auto-recompute", value=True)
    show_debug = st.toggle("Show scenario JSON", value=True)

st.markdown(
    """
    <div style="position:sticky;top:0;background:var(--background-color);padding:0.4rem 0;z-index:5;border-bottom:1px solid rgba(255,255,255,0.1)">
        <strong>F-14 Performance — DCS World</strong>
        <span style="opacity:0.7"> • UI-only v1.1.0 • Auto-recompute ON</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Section 1 — Aircraft
# =========================
with st.expander("1) Aircraft", expanded=True):
    ac = st.selectbox("Aircraft", ["F-14B (Tomcat)", "F-14A (future)", "F/A-18C (future)"], index=0)
    st.caption("Selecting the airframe sets defaults for flaps/thrust and W&B stations.")

# =========================
# Section 2 — Runway (picker + manual override)
# =========================
with st.expander("2) Runway", expanded=True):
    # Picker
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        maps = sorted(airports["map"].dropna().unique().tolist())
        map_sel = st.selectbox("Map", maps, key="rw_map")
        sub = airports[airports["map"] == map_sel]
        search = st.text_input("Search airport", placeholder="Type part of the name…")
        if search:
            sub = sub[sub["airport_name"].str.contains(search, case=False, na=False)]
        apt = st.selectbox("Airport", sorted(sub["airport_name"].dropna().unique().tolist()), key="rw_airport")
    with c2:
        rwy_rows = sub[sub["airport_name"] == apt]
        ends = rwy_rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        rwy_end = st.selectbox("Runway End / Intersection", sorted(ends) if ends else ["Full Length"], key="rw_end")
        # Available distance (TORA if present else length)
        tora_series = rwy_rows.loc[rwy_rows["runway_end"].astype(str) == str(rwy_end), "tora_ft"] if "runway_end" in rwy_rows.columns else pd.Series()
        tora = float(tora_series.max() if not tora_series.empty else rwy_rows.get("length_ft", pd.Series([0.0])).max())
        elev = float(rwy_rows.get("threshold_elev_ft", pd.Series([0.0])).max())
        hdg = float(rwy_rows.get("heading_deg", pd.Series([0.0])).max())
        slope = float(rwy_rows.get("slope_pct", pd.Series([0.0])).max())
        st.metric("TORA (ft)", f"{tora:.0f}")
        st.metric("Elev (ft)", f"{elev:.0f}")
        st.metric("Heading (°T)", f"{hdg:.0f}")
        st.metric("Slope (%)", f"{slope:.1f}")
    with c3:
        st.checkbox("Manual runway entry", value=False, key="rw_manual")
        if st.session_state["rw_manual"]:
            mr_len = st.text_input("Runway length (ft or NM)", placeholder="8500 or 1.4 NM")
            len_ft, unit_label = detect_length_unit(mr_len)
            st.caption(f"Detected: {unit_label or '—'} → {f'{len_ft:.0f} ft' if len_ft else ''}")
            mr_elev = st.number_input("Elevation (ft)", value=elev or 0.0, step=50.0)
            mr_hdg = st.number_input("Heading (°T)", value=hdg or 0.0, step=1.0)
            mr_slope = st.number_input("Slope (%)", value=slope or 0.0, step=0.1)
            mr_tora = st.number_input("TORA (ft)", value=float(len_ft or tora or 0.0), step=100.0)
            # Override visual
            if len_ft:
                tora = float(len_ft)
            elev, hdg, slope = mr_elev, mr_hdg, mr_slope
            st.info("Manual values override database for calculations.")

# =========================
# Section 3 — Environment (paste parser + manual)
# =========================
with st.expander("3) Environment", expanded=True):
    mode_env = st.radio("Input mode", ["Paste from DCS briefing", "Manual"], horizontal=True)

    if mode_env == "Paste from DCS briefing":
        blob = st.text_area("Paste briefing text", height=160, placeholder="Paste the DCS weather section here…")
        if blob:
            # Very light stub parse to illustrate UX
            temp_m = re.search(r"temp[^\d-]*(-?\d+)", blob, flags=re.I)
            qnh_m = re.search(r"qnh[^\d]*(\d{3,4}|\d+\.?\d*)", blob, flags=re.I)
            wind_m = re.search(r"(\d{2,3})\s*[/@]\s*(\d+\.?\d*)\s*(kt|kts|m/s)?", blob, flags=re.I)
            temp_sl = float(temp_m.group(1)) if temp_m else None
            qnh_text = qnh_m.group(1) + (" hPa" if (qnh_m and len(qnh_m.group(1)) >= 3) else "") if qnh_m else ""
            qnh_inhg, qnh_label = detect_pressure(qnh_text)
            wind = parse_wind(wind_m.group(0)) if wind_m else {"dir_deg": None, "spd_kts": None, "unit": ""}
            st.success("Parsed tokens:")
            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("Sea-level Temp (°C)", f"{temp_sl if temp_sl is not None else '—'}")
            with cB:
                st.metric("QNH (inHg)", f"{qnh_inhg:.2f}" if qnh_inhg else "—")
                st.caption(qnh_label)
            with cC:
                st.metric("Wind", f"{wind['dir_deg'] or '—'}/{wind['spd_kts']:.0f} kt" if wind["spd_kts"] else "—")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            temp_sl = st.number_input("Sea-level Temp (°C)", value=15.0, step=1.0)
            qnh_text = st.text_input("QNH (inHg or hPa)", value="29.92")
            qnh_inhg, qnh_label = detect_pressure(qnh_text)
            st.caption(f"Detected: {qnh_label}")
        with c2:
            wind_text = st.text_input("Wind (deg/speed)", value="270/10 kt", help="e.g. 090/7m/s or 270/10 kt")
            w = parse_wind(wind_text)
            st.caption(f"Detected: {w['unit']}")
        with c3:
            # Compute field temp from runway elevation
            field_temp = temp_at_elevation(temp_sl, elev or 0.0)
            st.metric("Field Temp est. (°C)", f"{field_temp:.1f}" if field_temp is not None else "—")
            hw, xw = hw_xw_components(w.get("dir_deg"), w.get("spd_kts"), hdg)
            st.metric("Head/Tailwind (kt)", f"{hw:+.0f}" if hw is not None else "—")
            st.metric("Crosswind (kt)", f"{xw:.0f}" if xw is not None else "—")

# =========================
# Section 4 — Weight & Balance
# =========================
with st.expander("4) Weight & Balance", expanded=True):
    wb_mode = st.radio("Mode", ["Simple (enter GTOW)", "Detailed (DCS-style loadout)"])

    if wb_mode.startswith("Simple"):
        gw_tow = st.number_input("Gross Takeoff Weight (lb)", value=70000.0, step=500.0)
        gw_ldg_plan = st.number_input("Planned Landing Weight (lb)", value=56000.0, step=500.0)
        st.caption("You can switch to Detailed mode to build weight via stations + fuel.")
    else:
        # DCS-like: top silhouette area (placeholder), station grid, fuel, running totals
        st.markdown("**Loadout (F-14B)** — select stores per station. This mimics the DCS Rearm/Refuel layout.")
        stores_categories = {
            "AIR-TO-AIR": ["AIM-9M", "AIM-7MH", "AIM-54C"],
            "BOMBS": ["Mk-82", "Mk-83", "GBU-12"],
            "ROCKETS": ["ZUNI LAU-10"],
            "FUEL TANKS": ["Drop Tank 267 gal"],
            "PODS": ["LANTIRN"],
        }
        # Minimal station schema (mirrors Heatblur naming for 1A/1B and 8A/8B)
        stations = [
            "1A", "1B", "2", "3", "4", "5", "6", "7", "8A", "8B"
        ]
        # Build UI grid
        cols = st.columns(5)
        sel: Dict[str, Dict[str, Any]] = {}
        for i, sta in enumerate(stations):
            with cols[i % 5]:
                st.write(f"**STA {sta}**")
                cat = st.selectbox(
                    f"Category {sta}", ["—"] + list(stores_categories.keys()), key=f"cat_{sta}"
                )
                item = "—"
                if cat != "—":
                    item = st.selectbox(
                        f"Store {sta}", ["—"] + stores_categories[cat], key=f"store_{sta}"
                    )
                qty = st.number_input(f"Qty {sta}", value=0, min_value=0, max_value=2, step=1, key=f"qty_{sta}")
                remove_pylon = st.checkbox("Remove pylon", value=False, key=f"pylon_{sta}")
                sel[sta] = {"category": cat, "store": item, "qty": qty, "remove_pylon": remove_pylon}

        st.markdown("**Fuel**")
        c1, c2, c3 = st.columns(3)
        with c1:
            fuel_int = st.slider("Internal Fuel (%)", min_value=0, max_value=100, value=80)
        with c2:
            fuel_ext_l = st.slider("Left Tank (%)", min_value=0, max_value=100, value=0)
        with c3:
            fuel_ext_r = st.slider("Right Tank (%)", min_value=0, max_value=100, value=0)

        # Running totals (placeholders; real math later)
        st.markdown("### Totals (mock)")
        t1, t2, t3 = st.columns(3)
        t1.metric("Gross Weight (lb)", "70,500")
        t2.metric("CG (%MAC)", "23.5")
        t3.metric("Trim (units)", "+2.0")
        st.caption("Weight/CG are placeholders. Real W&B math to be wired later.")

# =========================
# Section 5 — Takeoff Configuration
# =========================
with st.expander("5) Takeoff Configuration", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        flaps = st.radio("Flaps", ["AUTO", "UP", "MANEUVER", "FULL"], horizontal=True)
    with c2:
        thrust = st.radio("Thrust", ["AUTO", "MILITARY", "AFTERBURNER", "Manual derate"], horizontal=False)
    with c3:
        derate = 0
        if thrust == "Manual derate":
            derate = st.slider("Derate (RPM %)", min_value=70, max_value=100, value=95)
        st.metric("Req. climb grad (AEO)", "≥ 300 ft/NM")
    st.caption("AUTO thrust will target 14 CFR 121.189 and ≥300 ft/NM AEO using the minimum required setting (to be modeled).")

# =========================
# Section 6 — Climb Profile (updated spec)
# =========================
with st.expander("6) Climb Profile", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        cruise_alt = st.number_input("Cruise Altitude (MSL ft)", value=28000.0, step=1000.0)
    with c2:
        climb_profile = st.selectbox("Profile", ["Minimum time to altitude", "Most efficient climb"], index=0)
    with c3:
        ignore_reg = st.checkbox("Ignore regulatory speed restrictions (≤250 KIAS <10k)")

    st.markdown("### Results (mock)")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Time to 10k", "02:40")
    r2.metric("Time to TOC", "07:50")
    r3.metric("Fuel to TOC", "2,100 lb")
    r4.metric("TOC Distance", "37 NM")
    st.caption("Speed schedules and numbers are placeholders; real climb model will populate here.")

# =========================
# Section 7 — Landing Setup
# =========================
with st.expander("7) Landing Setup", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        dest_map = st.selectbox("Destination Map", maps, key="ldg_map")
        dest_sub = airports[airports["map"] == dest_map]
        dest_apt = st.selectbox("Destination Airport", sorted(dest_sub["airport_name"].dropna().unique().tolist()), key="ldg_airport")
    with c2:
        dest_rows = dest_sub[dest_sub["airport_name"] == dest_apt]
        dest_end = st.selectbox("Runway End", sorted(dest_rows.get("runway_end", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) or ["Full Length"], key="ldg_end")
        dest_tora_series = dest_rows.loc[dest_rows["runway_end"].astype(str) == str(dest_end), "tora_ft"] if "runway_end" in dest_rows.columns else pd.Series()
        dest_tora = float(dest_tora_series.max() if not dest_tora_series.empty else dest_rows.get("length_ft", pd.Series([0.0])).max())
        st.metric("TORA (ft)", f"{dest_tora:.0f}")
    with c3:
        plan_ldw = st.number_input("Planned Landing Weight (lb)", value=56000.0, step=500.0)
        cond = st.radio("Runway condition", ["DRY", "WET"], horizontal=True)
        st.caption("121.195 factors will apply (to be modeled).")

# =========================
# RESULTS — sticky panel with sub-tabs
# =========================
st.markdown("---")
with st.container():
    st.subheader("Results (all placeholder values)")
    tabs = st.tabs(["Takeoff", "Climb", "Landing"])  # results presentation only

    with tabs[0]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("V1 (kt)", "145")
        m2.metric("Vr (kt)", "150")
        m3.metric("V2 (kt)", "160")
        m4.metric("Vfs (kt)", "180")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("RDR (ft)", "8,200")
        d2.metric("RDA (ft)", f"{tora:.0f}")
        d3.metric("Accel-Go (ft)", "9,100")
        d4.metric("Accel-Stop (ft)", "8,600")

        dcs = st.columns(2)
        dcs[0].metric("DCS distance to Vr (ft)", "3,400")
        dcs[1].metric("DCS liftoff +35 ft (ft)", "5,100")
        st.warning("Most restrictive: ASDR > TORA (mock)")

    with tabs[1]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vy (kt)", "220")
        c2.metric("Time 10k", "02:40")
        c3.metric("Time TOC", "07:50")
        c4.metric("Fuel TOC", "2,100 lb")
        st.info("Regulatory: Compliant (≤250 KIAS below 10k)")

    with tabs[2]:
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("Vs (kt)", "121")
        l2.metric("Vref (kt)", "157")
        l3.metric("Vapp (kt)", "165")
        l4.metric("Va/c (kt)", "167")
        l5.metric("Vfs (kt)", "177")
        st.metric("LDR 50 ft (ft)", "4,200")
        st.metric("Max Landing Wt (lb)", "58,400")

    # Sticky footer actions
    st.markdown("---")
    a, b, c = st.columns(3)
    with a:
        st.button("Copy V-speed line", help="Copies V1/Vr/V2 and Vref/Vapp to clipboard (stub)")
    with b:
        st.button("Generate Kneeboard PDF", help="Stub — will export a print-friendly report")
    with c:
        st.button("Share Permalink", help="Stub — will encode scenario as URL")

# =========================
# Scenario JSON (debug)
# =========================
if show_debug:
    scenario = {
        "aircraft": ac,
        "runway": {
            "map": map_sel,
            "airport": apt,
            "end": rwy_end,
            "tora_ft": tora,
            "elev_ft": elev,
            "heading_deg": hdg,
            "slope_pct": slope,
            "manual_override": bool(st.session_state.get("rw_manual")),
        },
        "environment": {
            "temp_sl_c": temp_sl if 'temp_sl' in locals() else None,
            "qnh_inhg": qnh_inhg if 'qnh_inhg' in locals() else None,
            "wind": (w if 'w' in locals() else {}),
            "field_temp_c": (temp_at_elevation(temp_sl, elev) if 'temp_sl' in locals() else None),
        },
        "wb": {
            "mode": wb_mode,
            "simple": {"gtow_lb": (gw_tow if 'gw_tow' in locals() else None), "ldw_lb": (gw_ldg_plan if 'gw_ldg_plan' in locals() else None)},
            "detailed": st.session_state,  # contains per-station selections (keys cat_*, store_*, qty_*)
        },
        "takeoff_config": {"flaps": flaps, "thrust": thrust, "derate_rpm": (derate if 'derate' in locals() else None)},
        "climb": {"cruise_alt_ft": cruise_alt, "profile": climb_profile, "ignore_reg": ignore_reg},
        "landing": {"dest_map": dest_map, "dest_airport": dest_apt, "dest_end": dest_end, "dest_tora_ft": dest_tora, "plan_ldw_lb": plan_ldw, "cond": cond},
    }
    st.markdown("### Scenario JSON (debug)")
    st.code(json.dumps(scenario, indent=2))

st.caption("UI-only baseline v1.1.0. Once you sign off on UX, we will move logic into f14_takeoff_core.py and wire real models.")
