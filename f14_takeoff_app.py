# streamlit_app.py
# DCS F-14B Takeoff Performance Calculator (Heatblur) — Streamlit
# NOTE: This is an engineering aid for a flight sim, not real-world guidance.
# Focus areas: clean UI, transparent assumptions, pluggable performance model,
# V2-centric trim target, and reduced-thrust lower limit of 90% RPM.

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "F-14B Takeoff Performance — DCS (Heatblur)"

# --------------------------- Constants & Baseline Assumptions ---------------------------
FT2_TO_M2 = 0.092903
KT_TO_FPS = 1.68781
LBS_TO_N = 4.44822
RHO_SEA_LEVEL_SI = 1.225  # kg/m^3
G = 9.80665

# F-14 geometry (simplified for takeoff conf)
# Heatblur models variable geometry; for a forward-sweep takeoff we approximate:
WING_AREA_FT2 = 565.0  # forward sweep reference (approx)
WING_AREA_M2 = WING_AREA_FT2 * FT2_TO_M2

# Aerodynamic assumptions (tunable):
# CLmax at takeoff config ("Maneuver" flaps ~15-20° equivalent). Range 1.8–2.2.
CLMAX_TO = 2.00
# V2 factor over Vs. Transport-category style heuristic (sim context):
V2_FACTOR = 1.13
# VR target factor vs V2 (slightly below/near V2 for rotation initiation):
VR_FACTOR = 0.97  # rotate to arrive near V2 at lift-off
# V1 target factor vs VR (placeholder — in real calc depends on BFL & accelerate-stop):
V1_FACTOR = 0.94

# Engine / thrust lapse (very simplified):
# F110-GE-400 (F-14B). We treat thrust as a gross MIL/AB number with ISA lapse + OAT corrections.
# Numbers are placeholders tuned for sim-plausibility, NOT NATOPS.
T_MIL_SL_LBF = 2 * 16200  # ~16.2k lbf/engine MIL at SL (ballpark)
T_AB_SL_LBF = 2 * 30000   # ~30k lbf/engine AB at SL (ballpark)

# Reduced-thrust lower bound (per user ask): 90% RPM equivalent guard.
REDUCED_THRUST_FLOOR = 0.90  # i.e., do not allow below 90% of the selected regime

# Rolling friction & runway effects (coarse placeholders):
MU_DRY = 0.03
MU_WET = 0.05
SLOPE_GRAD_PER_UNIT = 0.01  # +1% slope increases required field length ~1% (coarse)
HEADWIND_FIELD_EFFECT = -0.01  # -1% field length per kt headwind per 10 kts (coarse)


# --------------------------- Utility Functions ---------------------------
def isa_density(altitude_ft: float, oat_c: float) -> float:
    """Return air density (kg/m^3) using a simple ISA + delta-T model."""
    # ISA temperature at altitude (°C)
    alt_m = altitude_ft * 0.3048
    T0 = 288.15
    L = 0.0065
    p0 = 101325.0
    R = 287.058
    g = 9.80665
    T_isa = T0 - L * alt_m
    T = T_isa + (oat_c - (T_isa - 273.15))  # apply OAT delta vs ISA
    # Baro pressure under ISA lapse (ignoring OAT delta on pressure for simplicity)
    p = p0 * (T_isa / T0) ** (g / (R * L))
    rho = p / (R * T)
    return rho


def stall_speed_ias_kt(gw_lbs: float, rho: float, cl_max: float = CLMAX_TO) -> float:
    """Compute Vs (IAS, kt) using lift = weight at stall. Simplified compressibility ignored."""
    W_N = gw_lbs * LBS_TO_N
    Vs_mps = math.sqrt((2 * W_N) / (rho * WING_AREA_M2 * cl_max))
    Vs_kt = Vs_mps / 0.514444
    return Vs_kt


def thrust_available_lbf(mode: str, rho: float, derate: float) -> float:
    """Crude thrust lapse: linear with density ratio. Apply derate w/90% floor."""
    rho_ratio = rho / RHO_SEA_LEVEL_SI
    if mode == "MIL":
        base = T_MIL_SL_LBF * rho_ratio
    else:
        base = T_AB_SL_LBF * rho_ratio
    derate_clamped = max(derate, REDUCED_THRUST_FLOOR)
    return base * derate_clamped


def balanced_field_length_ft(gw_lbs: float, thrust_lbf: float, rho: float,
                             mu_roll: float, headwind_kt: float, slope_pct: float) -> float:
    """Very simplified field length model combining ground roll + rotation + climb segment.
    Tuned for continuity and monotonicity rather than fidelity. Returns feet.
    """
    # Ground roll term ~ function of (W/T), density, and friction
    w_over_t = gw_lbs / max(thrust_lbf, 1.0)
    base_roll_ft = 1200 * w_over_t  # tuned scalar

    # Density effect: thinner air -> longer
    dens_factor = RHO_SEA_LEVEL_SI / rho

    # Rotation + air distance term grows with Vs
    Vs = stall_speed_ias_kt(gw_lbs, rho)
    air_dist_ft = 7.5 * (V2_FACTOR * Vs) * 1.6878  # 7.5 s @ ~V2 in ft (coarse)

    # Friction effect
    fric_factor = 1.0 + (mu_roll - MU_DRY) * 6.0  # wet adds ~12% vs dry

    # Wind & slope effects (coarse): headwind reduces, uphill increases
    wind_factor = 1.0 + HEADWIND_FIELD_EFFECT * (headwind_kt / 10.0)
    slope_factor = 1.0 + SLOPE_GRAD_PER_UNIT * (slope_pct)

    bfl = (base_roll_ft + air_dist_ft) * dens_factor * fric_factor * wind_factor * slope_factor

    # Ensure sane lower bound
    return float(max(bfl, 500.0))


# --------------------------- Trim Targeting (V2-centric) ---------------------------
# Goal: suggest a stabilator trim that tends to settle near V2 in TO config w/gear up.
# Without full F-14 pitch-moment data, use a pragmatic schedule keyed to GW and V2.
# The schedule is intentionally conservative (less nose-up) to avoid rotation overshoot.

def trim_units_from_v2(gw_lbs: float, v2_kt: float) -> float:
    """Heuristic: small positive units that scale modestly with weight and V2.
    Tuned to avoid over-rotation. You can tweak the coefficients from the sidebar.
    """
    gw_klb = gw_lbs / 1000.0
    # Base schedule (nose-up increases mildly with GW and V2)
    trim_units = 0.02 * gw_klb + 0.005 * (v2_kt - 130)  # ~1.4 units @ 70k & V2≈155
    return float(max(min(trim_units, 3.5), -1.0))  # clamp to plausible range


# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🛫", layout="wide")
st.title(APP_TITLE)
st.caption("Engineering aid for Heatblur F-14B in DCS. Not for real-world flight.")

colA, colB = st.columns([2, 1])

with colB:
    st.subheader("Configuration")
    gw_lbs = st.number_input("Gross Weight [lbs]", min_value=40000, max_value=74000, value=70100, step=100)
    oat_c = st.number_input("Outside Air Temp [°C]", min_value=-40, max_value=60, value=40, step=1)
    pa_ft = st.number_input("Pressure Altitude [ft]", min_value=-2000, max_value=12000, value=0, step=100)

    flap = st.selectbox("Flaps", ["Maneuver"], index=0, help="Heatblur F-14B standard sim takeoff setting.")
    thrust_mode = st.selectbox("Thrust Mode", ["MIL", "AB"], index=0)

    derate = st.slider("Reduced Thrust (as % of selected mode)", min_value=0.80, max_value=1.00, value=1.00, step=0.01)
    st.caption("Guard: calculator enforces a \u2265 90% floor per your requirement.")

    wind_comp = st.number_input("Headwind (+) / Tailwind (-) [kt]", min_value=-20, max_value=40, value=0, step=1)
    slope_pct = st.number_input("Runway Slope [% up]", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    runway_len_ft = st.number_input("Runway Available [ft]", min_value=1500, max_value=16000, value=8000, step=100)
    surface = st.selectbox("Runway Condition", ["Dry", "Wet"], index=0)

    # Advanced tuning
    with st.expander("Advanced Tuning (use sparingly)"):
        global CLMAX_TO, V2_FACTOR, VR_FACTOR, V1_FACTOR
        CLMAX_TO = st.slider("CLmax (TO config)", 1.6, 2.4, CLMAX_TO, 0.05)
        V2_FACTOR = st.slider("V2 / Vs factor", 1.08, 1.25, V2_FACTOR, 0.01)
        VR_FACTOR = st.slider("VR / V2 factor", 0.92, 1.02, VR_FACTOR, 0.005)
        V1_FACTOR = st.slider("V1 / VR factor", 0.90, 0.99, V1_FACTOR, 0.005)
        trim_bias = st.slider("Trim Bias [units] (adds to schedule)", -1.0, 1.0, 0.0, 0.1)

with colA:
    st.subheader("Results")

    rho = isa_density(pa_ft, oat_c)
    mu = MU_DRY if surface == "Dry" else MU_WET

    Vs = stall_speed_ias_kt(gw_lbs, rho, CLMAX_TO)
    V2 = V2_FACTOR * Vs
    VR = VR_FACTOR * V2
    V1 = V1_FACTOR * VR

    T_avail = thrust_available_lbf(thrust_mode, rho, derate)
    BFL = balanced_field_length_ft(gw_lbs, T_avail, rho, mu, wind_comp, slope_pct)

    trim_units = trim_units_from_v2(gw_lbs, V2)
    # Apply user bias after core schedule
    trim_units += trim_bias
    trim_units = float(max(min(trim_units, 3.5), -1.0))

    # Field limit check
    field_ok = runway_len_ft >= BFL

    # Output summary table
    df = pd.DataFrame(
        {
            "Parameter": [
                "Vs (stall, IAS)", "V2 (target)", "VR", "V1",
                "Thrust Available", "Balanced Field Length", "Runway Available",
                "Suggested Stab Trim"
            ],
            "Value": [
                f"{Vs:0.0f} kt", f"{V2:0.0f} kt", f"{VR:0.0f} kt", f"{V1:0.0f} kt",
                f"{T_avail:,.0f} lbf", f"{BFL:,.0f} ft", f"{runway_len_ft:,.0f} ft",
                f"{trim_units:0.1f} units"
            ],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Status messages
    if field_ok:
        st.success("Field limit: OK (Runway >= Required).")
    else:
        shortfall = BFL - runway_len_ft
        st.error(f"Field limit: INSUFFICIENT by ~{shortfall:,.0f} ft.")

    st.markdown(
        """
        **Notes**
        - Speeds and distances are heuristic and intended for DCS plausibility, not NATOPS accuracy.
        - Trim schedule targets settling near **V2** in takeoff configuration with **gear up**. If you still see over-rotation, nudge **Trim Bias** negative (nose-down).
        - Reduced thrust enforces a **90% floor** of the selected regime (MIL/AB) per your requirement.
        - Remove any concept of RTO/"spare" by relying directly on the required field length estimate.
        """
    )

    # Quick Test Card (from your reported scenario)
    with st.expander("Quick Test Card — GW 70,100 lbs, OAT +40°C, Flaps Maneuver, MIL"):
        test_rho = isa_density(0, 40)
        test_vs = stall_speed_ias_kt(70100, test_rho, CLMAX_TO)
        test_v2 = V2_FACTOR * test_vs
        test_trim = trim_units_from_v2(70100, test_v2) + trim_bias
        st.write({
            "Vs_kt": round(test_vs),
            "V2_kt": round(test_v2),
            "Trim_units": round(max(min(test_trim, 3.5), -1.0), 1),
        })

# Footer
st.caption(
    "This tool is community-built for sim use. If you have NATOPS-derived data or Heatblur-specific curves, "
    "drop them into the model for higher fidelity (see source: trim schedule, CLmax, thrust lapse, BFL)."
)
