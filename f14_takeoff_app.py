import streamlit as st
import math

# ---- Constants ----
g = 9.80665
ft2m = 0.3048
lb2kg = 0.45359237
kt2ms = 0.514444
mps2kt = 1/kt2ms
rho_sl = 1.225

# Heatblur F-14 approximations
WING_AREA_M2 = 52.5
TOTAL_MAX_THRUST_LBF = 34154 * 2
TOTAL_MAX_THRUST_N = TOTAL_MAX_THRUST_LBF * 4.44822162

# Aerodynamics
Cd0 = 0.025
k = 0.040
mu_roll = 0.02
brake_decel_g = 0.35
CLmax_by_flap = {0:1.2, 10:1.4, 20:1.7, 30:2.0, 40:2.1}
obstacle_height_ft = 35.0
rotation_time = 2.0

# ---- Helper functions ----
def air_density(alt_ft, oat_c):
    alt_m = alt_ft * ft2m
    T0 = 288.15
    L = 0.0065
    P0 = 101325
    R = 287.05
    T_std = T0 - L * alt_m
    T = oat_c + 273.15
    P = P0 * (T0 / T_std) ** (g / (R * L)) * (T / T_std) ** (g / (R * L))
    rho = P / (R * T)
    if rho <= 0:
        rho = rho_sl * math.exp(-alt_m / 8000.0)
    return rho

def v_stall(weight_kg, rho, S, CLmax):
    W = weight_kg * g
    return math.sqrt(2*W / (rho * S * CLmax))

def accelerate_stop_distance(V1, mass, thrust_N, rho, S, W):
    # super simplified accelerate-stop
    a_brake = brake_decel_g * g
    return (V1**2) / (2 * a_brake) + 500  # +500 m fudge factor for accel

def accelerate_go_distance(Vr, V2):
    # super simplified accelerate-go
    return Vr*rotation_time + V2*3  # fudge factor for airborne accel/climb

def find_takeoff_speeds(weight_lbs, oat_c, alt_ft, flap_deg, thrust_fraction):
    weight_kg = weight_lbs * lb2kg
    rho = air_density(alt_ft, oat_c)
    CLmax = CLmax_by_flap.get(flap_deg, 1.7)
    Vs = v_stall(weight_kg, rho, WING_AREA_M2, CLmax)
    Vr = Vs*1.05
    V2 = Vs*1.20
    V1 = Vs*1.04
    stop_dist = accelerate_stop_distance(V1, weight_kg, TOTAL_MAX_THRUST_N, rho, WING_AREA_M2, weight_kg*g)
    go_dist = accelerate_go_distance(Vr, V2)
    trim = 8.0 + (weight_lbs - 50000) / 10000.0
    trim = max(5.0, min(12.0, trim))
    return {
        "Vs": round(Vs*mps2kt,1),
        "Vr": round(Vr*mps2kt,1),
        "V2": round(V2*mps2kt,1),
        "V1": round(V1*mps2kt,1),
        "Stop Distance (m)": round(stop_dist,1),
        "Go Distance (m)": round(go_dist,1),
        "Flaps": flap_deg,
        "Trim": round(trim,1)
    }

# ---- Streamlit UI ----
st.title("F-14B Takeoff Calculator (DCS)")

weight = st.number_input("Gross Weight (lbs)", 40000, 74000, 60000, 500)
temp = st.number_input("OAT (°C)", -30, 50, 15, 1)
alt = st.number_input("Field Elevation (ft)", 0, 8000, 0, 100)
flaps = st.selectbox("Flap Setting", [0,10,20,30,40], index=2)
thrust_fraction = st.slider("Thrust Setting (fraction of MIL)", 0.5, 1.0, 1.0)

if st.button("Calculate"):
    res = find_takeoff_speeds(weight, temp, alt, flaps, thrust_fraction)
    st.subheader("Results")
    for k,v in res.items():
        st.write(f"**{k}:** {v}")
