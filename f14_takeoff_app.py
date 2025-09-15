from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from functools import lru_cache

import numpy as np
import pandas as pd

# ================= CORE CONSTANTS =================
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}
DERATE_FLOOR_BY_FLAP = {0: 0.90, 20: 0.90, 40: 1.00}

ALPHA_N1_DIST = 2.0
AEO_VR_FRAC   = 0.88
AEO_VR_FRAC_FULL = 0.82

OEI_AGD_FACTOR = 1.20
AEO_CAL_FACTOR = 1.00

WIND_FACTORS: Dict[str, Tuple[float, float]] = {
    "None": (1.0, 1.0),
    "50/150": (0.5, 1.5),
}

# ================= ATMOSPHERE HELPERS =================
def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft + (29.92 - qnh_inhg) * 1000.0)

def isa_temp_c_at_ft(h_ft: float) -> float:
    return 15.0 - 1.98 * (h_ft / 1000.0)

def density_altitude_ft(pa_ft: float, oat_c: float) -> float:
    return float(pa_ft + 120.0 * (oat_c - isa_temp_c_at_ft(pa_ft)))

def sigma_from_da(da_ft: float) -> float:
    h_m = da_ft * 0.3048
    T0 = 288.15; L = 0.0065; g = 9.80665; R = 287.05
    p0 = 101325.0; rho0 = 1.225
    T = T0 - L*h_m
    p = p0 * (T/T0)**(g/(R*L))
    rho = p/(R*T)
    return float(rho/rho0)

def da_out_of_grid_scale(pa_ft: float, oat_c: float) -> float:
    da_act = density_altitude_ft(pa_ft, oat_c)
    da_ref = density_altitude_ft(min(pa_ft, 5000.0), min(oat_c, 30.0))
    sig_act = sigma_from_da(da_act)
    sig_ref = sigma_from_da(da_ref)
    BETA = 0.85
    return (sig_ref / max(1e-6, sig_act)) ** BETA

# ================= WIND =================
def wind_components(speed_kn: float, dir_deg: float, rwy_heading_deg: float) -> Tuple[float, float]:
    delta = math.radians((dir_deg - rwy_heading_deg) % 360.0)
    hw = speed_kn * math.cos(delta)
    cw = speed_kn * math.sin(delta)
    return hw, cw

def apply_wind_slope(distance_ft: float, slope_pct: float, headwind_kn: float, policy: str) -> float:
    d = float(distance_ft)
    if slope_pct > 0:
        d *= (1.0 + 0.20 * slope_pct)
    head_fac, tail_fac = WIND_FACTORS.get(policy, (1.0, 1.0))
    if headwind_kn >= 0:
        d *= (1.0 - 0.005 * head_fac * headwind_kn)
    else:
        d *= (1.0 - 0.005 * tail_fac * headwind_kn)
    return max(d, 0.0)

# ================= PERF TABLE HELPERS =================
def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_flap20(df: pd.DataFrame) -> pd.DataFrame:
    if (df["flap_deg"] == 20).any():
        return df
    keys = ["thrust","gw_lbs","press_alt_ft","oat_c"]
    up   = df[df["flap_deg"] == 0]
    full = df[df["flap_deg"] == 40]
    if up.empty or full.empty:
        return df
    m = pd.merge(up, full, on=keys, suffixes=("_up","_full"))
    if m.empty:
        return df
    def blend(a, b, w=0.4):
        return w*a + (1.0 - w)*b
    new = pd.DataFrame({
        "model": "F-14B",
        "flap_deg": 20,
        "thrust": m["thrust"],
        "gw_lbs": m["gw_lbs"],
        "press_alt_ft": m["press_alt_ft"],
        "oat_c": m["oat_c"],
        "Vs_kt": blend(m["Vs_kt_up"], m["Vs_kt_full"]),
        "V1_kt": blend(m["V1_kt_up"], m["V1_kt_full"]),
        "Vr_kt": blend(m["Vr_kt_up"], m["Vr_kt_full"]),
        "V2_kt": blend(m["V2_kt_up"], m["V2_kt_full"]),
        "ASD_ft": blend(m["ASD_ft_up"], m["ASD_ft_full"]),
        "AGD_ft": blend(m["AGD_ft_up"], m["AGD_ft_full"]),
        "note": "synth-MAN(20) 0.4*UP + 0.6*FULL"
    })
    return pd.concat([df, new], ignore_index=True)

def load_perf_csv(path: str = "f14_perf.csv") -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["thrust"] = df["thrust"].astype(str).str.upper().replace({"MIL": "MILITARY", "AB": "AFTERBURNER"})
    df = ensure_numeric(df, ["flap_deg","gw_lbs","press_alt_ft","oat_c","Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"])
    df = df.dropna(subset=["flap_deg","gw_lbs","press_alt_ft","oat_c","Vr_kt","ASD_ft","AGD_ft"])
    df = ensure_flap20(df)
    return df
# ================= INTERPOLATION =================
def _bounds(vals, x):
    vals = sorted(set(map(float, vals)))
    lo = max([v for v in vals if v <= x], default=vals[0])
    hi = min([v for v in vals if v >= x], default=vals[-1])
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return lo, hi, w

def _interp_weight_at(sub: pd.DataFrame, pa: float, oat: float, field: str, gw_x: float) -> float:
    s = sub[(sub["press_alt_ft"] == pa) & (sub["oat_c"] == oat)].sort_values("gw_lbs")
    if s.empty:
        s = sub.sort_values(["press_alt_ft", "oat_c", "gw_lbs"])
    xs = s["gw_lbs"].values.astype(float)
    ys = s[field].values.astype(float)
    if len(xs) < 2:
        return float(ys[0])
    if gw_x <= xs[0]:
        x0, x1 = xs[0], xs[1]; y0, y1 = ys[0], ys[1]
        return float(y0 + (y1 - y0)/(x1 - x0) * (gw_x - x0))
    if gw_x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]; y0, y1 = ys[-2], ys[-1]
        return float(y0 + (y1 - y0)/(x1 - x0) * (gw_x - x0))
    return float(np.interp(gw_x, xs, ys))

def interp_perf(perf: pd.DataFrame, flap_deg: int, thrust: str, gw: float, pa: float, oat: float):
    use_flap = 20 if flap_deg == 0 else flap_deg  # UP uses MAN table as base
    sub = perf[(perf["flap_deg"] == use_flap) & (perf["thrust"] == thrust)]
    if sub.empty:
        sub = perf[(perf["flap_deg"] == use_flap)]
        if sub.empty:
            sub = perf[(perf["thrust"] == thrust)]
        if sub.empty:
            sub = perf
    pa0, pa1, wp = _bounds(sub["press_alt_ft"].unique(), pa)
    t0,  t1,  wt = _bounds(sub["oat_c"].unique(),        oat)
    out = {}
    for f in ["Vs_kt","V1_kt","Vr_kt","V2_kt","ASD_ft","AGD_ft"]:
        v00 = _interp_weight_at(sub, pa0, t0, f, gw)
        v01 = _interp_weight_at(sub, pa0, t1, f, gw)
        v10 = _interp_weight_at(sub, pa1, t0, f, gw)
        v11 = _interp_weight_at(sub, pa1, t1, f, gw)
        v0  = v00*(1-wt) + v01*wt
        v1  = v10*(1-wt) + v11*wt
        out[f] = v0*(1-wp) + v1*wp
    return out

# ================= NATOPS DETECTION =================
def agd_is_liftoff_mode(perfdb: pd.DataFrame, flap_deg: int, thrust: str) -> bool:
    sub = perfdb[(perfdb["flap_deg"] == flap_deg) & (perfdb["thrust"] == thrust)]
    if sub.empty:
        return False
    mask = sub["note"].astype(str).str.contains("NATOPS", case=False, na=False)
    return bool(mask.mean() >= 0.5)

# ================= SPEED FLOORS =================
def enforce_speed_floors(vs, v1, vr, v2, flap_deg: int):
    import math
    def num(x, default=float("nan")):
        try:
            y = float(x)
            return y if math.isfinite(y) else default
        except Exception:
            return default

    vs = num(vs); v1 = num(v1); vr = num(vr); v2 = num(v2)
    if not math.isfinite(vs) or vs <= 0:
        vs = 110.0 if flap_deg == 40 else 120.0

    if not math.isfinite(v1): v1 = vs * 1.10
    if not math.isfinite(vr): vr = vs * 1.20
    if not math.isfinite(v2): v2 = vs * 1.30

    vmcg = 112.0 if flap_deg == 40 else 118.0

    v1_min = max(vmcg + 3.0, 0.95 * vr)
    vr_min = max(vmcg + 8.0, v1 + 3.0, 1.05 * vs)
    v2_min = max(vr + 10.0, 1.18 * vs)

    v1f = max(v1, v1_min)
    vrf = max(vr, vr_min)
    v2f = max(v2, v2_min)

    return float(int(round(v1f))), float(int(round(vrf))), float(int(round(v2f)))

# ================= TRIM =================
def trim_anu(gw_lbs: float, flap_deg: int) -> float:
    base = 4.5 + (gw_lbs - 60000.0)/10000.0 * 0.8
    if flap_deg == 0: base -= 1.0
    if flap_deg == 40: base += 1.0
    return float(max(2.0, min(8.0, round(base, 1))))

# ================= WIND PARSING =================
def parse_wind_entry(entry: str, unit: str) -> Optional[Tuple[float,float]]:
    text = (entry or "").strip().replace('/', ' ').replace('@', ' ')
    parts = [p for p in text.split(' ') if p]
    if len(parts) >= 2:
        try:
            d = float(parts[0]) % 360.0
            s = float(parts[1])
            if unit == "m/s":
                s *= 1.943844
            return (d, s)
        except Exception:
            return None
    return None

# ================= RESULT DATACLASS =================
@dataclass
class Result:
    v1: float; vr: float; v2: float; vs: float
    flap_text: str; thrust_text: str; n1_pct: float
    asd_ft: float; agd_aeo_liftoff_ft: float; agd_reg_oei_ft: float
    req_ft: float; avail_ft: float; limiting: str
    hw_kn: float; cw_kn: float; notes: list

# ================= COMPUTE TAKEOFF =================
def compute_takeoff(perfdb: pd.DataFrame,
                    rwy_heading_deg: float, tora_ft: float, toda_ft: float, asda_ft: float,
                    field_elev_ft: float, slope_pct: float, shorten_ft: float,
                    oat_c: float, qnh_inhg: float,
                    wind_speed: float, wind_dir_deg: float, wind_units: str, wind_policy: str,
                    gw_lbs: float,
                    flap_mode: str,
                    thrust_mode: str,
                    target_n1_pct: float) -> Result:

    # Atmosphere & wind
    pa = pressure_altitude_ft(field_elev_ft, qnh_inhg)
    spd_kn = wind_speed if wind_units == "kts" else wind_speed * 1.943844
    hw, cw = wind_components(spd_kn, wind_dir_deg, rwy_heading_deg)

    # Flap handling
    flap_text = "MANEUVER" if flap_mode == "Auto-Select" else flap_mode
    flap_deg = 0 if flap_text.upper().startswith("UP") else (40 if flap_text.upper().startswith("FULL") else 20)

    # Interpolated base
    table_thrust = "AFTERBURNER" if thrust_mode == "AB" else "MILITARY"
    base = interp_perf(perfdb, flap_deg if flap_deg != 0 else 20, table_thrust, gw_lbs, pa, oat_c)

    vs = float(base.get("Vs_kt", np.nan))
    v1 = float(base.get("V1_kt", np.nan))
    vr = float(base.get("Vr_kt", np.nan))
    v2 = float(base.get("V2_kt", np.nan))
    v1, vr, v2 = enforce_speed_floors(vs, v1, vr, v2, flap_deg)

    asd = float(base["ASD_ft"])
    agd = float(base["AGD_ft"])
    agd_liftoff = agd * 1.42 if not agd_is_liftoff_mode(perfdb, flap_deg, table_thrust) else agd

    return Result(
        v1=v1, vr=vr, v2=v2, vs=vs,
        flap_text=flap_text, thrust_text=thrust_mode, n1_pct=target_n1_pct,
        asd_ft=asd, agd_aeo_liftoff_ft=agd_liftoff, agd_reg_oei_ft=agd*OEI_AGD_FACTOR,
        req_ft=max(asd, agd), avail_ft=tora_ft, limiting="ASD" if asd > agd else "AGD",
        hw_kn=hw, cw_kn=cw, notes=[]
    )
# ============================== STREAMLIT UI ==============================

import streamlit as st

st.set_page_config(page_title="DCS F-14B Takeoff", page_icon="✈️", layout="wide")

st.title("DCS F-14B Takeoff — Merged App (Core + UI)")

# Sidebar inputs
with st.sidebar:
    st.header("Runway & Weather")

    tora_ft = st.number_input("TORA (ft)", min_value=1000, max_value=20000, value=10000, step=100)
    toda_ft = st.number_input("TODA (ft)", min_value=1000, max_value=20000, value=int(tora_ft), step=100)
    asda_ft = st.number_input("ASDA (ft)", min_value=1000, max_value=20000, value=int(tora_ft), step=100)
    elev_ft = st.number_input("Field Elevation (ft)", min_value=-1000, max_value=10000, value=2000, step=50)
    slope_pct = st.number_input("Runway slope (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    rwy_hdg = st.number_input("Runway heading (deg)", min_value=0.0, max_value=360.0, value=350.0, step=0.1)

    shorten_ft = st.number_input("Shorten available (ft)", min_value=0.0, max_value=5000.0, value=0.0, step=50.0)

    oat_c = st.number_input("OAT (°C)", min_value=-40.0, max_value=60.0, value=15.0, step=1)
    qnh_inhg = st.number_input("QNH (inHg)", min_value=27.50, max_value=31.50, value=29.92, step=0.01)

    wind_dir = st.number_input("Wind direction (deg)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
    wind_spd = st.number_input("Wind speed", min_value=0.0, max_value=80.0, value=0.0, step=1.0)
    wind_units = st.selectbox("Wind units", ["kts", "m/s"], index=0)
    wind_policy = st.selectbox("Wind policy", ["None", "50/150"], index=0)

with st.sidebar:
    st.header("Weight & Config")

    gw_lbs = st.number_input("Gross Weight (lb)", min_value=40000.0, max_value=80000.0, value=70000.0, step=100.0)
    flap_mode = st.selectbox("Flaps", ["Auto-Select", "UP", "MANEUVER", "FULL"], index=2)
    thrust_mode = st.selectbox("Thrust", ["Auto-Select", "MIL", "AB", "DERATE"], index=1)
    n1_pct = st.slider("Target N1 % (for DERATE)", min_value=90, max_value=100, value=95)

# Load perf table
perfdb = load_perf_csv("f14_perf.csv")

# Compute
if st.button("Compute Takeoff"):
    res = compute_takeoff(perfdb,
                          rwy_hdg, tora_ft, toda_ft, asda_ft,
                          elev_ft, slope_pct, shorten_ft,
                          oat_c, qnh_inhg,
                          wind_spd, wind_dir, wind_units, wind_policy,
                          gw_lbs, flap_mode, thrust_mode, n1_pct)

    # Display results
    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("V1 (kt)", f"{res.v1:.0f}")
        st.metric("Vr (kt)", f"{res.vr:.0f}")
        st.metric("V2 (kt)", f"{res.v2:.0f}")
        st.metric("Vs (kt)", f"{res.vs:.0f}")
    with c2:
        st.metric("ASD (ft)", f"{res.asd_ft:,.0f}")
        st.metric("AGD OEI (ft)", f"{res.agd_reg_oei_ft:,.0f}")
        st.metric("AGD AEO (ft)", f"{res.agd_aeo_liftoff_ft:,.0f}")
    with c3:
        st.metric("Runway Required (ft)", f"{res.req_ft:,.0f}")
        st.metric("Runway Available (ft)", f"{res.avail_ft:,.0f}")
        st.metric("Limiting", res.limiting)

    st.caption(f"Headwind: {res.hw_kn:.1f} kt | Crosswind: {res.cw_kn:.1f} kt")
    if res.notes:
        for note in res.notes:
            st.warning(note)

    st.markdown("---")
    st.subheader("All-engines (for DCS)")
    vr_gr_roll = res.agd_aeo_liftoff_ft * (AEO_VR_FRAC_FULL if flap_mode.upper()=="FULL" else AEO_VR_FRAC)
    st.metric("Vr ground roll (ft)", f"{vr_gr_roll:,.0f}")
    st.metric("Liftoff to 35 ft (ft)", f"{res.agd_aeo_liftoff_ft:,.0f}")

    st.caption("AEO estimates shown for DCS reference. Regulatory checks use OEI factors per FAR 121.189.")

    # Kneeboard summary line
    kneeboard_line = (
        f"{int(gw_lbs):,} lb | {flap_mode} | {thrust_mode} {n1_pct:.0f}% | "
        f"OAT {oat_c:.0f}°C | QNH {qnh_inhg:.2f} | Wind {wind_dir:.0f}/{wind_spd:.0f}{wind_units} | "
        f"V1 {res.v1:.0f} | Vr {res.vr:.0f} | V2 {res.v2:.0f} | ASD {res.asd_ft:,.0f} ft | "
        f"AEO {res.agd_aeo_liftoff_ft:,.0f} ft"
    )
    st.text_area("Kneeboard line", kneeboard_line, height=80)

# Footer
st.markdown("---")
st.caption("Merged F-14B takeoff calculator • Stage 3 UI • Next: add trends, what-ifs, optimizer, exports.")
import altair as alt
import json
import io

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Results", "Trends", "Matrix", "Optimizer", "Data Checker"]
)

with tab1:
    st.write("Results are shown above in the main view.")

with tab2:
    st.subheader("Trend Charts")
    xaxis = st.selectbox("X-axis", ["Weight", "OAT", "Pressure Altitude"])
    if st.button("Generate Trends"):
        xs = []
        ys_asd = []
        ys_agd = []
        for w in np.linspace(50000, 75000, 6):
            base = compute_takeoff(perfdb, rwy_hdg, tora_ft, toda_ft, asda_ft,
                                   elev_ft, slope_pct, shorten_ft,
                                   oat_c, qnh_inhg,
                                   wind_spd, wind_dir, wind_units, wind_policy,
                                   w, flap_mode, thrust_mode, n1_pct)
            xs.append(w)
            ys_asd.append(base.asd_ft)
            ys_agd.append(base.agd_aeo_liftoff_ft)
        chart = alt.Chart(pd.DataFrame({
            "Weight": xs, "ASD": ys_asd, "AGD": ys_agd
        })).transform_fold(["ASD","AGD"], as_=["Type","Value"]).mark_line().encode(
            x="Weight", y="Value", color="Type"
        )
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.subheader("What-if Matrix")
    if st.button("Generate Matrix"):
        ws = [60000, 65000, 70000, 75000]
        ts = ["MIL", "AB"]
        data = []
        for w in ws:
            row = {"Weight": w}
            for t in ts:
                r = compute_takeoff(perfdb, rwy_hdg, tora_ft, toda_ft, asda_ft,
                                    elev_ft, slope_pct, shorten_ft,
                                    oat_c, qnh_inhg,
                                    wind_spd, wind_dir, wind_units, wind_policy,
                                    w, flap_mode, t, n1_pct)
                row[t] = f"{r.req_ft:,.0f}"
            data.append(row)
        st.dataframe(pd.DataFrame(data))

with tab4:
    st.subheader("Optimizer (find min runway required)")
    target = st.selectbox("Optimize for", ["Weight", "OAT", "Pressure Altitude"])
    if st.button("Run Optimizer"):
        best = None
        for w in np.linspace(50000, 75000, 11):
            r = compute_takeoff(perfdb, rwy_hdg, tora_ft, toda_ft, asda_ft,
                                elev_ft, slope_pct, shorten_ft,
                                oat_c, qnh_inhg,
                                wind_spd, wind_dir, wind_units, wind_policy,
                                w, flap_mode, thrust_mode, n1_pct)
            if best is None or r.req_ft < best.req_ft:
                best = r
        if best:
            st.success(f"Best case: {best.req_ft:,.0f} ft required at {best.vr:.0f} kt")

with tab5:
    st.subheader("Data Checker")
    st.write(perfdb.head())
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.subheader("Scenario Save/Load")

save_name = st.text_input("Scenario name")
if st.button("Save Scenario") and save_name:
    state = {
        "tora": tora_ft, "toda": toda_ft, "asda": asda_ft,
        "elev": elev_ft, "slope": slope_pct, "rwy_hdg": rwy_hdg,
        "shorten": shorten_ft, "oat": oat_c, "qnh": qnh_inhg,
        "wind_dir": wind_dir, "wind_spd": wind_spd,
        "wind_units": wind_units, "wind_policy": wind_policy,
        "gw": gw_lbs, "flap": flap_mode, "thrust": thrust_mode, "n1": n1_pct
    }
    st.download_button("Download JSON", data=json.dumps(state), file_name=f"{save_name}.json")

upload = st.file_uploader("Load Scenario", type="json")
if upload:
    state = json.load(upload)
    st.json(state)

st.subheader("Export Kneeboard PDF")
if st.button("Export PDF"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph("F-14B Takeoff Kneeboard", styles["Heading1"])]
    story.append(Paragraph(kneeboard_line, styles["Normal"]))
    doc.build(story)
    st.download_button("Download PDF", data=buffer.getvalue(), file_name="kneeboard.pdf")
st.subheader("A/B Compare")
colA, colB = st.columns(2)

with colA:
    st.write("Runway A")
    toraA = st.number_input("TORA A", 1000, 20000, 10000, 100)
    resA = compute_takeoff(perfdb, rwy_hdg, toraA, toraA, toraA,
                           elev_ft, slope_pct, shorten_ft,
                           oat_c, qnh_inhg,
                           wind_spd, wind_dir, wind_units, wind_policy,
                           gw_lbs, flap_mode, thrust_mode, n1_pct)
    st.metric("Req ft (A)", f"{resA.req_ft:,.0f}")

with colB:
    st.write("Runway B")
    toraB = st.number_input("TORA B", 1000, 20000, 12000, 100)
    resB = compute_takeoff(perfdb, rwy_hdg, toraB, toraB, toraB,
                           elev_ft, slope_pct, shorten_ft,
                           oat_c, qnh_inhg,
                           wind_spd, wind_dir, wind_units, wind_policy,
                           gw_lbs, flap_mode, thrust_mode, n1_pct)
    st.metric("Req ft (B)", f"{resB.req_ft:,.0f}")

if 'resA' in locals() and 'resB' in locals():
    if resA.req_ft < resB.req_ft:
        st.success("Runway A is more favorable")
    elif resB.req_ft < resA.req_ft:
        st.success("Runway B is more favorable")
    else:
        st.info("Both runways are equal")
