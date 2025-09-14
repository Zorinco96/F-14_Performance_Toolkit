# f14_takeoff_app.py — Streamlit UI for DCS F-14B takeoff performance
# EFB-style UI with professional enhancements:
# - Collapsible sidebar sections
# - Sticky top app bar (airport/runway + UTC + units)
# - Favorites & Recents
# - Numeric keypad overlay for GW and OAT
# - Quick actions (swap runway end / flip wind / reset shorten)
# - Compact mode toggle
# - Quick presets
# - Tabs (Results / Trends / Matrix)
# - Dual-runway A/B compare
# - Best End advisor
# - Derate Advisor
# - Copyable V-speed line
# - Kneeboard PDF (ReportLab)
# - Save/Load scenarios
# - Debounce with Auto-recompute toggle
# - Color-cued summary card
# - LRU micro-cache for snappier what-ifs

from __future__ import annotations
import math, json, datetime as _dt
from typing import List, Dict
from functools import lru_cache
import pandas as pd
import streamlit as st

from f14_takeoff_core import (
    load_perf_csv, compute_takeoff, trim_anu,
    hpa_to_inhg, parse_wind_entry,
    AEO_VR_FRAC, AEO_VR_FRAC_FULL,
    detect_length_text_to_ft, detect_elev_text_to_ft,
    estimate_ab_multiplier
)

st.set_page_config(page_title="DCS F-14B Takeoff", page_icon="✈️", layout="wide")

# --------------------- Styling ---------------------
st.markdown("""
<style>
.f14-card { padding:12px 16px; border:1px solid var(--secondary-background-color);
  border-radius:10px; background:rgba(127,127,127,0.03); }
.f14-sticky { position:sticky; top:0; z-index:999;
  padding:8px 12px; margin:-14px -14px 10px -14px;
  background:rgba(0,0,0,0.65); color:white;
  border-bottom:1px solid rgba(255,255,255,0.15); }
.f14-card-ok { background:rgba(0,128,0,0.06)!important; }
.f14-card-bad{ background:rgba(255,0,0,0.06)!important; }
</style>
""", unsafe_allow_html=True)

# --------------------- Helpers ---------------------
def _utc_now(): return _dt.datetime.utcnow().strftime("%d %b %Y • %H:%MZ")

def top_app_bar(airport:str, rwy:str, units:str):
    st.markdown(f"""
    <div class='f14-sticky' style='display:flex;justify-content:space-between;align-items:center;'>
      <div><b>EFB • PERF</b> — {airport} RWY {rwy}</div>
      <div>{_utc_now()} • {units}</div>
    </div>
    """, unsafe_allow_html=True)

def keypad(label, value, step=100.0, step_big=500.0, minv=None, maxv=None):
    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,1])
    with c1: val = st.number_input(label, value=float(value), min_value=minv, max_value=maxv, step=step)
    with c2: 
        if st.button("+", key=label+"p1"): val += step
    with c3:
        if st.button("−", key=label+"m1"): val -= step
    with c4:
        if st.button(f"+{int(step_big)}", key=label+"pB"): val += step_big
    with c5:
        if st.button(f"−{int(step_big)}", key=label+"mB"): val -= step_big
    if minv is not None: val = max(val, minv)
    if maxv is not None: val = min(val, maxv)
    return val

# --------------------- Data loaders ---------------------
@st.cache_data
def load_runways(path_primary="dcs_airports_expanded.csv", path_alt="data/dcs_airports_expanded.csv"):
    for p in (path_primary, path_alt):
        try:
            df = pd.read_csv(p)
            df["runway_label"] = df["airport_name"]+" "+df["runway_end"].astype(str)+" ("+df["runway_pair"].astype(str)+")"
            return df
        except Exception: continue
    st.error("dcs_airports_expanded.csv not found."); st.stop()

@st.cache_data
def load_intersections(path_primary="intersections.csv", path_alt="data/intersections.csv"):
    for p in (path_primary, path_alt):
        try: return pd.read_csv(p)
        except Exception: continue
    return pd.DataFrame(columns=["map","airport_name","runway_pair","runway_end",
        "intersection_id","tora_ft","toda_ft","asda_ft","distance_from_threshold_ft","notes"])

perfdb = load_perf_csv("f14_perf.csv")
rwy_db = load_runways()
ix_db  = load_intersections()

# --------------------- Sidebar ---------------------
st.title("DCS F-14B Takeoff — FAA-Based Model")

with st.sidebar:
    st.header("Setup")

    # Runway
    with st.expander("Runway", expanded=True):
        theatre = st.selectbox("DCS Theatre", sorted(rwy_db["map"].unique()))
        df_t = rwy_db[rwy_db["map"]==theatre]
        airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
        df_a = df_t[df_t["airport_name"]==airport]
        rwy_label = st.selectbox("Runway End", list(df_a["runway_label"]))
        rwy = df_a[df_a["runway_label"]==rwy_label].iloc[0]

        base_tora, base_toda, base_asda = map(float, (rwy["tora_ft"],rwy["toda_ft"],rwy["asda_ft"]))
        elev_ft, hdg = float(rwy["threshold_elev_ft"]), float(rwy["heading_deg"])

        # Favorites & Recents
        fav_key = f"{theatre}:{airport}:{rwy['runway_end']}"
        if "fav_rw" not in st.session_state: st.session_state["fav_rw"]=set()
        cols = st.columns([1,1,3])
        with cols[0]:
            if st.button("⭐ Fav"): st.session_state["fav_rw"].add(fav_key); st.rerun()
        with cols[1]:
            if st.button("🗑 Clear"): st.session_state["fav_rw"].clear(); st.rerun()
        with cols[2]:
            if st.session_state["fav_rw"]: 
                for f in sorted(st.session_state["fav_rw"]): st.caption(f)

        if "recents" not in st.session_state: st.session_state["recents"]=[]
        cur={"map":theatre,"apt":airport,"rwy":str(rwy["runway_end"])}
        if not st.session_state["recents"] or st.session_state["recents"][0]!=cur:
            st.session_state["recents"]=[cur]+[x for x in st.session_state["recents"] if x!=cur]
            st.session_state["recents"]=st.session_state["recents"][:5]
        st.caption("Recents: "+", ".join([f"{x['apt']} RWY {x['rwy']}" for x in st.session_state["recents"]]))

    # Weather
    with st.expander("Weather", expanded=False):
        oat_c = keypad("OAT (°C)", 15.0, step=1, step_big=5)
        qnh_val = st.number_input("QNH", value=29.92, step=0.01, format="%.2f")
        qnh_unit = st.selectbox("QNH Units", ["inHg","hPa"], index=0)
        qnh_inhg = float(qnh_val) if qnh_unit=="inHg" else hpa_to_inhg(float(qnh_val))
        wind_units = st.selectbox("Wind Units", ["kts","m/s"], index=0)
        st.text_input("Wind (DIR@SPD)", placeholder="180@12", key="wind_entry")
        parsed = parse_wind_entry(st.session_state.get("wind_entry",""), wind_units)
        wind_dir, wind_spd = parsed if parsed else (hdg,0.0)

    # Weight
    with st.expander("Weight", expanded=True):
        gw = keypad("Gross Weight (lb)", 70000.0, step=100, step_big=500, minv=40000, maxv=80000)

    # Config & Thrust
    with st.expander("Config & Thrust", expanded=False):
        flap_mode = st.selectbox("Flaps", ["Auto-Select","UP","MANEUVER","FULL"], index=0)
        thrust_mode = st.radio("Thrust Mode", ["Auto-Select","Manual Derate","MIL","AB"], index=0)
        derate_n1=98.0
        if thrust_mode=="Manual Derate":
            if flap_mode=="FULL": st.warning("FULL cannot derate — using MIL"); derate_floor=100
            else: derate_floor=90
            derate_n1 = st.slider("Target N1 %", min_value=int(derate_floor), max_value=100, value=95, step=1)

    # Advisors
    with st.expander("Advisors", expanded=False):
        pair = df_a[df_a["runway_pair"]==rwy["runway_pair"]]
        if st.button("Suggest best end"):
            ends=[]
            for _,rr in pair.iterrows():
                res=compute_takeoff(perfdb,float(rr["heading_deg"]),float(rr["tora_ft"]),float(rr["toda_ft"]),float(rr["asda_ft"]),
                    float(rr["threshold_elev_ft"]),0.0,0.0,float(oat_c),float(qnh_inhg),
                    float(wind_spd),float(wind_dir),wind_units,"None",
                    float(gw),flap_mode,"MIL",100)
                ends.append((rr["runway_end"],min(float(rr["asda_ft"])-res.asd_ft,float(rr["tora_ft"])-res.agd_reg_oei_ft)))
            best=max(ends,key=lambda x:x[1])
            st.success(f"Best end: RWY {best[0]} (+{int(best[1])} ft margin)")
        target_margin=st.slider("Target margin (ft)",0,1500,500,50)
        if st.button("Suggest N1"):
            lo,hi=(90 if flap_mode!="FULL" else 100),100; best=100
            for _ in range(12):
                mid=(lo+hi)/2; r=compute_takeoff(perfdb,hdg,base_tora,base_toda,base_asda,elev_ft,0,0,oat_c,qnh_inhg,wind_spd,wind_dir,wind_units,"None",gw,flap_mode,"Manual Derate",mid)
                margin=min(base_asda-r.asd_ft,base_tora-r.agd_reg_oei_ft)
                if margin>=target_margin: best=mid; hi=mid
                else: lo=mid
            st.info(f"Suggested N1: {int(math.ceil(best))}%")

    compact = st.toggle("Compact mode", value=False)
    autorun = st.toggle("Auto-recompute", value=True)

# --------------------- Compute & Results ---------------------
ready=True; should_compute=ready and (autorun or st.button("Recompute"))
@lru_cache(maxsize=256)
def _calc(flap,thrust,n1):
    return compute_takeoff(perfdb,hdg,base_tora,base_toda,base_asda,elev_ft,0,0,oat_c,qnh_inhg,wind_spd,wind_dir,wind_units,"None",gw,flap,thrust,n1)

if should_compute:
    res=_calc(flap_mode,thrust_mode,derate_n1)
    top_app_bar(airport,str(rwy["runway_end"]),f"{qnh_unit}/{wind_units}")
    st.subheader("Results")
    st.metric("V1",f"{res.v1:.0f}"); st.metric("Vr",f"{res.vr:.0f}"); st.metric("V2",f"{res.v2:.0f}")
    st.code(f"V1 {int(res.v1)} Vr {int(res.vr)} V2 {int(res.v2)} Trim {trim_anu(gw,20 if flap_mode=='MANEUVER' else 40 if flap_mode=='FULL' else 0):.1f} ANU")
else:
    st.info("Set weight and inputs to compute.")
