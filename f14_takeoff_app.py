# app.py — DCS F‑14B/D Takeoff Performance (Streamlit)
# Fully self‑contained: runway & performance data embedded (no external CSVs)
#
# Notes
# • Uses your provided dcs_airports.csv, perf_f14b.csv, perf_f14d.csv embedded below.
# • Balanced‑field style check with current single‑V1 rows: Required = max(ASD, AGD)
# • Wind/slope simple adjustments; refine with NATOPS corrections if desired.
# • For DCS sim planning only.

import math
from io import StringIO
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DCS F‑14B/D Takeoff Performance", page_icon="✈️", layout="wide")

# ==========================
# Embedded CSV data (verbatim)
# ==========================
RUNWAYS_CSV = """map,airport_name,runway_pair,runway_end,heading_deg,length_ft,tora_ft,toda_ft,asda_ft,threshold_elev_ft,opp_threshold_elev_ft,slope_percent,notes
Caucasus,Batumi,13/31,13,126,8530,8530,8530,8530,33,39,,Public real-world data; verify TORA/TODA/ASDA in DCS
Caucasus,Batumi,13/31,31,306,8530,8530,8530,8530,39,33,,Public real-world data; verify
Caucasus,Kobuleti,07/25,07,73,8080,8080,8080,8080,59,58,,Public real-world data; verify
Caucasus,Kobuleti,07/25,25,253,8080,8080,8080,8080,58,59,,Public real-world data; verify
Caucasus,Senaki-Kolkhi,09/27,09,94,7870,7870,7870,7870,98,98,,Public real-world data; verify
Caucasus,Senaki-Kolkhi,09/27,27,274,7870,7870,7870,7870,98,98,,Public real-world data; verify
Caucasus,Sukhumi-Babushara,07/25,07,74,8202,8202,8202,8202,39,33,,Public real-world data; verify
Caucasus,Sukhumi-Babushara,07/25,25,254,8202,8202,8202,8202,33,39,,Public real-world data; verify
Caucasus,Gudauta,08/26,08,80,6562,6562,6562,6562,131,121,,Public real-world data; verify
Caucasus,Gudauta,08/26,26,260,6562,6562,6562,6562,121,131,,Public real-world data; verify
Caucasus,Sochi-Adler,06/24,06,62,7218,7218,7218,7218,89,89,,Public real-world data; verify
Caucasus,Sochi-Adler,06/24,24,242,7218,7218,7218,7218,89,89,,Public real-world data; verify
Caucasus,Anapa-Vityazevo,04/22,04,40,8858,8858,8858,8858,174,174,,Public real-world data; verify
Caucasus,Anapa-Vityazevo,04/22,22,220,8858,8858,8858,8858,174,174,,Public real-world data; verify
Caucasus,Krasnodar-Pashkovsky,09/27,09,90,10056,10056,10056,10056,112,115,,Public real-world data; verify
Caucasus,Krasnodar-Pashkovsky,09/27,27,270,10056,10056,10056,10056,115,112,,Public real-world data; verify
Caucasus,Krymsk,04/22,04,40,8038,8038,8038,8038,118,118,,Public real-world data; verify
Caucasus,Krymsk,04/22,22,220,8038,8038,8038,8038,118,118,,Public real-world data; verify
Caucasus,Maykop-Khanskaya,04/22,04,40,8202,8202,8202,8202,607,607,,Public real-world data; verify
Caucasus,Maykop-Khanskaya,04/22,22,220,8202,8202,8202,8202,607,607,,Public real-world data; verify
Caucasus,Mineralnye Vody,12/30,12,120,12960,12960,12960,12960,1053,1053,,Public real-world data; verify
Caucasus,Mineralnye Vody,12/30,30,300,12960,12960,12960,12960,1053,1053,,Public real-world data; verify
Caucasus,Nalchik,06/24,06,60,7202,7202,7202,7202,1479,1479,,Public real-world data; verify
Caucasus,Nalchik,06/24,24,240,7202,7202,7202,7202,1479,1479,,Public real-world data; verify
Caucasus,Mozdok,08/26,08,80,12008,12008,12008,12008,512,512,,Public real-world data; verify
Caucasus,Mozdok,08/26,26,260,12008,12008,12008,12008,512,512,,Public real-world data; verify
Caucasus,Tbilisi-Lochini,13/31,13,134,9800,9800,9800,9800,1624,1624,,Public real-world data; verify
Caucasus,Tbilisi-Lochini,13/31,31,314,9800,9800,9800,9800,1624,1624,,Public real-world data; verify
Caucasus,Vaziani,13/31,13,134,8071,8071,8071,8071,1545,1545,,Public real-world data; verify
Caucasus,Vaziani,13/31,31,314,8071,8071,8071,8071,1545,1545,,Public real-world data; verify
Caucasus,Beslan,10/28,10,96,8176,8176,8176,8176,1710,1710,,Public real-world data; verify
Caucasus,Beslan,10/28,28,276,8176,8176,8176,8176,1710,1710,,Public real-world data; verify
Persian Gulf,Al Dhafra AB,13L/31R,13L,128,12400,12400,12400,12400,77,79,,Public real-world data; verify
Persian Gulf,Al Dhafra AB,13L/31R,31R,308,12400,12400,12400,12400,79,77,,Public real-world data; verify
Persian Gulf,Khasab,01/19,01,14,8074,8074,8074,8074,100,100,,Public real-world data; verify
Persian Gulf,Khasab,01/19,19,194,8074,8074,8074,8074,100,100,,Public real-world data; verify
Persian Gulf,Fujairah Intl,11/29,11,113,12467,12467,12467,12467,152,152,,Public real-world data; verify
Persian Gulf,Fujairah Intl,11/29,29,293,12467,12467,12467,12467,152,152,,Public real-world data; verify
Persian Gulf,Al Ain Intl,01/19,01,13,13123,13123,13123,13123,869,869,,Public real-world data; verify
Persian Gulf,Al Ain Intl,01/19,19,193,13123,13123,13123,13123,869,869,,Public real-world data; verify
Persian Gulf,Bandar Abbas Intl,09/27,09,91,12004,12004,12004,12004,22,22,,Public real-world data; verify
Persian Gulf,Bandar Abbas Intl,09/27,27,271,12004,12004,12004,12004,22,22,,Public real-world data; verify
Persian Gulf,Kish Intl,09/27,09,88,12121,12121,12121,12121,101,101,,Public real-world data; verify
Persian Gulf,Kish Intl,09/27,27,268,12121,12121,12121,12121,101,101,,Public real-world data; verify
Persian Gulf,Sirri Island,12/30,12,115,8530,8530,8530,8530,43,43,,Public real-world data; verify
Persian Gulf,Sirri Island,12/30,30,295,8530,8530,8530,8530,43,43,,Public real-world data; verify
Persian Gulf,Qeshm Intl,04/22,04,41,14000,14000,14000,14000,45,45,,Public real-world data; verify
Persian Gulf,Qeshm Intl,04/22,22,221,14000,14000,14000,14000,45,45,,Public real-world data; verify
Syria,Damascus Intl,05/23,05,47,11500,11500,11500,11500,2025,2025,,Public real-world data; verify
Syria,Damascus Intl,05/23,23,227,11500,11500,11500,11500,2025,2025,,Public real-world data; verify
Syria,Beirut-Rafic Hariri,16/34,16,157,10660,10660,11800,10660,87,64,,Public real-world data; verify
Syria,Beirut-Rafic Hariri,16/34,34,337,10660,10660,11800,10660,64,87,,Public real-world data; verify
Syria,Aleppo Intl,09/27,09,85,9514,9514,9514,9514,1276,1276,,Public real-world data; verify
Syria,Aleppo Intl,09/27,27,265,9514,9514,9514,9514,1276,1276,,Public real-world data; verify
Syria,Hama,09/27,09,90,10500,10500,10500,10500,1023,1023,,Public real-world data; verify
Syria,Hama,09/27,27,270,10500,10500,10500,10500,1023,1023,,Public real-world data; verify
Syria,Khmeimim,17/35,17,168,9730,9730,9730,9730,43,47,,Public real-world data; verify
Syria,Khmeimim,17/35,35,348,9730,9730,9730,9730,47,43,,Public real-world data; verify
Nevada,Nellis AFB,03L/21R,03L,34,10050,10050,10050,10050,1870,1873,,Public real-world data; verify
Nevada,Nellis AFB,03L/21R,21R,214,10050,10050,10050,10050,1873,1870,,Public real-world data; verify
Nevada,Creech AFB,08/26,08,76,7999,7999,7999,7999,3135,3135,,Public real-world data; verify
Nevada,Creech AFB,08/26,26,256,7999,7999,7999,7999,3135,3135,,Public real-world data; verify
Nevada,Henderson Executive,17R/35L,17R,173,6501,6501,6501,6501,2492,2492,,Public real-world data; verify
Nevada,Henderson Executive,17R/35L,35L,353,6501,6501,6501,6501,2492,2492,,Public real-world data; verify
Nevada,North Las Vegas,12R/30L,12R,120,5004,5004,5004,5004,2205,2205,,Public real-world data; verify
Nevada,North Las Vegas,12R/30L,30L,300,5004,5004,5004,5004,2205,2205,,Public real-world data; verify
Nevada,McCarran Intl (Harry Reid),08L/26R,08L,85,14712,14712,14712,14712,2181,2181,,Public real-world data; verify
Nevada,McCarran Intl (Harry Reid),08L/26R,26R,265,14712,14712,14712,14712,2181,2181,,Public real-world data; verify
Marianas,Andersen AFB,06L/24R,06L,62,11000,11000,11000,11000,610,631,,Public real-world data; verify
Marianas,Andersen AFB,06L/24R,24R,242,11000,11000,11000,11000,631,610,,Public real-world data; verify
Marianas,Antonio B. Won Pat Intl,06L/24R,06L,62,10018,10018,10018,10018,298,298,,Public real-world data; verify
Marianas,Antonio B. Won Pat Intl,06L/24R,24R,242,10018,10018,10018,10018,298,298,,Public real-world data; verify
Marianas,Saipan Intl,07/25,07,72,8700,8700,8700,8700,215,215,,Public real-world data; verify
Marianas,Saipan Intl,07/25,25,252,8700,8700,8700,8700,215,215,,Public real-world data; verify
Sinai,Cairo Intl,05R/23L,05R,52,13100,13100,13100,13100,382,455,,Public real-world data; verify
Sinai,Cairo Intl,05R/23L,23L,232,13100,13100,13100,13100,455,382,,Public real-world data; verify
Sinai,Almaza AFB,18/36,18,180,9860,9860,9860,9860,151,151,,Public real-world data; verify
Sinai,Almaza AFB,18/36,36,360,9860,9860,9860,9860,151,151,,Public real-world data; verify
Sinai,El Arish Intl,10/28,10,99,9840,9840,9840,9840,121,121,,Public real-world data; verify
Sinai,El Arish Intl,10/28,28,279,9840,9840,9840,9840,121,121,,Public real-world data; verify
South Atlantic,Mount Pleasant,10/28,10,97,10000,10000,10000,10000,246,247,,Public real-world data; verify
South Atlantic,Mount Pleasant,10/28,28,277,10000,10000,10000,10000,247,246,,Public real-world data; verify
South Atlantic,Stanley,09/27,09,89,4100,4100,4100,4100,75,75,,Public real-world data; verify
South Atlantic,Stanley,09/27,27,269,4100,4100,4100,4100,75,75,,Public real-world data; verify
Normandy 2.0,Carpiquet,13/31,13,130,6234,6234,6234,6234,256,272,,Public real-world data; verify
Normandy 2.0,Carpiquet,13/31,31,310,6234,6234,6234,6234,272,256,,Public real-world data; verify
The Channel,Manston,10/28,10,100,9010,9010,9010,9010,178,178,,Public real-world data; verify
The Channel,Manston,10/28,28,280,9010,9010,9010,9010,178,178,,Public real-world data; verify
"""

PERF_F14B_CSV = """model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note
F-14B,20,Military,60000,0,0,120,132,141,154,4300,5200,PLACEHOLDER_EST
F-14B,20,Military,60000,0,30,123,135,144,157,4700,5600,PLACEHOLDER_EST
F-14B,20,Military,60000,5000,0,123,136,146,158,5200,6100,PLACEHOLDER_EST
F-14B,20,Military,60000,5000,30,126,139,149,162,5600,6600,PLACEHOLDER_EST
F-14B,20,Military,65000,0,0,123,136,146,159,4800,5700,PLACEHOLDER_EST
F-14B,20,Military,65000,0,30,126,139,150,162,5200,6200,PLACEHOLDER_EST
F-14B,20,Military,65000,5000,0,126,139,150,162,5600,6600,PLACEHOLDER_EST
F-14B,20,Military,65000,5000,30,129,143,154,167,6000,7100,PLACEHOLDER_EST
F-14B,20,Military,70000,0,0,126,140,151,164,5300,6300,PLACEHOLDER_EST
F-14B,20,Military,70000,0,30,129,143,155,168,5800,6900,PLACEHOLDER_EST
F-14B,20,Military,70000,5000,0,130,144,156,169,6200,7400,PLACEHOLDER_EST
F-14B,20,Military,70000,5000,30,133,147,160,173,6700,7900,PLACEHOLDER_EST
F-14B,20,Military,74300,0,0,128,142,154,167,5600,6700,PLACEHOLDER_EST
F-14B,20,Military,74300,0,30,131,145,158,171,6100,7300,PLACEHOLDER_EST
F-14B,20,Military,74300,5000,0,133,147,160,173,6600,7900,PLACEHOLDER_EST
F-14B,20,Military,74300,5000,30,136,151,164,178,7100,8500,PLACEHOLDER_EST
F-14B,20,Afterburner,60000,0,0,118,130,139,152,3900,4200,PLACEHOLDER_EST
F-14B,20,Afterburner,60000,0,30,120,132,141,154,4200,4500,PLACEHOLDER_EST
F-14B,20,Afterburner,60000,5000,0,120,133,143,156,4400,4700,PLACEHOLDER_EST
F-14B,20,Afterburner,60000,5000,30,122,135,145,158,4700,5000,PLACEHOLDER_EST
F-14B,20,Afterburner,65000,0,0,121,133,143,156,4200,4500,PLACEHOLDER_EST
F-14B,20,Afterburner,65000,0,30,123,136,146,159,4500,4900,PLACEHOLDER_EST
F-14B,20,Afterburner,65000,5000,0,123,136,147,160,4800,5200,PLACEHOLDER_EST
F-14B,20,Afterburner,65000,5000,30,125,138,149,162,5200,5600,PLACEHOLDER_EST
F-14B,20,Afterburner,70000,0,0,124,137,148,161,4500,4900,PLACEHOLDER_EST
F-14B,20,Afterburner,70000,0,30,126,139,151,164,4900,5400,PLACEHOLDER_EST
F-14B,20,Afterburner,70000,5000,0,127,140,152,165,5300,5900,PLACEHOLDER_EST
F-14B,20,Afterburner,70000,5000,30,129,143,155,168,5800,6500,PLACEHOLDER_EST
F-14B,20,Afterburner,74300,0,0,126,139,151,164,4800,5300,PLACEHOLDER_EST
F-14B,20,Afterburner,74300,0,30,128,141,153,167,5200,5800,PLACEHOLDER_EST
F-14B,20,Afterburner,74300,5000,0,130,143,156,169,5700,6400,PLACEHOLDER_EST
F-14B,20,Afterburner,74300,5000,30,132,145,158,172,6200,7000,PLACEHOLDER_EST
F-14B,40,Military,60000,0,0,116,128,137,149,4200,5100,PLACEHOLDER_EST
F-14B,40,Military,60000,0,30,118,130,139,152,4500,5500,PLACEHOLDER_EST
F-14B,40,Military,60000,5000,0,118,131,141,153,5000,6000,PLACEHOLDER_EST
F-14B,40,Military,60000,5000,30,120,133,143,156,5400,6500,PLACEHOLDER_EST
F-14B,40,Military,70000,0,0,123,136,146,158,5200,6200,PLACEHOLDER_EST
F-14B,40,Military,70000,0,30,126,139,149,162,5700,6900,PLACEHOLDER_EST
F-14B,40,Military,70000,5000,0,128,141,152,165,6200,7600,PLACEHOLDER_EST
F-14B,40,Military,70000,5000,30,131,144,156,169,6800,8300,PLACEHOLDER_EST
F-14B,40,Afterburner,60000,0,0,114,126,135,148,3800,4100,PLACEHOLDER_EST
F-14B,40,Afterburner,60000,0,30,116,128,137,150,4100,4400,PLACEHOLDER_EST
F-14B,40,Afterburner,60000,5000,0,116,129,139,151,4300,4700,PLACEHOLDER_EST
F-14B,40,Afterburner,60000,5000,30,118,131,141,154,4600,5100,PLACEHOLDER_EST
F-14B,40,Afterburner,70000,0,0,121,134,144,156,4500,5400,PLACEHOLDER_EST
F-14B,40,Afterburner,70000,0,30,123,136,147,159,5000,6100,PLACEHOLDER_EST
F-14B,40,Afterburner,70000,5000,0,124,137,148,160,5400,6800,PLACEHOLDER_EST
F-14B,40,Afterburner,70000,5000,30,126,139,151,163,6000,7500,PLACEHOLDER_EST
"""

PERF_F14D_CSV = """model,flap_deg,thrust,gw_lbs,press_alt_ft,oat_c,Vs_kt,V1_kt,Vr_kt,V2_kt,ASD_ft,AGD_ft,note
F-14D,20,Military,60000,0,0,121,133,142,155,4200,5100,PLACEHOLDER_EST
F-14D,20,Military,60000,0,30,123,135,145,157,4600,5500,PLACEHOLDER_EST
F-14D,20,Military,60000,5000,0,123,136,146,158,5100,6000,PLACEHOLDER_EST
F-14D,20,Military,60000,5000,30,126,139,149,162,5500,6500,PLACEHOLDER_EST
F-14D,20,Military,65000,0,0,124,137,147,160,4700,5600,PLACEHOLDER_EST
F-14D,20,Military,65000,0,30,127,140,151,163,5200,6100,PLACEHOLDER_EST
F-14D,20,Military,65000,5000,0,127,140,151,163,5600,6600,PLACEHOLDER_EST
F-14D,20,Military,65000,5000,30,130,144,155,168,6000,7100,PLACEHOLDER_EST
F-14D,20,Military,70000,0,0,127,141,152,165,5200,6100,PLACEHOLDER_EST
F-14D,20,Military,70000,0,30,130,144,156,169,5700,6800,PLACEHOLDER_EST
F-14D,20,Military,70000,5000,0,131,145,157,170,6200,7400,PLACEHOLDER_EST
F-14D,20,Military,70000,5000,30,134,148,161,174,6700,8000,PLACEHOLDER_EST
F-14D,20,Military,74300,0,0,129,143,155,168,5500,6500,PLACEHOLDER_EST
F-14D,20,Military,74300,0,30,132,146,159,172,6000,7100,PLACEHOLDER_EST
F-14D,20,Military,74300,5000,0,134,148,161,174,6500,7700,PLACEHOLDER_EST
F-14D,20,Military,74300,5000,30,137,151,165,178,7000,8300,PLACEHOLDER_EST
F-14D,20,Afterburner,60000,0,0,119,131,140,153,3800,4000,PLACEHOLDER_EST
F-14D,20,Afterburner,60000,0,30,121,133,142,155,4100,4300,PLACEHOLDER_EST
F-14D,20,Afterburner,60000,5000,0,121,134,144,156,4300,4600,PLACEHOLDER_EST
F-14D,20,Afterburner,60000,5000,30,123,136,146,159,4600,4900,PLACEHOLDER_EST
F-14D,20,Afterburner,65000,0,0,122,134,144,157,4100,4300,PLACEHOLDER_EST
F-14D,20,Afterburner,65000,0,30,124,136,147,160,4400,4700,PLACEHOLDER_EST
F-14D,20,Afterburner,65000,5000,0,124,137,148,161,4700,5100,PLACEHOLDER_EST
F-14D,20,Afterburner,65000,5000,30,126,139,150,163,5100,5600,PLACEHOLDER_EST
F-14D,20,Afterburner,70000,0,0,125,138,149,162,4400,4700,PLACEHOLDER_EST
F-14D,20,Afterburner,70000,0,30,127,140,152,165,4800,5200,PLACEHOLDER_EST
F-14D,20,Afterburner,70000,5000,0,127,141,153,166,5200,5700,PLACEHOLDER_EST
F-14D,20,Afterburner,70000,5000,30,129,143,155,168,5700,6300,PLACEHOLDER_EST
F-14D,20,Afterburner,74300,0,0,127,140,152,165,4700,5100,PLACEHOLDER_EST
F-14D,20,Afterburner,74300,0,30,129,142,154,167,5100,5600,PLACEHOLDER_EST
F-14D,20,Afterburner,74300,5000,0,129,142,155,168,5600,6200,PLACEHOLDER_EST
F-14D,20,Afterburner,74300,5000,30,131,145,157,171,6100,6800,PLACEHOLDER_EST
F-14D,40,Military,60000,0,0,116,128,137,149,4100,5000,PLACEHOLDER_EST
F-14D,40,Military,60000,0,30,118,130,139,152,4400,5400,PLACEHOLDER_EST
F-14D,40,Military,60000,5000,0,118,131,141,153,4900,5900,PLACEHOLDER_EST
F-14D,40,Military,60000,5000,30,120,133,143,156,5300,6400,PLACEHOLDER_EST
F-14D,40,Military,70000,0,0,123,136,146,158,5100,6100,PLACEHOLDER_EST
F-14D,40,Military,70000,0,30,126,139,149,162,5600,6800,PLACEHOLDER_EST
F-14D,40,Military,70000,5000,0,127,140,151,164,6100,7500,PLACEHOLDER_EST
F-14D,40,Military,70000,5000,30,130,143,155,168,6700,8200,PLACEHOLDER_EST
F-14D,40,Afterburner,60000,0,0,114,126,135,148,3700,4000,PLACEHOLDER_EST
F-14D,40,Afterburner,60000,0,30,116,128,137,150,4000,4300,PLACEHOLDER_EST
F-14D,40,Afterburner,60000,5000,0,116,129,139,151,4200,4600,PLACEHOLDER_EST
F-14D,40,Afterburner,60000,5000,30,118,131,141,154,4500,5000,PLACEHOLDER_EST
F-14D,40,Afterburner,70000,0,0,121,134,144,156,4400,5200,PLACEHOLDER_EST
F-14D,40,Afterburner,70000,0,30,123,136,147,159,4900,5900,PLACEHOLDER_EST
F-14D,40,Afterburner,70000,5000,0,123,136,147,160,5300,6600,PLACEHOLDER_EST
F-14D,40,Afterburner,70000,5000,30,125,138,149,162,5900,7300,PLACEHOLDER_EST
"""

# ==========================
# Utility functions
# ==========================

def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return field_elev_ft + (29.92 - qnh_inhg) * 1000.0

def headwind_component(knots_wind: float, wind_dir_deg: float, rwy_heading_deg: float) -> float:
    delta = math.radians((wind_dir_deg - rwy_heading_deg) % 360)
    return knots_wind * math.cos(delta)

def apply_wind_slope(distance_ft: float, headwind_kn: float, slope_pct: float) -> float:
    adj = distance_ft
    adj *= 1.0 - 0.01 * min(0.5, max(-0.5, headwind_kn / 2.0))
    adj *= 1.0 + 0.20 * max(0.0, slope_pct)
    return adj

# ==========================
# Data loaders (from embedded strings)
# ==========================
@st.cache_data
def load_runways() -> pd.DataFrame:
    df = pd.read_csv(StringIO(RUNWAYS_CSV))
    df["rw_key"] = df["airport_name"] + " — " + df["runway_pair"].astype(str) + "/" + df["runway_end"].astype(str)
    df["runway_label"] = df["airport_name"] + " " + df["runway_end"].astype(str) + " (" + df["runway_pair"].astype(str) + ")"
    return df

@st.cache_data
def load_perf() -> pd.DataFrame:
    p_b = pd.read_csv(StringIO(PERF_F14B_CSV))
    p_d = pd.read_csv(StringIO(PERF_F14D_CSV))
    perf = pd.concat([p_b, p_d], ignore_index=True)
    perf["model"] = perf["model"].str.upper()
    perf["thrust"] = perf["thrust"].str.upper()
    return perf

# ==========================
# Core performance logic
# ==========================
@dataclass
class Inputs:
    model: str
    theatre: str
    airport: str
    runway_label: str
    heading_deg: float
    tora_ft: float
    toda_ft: float
    asda_ft: float
    elev_ft: float
    slope_pct: float
    shorten_ft: float
    oat_c: float
    qnh_inhg: float
    wind_knots: float
    wind_dir_deg: float
    gw_lbs: float
    cg_percent_mac: Optional[float]
    flap_sel: str
    thrust_sel: str

@dataclass
class PerfResult:
    v1: float
    vr: float
    v2: float
    vs: float
    flap_setting: str
    thrust_setting: str
    req_distance_ft: float
    avail_distance_ft: float
    limiting: str

def auto_flaps(gw_lbs: float, pa_ft: float, oat_c: float) -> str:
    if gw_lbs >= 70000 or pa_ft > 3000 or oat_c > 30:
        return "FULL"
    elif gw_lbs >= 62000:
        return "MAN"
    return "UP"

def auto_thrust(oat_c: float, pa_ft: float) -> str:
    if pa_ft > 3000 or oat_c > 30:
        return "MIL"
    return "MIL"

def flap_to_deg(flap_sel: str) -> int:
    s = flap_sel.upper()
    if s in ("UP", "0"):
        return 0
    if s in ("MAN", "MANEUVERING", "20"):
        return 20
    return 40

def thrust_label_to_table_mode(label: str) -> str:
    lab = label.upper()
    if lab in ("MIL", "MILITARY"): return "MILITARY"
    if lab in ("AB", "AFTERBURNER"): return "AFTERBURNER"
    return "MILITARY"

def nearest_perf_row(perf: pd.DataFrame, model: str, flap_deg: int, thrust_mode: str,
                     gw_lbs: float, pa_ft: float, oat_c: float) -> pd.Series:
    sub = perf[(perf["model"] == model.upper()) &
               (perf["flap_deg"] == flap_deg) &
               (perf["thrust"].str.upper() == thrust_mode.upper())]
    if sub.empty:
        sub = perf[(perf["model"] == model.upper()) & (perf["flap_deg"] == flap_deg)]
    if sub.empty:
        raise ValueError("No performance rows match model/flap.")
    sub = sub.assign(
        d_w = (sub["gw_lbs"] - gw_lbs).abs(),
        d_pa = (sub["press_alt_ft"] - pa_ft).abs(),
        d_t  = (sub["oat_c"] - oat_c).abs(),
    )
    return sub.sort_values(["d_w", "d_pa", "d_t"]).iloc[0]

def compute_performance(inp: Inputs, perfdb: pd.DataFrame) -> PerfResult:
    pa_ft = pressure_altitude_ft(inp.elev_ft, inp.qnh_inhg)
    headwind = headwind_component(inp.wind_knots, inp.wind_dir_deg, inp.heading_deg)

    flap = inp.flap_sel if inp.flap_sel != "Auto" else auto_flaps(inp.gw_lbs, pa_ft, inp.oat_c)
    thrust = inp.thrust_sel if inp.thrust_sel != "Auto" else auto_thrust(inp.oat_c, pa_ft)

    flap_deg = flap_to_deg(flap)
    if flap_deg == 0:
        # Promote to 20 if no 0° rows exist
        if not ((perfdb["flap_deg"] == 0) & (perfdb["model"] == inp.model.upper())).any():
            flap_deg = 20
            flap = "MAN"

    thrust_mode = thrust_label_to_table_mode(thrust)

    row = nearest_perf_row(perfdb, inp.model, flap_deg, thrust_mode, inp.gw_lbs, pa_ft, inp.oat_c)

    V1 = float(row["V1_kt"])
    Vr = float(row["Vr_kt"])
    V2 = float(row["V2_kt"])
    Vs = float(row.get("Vs_kt", np.nan))

    ASD = apply_wind_slope(float(row["ASD_ft"]), headwind, inp.slope_pct)
    AGD = apply_wind_slope(float(row["AGD_ft"]), headwind, inp.slope_pct)

    req_ft = max(ASD, AGD)
    avail_ft = max(0.0, float(inp.tora_ft) - float(inp.shorten_ft))
    limiting = "ASD" if ASD >= AGD else "AGD"

    return PerfResult(V1, Vr, V2, Vs, flap, thrust, req_ft, avail_ft, limiting)

# ==========================
# UI
# ==========================
st.title("DCS F‑14B/D Takeoff Performance")

rwy_df = load_runways()
perfdb = load_perf()

with st.sidebar:
    st.header("Runway")
    theatre = st.selectbox("DCS Map", sorted(rwy_df["map"].unique()))
    df_t = rwy_df[rwy_df["map"] == theatre]
    airport = st.selectbox("Airport", sorted(df_t["airport_name"].unique()))
    df_a = df_t[df_t["airport_name"] == airport]
    rw = st.selectbox("Runway End", list(df_a["runway_label"]))
    row_rwy = df_a[df_a["runway_label"] == rw].iloc[0]

    st.metric("TORA (ft)", f"{int(row_rwy['tora_ft']):,}")
    st.metric("TODA (ft)", f"{int(row_rwy['toda_ft']):,}")
    st.metric("ASDA (ft)", f"{int(row_rwy['asda_ft']):,}")

    shorten_ft = st.number_input("Manually shorten available runway (ft)", min_value=0.0, value=0.0, step=100.0)

    st.header("Weather")
    oat_c = st.number_input("OAT (°C)", value=15.0, step=1.0)
    qnh_mode = st.selectbox("QNH Input", ["inHg", "hPa"], index=0)
    if qnh_mode == "inHg":
        qnh_inhg = st.number_input("QNH (inHg)", value=29.92, step=0.01, format="%.2f")
    else:
        qnh_hpa = st.number_input("QNH (hPa)", value=1013.0, step=1.0)
        qnh_inhg = hpa_to_inhg(qnh_hpa)
    wind_knots = st.number_input("Surface wind (kts)", value=0.0, step=1.0)
    wind_dir_deg = st.number_input("Wind direction (deg true)", value=float(row_rwy["heading_deg"]), step=1.0, min_value=0.0, max_value=359.9)

    st.header("Weight & Config")
    model = st.selectbox("Model", ["F-14B", "F-14D"], index=0)
    gw_lbs = st.number_input("Gross Weight (lb)", value=70000.0, step=500.0)
    cg_percent_mac = st.number_input("CG (%MAC) — optional", value=25.0, step=0.5)

    flap_sel = st.selectbox("Takeoff Flaps", ["Auto", "UP", "MAN", "FULL"], index=0)
    thrust_sel = st.selectbox("Takeoff Thrust", ["Auto", "MIL", "AB"], index=0)

if st.button("Compute Takeoff Performance", type="primary"):
    inp = Inputs(
        model=model,
        theatre=theatre,
        airport=airport,
        runway_label=rw,
        heading_deg=float(row_rwy["heading_deg"]),
        tora_ft=float(row_rwy["tora_ft"]),
        toda_ft=float(row_rwy["toda_ft"]),
        asda_ft=float(row_rwy["asda_ft"]),
        elev_ft=float(row_rwy["threshold_elev_ft"]),
        slope_pct=float(row_rwy.get("slope_percent", 0.0) or 0.0),
        shorten_ft=float(shorten_ft),
        oat_c=float(oat_c), qnh_inhg=float(qnh_inhg),
        wind_knots=float(wind_knots), wind_dir_deg=float(wind_dir_deg),
        gw_lbs=float(gw_lbs), cg_percent_mac=float(cg_percent_mac),
        flap_sel=flap_sel, thrust_sel=thrust_sel,
    )
    try:
        perf = compute_performance(inp, perfdb)
        ok = perf.req_distance_ft <= perf.avail_distance_ft

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("V‑Speeds")
            st.metric("V1 (kt)", f"{perf.v1:.0f}")
            st.metric("Vr (kt)", f"{perf.vr:.0f}")
            st.metric("V2 (kt)", f"{perf.v2:.0f}")
            if not np.isnan(perf.vs):
                st.metric("Vs (kt)", f"{perf.vs:.0f}")
        with col2:
            st.subheader("Settings")
            st.metric("Flaps", perf.flap_setting)
            st.metric("Thrust", perf.thrust_setting)
            st.caption("Trim lookup can be added later via a trim table.")
        with col3:
            st.subheader("Runway")
            st.metric("Req. Distance (ft)", f"{perf.req_distance_ft:.0f}", help="max(ASD, AGD) with wind/slope adjustments")
            st.metric("Avail. Distance (ft)", f"{perf.avail_distance_ft:.0f}")
            st.metric("Limiting", perf.limiting)

        st.markdown("✅ **" + ("TAKEOFF POSSIBLE" if ok else "TAKEOFF NOT PERMITTED") + "** — using TORA as available distance.")
        if not ok:
            st.warning("Try AB, more flap, lower weight, cooler conditions, or a longer runway.")

        with st.expander("Debug details"):
            st.json(vars(inp))

    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.stop()
else:
    st.info("Set inputs and click **Compute Takeoff Performance**.")
