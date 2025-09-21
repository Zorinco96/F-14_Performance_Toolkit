# test_app.py — Calibration Harness for Thrust Exponent "m"
# Purpose:
#   Fit thrust→ground-roll exponent m per flap (UP/MAN/FULL) from simple DCS runs.
#   Data in: airport, runway, weight_lb, flaps_deg, derate_pct, oat_c, qnh_inhg,
#            headwind_kt, ground_roll_ft, notes
#   Method:  For each flap, take the MIL (100%) baseline ground-roll (GR_MIL),
#            and each observed derate point (derate_pct -> mult), then:
#              m_i = ln(GR_MIL / GR_obs) / ln(mult)
#            Aggregate per flap via robust median -> m_up, m_man, m_full.
#
# Outputs:
#   - On-screen table with per-point m and a robust median per flap.
#   - Download button for a merged derate_config.json that includes:
#       "thrust_exponent_m": {"0": m_up, "20": m_man, "35": m_full}
#
# Notes:
#   - This tool does NOT write to your repo automatically. It generates a merged JSON
#     for you to copy/paste into GitHub (/data/derate_config.json).
#   - Floors and safety policies are untouched here.
#
# Usage:
#   1) Run locally or on Streamlit Cloud side-by-side with main app.
#   2) Paste your quick sheet or upload CSV.
#   3) Review fits, download merged JSON, paste into GitHub.

from __future__ import annotations
import io
import json
import math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# --- Paths / Load existing config ---
DEFAULT_CFG_PATH = "data/derate_config.json"

def load_existing_config(path: str = DEFAULT_CFG_PATH) -> Dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # Minimal sensible defaults (your floors & policies are elsewhere)
        return {
            "allow_ab": False,
            "min_idle_ff_pph": 1200,
            "thrust_exponent_m": {"0": 0.75, "20": 0.75, "35": 0.75},
            "min_pct_by_flap_deg": {"0": 85, "20": 90, "35": 96},
            "safety": {"runway_factor": 1.10},  # 10% factor on ASDR/TODR (your new policy)
        }

def merge_config(existing: Dict, fitted_m: Dict[str, float]) -> Dict:
    cfg = dict(existing) if isinstance(existing, dict) else {}
    # Ensure keys exist
    cfg.setdefault("thrust_exponent_m", {})
    # Merge m
    for k, v in fitted_m.items():
        if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            cfg["thrust_exponent_m"][k] = float(v)
    # Keep your floors if they exist; otherwise set your locked values
    cfg.setdefault("min_pct_by_flap_deg", {"0": 85, "20": 90, "35": 96})
    # Keep your 10% runway factor as explicit policy knob (used by main app/core)
    safety = cfg.get("safety", {})
    safety.setdefault("runway_factor", 1.10)
    cfg["safety"] = safety
    return cfg

# --- UI ---
st.set_page_config(page_title="F-14B Calibration Harness — m Fit", layout="wide")
st.title("F-14B Calibration Harness — Fit Thrust Exponent m")

st.markdown("""
Paste your DCS test runs or upload a CSV with the following columns:

```
airport,runway,weight_lb,flaps_deg,derate_pct,oat_c,qnh_inhg,headwind_kt,ground_roll_ft,notes
```

**Rules for clean fits:**
- Use a **MIL (100%)** baseline for each flap state you’re fitting (UP=0, MAN=20, FULL=35).
- Keep **weight, runway, and wind** constant within a flap batch.
- Vary **derate_pct**: e.g., UP: 100, 95, 90, 85; MAN: 100, 95, 90; FULL: 100, 96 (and 98 optional).
""")

# Input options
tab_paste, tab_upload = st.tabs(["Paste data", "Upload CSV"])

with tab_paste:
    sample = """airport,runway,weight_lb,flaps_deg,derate_pct,oat_c,qnh_inhg,headwind_kt,ground_roll_ft,notes
BATUMI,13,70000,0,100,15,29.92,0,_____,MIL baseline
BATUMI,13,70000,0,95,15,29.92,0,_____, 
BATUMI,13,70000,0,90,15,29.92,0,_____, 
BATUMI,13,70000,0,85,15,29.92,0,_____, 
BATUMI,13,70000,20,100,15,29.92,0,_____,MAN baseline
BATUMI,13,70000,20,95,15,29.92,0,_____, 
BATUMI,13,70000,20,90,15,29.92,0,_____, 
BATUMI,13,70000,35,100,15,29.92,0,_____,FULL baseline
BATUMI,13,70000,35,98,15,29.92,0,_____,optional
BATUMI,13,70000,35,96,15,29.92,0,_____,floor
"""
    txt = st.text_area("Paste CSV rows here", sample, height=220)
    try:
        df_paste = pd.read_csv(io.StringIO(txt.strip()))
    except Exception:
        df_paste = pd.DataFrame()

with tab_upload:
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f is not None:
        try:
            df_upload = pd.read_csv(f)
        except Exception:
            df_upload = pd.DataFrame()
    else:
        df_upload = pd.DataFrame()

# Choose source
df_candidates = []
if not df_paste.empty:
    df_candidates.append(("Pasted", df_paste))
if not df_upload.empty:
    df_candidates.append(("Uploaded", df_upload))

if not df_candidates:
    st.info("Paste rows or upload a CSV to begin.")
    st.stop()

src_name, df = df_candidates[0]
st.subheader(f"Input data · {src_name}")
st.dataframe(df, use_container_width=True)

# Basic validation / normalization
required_cols = {
    "airport","runway","weight_lb","flaps_deg","derate_pct",
    "oat_c","qnh_inhg","headwind_kt","ground_roll_ft"
}
missing = required_cols - set(df.columns.str.lower())
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Normalize column case
df.columns = [c.strip().lower() for c in df.columns]
# Drop rows without essential fields
df = df.dropna(subset=["flaps_deg","derate_pct","ground_roll_ft"])

# Helper: compute per-point m given flap batch and MIL baseline
def fit_m_for_flap(batch: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    # Use MIL (100%) as baseline
    mil_rows = batch[np.isclose(batch["derate_pct"], 100.0)]
    if mil_rows.empty:
        return batch.assign(m_fit=np.nan), float("nan")
    # Prefer single baseline; average if multiple
    GR_MIL = float(np.nanmedian(mil_rows["ground_roll_ft"].astype(float)))
    out = batch.copy()
    m_list = []
    for i, row in out.iterrows():
        derate = float(row["derate_pct"])
        GR_obs = float(row["ground_roll_ft"])
        if derate <= 0 or GR_obs <= 0 or derate == 100.0:
            out.loc[i, "m_fit"] = np.nan
            continue
        mult = derate / 100.0
        try:
            m_i = math.log(GR_MIL / GR_obs) / math.log(mult)
        except Exception:
            m_i = float("nan")
        out.loc[i, "m_fit"] = m_i
        if not (math.isnan(m_i) or math.isinf(m_i)):
            m_list.append(m_i)
    m_med = float(np.nanmedian(m_list)) if m_list else float("nan")
    return out, m_med

st.subheader("Fits")
results = []
fitted = {"0": float("nan"), "20": float("nan"), "35": float("nan")}

for flap_key, flap_deg in [("0", 0), ("20", 20), ("35", 35)]:
    sub = df[np.isclose(df["flaps_deg"].astype(float), float(flap_deg))]
    with st.expander(f"Flaps {flap_deg}°", expanded=True):
        if sub.empty:
            st.info("No rows for this flap.")
            st.write(pd.DataFrame(columns=list(df.columns)+["m_fit"]))
        else:
            out, m_med = fit_m_for_flap(sub)
            st.dataframe(out, use_container_width=True)
            st.markdown(f"**Robust median m (Flaps {flap_deg}°):** `{m_med:.4f}`" if not math.isnan(m_med) else "**Insufficient data**")
            fitted[flap_key] = m_med
            results.append((flap_deg, m_med))

# Show summary + merged config
st.subheader("Summary")
summary_df = pd.DataFrame({
    "Flaps_deg": [0, 20, 35],
    "m_fitted": [fitted["0"], fitted["20"], fitted["35"]],
})
st.table(summary_df)

existing_cfg = load_existing_config()
merged_cfg = merge_config(existing_cfg, {
    "0": fitted["0"],
    "20": fitted["20"],
    "35": fitted["35"],
})

st.subheader("Write-back / Download")
st.code(json.dumps(merged_cfg, indent=2), language="json")
st.download_button(
    label="Download merged derate_config.json",
    data=json.dumps(merged_cfg, indent=2).encode("utf-8"),
    file_name="derate_config.json",
    mime="application/json"
)

st.caption("Paste this into /data/derate_config.json in your GitHub repo.")
