# natops_transcriber.py — Human-in-the-loop NATOPS table transcriber
# Shows a NATOPS page image on the left and a spreadsheet editor on the right,
# so you (or a helper) can key values safely into structured CSVs.
#
# Outputs two CSVs:
#  - f14_landing_natops.csv
#  - f14_cruise_natops.csv
#
# This tool does not guess numbers from scans; it ensures QA by pairing image preview
# with explicit data entry and validation.
#
# Usage:
#  1) Run this app in Streamlit.
#  2) Upload a page image (PNG/JPG) for the table you are transcribing.
#  3) Choose the table type (Landing/Cruise) and fill rows in the editor.
#  4) Download the CSVs when finished.
from __future__ import annotations
import io, json
from typing import List, Dict
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="NATOPS Transcriber — F-14B", layout="wide")
st.title("NATOPS Transcriber — F-14B (Landing & Cruise)")

with st.sidebar:
    st.markdown("**Workflow**")
    st.markdown("1. Upload the page image (PNG/JPG).")
    st.markdown("2. Select **Landing** or **Cruise** mode.")
    st.markdown("3. Enter values row-by-row in the grid.")
    st.markdown("4. Repeat for other pages as needed.")
    st.markdown("5. Download CSVs.")

tab_landing, tab_cruise = st.tabs(["Landing Tables", "Cruise Tables"])

def _download_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

with tab_landing:
    st.subheader("Landing — Approach Speeds & Ground Roll")
    col_img, col_edit = st.columns([1, 1.2], gap="large")

    with col_img:
        img_file = st.file_uploader("Upload landing page image (PNG/JPG)", type=["png","jpg","jpeg"], key="landing_img")
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Landing table reference", use_column_width=True)
        st.caption("Tip: You can upload multiple images sequentially and append rows below.")

    with col_edit:
        st.markdown("**Landing data editor**")
        default_cols = ["source_doc","source_page","gross_weight_lbs","flap_setting","aoa_units","vref_kts","ground_roll_ft_unfactored","runway_condition","notes"]
        df_landing = st.session_state.get("df_landing", pd.DataFrame(columns=default_cols))
        df_landing = st.data_editor(
            df_landing,
            num_rows="dynamic",
            use_container_width=True,
            key="landing_editor",
            column_config={
                "flap_setting": st.column_config.SelectboxColumn(options=["UP","DOWN"], help="NATOPS typically uses Flaps UP or Flaps DOWN for landing tables."),
                "runway_condition": st.column_config.SelectboxColumn(options=["DRY","WET","OTHER"], help="If the table is dry-only, use DRY."),
                "aoa_units": st.column_config.NumberColumn(format="%.1f"),
                "gross_weight_lbs": st.column_config.NumberColumn(format="%.0f"),
                "vref_kts": st.column_config.NumberColumn(format="%.0f"),
                "ground_roll_ft_unfactored": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.session_state["df_landing"] = df_landing
        st.divider()
        _download_button(df_landing, "f14_landing_natops.csv", "Download landing CSV")

with tab_cruise:
    st.subheader("Cruise — Optimum Altitude & Cruise Mach by Weight / Drag Index")
    col_img2, col_edit2 = st.columns([1, 1.2], gap="large")

    with col_img2:
        img_file2 = st.file_uploader("Upload cruise page image (PNG/JPG)", type=["png","jpg","jpeg"], key="cruise_img")
        if img_file2:
            img2 = Image.open(img_file2)
            st.image(img2, caption="Cruise table reference", use_column_width=True)
        st.caption("Tip: Use one row per (weight, DI) cell with its optimum altitude and cruise Mach.")

    with col_edit2:
        st.markdown("**Cruise data editor**")
        cruise_cols = ["source_doc","source_page","gross_weight_lbs","drag_index","optimum_alt_ft","optimum_mach","notes"]
        df_cruise = st.session_state.get("df_cruise", pd.DataFrame(columns=cruise_cols))
        df_cruise = st.data_editor(
            df_cruise,
            num_rows="dynamic",
            use_container_width=True,
            key="cruise_editor",
            column_config={
                "drag_index": st.column_config.NumberColumn(format="%.0f"),
                "gross_weight_lbs": st.column_config.NumberColumn(format="%.0f"),
                "optimum_alt_ft": st.column_config.NumberColumn(format="%.0f"),
                "optimum_mach": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.session_state["df_cruise"] = df_cruise
        st.divider()
        _download_button(df_cruise, "f14_cruise_natops.csv", "Download cruise CSV")
