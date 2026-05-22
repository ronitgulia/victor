import streamlit as st
import pandas as pd
from ui_utils import setup_page, load_data, render_sidebar, FEATURE_COLS
from config_loader import Config

setup_page("Victor — Raw Data")
preds, features = load_data()

if preds is None:
    st.error("Data files not found! Run the pipeline first.")
    st.stop()

threshold = render_sidebar(preds)

st.markdown("# Raw Predictions Log")
st.markdown("_All requests with their scores and verdict_")
st.divider()

col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
with col_f1:
    filter_opt = st.radio("Show:", ["All", "Bots only", "Humans only"], horizontal=True)
with col_f2:
    min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)
with col_f3:
    max_rows = st.number_input(
        "Max rows:", 10, 10000,
        Config.get("dashboard.default_max_rows", 200)
    )

display_df = preds.copy()
display_df["victor_flag"] = (display_df["ensemble_score"] > threshold).astype(int)

if filter_opt == "Bots only":
    display_df = display_df[display_df["victor_flag"] == 1]
elif filter_opt == "Humans only":
    display_df = display_df[display_df["victor_flag"] == 0]

display_df = display_df[display_df["ensemble_score"] >= min_score]
display_df = display_df.sort_values("ensemble_score", ascending=False).head(max_rows)

cols_to_show = [c for c in ["ip", *FEATURE_COLS, "iso_score", "xgb_score",
                             "ensemble_score", "victor_flag"] if c in display_df.columns]
st.markdown(f"**{len(display_df):,} records shown**")
st.dataframe(display_df[cols_to_show].round(4).reset_index(drop=True),
             use_container_width=True, height=500)

csv = display_df.to_csv(index=False)
st.download_button("⬇ Download CSV", csv, "victor_predictions.csv", "text/csv")
