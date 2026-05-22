import streamlit as st
import plotly.express as px
from ui_utils import setup_page, load_shap_csv, render_sidebar, CHART_LAYOUT
from config import Paths
from pathlib import Path

setup_page("Victor — Model Explainability")
# We only pass None since we don't need preds for the sidebar metrics on this page, or we can load them
from ui_utils import load_data
preds, _ = load_data()
render_sidebar(preds)

st.markdown("# Model Explainability")
st.markdown("_Understand why Victor classifies a request as a bot_")
st.divider()

shap_img_global = Paths.SHAP_GLOBAL_SUMMARY
shap_img_bar    = Paths.SHAP_FEATURE_BAR
shap_csv        = load_shap_csv()

if Path(shap_img_global).exists() or Path(shap_img_bar).exists():
    col1, col2 = st.columns(2)
    with col1:
        if Path(shap_img_global).exists():
            st.subheader("Global Feature Importance")
            st.image(shap_img_global, use_container_width=True)
    with col2:
        if Path(shap_img_bar).exists():
            st.subheader("Average Feature Impact")
            st.image(shap_img_bar, use_container_width=True)
else:
    st.warning("SHAP plots not found. Run `python explain.py` first.")

if shap_csv is not None:
    st.divider()
    st.subheader("SHAP Heatmap (top 200 rows)")
    shap_display = shap_csv.head(200)
    shap_renamed = shap_display.rename(
        columns={c: c.replace("shap_", "") for c in shap_display.columns}
    )
    fig_heat = px.imshow(
        shap_renamed.values.T,
        x=list(range(len(shap_renamed))),
        y=list(shap_renamed.columns),
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(x="Request Index", y="Feature", color="SHAP")
    )
    fig_heat.update_layout(**CHART_LAYOUT, height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()
st.subheader("What Each Feature Means")
explanations = {
    "ua_is_suspicious":        "User agent contains bot keywords (python-requests, curl, etc.)",
    "has_referer":             "Did the request include a Referer header? Humans typically send one.",
    "has_accept_lang":         "Was the Accept-Language header present? Bots often omit it.",
    "hit_secret_page":         "Did the IP visit the hidden honeypot endpoint /secret-data?",
    "ua_length":               "Length of the user agent string — bots tend to have shorter ones.",
    "time_gap_seconds":        "Time between requests — bots fire requests much faster.",
    "unique_pages_visited":    "Number of distinct pages visited — bots sweep more pages.",
    "total_requests_from_ip":  "Total number of requests made from this IP.",
}
for feat, desc in explanations.items():
    st.markdown(f"**`{feat}`** — {desc}")
