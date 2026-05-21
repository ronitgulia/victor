# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os
from datetime import datetime

st.set_page_config(
    page_title="Victor — Bot Detection",
    page_icon="⚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2d3436; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #2d3436 0%, #636e72 100%); border-right: 3px solid #0984e3; }
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] h2 { font-size: 1.6rem !important; margin-bottom: 8px !important; font-weight: 800 !important; }
[data-testid="stMetric"] { background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); border: 2px solid #e0e6ed; border-radius: 16px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); transition: all 0.3s ease; }
[data-testid="stMetric"]:hover { border-color: #0984e3; box-shadow: 0 12px 32px rgba(9,132,227,0.15); transform: translateY(-2px); }
[data-testid="stMetricLabel"] { color: #636e72 !important; font-size: 0.8rem !important; font-weight: 700 !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #2d3436 !important; font-size: 2.4rem !important; font-weight: 800 !important; }
h1, h2, h3 { color: #2d3436 !important; font-weight: 800 !important; }
h1 { font-size: 2.8rem !important; margin-top: 0 !important; }
h2 { font-size: 1.8rem !important; margin-top: 24px !important; }
hr { border-color: #dfe6e9 !important; margin: 24px 0 !important; }
.stButton > button { background: linear-gradient(135deg, #0984e3 0%, #0770d4 100%); color: white !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 12px 24px !important; box-shadow: 0 4px 12px rgba(9,132,227,0.3) !important; transition: all 0.3s ease !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: rgba(255,255,255,0.15) !important; border: 1px solid rgba(255,255,255,0.3) !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
.activity-feed { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-left: 4px solid #0984e3; border-radius: 8px; padding: 20px; margin: 12px 0; }
.activity-item { padding: 12px; margin: 10px 0; background: white; border-radius: 6px; border: 1px solid #e0e6ed; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#2d3436", family="system-ui, -apple-system, sans-serif", size=12),
    xaxis=dict(gridcolor="#e0e6ed", zerolinecolor="#e0e6ed"),
    yaxis=dict(gridcolor="#e0e6ed", zerolinecolor="#e0e6ed"),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#dfe6e9", borderwidth=2),
    margin=dict(t=40, b=50, l=50, r=20),
    hovermode="x unified",
)

# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_data():
    try:
        preds    = pd.read_csv("data/predictions.csv")
        features = pd.read_csv("data/features.csv")
        return preds, features
    except FileNotFoundError:
        return None, None

@st.cache_data(ttl=10)
def load_metrics():
    if os.path.exists("data/model_metrics.json"):
        with open("data/model_metrics.json") as f:
            return json.load(f)
    return None

@st.cache_data(ttl=10)
def load_shap_csv():
    path = "data/shap/shap_values.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

FEATURE_COLS = [
    "ua_is_suspicious", "has_referer", "has_accept_lang",
    "hit_secret_page", "ua_length", "time_gap_seconds",
    "unique_pages_visited", "total_requests_from_ip"
]

preds, features = load_data()
metrics = load_metrics()

if preds is None or features is None:
    st.error(
        "Data files not found! Run the pipeline first:\n"
        "- `python honeypot.py` (separate terminal)\n"
        "- `python simulate_traffic.py`\n"
        "- `python feature_engineering.py`\n"
        "- `python train_model.py`"
    )
    st.stop()

total         = len(preds)
bots_count    = int((preds["victor_flag"] == 1).sum())
human_count   = total - bots_count
bot_pct       = (bots_count / total * 100) if total > 0 else 0
human_pct     = (human_count / total * 100) if total > 0 else 0
avg_bot_score = preds["ensemble_score"].mean()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚔ VICTOR")
    st.markdown("**Bot Detection Watchdog**")
    st.divider()

    page = st.radio(
        "Navigate:",
        ["Dashboard", "IP Lookup", "Model Explainability", "Raw Data", "Settings"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("### Live Stats")
    col_a, col_b = st.columns(2)
    col_a.metric("Bots",   f"{bots_count:,}",  f"{bot_pct:.1f}%")
    col_b.metric("Humans", f"{human_count:,}", f"{human_pct:.1f}%")
    st.divider()

    if metrics:
        st.markdown("### Model Quality")
        st.metric("XGBoost AUC",       f"{metrics.get('xgb_auc', 0):.3f}")
        st.metric("Isolation Forest",  f"{metrics.get('iso_auc', 0):.3f}")
        st.divider()

    threshold = st.slider("Bot score threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ──────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown("# Victor Dashboard")
    st.markdown("_Real-time bot detection powered by ensemble ML_")
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Requests", f"{total:,}")
    m2.metric("Bots Detected",  f"{bots_count:,}", f"{bot_pct:.1f}%",   delta_color="inverse")
    m3.metric("Clean Traffic",  f"{human_count:,}", f"{human_pct:.1f}%")
    m4.metric("Avg Score",      f"{avg_bot_score:.3f}", "threshold: 0.50")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Traffic Breakdown")
        pie_data = pd.DataFrame({
            "Type":  ["Bots", "Humans"],
            "Count": [bots_count, human_count],
        })
        fig_pie = px.pie(
            pie_data, names="Type", values="Count",
            color_discrete_map={"Bots": "#d63031", "Humans": "#00b894"},
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              marker=dict(line=dict(color="#ffffff", width=4)))
        fig_pie.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Bot Confidence Score Distribution")
        fig_hist = px.histogram(
            preds, x="ensemble_score", nbins=40,
            labels={"ensemble_score": "Bot Probability", "count": "Requests"}
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#d63031",
                           annotation_text=f"Threshold: {threshold:.2f}",
                           annotation_position="top right")
        fig_hist.update_traces(marker_color="#3498db")
        fig_hist.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.subheader("What Separates Bots from Real Users?")

    bot_means   = features[features["label"] == 1][FEATURE_COLS].mean()
    human_means = features[features["label"] == 0][FEATURE_COLS].mean()
    compare_df  = pd.DataFrame({
        "Feature": [c.replace("_", " ").title() for c in FEATURE_COLS],
        "Bots":    bot_means.values,
        "Humans":  human_means.values,
    })
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Bots",   x=compare_df["Feature"], y=compare_df["Bots"],
                             marker=dict(color="#d63031", opacity=0.85)))
    fig_bar.add_trace(go.Bar(name="Humans", x=compare_df["Feature"], y=compare_df["Humans"],
                             marker=dict(color="#00b894", opacity=0.85)))
    fig_bar.update_layout(barmode="group", xaxis_tickangle=-45, **CHART_LAYOUT, height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("When Do Bots Attack?")
    if "timestamp" in preds.columns:
        preds_time = preds.copy()
        preds_time["timestamp"] = pd.to_datetime(preds_time["timestamp"], errors="coerce")
        preds_time = preds_time.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(preds_time) > 0:
            fig_tl = px.line(preds_time, x="timestamp", y="ensemble_score",
                             labels={"timestamp": "Time", "ensemble_score": "Bot Score"})
            fig_tl.add_hline(y=threshold, line_dash="dash", line_color="#d63031",
                             annotation_text="Threshold")
            fig_tl.update_traces(line=dict(color="#3498db", width=2))
            fig_tl.update_layout(**CHART_LAYOUT, height=350)
            st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Timestamp column not found in data")

    st.divider()
    st.markdown("## Recent Activity")
    recent = preds.tail(10).copy()
    activity_html = '<div class="activity-feed">'
    for _, row in recent.iterrows():
        score  = row.get("ensemble_score", 0)
        is_bot = score > threshold
        badge  = "BOT" if is_bot else "HUMAN"
        color  = "#d63031" if is_bot else "#00b894"
        bg     = "#ffe0e0" if is_bot else "#e0f7e0"
        ip     = row.get("ip", "Unknown")
        activity_html += f"""
        <div class="activity-item" style="border-left: 4px solid {color};">
          <div style="display:flex; justify-content:space-between;">
            <code style="color:#2d3436;">{ip}</code>
            <span style="background:{bg}; color:{color}; padding:2px 10px;
                  border-radius:12px; font-weight:700; font-size:0.8rem;">{badge}</span>
          </div>
          <div style="margin-top:6px; color:#636e72; font-size:0.85rem;">
            Score: <b style="color:{color};">{score:.2%}</b>
          </div>
        </div>"""
    activity_html += '</div>'
    st.markdown(activity_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# PAGE: IP LOOKUP
# ──────────────────────────────────────────────────────────────────
elif page == "IP Lookup":
    st.markdown("# IP Lookup")
    st.markdown("_View the full record for any IP address_")
    st.divider()

    ip_input = st.text_input("Enter IP address:", placeholder="e.g. 127.0.0.1")

    if ip_input:
        has_ip_col = "ip" in preds.columns
        if not has_ip_col:
            st.error("'ip' column not found in predictions.csv. "
                     "Please re-run feature_engineering.py.")
        else:
            ip_data = preds[preds["ip"] == ip_input]

            if len(ip_data) == 0:
                st.warning(f"No records found for **{ip_input}**.")
            else:
                avg_score = ip_data["ensemble_score"].mean()
                max_score = ip_data["ensemble_score"].max()
                req_count = len(ip_data)
                is_bot    = avg_score > threshold

                verdict_color = "#d63031" if is_bot else "#00b894"
                verdict_label = "🤖 BOT DETECTED" if is_bot else "✅ LEGITIMATE"
                verdict_bg    = "#fee2e2" if is_bot else "#dcfce7"

                st.markdown(f"""
                <div style="background:{verdict_bg}; border:2px solid {verdict_color};
                     border-radius:12px; padding:24px; margin-bottom:20px;">
                  <h3 style="color:{verdict_color}; margin:0;">{verdict_label}</h3>
                  <p style="color:#374151; margin:8px 0 0;">
                    IP: <b>{ip_input}</b> &nbsp;·&nbsp;
                    {req_count} requests &nbsp;·&nbsp;
                    Avg score: <b style="color:{verdict_color};">{avg_score:.3f}</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                k1, k2, k3 = st.columns(3)
                k1.metric("Requests", f"{req_count:,}")
                k2.metric("Avg Score", f"{avg_score:.3f}")
                k3.metric("Max Score", f"{max_score:.3f}")

                st.subheader("Score Distribution")
                fig_ip = px.histogram(ip_data, x="ensemble_score", nbins=20,
                                      color_discrete_sequence=["#3498db"])
                fig_ip.add_vline(x=threshold, line_dash="dash", line_color="#d63031")
                fig_ip.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_ip, use_container_width=True)

                st.subheader("All Records for this IP")
                show_cols = [c for c in ["ip", "ensemble_score", "xgb_score",
                                          "iso_score", "victor_flag"] if c in ip_data.columns]
                st.dataframe(ip_data[show_cols].round(4), use_container_width=True)
    else:
        st.info("Enter an IP address above to look it up")

# ──────────────────────────────────────────────────────────────────
# PAGE: MODEL EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────
elif page == "Model Explainability":
    st.markdown("# Model Explainability")
    st.markdown("_Understand why Victor classifies a request as a bot_")
    st.divider()

    shap_img_global = "data/shap/global_summary.png"
    shap_img_bar    = "data/shap/feature_bar.png"
    shap_csv        = load_shap_csv()

    if os.path.exists(shap_img_global) or os.path.exists(shap_img_bar):
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(shap_img_global):
                st.subheader("Global Feature Importance")
                st.image(shap_img_global, use_container_width=True)
        with col2:
            if os.path.exists(shap_img_bar):
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

# ──────────────────────────────────────────────────────────────────
# PAGE: RAW DATA
# ──────────────────────────────────────────────────────────────────
elif page == "Raw Data":
    st.markdown("# Raw Predictions Log")
    st.markdown("_All requests with their scores and verdict_")
    st.divider()

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        filter_opt = st.radio("Show:", ["All", "Bots only", "Humans only"], horizontal=True)
    with col_f2:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)
    with col_f3:
        max_rows = st.number_input("Max rows:", 10, 10000, 200)

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

# ──────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ──────────────────────────────────────────────────────────────────
elif page == "Settings":
    st.markdown("# Settings & Info")
    st.divider()

    st.markdown("### Dataset Info")
    st.json({
        "total_records": int(len(preds)),
        "bots_flagged":  int(bots_count),
        "clean_traffic": int(human_count),
        "bot_percentage": f"{bot_pct:.2f}%",
        "features_used":  len(FEATURE_COLS),
        "ip_column_present": "ip" in preds.columns,
    })

    st.divider()
    st.markdown("### Current Threshold")
    st.info(f"Threshold: **{threshold}** — any score above this is flagged as a bot")

    st.divider()
    st.markdown("### Model Metrics")
    if metrics:
        st.json(metrics)
    else:
        st.warning("Metrics file not found — run train_model.py first")

    st.divider()
    st.markdown("### File Structure")
    st.code("""data/
├── features.csv
├── predictions.csv
├── model_metrics.json
├── victor_traffic.db
└── shap/
    ├── shap_values.csv
    ├── global_summary.png
    └── feature_bar.png
models/
├── isolation_forest.pkl
└── xgboost_model.pkl""", language="bash")