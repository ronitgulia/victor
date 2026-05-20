import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os
from datetime import datetime

st.set_page_config(
    page_title="Victor — Bot Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
    
    /* Main background - modern gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2d3436;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3436 0%, #636e72 100%);
        border-right: 3px solid #0984e3;
    }
    
    [data-testid="stSidebar"] * { color: #fff !important; }
    [data-testid="stSidebar"] h2 { font-size: 1.6rem !important; margin-bottom: 8px !important; font-weight: 800 !important; }
    [data-testid="stSidebar"] [role="radio"] label { color: #f1f2f6 !important; font-weight: 600 !important; }
    
    /* Card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e0e6ed;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: #0984e3;
        box-shadow: 0 12px 32px rgba(9, 132, 227, 0.15);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetricLabel"] { 
        color: #636e72 !important; 
        font-size: 0.8rem !important; 
        font-weight: 700 !important; 
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    
    [data-testid="stMetricValue"] { 
        color: #2d3436 !important; 
        font-size: 2.4rem !important; 
        font-weight: 800 !important; 
    }
    
    /* Headers */
    h1, h2, h3 { color: #2d3436 !important; font-weight: 800 !important; }
    h1 { font-size: 2.8rem !important; margin-top: 0 !important; }
    h2 { font-size: 1.8rem !important; margin-top: 24px !important; }
    
    /* Dividers */
    hr { border-color: #dfe6e9 !important; margin: 24px 0 !important; }
    
    /* Alerts */
    .stSuccess, [data-testid="stSuccess"] { background: #d4edda !important; border: 2px solid #28a745 !important; }
    .stWarning, [data-testid="stWarning"] { background: #fff3cd !important; border: 2px solid #ffc107 !important; }
    .stError, [data-testid="stError"] { background: #f8d7da !important; border: 2px solid #dc3545 !important; }
    .stInfo, [data-testid="stInfo"] { background: #d1ecf1 !important; border: 2px solid #17a2b8 !important; }
    
    /* Status badges */
    .badge-bot {
        background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(214, 48, 49, 0.3);
    }
    
    .badge-human {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(0, 184, 148, 0.3);
    }
    
    /* Live indicator */
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #d63031;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Charts */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0984e3 0%, #0770d4 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(9, 132, 227, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 20px rgba(9, 132, 227, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Text input */
    .stTextInput input {
        border: 2px solid #dfe6e9 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    .stTextInput input:focus {
        border-color: #0984e3 !important;
        box-shadow: 0 0 0 3px rgba(9, 132, 227, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CHART LAYOUT CONFIG
# ─────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#2d3436", family="system-ui, -apple-system, sans-serif", size=12),
    xaxis=dict(gridcolor="#e0e6ed", zerolinecolor="#e0e6ed", showgrid=True),
    yaxis=dict(gridcolor="#e0e6ed", zerolinecolor="#e0e6ed", showgrid=True),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(color="#2d3436", size=11), 
                bordercolor="#dfe6e9", borderwidth=2),
    margin=dict(t=40, b=50, l=50, r=20),
    hovermode="x unified",
)

# ─────────────────────────────────────────────────────────────────
# CACHING & DATA LOADING
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)  # 5 second cache for live updates
def load_data():
    """Load predictions and features with auto-refresh"""
    try:
        preds = pd.read_csv("data/predictions.csv")
        features = pd.read_csv("data/features.csv")
        return preds, features
    except FileNotFoundError:
        return None, None

@st.cache_data(ttl=10)
def load_metrics():
    """Load model metrics"""
    if os.path.exists("data/model_metrics.json"):
        with open("data/model_metrics.json") as f:
            return json.load(f)
    return None

@st.cache_data(ttl=10)
def load_shap_csv():
    """Load SHAP values for explainability"""
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
    st.error("❌ Data files not found! Run the pipeline first:\n"
             "- `python honeypot.py` (in separate terminal)\n"
             "- `python simulate_traffic.py`\n"
             "- `python feature_engineering.py`\n"
             "- `python train_model.py`")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR - Navigation & Settings
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ VICTOR")
    st.markdown("**Bot Detection System**")
    st.markdown("_Real-time behavioral analysis_")
    st.divider()
    
    # Live indicator
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown('<span class="live-indicator"></span>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Live** • Last update: `just now`")
    
    st.divider()
    
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select View:",
        ["📈 Dashboard", "🔍 IP Lookup", "🧠 Model Explainability", "📋 Raw Data", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ⚡ Quick Stats")
    total = len(preds)
    bots_count = int((preds["victor_flag"] == 1).sum())
    human_count = total - bots_count
    
    col_a, col_b = st.columns(2)
    col_a.metric("🤖 Bots", f"{bots_count:,}", f"{bots_count/total*100:.1f}%")
    col_b.metric("👥 Humans", f"{human_count:,}", f"{human_count/total*100:.1f}%")
    
    st.divider()
    
    if metrics:
        st.markdown("### 🎯 Model Performance")
        st.metric("XGBoost AUC", f"📊 {metrics.get('xgb_auc', 0):.3f}")
        st.metric("Isolation Forest", f"🌳 {metrics.get('iso_auc', 0):.3f}")
    
    st.divider()
    
    threshold = st.slider(
        "🎚️ Bot Score Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Requests scoring above this are flagged as bots"
    )
    
    st.markdown("### 🔄 Refresh Settings")
    refresh_now = st.button("🔄 Refresh Data Now", use_container_width=True)
    if refresh_now:
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────

if page == "📈 Dashboard":
    st.markdown("# 🛡️ Victor Dashboard")
    st.markdown("_Live bot detection powered by ensemble ML • Auto-refreshes every 5 seconds_")
    st.divider()
    
    # Key metrics with humanized display
    m1, m2, m3, m4 = st.columns(4)
    
    bot_pct = (bots_count / total * 100) if total > 0 else 0
    human_pct = (human_count / total * 100) if total > 0 else 0
    avg_bot_score = preds["ensemble_score"].mean()
    
    with m1:
        st.metric(
            "📊 Total Requests",
            f"{total:,}",
            "incoming stream"
        )
    
    with m2:
        st.metric(
            "🤖 Bots Detected",
            f"{bots_count:,}",
            f"{bot_pct:.1f}%",
            delta_color="inverse"
        )
    
    with m3:
        st.metric(
            "👥 Clean Traffic",
            f"{human_count:,}",
            f"{human_pct:.1f}%"
        )
    
    with m4:
        st.metric(
            "📈 Avg Score",
            f"{avg_bot_score:.3f}",
            "threshold: 0.50"
        )
    
    st.divider()
    
    # Charts row 1
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🥧 Traffic Composition")
        pie_data = pd.DataFrame({
            "Type": ["🤖 Bots", "👥 Humans"],
            "Count": [bots_count, human_count],
            "Pct": [bot_pct, human_pct]
        })
        fig_pie = px.pie(
            pie_data,
            names="Type",
            values="Count",
            color_discrete_map={"🤖 Bots": "#d63031", "👥 Humans": "#00b894"},
            hole=0.4
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#ffffff", width=4))
        )
        fig_pie.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.subheader("📊 Bot Confidence Scores")
        fig_hist = px.histogram(
            preds,
            x="ensemble_score",
            nbins=40,
            title="Distribution of bot detection confidence",
            labels={"ensemble_score": "Bot Probability", "count": "Requests"}
        )
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#d63031",
            annotation_text=f"Threshold: {threshold:.2f}",
            annotation_position="top right"
        )
        fig_hist.update_traces(marker_color="#3498db")
        fig_hist.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()
    
    # Feature importance
    st.subheader("🔍 Feature Importance: Bots vs Humans")
    st.markdown("_How different behavior patterns differentiate bots from legitimate users_")
    
    bot_means = features[features["label"] == 1][FEATURE_COLS].mean()
    human_means = features[features["label"] == 0][FEATURE_COLS].mean()
    
    compare_df = pd.DataFrame({
        "Feature": [col.replace("_", " ").title() for col in FEATURE_COLS],
        "🤖 Bots": bot_means.values,
        "👥 Humans": human_means.values
    })
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="🤖 Bots",
        x=compare_df["Feature"],
        y=compare_df["🤖 Bots"],
        marker=dict(color="#d63031", opacity=0.85, line=dict(color="#c92a1f", width=1.5))
    ))
    fig_bar.add_trace(go.Bar(
        name="👥 Humans",
        x=compare_df["Feature"],
        y=compare_df["👥 Humans"],
        marker=dict(color="#00b894", opacity=0.85, line=dict(color="#008c5c", width=1.5))
    ))
    fig_bar.update_layout(barmode="group", xaxis_tickangle=-45, **CHART_LAYOUT, height=450)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Timeline (if timestamp exists)
    st.divider()
    st.subheader("⏱️ Detection Timeline")
    
    if "timestamp" in preds.columns:
        preds_time = preds.copy()
        preds_time["timestamp"] = pd.to_datetime(preds_time["timestamp"], errors="coerce")
        preds_time = preds_time.dropna(subset=["timestamp"]).sort_values("timestamp")
        
        if len(preds_time) > 0:
            fig_timeline = px.line(
                preds_time,
                x="timestamp",
                y="ensemble_score",
                hover_data={"ensemble_score": ":.3f"},
                labels={"timestamp": "Time", "ensemble_score": "Bot Score"}
            )
            fig_timeline.add_hline(y=threshold, line_dash="dash", line_color="#d63031", 
                                   annotation_text="Threshold")
            fig_timeline.update_traces(line=dict(color="#3498db", width=2))
            fig_timeline.update_layout(**CHART_LAYOUT, height=350)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("⏰ No timestamp data available")
    else:
        st.info("⏰ Timestamp column not found in data")

# ─────────────────────────────────────────────────────────────────
# PAGE: IP LOOKUP
# ─────────────────────────────────────────────────────────────────

elif page == "🔍 IP Lookup":
    st.markdown("# 🔍 IP Address Lookup")
    st.markdown("_Inspect any IP address and see its full activity profile_")
    st.divider()
    
    ip_input = st.text_input(
        "🔎 Enter IP address to investigate",
        placeholder="e.g., 192.168.1.1 or 127.0.0.1",
        label_visibility="collapsed"
    )
    
    if ip_input:
        # Find matching IP
        if "ip" in preds.columns:
            ip_data = preds[preds["ip"] == ip_input]
        else:
            st.warning("⚠️ IP column not found in predictions data")
            ip_data = pd.DataFrame()
        
        if len(ip_data) == 0:
            st.error(f"❌ No records found for IP: **{ip_input}**")
            st.info("💡 Make sure the IP exists in your traffic data")
        else:
            avg_score = ip_data["ensemble_score"].mean()
            max_score = ip_data["ensemble_score"].max()
            req_count = len(ip_data)
            is_bot = avg_score > threshold
            
            # Verdict card
            verdict_gradient = "linear-gradient(135deg, #d63031 0%, #c92a1f 100%)" if is_bot else "linear-gradient(135deg, #00b894 0%, #008c5c 100%)"
            verdict_label = "🤖 BOT DETECTED" if is_bot else "✅ LEGITIMATE"
            
            st.markdown(f"""
            <div style="background: {verdict_gradient}; color: white; border-radius: 16px; 
                        padding: 32px; margin: 20px 0; box-shadow: 0 12px 32px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; color: white; font-size: 2.2rem;">{verdict_label}</h2>
                <p style="margin: 12px 0 0; font-size: 1rem; opacity: 0.95;">
                    IP: <code style="background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 4px;">{ip_input}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📍 Requests", f"{req_count:,}")
            k2.metric("📊 Avg Score", f"{avg_score:.3f}")
            k3.metric("📈 Max Score", f"{max_score:.3f}")
            k4.metric("🎯 Status", "BOT" if is_bot else "HUMAN")
            
            st.divider()
            
            st.subheader("📋 Full Activity")
            display_cols = ["ip", "ensemble_score", "xgb_score", "iso_score", "victor_flag"] \
                          if "ip" in ip_data.columns else ["ensemble_score", "xgb_score", "iso_score", "victor_flag"]
            
            display_data = ip_data[display_cols].copy() if all(c in ip_data.columns for c in display_cols) else ip_data
            display_data = display_data.round(4)
            
            st.dataframe(display_data, use_container_width=True, height=300)
            
            # Score distribution chart
            st.subheader("📊 Score Distribution for This IP")
            fig_ip = px.histogram(ip_data, x="ensemble_score", nbins=20,
                                 title=f"Bot confidence scores for {ip_input}",
                                 labels={"ensemble_score": "Bot Score", "count": "Occurrences"})
            fig_ip.add_vline(x=threshold, line_dash="dash", line_color="#d63031",
                            annotation_text=f"Threshold ({threshold})")
            fig_ip.update_traces(marker_color="#3498db")
            fig_ip.update_layout(**CHART_LAYOUT, height=350)
            st.plotly_chart(fig_ip, use_container_width=True)
    else:
        st.info("👆 Enter an IP address above to investigate its activity")

# ─────────────────────────────────────────────────────────────────
# PAGE: MODEL EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────

elif page == "🧠 Model Explainability":
    st.markdown("# 🧠 Understanding the Model")
    st.markdown("_How does Victor decide if traffic is a bot? Learn the decision-making process._")
    st.divider()
    
    st.subheader("📚 Feature Explanations")
    st.markdown("Each of these signals helps Victor identify bot behavior:")
    
    explanations = {
        "👤 User Agent Suspicious": "Does the User-Agent match known bot signatures (curl, python-requests, etc.)?",
        "🔗 Has Referer": "Did the request include a Referer header? Humans usually do.",
        "🌍 Accept-Language": "Was an Accept-Language header sent? Bots often skip it.",
        "🔓 Hit Secret Page": "Did the IP visit the hidden honeypot endpoint `/secret-data`? Only bots do.",
        "📏 User-Agent Length": "How long is the User-Agent string? Bots tend to use shorter ones.",
        "⏱️ Time Gap Between Requests": "How many seconds between requests? Bots move very fast (< 1 sec).",
        "📑 Unique Pages Visited": "How many different pages? Bots typically sweep many pages quickly.",
        "📡 Total Requests from IP": "How many total requests from this IP? Bots make many requests.",
    }
    
    for feature, explanation in explanations.items():
        st.markdown(f"**{feature}**  \n_{explanation}_  ")
    
    st.divider()
    
    # SHAP visualizations
    st.subheader("🔬 SHAP Analysis")
    st.markdown("_Feature importance as measured by SHAP values_")
    
    shap_img_global = "data/shap/global_summary.png"
    shap_img_bar = "data/shap/feature_bar.png"
    
    if os.path.exists(shap_img_global) or os.path.exists(shap_img_bar):
        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            if os.path.exists(shap_img_global):
                st.markdown("**Global Feature Importance**")
                st.image(shap_img_global, use_container_width=True)
        
        with col_shap2:
            if os.path.exists(shap_img_bar):
                st.markdown("**Average Feature Impact**")
                st.image(shap_img_bar, use_container_width=True)
    else:
        st.warning("📊 SHAP plots not generated yet. Run `python explain.py` to generate them.")
    
    shap_csv = load_shap_csv()
    if shap_csv is not None:
        st.divider()
        st.subheader("🗺️ SHAP Value Heatmap")
        st.markdown("_Top 200 requests showing SHAP values per feature_")
        
        shap_display = shap_csv.head(200)
        fig_heat = px.imshow(
            shap_display.values.T,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(x="Request Index", y="Feature", color="SHAP Value")
        )
        fig_heat.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: RAW DATA
# ─────────────────────────────────────────────────────────────────

elif page == "📋 Raw Data":
    st.markdown("# 📋 Raw Predictions")
    st.markdown("_View and filter the complete detection log_")
    st.divider()
    
    col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])
    
    with col_filter1:
        filter_type = st.selectbox(
            "Filter by type:",
            ["📊 All Traffic", "🤖 Bots Only", "👥 Humans Only"],
            label_visibility="collapsed"
        )
    
    with col_filter2:
        min_score = st.slider("Min score:", 0.0, 1.0, 0.0, label_visibility="collapsed")
    
    with col_filter3:
        max_rows = st.number_input("Show rows:", 10, 10000, 100, label_visibility="collapsed")
    
    # Apply filters
    display_df = preds.copy()
    
    if filter_type == "🤖 Bots Only":
        display_df = display_df[display_df["ensemble_score"] > threshold]
    elif filter_type == "👥 Humans Only":
        display_df = display_df[display_df["ensemble_score"] <= threshold]
    
    display_df = display_df[display_df["ensemble_score"] >= min_score]
    display_df = display_df.head(max_rows)
    
    st.markdown(f"**Showing {len(display_df):,} records**")
    st.dataframe(display_df.round(4), use_container_width=True, height=500)
    
    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download as CSV",
        csv,
        "victor_predictions.csv",
        "text/csv"
    )

# ─────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.markdown("# ⚙️ Settings & Configuration")
    st.divider()
    
    st.markdown("### 📊 Dataset Information")
    st.json({
        "total_records": int(len(preds)),
        "bots_flagged": int(bots_count),
        "clean_traffic": int(human_count),
        "bot_percentage": f"{bot_pct:.2f}%",
        "features_used": len(FEATURE_COLS)
    })
    
    st.divider()
    
    st.markdown("### 🎯 Current Threshold")
    st.info(f"Bot Score Threshold: **{threshold}**  \n"
            f"Requests scoring above this value are flagged as bots.")
    
    st.divider()
    
    st.markdown("### 🔧 Model Configuration")
    if metrics:
        st.json(metrics)
    else:
        st.warning("⚠️ Metrics file not found")
    
    st.divider()
    
    st.markdown("### 💾 File Locations")
    st.code(
        """data/
├── features.csv
├── predictions.csv
├── model_metrics.json
├── traffic_logs.json
└── shap/
    └── shap_values.csv

models/
├── isolation_forest.pkl
└── xgboost_model.pkl
""",
        language="bash"
    )

    st.divider()

    avg_bot_score = round(preds["ensemble_score"].mean(), 3)
    c1.metric("Total Requests",  f"{total:,}")
    c2.metric("Bots Flagged",    f"{flagged_bots:,}",  delta=f"{bot_pct:.1f}%",   delta_color="inverse")
    c3.metric("Clean (Human)",   f"{clean_humans:,}",  delta=f"{human_pct:.1f}%")
    c4.metric("Avg Bot Score",   avg_bot_score,         delta="threshold 0.5",     delta_color="off")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Traffic Breakdown")
        pie_df = pd.DataFrame({"Type": ["Bot", "Human"], "Count": [flagged_bots, clean_humans]})
        fig_pie = px.pie(pie_df, names="Type", values="Count",
                         color="Type",
                         color_discrete_map={"Bot": "#dc2626", "Human": "#16a34a"},
                         hole=0.5)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              marker=dict(line=dict(color="#ffffff", width=3)))
        fig_pie.update_layout(**CHART_LAYOUT)
        fig_pie.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.subheader("Bot Confidence Score Distribution")
        fig_hist = px.histogram(preds, x="ensemble_score", nbins=35,
                                color_discrete_sequence=["#3b82f6"],
                                labels={"ensemble_score": "Bot Probability"})
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#dc2626",
                           annotation_text=f"Threshold ({threshold})",
                           annotation_font_color="#dc2626")
        fig_hist.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    st.subheader("Feature Comparison — Bots vs Humans")

    bot_means   = features[features["label"] == 1][FEATURE_COLS].mean()
    human_means = features[features["label"] == 0][FEATURE_COLS].mean()
    compare_df  = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Bot":     bot_means.values,
        "Human":   human_means.values
    })

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Bot",   x=compare_df["Feature"], y=compare_df["Bot"],
                             marker_color="#dc2626", marker_opacity=0.85))
    fig_bar.add_trace(go.Bar(name="Human", x=compare_df["Feature"], y=compare_df["Human"],
                             marker_color="#16a34a", marker_opacity=0.85))
    fig_bar.update_layout(barmode="group", xaxis_tickangle=-20, **CHART_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── row 3: score over time (if timestamp available) ──
        st.divider()
        st.subheader("Bot Score Over Time")
        preds_t = preds.copy()
        preds_t["timestamp"] = pd.to_datetime(preds_t["timestamp"], errors="coerce")
        preds_t = preds_t.dropna(subset=["timestamp"]).sort_values("timestamp")
        fig_line = px.line(preds_t, x="timestamp", y="ensemble_score",
                           color_discrete_sequence=["#3b82f6"])
        fig_line.add_hline(y=threshold, line_dash="dash", line_color="#dc2626")
        fig_line.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_line, use_container_width=True)


elif page == "IP Lookup":
    st.markdown("# IP Lookup")
    st.markdown("<p style='color:#6b7280;margin-top:-12px;font-size:1rem'>Check any IP address against the logged traffic data</p>", unsafe_allow_html=True)
    ip_input = st.text_input("Enter an IP address to inspect", placeholder="e.g. 127.0.0.1")

    if ip_input:
        ip_rows = preds[features["label"].index.isin(
            features[preds.index].index  # align on same index
        )].copy() if "ip" not in preds.columns else preds[preds.get("ip", pd.Series()) == ip_input]

        # fallback: search in features if ip column exists there
        feat_ip = features[features["ip"] == ip_input] if "ip" in features.columns else pd.DataFrame()
        pred_ip = preds.loc[feat_ip.index] if len(feat_ip) > 0 else pd.DataFrame()

        if len(pred_ip) == 0 and "ip" in preds.columns:
            pred_ip = preds[preds["ip"] == ip_input]

        if len(pred_ip) == 0:
            st.warning(f"No traffic records found for **{ip_input}**.")
        else:
            avg_score = pred_ip["ensemble_score"].mean()
            is_bot    = avg_score > threshold

            verdict_color = "#dc2626" if is_bot else "#16a34a"
            verdict_label = "BOT" if is_bot else "HUMAN"
            verdict_bg = "#fee2e2" if is_bot else "#dcfce7"
            verdict_border = "#fecaca" if is_bot else "#bbf7d0"

            st.markdown(f"""
            <div style='background:{verdict_bg};border:2px solid {verdict_border};
                        border-radius:12px;padding:24px;margin-bottom:20px'>
              <h3 style='color:{verdict_color};margin:0;font-size:1.5rem'>{verdict_label}</h3>
              <p style='color:#374151;margin:8px 0 0;font-size:0.95rem'>IP: <b>{ip_input}</b>
                 &nbsp;·&nbsp; {len(pred_ip)} requests logged
                 &nbsp;·&nbsp; Avg score: <b style='color:{verdict_color}'>{avg_score:.3f}</b>
              </p>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Requests",     len(pred_ip))
            col_b.metric("Avg Bot Score", f"{avg_score:.3f}")
            col_c.metric("Verdict",       "BOT" if is_bot else "HUMAN")

            st.markdown("**Score distribution for this IP**")
            fig_ip = px.histogram(pred_ip, x="ensemble_score", nbins=15,
                                  color_discrete_sequence=["#3b82f6"])
            fig_ip.add_vline(x=threshold, line_dash="dash", line_color="#dc2626")
            fig_ip.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_ip, use_container_width=True)
    else:
        st.info("💡 Enter an IP address above to see its full activity profile.")


# ─────────────────────────────────────────────────────────────────
# ── PAGE 3: MODEL EXPLAINABILITY ──
# ─────────────────────────────────────────────────────────────────

elif page == "Model Explainability":
    st.markdown("# Model Explainability")
    st.markdown("<p style='color:#6b7280;margin-top:-12px;font-size:1rem'>Understanding what drives bot detection predictions</p>", unsafe_allow_html=True)
    st.divider()

    shap_img_bar    = "data/shap/feature_bar.png"
    shap_img_global = "data/shap/global_summary.png"
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
        st.warning("SHAP plots not found. Run `python explain.py` to generate them.")

    if shap_csv is not None:
        st.divider()
        st.subheader("SHAP Value Heatmap (top 200 rows)")
        shap_display = shap_csv.head(200)
        shap_renamed = shap_display.rename(columns={c: c.replace("shap_", "") for c in shap_display.columns})
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
        "ua_is_suspicious":      "User agent matches known bot signatures (python-requests, curl, etc.)",
        "has_referer":           "Whether the request came with a Referer header (humans usually do)",
        "has_accept_lang":       "Whether Accept-Language header was sent (bots often skip it)",
        "hit_secret_page":       "Whether the IP visited the hidden honeypot endpoint /secret-data",
        "ua_length":             "Length of the user agent string (bots tend to have shorter ones)",
        "time_gap_seconds":      "Seconds between requests from the same IP (bots move very fast)",
        "unique_pages_visited":  "Number of distinct pages visited (bots sweep many pages)",
        "total_requests_from_ip":"Total request count from this IP in the dataset",
    }
    for feat, desc in explanations.items():
        st.markdown(f"**`{feat}`** — {desc}")


elif page == "Raw Data":
    st.markdown("# Raw Predictions Log")
    st.markdown("<p style='color:#6b7280;margin-top:-12px;font-size:1rem'>Full dataset with scores, flags, and filtering</p>", unsafe_allow_html=True)
    st.divider()

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        filter_opt = st.radio("Show:", ["All", "Bots only", "Humans only"], horizontal=True)
    with col_f2:
        search_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)

    display_df = preds.copy()
    display_df["victor_flag"] = (display_df["ensemble_score"] > threshold).astype(int)

    if filter_opt == "Bots only":
        display_df = display_df[display_df["victor_flag"] == 1]
    elif filter_opt == "Humans only":
        display_df = display_df[display_df["victor_flag"] == 0]

    display_df = display_df[display_df["ensemble_score"] >= search_score]

    cols_to_show = [c for c in [*FEATURE_COLS, "iso_score", "xgb_score", "ensemble_score", "victor_flag"] if c in display_df.columns]
    display_df = display_df[cols_to_show].sort_values("ensemble_score", ascending=False).reset_index(drop=True)

    for col in ["iso_score", "xgb_score", "ensemble_score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    def style_row(row):
        status = "<span class='badge-bot'>BOT</span>" if row.get("victor_flag", 0) == 1 else "<span class='badge-human'>HUMAN</span>"
        return status

    display_df["status"] = display_df.apply(style_row, axis=1)
    display_df = display_df.drop(columns=["victor_flag"], errors="ignore")

    html_table = display_df.to_html(index=False, escape=False)
    st.markdown(f"<div class='victor-table-wrap'>{html_table}</div>", unsafe_allow_html=True)
    st.caption(f"Showing {len(display_df):,} of {len(preds):,} total requests  •  threshold = {threshold}")