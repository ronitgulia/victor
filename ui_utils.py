import streamlit as st
import pandas as pd
import json, requests
from pathlib import Path
from config import Paths
from config_loader import Config
import time

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

FEATURE_COLS = [
    "ua_is_suspicious", "has_referer", "has_accept_lang",
    "hit_secret_page", "ua_length", "time_gap_seconds",
    "unique_pages_visited", "total_requests_from_ip"
]

def setup_page(title="Victor — Bot Detection"):
    st.set_page_config(
        page_title=title,
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

@st.cache_data(ttl=Config.get("dashboard.cache_ttl", 5))
def load_data():
    try:
        preds    = pd.read_csv(Paths.PREDICTIONS)
        features = pd.read_csv(Paths.FEATURES)
        return preds, features
    except FileNotFoundError:
        return None, None

@st.cache_data(ttl=Config.get("dashboard.cache_ttl_metrics", 10))
def load_metrics():
    if Path(Paths.MODEL_METRICS).exists():
        with open(Paths.MODEL_METRICS) as f:
            return json.load(f)
    return None

@st.cache_data(ttl=Config.get("dashboard.cache_ttl_metrics", 10))
def load_shap_csv():
    if Path(Paths.SHAP_VALUES).exists():
        return pd.read_csv(Paths.SHAP_VALUES)
    return None

@st.cache_data(ttl=3600)
def get_geolocation(ip: str):
    """Fetch IP geolocation from ip-api.com"""
    try:
        resp = requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,country,regionName,city,isp,query",
            timeout=4
        )
        data = resp.json()
        if data.get("status") == "success":
            return data
    except Exception:
        pass
    return None

def render_sidebar(preds=None, metrics=None):
    with st.sidebar:
        st.markdown("## ⚔ VICTOR")
        st.markdown("**Bot Detection Watchdog**")
        st.divider()

        if preds is not None:
            total         = len(preds)
            bots_count    = int((preds["victor_flag"] == 1).sum()) if "victor_flag" in preds.columns else 0
            human_count   = total - bots_count
            bot_pct       = (bots_count / total * 100) if total > 0 else 0
            human_pct     = (human_count / total * 100) if total > 0 else 0
            
            st.markdown("### Live Stats")
            col_a, col_b = st.columns(2)
            col_a.metric("Bots",   f"{bots_count:,}",  f"{bot_pct:.1f}%")
            col_b.metric("Humans", f"{human_count:,}", f"{human_pct:.1f}%")
            st.divider()

        st.markdown("### 🛡 Real-Time Shield")
        _rt_stats = {}
        if Path(Paths.REALTIME_STATS).exists():
            try:
                with open(Paths.REALTIME_STATS) as _f:
                    _rt_stats = json.load(_f)
            except Exception:
                pass

        if _rt_stats:
            _model_ok = _rt_stats.get("model_loaded", False)
            _scored   = _rt_stats.get("scored",  0)
            _blocked  = _rt_stats.get("blocked", 0)
            _mode     = "Blocking" if _rt_stats.get("blocking_mode") else "Log-only"
            st.caption(f"Model: {'✅ Loaded' if _model_ok else '❌ Not loaded'}  |  {_mode}")
            col_s, col_b2 = st.columns(2)
            col_s.metric("Scored",  f"{_scored:,}")
            col_b2.metric("Blocked", f"{_blocked:,}",
                          delta=f"{_blocked/_scored*100:.1f}%" if _scored else None,
                          delta_color="inverse")
        else:
            st.caption("Honeypot not running")
            st.caption("Start: `python honeypot.py`")
        st.divider()

        if metrics:
            st.markdown("### Model Quality")
            st.metric("XGBoost AUC",       f"{metrics.get('xgb_auc', 0):.3f}")
            st.metric("Isolation Forest",  f"{metrics.get('iso_auc', 0):.3f}")
            st.divider()

        threshold = st.slider(
            "Bot score threshold",
            float(Config.get("detection.threshold_min", 0.0)),
            float(Config.get("detection.threshold_max", 1.0)),
            float(Config.get("detection.default_threshold", 0.5)),
            float(Config.get("detection.threshold_step", 0.05)),
            key="threshold"
        )

        if preds is not None:
            spike_limit = Config.get("detection.bot_spike_threshold", 60.0)
            if bot_pct > spike_limit:
                st.error(f"⚠️ Bot Spike! {bot_pct:.1f}% of traffic is bots")

        st.divider()
        st.markdown("### Live Mode")
        live_mode = st.toggle(
            "🔴 Auto-Refresh",
            value=bool(Config.get("dashboard.live_mode_default", False)),
            help="Automatically refreshes data on a fixed interval"
        )
        refresh_interval = (
            st.slider("Refresh every (s)", 3, 60,
                      int(Config.get("dashboard.live_mode_refresh_interval", 5)))
            if live_mode else 5
        )

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if live_mode:
            time.sleep(refresh_interval)
            st.cache_data.clear()
            st.rerun()

        return threshold
