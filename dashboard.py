import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ui_utils import setup_page, load_data, load_metrics, render_sidebar, CHART_LAYOUT, FEATURE_COLS

setup_page("Victor — Dashboard")

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

threshold = render_sidebar(preds, metrics)

total         = len(preds)
bots_count    = int((preds["victor_flag"] == 1).sum()) if "victor_flag" in preds.columns else 0
human_count   = total - bots_count
bot_pct       = (bots_count / total * 100) if total > 0 else 0
human_pct     = (human_count / total * 100) if total > 0 else 0
avg_bot_score = preds["ensemble_score"].mean()

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

if "label" in features.columns:
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