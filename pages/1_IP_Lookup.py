import streamlit as st
import plotly.express as px
from ui_utils import setup_page, load_data, render_sidebar, get_geolocation, CHART_LAYOUT

setup_page("Victor — IP Lookup")
preds, features = load_data()

if preds is None:
    st.error("Data files not found! Run the pipeline first.")
    st.stop()

threshold = render_sidebar(preds)

st.markdown("# IP Lookup")
st.markdown("_View the full record for any IP address_")
st.divider()

ip_input = st.text_input("Enter IP address:", placeholder="e.g. 127.0.0.1")

if ip_input:
    has_ip_col = "ip" in preds.columns
    if not has_ip_col:
        st.error("'ip' column not found in predictions.csv. Please re-run feature_engineering.py.")
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

            # Geolocation enrichment
            with st.spinner("Looking up location..."):
                geo = get_geolocation(ip_input)
            if geo:
                st.info(
                    f"📍 **{geo.get('city', '—')}, {geo.get('regionName', '—')}, "
                    f"{geo.get('country', '—')}**  ·  "
                    f"🏢 **ISP**: {geo.get('isp', '—')}"
                )

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
