import streamlit as st
from ui_utils import setup_page, load_data, load_metrics, render_sidebar, FEATURE_COLS

setup_page("Victor — Settings")
preds, _ = load_data()
metrics = load_metrics()

threshold = render_sidebar(preds, metrics)

st.markdown("# Settings & Info")
st.divider()

if preds is not None:
    total         = len(preds)
    bots_count    = int((preds["victor_flag"] == 1).sum()) if "victor_flag" in preds.columns else 0
    human_count   = total - bots_count
    bot_pct       = (bots_count / total * 100) if total > 0 else 0

    st.markdown("### Dataset Info")
    st.json({
        "total_records": int(len(preds)),
        "bots_flagged":  int(bots_count),
        "clean_traffic": int(human_count),
        "bot_percentage": f"{bot_pct:.2f}%",
        "features_used":  len(FEATURE_COLS),
        "ip_column_present": "ip" in preds.columns,
    })
else:
    st.warning("Data not available. Please run the pipeline.")

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
├── xgboost_model.pkl
└── versions/
    ├── isolation_forest_<timestamp>.pkl
    └── xgboost_model_<timestamp>.pkl""", language="bash")
