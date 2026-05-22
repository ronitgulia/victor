import shap
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
from config import Paths
from logger import get_logger

logger = get_logger(__name__)



model = joblib.load(Paths.XGB_MODEL)
df    = pd.read_csv(Paths.FEATURES)

# Always use the exact feature list the model was trained on
_feat_path = "models/feature_cols.json"
if os.path.exists(_feat_path):
    with open(_feat_path) as _f:
        FEATURE_COLS = json.load(_f)
    logger.info(f"Feature list loaded from {_feat_path}  ({len(FEATURE_COLS)} features)")
else:
    # Fallback for older setups without feature_cols.json
    FEATURE_COLS = [
        "ua_is_suspicious", "has_referer", "has_accept_lang",
        "hit_secret_page", "ua_length", "time_gap_seconds",
        "unique_pages_visited", "total_requests_from_ip"
    ]
    logger.warning("Warning: models/feature_cols.json not found — using default 8 features")

X = df[FEATURE_COLS]


logger.info("Computing SHAP values (this takes a few seconds)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

logger.info("Done.")

logger.info("Saving global feature importance plot...")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("Victor — Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(Paths.SHAP_GLOBAL_SUMMARY, dpi=150, bbox_inches="tight")
plt.close()

logger.info("  Saved -> data/shap/global_summary.png")

logger.info("Saving bar chart version...")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Victor — Average Feature Impact")
plt.tight_layout()
plt.savefig(Paths.SHAP_FEATURE_BAR, dpi=150, bbox_inches="tight")
plt.close()

logger.info("  Saved -> data/shap/feature_bar.png")

shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLS])
shap_df.to_csv(Paths.SHAP_VALUES, index=False)

logger.info("  Saved -> data/shap/shap_values.csv")
logger.info("\nSHAP explainability complete.")