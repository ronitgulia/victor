# explain.py
# SHAP tells us WHY Victor flagged something as a bot
# instead of just saying "this is a bot", it shows which features caused it
# this is what separates a good ML project from a great one

import shap
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # stops matplotlib from trying to open a window
import matplotlib.pyplot as plt
import os

os.makedirs("data/shap", exist_ok=True)

# ─────────────────────────────────────────────
# load our trained XGBoost model + feature data
# ─────────────────────────────────────────────

model = joblib.load("models/xgboost_model.pkl")
df    = pd.read_csv("data/features.csv")

FEATURE_COLS = [
    "ua_is_suspicious",
    "has_referer",
    "has_accept_lang",
    "hit_secret_page",
    "ua_length",
    "time_gap_seconds",
    "unique_pages_visited",
    "total_requests_from_ip"
]

X = df[FEATURE_COLS]

# ─────────────────────────────────────────────
# STEP 1 — compute SHAP values
# ─────────────────────────────────────────────
# TreeExplainer is the fast version made specifically for tree-based models
# it calculates how much each feature "pushed" the prediction toward bot or human

print("Computing SHAP values (this takes a few seconds)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Done.")

# ─────────────────────────────────────────────
# STEP 2 — global summary plot
# which features matter most across ALL requests?
# ─────────────────────────────────────────────

print("Saving global feature importance plot...")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("Victor — Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("data/shap/global_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Saved -> data/shap/global_summary.png")

# ─────────────────────────────────────────────
# STEP 3 — bar chart version (cleaner for dashboards)
# ─────────────────────────────────────────────

print("Saving bar chart version...")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Victor — Average Feature Impact")
plt.tight_layout()
plt.savefig("data/shap/feature_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Saved -> data/shap/feature_bar.png")

# ─────────────────────────────────────────────
# STEP 4 — save SHAP values to CSV for the dashboard
# ─────────────────────────────────────────────

shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLS])
shap_df.to_csv("data/shap/shap_values.csv", index=False)

print("  Saved -> data/shap/shap_values.csv")
print("\nSHAP explainability complete.")