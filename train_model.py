# train_model.py  —  Victor's ensemble ML pipeline
# Trains Isolation Forest (unsupervised) + XGBoost (supervised)
# then blends their scores into a final ensemble bot probability.
#
# Run:  python train_model.py

import pandas as pd
import numpy as np
import joblib, json, os
from sklearn.ensemble        import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (classification_report,
                                     roc_auc_score,
                                     precision_score,
                                     recall_score,
                                     f1_score)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────
# STEP 1 — load features
# ─────────────────────────────────────────────────────────────────

df = pd.read_csv("data/features.csv")

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
y = df["label"]

print(f"Dataset: {len(df)} rows | Bots: {y.sum()} | Humans: {(y==0).sum()}")

# safety check — we need both classes to train
if y.nunique() < 2:
    print("\nERROR: Only one class found in labels.")
    print("Please delete data/traffic_logs.json and re-run the pipeline:")
    print("  python honeypot.py        (start server, Ctrl+C after traffic sim)")
    print("  python simulate_traffic.py")
    print("  python feature_engineering.py")
    print("  python train_model.py")
    exit(1)

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Isolation Forest (unsupervised anomaly detection)
# ─────────────────────────────────────────────────────────────────

contamination_rate = round(float(y.mean()), 2)
contamination_rate = max(0.05, min(contamination_rate, 0.45))

iso_forest = IsolationForest(
    n_estimators  = 200,
    contamination = contamination_rate,
    random_state  = 42
)
iso_forest.fit(X)

iso_preds_raw = iso_forest.predict(X)
iso_preds_bin = (iso_preds_raw == -1).astype(int)          # -1 → anomaly → bot

iso_raw    = iso_forest.decision_function(X)
iso_scores = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min())  # 0-1 scaled

iso_auc = roc_auc_score(y, iso_scores)
print("\n--- Isolation Forest ---")
print(classification_report(y, iso_preds_bin, target_names=["Human", "Bot"], zero_division=0))
print(f"ROC-AUC: {iso_auc:.3f}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — XGBoost (supervised classification)
# ─────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# class weight to handle any imbalance
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_model = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 4,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos_weight,
    eval_metric       = "logloss",
    early_stopping_rounds = 20,
    random_state      = 42,
    verbosity         = 0
)

xgb_model.fit(
    X_train, y_train,
    eval_set          = [(X_test, y_test)],
    verbose           = False
)

xgb_preds       = xgb_model.predict(X_test)
xgb_scores_test = xgb_model.predict_proba(X_test)[:, 1]

xgb_auc  = roc_auc_score(y_test, xgb_scores_test)
xgb_prec = precision_score(y_test, xgb_preds, zero_division=0)
xgb_rec  = recall_score(y_test, xgb_preds, zero_division=0)
xgb_f1   = f1_score(y_test, xgb_preds, zero_division=0)

print("\n--- XGBoost ---")
print(classification_report(y_test, xgb_preds, target_names=["Human", "Bot"], zero_division=0))
print(f"ROC-AUC: {xgb_auc:.3f}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — ensemble: blend both scores
# ─────────────────────────────────────────────────────────────────

xgb_scores_full = xgb_model.predict_proba(X)[:, 1]
ensemble_score  = (iso_scores * 0.4) + (xgb_scores_full * 0.6)   # XGBoost weighted more

df["iso_score"]      = iso_scores
df["xgb_score"]      = xgb_scores_full
df["ensemble_score"] = ensemble_score
df["victor_flag"]    = (ensemble_score > 0.5).astype(int)

df.to_csv("data/predictions.csv", index=False)

total   = len(df)
flagged = df["victor_flag"].sum()
print(f"\nVictor flagged {flagged}/{total} requests as bots ({flagged/total*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — save models + metrics JSON
# ─────────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(iso_forest, "models/isolation_forest.pkl")
joblib.dump(xgb_model,  "models/xgboost_model.pkl")
print("Models saved to models/")

metrics = {
    "xgb_auc"        : round(xgb_auc,  4),
    "xgb_precision"  : round(xgb_prec, 4),
    "xgb_recall"     : round(xgb_rec,  4),
    "xgb_f1"         : round(xgb_f1,   4),
    "iso_auc"        : round(iso_auc,   4),
    "total_requests" : int(total),
    "bots_flagged"   : int(flagged),
    "contamination"  : contamination_rate
}
with open("data/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to data/model_metrics.json")
print("\nDone! Run: streamlit run dashboard.py")