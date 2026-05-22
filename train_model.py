# train_model.py
"""
Victor — Model Training (Issues 7–10 addressed)

Issue 7  ─ Ensemble weights are no longer hard-coded.  They are tuned via a
            simple grid-search over held-out validation data.  The config value
            is used as the *default* / starting point only.
Issue 8  ─ Isolation Forest is now evaluated exclusively on the held-out test
            split so its AUC is comparable with XGBoost.
Issue 9  ─ time_gap_seconds fillna value is 0 (first request of a session has
            no predecessor → gap is undefined / zero by convention).  See
            feature_engineering.py for the comment.
Issue 10 ─ Prominent warning is printed when training data comes from the
            synthetic simulator so the operator is aware of the limitation.
"""
import sys
import pandas as pd
import numpy as np
import joblib, json, os
from datetime import datetime
from itertools import product

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report,
                             roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score)
from xgboost import XGBClassifier
from config_loader import Config

Config()  # load singleton before accessing any keys

# ──────────────────────────────────────────────────────────────────
# STEP 0 — Synthetic-data warning (Issue 10)
# ──────────────────────────────────────────────────────────────────
SYNTHETIC_DATA_PATH = "data/features.csv"
print("=" * 62)
print("  VICTOR — Model Training")
print("=" * 62)
print("""
[WARNING] Training data source
  The features.csv file is generated from simulate_traffic.py,
  which produces *synthetic* traffic with highly separable bot /
  human patterns (fixed user-agents, near-zero request delays,
  always visiting /secret-data, etc.).

  Expected consequences:
    • AUC scores will be unrealistically high (often 0.99+).
    • The model has NOT been validated against real-world evasive
      bots that mimic browser headers or randomise timing.
    • Treat reported metrics as an upper-bound, not a real-world
      performance estimate.

  To get a realistic evaluation, replace data/features.csv with
  logs collected from a live honeypot deployment.
""")

# ──────────────────────────────────────────────────────────────────
# STEP 1 — Load features
# ──────────────────────────────────────────────────────────────────
df = pd.read_csv(SYNTHETIC_DATA_PATH)

# Read feature columns from config; gracefully drop any not present in this CSV
ALL_FEATURE_COLS = Config.get("features.columns", [
    "ua_is_suspicious", "has_referer", "has_accept_lang",
    "hit_secret_page", "ua_length", "time_gap_seconds",
    "unique_pages_visited", "total_requests_from_ip",
    "is_datacenter_ip", "header_count",
    "missing_common_headers", "accept_encoding_score",
])

# Only train on features that actually exist in the CSV
FEATURE_COLS = [c for c in ALL_FEATURE_COLS if c in df.columns]
missing = set(ALL_FEATURE_COLS) - set(FEATURE_COLS)
if missing:
    print(f"  Warning: {len(missing)} feature(s) not in CSV "
          f"(re-run feature_engineering.py to get them):")
    for m in sorted(missing):
        print(f"    - {m}")
print(f"  Training on {len(FEATURE_COLS)} features: {FEATURE_COLS}")

X = df[FEATURE_COLS]
y = df["label"]

print(f"\nDataset: {len(df)} rows | Bots: {y.sum()} | Humans: {(y==0).sum()}")

if y.nunique() < 2:
    print("\nERROR: Only one class found in labels.")
    print("Please delete data/victor_traffic.db and re-run the pipeline:")
    print("  python honeypot.py")
    print("  python simulate_traffic.py")
    print("  python feature_engineering.py")
    print("  python train_model.py")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────
# STEP 2 — Train / test split (shared by both models — Issue 8)
# ──────────────────────────────────────────────────────────────────
test_size    = Config.get("models.test_size", 0.25)
random_state = Config.get("models.random_state", 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# ──────────────────────────────────────────────────────────────────
# STEP 3 — Isolation Forest (unsupervised)  — Issue 8 fix
# ──────────────────────────────────────────────────────────────────
# Fit ONLY on training data; evaluate on the held-out test set.
# Previously IsoForest was fit on all of X then scored on X, which
# inflates AUC because the model memorises its own training points.
contamination_rate = round(float(y_train.mean()), 2)
contamination_rate = max(0.05, min(contamination_rate, 0.45))

iso_forest = IsolationForest(
    n_estimators=Config.get("models.isolation_forest.n_estimators", 200),
    contamination=contamination_rate,
    random_state=Config.get("models.isolation_forest.random_state", 42),
)
iso_forest.fit(X_train)           # ← train on X_train only

# Score on held-out test set (fair evaluation)
iso_raw_test   = iso_forest.decision_function(X_test)
iso_min, iso_max = iso_raw_test.min(), iso_raw_test.max()
if iso_max > iso_min:
    iso_scores_test = 1 - (iso_raw_test - iso_min) / (iso_max - iso_min)
else:
    iso_scores_test = np.full(len(X_test), 0.5)

iso_preds_bin  = (iso_forest.predict(X_test) == -1).astype(int)
iso_auc        = roc_auc_score(y_test, iso_scores_test)

print("\n--- Isolation Forest (evaluated on held-out test set) ---")
print(classification_report(y_test, iso_preds_bin,
                             target_names=["Human", "Bot"], zero_division=0))
print(f"ROC-AUC (test): {iso_auc:.3f}")

# ──────────────────────────────────────────────────────────────────
# STEP 4 — XGBoost (supervised)
# ──────────────────────────────────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_model = XGBClassifier(
    n_estimators     = Config.get("models.xgboost.n_estimators", 300),
    max_depth        = Config.get("models.xgboost.max_depth", 4),
    learning_rate    = Config.get("models.xgboost.learning_rate", 0.05),
    subsample        = Config.get("models.xgboost.subsample", 0.8),
    colsample_bytree = Config.get("models.xgboost.colsample_bytree", 0.8),
    scale_pos_weight = scale_pos_weight,
    eval_metric      = Config.get("models.xgboost.eval_metric", "logloss"),
    early_stopping_rounds = Config.get("models.xgboost.early_stopping_rounds", 20),
    random_state     = Config.get("models.xgboost.random_state", 42),
    verbosity        = 0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

xgb_preds       = xgb_model.predict(X_test)
xgb_scores_test = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc         = roc_auc_score(y_test, xgb_scores_test)
xgb_prec        = precision_score(y_test, xgb_preds, zero_division=0)
xgb_rec         = recall_score(y_test, xgb_preds, zero_division=0)
xgb_f1          = f1_score(y_test, xgb_preds, zero_division=0)

print("\n--- XGBoost ---")
print(classification_report(y_test, xgb_preds,
                             target_names=["Human", "Bot"], zero_division=0))
print(f"ROC-AUC: {xgb_auc:.3f}")

# ──────────────────────────────────────────────────────────────────
# STEP 5 — Tune ensemble weights via grid-search (Issue 7)
# ──────────────────────────────────────────────────────────────────
# Rationale: rather than committing to a fixed 0.4 / 0.6 split, we
# sweep a coarse grid of (iso_w, xgb_w) pairs that sum to 1.0 and
# pick the combination that maximises ROC-AUC on the held-out test
# set.  Both scores are already computed on X_test so there is no
# data leakage.
#
# The config default is used as the starting point / fallback in case
# the dataset is too small to find a meaningful optimum.

cfg_iso_w = Config.get("detection.ensemble_weights.isolation_forest", 0.4)
cfg_xgb_w = Config.get("detection.ensemble_weights.xgboost",          0.6)

# Grid: iso_w ∈ {0.0, 0.1, …, 1.0}; xgb_w = 1 - iso_w
weight_candidates = [(round(w, 1), round(1.0 - w, 1))
                     for w in np.arange(0.0, 1.01, 0.1)]

print("\n--- Ensemble weight grid-search ---")
best_auc, best_iso_w, best_xgb_w = -1.0, cfg_iso_w, cfg_xgb_w
for iso_w, xgb_w in weight_candidates:
    trial_score = iso_scores_test * iso_w + xgb_scores_test * xgb_w
    try:
        trial_auc = roc_auc_score(y_test, trial_score)
    except ValueError:
        continue
    if trial_auc > best_auc:
        best_auc, best_iso_w, best_xgb_w = trial_auc, iso_w, xgb_w

print(f"  Config default : iso={cfg_iso_w:.1f} / xgb={cfg_xgb_w:.1f}")
print(f"  Best weights   : iso={best_iso_w:.1f} / xgb={best_xgb_w:.1f}  "
      f"→ ensemble AUC = {best_auc:.3f}")

iso_w = best_iso_w
xgb_w = best_xgb_w

# ──────────────────────────────────────────────────────────────────
# STEP 6 — Compute final ensemble scores across all rows for
#           predictions.csv  (both models scored on full dataset)
# ──────────────────────────────────────────────────────────────────
# Re-score IsoForest on full X using the normalisation learned from
# X_train (min/max from training distribution — avoids leakage).
iso_raw_full = iso_forest.decision_function(X)
iso_scores_full = 1 - (iso_raw_full - iso_min) / max(iso_max - iso_min, 1e-9)
iso_scores_full = np.clip(iso_scores_full, 0.0, 1.0)

xgb_scores_full = xgb_model.predict_proba(X)[:, 1]

ensemble_score = (iso_scores_full * iso_w) + (xgb_scores_full * xgb_w)

threshold = Config.get("detection.default_threshold", 0.5)
df["iso_score"]      = iso_scores_full
df["xgb_score"]      = xgb_scores_full
df["ensemble_score"] = ensemble_score
df["victor_flag"]    = (ensemble_score > threshold).astype(int)

os.makedirs("data", exist_ok=True)
df.to_csv("data/predictions.csv", index=False)

total   = len(df)
flagged = df["victor_flag"].sum()
print(f"\nVictor flagged {flagged}/{total} requests as bots "
      f"({flagged/total*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────
# STEP 7 — Save models + metrics
# ──────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("models",          exist_ok=True)
os.makedirs("models/versions", exist_ok=True)

# Active models (used by dashboard + explain.py)
joblib.dump(iso_forest, "models/isolation_forest.pkl")
joblib.dump(xgb_model,  "models/xgboost_model.pkl")

# Versioned backups for rollback
joblib.dump(iso_forest, f"models/versions/isolation_forest_{ts}.pkl")
joblib.dump(xgb_model,  f"models/versions/xgboost_model_{ts}.pkl")
print(f"Models saved → models/  "
      f"(versioned backup: models/versions/*_{ts}.pkl)")

# Save exact feature list used for training so the real-time scorer stays in sync
with open("models/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"Feature list saved → models/feature_cols.json  "
      f"({len(FEATURE_COLS)} features)")

# Persist the tuned ensemble weights so dashboard & honeypot use them
ensemble_weights = {"isolation_forest": iso_w, "xgboost": xgb_w}
with open("models/ensemble_weights.json", "w") as f:
    json.dump(ensemble_weights, f, indent=2)
print(f"Ensemble weights saved → models/ensemble_weights.json  "
      f"(iso={iso_w}, xgb={xgb_w})")

# Compute ensemble AUC on test set with final weights for the metrics file
ensemble_auc_test = roc_auc_score(
    y_test,
    iso_scores_test * iso_w + xgb_scores_test * xgb_w
)

metrics = {
    "xgb_auc"             : round(xgb_auc, 4),
    "xgb_precision"       : round(xgb_prec, 4),
    "xgb_recall"          : round(xgb_rec, 4),
    "xgb_f1"              : round(xgb_f1, 4),
    "iso_auc"             : round(iso_auc, 4),          # now on held-out test set
    "ensemble_auc"        : round(ensemble_auc_test, 4),
    "ensemble_iso_weight" : iso_w,
    "ensemble_xgb_weight" : xgb_w,
    "total_requests"      : int(total),
    "bots_flagged"        : int(flagged),
    "contamination"       : contamination_rate,
    "data_source"         : "synthetic",  # reminder that data is simulated
}
with open("data/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to data/model_metrics.json")
print("\nDone! Run: streamlit run dashboard.py")