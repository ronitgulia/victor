from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SHAP_DIR = DATA_DIR / "shap"
VERSIONS_DIR = MODELS_DIR / "versions"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SHAP_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)

class Paths:
    TRAFFIC_DB = DATA_DIR / "victor_traffic.db"
    TRAFFIC_LOGS = DATA_DIR / "traffic_logs.json"
    FEATURES = DATA_DIR / "features.csv"
    PREDICTIONS = DATA_DIR / "predictions.csv"
    MODEL_METRICS = DATA_DIR / "model_metrics.json"
    DATACENTER_RANGES = DATA_DIR / "datacenter_ranges.json"
    REALTIME_STATS = DATA_DIR / "realtime_stats.json"
    
    SHAP_VALUES = SHAP_DIR / "shap_values.csv"
    SHAP_GLOBAL_SUMMARY = SHAP_DIR / "global_summary.png"
    SHAP_FEATURE_BAR = SHAP_DIR / "feature_bar.png"
    
    ISO_FOREST_MODEL = MODELS_DIR / "isolation_forest.pkl"
    XGB_MODEL = MODELS_DIR / "xgboost_model.pkl"
    FEATURE_COLS = MODELS_DIR / "feature_cols.json"
    ENSEMBLE_WEIGHTS = MODELS_DIR / "ensemble_weights.json"
    VERSIONS_DIR = VERSIONS_DIR
