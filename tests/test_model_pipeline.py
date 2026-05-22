import os
import pandas as pd
from unittest.mock import patch
import pytest
from pathlib import Path

from train_model import main
from config import Paths

@pytest.fixture
def synthetic_features(tmp_path):
    """Create a small synthetic dataset for training."""
    test_features_csv = tmp_path / "test_features.csv"
    
    # Needs a minimum number of records and both classes
    data = {
        "ip": ["1.2.3.4", "5.6.7.8", "9.10.11.12", "13.14.15.16"] * 5,
        "timestamp": [
            "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z", 
            "2026-01-01T10:00:02Z", "2026-01-01T10:00:03Z"
        ] * 5,
        "ua_is_suspicious": [0, 1, 0, 1] * 5,
        "has_referer": [1, 0, 1, 0] * 5,
        "has_accept_lang": [1, 0, 1, 0] * 5,
        "hit_secret_page": [0, 1, 0, 1] * 5,
        "ua_length": [120, 20, 115, 25] * 5,
        "time_gap_seconds": [5.0, 0.1, 4.5, 0.2] * 5,
        "unique_pages_visited": [3, 15, 4, 20] * 5,
        "total_requests_from_ip": [10, 50, 12, 60] * 5,
        "is_datacenter_ip": [0, 1, 0, 1] * 5,
        "header_count": [8, 3, 7, 2] * 5,
        "missing_common_headers": [0, 2, 0, 3] * 5,
        "accept_encoding_score": [2, 0, 1, 0] * 5,
        "label": [0, 1, 0, 1] * 5
    }
    df = pd.DataFrame(data)
    df.to_csv(test_features_csv, index=False)
    
    return str(test_features_csv)

def test_model_training_pipeline(synthetic_features, tmp_path):
    """Test that the model pipeline runs end-to-end without crashing."""
    
    test_predictions = str(tmp_path / "test_predictions.csv")
    test_iso_model = str(tmp_path / "test_iso.pkl")
    test_xgb_model = str(tmp_path / "test_xgb.pkl")
    test_versions_dir = tmp_path / "versions"
    test_versions_dir.mkdir(exist_ok=True)
    test_feature_cols = str(tmp_path / "test_feature_cols.json")
    test_ensemble_weights = str(tmp_path / "test_ensemble_weights.json")
    test_metrics = str(tmp_path / "test_model_metrics.json")
    
    # Patch all the paths in train_model
    with patch("train_model.Paths.FEATURES", synthetic_features), \
         patch("train_model.Paths.PREDICTIONS", test_predictions), \
         patch("train_model.Paths.ISO_FOREST_MODEL", test_iso_model), \
         patch("train_model.Paths.XGB_MODEL", test_xgb_model), \
         patch("train_model.Paths.VERSIONS_DIR", test_versions_dir), \
         patch("train_model.Paths.FEATURE_COLS", test_feature_cols), \
         patch("train_model.Paths.ENSEMBLE_WEIGHTS", test_ensemble_weights), \
         patch("train_model.Paths.MODEL_METRICS", test_metrics):
         
        # Also need to patch config values if they depend on real paths? No, train_model uses Paths directly.
        
        main()
        
        # Verify outputs were generated
        assert Path(test_predictions).exists()
        assert Path(test_iso_model).exists()
        assert Path(test_xgb_model).exists()
        assert Path(test_feature_cols).exists()
        assert Path(test_ensemble_weights).exists()
        assert Path(test_metrics).exists()
        
        # Check predictions df
        preds_df = pd.read_csv(test_predictions)
        assert len(preds_df) == 20
        assert "iso_score" in preds_df.columns
        assert "xgb_score" in preds_df.columns
        assert "ensemble_score" in preds_df.columns
        assert "victor_flag" in preds_df.columns
