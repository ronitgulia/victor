import os
import sqlite3
import pandas as pd
from unittest.mock import patch
import pytest

from feature_engineering import engineer_features, FEATURE_COLS
from database import TrafficDatabase
from config import Paths

@pytest.fixture
def mock_db(tmp_path):
    """Sets up a temporary SQLite database for testing."""
    test_db_path = tmp_path / "test_traffic.db"
    test_features_path = tmp_path / "test_features.csv"
    
    with patch("database.DB_PATH", str(test_db_path)), \
         patch("database.Paths.TRAFFIC_DB", str(test_db_path)), \
         patch("feature_engineering.Paths.FEATURES", str(test_features_path)):
        
        db = TrafficDatabase()
        
        # Insert test data: 1 Human, 1 Bot
        db.log_request(
            timestamp="2026-01-01T10:00:00Z", ip="1.2.3.4", 
            user_agent="Mozilla/5.0", referer="https://google.com", 
            accept_lang="en-US", path="/home", method="GET", 
            status_code=200, session_id="s1", header_count=8, 
            accept_encoding="gzip, deflate, br", has_accept=1
        )
        db.log_request(
            timestamp="2026-01-01T10:00:01Z", ip="5.6.7.8", 
            user_agent="python-requests/2.25", referer="none", 
            accept_lang="none", path="/secret", method="GET", 
            status_code=403, session_id="s2", header_count=3, 
            accept_encoding="none", has_accept=0
        )
        db.close()
        yield str(test_features_path)

def test_engineer_features(mock_db):
    """Test feature extraction generates the correct features."""
    # Ensure engineer_features uses the patched paths
    with patch("feature_engineering.Paths.FEATURES", mock_db), \
         patch("database.DB_PATH", mock_db.replace("test_features.csv", "test_traffic.db")):
        
        features_df = engineer_features()
        
        assert not features_df.empty, "Features dataframe should not be empty"
        assert len(features_df) == 2, "Should have processed exactly 2 records"
        
        # Check if all required columns are present
        for col in FEATURE_COLS:
            assert col in features_df.columns, f"Missing feature column: {col}"
        assert "label" in features_df.columns, "Missing label column"
        
        # Check specific computed values
        # IP 1.2.3.4 is Human (label=0)
        human = features_df[features_df["ip"] == "1.2.3.4"].iloc[0]
        assert human["label"] == 0
        assert human["ua_is_suspicious"] == 0
        assert human["has_referer"] == 1
        assert human["has_accept_lang"] == 1
        assert human["hit_secret_page"] == 0
        assert human["accept_encoding_score"] == 2  # contains 'br'
        
        # IP 5.6.7.8 is Bot (label=1)
        bot = features_df[features_df["ip"] == "5.6.7.8"].iloc[0]
        assert bot["label"] == 1
        assert bot["ua_is_suspicious"] == 1
        assert bot["has_referer"] == 0
        assert bot["has_accept_lang"] == 0
        assert bot["hit_secret_page"] == 1
        assert bot["accept_encoding_score"] == 0  # contains 'none'
