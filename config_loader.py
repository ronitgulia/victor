"""
Config loader for Victor bot detection system.
Centralizes all configuration values from config.yaml.
"""

import yaml
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Load and provide access to config.yaml values"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - only load config once"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """Load config.yaml from project root"""
        config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        
        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)
    
    @classmethod
    def get(cls, key: str, default=None):
        """Get config value by dot-notation key
        
        Examples:
            Config.get('paths.features')  -> Path('data/features.csv')
            Config.get('detection.default_threshold')  -> 0.5
            Config.get('models.xgboost.n_estimators')  -> 300
        """
        if cls._config is None:
            cls()
        config = cls._config
        keys = key.split('.')
        
        try:
            for k in keys:
                config = config[k]
            
            # If it's a path, resolve it relative to project root
            if keys[0] == 'paths' and isinstance(config, str):
                return Path(__file__).parent / config
                
            return config
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config key not found: {key}")
    
    @classmethod
    def get_section(cls, section: str):
        """Get entire config section as dict
        
        Examples:
            Config.get_section('paths')
            Config.get_section('models')
        """
        if cls._config is None:
            cls()
        return cls._config.get(section, {})
    
    @classmethod
    def reload(cls):
        """Force reload config from disk"""
        cls._config = None
        cls._instance = None
        return Config()


# Convenience shortcuts
def get_paths():
    """Get all file paths"""
    return Config.get_section('paths')


def get_features():
    """Get feature configuration"""
    return Config.get_section('features')


def get_models_config():
    """Get model training configuration"""
    return Config.get_section('models')


def get_detection_config():
    """Get detection settings"""
    return Config.get_section('detection')


def get_simulation_config():
    """Get traffic simulation settings"""
    return Config.get_section('simulation')


def get_dashboard_config():
    """Get dashboard settings"""
    return Config.get_section('dashboard')


# Example usage
if __name__ == "__main__":
    config = Config()
    
    logger.info("=== Victor Configuration ===\n")
    
    logger.info("FILE PATHS:")
    logger.info(f"  Features: {config.get('paths.features')}")
    logger.info(f"  Predictions: {config.get('paths.predictions')}")
    logger.info(f"  XGBoost Model: {config.get('paths.xgboost_model')}")
    
    logger.info("\nMODEL TRAINING:")
    logger.info(f"  XGBoost n_estimators: {config.get('models.xgboost.n_estimators')}")
    logger.info(f"  Isolation Forest n_estimators: {config.get('models.isolation_forest.n_estimators')}")
    
    logger.info("\nDETECTION:")
    logger.info(f"  Default Threshold: {config.get('detection.default_threshold')}")
    logger.info(f"  Ensemble Method: {config.get('detection.ensemble_method')}")
    
    logger.info("\nSIMULATION:")
    logger.info(f"  Base URL: {config.get('simulation.base_url')}")
    logger.info(f"  Human Sessions: {config.get('simulation.human.num_sessions')}")
    logger.info(f"  Bot Sessions: {config.get('simulation.bot.num_sessions')}")
    
    logger.info("\nDASHBOARD:")
    logger.info(f"  Cache TTL: {config.get('dashboard.cache_ttl')} seconds")
    logger.info(f"  Activity Feed Size: {config.get('dashboard.activity_feed_size')}")
    
    logger.info("\nFEATURES:")
    features = config.get('features.columns')
    logger.info(f"  Total Features: {len(features)}")
    for feat in features:
        logger.info(f"    - {feat}")
