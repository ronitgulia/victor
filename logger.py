import logging
import sys
def setup_logger(name: str) -> logging.Logger:
    """Creates a configured logger based on settings in config.yaml."""
    from config_loader import Config
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, it's already configured
    if logger.hasHandlers():
        return logger

    log_level_str = Config.get("logging.level", "INFO").upper()
    log_format = Config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    
    # Prevent duplicate logs if attaching to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    return setup_logger(name)
