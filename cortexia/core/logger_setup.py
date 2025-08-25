import logging
import os
from pathlib import Path
from typing import Optional

def setup_logging(config_manager) -> None:
    """Configure application logging based on config settings.
    
    Args:
        config_manager: ConfigManager instance with logging settings
    """
    log_level = config_manager.get_param('logging.level', 'INFO')
    log_file = config_manager.get_param('logging.file', 'app.log')
    
    # Ensure logs directory exists
    log_dir = Path(log_file).parent
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    # Create logger
    logger = logging.getLogger('cortexia')
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)