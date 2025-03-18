"""
Logging Utility Module
"""
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_level=logging.INFO, log_to_console=True, log_to_file=True, log_dir="./logs"):
    """
    Set up a logger with the specified configuration
    
    Args:
        name (str): Logger name
        log_level (int): Logging level
        log_to_console (bool): Whether to log to console
        log_to_file (bool): Whether to log to file
        log_dir (str): Directory for log files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if log_to_file:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log file path with date in name
        current_date = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{name}_{current_date}.log")
        
        # Create rotating file handler (max 10MB per file, max 10 files)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=10
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def configure_global_logging(log_level=logging.INFO, log_dir="./logs"):
    """
    Configure global logging settings
    
    Args:
        log_level (int): Logging level
        log_dir (str): Directory for log files
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = setup_logger("root", log_level=log_level, log_dir=log_dir)
    
    # Set default exception handler
    sys.excepthook = handle_exception
    
    return root_logger

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Custom exception handler to log unhandled exceptions
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    # Skip KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Log the exception
    logger = logging.getLogger("root")
    logger.error(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
