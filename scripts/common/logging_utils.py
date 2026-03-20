"""
Shared logging setup for all pipeline scripts.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path, log_filename: str) -> logging.Logger:
    """
    Configure logging to both file and console.

    Args:
        log_dir: Directory for log files
        log_filename: Name of the log file (e.g., "train_embeddings.log")

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename

    logger = logging.getLogger(log_filename.replace(".log", ""))
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
