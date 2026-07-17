"""
utils.py

Small reusable helper functions used across the project.
Currently: centralized logging setup.
"""

import logging
import time
from functools import wraps

import config


def setup_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger that writes to both
    the console and a log file (config.LOG_FILE).

    Args:
        name: usually __name__ from the calling module,
              so log lines show which file they came from.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — so you still see logs live while testing
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — so logs persist across runs for later debugging
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def timeit(func):
    """
    Decorator that logs how long a function took to run.
    Useful for measuring latency of each pipeline stage
    (STT, LLM, TTS) to optimize later.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper