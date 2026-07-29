import logging
import sys

from src.config import LOGGING_FORMAT


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger bound to `name` (pass __name__)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger