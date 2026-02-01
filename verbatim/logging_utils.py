import logging
import sys
from typing import Optional

STATUS_LOGGER_NAME = "verbatim.status"


def get_status_logger() -> logging.Logger:
    return logging.getLogger(STATUS_LOGGER_NAME)


def status_enabled() -> bool:
    logger = get_status_logger()
    if not logger.handlers and not logger.propagate:
        return False
    return logger.isEnabledFor(logging.INFO)


def configure_status_logger(*, verbose: int, fmt: str, datefmt: Optional[str]) -> None:
    logger = get_status_logger()
    logger.handlers.clear()
    logger.propagate = False

    if verbose <= 0:
        logger.setLevel(logging.CRITICAL + 1)
        return

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
