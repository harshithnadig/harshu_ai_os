"""Tiny application-wide helpers shared by the API and local scripts."""

import logging
import os

from dotenv import load_dotenv


class MissingConfigError(Exception):
    """Raised when a required local configuration value is missing."""


def get_app_mode() -> str:
    """Load the local environment and return the selected app mode."""
    load_dotenv()
    app_mode = os.getenv("HARSHU_AI_OS_MODE")

    if app_mode is None:
        raise MissingConfigError("HARSHU_AI_OS_MODE is missing")

    return app_mode


def get_logger(name: str) -> logging.Logger:
    """Return one readable console logger without duplicate handlers."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger
