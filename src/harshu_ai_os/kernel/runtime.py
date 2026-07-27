"""Shared runtime configuration, identity, and logging for Harshu AI OS."""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv


class MissingConfigError(Exception):
    """Raised when a required runtime setting is unavailable."""


def get_app_mode() -> str:
    """Load the local environment and return the configured application mode."""
    load_dotenv()
    app_mode = os.getenv("HARSHU_AI_OS_MODE")

    if app_mode is None:
        raise MissingConfigError("HARSHU_AI_OS_MODE is missing")

    return app_mode


@dataclass(frozen=True)
class RuntimeProfile:
    """Small immutable identity object shared by runtime entry points."""

    system_name: str
    mode: str
    version: str = "0.1.0"

    def show_summary(self) -> str:
        return (
            f"System Name: {self.system_name}\n"
            f"Mode: {self.mode}\n"
            f"Version: {self.version}"
        )


def get_logger(name: str) -> logging.Logger:
    """Return one configured logger without duplicating handlers on reload."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger
