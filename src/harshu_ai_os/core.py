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


def get_omniroute_config() -> tuple[str, str]:
    """Return (base_url, api_key) for the OmniRoute gateway without printing secrets."""
    load_dotenv()
    base_url = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1")
    api_key = os.getenv("OMNIROUTE_API_KEY")
    if not api_key:
        from pathlib import Path
        local_env = Path(__file__).resolve().parent.parent.parent / "omiroute" / ".env"
        if local_env.exists():
            with open(local_env, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("OMNIROUTE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    return base_url, api_key or "sk-dummy"

