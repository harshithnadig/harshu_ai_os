"""Tests for the few helpers shared by application entry points."""

import logging

import pytest

from harshu_ai_os.core import MissingConfigError, get_app_mode, get_logger


def test_get_app_mode_rejects_missing_setting(monkeypatch) -> None:
    monkeypatch.delenv("HARSHU_AI_OS_MODE", raising=False)
    monkeypatch.setattr("harshu_ai_os.core.load_dotenv", lambda: None)

    with pytest.raises(MissingConfigError, match="HARSHU_AI_OS_MODE is missing"):
        get_app_mode()


def test_logger_does_not_duplicate_handlers() -> None:
    logger_name = "harshu_ai_os.tests.core"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    assert get_logger(logger_name) is get_logger(logger_name)
    assert len(logger.handlers) == 1
