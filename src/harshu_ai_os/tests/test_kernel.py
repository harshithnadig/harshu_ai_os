import logging

import pytest

from harshu_ai_os.kernel.performance import (
    count_quadratic_operations,
    measure_membership,
)
from harshu_ai_os.kernel.runtime import (
    MissingConfigError,
    RuntimeProfile,
    get_app_mode,
    get_logger,
)


def test_runtime_profile_formats_identity() -> None:
    profile = RuntimeProfile(
        system_name="Harshu AI OS",
        mode="test",
    )

    assert profile.show_summary() == (
        "System Name: Harshu AI OS\nMode: test\nVersion: 0.1.0"
    )


def test_get_app_mode_rejects_missing_setting(monkeypatch) -> None:
    monkeypatch.delenv("HARSHU_AI_OS_MODE", raising=False)
    monkeypatch.setattr(
        "harshu_ai_os.kernel.runtime.load_dotenv",
        lambda: None,
    )

    with pytest.raises(
        MissingConfigError,
        match="HARSHU_AI_OS_MODE is missing",
    ):
        get_app_mode()


def test_logger_does_not_duplicate_handlers() -> None:
    logger_name = "harshu_ai_os.tests.kernel"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    first_logger = get_logger(logger_name)
    second_logger = get_logger(logger_name)

    assert first_logger is second_logger
    assert len(first_logger.handlers) == 1


def test_performance_helpers_return_structured_evidence() -> None:
    results = measure_membership(data_size=10, repeat_count=1)

    assert set(results) == {"list", "set", "dictionary"}
    assert all(elapsed >= 0 for elapsed in results.values())
    assert count_quadratic_operations(4) == 16
