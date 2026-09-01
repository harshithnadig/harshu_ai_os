"""Global pytest fixtures and test environment defaults."""

import os
import pytest


@pytest.fixture(autouse=True)
def default_test_auth_environment(monkeypatch):
    """Ensure tests run with HARSHU_AUTH_DISABLED=true by default unless overridden.

    Security tests in test_security.py explicitly override or delete these environment
    variables to verify fail-closed behavior, 401s, and disabled states.
    """
    if "HARSHU_AUTH_DISABLED" not in os.environ and "HARSHU_API_KEY" not in os.environ:
        monkeypatch.setenv("HARSHU_AUTH_DISABLED", "true")
