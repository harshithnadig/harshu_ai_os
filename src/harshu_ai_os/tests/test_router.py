import pytest
from harshu_ai_os.llm.router import (
    choose_route,
    SIMPLE_MODEL,
    GENERAL_MODEL,
    REASONING_MODEL,
)


def test_choose_route_invalid():
    with pytest.raises(ValueError):
        choose_route("unknown")


def test_choose_route_simple():
    route = choose_route("simple")

    assert route["model"] == SIMPLE_MODEL
    assert route["max_tokens"] == 80


def test_choose_route_general():
    route = choose_route("general")

    assert route["model"] == GENERAL_MODEL
    assert route["max_tokens"] == 500


def test_choose_route_complex():
    route = choose_route("complex")

    assert route["model"] == REASONING_MODEL
    assert route["max_tokens"] == 1000
