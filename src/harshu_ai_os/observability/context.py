"""Concurrency-safe request context for request IDs using contextvars."""

import re
import uuid
from contextvars import ContextVar, Token

# ContextVar for storing the current request ID per async task / thread context.
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

# Allow alphanumeric, hyphen, underscore, dot, colon. Length 1-128.
REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9_\.\:\-]+$")
MAX_REQUEST_ID_LENGTH = 128


def sanitize_request_id(raw_id: str | None) -> str | None:
    """Validate and sanitize a caller-supplied request ID.

    Returns the sanitized string if valid, or None if missing, malformed,
    empty, or overly long.
    """
    if raw_id is None:
        return None

    cleaned = raw_id.strip()
    if not cleaned or len(cleaned) > MAX_REQUEST_ID_LENGTH:
        return None

    if not REQUEST_ID_REGEX.match(cleaned):
        return None

    return cleaned


def generate_request_id() -> str:
    """Generate a standard UUID4 string for a new request."""
    return str(uuid.uuid4())


def get_request_id() -> str | None:
    """Get the active request ID for the current context."""
    return _request_id_ctx.get()


def set_request_id(request_id: str) -> Token[str | None]:
    """Set the active request ID in contextvars and return the reset token."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Reset the request ID contextvar to its previous state using the token."""
    _request_id_ctx.reset(token)
