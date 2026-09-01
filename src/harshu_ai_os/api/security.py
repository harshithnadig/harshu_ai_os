"""Security baseline: authentication, deterministic rate limiting, and request limits."""

import hmac
import os
import threading
import time

from fastapi import Header, HTTPException, Request

MAX_PAYLOAD_BYTES = 65536  # 64 KB max payload size for AI endpoints
DEFAULT_RATE_LIMIT_PER_MINUTE = 60


def is_auth_disabled() -> bool:
    """Check if authentication is explicitly disabled for local development only.

    Accepts deliberately narrow truthy values: '1', 'true', 'yes' (case-insensitive).
    Any other value (including unset or empty) evaluates to False (fail-closed).
    """
    val = os.getenv("HARSHU_AUTH_DISABLED", "").strip().lower()
    return val in ("1", "true", "yes")


def get_configured_api_key() -> str | None:
    """Retrieve the configured API key from environment, or None if unconfigured."""
    key = os.getenv("HARSHU_API_KEY", "").strip()
    return key if key else None


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> None:
    """Enforce fail-closed API key authentication for protected endpoints.

    State Machine:
    1. If HARSHU_AUTH_DISABLED is explicitly truthy ('1', 'true', 'yes'):
       -> Request permitted (explicit local development bypass only).
    2. If HARSHU_API_KEY is not configured:
       -> Fails closed with HTTP 503 Service Unavailable (server misconfigured).
    3. If HARSHU_API_KEY is configured:
       -> If client provides matching X-API-Key or Bearer token (constant-time):
          -> Request permitted.
       -> Otherwise:
          -> HTTP 401 Unauthorized.
    """
    if is_auth_disabled():
        return

    required_key = get_configured_api_key()
    if not required_key:
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable: server authentication is unconfigured",
        )

    provided_key: str | None = x_api_key
    if not provided_key and authorization:
        if authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()

    if not provided_key or not hmac.compare_digest(provided_key, required_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


class InMemoryRateLimiter:
    """Deterministic thread-safe sliding-window rate limiter."""

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE) -> None:
        self.requests_per_minute = requests_per_minute
        self._window_seconds = 60.0
        self._records: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str, now: float | None = None) -> tuple[bool, int]:
        """Check if client request is within allowed rate.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        current_time = now if now is not None else time.time()
        cutoff = current_time - self._window_seconds

        with self._lock:
            timestamps = self._records.get(client_id, [])
            valid = [t for t in timestamps if t > cutoff]

            if len(valid) >= self.requests_per_minute:
                oldest = valid[0]
                retry_after = max(1, int(oldest + self._window_seconds - current_time))
                self._records[client_id] = valid
                return False, retry_after

            valid.append(current_time)
            self._records[client_id] = valid
            return True, 0

    def reset(self) -> None:
        """Clear all rate limiting records."""
        with self._lock:
            self._records.clear()


# Default process-level rate limiter
global_rate_limiter = InMemoryRateLimiter()


def check_rate_limit(request: Request) -> None:
    """FastAPI dependency to enforce rate limits per client IP / auth token."""
    if os.getenv("HARSHU_RATE_LIMIT_DISABLED", "").lower() in ("1", "true"):
        return

    client_id = "unknown"
    if request.client and request.client.host:
        client_id = request.client.host

    allowed, retry_after = global_rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )
