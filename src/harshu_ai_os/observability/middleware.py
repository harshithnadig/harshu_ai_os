"""Pure ASGI HTTP observability middleware for request tracing and metrics."""

import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from harshu_ai_os.observability.context import (
    generate_request_id,
    reset_request_id,
    sanitize_request_id,
    set_request_id,
)
from harshu_ai_os.observability.logging import log_event


class ObservabilityMiddleware:
    """Pure ASGI middleware that manages request IDs, duration, and structured logs.

    Implemented as pure ASGI to avoid ContextVar propagation issues in BaseHTTPMiddleware.
    All per-request mutable state is local to __call__ or stored in ContextVars.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_request_id: str | None = None
        for header_name, header_val in scope.get("headers", []):
            if header_name.lower() == b"x-request-id":
                try:
                    raw_request_id = header_val.decode("latin1")
                except Exception:
                    raw_request_id = None
                break

        request_id = sanitize_request_id(raw_request_id) or generate_request_id()
        token = set_request_id(request_id)

        method = scope.get("method", "")
        path = scope.get("path", "")
        start_time = time.perf_counter()
        status_code = 500

        log_event(
            "INFO",
            "http_request_started",
            request_id=request_id,
            method=method,
            path=path,
        )

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
                headers = list(message.get("headers", []))
                has_req_id = any(k.lower() == b"x-request-id" for k, _ in headers)
                if not has_req_id:
                    headers.append((b"x-request-id", request_id.encode("latin1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            duration_ms = max(0.0, round((time.perf_counter() - start_time) * 1000, 2))
            log_event(
                "INFO",
                "http_request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = max(0.0, round((time.perf_counter() - start_time) * 1000, 2))
            log_event(
                "ERROR",
                "http_request_failed",
                request_id=request_id,
                method=method,
                path=path,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            reset_request_id(token)
