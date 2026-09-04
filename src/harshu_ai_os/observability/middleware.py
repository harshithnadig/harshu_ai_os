"""Pure ASGI HTTP observability middleware for request tracing and metrics."""

import json
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from harshu_ai_os.observability.context import (
    generate_request_id,
    reset_request_id,
    sanitize_request_id,
    set_request_id,
)
from harshu_ai_os.observability.logging import log_event

MAX_PAYLOAD_BYTES = 65536  # 64 KiB


class ObservabilityMiddleware:
    """Pure ASGI middleware that manages request IDs, duration, and structured logs.

    Implemented as pure ASGI to avoid ContextVar propagation issues in BaseHTTPMiddleware.
    All per-request mutable state is local to __call__ or stored in ContextVars.
    Enforces maximum payload size against both Content-Length headers and actual received bytes.
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

        async def send_413_payload_too_large() -> None:
            resp_body = json.dumps({"detail": "Request payload too large (max 64KB)"}).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(resp_body)).encode("latin1")),
                    (b"x-request-id", request_id.encode("latin1")),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": resp_body,
            })
            duration_ms = max(0.0, round((time.perf_counter() - start_time) * 1000, 2))
            log_event(
                "INFO",
                "http_request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status_code=413,
                duration_ms=duration_ms,
            )

        # 1. Early rejection optimization if declared Content-Length exceeds limit
        content_length: int | None = None
        for h_name, h_val in scope.get("headers", []):
            if h_name.lower() == b"content-length":
                try:
                    content_length = int(h_val.decode("latin1"))
                except Exception:
                    pass
                break

        if content_length is not None and content_length > MAX_PAYLOAD_BYTES:
            await send_413_payload_too_large()
            reset_request_id(token)
            return

        # 2. Enforce limit against actual bytes received
        received_chunks: list[bytes] = []
        total_bytes = 0
        payload_too_large = False

        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                total_bytes += len(chunk)
                if total_bytes > MAX_PAYLOAD_BYTES:
                    payload_too_large = True
                    break
                received_chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                reset_request_id(token)
                return

        if payload_too_large:
            await send_413_payload_too_large()
            reset_request_id(token)
            return

        # 3. Replay received chunks for downstream application
        chunk_idx = 0

        async def replay_receive() -> Message:
            nonlocal chunk_idx
            if chunk_idx < len(received_chunks):
                chunk = received_chunks[chunk_idx]
                chunk_idx += 1
                more = chunk_idx < len(received_chunks)
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": more,
                }
            if not received_chunks and chunk_idx == 0:
                chunk_idx = 1
                return {
                    "type": "http.request",
                    "body": b"",
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

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
            await self.app(scope, replay_receive, send_wrapper)
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
