"""Structured JSON logging with request ID correlation."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from harshu_ai_os.observability.context import get_request_id


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
        }

        # Structured event name if present
        event = getattr(record, "event", None)
        if event is not None:
            data["event"] = event

        # Resolve request_id (from record extra or active contextvar)
        req_id = getattr(record, "request_id", None) or get_request_id()
        if req_id is not None:
            data["request_id"] = req_id

        # Standard and AI-specific structured metadata
        telemetry_fields = (
            "method",
            "path",
            "status_code",
            "duration_ms",
            "error_type",
            "workflow",
            "model",
            "provider",
            "complexity",
            "retry_count",
            "fallback_used",
            "tool_calls_count",
            "tool_name",
            "tool_used",
            "abstained",
            "abstention_reason",
            "retrieval_ms",
            "reranking_ms",
            "judge_ms",
            "generation_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost",
        )
        for field in telemetry_fields:
            if hasattr(record, field):
                data[field] = getattr(record, field)

        # Include message if distinct from the event name
        message = record.getMessage()
        if message and message != event:
            data["message"] = message

        return json.dumps(data)


def get_structured_logger(name: str) -> logging.Logger:
    """Return a logger configured with StructuredJsonFormatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_event(
    level: str,
    event: str,
    *,
    request_id: str | None = None,
    method: str | None = None,
    path: str | None = None,
    status_code: int | None = None,
    duration_ms: float | None = None,
    error_type: str | None = None,
) -> None:
    """Emit a structured event log line using the observability logger."""
    obs_logger = get_structured_logger("harshu_ai_os.observability")
    extra: dict[str, Any] = {"event": event}
    if request_id is not None:
        extra["request_id"] = request_id
    if method is not None:
        extra["method"] = method
    if path is not None:
        extra["path"] = path
    if status_code is not None:
        extra["status_code"] = status_code
    if duration_ms is not None:
        extra["duration_ms"] = duration_ms
    if error_type is not None:
        extra["error_type"] = error_type

    log_level = getattr(logging, level.upper(), logging.INFO)
    obs_logger.log(log_level, event, extra=extra)
