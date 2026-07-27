"""Application-level LLM error contract used by API-facing code."""


class LLMServiceError(Exception):
    """Raised when a provider failure should become a stable 503 response."""
