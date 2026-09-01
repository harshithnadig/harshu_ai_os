"""Application-level LLM error contract used by API-facing code."""


class LLMServiceError(Exception):
    """Raised when a provider failure should become a stable 503 response."""


class LLMTimeoutError(LLMServiceError):
    """Raised when an LLM provider request times out."""


class LLMRateLimitError(LLMServiceError):
    """Raised when an LLM provider request is throttled or rate-limited."""


class LLMAuthenticationError(LLMServiceError):
    """Raised when an LLM provider rejects authentication credentials."""
