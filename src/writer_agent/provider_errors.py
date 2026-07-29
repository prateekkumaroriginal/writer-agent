"""Classification for provider failures that are safe to retry later."""

from __future__ import annotations


class RetryableProviderError(RuntimeError):
    """A temporary model or search-provider failure."""


def raise_if_retryable_provider_error(exc: Exception) -> None:
    """Convert rate limits and transient transport failures to a safe error."""
    status_code = getattr(exc, "status_code", None)
    message = str(exc).casefold()
    retryable_markers = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "timed out",
        "timeout",
        "service unavailable",
        "connection error",
        "connection reset",
        "temporarily unavailable",
    )
    if status_code in {408, 429, 500, 502, 503, 504} or any(
        marker in message for marker in retryable_markers
    ):
        raise RetryableProviderError(
            "The AI provider is temporarily unavailable or rate-limited."
        ) from exc
