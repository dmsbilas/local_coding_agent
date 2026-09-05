"""Bounded retries for LLM / tool calls (rate limits, timeouts)."""

from __future__ import annotations

import random
import time

from config.settings import RETRY_BASE_SECONDS, RETRY_MAX_SECONDS


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort classification for provider rate-limit errors."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate-limit",
            "too many requests",
            "quota exceeded",
        )
    )


def is_timeout_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, TimeoutError) or any(
        marker in text for marker in ("timeout", "timed out", "deadline exceeded")
    )


def backoff(attempt: int) -> float:
    """Exponential backoff with small jitter, capped by configuration."""
    delay = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**attempt))
    return delay + random.uniform(0, delay * 0.1)


def invoke_with_retry(callable_fn, *, attempts: int, operation: str):
    """Retry transient failures while keeping a strict retry bound."""
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return callable_fn()
        except Exception as exc:  # noqa: BLE001 - boundary around external systems
            last_error = exc
            transient = is_rate_limit_error(exc) or is_timeout_error(exc)

            if not transient or attempt >= attempts - 1:
                break

            time.sleep(backoff(attempt))

    raise RuntimeError(
        f"{operation} failed after {attempts} attempt(s): {last_error}"
    ) from last_error
