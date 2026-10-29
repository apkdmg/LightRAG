"""Detection and reporting of LLM provider credential / budget rejections.

A rejected API key (HTTP 401/403) or an exhausted spend limit (HTTP 402/412)
fails every call until someone changes the key or raises the limit, so retrying
document after document only floods the provider and the logs. The pipeline
uses :func:`classify_provider_rejection` to recognise these errors, pauses the
workspace's ingestion, and reports a :class:`ProviderRejectionEvent` to the
listeners registered here (the API server registers an email notifier).

This module has no API-layer dependencies; listeners are plain async callables.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from lightrag.utils import logger

# HTTP status -> rejection kind. 429 is deliberately absent: it is a transient
# rate limit that the LLM bindings already retry with backoff.
_REJECTION_KINDS = {
    401: "invalid_key",
    403: "invalid_key",
    402: "budget_exhausted",
    412: "budget_exhausted",
}

# Roles as named by the LLM role registry (see lightrag/llm_roles.py).
EXTRACT_ROLE = "extract"
VISION_ROLE = "vlm"
QUERY_ROLE = "query"

_MAX_MESSAGE_LEN = 300
_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]+")


@dataclass(frozen=True)
class ProviderRejection:
    """A provider refusal that will not go away by retrying."""

    kind: str  # "invalid_key" | "budget_exhausted"
    status_code: int
    message: str


@dataclass(frozen=True)
class ProviderRejectionEvent:
    """What a listener receives when a workspace's provider rejects a call."""

    workspace: str
    role: str
    rejection: ProviderRejection
    host: Optional[str] = None
    model: Optional[str] = None
    paused: bool = False
    retry_after_minutes: Optional[int] = None
    # True/False when the caller knows whether the workspace's own (BYO) key
    # was used; None lets the listener work it out from the provider config.
    byo_key: Optional[bool] = None
    occurred_at: float = field(default_factory=time.time)


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _provider_message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    message = None
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            message = inner.get("message")
        elif isinstance(inner, str):
            message = inner
    if not message:
        message = getattr(exc, "message", None) or str(exc)
    message = _SECRET_PATTERN.sub(r"\1…", str(message))
    return message[:_MAX_MESSAGE_LEN]


def classify_provider_rejection(
    exc: Optional[BaseException],
) -> Optional[ProviderRejection]:
    """Return the rejection behind ``exc`` (following its cause chain), if any.

    The pipeline wraps provider errors in ``RuntimeError`` layers with
    ``raise ... from``, so the original HTTP error sits in ``__cause__`` /
    ``__context__``.
    """
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        code = _status_code(current)
        if code in _REJECTION_KINDS:
            return ProviderRejection(
                kind=_REJECTION_KINDS[code],
                status_code=code,
                message=_provider_message(current),
            )
        current = current.__cause__ or current.__context__
    return None


ProviderRejectionListener = Callable[[ProviderRejectionEvent], Awaitable[None]]
_listeners: list[ProviderRejectionListener] = []


def add_provider_rejection_listener(listener: ProviderRejectionListener) -> None:
    if listener not in _listeners:
        _listeners.append(listener)


def remove_provider_rejection_listener(listener: ProviderRejectionListener) -> None:
    if listener in _listeners:
        _listeners.remove(listener)


async def emit_provider_rejection(event: ProviderRejectionEvent) -> None:
    """Deliver ``event`` to every listener; a failing listener never propagates."""
    for listener in list(_listeners):
        try:
            await listener(event)
        except Exception as e:
            logger.error(f"Provider rejection listener failed: {e}")
