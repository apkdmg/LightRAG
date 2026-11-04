"""Capture of per-call LLM token usage, and of system events.

The role wrapper in :mod:`lightrag.llm_roles` builds a :class:`UsageRecord` for
every provider call, and :func:`lightrag.utils.handle_cache` for every LLM cache
hit. Records go to whatever sinks are registered here (the API server registers
a Postgres writer). With no sink registered nothing is captured, so library use
of LightRAG behaves exactly as before.

What each call was for comes from :func:`usage_scope`. The scope lives in a
context variable, so it is read in the caller's task before the call joins the
role queue: queue workers run in their own long-lived context and can't see it.

Records carry counts, timing and identifiers only, never prompt or answer text.
This module has no API-layer dependencies. A failing sink never affects a call.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional

logger = logging.getLogger("lightrag")

EXTRACT_PURPOSE = "extract"
KEYWORD_PURPOSE = "keyword"
QUERY_PURPOSE = "query"
VISION_PURPOSE = "vlm"
PURPOSES = (EXTRACT_PURPOSE, VISION_PURPOSE, KEYWORD_PURPOSE, QUERY_PURPOSE)

SYSTEM_BILLING = "system"
BYO_BILLING = "byo"

# Bindings whose completion functions accept and fill ``token_tracker``.
TOKEN_TRACKING_BINDINGS = frozenset({"openai", "azure_openai", "gemini"})

_DEFAULT_OPERATIONS = {
    EXTRACT_PURPOSE: "extraction",
    KEYWORD_PURPOSE: "mix",
    QUERY_PURPOSE: "bypass",
    VISION_PURPOSE: "document_image",
}

_CACHE_TYPE_PURPOSES = {
    "extract": EXTRACT_PURPOSE,
    "summary": EXTRACT_PURPOSE,
    "keywords": KEYWORD_PURPOSE,
    "query": QUERY_PURPOSE,
}

# generate_track_id() prefix -> where the document came from.
_TRACK_ID_SOURCES = {
    "upload": "file_upload",
    "email": "email",
    "scan": "folder_scan",
    "insert": "text_api",
    "enqueue": "text_api",
}


@dataclass(frozen=True)
class UsageScope:
    """What the LLM calls made in the current context are for."""

    workspace: Optional[str] = None
    purpose: Optional[str] = None
    operation: Optional[str] = None
    # Used when nothing outer has set ``operation``.
    default_operation: Optional[str] = None
    source: Optional[str] = None
    client: Optional[str] = None
    doc_id: Optional[str] = None
    track_id: Optional[str] = None

    def operation_for(self, purpose: str) -> str:
        return (
            self.operation
            or self.default_operation
            or _DEFAULT_OPERATIONS.get(purpose, "other")
        )


_scope: ContextVar[UsageScope] = ContextVar(
    "lightrag_usage_scope", default=UsageScope()
)


def current_usage_scope() -> UsageScope:
    return _scope.get()


@contextmanager
def usage_scope(**fields: Any) -> Iterator[UsageScope]:
    """Label LLM calls made inside this block. ``None`` values change nothing."""
    updated = replace(
        _scope.get(), **{k: v for k, v in fields.items() if v is not None}
    )
    token = _scope.set(updated)
    try:
        yield updated
    finally:
        _scope.reset(token)


async def run_in_usage_scope(awaitable: Awaitable[Any], **fields: Any) -> Any:
    """Await ``awaitable`` inside a :func:`usage_scope`."""
    with usage_scope(**fields):
        return await awaitable


def document_usage_fields(doc_id: str, track_id: Optional[str]) -> dict[str, Any]:
    """Scope fields for work done on one document."""
    prefix = (track_id or "").split("_", 1)[0]
    return {
        "doc_id": doc_id,
        "track_id": track_id or None,
        "source": _TRACK_ID_SOURCES.get(prefix, "other"),
    }


@dataclass(frozen=True)
class UsageRecord:
    """One provider call or one LLM cache hit. Never holds any text."""

    workspace: str
    purpose: str
    operation: str
    source: str = "other"
    client: Optional[str] = None
    model: str = ""
    host: Optional[str] = None
    billing: str = SYSTEM_BILLING
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    # Cost the provider reported for this call, when it reports one.
    cost: Optional[float] = None
    latency_ms: int = 0
    # "ok" | "error" | "truncated" | "abandoned"
    status: str = "ok"
    cache_hit: bool = False
    # Token counts came from a local estimate, not from the provider.
    estimated: bool = False
    # False when no figure exists at all: images can't be estimated locally.
    measured: bool = True
    doc_id: Optional[str] = None
    track_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    # Failure class (``lightrag.api.ops.classify``) of an ``error`` call and the
    # API request it served; filled in by the usage recorder.
    error_kind: Optional[str] = None
    request_id: Optional[str] = None
    # The exception behind an ``error`` call, for the recorder to classify.
    # Never stored: the recorder drops it before buffering the record.
    error: Optional[BaseException] = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SystemEvent:
    """Something that happened in the system, for chart pins and explanations."""

    kind: str
    workspace: Optional[str] = None
    purpose: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


UsageSink = Callable[[UsageRecord], None]
EventSink = Callable[[SystemEvent], None]
_usage_sinks: list[UsageSink] = []
_event_sinks: list[EventSink] = []


def add_usage_sink(sink: UsageSink) -> None:
    if sink not in _usage_sinks:
        _usage_sinks.append(sink)


def remove_usage_sink(sink: UsageSink) -> None:
    if sink in _usage_sinks:
        _usage_sinks.remove(sink)


def add_event_sink(sink: EventSink) -> None:
    if sink not in _event_sinks:
        _event_sinks.append(sink)


def remove_event_sink(sink: EventSink) -> None:
    if sink in _event_sinks:
        _event_sinks.remove(sink)


def usage_enabled() -> bool:
    return bool(_usage_sinks)


def emit_usage(record: UsageRecord) -> None:
    for sink in list(_usage_sinks):
        try:
            sink(record)
        except Exception as e:
            logger.warning(f"Usage sink failed: {e}")


def emit_event(
    kind: str,
    workspace: Optional[str] = None,
    purpose: Optional[str] = None,
    **detail: Any,
) -> None:
    event = SystemEvent(kind=kind, workspace=workspace, purpose=purpose, detail=detail)
    for sink in list(_event_sinks):
        try:
            sink(event)
        except Exception as e:
            logger.warning(f"Event sink failed: {e}")


def emit_cache_hit(cache_type: str, workspace: Optional[str]) -> None:
    """Record an LLM cache hit: no provider call, and no tokens spent."""
    if not usage_enabled():
        return
    scope = current_usage_scope()
    purpose = _CACHE_TYPE_PURPOSES.get(cache_type) or scope.purpose or EXTRACT_PURPOSE
    emit_usage(
        UsageRecord(
            workspace=scope.workspace or workspace or "default",
            purpose=purpose,
            operation=scope.operation_for(purpose),
            source=scope.source or "other",
            client=scope.client,
            cache_hit=True,
            doc_id=scope.doc_id,
            track_id=scope.track_id,
        )
    )


class UsageCapture:
    """Passed to a binding as ``token_tracker`` for one call."""

    def __init__(self) -> None:
        self.counts: Optional[dict[str, Any]] = None

    def add_usage(self, token_counts: dict[str, Any]) -> None:
        self.counts = token_counts

    def __bool__(self) -> bool:
        # Bindings test ``if token_tracker`` before filling it in.
        return True


def provider_usage_fields(usage: Any) -> dict[str, Any]:
    """Optional detail from an OpenAI-compatible ``usage`` object."""
    fields: dict[str, Any] = {}
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", None)
    reasoning = getattr(completion_details, "reasoning_tokens", None)
    # Some OpenAI-compatible routers report the cost of the call here.
    cost = getattr(usage, "cost", None)
    if isinstance(cached, int):
        fields["cached_tokens"] = cached
    if isinstance(reasoning, int):
        fields["reasoning_tokens"] = reasoning
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        fields["cost"] = float(cost)
    return fields


async def recorded_stream(
    stream: AsyncIterator[Any], finish: Callable[..., None]
) -> AsyncIterator[Any]:
    """Yield ``stream`` and report how it ended once it is finished or abandoned.

    ``finish(parts, status)`` is called once; a stream that failed also passes
    the exception as ``error=``.
    """
    parts: list[str] = []
    status = "abandoned"
    error: Optional[BaseException] = None
    try:
        async for chunk in stream:
            if isinstance(chunk, str):
                parts.append(chunk)
            yield chunk
        status = "ok"
    except GeneratorExit:
        raise
    except Exception as e:
        status = "error"
        error = e
        raise
    finally:
        if error is not None:
            finish(parts, status, error=error)
        else:
            finish(parts, status)
