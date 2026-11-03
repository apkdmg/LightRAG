"""Failure taxonomy for documents that failed in the ingestion pipeline.

:func:`classify_failure` turns an exception into a small, stable set of
``error_kind`` values plus the HTTP status and exception type behind it, so
failures can be grouped by what went wrong instead of by message text. It
recognises exceptions by class name, module and status code only: no
provider SDK is imported, and the module stays as cheap to import as the
rest of :mod:`lightrag.api.ops` (the pipeline calls it).

The pipeline wraps provider errors in ``RuntimeError`` layers with ``raise
... from``, so the cause chain is followed (outermost first); the first link
that says something specific decides.

Kinds:

``cancelled``        the user cancelled the pipeline
``duplicate``        identical content or file name already stored
``provider_auth``    key rejected (401/403)
``provider_budget``  spend limit or payment required (402/412)
``rate_limited``     429 / rate-limit errors
``timeout``          timeouts (LLM, embedding, Docling, storage)
``connection``       could not connect / connection dropped
``context_length``   input longer than the model accepts
``provider_4xx``     other client errors from a provider
``provider_5xx``     server errors from a provider (incl. overloaded)
``llm_output``       the model's output could not be parsed (JSON etc.)
``storage``          database / vector store / graph store errors
``content_missing``  the document's stored content is gone
``multimodal``       image/table/equation analysis failed
``internal``         a programming or data error inside LightRAG
``unknown``          nothing above matched
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

ERROR_KINDS = (
    "cancelled",
    "duplicate",
    "provider_auth",
    "provider_budget",
    "rate_limited",
    "timeout",
    "connection",
    "context_length",
    "provider_4xx",
    "provider_5xx",
    "llm_output",
    "storage",
    "content_missing",
    "multimodal",
    "internal",
    "unknown",
)
# Where in the pipeline a document failed.
STAGES = ("enqueue", "parse", "analyze", "extract", "merge")

CANCELLED_PREFIX = "User cancelled"
_MAX_CHAIN = 16
_EXC_TYPE_MAX = 128

_STATUS_KINDS = {
    401: "provider_auth",
    403: "provider_auth",
    402: "provider_budget",
    412: "provider_budget",
    408: "timeout",
    413: "context_length",
    429: "rate_limited",
    504: "timeout",
}
# Matched against every class name in the exception's MRO.
_NAME_KINDS = {
    "PipelineCancelledException": "cancelled",
    "AuthenticationError": "provider_auth",
    "PermissionDeniedError": "provider_auth",
    "RateLimitError": "rate_limited",
    "APITimeoutError": "timeout",
    "TimeoutError": "timeout",
    "TimeoutException": "timeout",
    "ReadTimeout": "timeout",
    "ConnectTimeout": "timeout",
    "WriteTimeout": "timeout",
    "PoolTimeout": "timeout",
    "ServerTimeoutError": "timeout",
    "QueryCanceledError": "timeout",
    "APIConnectionError": "connection",
    "ConnectError": "connection",
    "RemoteProtocolError": "connection",
    "ConnectionError": "connection",
    "ClientConnectorError": "connection",
    "ServerDisconnectedError": "connection",
    "ServiceUnavailable": "connection",
    "InternalServerError": "provider_5xx",
    "OverloadedError": "provider_5xx",
    "ServiceUnavailableError": "provider_5xx",
    "JSONDecodeError": "llm_output",
    "OutputParserException": "llm_output",
    "ValidationError": "llm_output",
    "MultimodalAnalysisError": "multimodal",
    "MilvusException": "storage",
    "PostgresError": "storage",
    "InterfaceError": "storage",
    "Neo4jError": "storage",
    "DriverError": "storage",
    "RedisError": "storage",
    "UnexpectedResponse": "storage",
    "ResponseHandlingException": "storage",
}
# Exception modules that mean the storage layer failed.
_STORAGE_MODULES = (
    "asyncpg",
    "psycopg",
    "pymilvus",
    "neo4j",
    "redis",
    "qdrant_client",
    "pymongo",
    "motor",
    "opensearchpy",
    "faiss",
)
_GENERIC_NAMES = frozenset(
    {"Exception", "RuntimeError", "BaseException", "ExceptionGroup", "OSError"}
)
_INTERNAL_NAMES = frozenset(
    {
        "TypeError",
        "AttributeError",
        "KeyError",
        "IndexError",
        "AssertionError",
        "NameError",
        "ZeroDivisionError",
        "UnboundLocalError",
        "NotImplementedError",
    }
)
_CONTEXT_LENGTH = re.compile(
    r"context[_ ]length|maximum context|context window|too many tokens|"
    r"prompt is too long|input is too long|max_tokens.*exceed",
    re.IGNORECASE,
)
_RATE_LIMIT = re.compile(r"rate.?limit|too many requests", re.IGNORECASE)
_TIMEOUT = re.compile(r"timed? ?out|timeout", re.IGNORECASE)
_CONTENT_MISSING = re.compile(
    r"content not found in full_docs|document content not found", re.IGNORECASE
)
_JSON_OUTPUT = re.compile(
    r"json|expecting value|unterminated string|invalid control character",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FailureClass:
    """What :func:`classify_failure` returns."""

    error_kind: str
    http_status: Optional[int]
    exc_type: str


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    code = getattr(exc, "status", None)
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code < 600:
        return code
    return None


def _type_name(exc: BaseException) -> str:
    cls = type(exc)
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__qualname__", cls.__name__)
    if module in ("builtins", "__main__", ""):
        full = name
    else:
        full = f"{module.split('.')[0]}.{name}"
    return full[:_EXC_TYPE_MAX]


def _mro_names(exc: BaseException) -> list[str]:
    return [klass.__name__ for klass in type(exc).__mro__]


def _message(exc: BaseException) -> str:
    """The exception text plus a provider error body's message, if any."""
    try:
        text = str(exc)[:2000]
    except Exception:
        text = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        detail = inner.get("message") if isinstance(inner, dict) else inner
        if isinstance(detail, str) and detail:
            text = f"{text} {detail[:2000]}"
    return text


def _kind_of_link(exc: BaseException) -> tuple[Optional[str], Optional[int]]:
    """``(kind, status)`` for one link of the chain; kind None when generic."""
    names = _mro_names(exc)
    status = _status_code(exc)
    message = _message(exc)
    if "PipelineCancelledException" in names:
        return "cancelled", status
    if status is not None and 400 <= status < 600:
        if status == 400 and _CONTEXT_LENGTH.search(message):
            return "context_length", status
        kind = _STATUS_KINDS.get(status)
        if kind is not None:
            return kind, status
        return ("provider_5xx" if status >= 500 else "provider_4xx"), status
    for name in names:
        kind = _NAME_KINDS.get(name)
        if kind is not None:
            if kind == "llm_output" and name == "ValidationError":
                # pydantic ValidationError comes from LLM output parsing only
                # when the module says so; elsewhere it is a data error.
                module = getattr(type(exc), "__module__", "") or ""
                if not module.startswith("pydantic"):
                    continue
            return kind, status
    module = (getattr(type(exc), "__module__", "") or "").split(".")[0]
    if module in _STORAGE_MODULES:
        return "storage", status
    if _CONTENT_MISSING.search(message):
        return "content_missing", status
    if names[0] in _INTERNAL_NAMES:
        return "internal", status
    if names[0] == "ValueError" and _JSON_OUTPUT.search(message):
        return "llm_output", status
    if names[0] in _GENERIC_NAMES or names[0] == "ValueError":
        if message.startswith(CANCELLED_PREFIX):
            return "cancelled", status
        if _RATE_LIMIT.search(message):
            return "rate_limited", status
        if _CONTEXT_LENGTH.search(message):
            return "context_length", status
        if _TIMEOUT.search(message):
            return "timeout", status
    return None, status


def classify_failure(exc: Optional[BaseException]) -> FailureClass:
    """Classify ``exc`` (following its cause chain). Never raises."""
    if exc is None:
        return FailureClass("unknown", None, "")
    try:
        seen: set[int] = set()
        current: Optional[BaseException] = exc
        first_status: Optional[int] = None
        while current is not None and id(current) not in seen:
            if len(seen) >= _MAX_CHAIN:
                break
            seen.add(id(current))
            kind, status = _kind_of_link(current)
            if first_status is None and status is not None:
                first_status = status
            if kind is not None:
                return FailureClass(kind, status or first_status, _type_name(current))
            current = current.__cause__ or current.__context__
        return FailureClass("unknown", first_status, _type_name(exc))
    except Exception:
        return FailureClass("unknown", None, "")


def is_cancellation(exc: Optional[BaseException]) -> bool:
    return classify_failure(exc).error_kind == "cancelled"


def classify_message(message: Any) -> str:
    """Best-effort kind for a stored ``error_msg`` with no exception at hand."""
    text = message if isinstance(message, str) else ""
    if text.startswith(CANCELLED_PREFIX):
        return "cancelled"
    if _CONTENT_MISSING.search(text):
        return "content_missing"
    if _RATE_LIMIT.search(text) or "429" in text:
        return "rate_limited"
    if _CONTEXT_LENGTH.search(text):
        return "context_length"
    if _TIMEOUT.search(text):
        return "timeout"
    return "unknown"
