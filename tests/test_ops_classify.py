"""Failure taxonomy: real exception shapes from the providers, storages and
the pipeline map to stable ``error_kind`` values."""

from __future__ import annotations

import asyncio
import json

import anthropic
import asyncpg
import httpx
import openai
import pytest
from pymilvus.exceptions import MilvusException

from lightrag.api.ops import runs
from lightrag.api.ops.classify import (
    ERROR_KINDS,
    classify_failure,
    classify_message,
    is_cancellation,
)
from lightrag.exceptions import PipelineCancelledException

pytestmark = pytest.mark.offline

_REQUEST = httpx.Request("POST", "https://router.example.test/v1/chat/completions")


def _openai_status(cls, code: int, message: str = "boom"):
    return cls(
        f"Error code: {code}",
        response=httpx.Response(code, request=_REQUEST),
        body={"error": {"message": message}},
    )


def _anthropic_status(cls, code: int, message: str = "boom"):
    return cls(
        message,
        response=httpx.Response(code, request=_REQUEST),
        body={"type": "error", "error": {"type": "x", "message": message}},
    )


def _wrapped(inner: BaseException) -> BaseException:
    """The pipeline's ``raise RuntimeError(...) from e`` layering."""
    try:
        try:
            raise inner
        except BaseException as e:
            raise RuntimeError("chunk-000: extraction failed") from e
    except RuntimeError as outer:
        return outer


@pytest.mark.parametrize(
    "exc, kind, status",
    [
        (_openai_status(openai.RateLimitError, 429), "rate_limited", 429),
        (_openai_status(openai.AuthenticationError, 401), "provider_auth", 401),
        (_openai_status(openai.PermissionDeniedError, 403), "provider_auth", 403),
        (_openai_status(openai.APIStatusError, 402), "provider_budget", 402),
        (_openai_status(openai.APIStatusError, 412), "provider_budget", 412),
        (_openai_status(openai.InternalServerError, 500), "provider_5xx", 500),
        (_openai_status(openai.APIStatusError, 503), "provider_5xx", 503),
        (_openai_status(openai.APIStatusError, 504), "timeout", 504),
        (_openai_status(openai.NotFoundError, 404), "provider_4xx", 404),
        (
            _openai_status(
                openai.BadRequestError,
                400,
                "This model's maximum context length is 8192 tokens",
            ),
            "context_length",
            400,
        ),
        (
            _openai_status(openai.BadRequestError, 400, "invalid image"),
            "provider_4xx",
            400,
        ),
        (openai.APITimeoutError(request=_REQUEST), "timeout", None),
        (openai.APIConnectionError(request=_REQUEST), "connection", None),
        (_anthropic_status(anthropic.RateLimitError, 429), "rate_limited", 429),
        (_anthropic_status(anthropic.AuthenticationError, 401), "provider_auth", 401),
        (_anthropic_status(anthropic.InternalServerError, 500), "provider_5xx", 500),
        (_anthropic_status(anthropic.APIStatusError, 529), "provider_5xx", 529),
        (anthropic.APITimeoutError(request=_REQUEST), "timeout", None),
        (httpx.ReadTimeout("docling:5001 read timed out"), "timeout", None),
        (httpx.ConnectError("connection refused"), "connection", None),
        (asyncio.TimeoutError(), "timeout", None),
        (TimeoutError("timed out"), "timeout", None),
        (ConnectionResetError("reset by peer"), "connection", None),
        (asyncpg.exceptions.UniqueViolationError("duplicate key"), "storage", None),
        (asyncpg.exceptions.QueryCanceledError("statement timeout"), "timeout", None),
        (asyncpg.exceptions.InterfaceError("connection is closed"), "storage", None),
        (MilvusException(code=1, message="collection not loaded"), "storage", None),
        (json.JSONDecodeError("Expecting value", "{", 0), "llm_output", None),
        (ValueError("Unterminated string in JSON output"), "llm_output", None),
        (
            Exception("Document content not found in full_docs for doc_id: doc-1"),
            "content_missing",
            None,
        ),
        (TypeError("unsupported operand"), "internal", None),
        (KeyError("chunk_order_index"), "internal", None),
        (RuntimeError("something odd"), "unknown", None),
    ],
)
def test_exception_corpus(exc, kind, status):
    result = classify_failure(exc)
    assert result.error_kind == kind
    assert result.error_kind in ERROR_KINDS
    assert result.http_status == status


def test_neo4j_errors_are_storage():
    from neo4j.exceptions import ClientError, ServiceUnavailable

    assert classify_failure(ClientError("bad cypher")).error_kind == "storage"
    assert classify_failure(ServiceUnavailable("no route")).error_kind in (
        "connection",
        "storage",
    )


def test_wrapped_chains_find_the_cause():
    inner = _openai_status(openai.RateLimitError, 429)
    result = classify_failure(_wrapped(inner))
    assert result.error_kind == "rate_limited"
    assert result.http_status == 429
    assert result.exc_type == "openai.RateLimitError"
    budget = _wrapped(_openai_status(openai.APIStatusError, 402))
    assert classify_failure(budget).error_kind == "provider_budget"
    timeout = _wrapped(httpx.ReadTimeout("slow"))
    assert classify_failure(timeout).error_kind == "timeout"


@pytest.mark.parametrize(
    "message",
    [
        "User cancelled",
        "User cancelled during entity extraction",
        "User cancelled during merge phase",
        "User cancelled 3/7: report.pdf",
        "User cancelled during merge 1/2: memo.docx",
    ],
)
def test_every_cancellation_variant(message):
    assert classify_failure(PipelineCancelledException(message)).error_kind == (
        "cancelled"
    )
    assert is_cancellation(PipelineCancelledException(message))
    # Legacy rows only keep the message.
    assert classify_message(message) == "cancelled"
    # A cancellation re-raised as a generic error still reads as one.
    assert classify_failure(Exception(message)).error_kind == "cancelled"


def test_never_raises_and_handles_cycles():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert classify_failure(a).error_kind == "unknown"
    assert classify_failure(None).error_kind == "unknown"

    class Weird(Exception):
        def __str__(self):
            raise ValueError("no text")

    assert classify_failure(Weird()).error_kind == "unknown"


def test_legacy_message_classification():
    assert classify_message("RateLimitError: Error code: 429 - slow") == "rate_limited"
    assert classify_message("Request timed out after 120 seconds") == "timeout"
    assert classify_message(None) == "unknown"


def test_duplicate_failure_needs_tracking(monkeypatch):
    runs.install(None)
    assert runs.duplicate_failure("parse") is None
    tracker = runs.RunTracker(store=None, worker_id="w" * 32)
    runs.install(tracker)
    try:
        failure = runs.duplicate_failure("enqueue")
        assert failure["error_kind"] == "duplicate"
        assert failure["stage"] == "enqueue"
        assert failure["attempt"] == 1
        # Duplicates never produce a failure history row.
        assert not tracker._failures
    finally:
        runs.install(None)
