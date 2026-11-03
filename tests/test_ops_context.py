"""Request IDs, the 500 boundary and log context fields."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import logging
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from lightrag.api.ops.context import (
    CONTEXT_FIELDS,
    FUNCTION_LOG_TAGS,
    ContextFilter,
    ContextFormatter,
    RequestIdMiddleware,
    ServerErrorBoundary,
    background_log_context,
    bind_log_workspace,
    current_client_request_id,
    current_log_workspace,
    current_request_id,
    reset_run_id,
    set_run_id,
    set_worker_id,
)
from lightrag.api.security_headers import (
    BASELINE_SECURITY_HEADERS,
    SecurityHeadersMiddleware,
)
from lightrag.usage import usage_scope

pytestmark = pytest.mark.offline

HEX_ID = re.compile(r"^[0-9a-f]{32}$")


def _app(trusted: str = "", observers=()) -> FastAPI:
    app = FastAPI()

    @app.get("/ids")
    async def ids():
        return {
            "request_id": current_request_id(),
            "client_request_id": current_client_request_id(),
        }

    @app.get("/boom")
    async def boom():
        raise RuntimeError("database said sk-test-SECRET1234567890")

    @app.get("/stream-boom")
    async def stream_boom():
        async def body():
            yield b"partial"
            raise RuntimeError("late failure")

        return StreamingResponse(body())

    # Same order as the server: boundary innermost, request ID outermost.
    app.add_middleware(ServerErrorBoundary)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        RequestIdMiddleware,
        trusted_proxies=trusted,
        observers=observers,
        error_headers=BASELINE_SECURITY_HEADERS,
    )
    return app


def test_generated_request_id_is_echoed_and_bound():
    response = TestClient(_app()).get("/ids")
    request_id = response.headers["x-request-id"]
    assert HEX_ID.match(request_id)
    assert response.json() == {"request_id": request_id, "client_request_id": None}
    # The ContextVar does not outlive the request.
    assert current_request_id() is None


def test_untrusted_inbound_id_is_kept_only_as_client_request_id():
    client = TestClient(_app(trusted="10.0.0.0/8"))  # peer is "testclient"
    response = client.get("/ids", headers={"X-Request-ID": "client-abcdef12"})
    assert response.headers["x-request-id"] != "client-abcdef12"
    assert HEX_ID.match(response.headers["x-request-id"])
    assert response.json()["client_request_id"] == "client-abcdef12"


def test_trusted_proxy_id_is_adopted():
    client = TestClient(
        _app(trusted="10.0.0.0/24, 192.168.1.7"), client=("10.0.0.5", 1)
    )
    response = client.get("/ids", headers={"X-Request-ID": "nginx-0123456789abcdef"})
    assert response.headers["x-request-id"] == "nginx-0123456789abcdef"
    assert response.json() == {
        "request_id": "nginx-0123456789abcdef",
        "client_request_id": None,
    }


@pytest.mark.parametrize(
    "inbound", ["short", "has spaces in it", "x" * 65, "a/b/c/d/e/f"]
)
def test_malformed_inbound_id_is_ignored_even_from_trusted_proxy(inbound):
    client = TestClient(_app(trusted="10.0.0.5"), client=("10.0.0.5", 1))
    response = client.get("/ids", headers={"X-Request-ID": inbound})
    assert HEX_ID.match(response.headers["x-request-id"])
    assert response.json()["client_request_id"] is None


def test_default_trusts_nobody(monkeypatch):
    monkeypatch.delenv("OPS_TRUSTED_PROXIES", raising=False)
    app = FastAPI()

    @app.get("/ids")
    async def ids():
        return {"request_id": current_request_id()}

    app.add_middleware(RequestIdMiddleware)
    client = TestClient(app, client=("127.0.0.1", 1))
    response = client.get("/ids", headers={"X-Request-ID": "proxy-0123456789"})
    assert response.headers["x-request-id"] != "proxy-0123456789"


def test_unhandled_500_carries_request_id_and_headers(caplog):
    observed = []
    client = TestClient(
        _app(observers=[lambda scope, status, took: observed.append((status, took))]),
        raise_server_exceptions=False,
    )
    response = client.get("/boom")
    assert response.status_code == 500
    request_id = response.headers["x-request-id"]
    assert response.json() == {"detail": "internal_error", "request_id": request_id}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert "SECRET" not in response.text
    assert observed and observed[0][0] == 500 and observed[0][1] >= 0


def test_error_raised_outside_the_boundary_still_gets_request_id():
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    class Exploding:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            raise RuntimeError("middleware failure")

    app.add_middleware(Exploding)
    app.add_middleware(
        RequestIdMiddleware, trusted_proxies="", error_headers=BASELINE_SECURITY_HEADERS
    )
    response = TestClient(app, raise_server_exceptions=False).get("/ok")
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert response.headers["x-frame-options"] == "SAMEORIGIN"


def test_failure_after_response_started_is_not_masked():
    client = TestClient(_app())
    with pytest.raises(RuntimeError, match="late failure"):
        client.get("/stream-boom")


def test_observer_failure_does_not_break_requests():
    def broken(scope, status, took):
        raise ValueError("observer bug")

    response = TestClient(_app(observers=[broken])).get("/ids")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# ContextFilter


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("lightrag", logging.WARNING, __file__, 1, msg, (), None)


def test_context_fields_from_usage_scope_request_and_run():
    from lightrag.api.ops import context as ctx

    def run():
        rid = ctx._request_id.set("req-1234567890")
        token = set_run_id("run-42")
        try:
            with usage_scope(
                workspace="ws_one",
                doc_id="doc-1",
                track_id="upload_20261103_101010_abcd1234",
                operation="extraction",
                source="email",
            ):
                record = _record("anything")
                ContextFilter().filter(record)
                return record
        finally:
            reset_run_id(token)
            ctx._request_id.reset(rid)

    set_worker_id("worker-a")
    try:
        record = contextvars.copy_context().run(run)
    finally:
        set_worker_id(None)
    assert record.request_id == "req-1234567890"
    assert record.run_id == "run-42"
    assert record.worker_id == "worker-a"
    assert record.workspace == "ws_one"
    assert record.doc_id == "doc-1"
    assert record.track_id == "upload_20261103_101010_abcd1234"
    assert record.operation == "extraction"
    assert record.source == "email"


def test_workspace_falls_back_to_bound_workspace_then_message_prefix():
    def bound():
        bind_log_workspace("ws_bound")
        record = _record("[ws_prefix] message")
        ContextFilter().filter(record)
        return record

    assert contextvars.copy_context().run(bound).workspace == "ws_bound"

    record = _record("[ws_prefix] message")
    ContextFilter().filter(record)
    assert record.workspace == "ws_prefix"
    assert record.request_id == "-"

    record = _record("[email_20261103_101010_abcd1234] queued")
    ContextFilter().filter(record)
    assert record.workspace == "-"
    assert record.track_id == "email_20261103_101010_abcd1234"


@pytest.mark.parametrize(
    "msg",
    [
        "[analyze_multimodal] table/x: missing table content",
        "[sidecar] wrote blocks",
        "[purge] removed 3 rows",
        "[kg_query] no keywords",
        "[VECTOR_SIMILARITY] Traceback: boom",
        "[MinerU] parse failed",
    ],
)
def test_function_tag_prefix_is_not_taken_as_workspace(msg):
    record = _record(msg)
    ContextFilter().filter(record)
    assert record.workspace == "-"


def _literal_log_tags() -> set[str]:
    """Literal "[tag]" prefixes of logger calls in the lightrag package."""
    levels = {"debug", "info", "warning", "error", "exception", "critical"}
    tag = re.compile(r"^\[([A-Za-z0-9_-]{1,128})\]")
    found: set[str] = set()
    root = Path(__file__).resolve().parents[1] / "lightrag"
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in levels
                and node.args
            ):
                continue
            first = node.args[0]
            if isinstance(first, ast.JoinedStr) and first.values:
                first = first.values[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                match = tag.match(first.value)
                if match:
                    found.add(match.group(1).lower())
    return found


def test_every_literal_log_tag_is_a_known_function_tag():
    """A new "[tag]" call site must be listed, or it is labelled a workspace."""
    missing = _literal_log_tags() - FUNCTION_LOG_TAGS
    assert not missing, f"add to FUNCTION_LOG_TAGS: {sorted(missing)}"


def test_background_log_context_detaches_request_id_and_restores_it():
    from lightrag.api.ops import context as ctx

    def run():
        rid = ctx._request_id.set("req-aaaaaaaaaa")
        cid = ctx._client_request_id.set("client-bbbbbbbb")
        bind_log_workspace("ws_request")
        try:
            with background_log_context("ws_pipeline"):
                inside = (
                    current_request_id(),
                    current_client_request_id(),
                    current_log_workspace(),
                )
                record = _record("[ws_pipeline] processing")
                ContextFilter().filter(record)
            after = (current_request_id(), current_log_workspace())
            return inside, record, after
        finally:
            ctx._client_request_id.reset(cid)
            ctx._request_id.reset(rid)

    inside, record, after = contextvars.copy_context().run(run)
    assert inside == (None, None, "ws_pipeline")
    assert record.request_id == "-" and record.workspace == "ws_pipeline"
    assert after == ("req-aaaaaaaaaa", "ws_request")


def test_background_task_started_in_a_request_does_not_carry_its_id():
    """A pipeline run created during request A logs without A's ID."""
    from lightrag.api.ops import context as ctx

    seen: dict[str, object] = {}

    async def pipeline_run():
        seen["before"] = current_request_id()
        with background_log_context(""):
            await asyncio.sleep(0)
            seen["inside"] = current_request_id()
            seen["workspace"] = current_log_workspace()

    async def request_a():
        ctx._request_id.set("req-request-a")
        await asyncio.create_task(pipeline_run())

    asyncio.run(request_a())
    assert seen == {"before": "req-request-a", "inside": None, "workspace": None}


def test_pipeline_entry_and_drainer_run_detached_from_the_request():
    """Both long-lived entry points log the workspace, not the request ID."""
    from types import SimpleNamespace

    from lightrag.api.email_resync_queue import DeferredResyncQueue
    from lightrag.api.ops import context as ctx
    from lightrag.pipeline import _PipelineMixin

    seen: list[tuple] = []

    async def record_context(*_args):
        seen.append((current_request_id(), current_log_workspace()))

    rag = SimpleNamespace(workspace="ws_pipe", _process_enqueued_documents=None)
    rag._process_enqueued_documents = record_context
    queue = SimpleNamespace(_drain_loop=record_context)

    async def request():
        ctx._request_id.set("req-request-x")
        await _PipelineMixin.apipeline_process_enqueue_documents(rag)
        await DeferredResyncQueue._drain(queue, "ws_mail")
        seen.append((current_request_id(), None))

    asyncio.run(request())
    assert seen == [
        (None, "ws_pipe"),
        (None, "ws_mail"),
        ("req-request-x", None),
    ]


def test_formatter_tolerates_records_without_filter():
    record = _record("plain")
    text = ContextFormatter("[%(request_id)s] [%(workspace)s] %(message)s").format(
        record
    )
    assert text == "[-] [-] plain"
    for field in CONTEXT_FIELDS:
        assert getattr(record, field) == "-"


# --------------------------------------------------------------------------- #
# The real server app


_ENV_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "OPS_TRUSTED_PROXIES",
)


@pytest.fixture
def make_server_app(monkeypatch):
    for var in _ENV_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    def make(multi_tenant: bool = True) -> FastAPI:
        monkeypatch.setenv("ENABLE_MULTI_TENANCY", "true" if multi_tenant else "false")
        original_argv = sys.argv.copy()
        try:
            # The API config parses sys.argv on first use; keep pytest's flags out.
            sys.argv = ["lightrag-server"]
            from lightrag.api.config import parse_args
            from lightrag.api.lightrag_server import create_app

            args = parse_args()
            with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
                mock_rag.return_value = MagicMock()
                return create_app(args)
        finally:
            sys.argv = original_argv

    return make


@pytest.fixture
def server_app(make_server_app):
    return make_server_app()


def test_server_500_has_request_id_security_and_cors_headers(server_app):
    @server_app.get("/__ops_test_boom")
    async def boom():
        raise RuntimeError("leaked sk-test-SECRET1234567890")

    client = TestClient(server_app, raise_server_exceptions=False)
    response = client.get(
        "/__ops_test_boom", headers={"Origin": "http://webui.example"}
    )
    assert response.status_code == 500
    request_id = response.headers["x-request-id"]
    assert HEX_ID.match(request_id)
    assert response.json() == {"detail": "internal_error", "request_id": request_id}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" in response.headers
    assert "x-request-id" in response.headers["access-control-expose-headers"].lower()
    assert "SECRET" not in response.text


def test_server_mounts_ops_status_in_single_tenant_mode(make_server_app):
    from lightrag.api.dependencies import UserInfo, require_admin

    server_app = make_server_app(multi_tenant=False)

    assert not hasattr(server_app.state, "default_rag")
    server_app.dependency_overrides[require_admin] = lambda: UserInfo(
        username="root", role="admin", workspace_id="w", metadata={}
    )
    response = TestClient(server_app).get("/admin/ops/status")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["process"]["mode"] in ("uvicorn", "gunicorn")


def test_invalid_workspace_header_on_health_logs_no_warning(server_app):
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    lightrag_logger = logging.getLogger("lightrag")
    previous_level = lightrag_logger.level
    lightrag_logger.addHandler(handler)
    lightrag_logger.setLevel(logging.DEBUG)
    raw = "evil@example.com!<script>"
    try:
        TestClient(server_app, raise_server_exceptions=False).get(
            "/health", headers={"LIGHTRAG-WORKSPACE": raw}
        )
    finally:
        lightrag_logger.removeHandler(handler)
        lightrag_logger.setLevel(previous_level)
    assert not [r for r in records if r.levelno == logging.WARNING]
    assert all(raw not in r.getMessage() for r in records)
    header_lines = [r for r in records if "LIGHTRAG-WORKSPACE" in r.getMessage()]
    assert header_lines and all(r.levelno == logging.DEBUG for r in header_lines)
    assert all("sha=" in r.getMessage() for r in header_lines)
