"""Source-level PII fixes (§3.8.1) that the redaction regexes cannot catch.

Entity names, query text, chunk content, LLM kwargs and file names with
spaces look like ordinary words, so only the call sites themselves can keep
them out of the log. These tests drive the real code paths with sentinel
values and assert that no sentinel reaches a log record or an exception
message, plus a static guard over the fixed modules so a later edit that
reverts to ``f"... {entity_name}"`` fails here.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.offline

REPO = Path(__file__).resolve().parents[1]


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def text(self) -> str:
        parts = []
        for record in self.records:
            parts.append(record.getMessage())
            if record.exc_info:
                parts.append(logging.Formatter().formatException(record.exc_info))
        return "\n".join(parts)


@pytest.fixture
def lightrag_log():
    """Every record the ``lightrag`` logger sees, at DEBUG and above."""
    target = logging.getLogger("lightrag")
    handler = _ListHandler()
    old_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(old_level)


def _assert_absent(text: str, *sentinels: str) -> None:
    for sentinel in sentinels:
        assert sentinel not in text, (sentinel, text[-400:])


# --------------------------------------------------------------------------- #
# operate.py


def test_invalid_chunk_data_logs_no_content(lightrag_log):
    from lightrag.operate import _get_cached_extraction_results

    chunks = MagicMock()
    chunks.get_by_ids = AsyncMock(return_value=["SENTINEL_CHUNK Ali bin Abu text"])
    cache = MagicMock()
    cache.get_by_ids = AsyncMock(return_value=[])
    asyncio.run(_get_cached_extraction_results(cache, {"chunk-1"}, chunks))
    assert "Chunk data is invalid" in lightrag_log.text()
    _assert_absent(lightrag_log.text(), "SENTINEL_CHUNK")


def test_missing_node_logs_no_entity_name(lightrag_log):
    from lightrag.base import QueryParam
    from lightrag.operate import _find_most_related_entities_from_relationships

    graph = MagicMock()
    graph.get_nodes_batch = AsyncMock(return_value={})
    edges = [{"src_id": "SENTINEL_ENTITY_A", "tgt_id": "SENTINEL_ENTITY_B"}]
    asyncio.run(
        _find_most_related_entities_from_relationships(edges, QueryParam(), graph)
    )
    assert "not found in batch retrieval" in lightrag_log.text()
    _assert_absent(lightrag_log.text(), "SENTINEL_ENTITY")


def test_relation_without_description_raises_without_entity_names(lightrag_log):
    """The pipeline failure path logs the traceback, so the exception text
    must not carry the names either."""
    from lightrag.operate import _merge_edges_then_upsert

    graph = MagicMock()
    graph.has_edge = AsyncMock(return_value=False)
    with pytest.raises(ValueError) as caught:
        asyncio.run(
            _merge_edges_then_upsert(
                "SENTINEL_SRC Ali bin Abu",
                "SENTINEL_TGT UNIMAS Bursary",
                [],
                graph,
                None,
                None,
                {"max_source_ids_per_relation": 10},
            )
        )
    message = str(caught.value)
    assert "has no description" in message or "already_node missing" in message
    _assert_absent(message + lightrag_log.text(), "SENTINEL_SRC", "SENTINEL_TGT")


# --------------------------------------------------------------------------- #
# LLM bindings


def test_openai_failure_logs_param_keys_not_values(lightrag_log, monkeypatch):
    from lightrag.llm import openai as openai_mod

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    client.close = AsyncMock()
    monkeypatch.setattr(
        openai_mod, "create_openai_async_client", lambda **_kwargs: client
    )
    with pytest.raises(RuntimeError):
        asyncio.run(
            openai_mod.openai_complete_if_cache(
                "gpt-test",
                "SENTINEL_PROMPT",
                api_key="SENTINEL_API_KEY",
                openai_client_configs={"default_headers": {"X": "SENTINEL_HDR"}},
                extra_body={"secret": "SENTINEL_BODY"},
            )
        )
    text = lightrag_log.text()
    assert "OpenAI API Call Failed" in text and "extra_body" in text
    _assert_absent(text, "SENTINEL_API_KEY", "SENTINEL_HDR", "SENTINEL_BODY")


# --------------------------------------------------------------------------- #
# graph_routes.py


def _endpoint(router, path: str, method: str = "GET"):
    for route in router.routes:
        if getattr(route, "path", None) == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


def test_graph_route_errors_log_no_query_or_label(lightrag_log, monkeypatch):
    import sys

    from fastapi import HTTPException

    # The API config parses argv on first import.
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api.routers.graph_routes import create_graph_routes

    rag = MagicMock()
    rag.chunk_entity_relation_graph.search_labels = AsyncMock(
        side_effect=RuntimeError("down")
    )
    rag.get_knowledge_graph = AsyncMock(side_effect=RuntimeError("down"))
    router = create_graph_routes(rag)

    search = _endpoint(router, "/graph/label/search")
    with pytest.raises(HTTPException):
        asyncio.run(search(q="SENTINEL_QUERY", limit=5, rag=rag))
    graphs = _endpoint(router, "/graphs")
    with pytest.raises(HTTPException):
        asyncio.run(graphs(label="SENTINEL_LABEL", max_depth=1, max_nodes=5, rag=rag))
    errors = "\n".join(
        r.getMessage() for r in lightrag_log.records if r.levelno >= logging.WARNING
    )
    assert "Error searching labels" in errors
    _assert_absent(errors, "SENTINEL_QUERY", "SENTINEL_LABEL")


# --------------------------------------------------------------------------- #
# Static guard over the fixed modules

_LEVELS = {"info", "warning", "error", "exception", "critical"}

# module -> expressions that must never be interpolated raw into a log call at
# INFO or above (wrap them in mask_path / text_ref / len instead).
_FORBIDDEN = {
    "lightrag/operate.py": {
        "entity_name",
        "src_id",
        "tgt_id",
        "src",
        "tgt",
        "query",
        "chunk",
        "chunk_data",
    },
    "lightrag/pipeline.py": {"file_path", "source_file_path"},
    "lightrag/api/routers/document_routes.py": {
        "file_path",
        "file_path.name",
        "file_path_str",
        "filename",
        "safe_filename",
        "file.filename",
        "duplicate.name",
    },
    "lightrag/api/routers/email_routes.py": {
        "filename",
        "subject",
        "email.subject",
        "attachment.filename",
        "inline_img.filename",
        "image.filename",
    },
    "lightrag/api/routers/graph_routes.py": {
        "q",
        "label",
        "name",
        "request.entity_name",
        "request.source_id",
        "request.target_id",
        "request.source_entity",
        "request.target_entity",
        "request.entity_to_change_into",
        "request.entities_to_change",
    },
    "lightrag/llm/openai.py": {"kwargs", "client_configs"},
    "lightrag/llm/anthropic.py": {"kwargs"},
}

# Exceptions whose text reaches the log through the pipeline's traceback.
_FORBIDDEN_IN_RAISE = {
    "lightrag/operate.py": {"entity_name", "src_id", "tgt_id"},
    "lightrag/pipeline.py": {"file_path", "source_file_path"},
}


def _interpolations(node: ast.AST) -> set[str]:
    return {
        ast.unparse(value.value)
        for value in ast.walk(node)
        if isinstance(value, ast.FormattedValue)
    }


@pytest.mark.parametrize("module", sorted(_FORBIDDEN))
def test_fixed_modules_do_not_log_raw_values(module):
    tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
    leaks = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LEVELS
            and getattr(node.func.value, "id", None) == "logger"
        ):
            bad = _interpolations(node) & _FORBIDDEN[module]
            if bad:
                leaks.append((node.lineno, sorted(bad)))
        elif isinstance(node, ast.Raise) and module in _FORBIDDEN_IN_RAISE:
            bad = _interpolations(node) & _FORBIDDEN_IN_RAISE[module]
            if bad:
                leaks.append((node.lineno, sorted(bad)))
    assert not leaks, f"{module}: raw values logged at {leaks}"
