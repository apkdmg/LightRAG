"""Tests for the HTTP security headers added to the API server and WebUI.

Runs offline: a minimal FastAPI app exercises the middleware, and the CSP
builder is checked directly.
"""

import base64
import hashlib

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.testclient import TestClient

from lightrag.api.security_headers import (
    BASELINE_SECURITY_HEADERS,
    SecurityHeadersMiddleware,
    build_webui_csp,
    script_hash_source,
)

pytestmark = pytest.mark.offline


def _client():
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/plain")
    async def plain():
        return {"ok": True}

    @app.get("/framed")
    async def framed():
        return PlainTextResponse("x", headers={"X-Frame-Options": "DENY"})

    @app.get("/stream")
    async def stream():
        async def chunks():
            for part in ("a", "b", "c"):
                yield part

        return StreamingResponse(chunks(), media_type="text/plain")

    return TestClient(app)


def test_baseline_headers_on_every_response():
    client = _client()
    for path in ("/plain", "/stream", "/missing"):
        response = client.get(path)
        for name, value in BASELINE_SECURITY_HEADERS.items():
            assert response.headers.get(name) == value, (path, name)


def test_streaming_body_passes_through():
    assert _client().get("/stream").text == "abc"


def test_existing_header_is_not_overridden():
    response = _client().get("/framed")
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert len(response.headers.get_list("X-Frame-Options")) == 1


def test_script_hash_source_matches_sha256_of_body():
    body = 'window.__LIGHTRAG_CONFIG__ = {"apiPrefix": ""};'
    expected = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
    assert script_hash_source(body) == f"'sha256-{expected}'"


def test_webui_csp_blocks_remote_loads_and_unhashed_inline_scripts():
    body = "window.__LIGHTRAG_CONFIG__ = {};"
    directives = {
        part.split(" ", 1)[0]: part.split(" ", 1)[1].split()
        for part in build_webui_csp([body]).split("; ")
    }

    assert directives["script-src"] == ["'self'", script_hash_source(body)]
    assert "'unsafe-inline'" not in directives["script-src"]
    assert "'unsafe-eval'" not in directives["script-src"]
    assert directives["img-src"] == ["'self'", "data:", "blob:"]
    assert directives["connect-src"] == ["'self'"]
    assert directives["object-src"] == ["'none'"]
    assert directives["frame-ancestors"] == ["'self'"]
