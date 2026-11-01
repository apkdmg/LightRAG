"""Tags each request with where its questions come from, for usage records."""

from __future__ import annotations

from typing import Any, Optional

from lightrag.usage import usage_scope

CLIENT_HEADER = b"x-lightrag-client"


def _source_for(path: str, client: Optional[str]) -> Optional[str]:
    if "/query/data" in path:
        return "retrieval"
    if "/query" in path:
        return "webui" if client == "webui" else "rest_api"
    if path.startswith("/api/") or "/api/" in path:
        return "ollama"
    if "/v1/" in path:
        return "openai_api"
    return None


class UsageContextMiddleware:
    """Pure ASGI, so the scope stays visible through streaming responses."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        client = headers.get(CLIENT_HEADER, b"").decode("latin-1")[:64] or None
        agent = headers.get(b"user-agent", b"").decode("latin-1").split("/")[0][:64]
        source = _source_for(scope.get("path", ""), client)
        if source is None:
            await self.app(scope, receive, send)
            return
        with usage_scope(source=source, client=client or agent or None):
            await self.app(scope, receive, send)
