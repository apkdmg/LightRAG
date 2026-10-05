"""
HTTP security headers for the LightRAG API server and WebUI.

- ``SecurityHeadersMiddleware`` adds baseline headers to every response.
- ``build_webui_csp`` returns the Content-Security-Policy for WebUI HTML pages.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Iterable

# Applied to every response unless the response already sets the header.
BASELINE_SECURITY_HEADERS: dict[str, str] = {
    # Never let a browser reinterpret a JSON or text response as HTML/script.
    "X-Content-Type-Options": "nosniff",
    # Only the WebUI itself may frame our pages (it embeds /docs).
    "X-Frame-Options": "SAMEORIGIN",
    # Do not leak full URLs (which can contain document names or queries)
    # to other sites in the Referer header.
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware that adds baseline security headers.

    Implemented at the ASGI level (rather than with BaseHTTPMiddleware) so
    streaming responses such as /query/stream pass through untouched.
    """

    def __init__(self, app, headers: dict[str, str] | None = None):
        self.app = app
        self._headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or BASELINE_SECURITY_HEADERS).items()
        ]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                existing = {name.lower() for name, _ in message.get("headers", [])}
                extra = [(n, v) for n, v in self._headers if n not in existing]
                if extra:
                    message = {
                        **message,
                        "headers": list(message.get("headers", [])) + extra,
                    }
            await send(message)

        await self.app(scope, receive, send_with_headers)


def script_hash_source(script_body: str) -> str:
    """Return the CSP ``'sha256-…'`` source for an inline script's text."""
    digest = hashlib.sha256(script_body.encode("utf-8")).digest()
    return f"'sha256-{base64.b64encode(digest).decode('ascii')}'"


def build_webui_csp(inline_script_bodies: Iterable[str] = ()) -> str:
    """Build the Content-Security-Policy for WebUI HTML responses.

    The policy is the backstop against injected content in rendered answers:
    scripts only from our own origin (plus the hashed runtime-config script),
    no plugins, no framing by other sites, and images/network requests only
    to our own origin, so injected markup cannot load remote resources to
    exfiltrate data. Inline styles stay allowed because KaTeX, Mermaid and
    the UI components set style attributes.
    """
    script_sources = ["'self'"] + [script_hash_source(b) for b in inline_script_bodies]
    directives = {
        "default-src": ["'self'"],
        "script-src": script_sources,
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "blob:"],
        "font-src": ["'self'", "data:"],
        "connect-src": ["'self'"],
        "worker-src": ["'self'", "blob:"],
        "frame-src": ["'self'"],
        "object-src": ["'none'"],
        "base-uri": ["'self'"],
        "form-action": ["'self'"],
        "frame-ancestors": ["'self'"],
    }
    return "; ".join(
        f"{name} {' '.join(values)}" for name, values in directives.items()
    )
