"""Extra checks for Operations console requests that do something.

``require_admin`` (router-wide) trusts the token's ``role`` claim, which token
renewal copies forward for up to a week. Every ops POST therefore also uses:

- :func:`require_live_admin`: the caller must be an admin *now*, by the
  current ``ADMIN_ACCOUNTS`` (or, for service accounts, the current admin
  client allowlist or shared-key role), and must not be an impersonated
  session;
- :func:`require_operator` (exports, reveals and other forceful actions):
  when ``OPS_OPERATOR_ACCOUNTS`` is set, the caller must be listed there
  (D8); when it is unset every admin is an operator;
- :func:`require_ops_client`: CSRF protection (FR-77). The request must carry
  ``X-LightRAG-Client: webui`` (axios sends it on every call; a cross-site form
  cannot), a body, if any, must be JSON, and a browser request must come from
  this server's own pages: ``Sec-Fetch-Site: same-origin`` (set by the browser,
  so it holds behind any reverse proxy), or else an ``Origin`` matching the
  request's ``Host`` / ``X-Forwarded-Host`` / ``Forwarded`` host, the SSO
  redirect URI's origin, or an explicit ``CORS_ORIGINS`` entry.

Refusals are fixed codes; nothing about the account is echoed back.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status

from lightrag.api.dependencies import UserInfo, require_admin

logger = logging.getLogger("lightrag.api.ops")

CLIENT_HEADER = "X-LightRAG-Client"
CLIENT_VALUE = "webui"


def _global_args() -> Any:
    from lightrag.api.config import global_args

    return global_args


def is_live_admin(user: UserInfo) -> bool:
    """Whether ``user`` is an admin by the configuration in force now."""
    metadata = user.metadata or {}
    auth_mode = metadata.get("auth_mode", "")
    if auth_mode == "api_key":
        return getattr(_global_args(), "api_key_role", "admin") == "admin"
    if auth_mode == "client_credentials":
        client_id = metadata.get("client_id")
        if not client_id:
            return False
        from lightrag.api.obo_allowlist import check_admin_client

        return bool(check_admin_client(client_id))
    from lightrag.api.auth import _is_admin_user

    candidates = [user.username]
    for key in ("email", "preferred_username"):
        value = metadata.get(key)
        if isinstance(value, str):
            candidates.append(value)
    return _is_admin_user(*candidates)


def is_impersonated(user: UserInfo) -> bool:
    metadata = user.metadata or {}
    return bool(metadata.get("impersonation") or metadata.get("impersonated_by"))


async def require_live_admin(user: UserInfo = Depends(require_admin)) -> UserInfo:
    """Refuse removed admins (stale tokens) and impersonated sessions."""
    if is_impersonated(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "impersonation_not_allowed")
    try:
        live = is_live_admin(user)
    except Exception as e:
        logger.warning("Live admin check failed (%s)", type(e).__name__)
        live = False
    if not live:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin_not_current")
    return user


def operator_accounts() -> set[str]:
    """``OPS_OPERATOR_ACCOUNTS`` (comma-separated, case-insensitive)."""
    raw = os.getenv("OPS_OPERATOR_ACCOUNTS", "")
    return {a.strip().lower() for a in raw.split(",") if a.strip()}


def is_operator(user: UserInfo) -> bool:
    """When no operator list is set, every admin is an operator (D8)."""
    accounts = operator_accounts()
    if not accounts:
        return True
    candidates = {user.username}
    metadata = user.metadata or {}
    for key in ("email", "preferred_username"):
        value = metadata.get(key)
        if isinstance(value, str):
            candidates.add(value)
    return any(c and c.strip().lower() in accounts for c in candidates)


async def require_operator(user: UserInfo = Depends(require_live_admin)) -> UserInfo:
    """Operator-only routes: a live admin who is also an operator (D8)."""
    if not is_operator(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not_operator")
    return user


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin_of(url: str) -> Optional[tuple[str, str, Optional[int]]]:
    """``(scheme, hostname, port)`` of an origin or URL; the port is explicit
    or the scheme's default."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    return scheme, parts.hostname.lower(), port or _DEFAULT_PORTS.get(scheme)


def _host_of(value: str) -> Optional[tuple[str, Optional[int]]]:
    """``(hostname, port or None)`` of a ``Host``-style header value."""
    value = value.strip().strip('"')
    if not value:
        return None
    try:
        parts = urlsplit(f"//{value}")
        return (parts.hostname or "").lower() or None, parts.port
    except ValueError:
        return None


def _request_hosts(request: Request) -> list[tuple[str, Optional[int]]]:
    """The hosts the request was addressed to, as the proxy chain reports them."""
    values: list[str] = []
    for header in ("host", "x-forwarded-host"):
        values.extend((request.headers.get(header) or "").split(","))
    for element in (request.headers.get("forwarded") or "").split(","):
        for pair in element.split(";"):
            key, _, value = pair.partition("=")
            if key.strip().lower() == "host":
                values.append(value)
    hosts = []
    for value in values:
        parsed = _host_of(value)
        if parsed is not None and parsed[0]:
            hosts.append(parsed)
    return hosts


def _configured_origins() -> list[tuple[str, str, Optional[int]]]:
    args = _global_args()
    origins = []
    configured = getattr(args, "cors_origins", "") or ""
    for origin in str(configured).split(","):
        parsed = _origin_of(origin.strip())
        if parsed is not None:
            origins.append(parsed)
    try:
        from lightrag.api.config import is_oauth2_usable

        sso = is_oauth2_usable(args)
    except Exception:
        sso = False
    redirect = getattr(args, "oauth2_redirect_uri", "") if sso else ""
    if isinstance(redirect, str) and redirect:
        parsed = _origin_of(redirect)
        if parsed is not None:
            origins.append(parsed)
    return origins


def origin_allowed(request: Request, origin: str) -> bool:
    """Whether a browser ``Origin`` is this server's own (or configured) origin.

    A host header without a port (nginx's ``$host``) matches on the hostname.
    """
    parsed = _origin_of(origin)
    if parsed is None:
        return False
    scheme, hostname, port = parsed
    for host, host_port in _request_hosts(request):
        if host == hostname and (host_port is None or host_port == port):
            return True
    return parsed in _configured_origins()


_refusal_logged = False


def _log_refusal(request: Request, origin: str) -> None:
    """Log the first refusal at WARNING: usually a proxy that rewrites Host."""
    global _refusal_logged
    if _refusal_logged:
        return
    _refusal_logged = True
    hosts = (
        ", ".join(f"{h}:{p}" if p else h for h, p in _request_hosts(request)) or "none"
    )
    logger.warning(
        "Ops POST refused: Origin %s does not match the request host (%s). "
        "Behind a reverse proxy, forward Host or X-Forwarded-Host, or list "
        "the public origin in CORS_ORIGINS.",
        origin[:200],
        hosts[:200],
    )


async def require_ops_client(request: Request) -> None:
    """CSRF checks for ops POSTs (see the module docstring)."""
    if request.headers.get(CLIENT_HEADER, "").strip().lower() != CLIENT_VALUE:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "client_header_required")
    content_type = request.headers.get("content-type", "")
    has_body = request.headers.get("content-length", "0") not in ("", "0") or (
        "transfer-encoding" in request.headers
    )
    if has_body and not content_type.lower().startswith("application/json"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "json_required")
    fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
    if fetch_site == "same-origin":
        return  # the browser itself vouches for the page's origin
    origin = request.headers.get("origin")
    if origin and not origin_allowed(request, origin):
        _log_refusal(request, origin)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "origin_not_allowed")
    if not origin and fetch_site == "cross-site":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "origin_not_allowed")
