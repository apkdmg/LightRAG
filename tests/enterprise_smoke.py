"""Enterprise smoke checks against a running LightRAG server.

Black-box HTTP probes for the fork's enterprise surface — auth, per-user API
keys, multitenancy, OpenAI-compatible endpoint, document/query routing, admin
routes. Designed to be run before and after the v1.5.0.x upstream catch-up so
regressions are detected by diffing two runs.

Usage:
    python tests/enterprise_smoke.py --base-url http://localhost:9621 \\
        --username admin --password admin

    # Subset of checks
    python tests/enterprise_smoke.py --only auth --only apikeys

    # Skip a check
    python tests/enterprise_smoke.py --skip openai_compat

    # Include ingestion checks (slow — issues real LLM calls)
    python tests/enterprise_smoke.py --with-ingestion

Exit code: 0 if all non-skipped checks pass, 1 otherwise.

Design notes:
- No pytest. This is a standalone script so it can run against a live deployed
  server with whatever Python is available, without a test-runner config.
- Each check is a top-level function returning (Status, message). The runner
  tallies and prints; nothing relies on global mutable state except the bearer
  token cache between auth_login and downstream checks.
- Tier 1 (fast, default): read-only + cheap mutations. Runs in seconds.
- Tier 2 (--with-ingestion): real ingestion + query. Minutes per check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    duration_ms: int = 0


@dataclass
class Context:
    base_url: str
    username: Optional[str]
    password: Optional[str]
    admin_username: Optional[str]
    admin_password: Optional[str]
    static_api_key: Optional[str]
    timeout: int
    verbose: bool

    user_token: Optional[str] = None
    admin_token: Optional[str] = None
    auth_mode: Optional[str] = None
    oauth2_enabled: bool = False
    created_apikey_id: Optional[str] = None
    created_apikey_value: Optional[str] = None
    created_workspace_id: Optional[str] = None
    results: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _headers(token: Optional[str], api_key: Optional[str] = None) -> dict:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    elif api_key:
        h["X-API-Key"] = api_key
    return h


def _url(ctx: Context, path: str) -> str:
    return ctx.base_url.rstrip("/") + path


def _request(ctx: Context, method: str, path: str, **kwargs):
    kwargs.setdefault("timeout", ctx.timeout)
    return requests.request(method, _url(ctx, path), **kwargs)


# ---------------------------------------------------------------------------
# Tier 1 checks
# ---------------------------------------------------------------------------


def check_health(ctx: Context) -> tuple[str, str]:
    r = _request(ctx, "GET", "/health")
    if r.status_code != 200:
        return FAIL, f"/health returned {r.status_code}: {r.text[:200]}"
    return PASS, "200 OK"


def check_auth_status(ctx: Context) -> tuple[str, str]:
    r = _request(ctx, "GET", "/auth-status")
    if r.status_code != 200:
        return FAIL, f"/auth-status returned {r.status_code}"
    try:
        body = r.json()
    except ValueError:
        return FAIL, "/auth-status response was not JSON"
    if "auth_configured" not in body and "auth_mode" not in body:
        return FAIL, f"/auth-status missing expected keys: {list(body)}"
    ctx.oauth2_enabled = bool(body.get("oauth2_enabled"))
    ctx.auth_mode = body.get("auth_mode") or (
        "oauth2"
        if ctx.oauth2_enabled
        else ("jwt" if body.get("auth_configured") else "disabled")
    )
    return PASS, f"auth_mode={ctx.auth_mode} oauth2={ctx.oauth2_enabled}"


def check_oauth2_config(ctx: Context) -> tuple[str, str]:
    if not ctx.oauth2_enabled:
        return SKIP, "OAuth2 not enabled on this server"
    r = _request(ctx, "GET", "/oauth2/config")
    if r.status_code != 200:
        return FAIL, f"/oauth2/config returned {r.status_code}"
    body = r.json()
    if not body.get("authorization_url") and not body.get("client_id"):
        return FAIL, f"/oauth2/config missing client_id/authorization_url: {body}"
    return PASS, f"client_id={body.get('client_id', '?')[:12]}…"


def check_login_valid(ctx: Context) -> tuple[str, str]:
    if ctx.auth_mode == "disabled":
        return SKIP, "auth disabled — no /login flow to test"
    if not ctx.username or not ctx.password:
        return SKIP, "no --username/--password provided"
    r = _request(
        ctx,
        "POST",
        "/login",
        data={"username": ctx.username, "password": ctx.password},
    )
    if r.status_code != 200:
        return FAIL, f"/login returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    token = body.get("access_token")
    if not token:
        return FAIL, f"/login response missing access_token: {list(body)}"
    ctx.user_token = token
    role = body.get("role", "?")
    return PASS, f"got JWT (role={role})"


def check_login_invalid(ctx: Context) -> tuple[str, str]:
    if ctx.auth_mode == "disabled":
        return SKIP, "auth disabled"
    r = _request(
        ctx,
        "POST",
        "/login",
        data={"username": "definitely-not-a-real-user-xyzzy", "password": "wrong"},
    )
    if r.status_code not in (401, 403):
        return FAIL, f"bad-creds login returned {r.status_code} (expected 401/403)"
    return PASS, f"rejected with {r.status_code}"


def check_admin_login(ctx: Context) -> tuple[str, str]:
    if ctx.auth_mode == "disabled":
        return SKIP, "auth disabled"
    if not ctx.admin_username or not ctx.admin_password:
        return SKIP, "no --admin-username/--admin-password provided"
    r = _request(
        ctx,
        "POST",
        "/login",
        data={"username": ctx.admin_username, "password": ctx.admin_password},
    )
    if r.status_code != 200:
        return FAIL, f"admin /login returned {r.status_code}"
    body = r.json()
    if body.get("role") != "admin":
        return FAIL, f"admin login succeeded but role={body.get('role')} (expected 'admin')"
    ctx.admin_token = body["access_token"]
    return PASS, "admin JWT acquired"


def check_documents_paginated(ctx: Context) -> tuple[str, str]:
    token = ctx.user_token
    r = _request(
        ctx,
        "POST",
        "/documents/paginated",
        headers=_headers(token, ctx.static_api_key),
        json={"page": 1, "page_size": 5},
    )
    if r.status_code == 405:
        r = _request(
            ctx,
            "GET",
            "/documents/paginated",
            headers=_headers(token, ctx.static_api_key),
            params={"page": 1, "page_size": 5},
        )
    if r.status_code != 200:
        return FAIL, f"/documents/paginated returned {r.status_code}: {r.text[:200]}"
    return PASS, "200 OK"


def check_status_counts(ctx: Context) -> tuple[str, str]:
    r = _request(
        ctx,
        "GET",
        "/documents/status_counts",
        headers=_headers(ctx.user_token, ctx.static_api_key),
    )
    if r.status_code != 200:
        return FAIL, f"/documents/status_counts returned {r.status_code}"
    body = r.json()
    expected_keys = {"processed", "processing", "pending", "failed"}
    if not (expected_keys & set(body.keys())):
        return FAIL, f"unexpected response shape: {list(body)}"
    return PASS, f"counts={body}"


def check_pipeline_status(ctx: Context) -> tuple[str, str]:
    r = _request(
        ctx,
        "GET",
        "/documents/pipeline_status",
        headers=_headers(ctx.user_token, ctx.static_api_key),
    )
    if r.status_code != 200:
        return FAIL, f"/documents/pipeline_status returned {r.status_code}"
    body = r.json()
    for k in ("busy", "history_messages"):
        if k not in body:
            return FAIL, f"missing key {k!r} in pipeline_status response"
    return PASS, f"busy={body.get('busy')}"


def check_graphs_list(ctx: Context) -> tuple[str, str]:
    r = _request(
        ctx,
        "GET",
        "/graphs",
        headers=_headers(ctx.user_token, ctx.static_api_key),
        params={"label": "*"},
    )
    if r.status_code != 200:
        return FAIL, f"/graphs returned {r.status_code}: {r.text[:200]}"
    return PASS, "200 OK"


def check_graph_label_list(ctx: Context) -> tuple[str, str]:
    r = _request(
        ctx,
        "GET",
        "/graph/label/list",
        headers=_headers(ctx.user_token, ctx.static_api_key),
    )
    if r.status_code != 200:
        return FAIL, f"/graph/label/list returned {r.status_code}"
    return PASS, "200 OK"


def check_apikey_create(ctx: Context) -> tuple[str, str]:
    if not ctx.user_token:
        return SKIP, "no user JWT (auth disabled or login skipped)"
    r = _request(
        ctx,
        "POST",
        "/api-keys",
        headers=_headers(ctx.user_token),
        json={"name": "enterprise-smoke-test", "expires_in_days": 1},
    )
    if r.status_code not in (200, 201):
        return FAIL, f"POST /api-keys returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    key_id = body.get("id") or body.get("key_id")
    key_value = body.get("key") or body.get("api_key") or body.get("token")
    if not key_id or not key_value:
        return FAIL, f"create response missing id/key: {list(body)}"
    ctx.created_apikey_id = str(key_id)
    ctx.created_apikey_value = key_value
    return PASS, f"created key id={key_id}"


def check_apikey_usable(ctx: Context) -> tuple[str, str]:
    if not ctx.created_apikey_value:
        return SKIP, "no API key created (prior check skipped/failed)"
    r = _request(
        ctx,
        "GET",
        "/documents/status_counts",
        headers={"X-API-Key": ctx.created_apikey_value},
    )
    if r.status_code != 200:
        return FAIL, f"using created API key returned {r.status_code}"
    return PASS, "API key authenticates"


def check_apikey_delete(ctx: Context) -> tuple[str, str]:
    if not ctx.created_apikey_id:
        return SKIP, "no API key to delete"
    r = _request(
        ctx,
        "DELETE",
        f"/api-keys/{ctx.created_apikey_id}",
        headers=_headers(ctx.user_token),
    )
    if r.status_code not in (200, 204):
        return FAIL, f"DELETE /api-keys/{ctx.created_apikey_id} returned {r.status_code}"
    return PASS, f"deleted key id={ctx.created_apikey_id}"


def check_apikey_list(ctx: Context) -> tuple[str, str]:
    if not ctx.user_token:
        return SKIP, "no user JWT"
    r = _request(ctx, "GET", "/api-keys", headers=_headers(ctx.user_token))
    if r.status_code != 200:
        return FAIL, f"GET /api-keys returned {r.status_code}"
    body = r.json()
    count = len(body) if isinstance(body, list) else len(body.get("keys", []))
    return PASS, f"{count} keys for user"


def check_admin_status(ctx: Context) -> tuple[str, str]:
    if not ctx.admin_token:
        return SKIP, "no admin JWT"
    r = _request(ctx, "GET", "/admin/status", headers=_headers(ctx.admin_token))
    if r.status_code != 200:
        return FAIL, f"GET /admin/status returned {r.status_code}"
    return PASS, "200 OK"


def check_admin_workspaces_list(ctx: Context) -> tuple[str, str]:
    if not ctx.admin_token:
        return SKIP, "no admin JWT"
    r = _request(ctx, "GET", "/admin/workspaces", headers=_headers(ctx.admin_token))
    if r.status_code != 200:
        return FAIL, f"GET /admin/workspaces returned {r.status_code}"
    body = r.json()
    items = body if isinstance(body, list) else body.get("workspaces", [])
    return PASS, f"{len(items)} workspaces"


def check_admin_routes_require_admin(ctx: Context) -> tuple[str, str]:
    if not ctx.user_token:
        return SKIP, "no non-admin JWT to verify gate"
    if ctx.admin_username and ctx.username == ctx.admin_username:
        return SKIP, "user is admin — cannot test the negative case"
    r = _request(ctx, "GET", "/admin/status", headers=_headers(ctx.user_token))
    if r.status_code not in (401, 403):
        return FAIL, f"non-admin reached /admin/status with {r.status_code} (expected 401/403)"
    return PASS, f"non-admin blocked with {r.status_code}"


def check_openai_models(ctx: Context) -> tuple[str, str]:
    token = ctx.user_token
    r = _request(ctx, "GET", "/v1/models", headers=_headers(token, ctx.static_api_key))
    if r.status_code == 404:
        return FAIL, "/v1/models 404 — OpenAI-compat endpoint not mounted"
    if r.status_code != 200:
        return FAIL, f"/v1/models returned {r.status_code}"
    body = r.json()
    if not body.get("data"):
        return FAIL, f"/v1/models response missing data: {body}"
    return PASS, f"{len(body['data'])} model(s) listed"


def check_openai_chat_shape(ctx: Context) -> tuple[str, str]:
    """Issues a real chat call. Skipped unless --with-ingestion is set, since
    this triggers an LLM call against the configured backend."""
    if not ctx.run_ingestion_checks:  # type: ignore[attr-defined]
        return SKIP, "needs --with-ingestion (issues real LLM call)"
    token = ctx.user_token
    r = _request(
        ctx,
        "POST",
        "/v1/chat/completions",
        headers=_headers(token, ctx.static_api_key),
        json={
            "model": "lightrag:latest",
            "messages": [{"role": "user", "content": "Reply with the word OK."}],
            "max_tokens": 16,
            "stream": False,
        },
        timeout=ctx.timeout * 3,
    )
    if r.status_code != 200:
        return FAIL, f"/v1/chat/completions returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    choices = body.get("choices") or []
    if not choices:
        return FAIL, f"no choices in response: {body}"
    return PASS, "got chat completion"


# ---------------------------------------------------------------------------
# Tier 2 — ingestion + query (slow)
# ---------------------------------------------------------------------------


def _minimal_pdf_bytes() -> bytes:
    """Smallest possible valid PDF carrying the literal text 'SmokeTestProbe'."""
    return (
        b"%PDF-1.1\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 55>>stream\n"
        b"BT /F1 12 Tf 20 100 Td (SmokeTestProbe lightrag) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000052 00000 n \n"
        b"0000000095 00000 n \n"
        b"0000000183 00000 n \n"
        b"0000000280 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n340\n%%EOF"
    )


def check_text_ingestion(ctx: Context) -> tuple[str, str]:
    if not ctx.run_ingestion_checks:  # type: ignore[attr-defined]
        return SKIP, "needs --with-ingestion"
    payload = {
        "text": "The enterprise smoke test marker is FuchsiaPenguin42.",
        "file_source": "enterprise-smoke-test.txt",
    }
    r = _request(
        ctx,
        "POST",
        "/documents/text",
        headers=_headers(ctx.user_token, ctx.static_api_key),
        json=payload,
        timeout=ctx.timeout * 3,
    )
    if r.status_code not in (200, 202):
        return FAIL, f"/documents/text returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    track_id = body.get("track_id")
    if not track_id:
        return FAIL, f"text insert response missing track_id: {body}"
    return PASS, f"queued track_id={track_id}"


def check_pdf_upload_routes_to_docling(ctx: Context) -> tuple[str, str]:
    if not ctx.run_ingestion_checks:  # type: ignore[attr-defined]
        return SKIP, "needs --with-ingestion"
    files = {"file": ("smoke.pdf", _minimal_pdf_bytes(), "application/pdf")}
    r = _request(
        ctx,
        "POST",
        "/documents/upload",
        headers=_headers(ctx.user_token, ctx.static_api_key),
        files=files,
        timeout=ctx.timeout * 6,
    )
    if r.status_code not in (200, 202):
        return FAIL, f"PDF upload returned {r.status_code}: {r.text[:200]}"
    return PASS, "PDF accepted"


def check_query_returns_text(ctx: Context) -> tuple[str, str]:
    if not ctx.run_ingestion_checks:  # type: ignore[attr-defined]
        return SKIP, "needs --with-ingestion"
    r = _request(
        ctx,
        "POST",
        "/query",
        headers=_headers(ctx.user_token, ctx.static_api_key),
        json={"query": "What is the smoke test marker?", "mode": "naive"},
        timeout=ctx.timeout * 6,
    )
    if r.status_code != 200:
        return FAIL, f"/query returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    if not body.get("response"):
        return FAIL, f"/query response missing 'response' field: {list(body)}"
    return PASS, f"got {len(body['response'])} chars"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


CHECKS: list[tuple[str, str, Callable[[Context], tuple[str, str]]]] = [
    # group, name, fn
    ("connectivity", "health", check_health),
    ("auth",         "auth_status", check_auth_status),
    ("auth",         "oauth2_config", check_oauth2_config),
    ("auth",         "login_valid", check_login_valid),
    ("auth",         "login_invalid", check_login_invalid),
    ("auth",         "admin_login", check_admin_login),
    ("documents",    "documents_paginated", check_documents_paginated),
    ("documents",    "status_counts", check_status_counts),
    ("documents",    "pipeline_status", check_pipeline_status),
    ("graph",        "graphs_list", check_graphs_list),
    ("graph",        "graph_label_list", check_graph_label_list),
    ("apikeys",      "apikey_list", check_apikey_list),
    ("apikeys",      "apikey_create", check_apikey_create),
    ("apikeys",      "apikey_usable", check_apikey_usable),
    ("apikeys",      "apikey_delete", check_apikey_delete),
    ("admin",        "admin_status", check_admin_status),
    ("admin",        "admin_workspaces_list", check_admin_workspaces_list),
    ("admin",        "admin_routes_require_admin", check_admin_routes_require_admin),
    ("openai_compat", "openai_models", check_openai_models),
    ("openai_compat", "openai_chat_shape", check_openai_chat_shape),
    ("ingestion",    "text_ingestion", check_text_ingestion),
    ("ingestion",    "pdf_upload_routes_to_docling", check_pdf_upload_routes_to_docling),
    ("ingestion",    "query_returns_text", check_query_returns_text),
]


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _format_status(status: str) -> str:
    return {
        PASS: _color("PASS", "32"),
        FAIL: _color("FAIL", "31"),
        SKIP: _color("SKIP", "33"),
    }.get(status, status)


def run(ctx: Context, only: set[str], skip: set[str]) -> int:
    for group, name, fn in CHECKS:
        if only and not (group in only or name in only):
            continue
        if name in skip or group in skip:
            ctx.results.append(CheckResult(name, SKIP, "filtered by --skip", 0))
            continue
        t0 = time.time()
        try:
            status, msg = fn(ctx)
        except requests.RequestException as e:
            status, msg = FAIL, f"HTTP error: {e.__class__.__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            status, msg = FAIL, f"exception: {e.__class__.__name__}: {e}"
        dt = int((time.time() - t0) * 1000)
        ctx.results.append(CheckResult(name, status, msg, dt))
        print(f"  [{_format_status(status):>14}] {name:<32} {dt:>5}ms  {msg}")

    passed = sum(1 for r in ctx.results if r.status == PASS)
    failed = sum(1 for r in ctx.results if r.status == FAIL)
    skipped = sum(1 for r in ctx.results if r.status == SKIP)
    print()
    print(f"Summary: {_format_status(PASS)} {passed}   "
          f"{_format_status(FAIL)} {failed}   "
          f"{_format_status(SKIP)} {skipped}")
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://localhost:9621",
                   help="Server base URL (default: %(default)s)")
    p.add_argument("--username", help="JWT username (omit to skip /login checks)")
    p.add_argument("--password", help="JWT password")
    p.add_argument("--admin-username", help="Admin JWT username (for /admin checks)")
    p.add_argument("--admin-password", help="Admin JWT password")
    p.add_argument("--api-key", help="Static API key (use instead of JWT)")
    p.add_argument("--timeout", type=int, default=15,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--only", action="append", default=[],
                   metavar="GROUP_OR_NAME",
                   help="Run only checks matching group or name (repeatable)")
    p.add_argument("--skip", action="append", default=[],
                   metavar="GROUP_OR_NAME",
                   help="Skip checks matching group or name (repeatable)")
    p.add_argument("--with-ingestion", action="store_true",
                   help="Enable Tier 2 checks (real LLM + ingestion calls)")
    p.add_argument("--json", action="store_true",
                   help="Also emit machine-readable JSON results to stdout")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = Context(
        base_url=args.base_url,
        username=args.username,
        password=args.password,
        admin_username=args.admin_username,
        admin_password=args.admin_password,
        static_api_key=args.api_key,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    ctx.run_ingestion_checks = args.with_ingestion  # type: ignore[attr-defined]

    print(f"Enterprise smoke checks against {ctx.base_url}")
    print()
    code = run(ctx, set(args.only), set(args.skip))

    if args.json:
        payload = {
            "base_url": ctx.base_url,
            "auth_mode": ctx.auth_mode,
            "results": [
                {"name": r.name, "status": r.status, "message": r.message,
                 "duration_ms": r.duration_ms}
                for r in ctx.results
            ],
        }
        print()
        print(json.dumps(payload, indent=2))

    return code


if __name__ == "__main__":
    sys.exit(main())
