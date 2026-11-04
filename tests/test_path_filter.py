"""Access-log path filter: query strings, GET-only prefixes, POSTs kept."""

from __future__ import annotations

import logging

import pytest

from lightrag.utils import LightragPathFilter

pytestmark = pytest.mark.offline


def _access(method: str, path: str, status: int = 200) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", method, path, "1.1", status),
        None,
    )


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/health"),
        ("GET", "/documents"),
        ("POST", "/documents/paginated"),
        ("GET", "/documents/pipeline_status"),
        ("GET", "/documents/pipeline_status?workspace=a"),
        ("POST", "/documents/paginated?page=2"),
        ("GET", "/webui/"),
        ("GET", "/admin/ops/status"),
        ("GET", "/admin/ops/processing/active?page=2"),
        ("GET", "/admin/insights/live"),
        ("GET", "/admin/insights/live?minutes=5"),
        ("GET", "/api-prefix/admin/ops/overview"),
    ],
)
def test_successful_polls_are_dropped(method, path):
    assert LightragPathFilter().filter(_access(method, path)) is False


@pytest.mark.parametrize(
    "method,path,status",
    [
        # Ops actions are always kept in the access log.
        ("POST", "/admin/ops/workspaces/ws/actions/cancel", 200),
        ("POST", "/admin/ops/logs/export", 200),
        # Failures are always kept.
        ("GET", "/admin/ops/status", 403),
        ("GET", "/health", 500),
        # Other routes are kept.
        ("GET", "/admin/insights/overview", 200),
        ("DELETE", "/documents", 200),
        ("GET", "/documents/abc", 200),
        ("GET", "/admin/opsx", 200),
    ],
)
def test_other_requests_are_kept(method, path, status):
    assert LightragPathFilter().filter(_access(method, path, status)) is True


def test_non_access_records_pass_through():
    flt = LightragPathFilter()
    plain = logging.LogRecord("lightrag", logging.INFO, __file__, 1, "hi", (), None)
    assert flt.filter(plain) is True
    odd = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, "%s", ("x",), None
    )
    assert flt.filter(odd) is True
    broken = _access("GET", "/health")
    broken.args = ("a", "GET", None, "1.1", 200)
    assert flt.filter(broken) is True
