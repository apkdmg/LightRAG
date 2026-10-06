"""
First-start Milvus authentication setup.

When Milvus runs with authentication enabled
(``common.security.authorizationEnabled: true``), LightRAG should connect as
a least-privilege application user rather than as ``root``. If
``MILVUS_ROOT_PASSWORD`` is set, :func:`ensure_milvus_app_user` prepares that
on first start, idempotently:

1. Connects as root with ``MILVUS_ROOT_PASSWORD``; if that fails but the
   factory default root password still works, rotates it to
   ``MILVUS_ROOT_PASSWORD``.
2. Creates the configured database if it does not exist.
3. Creates the ``lightrag_app`` role with ``DatabaseAdmin`` + ``CollectionAdmin``
   on that database (create/drop collections, indexes, read/write data; no
   database creation or user management).
4. Creates ``MILVUS_USER`` with ``MILVUS_PASSWORD`` and grants it the role.

Without ``MILVUS_ROOT_PASSWORD`` nothing is done, so deployments that manage
Milvus users themselves are unaffected. The root password is only needed for
the first start and can be removed from the environment afterwards.

Note: Milvus silently ignores user and privilege changes while authentication
is disabled, so setup is skipped (with a warning) in that case.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from pymilvus import MilvusClient

from lightrag.utils import logger

APP_ROLE = "lightrag_app"
APP_PRIVILEGE_GROUPS = ("DatabaseAdmin", "CollectionAdmin")
_DEFAULT_ROOT_PASSWORD = "Milvus"

_lock = threading.Lock()
_done: set[tuple[str, str, str]] = set()


class MilvusAuthSetupError(RuntimeError):
    """Raised when Milvus authentication cannot be prepared."""


def _try_client(**kwargs) -> Optional[MilvusClient]:
    """Connect and run a trivial call; None when the credentials are refused.

    Refusals are expected while probing, so pymilvus's own error logging (full
    gRPC tracebacks) is muted for the duration of the probe.
    """
    pymilvus_logger = logging.getLogger("pymilvus")
    previous_level = pymilvus_logger.level
    pymilvus_logger.setLevel(logging.CRITICAL)
    try:
        client = MilvusClient(**kwargs)
        client.list_collections()
        return client
    except Exception:
        return None
    finally:
        pymilvus_logger.setLevel(previous_level)


def _root_client(uri: str, root_password: str) -> Optional[MilvusClient]:
    """Return a root client, rotating the default root password if needed.

    Returns None when Milvus accepts anonymous connections (auth disabled).
    """
    if _try_client(uri=uri) is not None:
        return None

    client = _try_client(uri=uri, token=f"root:{root_password}")
    if client is not None:
        return client

    default_client = _try_client(uri=uri, token=f"root:{_DEFAULT_ROOT_PASSWORD}")
    if default_client is not None:
        default_client.update_password("root", _DEFAULT_ROOT_PASSWORD, root_password)
        logger.info("Milvus: rotated the default root password")
        client = _try_client(uri=uri, token=f"root:{root_password}")
        if client is not None:
            return client

    raise MilvusAuthSetupError(
        "MILVUS_ROOT_PASSWORD does not match the Milvus root password, and the "
        "factory default no longer works. Set MILVUS_ROOT_PASSWORD to the current "
        "root password, or remove it if the app user already exists."
    )


def ensure_milvus_app_user(
    uri: str,
    app_user: Optional[str],
    app_password: Optional[str],
    root_password: Optional[str],
    db_name: Optional[str] = None,
) -> None:
    """Idempotently prepare the least-privilege Milvus app user (see module doc)."""
    if not (root_password and app_user and app_password):
        return
    if not uri.startswith(("http://", "https://", "tcp://", "unix://")):
        return  # Milvus Lite (local file): no authentication

    database = (db_name or "").strip() or "default"
    key = (uri, app_user, database)
    with _lock:
        if key in _done:
            return

        root = _root_client(uri, root_password)
        if root is None:
            logger.warning(
                "Milvus: authentication is disabled on the server, so the app user "
                "was not created. Enable common.security.authorizationEnabled."
            )
            _done.add(key)
            return

        if database != "default" and database not in root.list_databases():
            root.create_database(database)
            logger.info(f"Milvus: created database '{database}'")

        if APP_ROLE not in root.list_roles():
            root.create_role(APP_ROLE)
        for privilege_group in APP_PRIVILEGE_GROUPS:
            root.grant_privilege_v2(
                role_name=APP_ROLE,
                privilege=privilege_group,
                collection_name="*",
                db_name=database,
            )

        if app_user not in root.list_users():
            root.create_user(app_user, app_password)
            logger.info(f"Milvus: created app user '{app_user}'")
        root.grant_role(app_user, APP_ROLE)

        if (
            _try_client(uri=uri, user=app_user, password=app_password, db_name=database)
            is None
        ):
            raise MilvusAuthSetupError(
                f"Milvus user '{app_user}' exists but MILVUS_PASSWORD does not match it."
            )
        _done.add(key)
