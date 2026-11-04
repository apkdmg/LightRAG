"""Start-up DDL is serialised across processes with a Postgres advisory lock.

``CREATE TABLE IF NOT EXISTS`` is not safe when two sessions create the same
table at once: the loser fails with a unique violation on ``pg_type``. Gunicorn
workers starting against an empty database hit exactly that, so every start-up
DDL sequence runs under ``PostgreSQLDB.ddl_lock()``.

The offline tests fake the advisory lock; the live test starts several fresh
processes against a newly created database at the same instant.
"""

import asyncio
import os
import subprocess
import sys
import textwrap
import time
import uuid
from contextlib import asynccontextmanager

import pytest

from lightrag.kg import postgres_impl
from lightrag.kg.postgres_impl import (
    STARTUP_DDL_LOCK_KEY,
    ClientManager,
    PostgreSQLDB,
)


class _FakeAdvisoryServer:
    """Stands in for Postgres: one advisory lock shared by all sessions."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.connections: list["_FakeConnection"] = []

    async def connect(self, **_kwargs):
        conn = _FakeConnection(self)
        self.connections.append(conn)
        return conn


class _FakeConnection:
    def __init__(self, server: _FakeAdvisoryServer):
        self.server = server
        self.holds = False
        self.closed = False

    async def fetchval(self, sql, *args):
        assert sql == "SELECT pg_try_advisory_lock($1)"
        assert args == (STARTUP_DDL_LOCK_KEY,)
        if self.holds:  # session locks stack; one is enough here
            return True
        if self.server.lock.locked():
            return False
        await self.server.lock.acquire()
        self.holds = True
        return True

    async def close(self):
        # Like Postgres, ending the session releases its advisory locks.
        if self.holds:
            self.server.lock.release()
            self.holds = False
        self.closed = True


def _db() -> PostgreSQLDB:
    db = PostgreSQLDB(
        {
            **ClientManager.get_config(),
            "user": "u",
            "password": "p",
            "database": "d",
        }
    )
    db._direct_connect_params = {}  # skip initdb(); connect is faked
    return db


@pytest.fixture
def fake_server(monkeypatch):
    server = _FakeAdvisoryServer()
    monkeypatch.setattr(postgres_impl.asyncpg, "connect", server.connect)
    monkeypatch.setattr(postgres_impl, "_DDL_LOCK_POLL_SECONDS", 0.001)
    return server


@pytest.mark.offline
async def test_ddl_lock_serialises_separate_clients(fake_server):
    inside = 0
    peak = 0

    async def start_worker(db):
        nonlocal inside, peak
        async with db.ddl_lock():
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0.01)
            inside -= 1

    await asyncio.gather(*(start_worker(_db()) for _ in range(5)))

    assert peak == 1
    assert len(fake_server.connections) == 5
    assert all(c.closed for c in fake_server.connections)
    assert not fake_server.lock.locked()


@pytest.mark.offline
async def test_ddl_lock_is_reentrant_within_a_task(fake_server):
    db = _db()
    async with db.ddl_lock():
        async with asyncio.timeout(1):  # a second session would block forever
            async with db.ddl_lock():
                pass
    assert len(fake_server.connections) == 1
    assert not fake_server.lock.locked()


@pytest.mark.offline
async def test_ddl_lock_released_when_ddl_fails(fake_server):
    db = _db()
    with pytest.raises(RuntimeError):
        async with db.ddl_lock():
            raise RuntimeError("DDL failed")
    assert fake_server.connections[0].closed
    assert not fake_server.lock.locked()
    async with asyncio.timeout(1):
        async with db.ddl_lock():
            pass


class _RecordingDB:
    """Shared-pool stand-in that records whether DDL ran under the lock."""

    def __init__(self):
        self.locked = False
        self.statements: list[tuple[str, bool]] = []

    @asynccontextmanager
    async def ddl_lock(self):
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    async def execute(self, sql, *args, **kwargs):
        self.statements.append((sql, self.locked))


@pytest.mark.offline
async def test_api_stores_create_tables_under_the_lock(monkeypatch):
    from lightrag.api.insights.store import PostgresUsageStore
    from lightrag.api.workspace_registry import PostgresWorkspaceRegistry

    db = _RecordingDB()

    async def get_client(_vector_storage=None):
        return db

    monkeypatch.setattr(postgres_impl.ClientManager, "get_client", get_client)

    await PostgresWorkspaceRegistry(None).initialize()
    await PostgresUsageStore(None).initialize()

    created = [sql for sql, _ in db.statements if "CREATE" in sql]
    assert len(created) >= 2 + 5  # registry tables + usage tables
    assert all(locked for _, locked in db.statements), db.statements


# --------------------------------------------------------------------------- #
# Live: several processes start against one empty database at the same time.
# --------------------------------------------------------------------------- #

_WORKER = textwrap.dedent(
    """
    import asyncio, os, sys, time

    from lightrag.api.insights.store import PostgresUsageStore
    from lightrag.api.workspace_registry import PostgresWorkspaceRegistry


    async def main():
        # Line up with the other processes so the CREATE statements overlap.
        time.sleep(max(0.0, float(os.environ["START_AT"]) - time.time()))
        registry = PostgresWorkspaceRegistry(None)
        usage = PostgresUsageStore(None)
        await registry.initialize()  # also creates the core LightRAG tables
        await usage.initialize()
        await usage.finalize()
        await registry.finalize()


    asyncio.run(main())
    """
)


def _admin_connect_params() -> dict:
    return {
        "host": os.environ["POSTGRES_HOST"],
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "database": os.environ.get("POSTGRES_DATABASE", "postgres"),
    }


@pytest.mark.integration
@pytest.mark.requires_db
def test_concurrent_startup_against_empty_database():
    if os.getenv("LIGHTRAG_RUN_INTEGRATION", "false").lower() != "true" or not (
        os.getenv("POSTGRES_HOST")
    ):
        pytest.skip("needs LIGHTRAG_RUN_INTEGRATION=true and POSTGRES_*")
    import asyncpg

    database = f"lightrag_ddl_race_{uuid.uuid4().hex[:8]}"

    async def admin(sql: str):
        conn = await asyncpg.connect(**_admin_connect_params())
        try:
            await conn.execute(sql)
        finally:
            await conn.close()

    async def table_count() -> int:
        conn = await asyncpg.connect(
            **{**_admin_connect_params(), "database": database}
        )
        try:
            return await conn.fetchval(
                "SELECT count(*) FROM pg_tables WHERE tablename LIKE 'lightrag_%'"
            )
        finally:
            await conn.close()

    asyncio.run(admin(f'CREATE DATABASE "{database}"'))
    try:
        env = {
            **os.environ,
            "POSTGRES_DATABASE": database,
            "START_AT": str(time.time() + 4),
        }
        env.pop("POSTGRES_WORKSPACE", None)
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _WORKER],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for _ in range(4)
        ]
        try:
            outputs = [p.communicate(timeout=120)[0] for p in procs]
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
        failures = [
            out for p, out in zip(procs, outputs, strict=True) if p.returncode != 0
        ]
        assert not failures, "\n\n".join(failures)
        assert asyncio.run(table_count()) > 0
    finally:
        asyncio.run(admin(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
