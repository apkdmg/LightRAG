"""
Workspace ownership registry for multi-tenant LightRAG.

A user's workspace ID used to be recomputed from their email or username on
every login. That mapping is lossy (``john.doe@x`` and ``john_doe@x`` give the
same ID) and has no memory (a reassigned email inherits the previous owner's
workspace). The registry records, for each workspace, the stable identity that
owns it:

- ``kc:<sub>``       SSO users (the Keycloak subject never changes)
- ``local:<name>``   password accounts from AUTH_ACCOUNTS

``resolve_workspace`` is called at sign-in. It returns the identity's existing
workspace, claims the derived ID when nobody owns it, gives a distinct ID when
it is owned by someone else, and refuses when the same email now belongs to a
different identity (a reassigned address), so an administrator decides.

Existing workspaces keep their names: each is bound to its owner on that
person's first sign-in after this feature is deployed.

Backends: a Postgres table when Postgres is the KV storage, otherwise a JSON
file protected by an exclusive file lock (safe across Gunicorn workers).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

try:  # POSIX only; on Windows the JSON registry runs without a file lock
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

logger = logging.getLogger("lightrag.api.workspace_registry")

# Short cache for identity -> workspace lookups made on every request with a
# Keycloak access token. Sign-ins always go to the backing store.
_CACHE_TTL_SECONDS = 60


@dataclass
class WorkspaceOwner:
    workspace_id: str
    identity: str
    email: Optional[str]
    created_at: float
    last_login_at: float


class WorkspaceOwnershipError(Exception):
    """The identity may not use the workspace its email maps to."""

    def __init__(self, message: str, workspace_id: str, identity: str):
        super().__init__(message)
        self.workspace_id = workspace_id
        self.identity = identity


def sso_identity(sub: str) -> str:
    return f"kc:{sub}"


def local_identity(username: str) -> str:
    return f"local:{username}"


def storage_safe_workspace_id(source: str) -> str:
    """Derive a workspace ID that every storage backend accepts as a name.

    Lowercase ``[a-z0-9_]`` only, starting with a letter (Milvus collection
    names reject ``-`` and a leading digit). For the usual ``name@domain``
    emails this equals the historical ``sanitize_workspace_id`` result, so
    existing workspaces keep their IDs.
    """
    workspace_id = re.sub(r"[^a-z0-9_]", "_", (source or "").lower())
    workspace_id = re.sub(r"_+", "_", workspace_id).strip("_")
    if not workspace_id:
        workspace_id = "default"
    if not workspace_id[0].isalpha():
        workspace_id = f"u_{workspace_id}"
    return workspace_id


def _identity_suffix(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]


def _same_email(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


class WorkspaceRegistry(ABC):
    """Persistent workspace -> owner mapping. Implementations must make
    ``claim`` atomic: it succeeds only if neither the workspace nor the
    identity is bound yet."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, float]] = {}

    async def initialize(self) -> None:  # pragma: no cover - default no-op
        return None

    async def finalize(self) -> None:  # pragma: no cover - default no-op
        return None

    @abstractmethod
    async def get_by_identity(self, identity: str) -> Optional[WorkspaceOwner]: ...

    @abstractmethod
    async def get_by_workspace(self, workspace_id: str) -> Optional[WorkspaceOwner]: ...

    @abstractmethod
    async def claim(
        self, workspace_id: str, identity: str, email: Optional[str]
    ) -> bool: ...

    @abstractmethod
    async def touch(self, identity: str, email: Optional[str]) -> None: ...

    @abstractmethod
    async def set_owner(
        self, workspace_id: str, identity: str, email: Optional[str]
    ) -> None:
        """Bind ``workspace_id`` to ``identity`` (admin action). Any previous
        owner of the workspace and any previous workspace of the identity are
        unbound."""

    @abstractmethod
    async def release(self, workspace_id: str) -> bool: ...

    @abstractmethod
    async def list_owners(self) -> list[WorkspaceOwner]: ...

    def invalidate_cache(self) -> None:
        self._cache.clear()

    async def cached_workspace_for(self, identity: str) -> Optional[str]:
        hit = self._cache.get(identity)
        now = time.monotonic()
        if hit and now - hit[1] < _CACHE_TTL_SECONDS:
            return hit[0]
        owner = await self.get_by_identity(identity)
        if owner is None:
            self._cache.pop(identity, None)
            return None
        self._cache[identity] = (owner.workspace_id, now)
        return owner.workspace_id


async def resolve_workspace(
    registry: WorkspaceRegistry,
    identity: str,
    email: Optional[str],
    source: str,
) -> str:
    """Return the workspace an identity should use, binding it on first use.

    ``source`` is what the workspace name is derived from (the email for SSO
    users, the username for password accounts).
    """
    owner = await registry.get_by_identity(identity)
    if owner is not None:
        await registry.touch(identity, email)
        return owner.workspace_id

    candidate = storage_safe_workspace_id(source)
    for _ in range(3):  # retry if a concurrent sign-in wins a claim race
        current = await registry.get_by_workspace(candidate)
        if current is None:
            if await registry.claim(candidate, identity, email):
                logger.info(f"Workspace '{candidate}' bound to {identity}")
                return candidate
            continue

        if current.identity == identity:
            return candidate

        if _same_email(current.email, email):
            # Same email address, different person (or a re-created account):
            # never hand over the previous owner's data automatically.
            logger.warning(
                f"Refused workspace '{candidate}' for {identity}: it belongs to "
                f"{current.identity} with the same email address"
            )
            raise WorkspaceOwnershipError(
                "This account does not match the owner of its workspace. "
                "Ask an administrator to review it.",
                workspace_id=candidate,
                identity=identity,
            )

        # Different person whose name maps to the same ID: give them their own.
        distinct = f"{storage_safe_workspace_id(source)}_{_identity_suffix(identity)}"
        if candidate == distinct:
            break
        logger.warning(
            f"Workspace ID '{candidate}' already belongs to {current.identity}; "
            f"assigning '{distinct}' to {identity}"
        )
        candidate = distinct

    owner = await registry.get_by_identity(identity)
    if owner is not None:
        return owner.workspace_id
    raise WorkspaceOwnershipError(
        "Could not assign a workspace for this account.",
        workspace_id=candidate,
        identity=identity,
    )


# ---------------------------------------------------------------------------
# JSON file backend
# ---------------------------------------------------------------------------


class JsonWorkspaceRegistry(WorkspaceRegistry):
    """Registry stored in ``<working_dir>/.workspace_owners.json``.

    Every read-modify-write holds an exclusive ``flock`` on a sidecar lock
    file, so concurrent Gunicorn workers cannot both claim the same workspace.
    """

    def __init__(self, working_dir: str | os.PathLike[str]):
        super().__init__()
        self._path = Path(working_dir) / ".workspace_owners.json"
        self._lock_path = Path(working_dir) / ".workspace_owners.lock"

    def _locked(self, mutate):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                data: dict[str, dict[str, Any]] = {}
                if self._path.exists():
                    data = json.loads(self._path.read_text(encoding="utf-8") or "{}")
                result, changed = mutate(data)
                if changed:
                    tmp = self._path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    os.replace(tmp, self._path)
                return result
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    async def _run(self, mutate):
        return await asyncio.to_thread(self._locked, mutate)

    @staticmethod
    def _owner(entry: dict[str, Any]) -> WorkspaceOwner:
        return WorkspaceOwner(**entry)

    async def get_by_identity(self, identity: str) -> Optional[WorkspaceOwner]:
        def read(data):
            for entry in data.values():
                if entry["identity"] == identity:
                    return self._owner(entry), False
            return None, False

        return await self._run(read)

    async def get_by_workspace(self, workspace_id: str) -> Optional[WorkspaceOwner]:
        def read(data):
            entry = data.get(workspace_id)
            return (self._owner(entry) if entry else None), False

        return await self._run(read)

    async def claim(
        self, workspace_id: str, identity: str, email: Optional[str]
    ) -> bool:
        def write(data):
            if workspace_id in data or any(
                e["identity"] == identity for e in data.values()
            ):
                return False, False
            now = time.time()
            data[workspace_id] = asdict(
                WorkspaceOwner(workspace_id, identity, email, now, now)
            )
            return True, True

        claimed = await self._run(write)
        self.invalidate_cache()
        return claimed

    async def touch(self, identity: str, email: Optional[str]) -> None:
        def write(data):
            for entry in data.values():
                if entry["identity"] == identity:
                    entry["last_login_at"] = time.time()
                    if email:
                        entry["email"] = email
                    return None, True
            return None, False

        await self._run(write)

    async def set_owner(
        self, workspace_id: str, identity: str, email: Optional[str]
    ) -> None:
        def write(data):
            for key in [k for k, e in data.items() if e["identity"] == identity]:
                del data[key]
            now = time.time()
            previous = data.get(workspace_id)
            data[workspace_id] = asdict(
                WorkspaceOwner(
                    workspace_id,
                    identity,
                    email,
                    previous["created_at"] if previous else now,
                    now,
                )
            )
            return None, True

        await self._run(write)
        self.invalidate_cache()

    async def release(self, workspace_id: str) -> bool:
        def write(data):
            if workspace_id in data:
                del data[workspace_id]
                return True, True
            return False, False

        released = await self._run(write)
        self.invalidate_cache()
        return released

    async def list_owners(self) -> list[WorkspaceOwner]:
        def read(data):
            return [self._owner(e) for e in data.values()], False

        return await self._run(read)


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

_PG_TABLE = "LIGHTRAG_WORKSPACE_OWNERS"


class PostgresWorkspaceRegistry(WorkspaceRegistry):
    """Registry stored in a Postgres table, using the shared LightRAG pool.

    The primary key on ``workspace_id`` and the unique ``identity`` column make
    ``claim`` (INSERT ... ON CONFLICT DO NOTHING) atomic across workers.
    """

    def __init__(self, vector_storage: str | None = None):
        super().__init__()
        self._vector_storage = vector_storage
        self._db = None

    async def initialize(self) -> None:
        from lightrag.kg.postgres_impl import ClientManager

        self._db = await ClientManager.get_client(self._vector_storage)
        await self._db.execute(
            f"""CREATE TABLE IF NOT EXISTS {_PG_TABLE} (
                workspace_id VARCHAR(255) PRIMARY KEY,
                identity VARCHAR(512) NOT NULL UNIQUE,
                email VARCHAR(512),
                created_at DOUBLE PRECISION NOT NULL,
                last_login_at DOUBLE PRECISION NOT NULL
            )"""
        )

    async def finalize(self) -> None:
        if self._db is not None:
            from lightrag.kg.postgres_impl import ClientManager

            await ClientManager.release_client(self._db)
            self._db = None

    @staticmethod
    def _owner(row: Optional[dict[str, Any]]) -> Optional[WorkspaceOwner]:
        if not row:
            return None
        return WorkspaceOwner(
            workspace_id=row["workspace_id"],
            identity=row["identity"],
            email=row["email"],
            created_at=float(row["created_at"]),
            last_login_at=float(row["last_login_at"]),
        )

    async def get_by_identity(self, identity: str) -> Optional[WorkspaceOwner]:
        row = await self._db.query(
            f"SELECT * FROM {_PG_TABLE} WHERE identity = $1", [identity]
        )
        return self._owner(row)

    async def get_by_workspace(self, workspace_id: str) -> Optional[WorkspaceOwner]:
        row = await self._db.query(
            f"SELECT * FROM {_PG_TABLE} WHERE workspace_id = $1", [workspace_id]
        )
        return self._owner(row)

    async def claim(
        self, workspace_id: str, identity: str, email: Optional[str]
    ) -> bool:
        now = time.time()
        row = await self._db.query(
            f"""INSERT INTO {_PG_TABLE}
                (workspace_id, identity, email, created_at, last_login_at)
                VALUES ($1, $2, $3, $4, $4)
                ON CONFLICT DO NOTHING
                RETURNING workspace_id""",
            [workspace_id, identity, email, now],
        )
        self.invalidate_cache()
        return bool(row)

    async def touch(self, identity: str, email: Optional[str]) -> None:
        await self._db.execute(
            f"""UPDATE {_PG_TABLE}
                SET last_login_at = $2, email = COALESCE($3, email)
                WHERE identity = $1""",
            {"identity": identity, "now": time.time(), "email": email},
        )

    async def set_owner(
        self, workspace_id: str, identity: str, email: Optional[str]
    ) -> None:
        now = time.time()
        await self._db.execute(
            f"DELETE FROM {_PG_TABLE} WHERE identity = $1 AND workspace_id <> $2",
            {"identity": identity, "workspace_id": workspace_id},
        )
        await self._db.execute(
            f"""INSERT INTO {_PG_TABLE}
                (workspace_id, identity, email, created_at, last_login_at)
                VALUES ($1, $2, $3, $4, $4)
                ON CONFLICT (workspace_id) DO UPDATE
                SET identity = EXCLUDED.identity,
                    email = EXCLUDED.email,
                    last_login_at = EXCLUDED.last_login_at""",
            {
                "workspace_id": workspace_id,
                "identity": identity,
                "email": email,
                "now": now,
            },
        )
        self.invalidate_cache()

    async def release(self, workspace_id: str) -> bool:
        row = await self._db.query(
            f"DELETE FROM {_PG_TABLE} WHERE workspace_id = $1 RETURNING workspace_id",
            [workspace_id],
        )
        self.invalidate_cache()
        return bool(row)

    async def list_owners(self) -> list[WorkspaceOwner]:
        rows = await self._db.query(
            f"SELECT * FROM {_PG_TABLE} ORDER BY workspace_id", multirows=True
        )
        return [self._owner(r) for r in rows or []]


def create_workspace_registry(args: Any) -> WorkspaceRegistry:
    """Pick the registry backend for this deployment."""
    if getattr(args, "kv_storage", "") == "PGKVStorage":
        return PostgresWorkspaceRegistry(getattr(args, "vector_storage", None))
    return JsonWorkspaceRegistry(args.working_dir)
