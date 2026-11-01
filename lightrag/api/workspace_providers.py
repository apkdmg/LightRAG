"""
Per-workspace bring-your-own (BYO) LLM / Vision-LLM provider credentials.

A workspace owner may optionally supply their own OpenAI-compatible provider
(base URL + API key + model, plus an optional reasoning effort) for each role
group:

- ``extraction`` → applied to the ``extract`` and ``keyword`` roles
  (ingestion + query-keyword extraction; a fast, non-thinking model is ideal).
- ``query``      → applied to the ``query`` role (answer generation; a reasoning
  model is usually preferred).
- ``vision``     → applied to the ``vlm`` role (image/infographic description).
  Email images are the exception: they always use the system key (see
  ``_system_vision_func`` in ``routers/email_routes.py``).

When a slot has no override, the system default applies — that fallback is
handled by the role-config resolution in ``lightrag_server.py``; this module
only owns *storage* of the overrides and *applying* them to a LightRAG instance.

Design notes (fork-only, self-contained to keep upstream merges trivial):

- API keys are encrypted at rest with ``cryptography.fernet.Fernet``. The key is
  derived from the ``WORKSPACE_PROVIDER_SECRET`` server secret.
- Persistence is one JSON file per workspace under
  ``working_dir/.workspace_providers/<workspace>.json``.
- The persistence detail sits behind :class:`WorkspaceProviderStore` so a
  DB-backed implementation can replace the file store later without touching
  any caller.
- Overrides are applied purely via the public ``rag.aupdate_llm_role_config``
  API (no core LightRAG code is modified).

An admin can also set the **server default**, stored in the same encrypted store
under :data:`SYSTEM_RECORD_ID`. For each slot, the order of precedence is: the
workspace's own provider, then the server default, then ``.env``. Unlike a
workspace's provider, the server default is billed as a system cost.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, SecretStr

from lightrag.usage import BYO_BILLING, SYSTEM_BILLING

logger = logging.getLogger("lightrag.api.workspace_providers")

# Role groups the three owner-facing slots map onto. Imported by the routes.
EXTRACTION_ROLES: tuple[str, ...] = ("extract", "keyword")
QUERY_ROLE: str = "query"
VISION_LLM_ROLE: str = "vlm"

# Slot name -> the roles it drives.
SLOT_ROLES: dict[str, tuple[str, ...]] = {
    "extraction": EXTRACTION_ROLES,
    "query": (QUERY_ROLE,),
    "vision": (VISION_LLM_ROLE,),
}

# All BYO overrides are OpenAI-compatible by contract.
OVERRIDE_BINDING = "openai"

# Accepted reasoning-effort values (OpenAI-compatible; Gemini maps these to its
# thinking budget). None/"" means "leave the provider default".
REASONING_EFFORTS = ("none", "low", "medium", "high")

_SUBDIR = ".workspace_providers"
_ENV_SECRET = "WORKSPACE_PROVIDER_SECRET"

# Store record for the server default. Sanitized workspace ids can't contain a
# dot, so no workspace, and no X-Target-Workspace header, can reach it.
SYSTEM_RECORD_ID = ".system"


class WorkspaceProviderError(Exception):
    """Raised when an operation cannot proceed (e.g. missing encryption secret)."""


class ProviderSlot(BaseModel):
    """One OpenAI-compatible provider override for a role group."""

    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    model: Optional[str] = None
    # Optional thinking/reasoning control (none|low|medium|high). None = default.
    reasoning_effort: Optional[str] = None
    # Opaque UI hint (which preset the owner picked). Stored verbatim.
    preset_id: Optional[str] = None

    def is_active(self) -> bool:
        """An override only takes effect when both endpoint and key are present."""
        return bool(self.base_url and self.api_key and self.api_key.get_secret_value())


class WorkspaceProviderConfig(BaseModel):
    """Full per-workspace provider configuration (three role-group slots)."""

    extraction: ProviderSlot = Field(default_factory=ProviderSlot)
    query: ProviderSlot = Field(default_factory=ProviderSlot)
    vision: ProviderSlot = Field(default_factory=ProviderSlot)
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

    def slot(self, name: str) -> ProviderSlot:
        return getattr(self, name)

    def is_empty(self) -> bool:
        return not (
            self.extraction.is_active()
            or self.query.is_active()
            or self.vision.is_active()
        )


def generate_secret() -> str:
    """Return a fresh Fernet-compatible secret for WORKSPACE_PROVIDER_SECRET."""
    return Fernet.generate_key().decode("utf-8")


def _build_fernet(secret: str) -> Fernet:
    """Build a Fernet from an operator-supplied secret.

    Accepts either a canonical Fernet key (urlsafe-base64, 32 bytes) or any
    arbitrary passphrase, which is deterministically stretched to a 32-byte
    urlsafe-base64 key via SHA-256.
    """
    secret = secret.strip()
    try:
        return Fernet(secret.encode("utf-8"))
    except (ValueError, TypeError):
        derived = base64.urlsafe_b64encode(
            hashlib.sha256(secret.encode("utf-8")).digest()
        )
        return Fernet(derived)


def _mask_key(plain: str) -> str:
    """Return a non-reversible preview of a secret (last 4 chars only)."""
    if not plain:
        return ""
    tail = plain[-4:] if len(plain) >= 4 else plain
    return f"...{tail}"


class WorkspaceProviderStore:
    """File-backed, encrypted store for per-workspace provider overrides.

    The storage detail is intentionally hidden behind this class; a DB-backed
    implementation can be substituted later without changing callers.
    """

    def __init__(
        self,
        working_dir: str | os.PathLike[str],
        get_secret: Callable[[], Optional[str]] | None = None,
    ) -> None:
        self._base_dir = Path(working_dir) / _SUBDIR
        self._get_secret = get_secret or (lambda: os.getenv(_ENV_SECRET))

    # -- secret handling -------------------------------------------------
    def has_secret(self) -> bool:
        secret = self._get_secret()
        return bool(secret and secret.strip())

    def _fernet(self) -> Fernet:
        secret = self._get_secret()
        if not secret or not secret.strip():
            raise WorkspaceProviderError(
                f"{_ENV_SECRET} is not configured; cannot encrypt/decrypt "
                "workspace provider credentials. Set it to a stable secret "
                "(generate one via WorkspaceProviderStore.generate_secret())."
            )
        return _build_fernet(secret)

    # -- paths -----------------------------------------------------------
    def _path(self, workspace_id: str) -> Path:
        return self._base_dir / f"{workspace_id}.json"

    def record_mtime(self, workspace_id: str) -> Optional[float]:
        """When a record last changed, or None if there isn't one."""
        try:
            return self._path(workspace_id).stat().st_mtime
        except OSError:
            return None

    def is_unreadable(self, workspace_id: str) -> bool:
        """True when a record exists but can't be used (e.g. the secret changed)."""
        return self._path(workspace_id).exists() and self.get(workspace_id) is None

    def workspaces_with_own_provider(self) -> dict[str, list[str]]:
        """Slot name -> ids of the workspaces with their own provider for it."""
        result: dict[str, list[str]] = {name: [] for name in SLOT_ROLES}
        if not self._base_dir.exists():
            return result
        for path in sorted(self._base_dir.glob("*.json")):
            if path.name.startswith("."):
                continue
            config = self.get(path.stem)
            if config is None:
                continue
            for name in SLOT_ROLES:
                if config.slot(name).is_active():
                    result[name].append(path.stem)
        return result

    # -- read ------------------------------------------------------------
    def get(self, workspace_id: str) -> Optional[WorkspaceProviderConfig]:
        """Load and decrypt a workspace's provider config.

        Returns ``None`` when no override is stored. On a decryption failure
        (e.g. the secret was rotated/lost) this logs an error and returns
        ``None`` so the instance falls back to the system default rather than
        failing to start.

        Backward compatibility: an older config used a single ``llm`` slot
        (applied to extract/keyword/query). It is migrated on read by mapping
        ``llm`` onto both ``extraction`` and ``query``.
        """
        path = self._path(workspace_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read provider config for '{workspace_id}': {e}")
            return None

        try:
            fernet = self._fernet()
        except WorkspaceProviderError as e:
            logger.error(f"Cannot decrypt provider config for '{workspace_id}': {e}")
            return None

        legacy = raw.get("llm")  # pre role-aware format
        try:
            return WorkspaceProviderConfig(
                extraction=self._decode_slot(raw.get("extraction") or legacy, fernet),
                query=self._decode_slot(raw.get("query") or legacy, fernet),
                vision=self._decode_slot(raw.get("vision"), fernet),
                updated_at=raw.get("updated_at"),
                updated_by=raw.get("updated_by"),
            )
        except InvalidToken:
            logger.error(
                f"Provider config for '{workspace_id}' could not be decrypted "
                f"(wrong {_ENV_SECRET}?); ignoring override."
            )
            return None

    def _decode_slot(self, data: Optional[dict], fernet: Fernet) -> ProviderSlot:
        if not data:
            return ProviderSlot()
        api_key: Optional[SecretStr] = None
        enc = data.get("api_key_enc")
        if enc:
            api_key = SecretStr(fernet.decrypt(enc.encode("utf-8")).decode("utf-8"))
        return ProviderSlot(
            base_url=data.get("base_url"),
            api_key=api_key,
            model=data.get("model"),
            reasoning_effort=data.get("reasoning_effort"),
            preset_id=data.get("preset_id"),
        )

    # -- write -----------------------------------------------------------
    def set(self, workspace_id: str, config: WorkspaceProviderConfig) -> None:
        """Encrypt secrets and persist a workspace's provider config."""
        fernet = self._fernet()  # fail closed if no secret
        payload: dict[str, Any] = {
            "extraction": self._encode_slot(config.extraction, fernet),
            "query": self._encode_slot(config.query, fernet),
            "vision": self._encode_slot(config.vision, fernet),
            "updated_at": config.updated_at or datetime.now(timezone.utc).isoformat(),
            "updated_by": config.updated_by,
        }
        path = self._path(workspace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        logger.info(f"Stored provider config for workspace '{workspace_id}'")

    def _encode_slot(self, slot: ProviderSlot, fernet: Fernet) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base_url": slot.base_url,
            "model": slot.model,
            "reasoning_effort": slot.reasoning_effort,
            "preset_id": slot.preset_id,
        }
        if slot.api_key and slot.api_key.get_secret_value():
            out["api_key_enc"] = fernet.encrypt(
                slot.api_key.get_secret_value().encode("utf-8")
            ).decode("utf-8")
        return out

    # -- delete ----------------------------------------------------------
    def delete(self, workspace_id: str, which: str = "all") -> bool:
        """Remove an override.

        ``which`` may be ``"all"`` (delete the whole record) or a slot name
        (``"extraction"`` / ``"query"`` / ``"vision"``) to clear one slot,
        keeping the others. Returns True if anything was removed.
        """
        path = self._path(workspace_id)
        if not path.exists():
            return False
        if which == "all":
            try:
                path.unlink()
                logger.info(f"Deleted provider config for workspace '{workspace_id}'")
                return True
            except OSError as e:
                logger.error(
                    f"Failed to delete provider config for '{workspace_id}': {e}"
                )
                return False

        if which not in SLOT_ROLES:
            raise ValueError(
                f"Invalid slot '{which}'; expected 'all' or one of {list(SLOT_ROLES)}"
            )

        config = self.get(workspace_id)
        if config is None:
            return False
        setattr(config, which, ProviderSlot())
        if config.is_empty():
            return self.delete(workspace_id, "all")
        self.set(workspace_id, config)
        return True

    # -- masked view (safe for API responses) ----------------------------
    def get_masked(self, workspace_id: str) -> dict[str, Any]:
        """Return a secret-free view of the stored config for API responses."""
        config = self.get(workspace_id) or WorkspaceProviderConfig()
        return {
            "extraction": _masked_slot(config.extraction),
            "query": _masked_slot(config.query),
            "vision": _masked_slot(config.vision),
            "updated_at": config.updated_at,
            "updated_by": config.updated_by,
        }


async def _apply_slot(rag: Any, roles: tuple[str, ...], slot: ProviderSlot) -> None:
    """Apply one slot's override to its role(s) via the public role API."""
    api_key = slot.api_key.get_secret_value()
    for role in roles:
        provider_options = None
        if slot.reasoning_effort:
            # Merge onto the role's existing (scrubbed) provider options so we
            # don't drop temperature/etc.; get_llm_role_config never leaks keys.
            current = rag.get_llm_role_config(role)
            provider_options = dict(
                (current.get("metadata") or {}).get("provider_options") or {}
            )
            provider_options["reasoning_effort"] = slot.reasoning_effort
        await rag.aupdate_llm_role_config(
            role,
            binding=OVERRIDE_BINDING,
            host=slot.base_url,
            api_key=api_key,
            model=slot.model,
            provider_options=provider_options,
        )
        # Usage recording reads this: calls on this role now bill the owner's key.
        rag.set_role_llm_metadata(role, billing=BYO_BILLING)


async def apply_workspace_provider_overrides(
    rag: Any,
    workspace_id: str,
    store: Optional["WorkspaceProviderStore"],
) -> None:
    """Apply a workspace's stored provider overrides to its LightRAG roles.

    ``extraction`` → ``extract``/``keyword``; ``query`` → ``query``;
    ``vision`` → ``vlm``. Roles without an override keep their env/default
    configuration (the system-default fallback). Safe to call on every instance
    build; a no-op when nothing is stored.
    """
    if store is None:
        return
    config = store.get(workspace_id)
    if config is None or config.is_empty():
        return

    if config.extraction.is_active():
        await _apply_slot(rag, EXTRACTION_ROLES, config.extraction)
        logger.info(f"Applied BYO extraction provider for workspace '{workspace_id}'")
    if config.query.is_active():
        await _apply_slot(rag, (QUERY_ROLE,), config.query)
        logger.info(f"Applied BYO query provider for workspace '{workspace_id}'")
    if config.vision.is_active():
        await _apply_slot(rag, (VISION_LLM_ROLE,), config.vision)
        logger.info(f"Applied BYO vision provider for workspace '{workspace_id}'")


def _system_role_target(
    slot: Optional[ProviderSlot], env: dict[str, Any]
) -> dict[str, Any]:
    """The settings a role should run with when no workspace provider applies.

    Always built from the ``.env`` settings rather than the role's current ones,
    so that clearing a value restores what ``.env`` says.
    """
    if slot is None:
        return {
            "binding": env["binding"],
            "model": env["model"],
            "host": env["host"],
            # Empty, not None: None would leave a stale key on the role, and
            # the Bedrock binding refuses to build with a key.
            "api_key": env["api_key"] or "",
            "provider_options": env["provider_options"] or {},
        }
    # Options for another binding don't apply to an OpenAI-compatible provider.
    same_binding = env["binding"] == OVERRIDE_BINDING
    options = dict(env["provider_options"] or {}) if same_binding else {}
    if slot.reasoning_effort:
        options["reasoning_effort"] = slot.reasoning_effort
    return {
        "binding": OVERRIDE_BINDING,
        "model": slot.model,
        "host": slot.base_url,
        "api_key": slot.api_key.get_secret_value(),
        "provider_options": options,
    }


def reconcile_system_provider(
    rag: Any,
    workspace_id: Optional[str],
    store: WorkspaceProviderStore,
    env_settings: Callable[[str], dict[str, Any]],
) -> list[str]:
    """Put a LightRAG instance on the server default, or back on ``.env``.

    Roles covered by the workspace's own provider are left alone. Pass
    ``workspace_id=None`` for the server's default instance, which never uses a
    workspace's provider because email images must stay on the system key.

    Swaps wrappers in place instead of rebuilding the instance, so a running
    pipeline keeps going: calls already sent finish on the old provider, and
    later ones use the new one. Returns the roles that changed.
    """
    system = store.get(SYSTEM_RECORD_ID)
    own = store.get(workspace_id) if workspace_id is not None else None
    changed: list[str] = []
    for slot_name, roles in SLOT_ROLES.items():
        if own is not None and own.slot(slot_name).is_active():
            continue
        slot = system.slot(slot_name) if system is not None else None
        if slot is not None and not slot.is_active():
            slot = None
        for role in roles:
            try:
                target = _system_role_target(slot, env_settings(role))
                if rag.role_llm_config_matches(role, **target):
                    continue
                rag.update_llm_role_config(role, **target)
                rag.set_role_llm_metadata(role, billing=SYSTEM_BILLING)
                changed.append(role)
            except Exception as e:
                # One role failing to build must not stop the others.
                logger.error(
                    f"Could not apply the server default to role '{role}' "
                    f"of workspace '{workspace_id}': {e}"
                )
    return changed


async def reconcile_loaded_instances(
    default_rag: Any,
    workspace_manager: Any,
    store: WorkspaceProviderStore,
    env_settings: Callable[[str], dict[str, Any]],
) -> None:
    """Reconcile the default instance and every workspace instance in memory.

    Instances built later pick up the server default as they are created. Yields
    between instances, because thousands can be loaded and requests keep running.
    """
    reconcile_system_provider(default_rag, None, store, env_settings)
    for rag in workspace_manager.loaded_instances():
        reconcile_system_provider(rag, rag.workspace, store, env_settings)
        await asyncio.sleep(0)


async def watch_system_provider(
    default_rag: Any,
    workspace_manager: Any,
    store: WorkspaceProviderStore,
    env_settings: Callable[[str], dict[str, Any]],
    interval_seconds: float = 10.0,
) -> None:
    """Keep this process on the current server default.

    The record is shared on disk, so this is how a save handled by one Gunicorn
    worker reaches the others. In the worker that handled it, nothing is left to
    do by the time this notices.
    """
    last_seen: Optional[float] = None
    while True:
        await asyncio.sleep(interval_seconds)
        mtime = store.record_mtime(SYSTEM_RECORD_ID)
        if mtime == last_seen:
            continue
        last_seen = mtime
        try:
            await reconcile_loaded_instances(
                default_rag, workspace_manager, store, env_settings
            )
        except Exception as e:
            logger.error(f"Could not apply the server default: {e}")


def resolve_server_slot_key(
    slot_name: str,
    base_url: str,
    config: WorkspaceProviderConfig,
    env_settings: Callable[[str], dict[str, Any]],
) -> Optional[SecretStr]:
    """Find the key already in use for this base URL, for a blank key field.

    Admins may never have seen the ``.env`` key, so they can change a model
    without retyping it. A key is only reused for the base URL it belongs to, so
    it can't be sent to a different host.
    """

    def same_host(url: Optional[str]) -> bool:
        return (url or "").rstrip("/") == base_url.rstrip("/")

    for name in (slot_name, *(n for n in SLOT_ROLES if n != slot_name)):
        slot = config.slot(name)
        if slot.is_active() and same_host(slot.base_url):
            return slot.api_key
    for name in (slot_name, *(n for n in SLOT_ROLES if n != slot_name)):
        env = env_settings(SLOT_ROLES[name][0])
        if env["api_key"] and same_host(env["host"]):
            return SecretStr(env["api_key"])
    return None


def slot_effective_view(role_cfg: dict[str, Any], active: bool) -> dict[str, Any]:
    """Build the effective-provider view for one owner-facing slot.

    ``role_cfg`` is a credential-scrubbed entry from
    ``rag.get_llm_role_config(role)``. ``source`` tells the owner whether the
    value in effect comes from their own override (``custom``) or the system
    default fallback (``system_default``).
    """
    meta = role_cfg.get("metadata") or {}
    options = meta.get("provider_options") or {}
    return {
        "binding": role_cfg.get("binding"),
        "model": role_cfg.get("model"),
        "host": role_cfg.get("host"),
        "reasoning_effort": options.get("reasoning_effort"),
        "source": "custom" if active else "system_default",
    }


def _masked_slot(slot: ProviderSlot) -> dict[str, Any]:
    plain = slot.api_key.get_secret_value() if slot.api_key else ""
    return {
        "base_url": slot.base_url,
        "model": slot.model,
        "reasoning_effort": slot.reasoning_effort,
        "preset_id": slot.preset_id,
        "api_key_set": bool(plain),
        "api_key_preview": _mask_key(plain),
        "active": slot.is_active(),
    }
