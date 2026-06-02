"""
Per-workspace bring-your-own (BYO) LLM / Vision-LLM provider credentials.

A workspace owner may optionally supply their own OpenAI-compatible provider
(base URL + API key + model) for the **text LLM** and the **vision LLM (VLM)**.
When a workspace has no override, the system default applies — that fallback is
handled by the role-config resolution in ``lightrag_server.py``; this module only
owns *storage* of the overrides.

Design notes (fork-only, self-contained to keep upstream merges trivial):

- API keys are encrypted at rest with ``cryptography.fernet.Fernet``. The key is
  derived from the ``WORKSPACE_PROVIDER_SECRET`` server secret.
- Persistence is one JSON file per workspace under
  ``working_dir/.workspace_providers/<workspace>.json`` — mirroring the existing
  ``.api_keys.json`` / ``.obo_allowlist`` file precedents, and readable
  synchronously at instance-build time (no async storage dependency).
- The persistence detail sits behind :class:`WorkspaceProviderStore` so a
  DB-backed implementation can replace the file store later for multi-replica
  deployments without touching any caller.

Two logical "slots" map onto LightRAG roles (see ``lightrag/llm_roles.py``):

- ``llm``    → applied to the ``extract``, ``keyword`` and ``query`` roles.
- ``vision`` → applied to the ``vlm`` role.
"""

from __future__ import annotations

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

logger = logging.getLogger("lightrag.api.workspace_providers")

# Role groups the two owner-facing slots map onto. Imported by the apply helper.
TEXT_LLM_ROLES: tuple[str, ...] = ("extract", "keyword", "query")
VISION_LLM_ROLE: str = "vlm"

# All BYO overrides are OpenAI-compatible by contract.
OVERRIDE_BINDING = "openai"

_SUBDIR = ".workspace_providers"
_ENV_SECRET = "WORKSPACE_PROVIDER_SECRET"


class WorkspaceProviderError(Exception):
    """Raised when an operation cannot proceed (e.g. missing encryption secret)."""


class ProviderSlot(BaseModel):
    """One OpenAI-compatible provider override (text LLM or vision LLM)."""

    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    model: Optional[str] = None
    # Opaque UI hint (which preset the owner picked). Stored verbatim, never
    # interpreted by the apply logic.
    preset_id: Optional[str] = None

    def is_active(self) -> bool:
        """An override only takes effect when both endpoint and key are present."""
        return bool(self.base_url and self.api_key and self.api_key.get_secret_value())


class WorkspaceProviderConfig(BaseModel):
    """Full per-workspace provider configuration (both slots)."""

    llm: ProviderSlot = Field(default_factory=ProviderSlot)
    vision: ProviderSlot = Field(default_factory=ProviderSlot)
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.llm.is_active() or self.vision.is_active())


def generate_secret() -> str:
    """Return a fresh Fernet-compatible secret suitable for WORKSPACE_PROVIDER_SECRET."""
    return Fernet.generate_key().decode("utf-8")


def _build_fernet(secret: str) -> Fernet:
    """Build a Fernet from an operator-supplied secret.

    Accepts either a canonical Fernet key (urlsafe-base64, 32 bytes) or any
    arbitrary passphrase, which is deterministically stretched to a 32-byte
    urlsafe-base64 key via SHA-256. This keeps configuration forgiving while
    still requiring an explicit, stable secret.
    """
    secret = secret.strip()
    try:
        return Fernet(secret.encode("utf-8"))
    except (ValueError, TypeError):
        derived = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
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
        # Default secret source: env var. The server wires this to global_args.
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

    # -- read ------------------------------------------------------------
    def get(self, workspace_id: str) -> Optional[WorkspaceProviderConfig]:
        """Load and decrypt a workspace's provider config.

        Returns ``None`` when no override is stored. On a decryption failure
        (e.g. the secret was rotated/lost) this logs an error and returns
        ``None`` so the instance falls back to the system default rather than
        failing to start — availability over hard failure.
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

        try:
            return WorkspaceProviderConfig(
                llm=self._decode_slot(raw.get("llm"), fernet),
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
            preset_id=data.get("preset_id"),
        )

    # -- write -----------------------------------------------------------
    def set(self, workspace_id: str, config: WorkspaceProviderConfig) -> None:
        """Encrypt secrets and persist a workspace's provider config."""
        fernet = self._fernet()  # fail closed if no secret
        payload: dict[str, Any] = {
            "llm": self._encode_slot(config.llm, fernet),
            "vision": self._encode_slot(config.vision, fernet),
            "updated_at": config.updated_at or datetime.now(timezone.utc).isoformat(),
            "updated_by": config.updated_by,
        }
        path = self._path(workspace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: tmp then replace.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        logger.info(f"Stored provider config for workspace '{workspace_id}'")

    def _encode_slot(self, slot: ProviderSlot, fernet: Fernet) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base_url": slot.base_url,
            "model": slot.model,
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

        ``which`` may be ``"all"`` (delete the whole record), ``"llm"`` or
        ``"vision"`` (clear a single slot, keeping the other). Returns True if
        anything was removed.
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
                logger.error(f"Failed to delete provider config for '{workspace_id}': {e}")
                return False

        if which not in ("llm", "vision"):
            raise ValueError(f"Invalid slot '{which}'; expected 'all', 'llm' or 'vision'")

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
        config = self.get(workspace_id)
        if config is None:
            return {
                "llm": _masked_slot(ProviderSlot()),
                "vision": _masked_slot(ProviderSlot()),
                "updated_at": None,
                "updated_by": None,
            }
        return {
            "llm": _masked_slot(config.llm),
            "vision": _masked_slot(config.vision),
            "updated_at": config.updated_at,
            "updated_by": config.updated_by,
        }


async def apply_workspace_provider_overrides(
    rag: Any,
    workspace_id: str,
    store: Optional["WorkspaceProviderStore"],
) -> None:
    """Apply a workspace's stored provider overrides to its LightRAG roles.

    The text-LLM override is applied to the ``extract``/``keyword``/``query``
    roles; the vision override to the ``vlm`` role. Roles without an override
    keep their env/default configuration (the system-default fallback).

    Implemented purely via the public ``rag.aupdate_llm_role_config`` API so no
    core LightRAG code is modified. Safe to call on every instance build; when
    no override is stored it is a no-op (system default applies).
    """
    if store is None:
        return
    config = store.get(workspace_id)
    if config is None or config.is_empty():
        return

    if config.llm.is_active():
        for role in TEXT_LLM_ROLES:
            await rag.aupdate_llm_role_config(
                role,
                binding=OVERRIDE_BINDING,
                host=config.llm.base_url,
                api_key=config.llm.api_key.get_secret_value(),
                model=config.llm.model,
            )
        logger.info(
            f"Applied BYO text-LLM provider for workspace '{workspace_id}'"
        )

    if config.vision.is_active():
        await rag.aupdate_llm_role_config(
            VISION_LLM_ROLE,
            binding=OVERRIDE_BINDING,
            host=config.vision.base_url,
            api_key=config.vision.api_key.get_secret_value(),
            model=config.vision.model,
        )
        logger.info(
            f"Applied BYO vision-LLM provider for workspace '{workspace_id}'"
        )


def slot_effective_view(role_cfg: dict[str, Any], active: bool) -> dict[str, Any]:
    """Build the effective-provider view for one owner-facing slot.

    ``role_cfg`` is a credential-scrubbed entry from
    ``rag.get_llm_role_config(role)``. ``source`` tells the owner whether the
    value in effect comes from their own override (``custom``) or the system
    default fallback (``system_default``). Only an allow-list of non-secret
    fields is returned.
    """
    return {
        "binding": role_cfg.get("binding"),
        "model": role_cfg.get("model"),
        "host": role_cfg.get("host"),
        "source": "custom" if active else "system_default",
    }


def _masked_slot(slot: ProviderSlot) -> dict[str, Any]:
    plain = slot.api_key.get_secret_value() if slot.api_key else ""
    return {
        "base_url": slot.base_url,
        "model": slot.model,
        "preset_id": slot.preset_id,
        "api_key_set": bool(plain),
        "api_key_preview": _mask_key(plain),
        "active": slot.is_active(),
    }
