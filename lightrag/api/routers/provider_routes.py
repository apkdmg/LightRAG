"""
Per-workspace BYO LLM / Vision-LLM provider routes.

Lets a workspace **owner** (self-service) — or an admin acting on behalf of a
workspace via ``X-Target-Workspace`` — register, view, update and clear their own
OpenAI-compatible provider (base URL + API key + model) for the text LLM and the
vision LLM. The system default is used as fallback when no override is set.

Changes take effect with **no server restart**: after persisting, the workspace's
cached LightRAG instance is invalidated, so the next request rebuilds it and the
build-time hook re-applies the (new) override or falls back to the system default.

This router is registered only when both multi-tenancy and the feature flag
(``ENABLE_WORKSPACE_PROVIDERS``) are enabled. It is fork-only and self-contained.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, SecretStr

from lightrag.api.dependencies import (
    UserInfo,
    get_current_user,
    get_current_workspace,
    get_workspace_manager,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.workspace_providers import (
    VISION_LLM_ROLE,
    ProviderSlot,
    WorkspaceProviderConfig,
    WorkspaceProviderError,
    WorkspaceProviderStore,
    slot_effective_view,
)

# Role used to represent each owner-facing slot in the effective view. The three
# text roles share the same override, so any one of them is representative.
_SLOT_REPRESENTATIVE_ROLE = {"llm": "query", "vision": VISION_LLM_ROLE}

logger = logging.getLogger("lightrag.api.provider_routes")


# ----------------------------- request/response models -----------------------------
class ProviderSlotInput(BaseModel):
    """One provider slot as submitted by the client (plaintext key over TLS)."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None  # omit to keep the existing stored key
    model: Optional[str] = None
    preset_id: Optional[str] = None


class UpdateProviderConfigRequest(BaseModel):
    """Update request. Omitted slots are left unchanged; use DELETE to clear."""

    llm: Optional[ProviderSlotInput] = None
    vision: Optional[ProviderSlotInput] = None


def _get_store(request: Request) -> WorkspaceProviderStore:
    store = getattr(request.app.state, "workspace_provider_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Per-workspace providers are not enabled on this server",
        )
    return store


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base URL '{base_url}'; must be an http(s) URL",
        )


def _merge_slot(existing: ProviderSlot, incoming: ProviderSlotInput) -> ProviderSlot:
    """Merge a submitted slot onto the existing stored slot.

    A missing ``api_key`` keeps the previously stored key (lets owners edit the
    base URL / model without re-entering their secret). The merged result is
    validated to be a complete, active override.
    """
    base_url = incoming.base_url if incoming.base_url is not None else existing.base_url
    model = incoming.model if incoming.model is not None else existing.model
    preset_id = (
        incoming.preset_id if incoming.preset_id is not None else existing.preset_id
    )
    if incoming.api_key:
        api_key: Optional[SecretStr] = SecretStr(incoming.api_key)
    else:
        api_key = existing.api_key

    if not base_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="base_url is required",
        )
    _validate_base_url(base_url)
    if not (api_key and api_key.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="api_key is required",
        )
    if not model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model is required",
        )
    return ProviderSlot(
        base_url=base_url, api_key=api_key, model=model, preset_id=preset_id
    )


async def _probe_provider(slot: ProviderSlot) -> None:
    """Best-effort connectivity/auth probe against the provider's /models.

    Raises 400 on a clear connection or auth failure; tolerates providers that
    do not implement /models (only hard auth/connection errors are fatal).
    """
    url = slot.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {slot.api_key.get_secret_value()}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not reach provider at {slot.base_url}: {e}",
        )
    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider rejected the API key (authentication failed)",
        )


def create_provider_routes(api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(prefix="/workspace", tags=["providers"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/provider-config",
        dependencies=[Depends(combined_auth)],
    )
    async def get_provider_config(
        request: Request,
        workspace: str = Depends(get_current_workspace),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Return the masked provider config plus the **effective** provider in
        use for each slot — including the system default's host/model when no
        override is set, so the owner always sees what is actually being called.
        """
        store = _get_store(request)
        masked = store.get_masked(workspace)
        # Build/fetch the workspace instance and read its live, scrubbed role
        # config so the effective host/model reflect any applied override.
        rag = await workspace_manager.get_instance(workspace)
        roles = rag.get_llm_role_config()
        masked["llm"]["effective"] = slot_effective_view(
            roles[_SLOT_REPRESENTATIVE_ROLE["llm"]], masked["llm"]["active"]
        )
        masked["vision"]["effective"] = slot_effective_view(
            roles[_SLOT_REPRESENTATIVE_ROLE["vision"]], masked["vision"]["active"]
        )
        return masked

    @router.get(
        "/provider-config/effective",
        dependencies=[Depends(combined_auth)],
    )
    async def get_effective_role_config(
        request: Request,
        workspace: str = Depends(get_current_workspace),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Return the live, credential-scrubbed config of every LLM role.

        Ground-truth introspection of what each role (``extract``/``keyword``/
        ``query`` = text LLM, ``vlm`` = vision LLM) is actually calling, with a
        ``source`` of ``custom`` (owner override) or ``system_default``.
        """
        store = _get_store(request)
        masked = store.get_masked(workspace)
        llm_active = masked["llm"]["active"]
        vision_active = masked["vision"]["active"]
        rag = await workspace_manager.get_instance(workspace)
        roles = rag.get_llm_role_config()

        out: dict[str, dict] = {}
        for role_name, cfg in roles.items():
            active = vision_active if role_name == VISION_LLM_ROLE else llm_active
            out[role_name] = {
                "binding": cfg.get("binding"),
                "model": cfg.get("model"),
                "host": cfg.get("host"),
                "is_cross_provider": cfg.get("is_cross_provider", False),
                "source": "custom" if active else "system_default",
            }
        return {"roles": out}

    @router.put(
        "/provider-config",
        dependencies=[Depends(combined_auth)],
    )
    async def update_provider_config(
        request: Request,
        body: UpdateProviderConfigRequest,
        test: bool = False,
        workspace: str = Depends(get_current_workspace),
        user: UserInfo = Depends(get_current_user),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Set / update provider credentials, then apply live (no restart)."""
        store = _get_store(request)
        if not store.has_secret():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is missing WORKSPACE_PROVIDER_SECRET; cannot store "
                "provider credentials securely.",
            )
        if body.llm is None and body.vision is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one of 'llm' or 'vision'",
            )

        existing = store.get(workspace) or WorkspaceProviderConfig()
        if body.llm is not None:
            existing.llm = _merge_slot(existing.llm, body.llm)
        if body.vision is not None:
            existing.vision = _merge_slot(existing.vision, body.vision)

        if test:
            if body.llm is not None:
                await _probe_provider(existing.llm)
            if body.vision is not None:
                await _probe_provider(existing.vision)

        existing.updated_at = datetime.now(timezone.utc).isoformat()
        existing.updated_by = user.username
        try:
            store.set(workspace, existing)
        except WorkspaceProviderError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            )

        # Apply with no restart: drop the cached instance so the next request
        # rebuilds it and re-applies the override via the on_instance_ready hook.
        await workspace_manager.invalidate(workspace)
        logger.info(
            f"Provider config updated for workspace '{workspace}' by "
            f"'{user.username}'"
        )
        return store.get_masked(workspace)

    @router.delete(
        "/provider-config",
        dependencies=[Depends(combined_auth)],
    )
    async def delete_provider_config(
        request: Request,
        slot: str = "all",
        workspace: str = Depends(get_current_workspace),
        user: UserInfo = Depends(get_current_user),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Clear a provider override (``slot`` = all | llm | vision).

        After clearing, the workspace falls back to the system default.
        """
        if slot not in ("all", "llm", "vision"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="slot must be 'all', 'llm' or 'vision'",
            )
        store = _get_store(request)
        removed = store.delete(workspace, which=slot)
        # Rebuild on next request regardless, so a previously-applied override on
        # the live instance is reset to the system default.
        await workspace_manager.invalidate(workspace)
        logger.info(
            f"Provider config slot '{slot}' cleared for workspace '{workspace}' "
            f"by '{user.username}' (removed={removed})"
        )
        return store.get_masked(workspace)

    return router
