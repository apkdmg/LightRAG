"""
Admin routes for the server default LLM / Vision-LLM provider.

The server default is what every workspace without its own provider uses. Admins
can set it per role group (see ``provider_routes.py`` for the groups) from the
WebUI, without access to the server's ``.env``, which stays as the fallback.

Changes apply with **no server restart** and without rebuilding any instance, so
running pipelines keep going. Saving also resumes ingestion that was paused
because the provider rejected the previous key.

This setting decides where document text is sent, so sign-ins only: API keys and
client-credentials tokens are refused, and every change is logged and emailed to
the admins.

Registered only when both multi-tenancy and ``ENABLE_WORKSPACE_PROVIDERS`` are
enabled. Fork-only and self-contained.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import SecretStr

from lightrag.api.dependencies import UserInfo, require_admin
from lightrag.api.provider_alerts import _ROLE_SLOTS, email_system_provider_change
from lightrag.api.routers.provider_routes import (
    ProviderSlotInput,
    UpdateProviderConfigRequest,
    _get_store,
    _merge_slot,
    _resume_paused_ingestion,
    _validate_base_url,
)
from lightrag.api.startup_resume import _candidate_workspaces
from lightrag.api.workspace_deletion import persisted_workspaces
from lightrag.api.workspace_providers import (
    SLOT_ROLES,
    SYSTEM_RECORD_ID,
    ProviderSlot,
    WorkspaceProviderConfig,
    WorkspaceProviderError,
    WorkspaceProviderStore,
    _mask_key,
    reconcile_loaded_instances,
    resolve_server_slot_key,
    slot_effective_view,
)
from lightrag.exceptions import PipelineNotInitializedError
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.pipeline import PROVIDER_PAUSE_KEY
from lightrag.usage import emit_event

logger = logging.getLogger("lightrag.api.system_provider_routes")

_TEST_TIMEOUT_SECONDS = 30.0
_TEST_MAX_TOKENS = 16
# An 8x8 white PNG, to check that a vision model accepts images.
_TEST_IMAGE = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFUlEQVR4"
    "nGP8//8/AzbAhFV00EoAAFbUAw037MyjAAAAAElFTkSuQmCC"
)

# Strong references so background emails are not garbage-collected mid-send.
_email_tasks: set[asyncio.Task] = set()


class ProviderTestRequest(ProviderSlotInput):
    """A slot as it stands in the form, to test before saving."""

    slot: Literal["extraction", "query", "vision"]


async def require_signed_in_admin(
    user: UserInfo = Depends(require_admin),
) -> UserInfo:
    """Admin role, and a person signed in rather than a service account."""
    if user.metadata.get("auth_mode") in ("api_key", "client_credentials"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The server default can only be changed by a signed-in admin",
        )
    return user


def _build_slot(
    name: str,
    incoming: ProviderSlotInput,
    config: WorkspaceProviderConfig,
    env_settings,
) -> ProviderSlot:
    """Merge a submitted slot, reusing the key in use if the provider is the same."""
    existing = config.slot(name)
    base_url = incoming.base_url or existing.base_url
    if not base_url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "base_url is required")
    _validate_base_url(base_url)
    key: Optional[SecretStr] = None
    if not incoming.api_key:
        key = resolve_server_slot_key(name, base_url, config, env_settings)
        if key is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"This provider needs its own API key: no key is in use for {base_url}",
            )
    return _merge_slot(existing.model_copy(update={"api_key": key}), incoming)


def _provider_error_message(resp: httpx.Response, api_key: str) -> str:
    """The provider's own error text, shortened, and never containing the key."""
    try:
        data = resp.json()
        error = data.get("error") if isinstance(data, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        message = str(message or data)
    except ValueError:
        message = resp.text
    return message.replace(api_key, "***")[:300]


async def _test_provider(slot: ProviderSlot, vision: bool) -> dict[str, Any]:
    """Send a very short real completion, to prove the key and the model work.

    Listing ``/models`` can't tell whether this model exists or accepts images.
    A failure the provider may recover from can still be applied anyway; one
    that says this configuration is wrong can't.
    """
    api_key = slot.api_key.get_secret_value()
    prompt = "Reply with OK."
    content: Any = (
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _TEST_IMAGE}},
        ]
        if vision
        else prompt
    )
    body: dict[str, Any] = {
        "model": slot.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _TEST_MAX_TOKENS,
    }
    if slot.reasoning_effort:
        body["reasoning_effort"] = slot.reasoning_effort
    url = slot.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    started = time.perf_counter()

    def result(
        outcome: str,
        message: str = "",
        status_code: Optional[int] = None,
        can_apply_anyway: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": outcome == "ok",
            "outcome": outcome,
            "status_code": status_code,
            "message": message,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "can_apply_anyway": can_apply_anyway,
        }

    try:
        async with httpx.AsyncClient(timeout=_TEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=body, headers=headers)
            # Newer OpenAI models refuse max_tokens by name.
            if resp.status_code == 400 and "max_tokens" in resp.text:
                body["max_completion_tokens"] = body.pop("max_tokens")
                resp = await client.post(url, json=body, headers=headers)
    except httpx.HTTPError as e:
        return result(
            "unreachable", str(e).replace(api_key, "***")[:300], can_apply_anyway=True
        )

    code = resp.status_code
    if code in (401, 403):
        return result("invalid_key", _provider_error_message(resp, api_key), code)
    if code in (402, 412):
        return result("spend_limit", _provider_error_message(resp, api_key), code)
    if code in (400, 404, 422):
        return result("model_rejected", _provider_error_message(resp, api_key), code)
    if code != 200:
        return result(
            "provider_error",
            _provider_error_message(resp, api_key),
            code,
            can_apply_anyway=True,
        )
    # A wrong base URL can answer 200 with a web page; some routers put errors
    # in a 200 response.
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict) or not data.get("choices"):
        return result("model_rejected", _provider_error_message(resp, api_key), code)
    return result("ok", status_code=code)


async def _paused_on_system_key(request: Request) -> list[dict[str, Any]]:
    """Workspaces whose ingestion is paused on a key the server default controls."""
    app_state = request.app.state
    store: WorkspaceProviderStore = app_state.workspace_provider_store
    try:
        workspaces = await _candidate_workspaces(
            app_state.default_rag, app_state.workspace_registry, True
        )
    except Exception as e:
        logger.warning(f"Could not list workspaces to check for pauses: {e}")
        return []
    own = store.workspaces_with_own_provider()
    paused: list[dict[str, Any]] = []
    for workspace in workspaces:
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )
        except PipelineNotInitializedError:
            continue
        pause = pipeline_status.get(PROVIDER_PAUSE_KEY)
        if not pause:
            continue
        slot = _ROLE_SLOTS.get(pause.get("role"))
        if slot is None or workspace in own[slot]:
            continue
        paused.append(
            {
                "workspace": workspace,
                "slot": slot,
                "kind": pause.get("kind"),
                "status_code": pause.get("status_code"),
                "since": pause.get("since"),
            }
        )
    return paused


async def _resume_paused(request: Request, slots: set[str]) -> int:
    """Resume ingestion paused on any of these slots. Returns how many workspaces."""
    app_state = request.app.state
    resumed = 0
    for pause in await _paused_on_system_key(request):
        if pause["slot"] not in slots:
            continue
        workspace = pause["workspace"]
        # The unnamed default workspace only exists on the default instance.
        await _resume_paused_ingestion(
            app_state.workspace_manager,
            workspace,
            rag=None if workspace else app_state.default_rag,
        )
        resumed += 1
    return resumed


async def _view(request: Request) -> dict[str, Any]:
    """Everything the server default page shows. Never includes a full key."""
    app_state = request.app.state
    store: WorkspaceProviderStore = app_state.workspace_provider_store
    env_settings = app_state.env_role_llm_settings
    masked = store.get_masked(SYSTEM_RECORD_ID)
    for name, roles in SLOT_ROLES.items():
        role = roles[0]
        slot = masked[name]
        env = env_settings(role)
        env_preview = _mask_key(env["api_key"] or "")
        slot["effective"] = {
            **slot_effective_view(
                app_state.default_rag.get_llm_role_config(role), slot["active"]
            ),
            "source": "webui" if slot["active"] else "env",
            "api_key_preview": slot["api_key_preview"]
            if slot["active"]
            else env_preview,
        }
        slot["env"] = {
            "binding": env["binding"],
            "host": env["host"],
            "model": env["model"],
            "api_key_preview": env_preview,
        }

    if not store.has_secret():
        record_status = "no_secret"
    elif store.is_unreadable(SYSTEM_RECORD_ID):
        record_status = "unreadable"
    else:
        record_status = "ok"

    _, total = app_state.workspace_manager.list_workspaces(
        page=1, page_size=1, persisted=await persisted_workspaces(request.app)
    )
    return {
        **masked,
        "status": record_status,
        "impact": {
            "total_workspaces": total,
            "own_provider": store.workspaces_with_own_provider(),
        },
        "paused": await _paused_on_system_key(request),
    }


def _in_use(request: Request, role: str) -> dict[str, Optional[str]]:
    config = request.app.state.default_rag.get_llm_role_config(role)
    return {"host": config.get("host"), "model": config.get("model")}


def _record_change(
    actor: str,
    before: dict[str, dict[str, Optional[str]]],
    request: Request,
) -> None:
    """Log and email what changed. Called after the change has been applied."""
    changes = {
        slot: {"old": old, "new": _in_use(request, SLOT_ROLES[slot][0])}
        for slot, old in before.items()
    }
    changes = {s: c for s, c in changes.items() if c["old"] != c["new"]}
    for slot, change in changes.items():
        emit_event(
            "system_provider_change",
            slot=slot,
            by=actor,
            old_model=change["old"]["model"],
            new_model=change["new"]["model"],
        )
        logger.info(
            f"Server default for '{slot}' changed by '{actor}': "
            f"{change['old']} -> {change['new']}"
        )
    if changes:
        task = asyncio.create_task(email_system_provider_change(actor, changes))
        _email_tasks.add(task)
        task.add_done_callback(_email_tasks.discard)


def create_system_provider_routes() -> APIRouter:
    router = APIRouter(prefix="/admin/system-provider", tags=["providers"])

    @router.get("")
    async def get_server_default(
        request: Request, _: UserInfo = Depends(require_signed_in_admin)
    ):
        """The server default, what's in use, and who a change would affect."""
        _get_store(request)
        return await _view(request)

    @router.post("/test")
    async def test_server_default(
        request: Request,
        body: ProviderTestRequest,
        _: UserInfo = Depends(require_signed_in_admin),
    ):
        """Try a slot as it stands in the form. Saves nothing."""
        store = _get_store(request)
        config = store.get(SYSTEM_RECORD_ID) or WorkspaceProviderConfig()
        slot = _build_slot(
            body.slot, body, config, request.app.state.env_role_llm_settings
        )
        return await _test_provider(slot, vision=body.slot == "vision")

    @router.put("")
    async def update_server_default(
        request: Request,
        body: UpdateProviderConfigRequest,
        user: UserInfo = Depends(require_signed_in_admin),
    ):
        """Set the server default for the given slots, and apply it now."""
        store = _get_store(request)
        if not store.has_secret():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is missing WORKSPACE_PROVIDER_SECRET; cannot store "
                "provider credentials securely.",
            )
        provided = {
            name: getattr(body, name)
            for name in SLOT_ROLES
            if getattr(body, name) is not None
        }
        if not provided:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provide at least one of {list(SLOT_ROLES)}",
            )

        app_state = request.app.state
        env_settings = app_state.env_role_llm_settings
        before = {name: _in_use(request, SLOT_ROLES[name][0]) for name in provided}
        config = store.get(SYSTEM_RECORD_ID) or WorkspaceProviderConfig()
        for name, incoming in provided.items():
            setattr(config, name, _build_slot(name, incoming, config, env_settings))
        config.updated_at = datetime.now(timezone.utc).isoformat()
        config.updated_by = user.username
        try:
            store.set(SYSTEM_RECORD_ID, config)
        except WorkspaceProviderError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            )

        reconcile_loaded_instances(
            app_state.default_rag, app_state.workspace_manager, store, env_settings
        )
        resumed = await _resume_paused(request, set(provided))
        _record_change(user.username, before, request)
        return {**await _view(request), "resumed": resumed}

    @router.delete("")
    async def clear_server_default(
        request: Request,
        slot: str = "all",
        user: UserInfo = Depends(require_signed_in_admin),
    ):
        """Return a slot (or ``all``) to the ``.env`` configuration."""
        if slot != "all" and slot not in SLOT_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"slot must be 'all' or one of {list(SLOT_ROLES)}",
            )
        store = _get_store(request)
        app_state = request.app.state
        slots = set(SLOT_ROLES) if slot == "all" else {slot}
        before = {name: _in_use(request, SLOT_ROLES[name][0]) for name in slots}
        store.delete(SYSTEM_RECORD_ID, which=slot)
        reconcile_loaded_instances(
            app_state.default_rag,
            app_state.workspace_manager,
            store,
            app_state.env_role_llm_settings,
        )
        resumed = await _resume_paused(request, slots)
        _record_change(user.username, before, request)
        return {**await _view(request), "resumed": resumed}

    return router
