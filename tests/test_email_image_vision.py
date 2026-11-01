"""Offline tests for the cost controls on email images.

Email images get the same controls as images in uploaded documents: size limits,
a cached description keyed on the image bytes, and the VLM_MAX_ASYNC_LLM limit.
They stay on the system key in every workspace. Runs against a LightRAG built with
dummy LLM/embedding and file-based storages, so no external services are needed.
"""

import asyncio
import base64
import importlib
import struct
import sys
from uuid import uuid4

import numpy as np
import pytest

# Neutralize pytest argv during the router import (auth handler parses args).
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_email_routes = importlib.import_module("lightrag.api.routers.email_routes")
sys.argv = _original_argv

EmailIngestionService = _email_routes.EmailIngestionService
ParsedAttachment = _email_routes.ParsedAttachment
ParsedEmail = _email_routes.ParsedEmail
_system_vision_cache_identity = _email_routes._system_vision_cache_identity
_system_vision_func = _email_routes._system_vision_func

from lightrag.base import DocStatus  # noqa: E402
from lightrag.lightrag import LightRAG  # noqa: E402
from lightrag.llm_roles import RoleLLMConfig  # noqa: E402
from lightrag.provider_rejection import (  # noqa: E402
    add_provider_rejection_listener,
    remove_provider_rejection_listener,
)
from lightrag.utils import EmbeddingFunc, Tokenizer  # noqa: E402

pytestmark = pytest.mark.offline

SYSTEM_IDENTITY = {"role": "vlm", "binding": "openai", "model": "m", "host": "h"}


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


class _CountingLLM:
    def __init__(self):
        self.calls = 0

    async def __call__(self, *args, **kwargs) -> str:
        self.calls += 1
        return "ok"


def _no_op_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


async def _build_rag(tmp_path, **overrides) -> LightRAG:
    overrides.setdefault("llm_model_func", _CountingLLM())
    rag = LightRAG(
        working_dir=str(tmp_path / uuid4().hex[:8]),
        workspace=f"emailvision_{uuid4().hex[:8]}",
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_no_op_chunking,
        max_parallel_insert=1,
        **overrides,
    )
    await rag.initialize_storages()
    return rag


@pytest.fixture
async def rag(tmp_path):
    instance = await _build_rag(tmp_path)
    yield instance
    await instance.finalize_storages()


class _Vision:
    """Answers differently on every call, as real vision models do."""

    def __init__(self, error: Exception | None = None, response: str | None = None):
        self.error = error
        self.response = response
        self.calls = 0

    async def __call__(self, prompt: str, image_data: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        return f"Description, attempt {self.calls}."


class _ProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _png(width: int = 200, height: int = 100, padding: int = 0) -> bytes:
    """Just enough of a PNG for its size to be read from the header."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\0" * padding
    )


def _image(content: bytes, filename: str = "image001.png") -> ParsedAttachment:
    return ParsedAttachment(
        filename=filename, content_type="image/png", content=content, is_inline=True
    )


def _service(rag, vision, identity=None) -> EmailIngestionService:
    return EmailIngestionService(
        rag, vision, vision_cache_identity=identity or SYSTEM_IDENTITY
    )


# --------------------------------------------------------------------------- #
# Size limits                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("size", [(1, 1), (31, 200), (200, 31)])
async def test_image_with_a_short_side_is_not_sent(rag, size):
    vision = _Vision()
    result = await _service(rag, vision)._describe_image(_image(_png(*size)))
    assert not result.ok
    assert result.note == "image too small"
    assert vision.calls == 0


async def test_image_at_the_minimum_size_is_sent(rag):
    vision = _Vision()
    result = await _service(rag, vision)._describe_image(_image(_png(32, 32)))
    assert result.ok
    assert vision.calls == 1


async def test_image_over_the_default_byte_limit_is_not_sent(rag):
    vision = _Vision()
    limit = 5 * 1024 * 1024
    service = _service(rag, vision)

    result = await service._describe_image(_image(_png(padding=limit)))
    assert not result.ok
    assert result.note == "image too large"
    assert vision.calls == 0

    at_limit = _png(padding=limit - len(_png()))
    assert (await service._describe_image(_image(at_limit))).ok
    assert vision.calls == 1


async def test_size_limits_follow_the_environment(rag, monkeypatch):
    vision = _Vision()
    service = _service(rag, vision)

    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", "64")
    result = await service._describe_image(_image(_png(50, 50)))
    assert result.note == "image too small"

    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", str(300 * 1024))
    result = await service._describe_image(_image(_png(padding=300 * 1024)))
    assert result.note == "image too large"
    assert vision.calls == 0


async def test_size_limits_keep_the_pipeline_floors(rag, monkeypatch):
    vision = _Vision()
    service = _service(rag, vision)
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", "1")
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", "0")

    assert (await service._describe_image(_image(_png(1, 1, padding=1024)))).ok
    result = await service._describe_image(_image(_png(padding=256 * 1024)))
    assert result.note == "image too large"


async def test_invalid_size_limits_fall_back_to_the_defaults(rag, monkeypatch):
    vision = _Vision()
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", "lots")
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", "")

    result = await _service(rag, vision)._describe_image(_image(_png(1, 1)))
    assert result.note == "image too small"


async def test_image_of_unknown_size_is_still_sent(rag):
    # As with document images: only a size we can read rules an image out.
    vision = _Vision()
    result = await _service(rag, vision)._describe_image(_image(b"not a raster"))
    assert result.ok
    assert vision.calls == 1


async def test_empty_image_is_not_sent(rag):
    vision = _Vision()
    result = await _service(rag, vision)._describe_image(_image(b""))
    assert result.status == "empty"
    assert vision.calls == 0


async def test_small_image_attachment_is_filtered_too(rag):
    vision = _Vision()
    attachment = ParsedAttachment(
        filename="pixel.gif",
        content_type="image/gif",
        content=b"GIF89a" + struct.pack("<HH", 1, 1),
    )
    result = await _service(rag, vision)._extract_attachment_content(attachment)
    assert result.note == "image too small"
    assert vision.calls == 0


# --------------------------------------------------------------------------- #
# Cached descriptions                                                         #
# --------------------------------------------------------------------------- #


async def test_same_image_is_described_once_whatever_its_filename(rag):
    vision = _Vision()
    service = _service(rag, vision)

    first = await service._describe_image(_image(_png(), "image001.png"))
    second = await service._describe_image(_image(_png(), "logo.png"))

    assert first.ok and second.ok
    assert second.text == first.text
    assert vision.calls == 1


async def test_other_image_or_model_is_described_afresh(rag):
    vision = _Vision()
    service = _service(rag, vision)
    await service._describe_image(_image(_png()))

    await service._describe_image(_image(_png(201, 100)))
    assert vision.calls == 2

    new_model = {**SYSTEM_IDENTITY, "model": "newer"}
    await _service(rag, vision, new_model)._describe_image(_image(_png()))
    assert vision.calls == 3


async def test_entry_is_an_analysis_entry_without_the_image(rag, monkeypatch):
    saved = {}
    upsert = rag.llm_response_cache.upsert

    async def spy(data):
        saved.update(data)
        await upsert(data)

    monkeypatch.setattr(rag.llm_response_cache, "upsert", spy)
    content = _png()
    result = await _service(rag, _Vision())._describe_image(_image(content))

    [(key, entry)] = saved.items()
    assert key.startswith("default:analysis:")
    assert entry["cache_type"] == "analysis"
    assert entry["return"] == result.text
    assert "image001.png" in entry["original_prompt"]
    assert base64.b64encode(content).decode() not in entry["original_prompt"]


async def test_cache_flag_off_neither_reads_nor_saves(tmp_path):
    rag = await _build_rag(tmp_path, enable_llm_cache_for_entity_extract=False)
    try:
        vision = _Vision()
        service = _service(rag, vision)
        await service._describe_image(_image(_png()))
        await service._describe_image(_image(_png()))
        assert vision.calls == 2
        assert await rag.llm_response_cache.is_empty()
    finally:
        await rag.finalize_storages()


# --------------------------------------------------------------------------- #
# Failure paths                                                               #
# --------------------------------------------------------------------------- #


async def test_failed_call_is_not_cached(rag):
    vision = _Vision(error=RuntimeError("provider down"))
    service = _service(rag, vision)

    result = await service._describe_image(_image(_png()))
    assert result.status == "failed"
    assert result.note == "vision error"
    assert await rag.llm_response_cache.is_empty()

    vision.error = None
    assert (await service._describe_image(_image(_png()))).ok
    assert vision.calls == 2


@pytest.mark.parametrize("response", ["", "   \n"])
async def test_empty_response_is_not_cached(rag, response):
    vision = _Vision(response=response)
    service = _service(rag, vision)

    result = await service._describe_image(_image(_png()))
    assert result.status == "empty"
    assert await rag.llm_response_cache.is_empty()

    vision.response = "A chart."
    assert (await service._describe_image(_image(_png()))).text == "A chart."


async def test_rejected_key_is_reported_as_the_system_key(rag):
    events = []

    async def listener(event):
        events.append(event)

    add_provider_rejection_listener(listener)
    try:
        vision = _Vision(error=_ProviderError(402))
        result = await _service(rag, vision)._describe_image(_image(_png()))
    finally:
        remove_provider_rejection_listener(listener)

    assert result.status == "failed"
    [event] = events
    assert event.workspace == rag.workspace
    assert event.role == "vlm"
    assert event.byo_key is False
    assert event.rejection.kind == "budget_exhausted"


async def test_cached_description_survives_a_provider_outage(rag):
    events = []

    async def listener(event):
        events.append(event)

    vision = _Vision()
    service = _service(rag, vision)
    first = await service._describe_image(_image(_png()))

    vision.error = _ProviderError(401)
    add_provider_rejection_listener(listener)
    try:
        second = await service._describe_image(_image(_png()))
    finally:
        remove_provider_rejection_listener(listener)

    assert second.ok and second.text == first.text
    assert vision.calls == 1
    assert events == []


async def test_failed_cache_write_keeps_the_paid_for_description(rag, monkeypatch):
    async def broken_upsert(data):
        raise RuntimeError("storage down")

    monkeypatch.setattr(rag.llm_response_cache, "upsert", broken_upsert)
    vision = _Vision()

    result = await _service(rag, vision)._describe_image(_image(_png()))

    assert result.ok
    assert vision.calls == 1


async def test_failed_cache_read_falls_back_to_the_model(rag, monkeypatch):
    async def broken_get(key):
        raise RuntimeError("storage down")

    monkeypatch.setattr(rag.llm_response_cache, "get_by_id", broken_get)
    vision = _Vision()

    result = await _service(rag, vision)._describe_image(_image(_png()))

    assert result.ok
    assert vision.calls == 1


# --------------------------------------------------------------------------- #
# Re-sync, end to end                                                         #
# --------------------------------------------------------------------------- #


def _email() -> ParsedEmail:
    return ParsedEmail(
        message_id="<resync-images@example.com>",
        from_address="sender@example.com",
        to_addresses=["r@example.com"],
        cc_addresses=[],
        subject="Quarterly charts",
        date=None,
        body_text="See the charts below.",
        body_html=None,
        inline_images=[
            _image(_png(1, 1), "tracking.png"),
            _image(_png(400, 300), "chart.png"),
            # The same picture twice in one email is one description.
            _image(_png(400, 300), "chart-again.png"),
        ],
        attachments=[
            ParsedAttachment(
                filename="photo.png",
                content_type="image/png",
                content=_png(800, 600),
            )
        ],
    )


async def _bundle_contents(rag) -> dict[str, str]:
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    return {
        st.file_path: (await rag.full_docs.get_by_id(doc_id))["content"]
        for doc_id, st in docs.items()
        if str(st.file_path).startswith("email_")
    }


async def test_resync_pays_for_no_image_or_extraction_again(tmp_path):
    llm = _CountingLLM()
    rag = await _build_rag(tmp_path, llm_model_func=llm)
    try:
        vision = _Vision()
        service = _service(rag, vision)

        first = await service.ingest_email(_email(), track_id="first")
        contents = await _bundle_contents(rag)
        calls_after_first = llm.calls

        assert vision.calls == 2  # chart + photo; the pixel never reaches it
        assert llm.calls > 0
        statuses = [c["status"] for c in first["components"]]
        assert statuses == ["ok", "skipped", "ok", "ok", "ok"]
        assert not any("tracking" in path for path in contents)

        second = await service.ingest_email(_email(), track_id="second")

        assert second["deferred"] is False
        assert vision.calls == 2
        # The vision model would answer differently now, yet the text is the
        # same, so entity extraction finds its own cache entries as well.
        assert await _bundle_contents(rag) == contents
        assert llm.calls == calls_after_first
    finally:
        await rag.finalize_storages()


# --------------------------------------------------------------------------- #
# The system vlm role                                                         #
# --------------------------------------------------------------------------- #


class _RoleFunc:
    """A raw role LLM function that records how it is called."""

    def __init__(self):
        self.kwargs: list[dict] = []
        self.running = 0
        self.peak = 0

    async def __call__(self, prompt: str, **kwargs) -> str:
        self.kwargs.append(kwargs)
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        return f"Described: {len(self.kwargs)}"


def _vlm_config(func: _RoleFunc, max_async: int = 4) -> dict[str, RoleLLMConfig]:
    return {
        "vlm": RoleLLMConfig(
            func=func,
            max_async=max_async,
            metadata={"binding": "openai", "model": "sys-vlm", "host": "https://sys"},
        )
    }


async def test_system_vision_func_calls_the_role_like_the_pipeline(tmp_path):
    role_func = _RoleFunc()
    system = await _build_rag(tmp_path, role_llm_configs=_vlm_config(role_func))
    try:
        answer = await _system_vision_func(system)("Describe", image_data="QUJD")
    finally:
        await system.finalize_storages()

    assert answer == "Described: 1"
    [kwargs] = role_func.kwargs
    assert kwargs["image_inputs"] == ["QUJD"]
    assert kwargs["stream"] is False
    assert "_priority" not in kwargs  # consumed by the queue


async def test_vlm_concurrency_limit_holds_across_workspaces(tmp_path):
    role_func = _RoleFunc()
    system = await _build_rag(
        tmp_path, role_llm_configs=_vlm_config(role_func, max_async=2)
    )
    first, second = await _build_rag(tmp_path), await _build_rag(tmp_path)
    try:
        vision = _system_vision_func(system)
        services = [_service(first, vision), _service(second, vision)]
        results = await asyncio.gather(
            *(
                services[i % 2]._describe_image(_image(_png(100 + i, 100)))
                for i in range(6)
            )
        )
    finally:
        await second.finalize_storages()
        await first.finalize_storages()
        await system.finalize_storages()

    assert all(result.ok for result in results)
    assert len(role_func.kwargs) == 6
    assert role_func.peak == 2


async def test_byo_vision_key_does_not_pay_for_email_images(tmp_path):
    system_func, byo_func = _RoleFunc(), _RoleFunc()
    system = await _build_rag(tmp_path, role_llm_configs=_vlm_config(system_func))
    workspace = await _build_rag(tmp_path, role_llm_configs=_vlm_config(_RoleFunc()))
    try:
        await workspace.aupdate_llm_role_config(
            "vlm", model_func=byo_func, model="byo-vlm", host="https://byo"
        )
        identity = _system_vision_cache_identity(system)
        service = EmailIngestionService(
            workspace, _system_vision_func(system), vision_cache_identity=identity
        )

        result = await service._describe_image(_image(_png()))

        assert result.ok
        assert len(system_func.kwargs) == 1
        assert byo_func.kwargs == []
        assert identity == {
            "role": "vlm",
            "binding": "openai",
            "model": "sys-vlm",
            "host": "https://sys",
        }
        # The description belongs to the workspace the email went into.
        assert not await workspace.llm_response_cache.is_empty()
        assert await system.llm_response_cache.is_empty()
    finally:
        await workspace.finalize_storages()
        await system.finalize_storages()
