"""
Tests for COT <think> tag handling in the OpenAI streaming path

Validates that a <think> block opened for reasoning_content deltas is always
closed, including when a provider sends the reasoning-to-answer transition in
a single delta carrying both reasoning_content and content (e.g. Gemini via
OpenAI-compatible routers). An unterminated <think> makes stream consumers
(like the WebUI) treat the entire answer as hidden reasoning.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.openai import openai_complete_if_cache


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


def _make_stream_chunk(content=None, reasoning_content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                )
            )
        ]
    )


def _make_fake_client(stream):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=stream),
            )
        ),
        close=AsyncMock(),
    )


async def _collect_stream(chunks) -> str:
    fake_client = _make_fake_client(_FakeAsyncStream(chunks))
    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        stream = await openai_complete_if_cache(
            model="test-model",
            prompt="question",
            stream=True,
            enable_cot=True,
        )
        parts = []
        async for part in stream:
            parts.append(part)
    return "".join(parts)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cot_closed_on_separate_content_delta():
    result = await _collect_stream(
        [
            _make_stream_chunk(reasoning_content="thinking..."),
            _make_stream_chunk(content="answer"),
        ]
    )
    assert result == "<think>thinking...</think>answer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cot_closed_when_transition_delta_has_both_fields():
    """Regression: reasoning_content and content in the same delta must not
    leave the <think> tag unterminated."""
    result = await _collect_stream(
        [
            _make_stream_chunk(reasoning_content="thinking..."),
            _make_stream_chunk(reasoning_content="tail", content="answer"),
            _make_stream_chunk(content=" continues"),
        ]
    )
    assert result.startswith("<think>")
    assert "</think>" in result
    assert result.split("</think>", 1)[1] == "answer continues"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_cot_tags_when_first_delta_has_both_fields():
    """Providers that duplicate reasoning into content from the first delta
    must not get <think> wrapping at all."""
    result = await _collect_stream(
        [
            _make_stream_chunk(reasoning_content="dup", content="answer"),
            _make_stream_chunk(content=" text"),
        ]
    )
    assert result == "answer text"
