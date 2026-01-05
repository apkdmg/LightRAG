"""
OpenAI-Compatible API Router for LightRAG.

This module provides OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`)
with full multi-tenancy support, enabling users to connect LightRAG to any
OpenAI-compatible client (Continue.dev, Cursor, LibreChat, etc.).

Supports:
- Streaming and non-streaming chat completions
- Query mode prefixes (/local, /global, /hybrid, etc.)
- Multi-tenancy via workspace resolution
- All authentication methods (JWT, API Key, OAuth2, per-user API keys)
"""

import json
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lightrag import LightRAG, QueryParam
from lightrag.utils import logger, TiktokenTokenizer
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.dependencies import resolve_workspace_from_request
from .ollama_api import parse_query_mode, SearchMode


def _is_raganything_instance(rag_instance) -> bool:
    """Check if the RAG instance is a RAGAnything instance."""
    return type(rag_instance).__name__ == "RAGAnything"


async def _call_aquery(rag_instance, query: str, param: "QueryParam"):
    """
    Call aquery on the RAG instance with proper parameter handling.

    Handles incompatibility between LightRAG and RAGAnything's aquery signatures.
    RAGAnything has a bug where passing param=QueryParam causes issues when VLM enhanced
    mode is active. Workaround: unpack QueryParam fields into keyword arguments.
    """
    if _is_raganything_instance(rag_instance):
        from dataclasses import asdict

        param_dict = asdict(param)
        kwargs = {k: v for k, v in param_dict.items() if v is not None}
        mode = kwargs.pop("mode", "mix")
        # Remove fields that RAGAnything sets internally in aquery_vlm_enhanced
        kwargs.pop("only_need_prompt", None)
        kwargs.pop("only_need_context", None)
        return await rag_instance.aquery(query, mode=mode, **kwargs)
    else:
        return await rag_instance.aquery(query, param=param)


# Pydantic models for OpenAI API
class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # LightRAG-specific extensions (optional)
    top_k: Optional[int] = None


class ChatChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int
    message: Optional[Dict[str, str]] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    """OpenAI-compatible model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "lightrag"


class ModelsResponse(BaseModel):
    """OpenAI-compatible models list response."""

    object: str = "list"
    data: List[ModelInfo]


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text using tiktoken."""
    tokens = TiktokenTokenizer().encode(text)
    return len(tokens)


class OpenAIAPI:
    """
    OpenAI-compatible API handler with multi-tenancy support.

    This class provides endpoints that match the OpenAI API specification,
    allowing LightRAG to be used with any OpenAI-compatible client.
    """

    def __init__(
        self,
        rag: LightRAG,
        top_k: int = 60,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI API handler.

        Args:
            rag: Default LightRAG instance (for single-instance mode fallback)
            top_k: Default number of top results to retrieve
            api_key: Optional API key for authentication
        """
        self._default_rag = rag
        self.top_k = top_k
        self.api_key = api_key
        self.router = APIRouter(tags=["openai"])
        self.setup_routes()

    async def get_rag_for_request(self, request: Request) -> LightRAG:
        """
        Get RAG instance - multi-tenant or single instance.

        In multi-tenant mode, resolves the workspace from the request
        and returns the appropriate RAG instance for that workspace.
        In single-instance mode, returns the default RAG instance.

        Args:
            request: The FastAPI request object

        Returns:
            LightRAG instance for the request's workspace
        """
        workspace_manager = getattr(request.app.state, "workspace_manager", None)

        if workspace_manager is not None:
            # Multi-tenant mode
            workspace = await resolve_workspace_from_request(request)
            return await workspace_manager.get_instance(workspace)
        else:
            # Single-instance mode
            return self._default_rag

    def setup_routes(self):
        """Setup OpenAI-compatible API routes."""

        # Create combined auth dependency
        combined_auth = get_combined_auth_dependency(self.api_key)

        @self.router.get("/v1/models", dependencies=[Depends(combined_auth)])
        async def list_models():
            """
            List available models (OpenAI format).

            Returns a list of model IDs that can be used with chat completions.
            The model ID can be used to specify query mode:
            - lightrag: Default (hybrid mode)
            - lightrag-local: Local search mode
            - lightrag-global: Global search mode
            - lightrag-hybrid: Hybrid search mode
            - lightrag-naive: Naive search mode
            - lightrag-mix: Mix search mode
            """
            current_time = int(time.time())
            return ModelsResponse(
                data=[
                    ModelInfo(id="lightrag", created=current_time),
                    ModelInfo(id="lightrag-local", created=current_time),
                    ModelInfo(id="lightrag-global", created=current_time),
                    ModelInfo(id="lightrag-hybrid", created=current_time),
                    ModelInfo(id="lightrag-naive", created=current_time),
                    ModelInfo(id="lightrag-mix", created=current_time),
                ]
            )

        @self.router.post("/v1/chat/completions", dependencies=[Depends(combined_auth)])
        async def chat_completions(request: Request):
            """
            Create a chat completion (OpenAI format).

            Supports both streaming and non-streaming responses.
            Query mode can be specified via:
            1. Model name (e.g., lightrag-local, lightrag-global)
            2. Message prefix (e.g., /local What is RAG?)

            Message prefixes take precedence over model name.
            """
            try:
                # Parse request body
                body = await request.json()
                chat_request = ChatCompletionRequest(**body)

                # Get RAG instance for this request (multi-tenant aware)
                rag = await self.get_rag_for_request(request)

                # Get messages
                messages = chat_request.messages
                if not messages:
                    raise HTTPException(status_code=400, detail="No messages provided")

                # Get the last user message as query
                query = messages[-1].content
                conversation_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages[:-1]
                ]

                # Parse query mode from message prefix
                cleaned_query, mode, only_need_context, user_prompt = parse_query_mode(
                    query
                )

                # If no prefix, check model name for mode
                if mode == SearchMode.mix and cleaned_query == query:
                    # No prefix was found, check model name
                    model_lower = chat_request.model.lower()
                    if model_lower.endswith("-local"):
                        mode = SearchMode.local
                    elif model_lower.endswith("-global"):
                        mode = SearchMode.global_
                    elif model_lower.endswith("-hybrid"):
                        mode = SearchMode.hybrid
                    elif model_lower.endswith("-naive"):
                        mode = SearchMode.naive
                    elif model_lower.endswith("-mix"):
                        mode = SearchMode.mix

                # Build query parameters
                param_dict = {
                    "mode": mode.value if mode != SearchMode.global_ else "global",
                    "stream": chat_request.stream,
                    "only_need_context": only_need_context,
                    "conversation_history": conversation_history,
                    "top_k": chat_request.top_k or self.top_k,
                }

                # Add user_prompt if specified
                if user_prompt is not None:
                    param_dict["user_prompt"] = user_prompt

                # Check for history_turns setting
                if hasattr(rag, "args") and rag.args.history_turns is not None:
                    param_dict["history_turns"] = rag.args.history_turns

                query_param = QueryParam(**param_dict)

                if chat_request.stream:
                    return StreamingResponse(
                        self._stream_response(
                            rag,
                            cleaned_query,
                            query_param,
                            chat_request.model,
                            mode,
                            conversation_history,
                        ),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )
                else:
                    return await self._non_streaming_response(
                        rag,
                        cleaned_query,
                        query_param,
                        chat_request.model,
                        mode,
                        conversation_history,
                    )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"OpenAI chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _stream_response(
        self,
        rag: LightRAG,
        query: str,
        param: QueryParam,
        model: str,
        mode: SearchMode,
        conversation_history: List[Dict[str, str]],
    ):
        """
        Stream response in OpenAI SSE format.

        Args:
            rag: LightRAG instance
            query: The cleaned query string
            param: Query parameters
            model: Model name for response
            mode: Search mode
            conversation_history: Previous messages
        """
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        try:
            # Handle bypass mode (direct LLM call)
            if mode == SearchMode.bypass:
                response = await rag.llm_model_func(
                    query,
                    stream=True,
                    history_messages=conversation_history,
                    **rag.llm_model_kwargs,
                )
            else:
                response = await _call_aquery(rag, query, param)

            # Handle string response (non-streaming from aquery)
            if isinstance(response, str):
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": response},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Stream async generator response
                try:
                    first_chunk = True
                    async for text in response:
                        if text:
                            delta = {"content": text}
                            if first_chunk:
                                delta["role"] = "assistant"
                                first_chunk = False

                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                except (asyncio.CancelledError, Exception) as e:
                    error_msg = str(e)
                    if isinstance(e, asyncio.CancelledError):
                        error_msg = "Stream was cancelled"
                    logger.error(f"Stream error: {error_msg}")

                    # Send error in stream
                    error_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"\n\nError: {error_msg}"},
                                "finish_reason": "error",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Send final chunk with finish_reason
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\nError: {str(e)}"},
                        "finish_reason": "error",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    async def _non_streaming_response(
        self,
        rag: LightRAG,
        query: str,
        param: QueryParam,
        model: str,
        mode: SearchMode,
        conversation_history: List[Dict[str, str]],
    ) -> dict:
        """
        Generate non-streaming response in OpenAI format.

        Args:
            rag: LightRAG instance
            query: The cleaned query string
            param: Query parameters
            model: Model name for response
            mode: Search mode
            conversation_history: Previous messages

        Returns:
            OpenAI-compatible chat completion response
        """
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        # Handle bypass mode (direct LLM call)
        if mode == SearchMode.bypass:
            response_text = await rag.llm_model_func(
                query,
                stream=False,
                history_messages=conversation_history,
                **rag.llm_model_kwargs,
            )
        else:
            response_text = await _call_aquery(rag, query, param)

        if not response_text:
            response_text = "No response generated"

        # Estimate token usage
        prompt_tokens = estimate_tokens(query)
        completion_tokens = estimate_tokens(str(response_text))

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response_text),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def create_openai_routes(
    rag: LightRAG,
    top_k: int = 60,
    api_key: Optional[str] = None,
) -> APIRouter:
    """
    Create OpenAI-compatible API router.

    Args:
        rag: Default LightRAG instance (for single-instance mode)
        top_k: Default number of top results
        api_key: Optional API key for authentication

    Returns:
        FastAPI router with OpenAI-compatible endpoints
    """
    openai_api = OpenAIAPI(rag=rag, top_k=top_k, api_key=api_key)
    return openai_api.router
