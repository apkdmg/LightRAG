# OpenAI-Compatible API

A drop-in OpenAI Chat Completions API surface for LightRAG — an **enterprise-fork feature** that lets any OpenAI-compatible client query your knowledge graph through the familiar `/v1/chat/completions` interface.

## Overview

The OpenAI-Compatible API exposes LightRAG's retrieval-augmented generation through endpoints that mirror the OpenAI API specification:

- `POST /v1/chat/completions` — chat completions (streaming and non-streaming), routed through LightRAG retrieval.
- `GET /v1/models` — lists the synthetic model IDs that map to LightRAG query modes.

Because the request/response shapes match OpenAI's, you can point existing tools at LightRAG with **no client code changes** — just change the base URL and API key. This makes it a drop-in backend for:

- The **OpenAI Python/Node SDKs** (set `base_url` to `http://<host>:9621/v1`).
- **Open WebUI** as an "OpenAI API" connection.
- **Continue.dev**, **Cursor**, **LibreChat**, and similar IDE/chat clients.

The router is implemented in [`lightrag/api/routers/openai_api.py`](../lightrag/api/routers/openai_api.py) and is registered unconditionally at server startup (no feature flag required) in `lightrag/api/lightrag_server.py`:

```python
app.include_router(create_openai_routes(rag, top_k=args.top_k, api_key=api_key))
```

It is **multi-tenancy aware**: when the server runs with a workspace manager, each request resolves to the caller's workspace via `resolve_workspace_from_request`, so a per-user credential automatically reads/writes only that user's data. In single-instance mode it falls back to the default RAG instance.

## Endpoints

All endpoints are mounted at the server root (no `/api` prefix). The default server port is `9621`.

| Method | Path | Auth required | Description |
|---|---|---|---|
| `GET` | `/v1/models` | Yes | List available model IDs (each maps to a query mode). |
| `POST` | `/v1/chat/completions` | Yes | Create a chat completion. Streaming and non-streaming. |

### Authentication

> **`/v1/*` endpoints REQUIRE authentication.** Unlike the Ollama-compatible
> `/api/*` endpoints — which are in the default `WHITELIST_PATHS`
> (`/health,/api/*`) and are therefore **public by default** — the OpenAI
> routes are **not whitelisted**. Every `/v1/*` request passes through the
> `combined_auth` dependency (`get_combined_auth_dependency` in
> [`lightrag/api/utils_api.py`](../lightrag/api/utils_api.py)).

The `combined_auth` dependency accepts any of the following credentials, evaluated in this order:

1. **Per-user API key** — `Authorization: Bearer sk-lightrag-…`. Checked first (these are not JWTs). The workspace is encoded in the key, so **no `X-Target-Workspace` header is needed**. This is the recommended credential for OpenAI clients.
2. **JWT** — `Authorization: Bearer <jwt>` (LightRAG-issued or Keycloak user/service-account token). A cookie token `lightrag_token` is also accepted as a fallback.
3. **Shared API key** — `X-API-Key: <LIGHTRAG_API_KEY>`. The caller is treated as an admin service account and **must** also send `X-Target-Workspace` (gated by the OBO allowlist) to target a workspace.

If **no** auth is configured on the server at all (no accounts, no API key), requests are accepted without credentials. As soon as either an API key or login accounts are configured, `/v1/*` requires a valid credential or returns `401`/`403`.

> **Note:** most OpenAI SDKs and chat clients send credentials via the
> `Authorization: Bearer …` header. This works directly with per-user API keys
> and JWTs. Clients that can only send a shared key as a plain "API key" will
> send `Authorization: Bearer <key>`; if you must use the *shared* server key,
> prefer a client that lets you set the `X-API-Key` header explicitly, or
> issue a per-user API key instead.

## Request & response schema

The Pydantic models live in `lightrag/api/routers/openai_api.py`.

### `ChatCompletionRequest`

```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_k: Optional[int] = None
```

| Field | Type | Default | Honored? | Notes |
|---|---|---|---|---|
| `model` | `string` | — (required) | **Yes** — for mode selection | Selects the query mode (see below). Not an LLM model name; the actual LLM is whatever the server is configured with. |
| `messages` | `array<ChatMessage>` | — (required) | **Yes** | Must be non-empty. The **last message** is used as the query; all earlier messages become `conversation_history`. An empty array returns **400**. |
| `temperature` | `float` | `0.7` | **Ignored** | Accepted for OpenAI-client compatibility but **not** passed to the LLM. The server uses its own configured LLM kwargs. |
| `max_tokens` | `int` | `null` | **Ignored** | Accepted but not forwarded to the LLM. |
| `stream` | `bool` | `false` | **Yes** | When `true`, the response is Server-Sent Events. |
| `top_k` | `int` | `null` | **Yes** — LightRAG extension | **Non-standard extension.** Number of top results to retrieve. Falls back to the server's configured `top_k` (default `60`) when omitted. |

`ChatMessage` has two fields, both required: `role` (`"system"` / `"user"` / `"assistant"`) and `content` (`string`).

**Other standard OpenAI fields** (`n`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `tools`, `response_format`, `seed`, etc.) are **not declared on the model and are silently ignored** — Pydantic drops unknown keys, so sending them does not error.

### Model names and query-mode selection

`GET /v1/models` returns six synthetic model IDs (all `owned_by: "lightrag"`):

| Model ID | Query mode |
|---|---|
| `lightrag` | `mix` (default) |
| `lightrag-local` | `local` |
| `lightrag-global` | `global` |
| `lightrag-hybrid` | `hybrid` |
| `lightrag-naive` | `naive` |
| `lightrag-mix` | `mix` |

The query mode is resolved in `chat_completions` as follows:

1. **Message prefix takes precedence.** `parse_query_mode` (from `ollama_api.py`) inspects the last message. A leading prefix selects the mode and is stripped from the query:
   - `/local …`, `/global …`, `/hybrid …`, `/naive …`, `/mix …`, `/bypass …`
   - `/context`, `/localcontext`, `/globalcontext`, `/hybridcontext`, `/naivecontext`, `/mixcontext` — same modes but set `only_need_context` (return retrieved context, no LLM generation).
   - A bracket form `/local[user prompt here] question …` additionally passes a per-query `user_prompt`.
2. **Model-name suffix is the fallback.** Only when no prefix was found, the `model` string's suffix is matched: `-local`, `-global`, `-hybrid`, `-naive`, `-mix`.
3. **Default.** If neither yields a mode, `mix` is used.

`/bypass` is special: it skips retrieval entirely and calls the LLM directly with the conversation history. Any model name not matching a known suffix simply falls through to `mix` — **unknown model names do not error.**

### `ChatCompletionResponse` (non-streaming)

```python
class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
```

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | Generated as `chatcmpl-<24 hex chars>`. |
| `object` | `string` | Always `"chat.completion"`. |
| `created` | `int` | Unix timestamp (seconds). |
| `model` | `string` | Echoes the `model` from the request. |
| `choices` | `array` | Always a single choice (`index: 0`). See `ChatChoice` below. |
| `usage` | `object` | `prompt_tokens`, `completion_tokens`, `total_tokens` — **estimated** (see [Token usage](#token-usage)). |

### `ChatChoice`

```python
class ChatChoice(BaseModel):
    index: int
    message: Optional[Dict[str, str]] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None
```

| Field | Used in | Notes |
|---|---|---|
| `index` | both | Always `0` (single choice). |
| `message` | non-streaming | `{"role": "assistant", "content": "…"}`. |
| `delta` | streaming | Incremental token payload (see [Streaming](#streaming)). |
| `finish_reason` | both | `"stop"` on success; `"error"` if the stream failed mid-flight; `null` on intermediate streamed chunks. |

> `ChatChoice` is defined for documentation/clarity; the handler builds the
> `choices` array as plain dicts, so both `message` and `delta` shapes are
> produced depending on the request.

A non-streaming response example:

```json
{
  "id": "chatcmpl-1a2b3c4d5e6f7g8h9i0j1k2l",
  "object": "chat.completion",
  "created": 1747900000,
  "model": "lightrag-hybrid",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "LightRAG combines …" },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 7, "completion_tokens": 142, "total_tokens": 149 }
}
```

When the underlying query yields no text, the content is replaced with the literal string `"No response generated"`.

## Streaming

When `stream: true`, the server returns `Content-Type: text/event-stream` with the headers `Cache-Control: no-cache`, `Connection: keep-alive`, and `X-Accel-Buffering: no` (the last disables proxy buffering so chunks arrive promptly).

The stream is standard OpenAI SSE: each line begins with `data: `, carries one JSON object with `object: "chat.completion.chunk"`, and the stream ends with the sentinel `data: [DONE]`.

- The **first** content chunk's `delta` includes `"role": "assistant"`; subsequent chunks carry only `"content"`.
- The **final** chunk has an empty `delta` (`{}`) and `finish_reason: "stop"`.
- If an error occurs mid-stream, an extra chunk is emitted with `delta.content` containing `"\n\nError: …"` and `finish_reason: "error"`, followed immediately by `data: [DONE]`.

Example streamed chunks:

```
data: {"id":"chatcmpl-1a2b3c4d5e6f7g8h9i0j1k2l","object":"chat.completion.chunk","created":1747900000,"model":"lightrag","choices":[{"index":0,"delta":{"role":"assistant","content":"Light"},"finish_reason":null}]}

data: {"id":"chatcmpl-1a2b3c4d5e6f7g8h9i0j1k2l","object":"chat.completion.chunk","created":1747900000,"model":"lightrag","choices":[{"index":0,"delta":{"content":"RAG combines …"},"finish_reason":null}]}

data: {"id":"chatcmpl-1a2b3c4d5e6f7g8h9i0j1k2l","object":"chat.completion.chunk","created":1747900000,"model":"lightrag","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

> **Note:** streamed chunks do **not** include a `usage` object. Token counts
> are only returned on non-streaming responses.

## Usage examples

In all examples, replace `sk-lightrag-…` with your per-user API key and `localhost:9621` with your server address.

### curl

Non-streaming:

```bash
curl -X POST http://localhost:9621/v1/chat/completions \
  -H "Authorization: Bearer sk-lightrag-7f3a9c2e-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lightrag-hybrid",
    "messages": [
      {"role": "user", "content": "What is LightRAG?"}
    ]
  }'
```

Streaming:

```bash
curl -N -X POST http://localhost:9621/v1/chat/completions \
  -H "Authorization: Bearer sk-lightrag-7f3a9c2e-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lightrag",
    "stream": true,
    "messages": [
      {"role": "user", "content": "/local Summarize the onboarding docs"}
    ]
  }'
```

List models:

```bash
curl http://localhost:9621/v1/models \
  -H "Authorization: Bearer sk-lightrag-7f3a9c2e-xxxxxxxx"
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9621/v1",
    api_key="sk-lightrag-7f3a9c2e-xxxxxxxx",  # your per-user API key
)

# Non-streaming
resp = client.chat.completions.create(
    model="lightrag-hybrid",
    messages=[{"role": "user", "content": "What is LightRAG?"}],
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="lightrag",
    messages=[{"role": "user", "content": "Explain the knowledge graph"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

The SDK sends the `api_key` as `Authorization: Bearer …`, which matches the per-user API key path in `combined_auth`. `temperature` and `max_tokens` may be passed for SDK compatibility but are ignored server-side.

### Open WebUI

1. In Open WebUI, open **Settings → Connections → OpenAI API**.
2. Add a new connection:
   - **API Base URL:** `http://<lightrag-host>:9621/v1`
   - **API Key:** your per-user API key (`sk-lightrag-…`).
3. Save. The LightRAG models (`lightrag`, `lightrag-local`, `lightrag-hybrid`, …) appear in the model picker — each one selects the corresponding retrieval mode.
4. To change retrieval mode per message without switching models, prefix the message, e.g. `/global What are the company-wide policies?`.

### Continue.dev

Add LightRAG as an OpenAI-compatible provider in `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "LightRAG (hybrid)",
      "provider": "openai",
      "model": "lightrag-hybrid",
      "apiBase": "http://localhost:9621/v1",
      "apiKey": "sk-lightrag-7f3a9c2e-xxxxxxxx"
    }
  ]
}
```

## Token usage

The `usage` object on non-streaming responses is **estimated**, not authoritative.

Counts come from `estimate_tokens(text)` in `openai_api.py`, which encodes the text with `TiktokenTokenizer` and returns the token count:

```python
def estimate_tokens(text: str) -> int:
    tokens = TiktokenTokenizer().encode(text)
    return len(tokens)
```

For non-streaming responses:

- `prompt_tokens` = `estimate_tokens(query)` — counts **only the last user message**, not the system prompt, conversation history, or the retrieved context that LightRAG actually feeds to the LLM.
- `completion_tokens` = `estimate_tokens(response_text)` — the generated answer.
- `total_tokens` = `prompt_tokens + completion_tokens`.

Because the estimate ignores history, the retrieval context, and prompt scaffolding — and because `tiktoken`'s tokenization may differ from your actual LLM's tokenizer — these numbers are a rough indication only. **Do not rely on them for billing-grade accuracy or quota enforcement.** Streaming responses omit `usage` entirely.

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `401 Unauthorized` (`"Invalid API key"`, `"Invalid token. Please login again."`, or `"No credentials provided"`) | Missing/expired/wrong credential. `/v1/*` is **not** whitelisted, so auth is always required when the server has auth configured. | Send a valid `Authorization: Bearer sk-lightrag-…` (per-user API key) or `Bearer <jwt>`. Re-mint the key/token if expired. |
| `403 Forbidden` (`"Invalid API Key"`, `"API Key required"`) | Used the shared `X-API-Key` and it is wrong or missing, or shared key was sent without `X-Target-Workspace`. | Verify `LIGHTRAG_API_KEY`; prefer a per-user API key which needs no extra headers. |
| Unknown model name behaves unexpectedly | A model string that matches no known suffix silently falls back to `mix` mode — it does **not** error. | Use one of the six IDs from `GET /v1/models`, or set the mode explicitly with a message prefix (`/local …`, `/global …`). |
| `400 Bad Request` — `"No messages provided"` | The `messages` array was empty. | Send at least one message; the last one is treated as the query. |
| `422 Unprocessable Entity` | Malformed body — e.g. missing `model`, `messages`, or a `ChatMessage` missing `role`/`content`. | Match the `ChatCompletionRequest` schema; `role` and `content` are both required on every message. |
| `500 Internal Server Error` | An unexpected failure in retrieval or the LLM call; the error message is returned in `detail`. | Check server logs (`OpenAI chat completion error: …`); verify the LLM/embedding bindings are reachable. |
| Streaming hangs or arrives all at once | A reverse proxy is buffering the SSE response. | The server sets `X-Accel-Buffering: no`; ensure your proxy honors it (e.g. nginx: `proxy_buffering off;`). Use `curl -N` to disable client buffering when testing. |
| Stream ends with `"Error: …"` and `finish_reason: "error"` | The query failed mid-stream (cancelled connection or backend error). | Retry; inspect server logs for the `Stream error:` entry. A cancelled client connection reports `"Stream was cancelled"`. |
| Empty answer / `"No response generated"` | Retrieval returned nothing for the query in the resolved workspace. | Confirm documents are ingested into the caller's workspace; try `/naive` or `/bypass` to isolate retrieval vs. LLM. |

## Related documentation

- [Integration Guide](./INTEGRATION_GUIDE.md) — section 5 covers the OpenAI-compatible API in the broader context of authentication, workspaces, and chat-client integration.
- [LightRAG API Server Guide](./LightRAG-API-Server.md) — full server configuration, startup options, and the complete endpoint reference.
- `PER_USER_API_KEYS.md` — minting and managing `sk-lightrag-…` per-user API keys (the recommended credential for OpenAI clients). See also Integration Guide §2.5 for the same workflow.
