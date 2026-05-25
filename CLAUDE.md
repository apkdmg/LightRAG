# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Canonical guidelines

**`AGENTS.md` is the source of truth** for repository structure, module layout, the `LightRAG` mixin composition, the pipeline concurrency contract, query modes, dev commands, and code style. Read it before non-trivial changes. `.clinerules/01-basic.md` contains additional implementation pattern notes (embedding format compatibility, async-generator lock rules, etc.).

This file only surfaces the highest-leverage orientation a fresh session needs immediately.

## Repository at a glance

- This is `lightrag-enterprise`, a **fork** of `HKUDS/LightRAG`. Open PRs against this fork's `main`, never upstream.
- Python package `lightrag/` (orchestrator + mixins, storage backends under `kg/`, LLM bindings under `llm/`, FastAPI service under `api/`).
- React 19 + TypeScript WebUI under `lightrag_webui/` — **Bun only**, not npm/yarn.
- Setup wizard generates `docker-compose.final.yml` from `scripts/setup/templates/*.yml`; drive it via `make env-*`, not direct `setup.sh` calls.

## Commands you'll reach for first

```bash
# Backend dev environment + frontend build in one shot
make dev

# Backend tests (resolves PYTHON/venv/uv automatically — prefer over raw pytest)
./scripts/test.sh tests
./scripts/test.sh test_graph_storage.py          # single file
./scripts/test.sh tests --test-workers 4

# Lint
ruff check .

# API server (after .env is configured via `make env-base`)
lightrag-server                                          # production entrypoint
uvicorn lightrag.api.lightrag_server:app --reload        # dev with reload

# WebUI (from lightrag_webui/)
bun install --frozen-lockfile
bun run dev          # or: bun run build, bun run lint, bun test
```

Pytest markers: `offline`, `integration`, `requires_db`, `requires_api`. Integration tests are skipped by default; opt in with `LIGHTRAG_RUN_INTEGRATION=true` plus the relevant storage connection strings.

## Non-obvious things that will bite you

1. **Always `await rag.initialize_storages()`** after constructing `LightRAG`. Skipping it surfaces later as `AttributeError: __aenter__` or `KeyError: 'history_messages'`. Pair with `await rag.finalize_storages()` on shutdown.
2. **Switching embedding models requires wiping the data directory** (you may keep `kv_store_llm_response_cache.json`). Existing vectors live in the old embedding space and will silently mis-rank.
3. **Wrapping an already-wrapped embedding function**: call `.func` to access the underlying. `EmbeddingFunc(func=openai_embed.func)` is right; `EmbeddingFunc(func=openai_embed)` is wrong.
4. **Pipeline concurrency uses a shared `pipeline_status` dict under `get_namespace_lock("pipeline_status", workspace=...)`** — `busy`, `destructive_busy`, `scanning`, `scanning_exclusive`, `pending_enqueues`, `request_pending` each have a distinct role. Concurrent enqueue + processing is permitted by design; destructive ops (`/documents/clear`, per-doc delete) are the exclusive subset. See the full state table in AGENTS.md before touching `lightrag/pipeline.py`.
5. **Embedding responses come in two formats** (base64 string or raw array) depending on the endpoint. `lightrag/llm/openai.py::openai_embed` handles both — preserve that dual handling if you refactor it.
6. **Lock keys on entity pairs must be sorted** (`sorted([src, tgt])`) to avoid deadlock between symmetric relationship workers.
7. **Never hold a storage lock across an `async generator` yield.** Snapshot under the lock, release, then iterate. See `lightrag/tools/migrate_llm_cache.py::stream_default_caches_json` for the correct pattern.
8. **`LightRAG` carries `@final`** even though it's composed from `_RoleLLMMixin` / `_StorageMigrationMixin` / `_PipelineMixin`. The layering is internal; don't expose it as a subclassing surface.
9. **Setup wizard outputs**: `.env` must stay host-usable. Container-only hostnames and staged SSL paths belong in the wizard-managed compose layer, not persisted back into `.env`.

## Storage layer mental model

Four storage roles (`KV_STORAGE`, `VECTOR_STORAGE`, `GRAPH_STORAGE`, `DOC_STATUS_STORAGE`), each pluggable. Backend registry lives in `lightrag/kg/__init__.py` (`STORAGE_IMPLEMENTATIONS` / `STORAGES`); resolution goes through `kg/factory.py::get_storage_class()`. Workspace isolation differs by backend: file-based uses subdirectories, collection-based uses name prefixes, relational uses column filters, Qdrant uses payload partitioning.

## Style

- English in comments, backend code, and log messages. WebUI strings go through i18next.
- Python: PEP 8, 4-space, type annotations, async/await throughout, `lightrag.utils.logger` (not `print`).
- TS/React: functional components + hooks, PascalCase components, 2-space indent, single quotes, Tailwind utility-first. `@typescript-eslint/no-explicit-any` is intentionally disabled.
