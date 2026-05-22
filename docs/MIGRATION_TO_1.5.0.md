# Migration: enterprise fork → LightRAG 1.5.0

> **Historical record.** This document captures the one-time migration that
> re-ported the fork's enterprise features onto upstream LightRAG 1.5.0. The
> migration is complete — the work shipped on the **`main`** branch (the
> `enterprise-1.5.0` branch referenced below was merged into `main` and then
> deleted). This page is kept for historical context; it is **not** needed to
> install or use LightRAG. For current documentation, see the
> [documentation index](./README.md).

Re-port the fork's enterprise features onto upstream **LightRAG 1.5.0** (targeting tag
`v1.5.0rc2`). The legacy `RAGAnything`-package integration is **dropped** — LightRAG 1.5.0
ships native multimodal processing that replaces it.

## Branches

| Branch | Role |
|---|---|
| `main` | Old upstream snapshot — left intact |
| `RAGAnything` | Old enterprise branch (LightRAG ~1.4 + raganything) — left intact as reference |
| `enterprise-1.5.0` | **Migration target** — based on `upstream/main` (v1.5.0rc2) |

## Decisions

- **Scheme feature dropped.** The `scheme_name` / `SchemeManager` / `/capabilities`
  framework-selection work existed to choose LightRAG-vs-RAGAnything per document.
  Native multimodal makes that automatic — the feature is not re-ported.
- **Target v1.5.0rc2 now.** Re-base onto v1.5.0 GA when released, before production.
- **Multimodal:** drop the `raganything` pip package. Uploaded files use the native
  pipeline (`VLM_PROCESS_ENABLE`, parser routing). Emails keep the fork's own inline-image
  vision extraction and insert assembled text via `ainsert()` (text path).

## Phases

### Phase 1 — Stage conflict-free enterprise modules
New files that do not exist upstream; copied verbatim from `RAGAnything`. They will not
compile until later phases wire them up and remove `raganything` references.

- `lightrag/api/oauth2.py`, `dependencies.py`, `obo_allowlist.py`
- `lightrag/api/routers/{admin,apikey,openai,email}_routes.py`
- `lightrag/api/workspace_manager.py`
- `lightrag/api/OBO_ALLOWLIST.md`, `.obo_allowlist.example`
- `docs/{KEYCLOAK_SSO_SETUP,OAuth2-SSO-Authentication,LINUX_INSTALLATION_GUIDE,Hybrid-Token-Authentication-Implementation-Plan}.md`
- `lightrag_webui/src/features/OAuth2Callback.tsx`

### Phase 2 — Re-port modified API files

**Verified:** `git diff 9bc5f157 RAGAnything` shows the 57 enterprise commits touched the
**API layer only**. `lightrag.py`, `base.py`, `operate.py`, `prompt.py` have **zero**
enterprise changes — the core-library edits seen in cruder `main...RAGAnything` analysis
belong to the dropped hzywhite / PR-#2042 commits. 1.5.0's core library is used unchanged.

Base = 1.5.0; re-apply enterprise hooks. `auth.py` is security-critical — hand-port.

- `lightrag/api/config.py` — DONE: enterprise auth / OAuth2 / multi-tenancy / OBO config
- `lightrag/api/auth.py` — `workspace_id` in the JWT, `sanitize_workspace_id`,
  `_is_admin_user`, `validate_any_token` hybrid validator (LightRAG JWT + Keycloak)
- `lightrag/api/utils_api.py` — per-user API key + cookie-token + hybrid auth (43 lines)
- `lightrag/api/lightrag_server.py` — register the enterprise routers, OAuth2 endpoints,
  WorkspaceManager init; drop raganything init (use 1.5.0's native `vlm` role)
- `lightrag/api/routers/{document,query,graph,ollama}_routes.py` — **no Phase 2 change.**
  Verified (residue analysis of `git diff 9bc5f157 RAGAnything`) that their entire
  enterprise diff is raganything `aquery` wrappers (dropped) + the scheme feature
  (dropped) + workspace resolution. The workspace resolution moves to Phase 3, bundled
  with the `WorkspaceManager`. 1.5.0's versions of these four files stand as-is.
- `routers/__init__.py` — no change (factories imported directly in `lightrag_server.py`)

**Phase 2 status: COMPLETE.** `config.py`, `auth.py`, `utils_api.py`,
`lightrag_server.py` re-ported and committed. `env.example` / `pyproject.toml` carry
only enterprise env vars + test config — folded into Phase 3 / Phase 5.

### Phase 3 — Native multimodal + multi-tenancy
- Remove all `raganything` imports / dependency
- `workspace_manager.py` — drop RAGAnything instance caching; per-tenant plain `LightRAG`
- Build the `WorkspaceManager` init in `lightrag_server.py` (currently stubbed to `None`)
- Thread `get_rag_for_request()` workspace resolution through the document / query /
  graph / ollama routers; drop the raganything `aquery` wrappers and the scheme feature
- Delete `ragmanager.py` if unused
- Emails — vision func from own `VLM_LLM_*` config, not a RAGAnything instance
- File uploads — native pipeline; `VLM_PROCESS_ENABLE=true`, `vlm` role
- `env.example` — add OAuth2 / OBO / multi-tenancy vars; drop `RAGANYTHING_*`

**Phase 3 status: COMPLETE.** Codebase is raganything-free (verified by a full
scan of `lightrag/`). Multi-tenancy fully wired — factory-based `WorkspaceManager`
+ `build_rag` + `resolve_rag`/`resolve_doc_manager` threaded through all four
routers. Email ingestion uses a native VLM vision function; document uploads use
1.5.0's native parser pipeline. `ragmanager.py` was never staged. `openai_api.py`
raganything wrapper removed. `lightrag/api/` compiles clean.

### Phase 4 — WebUI

**Phase 4 status: COMPLETE.** OAuth2/SSO source changes re-applied to `AppRouter.tsx`,
`stores/state.ts`, `features/LoginPage.tsx`, `api/lightrag.ts`, `App.tsx` (the two
`login()` calls needed the new `isSSO` arg), and the 5 locale files; `OAuth2Callback.tsx`
came from Phase 1. `bun run build` succeeds, `tsc -b --noEmit` is clean, `eslint` is
clean. `SchemeManager`/`SchemeContext` were hzywhite's (never staged); `vite.config.ts`
dev-proxy tweak skipped; no stray `package-lock.json`. Note: `lightrag/api/webui/` is
gitignored in 1.5.0 — built assets are generated at package time, not committed.

### Phase 5 — Tests & CI

**Phase 5 status: COMPLETE.** The enterprise "test infrastructure" (a 402-line
`conftest.py`, a `test.yml`, empty `tests/{api,unit,integration}/` dirs, `tests/README.md`)
is empty scaffolding — no actual `test_*.py` files — and is fully superseded by 1.5.0's
139-file test suite with its own `conftest.py` and CI workflows. Porting it would *break*
1.5.0's suite, so nothing is ported; 1.5.0's test infrastructure is kept.

Verified instead that the migration introduces **zero test regressions**: the 1.5.0
offline suite shows 270 failures on `enterprise-1.5.0` — identical to 270 on pristine
`upstream/main` (all environmental — this offline sandbox blocks network the suite needs).
One regression caught during the check — the Phase 3 `document_routes` `Depends`
parameters broke 1.5.0's direct-call unit tests — was fixed (`http_request: Request = None`
+ inline resolution).

### Phase 6 — Verification
- Enterprise: OAuth2 login/logout, multi-tenant isolation, OpenAI-compat endpoint,
  per-user API keys, OBO flow
- Multimodal: upload PDF + DOCX → VLM analysis runs; email with inline images
- `pytest tests/ -m offline` green; `ruff check .` clean; `bun run build` succeeds

## Open verification items

- Confirm 1.5.0's native `workspace` semantics vs. the fork's `workspace_manager` multitenancy
- Decide native-docx-only vs. standing up a mineru/docling endpoint
- Re-base onto v1.5.0 GA before production cutover
