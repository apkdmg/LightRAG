# Migration: enterprise fork → LightRAG 1.5.0

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
- `lightrag/api/routers/document_routes.py` — workspace scoping; drop raganything
  branching and the scheme feature
- `lightrag/api/routers/{query,graph,ollama}_routes.py` — workspace scoping; drop the
  raganything `aquery` wrappers
- `routers/__init__.py` — no change (factories imported directly in `lightrag_server.py`)
- `env.example`, `pyproject.toml` — enterprise vars; `RAGANYTHING_*` dropped

### Phase 3 — Native multimodal, drop raganything
- Remove all `raganything` imports / dependency
- `workspace_manager.py` — drop RAGAnything instance caching; per-tenant plain `LightRAG`
- Delete `ragmanager.py` if unused
- Emails — vision func from own `VLM_LLM_*` config, not a RAGAnything instance
- File uploads — native pipeline; `VLM_PROCESS_ENABLE=true`, `vlm` role

### Phase 4 — WebUI
Rebuild with Bun (`bun run build`) — do not port built assets. Re-apply source changes
(`App.tsx`, `AppRouter.tsx`, `LoginPage.tsx`, `api/lightrag.ts`, `stores/state.ts`,
`OAuth2Callback.tsx`). Drop `SchemeManager/*`, `SchemeContext.tsx`. Delete stray
`package-lock.json` (Bun-only project).

### Phase 5 — Tests & CI
Reconcile `tests/conftest.py` with 1.5.0; port `tests/{api,unit,integration}/`; reconcile
`.github/workflows/test.yml`.

### Phase 6 — Verification
- Enterprise: OAuth2 login/logout, multi-tenant isolation, OpenAI-compat endpoint,
  per-user API keys, OBO flow
- Multimodal: upload PDF + DOCX → VLM analysis runs; email with inline images
- `pytest tests/ -m offline` green; `ruff check .` clean; `bun run build` succeeds

## Open verification items

- Confirm 1.5.0's native `workspace` semantics vs. the fork's `workspace_manager` multitenancy
- Decide native-docx-only vs. standing up a mineru/docling endpoint
- Re-base onto v1.5.0 GA before production cutover
