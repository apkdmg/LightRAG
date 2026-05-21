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

### Phase 2 — Re-port modified API/core files
Base = 1.5.0; re-apply enterprise hooks. `auth.py` is security-critical — hand-port.

- `lightrag/api/routers/__init__.py` — export new routers
- `lightrag/api/lightrag_server.py` — register routers, OAuth2 middleware, auth deps; drop raganything init
- `lightrag/api/auth.py` — hybrid token validation, cookie-token, OBO
- `lightrag/api/config.py` — OAuth2/multitenancy/OpenAI config; `RAGANYTHING_*` → `VLM_LLM_*`
- `lightrag/api/utils_api.py` — auth helpers
- `lightrag/api/routers/{document,query,graph,ollama}_routes.py` — workspace scoping; drop raganything branching
- `lightrag/lightrag.py` — `move_file_to_enqueue`/`input_dir` if still wanted; drop `multimodal_content`/`scheme_name`
- `lightrag/base.py` — drop `multimodal_*` (native has them)
- `lightrag/operate.py`, `prompt.py` — re-assess; likely already covered upstream
- `env.example`, `pyproject.toml`

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
