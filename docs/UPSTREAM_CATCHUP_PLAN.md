# Upstream Catch-Up Plan: hold at v1.5.0rc2, merge at GA

This document tracks the deliberate catch-up of the `apkdmg/LightRAG` fork onto
upstream `HKUDS/LightRAG`. It is the working checklist; update the boxes as we
progress.

## DECISION (2026-05-27): HOLD — do not bulk-merge until upstream GA

After triaging all 135 upstream commits (see [Triage](#triage-135-upstream-commits-rc2--upstreammain)),
the decision is to **stay on `v1.5.0rc2` and defer the full catch-up until
upstream tags a GA `v1.5.0` release.** Rationale:

- There is **no stable upstream target newer than rc2.** All 135 commits are
  untagged dev churn on `upstream/main` — merging into them means resolving
  conflicts against code that will change again before GA, then again at GA.
- The fork **works in production** and is already on upstream's newest *release*
  (rc2). "135 behind" is behind unreleased dev, not behind a release.
- Triage found **exactly one must-have** (a concurrency/deadlock fix), and the
  team has **not observed deadlocks** in production (confirmed 2026-05-27) — so
  even that is not urgent. Everything else is irrelevant to this deployment or a
  refactor that is safer to take whole at GA.

**What to do until GA:** nothing, unless a *specific* problem appears (see the
cherry-pick exception in the triage). Phase 0–1 safety work is done and banked;
it does not expire and will be reused at GA. Phases 2+ are **on hold**.

## Current state (snapshot, corrected 2026-05-27)

- **Fork `main` base point**: upstream `v1.5.0rc2` (`b62c2606`) — the merge-base
- **Divergence from `upstream/main`**: 59 ahead / 135 behind (135 = untagged dev)
- **Latest upstream *release* tag**: `v1.5.0rc2` (2026-05-21). Last *stable*
  (non-rc) upstream release is `v1.4.16`. **No GA `v1.5.0` exists yet.**
- **CORRECTION**: `v1.5.0.1` / `v1.5.0.2` / `v1.5.0.3` are **this fork's own
  version tags** (already in `main`) — NOT upstream milestones. An earlier
  draft of this plan said to "walk" them; that was an error (merging them is a
  no-op). The only upstream anchors are `rc1`, `rc2`, then untagged `main`.
- **Reference for prior migration**: `docs/MIGRATION_TO_1.5.0.md`

## Guiding principles

1. `main` is never modified directly during the catch-up.
2. Every step is reversible.
3. Test gates between checkpoints.
4. **Merge, not rebase** — rebase rewrites 59 SHAs, breaks external references,
   and forces conflict resolution one commit at a time.
5. Conflict resolution is conservative — take upstream's structural changes,
   re-apply enterprise logic on top. No "improvements" mixed in.

## High-conflict zones (files upstream touched that we customized)

- `lightrag/api/lightrag_server.py`
- `lightrag/api/config.py`, `auth.py`
- `lightrag/api/routers/document_routes.py`, `graph_routes.py`, `email_routes.py`
- `lightrag/pipeline.py` (sensitive: see CLAUDE.md pipeline_status state table)
- `lightrag/kg/*` (Postgres 18 + AGE image, OpenSearch perf, Mongo sync, Redis refactor)
- `tests/` (upstream reorganized into subfolders)
- Docker compose templates and `env.example`

## Pass-through zones (upstream did not touch — should merge cleanly)

- `docs/MIGRATION_TO_1.5.0.md`, `OAuth2-SSO-Authentication.md`,
  `KEYCLOAK_SSO_SETUP.md`, `OPENAI_COMPATIBLE_API.md`, `PER_USER_API_KEYS.md`,
  `EMAIL_INGESTION.md`
- `lightrag/api/oauth2.py`, `obo_allowlist.py`, `dependencies.py`
- `lightrag/api/routers/apikey_routes.py`, `admin_routes.py`
- OBO allowlist files

---

# Phase 0 — Pre-merge safety net

- [x] **0.1** Confirm clean working tree (`git status` shows only known files)
  — done 2026-05-27. Working tree had only untracked plan/script files; no
  uncommitted modifications to tracked files. HEAD at `ffbacfb7` on `main`.
- [x] **0.2** Tag current `main` as the rollback point and push:
  ```bash
  git tag -a pre-1.5.0.3-merge -m "main before upstream v1.5.0.x catch-up" main
  git push origin pre-1.5.0.3-merge
  ```
  — done 2026-05-27. Annotated tag `pre-1.5.0.3-merge` (object
  `a5e5e308…`) created on `ffbacfb7` and pushed to `origin`.
  Rollback: `git reset --hard pre-1.5.0.3-merge`.
- [x] **0.3** Capture offline test baseline (`/tmp/baseline-offline-pre.log`):
  ```bash
  ./scripts/test.sh tests 2>&1 | tee /tmp/baseline-offline-pre.log
  ```
  — done 2026-05-27. Result: **21 collection errors, 5 skipped, 5.59s**.
  All 21 errors are `ModuleNotFoundError`-style collection failures for tests
  that need optional storage drivers (`asyncpg`, `pgvector`, `pymilvus`,
  `qdrant`, `memgraph`, etc.) which `pipmaster` could not install offline.
  This is the *environmental* baseline — not a regression signal by itself.
  Post-merge: same 21 errors expected; any *new* error indicates a regression.

- [x] **0.4** Capture lint baseline (`/tmp/baseline-ruff-pre.log`):
  `ruff check . 2>&1 | tee /tmp/baseline-ruff-pre.log`
  — done 2026-05-27. Result: **All checks passed!** (Post-merge must stay clean.)

- [x] **0.5** Capture WebUI baselines (from `lightrag_webui/`):
  - **0.5a** `bun run lint` (`/tmp/baseline-webui-lint-pre.log`) — done
    2026-05-27. Result: `eslint .` ran with no errors.
  - **0.5b** `bun run build` (`/tmp/baseline-webui-build-pre.log`) — done
    2026-05-27. Result: `✓ built in 720ms`. Largest chunk
    `index-Cac1tOSV.js` 2.9 MB (gzip 898 kB) — track this; a significant
    growth post-merge may indicate accidentally bundled deps.

- [x] **0.6** Smoke-check server boot (`/tmp/baseline-server-help.txt`):
  `.venv/bin/lightrag-server --help > /tmp/baseline-server-help.txt`
  — done 2026-05-27. Result: 292 lines of help text, exit 0.
  Confirms imports + entrypoint resolve cleanly.
- [ ] **0.7** Snapshot production storage volumes.
  Prod runs the **Full/GPU preset** (`docker-compose-full.yml`): Postgres
  (KV+DocStatus), Neo4j (Graph), Milvus+etcd+minio (Vector), plus bind-mounted
  `./data/{rag_storage,inputs,prompts}`. A maintenance window is acceptable, so
  use the cold-snapshot script:
  ```bash
  # On the production host, from the deploy dir:
  DRY_RUN=1 scripts/backup/pre-merge-snapshot.sh        # preview, changes nothing
  ENV_FILE=.env scripts/backup/pre-merge-snapshot.sh    # real run (point at prod .env)
  ```
  The script: pg_dump (portable) → stop stack → tar all 5 named volumes + 3
  bind dirs → restart → MANIFEST with image digests + sha256. **Copy the
  resulting `backups/pre-merge-<ts>/` dir OFF the host.** Restore steps:
  `docs/ROLLBACK.md` §C.4. _Script ready; awaiting execution on prod host with
  the real `.env`._
- [x] **0.8** Document the rollback procedure in `docs/ROLLBACK.md`
  — done 2026-05-27. Three scenarios (A: side branch only,
  B: main reset but not deployed, C: production rollback with optional
  data restore). Anchor tag is `pre-1.5.0.3-merge`; rollback verification
  uses `tests/enterprise_smoke.py`.

# Phase 1 — Build the test harness (do BEFORE any merge)

Tests are what convert "merge looks fine" into a verifiable claim. The
existing pytest suite covers core RAG mechanics but does not cover the
enterprise surface.

- [ ] **1.1** Write `tests/enterprise_smoke.py` — black-box HTTP smoke checks
  for each enterprise feature (OAuth2, multitenancy, per-user API keys,
  OpenAI-compatible endpoint, native .docx vs docling routing, query modes,
  admin/OBO routes)
- [ ] **1.2** Run the smoke script against the **current pre-merge build** and
  save the green baseline output to `/tmp/baseline-smoke-pre.log`
- [ ] **1.3** Stand up a staging environment with a snapshot of production
  data; confirm smoke script passes against it too

# Phase 2 — ON HOLD until upstream GA

> **Do not start Phase 2 until upstream tags a GA `v1.5.0`.** The earlier
> "walk v1.5.0.1 → .2 → .3" steps were removed — those are the fork's own tags,
> already in `main`, so merging them does nothing. There is no valid stepwise
> tag-walk on the upstream side between rc2 and `main`.

**When GA lands**, execute as a single merge to the GA tag on a side branch:

```bash
git fetch upstream --tags
git switch -c upstream-catchup main
git merge --no-ff --no-commit v1.5.0          # the GA tag, once it exists
```

- [ ] **2.1** Merge the GA tag (`--no-commit` first; review the conflict set)
- [ ] **2.2** Resolve conflicts — high-conflict zones above; conservative resolution
- [ ] **2.3** Commit the merge
- [ ] **2.4** Run gates: `ruff check .`, `./scripts/test.sh tests`,
  `bun run lint && bun run build`, `tests/enterprise_smoke.py` against a local server
- [ ] **2.5** Diff offline tests vs `baselines/pre-merge/baseline-offline-pre.log`
  — same 21 env errors = green; any *new* error = regression
- [ ] **2.6** Tag checkpoint: `git tag merge-ga`

**Re-run the triage at GA** — the must-have/nice/irrelevant split below was made
against `upstream/main` as of 2026-05-27 and will have shifted by GA.

# Phase 3 — Manual review of high-risk merged files

Even with green tests, eyeball these files post-merge:

- [ ] **3.1** `lightrag/api/lightrag_server.py` (boot order, middleware stack intact)
- [ ] **3.2** `lightrag/api/auth.py` + `config.py` (auth gates not bypassed)
- [ ] **3.3** `lightrag/pipeline.py` against the CLAUDE.md `pipeline_status` state table
- [ ] **3.4** `lightrag/api/routers/document_routes.py` (ingestion entry path)
- [ ] **3.5** `lightrag/api/routers/email_routes.py` (native VLM vision path preserved)

# Phase 4 — Live acceptance on staging

Restore a snapshot of production data into staging, deploy the
`upstream-catchup` build, and run the full acceptance matrix:

- [ ] **4.1** Keycloak OAuth2 login round-trip
- [ ] **4.2** Per-user API keys: create + use for `/query`
- [ ] **4.3** OpenAI-compatible `/v1/chat/completions` returns valid response
- [ ] **4.4** Multimodal ingestion (`vlm_process_enable` path) processes images
- [ ] **4.5** Docling routes PDF/PPTX/XLSX correctly
- [ ] **4.6** Native parser handles `.docx` (not routed to docling)
- [ ] **4.7** Multitenancy: workspace A docs invisible to workspace B
- [ ] **4.8** Admin routes (`/admin/users`) gated to admin JWT
- [ ] **4.9** OBO allowlist enforcement
- [ ] **4.10** Email ingestion pipeline ingests + extracts inline images
- [ ] **4.11** Soak test: 24h of synthetic traffic without error-rate spike

# Phase 5 — Merge `upstream-catchup` into `main`

Only after Phase 4 is green:

- [ ] **5.1** `git switch main`
- [ ] **5.2** `git merge --no-ff upstream-catchup -m "merge: catch up to upstream v1.5.0.3+"`
- [ ] **5.3** Push: `git push origin main`
- [ ] **5.4** Confirm GitHub banner now shows "59 ahead, 0 behind" (or similar)

# Phase 6 — Production cutover

- [ ] **6.1** Deploy merged build to production (blue-green or rolling, depending on infra)
- [ ] **6.2** Run smoke script against production
- [ ] **6.3** Watch error rates / latency for 24h
- [ ] **6.4** Confirm no on-call pages or user reports

# Phase 7 — Cleanup (only after Phase 6 is stable for a week+)

- [ ] **7.1** Delete intermediate tags: `git tag -d merge-step-1 merge-step-2 merge-step-3 merge-step-4`
- [ ] **7.2** Delete `upstream-catchup` branch
- [ ] **7.3** Keep `pre-1.5.0.3-merge` tag for at least 30 days as final safety net
- [ ] **7.4** Update `docs/MIGRATION_TO_1.5.0.md` with a "Status: superseded" note pointing here

---

## Rollback procedure

**See `docs/ROLLBACK.md` for the full operational runbook** — three scenarios
(pre-merge, post-merge-not-deployed, post-deploy), data-restore decision tree,
and the quick-reference card.

Quick anchor: `git reset --hard pre-1.5.0.3-merge && git push --force-with-lease origin main`.

**Rehearse the runbook on staging once before you need it in production.**

---

## Triage: 135 upstream commits (rc2 → upstream/main)

_As of 2026-05-27. Deployment profile: PG (KV+DocStatus) · Neo4j (graph) ·
Milvus (vector) · OpenAI LLM/embed · enterprise API (OAuth2/Keycloak,
multitenancy, OpenAI-compat, per-user API keys, email ingestion, admin/OBO) ·
docling (PDF/PPTX/XLSX) + native docx + VLM._

Of 135 commits, 40 are merge commits; most of the rest touch backends/providers
this deployment does not run. Filtered result:

### 🔴 MUST-HAVE (1 cluster)
- `0c7e16f0` + `ce60a7a4` — **fix(concurrency): business-layer locking gaps &
  batch deadlock.** Workspace-namespaced keyed locks in `lightrag.py` /
  `utils_graph.py` (backend-agnostic → protects Neo4j+Milvus writes). Matches
  the multitenant concurrent-ingestion risk profile.
  **Status: NOT pulled — team confirmed no deadlocks observed (2026-05-27).**

### 🟡 NICE-TO-HAVE (relevant, not urgent)
- `fb5a3c56` guard graph mutations when busy
- `13d7dda5`, `90e5573f` server proxy-prefix / `scope.path` fixes (only if
  running behind a path-prefix reverse proxy)
- `a5c52f1f`, `05f03262`, `8b3a02a2`, `f60067bf` pipeline parse/analyze progress
  + `/cancel_pipeline` into PARSE/ANALYZE + stage metadata (multimodal UX)
- `1ed9c074` docx: drop revision/comment markers, skip empty tables
  (improves native-docx output — **conflicts with the docx fork**)
- `9a1a0046`, `6d0c8361` expose `split_by_character` in `/documents/text`

### ⚪ IRRELEVANT (not in this stack, or cosmetic)
- OpenSearch / MongoDB / Redis / Qdrant / Memgraph backend commits
- PGGraphStorage advisory-lock fixes (`217ddec9`, `3c011cea`, `9ad5d89f`) —
  graph is Neo4j here, PG-graph code is dead
- Anthropic LLM fixes (`93fb37ee`, `6fe5faba`, `f65ff2c2`) — uses OpenAI
- Swagger dark-theme styling; `901edbdd` atomic index_done for Faiss/Json/Nano
- Test reorg (~290 file touches), docs, AGENTS.md, k8s values, setup-wizard

### ⛔ AVOID pre-GA (refactor / breaking, conflict-prone, no functional need)
- `84f1d9ff` refactor: consolidate parser modules under `lightrag/parser/` —
  collides with the docx-routing fork
- `43cfb39a` remove deprecated `QueryParam.model_func` + `history_turns` —
  **`history_turns` is used in `api/routers/openai_api.py:227-229` +
  `config.py:705`**; this refactor would break the OpenAI-compat history path
- `af5775c8` + `c3e665fc` EMBEDDING_TOKEN_LIMIT enforcement then reverted (net zero)

### Cherry-pick exception
The only reason to act before GA: a *specific* problem appears — e.g. ingestion
hangs / Postgres `deadlock detected` / `lock timeout` / `pipeline_status.busy`
stuck under concurrent load. Then cherry-pick `0c7e16f0` + `ce60a7a4` onto a
branch (business-layer hunks apply cleanly; the `postgres_impl.py` hunk may
conflict — take upstream or skip it, since PG-graph is unused), run the smoke +
offline baselines, and ship that alone. Otherwise, wait for GA.

---

## Decision log

- **2026-05-27** — **HOLD until GA.** Triaged all 135 commits; only 1 must-have
  (concurrency/deadlock fix), no deadlocks observed in prod, no stable upstream
  target past rc2. Bulk-merging unreleased dev = double conflict-resolution cost.
  Deferring full catch-up to upstream GA `v1.5.0`. Phase 0–1 safety work banked.
- **2026-05-27** — Corrected tag-walk error: `v1.5.0.1/.2/.3` are fork tags,
  not upstream milestones; removed the Phase 2 stepwise-tag-walk.
