# LightRAG — Docker Installation Guide

A comprehensive guide to installing and running the **LightRAG enterprise server**
with Docker, Docker Compose, or Podman.

> This is the enterprise fork — [apkdmg/LightRAG](https://github.com/apkdmg/LightRAG).
> **Multi-tenancy and OAuth2/Keycloak SSO are enabled by default** — see
> [Configuration](#configuration).
>
> For a non-containerized install, see
> [LINUX_INSTALLATION_GUIDE.md](./LINUX_INSTALLATION_GUIDE.md).

## Contents

- [Prerequisites](#prerequisites)
- [Installation methods at a glance](#installation-methods-at-a-glance)
- [Method 1 — Run the pre-built image](#method-1--run-the-pre-built-image)
- [Method 2 — Docker Compose (recommended)](#method-2--docker-compose-recommended)
- [Method 3 — Build the image yourself](#method-3--build-the-image-yourself)
- [Configuration](#configuration)
- [Data persistence](#data-persistence)
- [Full stack with storage backends](#full-stack-with-storage-backends)
- [No-GPU preset](#no-gpu-preset)
- [Podman](#podman)
- [Production notes](#production-notes)
- [Verifying the signed image](#verifying-the-signed-image)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## Prerequisites

- **Docker Engine 20.10+** (24.0+ recommended) — check with `docker --version`
- **Docker Compose v2** — check with `docker compose version`
  (note the space: `docker compose`, not the legacy `docker-compose`)
- *or* **Podman 4.0+** with `podman-compose` — see [Podman](#podman)
- **Hardware:** 2 CPU / 4 GB RAM minimum; 4 CPU / 8 GB+ recommended
- **Network:** outbound access to your LLM / embedding provider

The image is **multi-architecture** — it runs natively on `linux/amd64` and
`linux/arm64` (including Apple Silicon), with no emulation.

Don't have Docker yet? See the official install docs:
<https://docs.docker.com/engine/install/>.

---

## Installation methods at a glance

| Method | Best for | Clone repo? |
|--------|----------|-------------|
| **1. Pre-built image** (`docker pull`) | Fastest start, single container | No |
| **2. Docker Compose** | Standard deployment, easy upgrades | Yes |
| **3. Build it yourself** | You've modified the source | Yes |
| **Full stack** (Compose) | All-in-one with PostgreSQL / Neo4j / Milvus | Yes |
| **No-GPU preset** | Bundled Postgres + Neo4j, cloud LLM / embedding / reranker | Yes |

---

## Method 1 — Run the pre-built image

The fork publishes a **public, multi-arch, Cosign-signed** image to GitHub
Container Registry: **`ghcr.io/apkdmg/lightrag`**. It is public — no
`docker login` required.

### 1. Pull the image

```bash
docker pull ghcr.io/apkdmg/lightrag:latest
```

Available tags:

- `latest` — the newest stable release
- version tags, e.g. `v1.5.0.1` — **pin one of these for production**

### 2. Create a configuration file

The container needs an `.env` file with your LLM / embedding settings. Download
the template (no repo clone needed):

```bash
curl -fsSL https://raw.githubusercontent.com/apkdmg/LightRAG/main/env.example -o .env
```

Then edit `.env` — at a minimum set `TOKEN_SECRET` and your LLM / embedding
configuration. See [Configuration](#configuration).

### 3. Create data directories

```bash
mkdir -p data/rag_storage data/inputs data/prompts
```

### 4. Run the container

```bash
docker run -d --name lightrag -p 9621:9621 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/data/rag_storage:/app/data/rag_storage" \
  -v "$(pwd)/data/inputs:/app/data/inputs" \
  -v "$(pwd)/data/prompts:/app/data/prompts" \
  ghcr.io/apkdmg/lightrag:latest
```

`--add-host=host.docker.internal:host-gateway` lets the container reach services
running on the **host** (e.g. a local LLM or embedding server) via the hostname
`host.docker.internal`.

To pin a release instead of `latest`, swap the final argument for a version tag,
e.g. `ghcr.io/apkdmg/lightrag:v1.5.0.1`.

### 5. Verify

```bash
curl http://localhost:9621/health        # server health
curl http://localhost:9621/auth-status   # authentication / SSO status
```

Open the WebUI in a browser at **http://localhost:9621**.

### Managing the container

```bash
docker logs -f lightrag      # follow logs
docker stop lightrag         # stop
docker start lightrag        # start again
docker rm -f lightrag        # remove (data in mounted volumes is kept)
```

---

## Method 2 — Docker Compose (recommended)

Compose is the recommended path: configuration lives in a file, upgrades are one
command, and the bundled `docker-compose.yml` already wires up volumes, ports,
the restart policy, and host networking.

```bash
git clone https://github.com/apkdmg/LightRAG.git
cd LightRAG
cp env.example .env          # then edit .env — see Configuration below
docker compose up -d
```

The bundled `docker-compose.yml`:

- runs `ghcr.io/apkdmg/lightrag:latest`
- maps `${HOST:-0.0.0.0}:${PORT:-9621}:9621`
- mounts `./data/rag_storage`, `./data/inputs`, `./data/prompts`, and `./.env`
- restarts on failure, and maps `host.docker.internal` for host services

Common commands:

```bash
docker compose logs -f       # follow logs
docker compose ps            # status
docker compose restart       # restart
docker compose down          # stop and remove containers
```

### Updating

```bash
docker compose pull          # fetch the newest image
docker compose down
docker compose up -d
```

---

## Method 3 — Build the image yourself

If you have modified the source, build the image locally instead of pulling it.

```bash
git clone https://github.com/apkdmg/LightRAG.git
cd LightRAG
docker compose up -d --build      # build, then run via Compose
```

Or build the image directly:

```bash
docker build -t lightrag-local .
```

The `Dockerfile` is a three-stage build — frontend assets via Bun, Python
dependencies via `uv`, and a slim `python:3.12-slim` runtime. BuildKit is enabled
automatically by the `# syntax=docker/dockerfile:1` directive and caches
dependency layers between builds.

For multi-architecture builds and pushing to a registry, see
[DockerDeployment.md](./DockerDeployment.md#-build-docker-images).

---

## Configuration

LightRAG reads its configuration from the `.env` file, mounted into the container
at `/app/.env`.

### Required settings

| Setting | Purpose |
|---------|---------|
| `TOKEN_SECRET` | Signs every JWT (including SSO tokens). **Must** be set — use 32+ random characters. |
| `LLM_BINDING`, `LLM_BINDING_HOST`, `LLM_MODEL`, `LLM_BINDING_API_KEY` | LLM provider |
| `EMBEDDING_BINDING`, `EMBEDDING_BINDING_HOST`, `EMBEDDING_MODEL`, `EMBEDDING_DIM` | Embedding provider |

### Enterprise features — enabled by default

This fork turns on **OAuth2/Keycloak SSO** and **multi-tenancy** out of the box:

- `OAUTH2_ENABLED=true` — supply Keycloak credentials, **or** set
  `OAUTH2_ENABLED=false` to use username/password or guest login instead.
  If left enabled without credentials, the server still starts but logs a
  startup warning and the SSO login flow is inactive.
- `ENABLE_MULTI_TENANCY=true` — each user gets an isolated workspace.

For the full enterprise `.env` block — OAuth2 endpoints, `ADMIN_ACCOUNTS`,
`AUTH_ACCOUNTS`, OBO allowlist — see the OAuth2 section of
[LINUX_INSTALLATION_GUIDE.md](./LINUX_INSTALLATION_GUIDE.md) and
[KEYCLOAK_SSO_SETUP.md](./KEYCLOAK_SSO_SETUP.md). The annotated
[`env.example`](../env.example) lists every variable.

### Reaching services on the host

If your LLM or embedding server runs on the Docker **host**, point the binding
host at `http://host.docker.internal:<port>`. Compose maps that hostname
automatically; for `docker run`, add `--add-host=host.docker.internal:host-gateway`
(shown in [Method 1](#method-1--run-the-pre-built-image)).

### Container paths (baked into the image)

| Path | Env var | Contents |
|------|---------|----------|
| `/app/.env` | — | configuration file |
| `/app/data/rag_storage` | `WORKING_DIR` | knowledge graph + vector data |
| `/app/data/inputs` | `INPUT_DIR` | documents to ingest |
| `/app/data/prompts` | `PROMPT_DIR` | custom prompts |
| `/app/data/tiktoken` | `TIKTOKEN_CACHE_DIR` | pre-bundled tokenizer cache |

The container exposes port **9621** and runs `lightrag.api.lightrag_server` as its
entrypoint.

---

## Data persistence

Containers are ephemeral — mount host directories (or named volumes) so your data
survives `docker rm` and image upgrades:

| Mount | Keep it? | Why |
|-------|----------|-----|
| `data/rag_storage` | **Yes — back this up** | the knowledge graph and embeddings |
| `data/inputs` | Recommended | uploaded / scanned source documents |
| `data/prompts` | Optional | custom prompt overrides |

When you use database-backed storage (PostgreSQL / Neo4j / Milvus), the graph and
vectors live in those services instead — see [Full stack](#full-stack-with-storage-backends).

---

## Full stack with storage backends

The bundled `docker-compose-full.yml` brings up everything LightRAG needs for a
production-grade graph + vector + reranker stack: PostgreSQL, Neo4j, Milvus
(with its etcd and MinIO sidecars), and local vLLM embedding + rerank services.

### Services and ports

| Service | Image | Port | Role | GPU? |
|---------|-------|------|------|------|
| `lightrag` | `ghcr.io/apkdmg/lightrag:latest` | **9621** (exposed) | API + WebUI | No |
| `vllm-embed` | `vllm/vllm-openai:latest` | 8001 (exposed) | Embedding model server — `BAAI/bge-m3` | **Yes** |
| `vllm-rerank` | `vllm/vllm-openai:latest` | 8000 (exposed) | Reranker — `BAAI/bge-reranker-v2-m3` | **Yes** |
| `postgres` | `pgvector/pgvector:pg18` | 5432 (internal) | KV + vector storage (pgvector) | No |
| `neo4j` | `neo4j:5-community` | 7474, 7687 (internal) | Graph storage | No |
| `milvus` | `milvusdb/milvus:v2.6.11-gpu` | 19530 (internal) | Alternative vector storage | **Yes** |
| `milvus-etcd` | `quay.io/coreos/etcd:v3.5.25` | — | Milvus metadata sidecar | No |
| `milvus-minio` | `minio/minio` | — | Milvus object-store sidecar | No |

Services communicate over the compose network by hostname (`postgres`,
`neo4j`, `milvus`, `vllm-embed`, `vllm-rerank`).

### Prerequisites

A GPU host is required for the default stack — Milvus and both vLLM services
use `runtime: nvidia`. You need:

- An NVIDIA GPU with a recent driver
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  installed, with Docker configured to use the `nvidia` runtime
- ~20 GB+ free disk for model weights and database volumes

If you don't have a GPU, see [CPU-only setup](#cpu-only-setup) below.

### Required environment variables

Compose fails fast if the secrets below are missing. Export them in your shell
or place them in a top-level `.env` (Compose reads it automatically):

```bash
# Neo4j authentication
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-strong-password

# Milvus object store (MinIO)
MINIO_ACCESS_KEY_ID=your-minio-key
MINIO_SECRET_ACCESS_KEY=your-minio-secret

# vLLM API keys (must match the matching *_BINDING_API_KEY in LightRAG's .env)
VLLM_EMBED_API_KEY=any-shared-token
VLLM_RERANK_API_KEY=any-shared-token
```

In LightRAG's `.env`, point storage and bindings at the bundled services:

```bash
# Storage backends
KV_STORAGE=PGKVStorage
VECTOR_STORAGE=PGVectorStorage       # or MilvusVectorStorage to use Milvus
GRAPH_STORAGE=Neo4JStorage
DOC_STATUS_STORAGE=PGDocStatusStorage

# Embedding — talks to vllm-embed over the compose network
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024                   # bge-m3 is 1024-dim
EMBEDDING_BINDING_API_KEY=any-shared-token   # match VLLM_EMBED_API_KEY

# Reranker — vLLM exposes /v1/rerank, which works with the cohere binding
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_API_KEY=any-shared-token      # match VLLM_RERANK_API_KEY
```

PostgreSQL host/port and the Neo4j/Milvus URIs are already wired into the
`lightrag` service's `environment` block in the compose file — you only need to
override them when [using external backends](#using-external-backends).

### Bring it up

```bash
docker compose -f docker-compose-full.yml up -d
docker compose -f docker-compose-full.yml ps          # wait for all services to be "healthy"
docker compose -f docker-compose-full.yml logs -f lightrag
```

The first start downloads ~5–10 GB of model weights for the vLLM services —
expect them to take a few minutes to reach `healthy`.

### CPU-only setup

If you don't have a GPU, override the GPU-bound services with a
`docker-compose.override.yml` next to the full compose file:

```yaml
services:
  vllm-embed:
    image: vllm/vllm-openai-cpu:latest
    runtime: ~
    deploy:
      resources: {}
    command: >
      --model BAAI/bge-m3
      --port 8001
      --dtype float32
      --api-key ${VLLM_EMBED_API_KEY}

  vllm-rerank:
    image: vllm/vllm-openai-cpu:latest
    runtime: ~
    deploy:
      resources: {}
    command: >
      --model BAAI/bge-reranker-v2-m3
      --port 8000
      --dtype float32
      --api-key ${VLLM_RERANK_API_KEY}
```

Milvus does not have a comfortable CPU story. For CPU-only deployments, the
simplest path is to drop the milvus services and use **`VECTOR_STORAGE=PGVectorStorage`**
(vectors stored alongside KV in PostgreSQL). See
[Using external backends](#using-external-backends) for how to exclude services.

### Customizing each backend

#### PostgreSQL

Default image `pgvector/pgvector:pg18` supports `PGKVStorage`, `PGVectorStorage`,
and `PGDocStatusStorage` — **but not** graph storage. To get pgvector **and**
Apache AGE in one Postgres (so you can drop Neo4j and use `PGGraphStorage`),
override the image to the variant used by the interactive setup:

```yaml
services:
  postgres:
    image: gzdaniel/postgres-for-rag:16.6   # pgvector + Apache AGE
```

Default credentials are `rag` / `rag` / `rag` (user / password / database).
Override via the `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` keys on the
`postgres` service.

> ⚠️ Postgres major-version data files are not compatible across upgrades —
> pick the major version before you ingest data; don't roll it back later.

#### Neo4j

Authentication is sourced from the `NEO4J_USERNAME` / `NEO4J_PASSWORD`
environment variables; compose fails to start if either is unset. To expose
the browser UI for inspection, uncomment the `ports` block on the `neo4j`
service (`7474:7474`, `7687:7687`) — and only do this on a trusted network.

#### Milvus

The bundled image `milvusdb/milvus:v2.6.11-gpu` requires a GPU host. To use
Milvus instead of pgvector for the vector store, set
`VECTOR_STORAGE=MilvusVectorStorage` in `.env`; LightRAG uses
`MILVUS_URI=http://milvus:19530` from the compose env block. Milvus depends on
its `milvus-etcd` (metadata) and `milvus-minio` (object store) sidecars — both
come up automatically.

#### vLLM embedding / rerank

Defaults are `BAAI/bge-m3` for embeddings (1024-dim) and
`BAAI/bge-reranker-v2-m3` for reranking. Override at startup:

```bash
VLLM_EMBED_MODEL=Qwen/Qwen3-Embedding-8B \
VLLM_RERANK_MODEL=BAAI/bge-reranker-v2-m3 \
docker compose -f docker-compose-full.yml up -d
```

If you change the embedding model, **also update `EMBEDDING_DIM`** in `.env` to
match the new model's output dimension, and wipe vector storage before
re-indexing — the embedding dimension is baked into the vector tables /
collections, and changing it mid-flight will throw dimension-mismatch errors.

### Using external backends

If you already operate PostgreSQL, Neo4j, or Milvus, don't run the bundled
ones. Use the basic `docker-compose.yml` (LightRAG only) instead of
`docker-compose-full.yml`, and point `.env` at your existing services:

```bash
POSTGRES_HOST=postgres.internal.example.com
POSTGRES_PORT=5432
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=...
POSTGRES_DATABASE=lightrag

NEO4J_URI=neo4j://neo4j.internal.example.com:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

MILVUS_URI=http://milvus.internal.example.com:19530
```

To keep using the *full* stack but drop just one or two services (e.g. you
have an external Postgres but still want the bundled vLLM + Milvus), write a
`docker-compose.override.yml` that disables those services:

```yaml
services:
  postgres: !reset { }   # Compose v2.20+ — remove the service from the stack
```

(Older Compose: copy `docker-compose-full.yml` into a slimmer custom file
without the services you replace.)

### The interactive setup wizard

For a stack tailored to your chosen backends without hand-editing compose,
use the wizard — it generates a `docker-compose.final.yml` and a matching
`.env`:

```bash
make env-base             # 1. LLM, embedding, reranker (offers local vLLM)
make env-storage          # 2. storage backends + database services
make env-server           # 3. server port, auth, SSL
make env-security-check   # 4. audit the generated .env before going live

docker compose -f docker-compose.final.yml up -d
```

Reruns preserve manual edits inside wizard-managed service blocks; to force
regeneration from the templates, use `make env-base-rewrite` and
`make env-storage-rewrite`. See [InteractiveSetup.md](./InteractiveSetup.md)
for the full target reference.

---

## No-GPU preset

For deployments without GPU hardware, the bundled `docker-compose-no-gpu.yml`
runs **3 containers** — LightRAG, PostgreSQL (with pgvector, also serving KV +
DOC_STATUS), and Neo4j. Embeddings, reranker, and the LLM all come from cloud
APIs (e.g. OpenAI + Jina), so nothing local needs a GPU.

### What's in it

| Service | Image | Ports | Role |
|---------|-------|-------|------|
| `lightrag` | `ghcr.io/apkdmg/lightrag:latest` | `9621` | API + WebUI |
| `postgres` | `pgvector/pgvector:pg18` | `127.0.0.1:5432` (DBA-accessible) | KV + DOC_STATUS + VECTORS |
| `neo4j` | `neo4j:5-community` | `127.0.0.1:7474`, `127.0.0.1:7687` (DBA-accessible) | Graph |

Resource baseline: roughly 4 GB RAM and 10 GB disk for a starter deployment.

### Configure `.env`

A complete template is provided at
[`env.docker-compose-no-gpu`](../env.docker-compose-no-gpu). Copy and edit:

```bash
cp env.docker-compose-no-gpu .env
# edit .env — at minimum:
#   TOKEN_SECRET, POSTGRES_PASSWORD, NEO4J_PASSWORD,
#   LLM_BINDING_API_KEY, EMBEDDING_BINDING_API_KEY, RERANK_BINDING_API_KEY
```

Storage selectors are pre-set in the template:

```bash
KV_STORAGE=PGKVStorage
VECTOR_STORAGE=PGVectorStorage
GRAPH_STORAGE=Neo4JStorage
DOC_STATUS_STORAGE=PGDocStatusStorage
```

The reranker uses the native Jina binding (the binding's default `base_url`
already targets `https://api.jina.ai/v1/rerank`):

```bash
RERANK_BINDING=jina
RERANK_MODEL=jina-reranker-v2-base-multilingual
RERANK_BINDING_API_KEY=jina_...
```

### Bring it up

```bash
docker compose -f docker-compose-no-gpu.yml up -d
docker compose -f docker-compose-no-gpu.yml ps        # wait for healthy
docker compose -f docker-compose-no-gpu.yml logs -f lightrag
```

LightRAG enables the `pgvector` extension on first connect — no manual SQL
required.

### DBA access

Both database ports are published to **localhost** by default, so DBAs can
connect from the host with standard tooling:

```bash
# Postgres — psql, pg_dump, pgAdmin, DBeaver, DataGrip
psql -h 127.0.0.1 -U rag -d lightrag
pg_dump -h 127.0.0.1 -U rag lightrag > backup.sql

# Neo4j Browser (web UI)
open http://127.0.0.1:7474

# Neo4j Bolt — cypher-shell, official drivers, Neo4j Desktop
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p "$NEO4J_PASSWORD"
```

To expose the DB ports on your **admin network** rather than localhost, set in
`.env`:

```bash
POSTGRES_HOST_BIND=10.0.0.5     # the host's admin-network IP, or 0.0.0.0
NEO4J_HOST_BIND=10.0.0.5
```

When you bind to anything other than `127.0.0.1`, restrict the listening port
at the host firewall, use strong credentials (not the default `rag/rag`), and
put the Neo4j Browser behind a reverse proxy with HTTPS.

### When to choose this preset

| If you... | Use |
|-----------|-----|
| Have no GPU; want LightRAG's data on infra you fully control (DBA + in-house backups) | **This preset** |
| Have NVIDIA GPU and want to run vLLM / Milvus locally | [Full stack](#full-stack-with-storage-backends) |
| Use cloud-managed databases (RDS / Neo4j Aura / Zilliz) | The basic [Docker Compose](#method-2--docker-compose-recommended) — point `.env` at the cloud endpoints |

---

## Podman

A Podman-compatible Compose file is provided:

```bash
podman-compose -f docker-compose.podman.yml up -d
```

Differences from the Docker Compose file:

- uses a top-level `restart` key instead of `deploy.restart_policy`
- omits the `host-gateway` mapping — Podman auto-provides
  **`host.containers.internal`**; use that hostname (not `host.docker.internal`)
  in your `.env` binding hosts when reaching host services.

---

## Production notes

- **Pin a version tag** (`:v1.5.0.1`) rather than `:latest` so deployments are
  reproducible.
- **Set a strong `TOKEN_SECRET`** — never ship a default or weak value.
- **Configure OAuth2 or disable it explicitly** — don't leave it enabled but
  unconfigured.
- **Restart policy** — Compose restarts on failure; for `docker run`, add
  `--restart unless-stopped`.
- **Resource limits** — cap memory/CPU with `--memory` / `--cpus`
  (or `deploy.resources` in Compose).
- **HTTPS** — terminate TLS at a reverse proxy (nginx, Caddy, Traefik). See the
  reverse-proxy section of
  [LINUX_INSTALLATION_GUIDE.md](./LINUX_INSTALLATION_GUIDE.md).
- **Security audit** — run `make env-security-check` to audit `.env` for missing
  authentication, weak secrets, and unsafe settings before exposing the server.

---

## Verifying the signed image

Release images are signed with **Sigstore Cosign** using keyless GitHub OIDC
signing. Install [`cosign`](https://docs.sigstore.dev/cosign/installation/), then
verify the tag you intend to run:

```bash
cosign verify ghcr.io/apkdmg/lightrag:v1.5.0.1 \
  --certificate-identity-regexp '^https://github.com/apkdmg/LightRAG/\.github/workflows/.+\.yml@refs/.+$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

A successful verification prints the signing certificate details and confirms the
image was built by this repository's GitHub Actions.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **Port 9621 already in use** | Map a different host port: `-p 8621:9621` (or set `PORT` in `.env` for Compose). |
| **Container exits immediately** | Check `docker logs lightrag` — usually a missing `TOKEN_SECRET` or invalid LLM/embedding config. |
| **Can't reach a host LLM/embedding server** | Use `host.docker.internal` (Docker) or `host.containers.internal` (Podman), and ensure the host mapping (`--add-host` / Compose `extra_hosts`) is present. |
| **Permission denied on `data/`** | `chown -R` the host `data/` directory to the running user, or switch to named volumes. |
| **Embedding dimension mismatch** | `EMBEDDING_DIM` must match the model and stay constant. Changing embedding models requires wiping vector storage. |
| **OAuth2 startup warning** | OAuth2 is enabled by default — either configure the `OAUTH2_*` variables or set `OAUTH2_ENABLED=false`. |
| **PDF / Docling parsing unavailable** | The image excludes `torch` / `transformers` to stay slim. Route heavy document parsing to an external parser service. |
| **WebUI loads but API calls fail** | Confirm the LLM/embedding providers in `.env` are reachable from inside the container, not just the host. |

---

## Next steps

- [LINUX_INSTALLATION_GUIDE.md](./LINUX_INSTALLATION_GUIDE.md) — non-container
  install and the full `.env` reference
- [KEYCLOAK_SSO_SETUP.md](./KEYCLOAK_SSO_SETUP.md) — OAuth2 / Keycloak setup
- [OAuth2-SSO-Authentication.md](./OAuth2-SSO-Authentication.md) — SSO architecture
- [LightRAG-API-Server.md](./LightRAG-API-Server.md) — API reference
- [DockerDeployment.md](./DockerDeployment.md) — vLLM, SSL, and storage-backend details
- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) — integrating with external systems
