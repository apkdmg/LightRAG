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

For production with **PostgreSQL / Neo4j / Milvus**, use the full Compose file
(`docker-compose-full.yml`). It also bundles local vLLM embedding/rerank services
and requires an NVIDIA GPU host for Milvus and vLLM:

```bash
docker compose -f docker-compose-full.yml up -d
```

To generate a stack tailored to your chosen backends, use the interactive setup
wizard, which produces a `docker-compose.final.yml`:

```bash
make env-base       # LLM, embedding, reranker
make env-storage    # storage backends and database services
docker compose -f docker-compose.final.yml up -d
```

See [DockerDeployment.md](./DockerDeployment.md) for storage-backend, vLLM, SSL,
and PostgreSQL-image details, and [InteractiveSetup.md](./InteractiveSetup.md) for
the wizard.

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
