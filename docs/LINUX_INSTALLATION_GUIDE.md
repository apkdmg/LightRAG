# LightRAG Linux Server Installation Guide

A step-by-step guide to install and configure the LightRAG enterprise server
(LightRAG 1.5.0, branch `main`) on a Linux server. Multimodal
document processing is built in natively; OAuth2/Keycloak SSO and multi-tenant
workspace isolation are enabled by default.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
   - [Method 1: Docker (Recommended)](#method-1-docker-recommended)
   - [Method 2: Manual Installation](#method-2-manual-installation)
4. [Database Setup](#database-setup)
5. [Configuration](#configuration)
6. [Running the Server](#running-the-server)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Minimum Version | Purpose |
|----------|-----------------|---------|
| Python | 3.10+ | Runtime |
| pip | Latest | Package management |
| Git | 2.x | Clone repository |
| Docker | 20.x (optional) | Container deployment |
| Docker Compose | 2.x (optional) | Orchestration |

### External Services Required

1. **LLM API** (at least one):
   - OpenAI API
   - Ollama (local)
   - Azure OpenAI
   - AWS Bedrock
   - Any OpenAI-compatible API

2. **Embedding Model API** (at least one):
   - OpenAI Embeddings
   - Ollama Embeddings
   - Jina Embeddings
   - Azure OpenAI Embeddings

3. **Database** (optional, for production):
   - PostgreSQL 15+ with pgvector extension
   - Neo4j 5.x (recommended for graph storage)
   - Redis (for KV caching)

---

## System Requirements

### Minimum Requirements

| Resource | Specification |
|----------|---------------|
| CPU | 4 cores |
| RAM | 8 GB |
| Storage | 50 GB SSD |
| Network | Stable internet for API calls |

### Recommended for Production

| Resource | Specification |
|----------|---------------|
| CPU | 8+ cores |
| RAM | 32 GB |
| Storage | 200 GB+ NVMe SSD |
| GPU | Optional (for local LLM/embedding) |

---

## Installation Methods

### Method 1: Docker (Recommended)

Docker provides the easiest deployment with all dependencies bundled.

#### Step 1: Install Docker

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/apkdmg/LightRAG.git
cd LightRAG
```

#### Step 3: Configure Environment

```bash
# Copy example environment file
cp env.example .env

# Edit configuration
nano .env
```

**Minimal .env configuration:**

```bash
# Server
HOST=0.0.0.0
PORT=9621

# LLM Configuration (OpenAI example)
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=sk-your-api-key-here

# Embedding Configuration
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=sk-your-api-key-here

# Authentication — TOKEN_SECRET is required (it signs every JWT)
TOKEN_SECRET=your-random-secret-key-min-32-chars
LIGHTRAG_API_KEY=your-api-key-for-programmatic-access
# Username/password accounts (optional — omit for SSO-only or guest access)
AUTH_ACCOUNTS=admin:your-secure-password
ADMIN_ACCOUNTS=admin

# OAuth2 / Keycloak SSO is ENABLED BY DEFAULT — supply real credentials,
# or set OAUTH2_ENABLED=false to use username/password login only.
# These two lines assume the default UNIMAS realm. For any other Keycloak,
# you must also set the OAUTH2_ISSUER / *_ENDPOINT / *_JWKS_URI vars — see
# the full OAuth2 block in the .env reference below (they do NOT auto-derive).
OAUTH2_CLIENT_ID=lightrag
OAUTH2_CLIENT_SECRET=your-keycloak-client-secret
```

#### Step 4: Create Data Directories

```bash
mkdir -p data/rag_storage data/inputs data/prompts
chmod -R 755 data/
```

#### Step 5: Start with Docker Compose

```bash
# Start in detached mode
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

#### Step 6: Verify Installation

```bash
# Health check
curl http://localhost:9621/health

# Check authentication / SSO status
curl http://localhost:9621/auth-status
```

---

### Method 2: Manual Installation

For more control or when Docker is not available.

#### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    build-essential \
    libpq-dev \
    curl

# CentOS/RHEL/Rocky
sudo dnf install -y \
    python3.11 \
    python3.11-devel \
    git \
    gcc \
    postgresql-devel \
    curl
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/apkdmg/LightRAG.git
cd LightRAG
```

#### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 4: Install LightRAG and Build the WebUI

```bash
# Install core + API dependencies
pip install -e ".[api]"
# (upstream's recommended alternative: `uv sync --extra api`)
```

Multimodal document processing is **built into LightRAG 1.5.0** — no extra
package is required (RAGAnything is no longer used). Native DOCX parsing works
out of the box; for image/table extraction from PDFs, configure an external
parser (`mineru` or `docling`) via the `LIGHTRAG_PARSER` setting.

The WebUI's built assets are not committed to git, so build them once
([Bun](https://bun.sh) is required):

```bash
cd lightrag_webui
bun install --frozen-lockfile
bun run build          # outputs to lightrag/api/webui/
cd ..
```

#### Step 5: Configure Environment

```bash
cp env.example .env
nano .env
```

**Edit the .env file** with your API keys and settings (see Docker method above for example).

#### Step 6: Create Data Directories

```bash
mkdir -p data/rag_storage data/inputs data/prompts
```

#### Step 7: Start the Server

```bash
# Activate virtual environment if not already
source venv/bin/activate

# Start server
lightrag-server
```

---

## Database Setup

### Option A: Default (File-based, for Development)

No setup required. LightRAG uses:
- JsonKVStorage for documents
- NanoVectorDB for vectors
- NetworkX for graphs

### Option B: PostgreSQL (Recommended for Production)

#### Install PostgreSQL with pgvector

```bash
# Ubuntu/Debian
sudo apt install -y postgresql postgresql-contrib

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install pgvector extension
sudo apt install -y postgresql-15-pgvector
```

#### Create Database and User

```bash
sudo -u postgres psql << EOF
CREATE USER lightrag WITH PASSWORD 'your-secure-password';
CREATE DATABASE lightrag OWNER lightrag;
\c lightrag
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "\$user", public;
EOF
```

#### Configure LightRAG for PostgreSQL

LightRAG reads database connection details from a `config.ini` file in the
directory you start the server from. The `.env` file selects *which* storage
backend is active (the `*_STORAGE` variables below); `config.ini` holds the
*connection credentials* for it. Create `config.ini`:

```ini
[postgres]
host = localhost
port = 5432
user = lightrag
password = your-secure-password
database = lightrag
max_connections = 12
vector_index_type = HNSW
hnsw_m = 16
hnsw_ef = 64
```

Update `.env`:

```bash
# Storage configuration
KV_STORAGE=PGKVStorage
VECTOR_STORAGE=PGVectorStorage
GRAPH_STORAGE=PGGraphStorage
DOC_STATUS_STORAGE=PGDocStatusStorage
```

### Option C: Neo4j (Best for Graph Operations)

#### Install Neo4j

```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install -y neo4j

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

#### Configure Neo4j

Create `config.ini`:

```ini
[neo4j]
uri = bolt://localhost:7687
username = neo4j
password = your-neo4j-password
connection_pool_size = 100
connection_timeout = 30.0
```

Update `.env`:

```bash
GRAPH_STORAGE=Neo4JStorage
```

---

## Configuration

### Complete .env Reference

```bash
# ============================================================================
# SERVER CONFIGURATION
# ============================================================================
HOST=0.0.0.0
PORT=9621
WORKERS=2
TIMEOUT=150
SSL=false

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
# Binding options: openai, ollama, azure_openai, aws_bedrock, lollms
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=sk-your-key

# For Ollama (local LLM)
# LLM_BINDING=ollama
# LLM_MODEL=qwen2.5:32b
# LLM_BINDING_HOST=http://localhost:11434
# OLLAMA_LLM_NUM_CTX=32768

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
# WARNING: Cannot change after first document is indexed!
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=sk-your-key

# For Ollama embeddings
# EMBEDDING_BINDING=ollama
# EMBEDDING_MODEL=bge-m3
# EMBEDDING_DIM=1024
# EMBEDDING_BINDING_HOST=http://localhost:11434

# ============================================================================
# RERANKER CONFIGURATION (Optional)
# ============================================================================
# RERANK_BINDING=jina
# RERANK_MODEL=jina-reranker-v2-base-multilingual
# RERANK_BINDING_API_KEY=your-jina-key
# MIN_RERANK_SCORE=0.3

# ============================================================================
# VLM / MULTIMODAL (native — describes images, tables, equations)
# ============================================================================
# Enables VLM analysis of image/table/equation items in documents and the
# vision description of inline images in ingested emails.
# VLM_PROCESS_ENABLE=true
# VLM_LLM_BINDING=openai
# VLM_LLM_MODEL=gpt-4o
# VLM_LLM_BINDING_HOST=https://api.openai.com/v1
# VLM_LLM_BINDING_API_KEY=sk-your-key

# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================
# Options: JsonKVStorage, PGKVStorage, RedisKVStorage, MongoKVStorage
KV_STORAGE=JsonKVStorage

# Options: NanoVectorDBStorage, PGVectorStorage, MilvusVectorDBStorage, etc.
VECTOR_STORAGE=NanoVectorDBStorage

# Options: NetworkXStorage, Neo4JStorage, PGGraphStorage, MemgraphStorage
GRAPH_STORAGE=NetworkXStorage

# Options: JsonDocStatusStorage, PGDocStatusStorage, MongoDocStatusStorage
DOC_STATUS_STORAGE=JsonDocStatusStorage

# ============================================================================
# AUTHENTICATION
# ============================================================================
# TOKEN_SECRET is REQUIRED — it signs every JWT (including SSO-issued tokens).
# With AUTH_ACCOUNTS set, the server refuses to start unless TOKEN_SECRET is
# changed from its default value.
TOKEN_SECRET=your-random-secret-key-at-least-32-characters-long
TOKEN_EXPIRE_HOURS=48
LIGHTRAG_API_KEY=your-api-key-for-programmatic-access

# Username/password accounts (optional — omit for SSO-only or guest access).
AUTH_ACCOUNTS=admin:secure-password,user1:user-password
# Usernames granted the admin role (comma-separated) — required for the
# multi-tenancy admin API and on-behalf-of operations.
ADMIN_ACCOUNTS=admin

# ============================================================================
# OAUTH2 / KEYCLOAK SSO  (ENABLED BY DEFAULT)
# ============================================================================
# Set OAUTH2_ENABLED=false to disable SSO and use password/guest login only.
OAUTH2_ENABLED=true

# Confidential-client credentials — required for the built-in SSO login flow
# (the /oauth2/authorize -> /oauth2/callback authorization-code exchange).
OAUTH2_CLIENT_ID=lightrag
OAUTH2_CLIENT_SECRET=your-keycloak-client-secret

# Realm endpoints. LightRAG does NOT perform OIDC discovery — these are NOT
# derived from OAUTH2_ISSUER. For a non-UNIMAS Keycloak you MUST override all
# five values below (swap host + realm consistently); otherwise they keep
# pointing at the UNIMAS defaults shown here.
OAUTH2_ISSUER=https://id.unimas.my/realms/UNIMAS
OAUTH2_AUTHORIZATION_ENDPOINT=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/auth
OAUTH2_TOKEN_ENDPOINT=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/token
OAUTH2_USERINFO_ENDPOINT=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/userinfo
OAUTH2_JWKS_URI=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/certs

# Must exactly match a "Valid Redirect URI" registered on the Keycloak client.
OAUTH2_REDIRECT_URI=http://your-host:9621/oauth2/callback
OAUTH2_SCOPES=openid profile email
# Full Keycloak configuration: see docs/KEYCLOAK_SSO_SETUP.md

# ============================================================================
# QUERY PARAMETERS
# ============================================================================
TOP_K=60
CHUNK_TOP_K=20
MAX_ENTITY_TOKENS=6000
MAX_RELATION_TOKENS=8000
MAX_TOTAL_TOKENS=30000
COSINE_THRESHOLD=0.2

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
SUMMARY_LANGUAGE=English
ENABLE_LLM_CACHE=true
ENABLE_LLM_CACHE_FOR_EXTRACT=true

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================
MAX_ASYNC=4
MAX_PARALLEL_INSERT=2
EMBEDDING_FUNC_MAX_ASYNC=16
EMBEDDING_BATCH_NUM=32

# ============================================================================
# MULTI-TENANCY  (ENABLED BY DEFAULT — per-user workspace isolation)
# ============================================================================
# Set ENABLE_MULTI_TENANCY=false for a single shared workspace.
ENABLE_MULTI_TENANCY=true
MAX_WORKSPACE_INSTANCES=10000
WORKSPACE_TTL_MINUTES=60
AUTO_CREATE_WORKSPACE=true

# ============================================================================
# DOCUMENT PARSING (native multimodal — RAGAnything is no longer used)
# ============================================================================
# Native DOCX parsing is built in. For PDF / image parsing, route to an
# external parser (mineru or docling) via LIGHTRAG_PARSER. See env.example
# for the parser-routing syntax and the MINERU_* / DOCLING_* settings.
# LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

---

## Running the Server

### Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Start server with auto-reload
lightrag-server --reload
```

### Production Mode

#### Using systemd

Create service file `/etc/systemd/system/lightrag.service`:

```ini
[Unit]
Description=LightRAG Server
After=network.target postgresql.service

[Service]
Type=simple
User=lightrag
Group=lightrag
WorkingDirectory=/opt/lightrag
Environment="PATH=/opt/lightrag/venv/bin"
EnvironmentFile=/opt/lightrag/.env
ExecStart=/opt/lightrag/venv/bin/lightrag-server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lightrag
sudo systemctl start lightrag
sudo systemctl status lightrag
```

#### Using Docker Compose (Production)

```bash
# Start in production mode
docker compose -f docker-compose.yml up -d

# Scale workers
docker compose up -d --scale lightrag=3
```

---

## Production Deployment

### Nginx Reverse Proxy

Install Nginx:

```bash
sudo apt install -y nginx
```

Create `/etc/nginx/sites-available/lightrag`:

```nginx
upstream lightrag {
    server 127.0.0.1:9621;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Increase timeouts for LLM operations
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;

    # Increase body size for document uploads
    client_max_body_size 100M;

    location / {
        proxy_pass http://lightrag;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/lightrag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL Certificate (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port
sudo lsof -i :9621

# Kill process
sudo kill -9 <PID>
```

#### 2. Permission Denied on Data Directory

```bash
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

#### 3. Docker Container Won't Start

```bash
# Check logs
docker compose logs lightrag

# Rebuild container
docker compose build --no-cache
docker compose up -d
```

#### 4. Database Connection Failed

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U lightrag -d lightrag -c "SELECT 1"
```

#### 5. Embedding Dimension Mismatch

If you see vector dimension errors after changing embedding models:

```bash
# For PostgreSQL, drop and recreate vector tables
psql -U lightrag -d lightrag << EOF
DROP TABLE IF EXISTS lightrag_vectors CASCADE;
EOF
```

**Warning:** This deletes all indexed documents. Re-index after fixing.

#### 6. Out of Memory

Reduce concurrent operations in `.env`:

```bash
MAX_ASYNC=2
MAX_PARALLEL_INSERT=1
EMBEDDING_FUNC_MAX_ASYNC=8
EMBEDDING_BATCH_NUM=16
```

### Logs Location

- **Docker:** `docker compose logs -f`
- **systemd:** `journalctl -u lightrag -f`
- **Manual:** Check console output or redirect to file

### Health Checks

```bash
# Server health
curl http://localhost:9621/health

# Authentication / SSO status
curl http://localhost:9621/auth-status

# API documentation
# Open in browser: http://localhost:9621/docs
```

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/auth-status` | GET | Authentication / SSO status |
| `/login` | POST | Username/password login (JWT) |
| `/oauth2/authorize` | GET | Start Keycloak SSO login |
| `/documents/upload` | POST | Upload a document |
| `/documents/scan` | POST | Scan the input directory |
| `/documents/email` | POST | Ingest an email (.eml or structured) |
| `/query` | POST | Query the knowledge base |
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |
| `/api-keys` | GET/POST | Per-user API key management |
| `/admin/workspaces` | GET | Multi-tenancy admin |
| `/graphs` | GET | Knowledge graph |
| `/docs` | GET | Swagger UI |

### Login & Credentials

There is **no built-in default user**. How login works depends on configuration:

- **OAuth2 / SSO (default):** users sign in via Keycloak ("Sign in with SSO").
  Requires `OAUTH2_CLIENT_ID` / `OAUTH2_CLIENT_SECRET`.
- **Username/password:** available only when `AUTH_ACCOUNTS` is set
  (e.g. `admin:<password>`).
- **Guest mode:** if `AUTH_ACCOUNTS` is unset *and* SSO is not configured, the
  WebUI enters as a guest with no login.
- **Admin role:** granted to the usernames listed in `ADMIN_ACCOUNTS`.
- **API key:** the shared `LIGHTRAG_API_KEY`, or per-user keys minted via `/api-keys`.

### Useful Commands

```bash
# Start server
lightrag-server

# Start with custom config
lightrag-server --host 0.0.0.0 --port 8080

# Docker commands
docker compose up -d          # Start
docker compose down           # Stop
docker compose logs -f        # View logs
docker compose restart        # Restart

# systemd commands
sudo systemctl start lightrag
sudo systemctl stop lightrag
sudo systemctl restart lightrag
sudo systemctl status lightrag
```

---

## Next Steps

1. **Test the installation** by uploading a sample document
2. **Configure authentication** for production use
3. **Set up monitoring** with Prometheus/Grafana
4. **Configure backups** for your database
5. **Review security** settings and firewall rules

For more information, see:
- [Main Documentation](../README.md)
- [API Server Guide](./LightRAG-API-Server.md)
- [Frontend Build Guide](./FrontendBuildGuide.md)
- [Docker Deployment](./DockerDeployment.md)
- [Kubernetes Deployment](../k8s-deploy/README.md)
