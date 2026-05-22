# LightRAG WebUI Walkthrough

A first-use guide to the LightRAG web interface — log in, upload a document,
watch it index, run your first query, and explore the knowledge graph.

> **Prerequisite:** a running LightRAG server. See the
> [Linux Installation Guide](./LINUX_INSTALLATION_GUIDE.md), the
> [Docker Installation Guide](./DOCKER_INSTALLATION_GUIDE.md), or the
> "Try It in 2 Minutes" section of the [main README](../README.md).

## 1. Open the WebUI

Browse to your server — by default **http://localhost:9621**. The WebUI is
served by the same process as the API, so no separate frontend deployment is
needed.

## 2. Log in

How you log in depends on how the server is configured:

| Mode | When it applies | How to log in |
|------|-----------------|---------------|
| **OAuth2 / SSO** | `OAUTH2_ENABLED=true` (default) with Keycloak configured | Click **Sign in with SSO**, authenticate with Keycloak, and you are redirected back |
| **Username / password** | `AUTH_ACCOUNTS` is set (e.g. `admin:secret`) | Enter the username and password |
| **Guest** | `AUTH_ACCOUNTS` unset *and* SSO not configured | The WebUI opens directly — no login screen |

- Users listed in `ADMIN_ACCOUNTS` receive the **admin** role, which adds
  workspace-administration capabilities.
- With multi-tenancy enabled (the default), each user works in an **isolated
  workspace** — your documents and knowledge graph are private to you.

To configure SSO see [KEYCLOAK_SSO_SETUP.md](./KEYCLOAK_SSO_SETUP.md); to set
username/password accounts see the [installation guides](./README.md).

## 3. Upload a document — the Documents tab

1. Open the **Documents** tab.
2. Add a file — drag-and-drop, or use the upload control. Supported inputs
   include plain text, Markdown, PDF, Office documents, and `.eml` emails.
   Native DOCX parsing is built in; image/table extraction from PDFs requires
   an external parser (see [DockerDeployment.md](./DockerDeployment.md)).
3. Alternatively, place files in the server's input directory
   (`data/inputs`) and use **Scan** to ingest everything found there.

To ingest emails specifically, see [EMAIL_INGESTION.md](./EMAIL_INGESTION.md).

## 4. Watch it index

Indexing is asynchronous. Each document moves through a status lifecycle shown
in the Documents tab:

| Status | Meaning |
|--------|---------|
| `pending` | Queued, waiting to be processed |
| `analyzing` | Multimodal (VLM) analysis of images and tables — when enabled |
| `processing` | Entity and relationship extraction into the knowledge graph |
| `processed` | Fully indexed and ready to query |
| `failed` | Processing failed — check the server logs |

Large documents take longer. Wait for `processed` before querying for the best
results.

## 5. Run your first query — the Retrieval tab

1. Open the **Retrieval** tab.
2. Type a question about the document you uploaded.
3. Choose a **retrieval mode**:

| Mode | Best for |
|------|----------|
| `naive` | Plain vector search, no graph |
| `local` | Entity-centric questions — about specific things |
| `global` | Theme-centric questions — broad or summary-level |
| `hybrid` | Combines local and global retrieval |
| `mix` | Graph **and** vector retrieval combined — the recommended default |

4. Submit. LightRAG retrieves context from the knowledge graph and the vector
   store, then generates an answer.

The same queries are available over the REST API and the
[OpenAI-compatible API](./OPENAI_COMPATIBLE_API.md).

## 6. Explore the knowledge graph — the Knowledge Graph tab

The **Knowledge Graph** tab visualizes the entities and relationships LightRAG
extracted from your documents. You can pan and zoom, search for specific nodes,
and filter subgraphs to see how your knowledge has been structured.

## Next steps

- [Integration Guide](./INTEGRATION_GUIDE.md) — drive LightRAG from your own
  applications, scripts, and automation tools
- [Per-User API Keys](./PER_USER_API_KEYS.md) — mint keys for programmatic access
- [OpenAI-Compatible API](./OPENAI_COMPATIBLE_API.md) — use LightRAG as a
  drop-in OpenAI-style endpoint
- [LightRAG-API-Server.md](./LightRAG-API-Server.md) — full API and server reference
