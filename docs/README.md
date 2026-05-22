# LightRAG Documentation

Documentation for the **LightRAG enterprise server** — a graph-based RAG
framework extended with multi-tenancy, OAuth2/Keycloak SSO, per-user API keys,
an OpenAI-compatible API, and email ingestion.

> Main project README: [`../README.md`](../README.md) · Configuration
> reference: the annotated [`../env.example`](../env.example)

## New here? Start with these

1. Install a server — **[Linux Installation Guide](./LINUX_INSTALLATION_GUIDE.md)**
   or **[Docker Installation Guide](./DOCKER_INSTALLATION_GUIDE.md)**
2. **[WebUI Walkthrough](./WEBUI_WALKTHROUGH.md)** — log in, upload a document,
   and run your first query
3. **[Integration Guide](./INTEGRATION_GUIDE.md)** — connect your own
   applications, services, and tools

---

## Installation & Deployment

| Document | What it covers |
|----------|----------------|
| [LINUX_INSTALLATION_GUIDE.md](./LINUX_INSTALLATION_GUIDE.md) | Step-by-step server install on Linux — source and Docker, with PostgreSQL / Neo4j |
| [DOCKER_INSTALLATION_GUIDE.md](./DOCKER_INSTALLATION_GUIDE.md) | Comprehensive Docker / Docker Compose / Podman installation |
| [DockerDeployment.md](./DockerDeployment.md) | Deeper Docker topics — vLLM, SSL, storage-backend images |
| [OfflineDeployment.md](./OfflineDeployment.md) | Air-gapped / offline installation |
| [MultiSiteDeployment.md](./MultiSiteDeployment.md) | Running multiple isolated LightRAG sites |
| [InteractiveSetup.md](./InteractiveSetup.md) | The `make env-*` setup wizard |
| [FrontendBuildGuide.md](./FrontendBuildGuide.md) | Building the WebUI frontend |
| Kubernetes | See [`../k8s-deploy/README.md`](../k8s-deploy/README.md) |

## Enterprise Features

| Document | What it covers |
|----------|----------------|
| [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) | **Master integration guide** — authentication, multi-tenancy, the REST API, end-to-end patterns |
| [KEYCLOAK_SSO_SETUP.md](./KEYCLOAK_SSO_SETUP.md) | Setting up OAuth2 / Keycloak SSO |
| [OAuth2-SSO-Authentication.md](./OAuth2-SSO-Authentication.md) | SSO architecture, token flows, and automation |
| [PER_USER_API_KEYS.md](./PER_USER_API_KEYS.md) | Per-user API keys (`sk-lightrag-…`) |
| [OPENAI_COMPATIBLE_API.md](./OPENAI_COMPATIBLE_API.md) | The OpenAI-compatible `/v1` API |
| [EMAIL_INGESTION.md](./EMAIL_INGESTION.md) | Ingesting `.eml` emails into the knowledge graph |

## Using LightRAG

| Document | What it covers |
|----------|----------------|
| [WEBUI_WALKTHROUGH.md](./WEBUI_WALKTHROUGH.md) | First-use walkthrough of the web interface — login, upload, query |
| [LightRAG-API-Server.md](./LightRAG-API-Server.md) | API server & WebUI reference *(also in [中文](./LightRAG-API-Server-zh.md))* |
| [ProgramingWithCore.md](./ProgramingWithCore.md) | Using the LightRAG core library directly in Python |
| [AdvancedFeatures.md](./AdvancedFeatures.md) | Advanced features and options |

## Configuration & Tuning

| Document | What it covers |
|----------|----------------|
| [RoleSpecificLLMConfiguration.md](./RoleSpecificLLMConfiguration.md) | Per-role LLM config — extract / query / keyword / VLM *(also in [中文](./RoleSpecificLLMConfiguration-zh.md))* |
| [AsymmetricEmbedding.md](./AsymmetricEmbedding.md) | Asymmetric query/document embeddings |
| [ParagraphSemanticChunking.md](./ParagraphSemanticChunking.md) | Text chunking strategies *(also in [中文](./ParagraphSemanticChunking-zh.md))* |
| [MilvusConfigurationGuide.md](./MilvusConfigurationGuide.md) | Milvus vector-storage configuration |

## Document Processing Internals

| Document | What it covers |
|----------|----------------|
| [FileProcessingPipeline.md](./FileProcessingPipeline.md) | How documents are parsed and processed *(also in [中文](./FileProcessingPipeline-zh.md))* |
| [LightRAGSidecarFormat.md](./LightRAGSidecarFormat.md) | The LightRAG sidecar file format *(also in [中文](./LightRAGSidecarFormat-zh.md))* |
| [ParserDebugCLI.md](./ParserDebugCLI.md) | The parser debugging CLI *(also in [中文](./ParserDebugCLI-zh.md))* |
| [Algorithm.md](./Algorithm.md) | Algorithm notes |

## Maintenance & Reference

| Document | What it covers |
|----------|----------------|
| [UV_LOCK_GUIDE.md](./UV_LOCK_GUIDE.md) | Managing the `uv.lock` dependency lockfile |
| [Reproduce.md](./Reproduce.md) | Reproducing the LightRAG paper's results |

## Project History (internal)

These record past internal work and are not needed to install or use the project:

| Document | What it covers |
|----------|----------------|
| [MIGRATION_TO_1.5.0.md](./MIGRATION_TO_1.5.0.md) | Record of the migration onto upstream LightRAG 1.5.0 |
| [Hybrid-Token-Authentication-Implementation-Plan.md](./Hybrid-Token-Authentication-Implementation-Plan.md) | Historical implementation plan for hybrid-token auth (now shipped) |

---

Several documents have Chinese (`-zh.md`) translations — look for the *中文*
links above. Contributing and security policy: [`../.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md)
· [`../SECURITY.md`](../SECURITY.md).
