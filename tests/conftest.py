"""
Shared pytest fixtures for LightRAG tests.

This module provides common fixtures used across all test modules including:
- Mock LLM and embedding functions
- Storage backend fixtures
- API test client fixtures
- Sample test data
"""

import asyncio
import os
import sys
import tempfile
import shutil
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Event Loop Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Functions
# ============================================================================


@pytest.fixture
def mock_embedding_func():
    """Mock embedding function that returns deterministic vectors."""

    async def _embedding_func(texts: list[str]) -> np.ndarray:
        """Return deterministic embeddings based on text content."""
        # Generate reproducible embeddings based on text hash
        embeddings = []
        for text in texts:
            # Use hash to generate reproducible random seed
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.rand(384).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)

    return _embedding_func


@pytest.fixture
def mock_llm_func():
    """Mock LLM function for testing without API calls."""

    async def _llm_func(prompt: str, **kwargs) -> str:
        """Return mock LLM responses based on prompt content."""
        if "extract" in prompt.lower() and "entities" in prompt.lower():
            # Entity extraction response
            return """
            ("entity"<|>"LIGHTRAG"<|>"SOFTWARE"<|>"LightRAG is a retrieval-augmented generation framework")
            ##
            ("entity"<|>"KNOWLEDGE GRAPH"<|>"CONCEPT"<|>"A graph structure representing knowledge relationships")
            ##
            ("relationship"<|>"LIGHTRAG"<|>"KNOWLEDGE GRAPH"<|>"uses"<|>"LightRAG uses knowledge graphs for retrieval"<|>8)
            """
        elif "keywords" in prompt.lower():
            # Keyword extraction response
            return '{"high_level_keywords": ["RAG", "knowledge graph"], "low_level_keywords": ["entity", "retrieval"]}'
        else:
            # Default response
            return "This is a mock LLM response for testing purposes."

    return _llm_func


@pytest.fixture
def mock_llm_func_sync():
    """Synchronous mock LLM function."""

    def _llm_func(prompt: str, **kwargs) -> str:
        if "extract" in prompt.lower() and "entities" in prompt.lower():
            return """
            ("entity"<|>"LIGHTRAG"<|>"SOFTWARE"<|>"LightRAG is a retrieval-augmented generation framework")
            ##
            ("relationship"<|>"LIGHTRAG"<|>"KNOWLEDGE GRAPH"<|>"uses"<|>"LightRAG uses knowledge graphs"<|>8)
            """
        return "Mock response"

    return _llm_func


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_working_dir() -> Generator[str, None, None]:
    """Create a temporary working directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="lightrag_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_input_dir(temp_working_dir: str) -> str:
    """Create a temporary input directory."""
    input_dir = os.path.join(temp_working_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)
    return input_dir


# ============================================================================
# Sample Test Data
# ============================================================================


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing document processing."""
    return """
    LightRAG is a simple and fast retrieval-augmented generation framework.
    It uses knowledge graphs to enhance the retrieval process.
    The framework supports multiple storage backends including NetworkX, Neo4J, and PostgreSQL.

    Key features of LightRAG include:
    1. Graph-based knowledge representation
    2. Multiple query modes (local, global, hybrid, naive)
    3. Streaming responses
    4. Multi-tenant support

    LightRAG was developed to provide an efficient alternative to traditional RAG systems
    by leveraging the structure of knowledge graphs for better context retrieval.
    """


@pytest.fixture
def sample_documents() -> list[str]:
    """Return multiple sample documents for batch testing."""
    return [
        "Document 1: LightRAG is a RAG framework that uses knowledge graphs.",
        "Document 2: The framework supports multiple storage backends.",
        "Document 3: Query modes include local, global, hybrid, and naive.",
        "Document 4: Authentication supports JWT tokens and API keys.",
        "Document 5: The API server is built with FastAPI.",
    ]


@pytest.fixture
def sample_entities() -> list[dict]:
    """Return sample entity data for graph testing."""
    return [
        {
            "entity_id": "LIGHTRAG",
            "description": "A retrieval-augmented generation framework",
            "entity_type": "SOFTWARE",
            "source_id": "chunk_1",
        },
        {
            "entity_id": "KNOWLEDGE_GRAPH",
            "description": "A graph structure for representing knowledge",
            "entity_type": "CONCEPT",
            "source_id": "chunk_1",
        },
        {
            "entity_id": "FASTAPI",
            "description": "A modern Python web framework",
            "entity_type": "SOFTWARE",
            "source_id": "chunk_2",
        },
    ]


@pytest.fixture
def sample_edges() -> list[dict]:
    """Return sample edge data for graph testing."""
    return [
        {
            "src_id": "LIGHTRAG",
            "tgt_id": "KNOWLEDGE_GRAPH",
            "relationship": "uses",
            "description": "LightRAG uses knowledge graphs for retrieval",
            "weight": 1.0,
            "source_id": "chunk_1",
        },
        {
            "src_id": "LIGHTRAG",
            "tgt_id": "FASTAPI",
            "relationship": "built_with",
            "description": "LightRAG API is built with FastAPI",
            "weight": 0.8,
            "source_id": "chunk_2",
        },
    ]


# ============================================================================
# Storage Fixtures
# ============================================================================


@pytest.fixture
def global_config(temp_working_dir: str, mock_embedding_func) -> dict:
    """Return global configuration for storage initialization."""
    return {
        "working_dir": temp_working_dir,
        "embedding_batch_num": 10,
        "embedding_func": mock_embedding_func,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5,
        },
    }


@pytest.fixture
async def networkx_storage(global_config, mock_embedding_func):
    """Create a NetworkX graph storage instance for testing."""
    from lightrag.kg.networkx_impl import NetworkXStorage
    from lightrag.kg.shared_storage import initialize_share_data

    # Initialize shared storage for NetworkX
    initialize_share_data()

    storage = NetworkXStorage(
        namespace="test",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )
    await storage.initialize()
    yield storage
    await storage.finalize()


@pytest.fixture
async def json_kv_storage(global_config, mock_embedding_func):
    """Create a JSON KV storage instance for testing."""
    from lightrag.kg.json_kv_impl import JsonKVStorage

    storage = JsonKVStorage(
        namespace="test",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )
    await storage.initialize()
    yield storage
    await storage.finalize()


@pytest.fixture
async def nano_vector_storage(global_config, mock_embedding_func):
    """Create a NanoVectorDB storage instance for testing."""
    from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage

    storage = NanoVectorDBStorage(
        namespace="test",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )
    await storage.initialize()
    yield storage
    await storage.finalize()


# ============================================================================
# LightRAG Instance Fixtures
# ============================================================================


@pytest.fixture
async def lightrag_instance(
    temp_working_dir: str, mock_embedding_func, mock_llm_func
) -> AsyncGenerator:
    """Create a LightRAG instance for testing."""
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=temp_working_dir,
        llm_model_func=mock_llm_func,
        embedding_func=mock_embedding_func,
    )
    await rag.initialize_storages()
    yield rag
    await rag.finalize_storages()


# ============================================================================
# API Test Fixtures
# ============================================================================


@pytest.fixture
def mock_env_vars(temp_working_dir: str):
    """Set up mock environment variables for API testing."""
    env_vars = {
        "WORKING_DIR": temp_working_dir,
        "INPUT_DIR": os.path.join(temp_working_dir, "inputs"),
        "HOST": "127.0.0.1",
        "PORT": "9621",
        "LLM_BINDING": "openai",
        "LLM_MODEL_NAME": "gpt-3.5-turbo",
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_MODEL_NAME": "text-embedding-3-small",
        "AUTH_ACCOUNTS": "test_user:test_password",
        "TOKEN_SECRET": "test_secret_key_for_testing_only",
        "LIGHTRAG_API_KEY": "test-api-key-12345",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def api_test_client(mock_env_vars, mock_llm_func, mock_embedding_func):
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient

    # We need to patch the LLM and embedding functions before importing the app
    with patch("lightrag.llm.openai.openai_complete_if_cache", mock_llm_func):
        with patch("lightrag.llm.openai.openai_embed", mock_embedding_func):
            # Import here to use patched functions
            from lightrag.api.lightrag_server import create_app

            app = create_app()
            client = TestClient(app)
            yield client


@pytest.fixture
def auth_headers() -> dict:
    """Return authentication headers for API testing."""
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-api-key-12345",
    }


# ============================================================================
# Query Parameter Fixtures
# ============================================================================


@pytest.fixture
def default_query_param():
    """Return default query parameters."""
    from lightrag import QueryParam

    return QueryParam(
        mode="hybrid",
        top_k=10,
        max_token_for_text_unit=4000,
        max_token_for_global_context=4000,
        max_token_for_local_context=4000,
    )


@pytest.fixture
def local_query_param():
    """Return local mode query parameters."""
    from lightrag import QueryParam

    return QueryParam(mode="local", top_k=5)


@pytest.fixture
def global_query_param():
    """Return global mode query parameters."""
    from lightrag import QueryParam

    return QueryParam(mode="global", top_k=5)


# ============================================================================
# Markers Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "api: mark test as an API test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring real LLM API"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database connection"
    )
