# LightRAG Test Suite

This directory contains the test suite for LightRAG, designed for CI/CD integration.

## Test Structure

```
tests/
├── conftest.py                 # Shared pytest fixtures
├── pytest.ini                  # Pytest configuration (deprecated, use pyproject.toml)
├── unit/                       # Unit tests (fast, no external dependencies)
│   ├── test_chunking.py       # Text chunking tests
│   ├── test_query_param.py    # QueryParam validation tests
│   └── test_utils.py          # Utility function tests
├── integration/                # Integration tests
│   ├── test_storage_graph.py  # Graph storage tests
│   ├── test_storage_kv.py     # KV storage tests
│   ├── test_storage_vector.py # Vector storage tests
│   └── test_lightrag_core.py  # LightRAG class tests
├── api/                        # API endpoint tests
│   ├── test_query_routes.py   # Query API tests
│   ├── test_document_routes.py # Document API tests
│   └── test_graph_routes.py   # Graph API tests
└── fixtures/                   # Test data
    └── sample_documents/       # Sample documents for testing
```

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
# or for full development setup
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/ -m unit

# Integration tests only
pytest tests/integration/ -m integration

# API tests only
pytest tests/api/ -m api
```

### Run with Coverage

```bash
pytest --cov=lightrag --cov-report=html --cov-report=term-missing
```

### Run in Parallel (requires pytest-xdist)

```bash
pip install pytest-xdist
pytest -n auto
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Fast unit tests with no external dependencies
- `@pytest.mark.integration` - Tests requiring storage backend setup
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.requires_llm` - Tests requiring real LLM API calls
- `@pytest.mark.requires_db` - Tests requiring database connections

## Writing New Tests

### Unit Tests

Unit tests should:
- Test a single function or class
- Be fast (< 1 second)
- Not require external services
- Use mocks for dependencies

Example:
```python
@pytest.mark.unit
def test_chunking_basic(tokenizer):
    chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content="Test content",
        max_token_size=100,
    )
    assert len(chunks) > 0
```

### Integration Tests

Integration tests should:
- Test component interactions
- Use fixtures from conftest.py
- Clean up after themselves

Example:
```python
@pytest.mark.integration
async def test_storage_upsert(storage):
    await storage.upsert_node("test", {"data": "value"})
    result = await storage.get_node("test")
    assert result is not None
```

### API Tests

API tests should:
- Use FastAPI TestClient
- Mock external services
- Test both success and error cases

Example:
```python
@pytest.mark.api
def test_query_endpoint(client):
    response = client.post("/query/data", json={"query": "test"})
    assert response.status_code in [200, 401]
```

## CI/CD Integration

The test suite is designed to run in GitHub Actions. See `.github/workflows/test.yml` for the CI configuration.

### Test Matrix

- Python versions: 3.10, 3.11, 3.12
- Storage backends: Default (JSON/NanoVectorDB/NetworkX)

### Coverage Requirements

- Unit tests: Should cover core utility functions
- Integration tests: Should cover storage CRUD operations
- API tests: Should cover all major endpoints

## Fixtures

Key fixtures available in `conftest.py`:

- `mock_embedding_func` - Mock embedding function
- `mock_llm_func` - Mock LLM function
- `temp_working_dir` - Temporary directory for test data
- `global_config` - Storage configuration
- `networkx_storage` - NetworkX graph storage instance
- `json_kv_storage` - JSON KV storage instance
- `nano_vector_storage` - NanoVectorDB storage instance
- `lightrag_instance` - Full LightRAG instance
- `sample_text` - Sample document text
- `sample_documents` - Multiple sample documents

## Troubleshooting

### Tests fail with import errors

Make sure you've installed the package in development mode:
```bash
pip install -e ".[api,test]"
```

### Async tests hang

Ensure you're using the correct asyncio mode in pytest.ini or pyproject.toml:
```toml
asyncio_mode = "auto"
```

### Storage tests fail

Check that no previous test data is interfering. Tests should use `temp_working_dir` fixture.
