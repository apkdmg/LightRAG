"""
Tests for the Milvus vector index health check

Validates the self-search probe that runs at collection initialization: a
healthy index finds a stored vector at score ~1.0; a broken one (e.g. GPU
CAGRA built with the unsupported COSINE metric) returns garbage scores and
must be reported loudly.
"""

from unittest.mock import MagicMock, patch

import pytest

from lightrag.kg.milvus_impl import MilvusVectorDBStorage


def _make_storage() -> MilvusVectorDBStorage:
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128

    storage = MilvusVectorDBStorage(
        namespace="test_entities",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2,
                "index_type": "AUTOINDEX",
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields=set(),
    )
    storage._client = MagicMock()
    storage.final_namespace = "test_entities"
    return storage


def _run_health_check(storage, query_rows, search_result):
    storage._client.query.return_value = query_rows
    storage._client.search.return_value = search_result
    with patch.object(storage, "_ensure_collection_loaded"):
        storage._verify_index_health()


@pytest.mark.offline
class TestMilvusIndexHealthCheck:
    def test_healthy_index_logs_no_error(self):
        storage = _make_storage()
        with patch("lightrag.kg.milvus_impl.logger") as mock_logger:
            _run_health_check(
                storage,
                query_rows=[{"vector": [0.1] * 128}],
                search_result=[[{"id": "x", "distance": 1.0}]],
            )
        mock_logger.error.assert_not_called()

    def test_broken_index_logs_error(self):
        """Self-search score near zero (GPU CAGRA + COSINE) must be reported"""
        storage = _make_storage()
        with patch("lightrag.kg.milvus_impl.logger") as mock_logger:
            _run_health_check(
                storage,
                query_rows=[{"vector": [0.1] * 128}],
                search_result=[[{"id": "x", "distance": -0.0}]],
            )
        mock_logger.error.assert_called_once()
        msg = mock_logger.error.call_args.args[0]
        assert "health check FAILED" in msg
        assert "MILVUS_INDEX_TYPE=HNSW" in msg

    def test_empty_collection_skips_search(self):
        storage = _make_storage()
        _run_health_check(storage, query_rows=[], search_result=None)
        storage._client.search.assert_not_called()

    def test_disabled_via_env_skips_entirely(self, monkeypatch):
        monkeypatch.setenv("MILVUS_INDEX_HEALTH_CHECK", "false")
        storage = _make_storage()
        with patch.object(storage, "_ensure_collection_loaded"):
            storage._verify_index_health()
        storage._client.query.assert_not_called()

    def test_non_cosine_metric_skips(self):
        storage = _make_storage()
        storage.index_config.metric_type = "IP"
        with patch.object(storage, "_ensure_collection_loaded"):
            storage._verify_index_health()
        storage._client.query.assert_not_called()

    def test_probe_exception_does_not_raise(self):
        """A failing probe must never block storage initialization"""
        storage = _make_storage()
        storage._client.query.side_effect = RuntimeError("connection lost")
        with patch("lightrag.kg.milvus_impl.logger") as mock_logger:
            with patch.object(storage, "_ensure_collection_loaded"):
                storage._verify_index_health()
        mock_logger.error.assert_not_called()
