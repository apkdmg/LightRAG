"""
Tests for Milvus query() search behavior

Validates that vector queries never use server-side range search ("radius"),
which is not implemented by GPU indexes (AUTOINDEX resolves to the CAGRA
family on milvus-gpu images), and that the cosine similarity threshold is
applied client-side instead.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from lightrag.kg.milvus_impl import MilvusVectorDBStorage


def _make_storage(threshold: float = 0.2) -> MilvusVectorDBStorage:
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128

    storage = MilvusVectorDBStorage(
        namespace="test_entities",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": threshold,
                "index_type": "AUTOINDEX",
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields=set(),
    )
    storage._client = MagicMock()
    storage.final_namespace = "test_entities"
    return storage


@pytest.mark.offline
class TestMilvusQuerySearch:
    def test_query_does_not_use_range_search(self):
        """query() must not pass "radius" — GPU indexes reject RangeSearch"""
        storage = _make_storage(threshold=0.2)
        storage._client.search.return_value = [[]]

        with patch.object(storage, "_ensure_collection_loaded"):
            asyncio.run(storage.query("q", top_k=5, query_embedding=[0.1] * 128))

        search_kwargs = storage._client.search.call_args.kwargs
        assert "radius" not in search_kwargs["search_params"].get("params", {})
        assert search_kwargs["limit"] == 5

    def test_query_applies_threshold_client_side(self):
        """Hits at or below cosine_better_than_threshold are dropped"""
        storage = _make_storage(threshold=0.2)
        storage._client.search.return_value = [
            [
                {"id": "hit-above", "distance": 0.9, "entity": {}},
                {"id": "hit-at-threshold", "distance": 0.2, "entity": {}},
                {"id": "hit-below", "distance": 0.1, "entity": {}},
            ]
        ]

        with patch.object(storage, "_ensure_collection_loaded"):
            results = asyncio.run(
                storage.query("q", top_k=5, query_embedding=[0.1] * 128)
            )

        # Range-search semantics for similarity metrics are score > radius
        assert [r["id"] for r in results] == ["hit-above"]
