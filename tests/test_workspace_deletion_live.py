"""Live test: workspace deletion against Postgres, Neo4j and Milvus.

Two workspaces are filled through the real insert pipeline (with a scripted
LLM and a deterministic embedding), one is purged, and the databases are then
queried directly: nothing of the purged workspace may remain (rows, nodes,
indexes, collections) while the other workspace stays intact.

Runs only with ``LIGHTRAG_RUN_INTEGRATION=true`` and ``POSTGRES_HOST``,
``NEO4J_URI`` and ``MILVUS_URI`` set (plus their credentials).
"""

import asyncio
import os
import sys
import uuid

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

_REQUIRED_ENV = ("POSTGRES_HOST", "NEO4J_URI", "MILVUS_URI")

if os.getenv("LIGHTRAG_RUN_INTEGRATION", "false").lower() != "true" or not all(
    os.getenv(name) for name in _REQUIRED_ENV
):
    pytest.skip(
        "needs LIGHTRAG_RUN_INTEGRATION=true and " + ", ".join(_REQUIRED_ENV),
        allow_module_level=True,
    )

DIM = 8


async def _embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        rows.append(rng.random(DIM))
    return np.array(rows, dtype=np.float32)


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    text = f"{system_prompt or ''}\n{prompt}"
    if "entity_name" in text or "Entity row" in text:
        return (
            "entity<|#|>Zebra Project<|#|>organization<|#|>A private project.\n"
            "entity<|#|>Dr Nadia<|#|>person<|#|>Leads the Zebra Project.\n"
            "relation<|#|>Dr Nadia<|#|>Zebra Project<|#|>leads<|#|>"
            "Dr Nadia leads the Zebra Project.\n"
            "<|COMPLETE|>"
        )
    return "Summary."


def _build(workspace: str, working_dir: str):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    return LightRAG(
        working_dir=working_dir,
        workspace=workspace,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        vector_storage="MilvusVectorDBStorage",
        graph_storage="Neo4JStorage",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=DIM, func=_embed),
    )


async def _postgres_counts(workspace: str) -> dict[str, int]:
    import asyncpg

    conn = await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DATABASE"],
    )
    try:
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.columns "
            "WHERE column_name = 'workspace' AND table_schema = 'public'"
        )
        counts = {}
        for row in tables:
            name = row["table_name"]
            counts[name] = await conn.fetchval(
                f'SELECT count(*) FROM "{name}" WHERE workspace = $1', workspace
            )
        return counts
    finally:
        await conn.close()


async def _postgres_tombstones() -> int:
    import asyncpg

    conn = await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DATABASE"],
    )
    try:
        return await conn.fetchval("SELECT count(*) FROM lightrag_workspace_tombstones")
    finally:
        await conn.close()


async def _neo4j_state(workspace: str) -> tuple[int, list[str]]:
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        async with driver.session() as session:
            result = await session.run(f"MATCH (n:`{workspace}`) RETURN count(n) AS c")
            nodes = (await result.single())["c"]
            result = await session.run(
                "SHOW INDEXES YIELD name, labelsOrTypes "
                "WHERE $label IN labelsOrTypes RETURN name",
                label=workspace,
            )
            indexes = [record["name"] for record in await result.data()]
        return nodes, indexes
    finally:
        await driver.close()


def _milvus_collections(workspace: str) -> dict[str, int]:
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=os.environ["MILVUS_URI"],
        user=os.getenv("MILVUS_USER", ""),
        password=os.getenv("MILVUS_PASSWORD", ""),
    )
    try:
        out = {}
        for name in client.list_collections():
            if name.startswith(f"{workspace}_"):
                client.flush(name)
                out[name] = client.get_collection_stats(name)["row_count"]
        return out
    finally:
        client.close()


def test_purge_leaves_nothing_in_live_backends(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    for var in ("POSTGRES_WORKSPACE", "NEO4J_WORKSPACE", "MILVUS_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)

    from lightrag.api.workspace_deletion import purge_workspace
    from lightrag.api.workspace_manager import WorkspaceManager
    from lightrag.api.workspace_registry import (
        PostgresWorkspaceRegistry,
        local_identity,
    )
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    suffix = uuid.uuid4().hex[:6]
    target, bystander = f"purge_a_{suffix}", f"purge_b_{suffix}"
    working_dir, input_dir = tmp_path / "rag", tmp_path / "inputs"
    working_dir.mkdir()
    (input_dir / target).mkdir(parents=True)
    (input_dir / target / "notes.txt").write_text("Zebra Project notes")

    async def scenario():
        initialize_share_data()
        registry = PostgresWorkspaceRegistry(
            "MilvusVectorDBStorage"
        )  # as the server passes
        await registry.initialize()
        manager = WorkspaceManager(
            instance_factory=lambda ws: _build(ws, str(working_dir))
        )
        try:
            for workspace in (target, bystander):
                rag = await manager.get_instance(workspace)
                await rag.ainsert(
                    "Dr Nadia leads the Zebra Project, a private research effort.",
                    file_paths="notes.txt",
                )
            await registry.claim(target, local_identity(target), None)

            before = {ws: await _postgres_counts(ws) for ws in (target, bystander)}
            assert sum(before[target].values()) > 0, before
            assert (await _neo4j_state(target))[0] > 0
            assert any(_milvus_collections(target).values())

            result = await purge_workspace(
                target,
                workspace_manager=manager,
                registry=registry,
                working_dir=working_dir,
                input_dir=input_dir,
            )
            assert result.ok, result.errors

            after = await _postgres_counts(target)
            assert all(count == 0 for count in after.values()), after
            nodes, indexes = await _neo4j_state(target)
            assert nodes == 0 and indexes == [], (nodes, indexes)
            assert _milvus_collections(target) == {}
            assert not (input_dir / target).exists()
            assert await registry.get_by_workspace(target) is None
            assert await registry.get_tombstone(target) is not None
            assert await _postgres_tombstones() >= 1

            # The other workspace is untouched
            assert await _postgres_counts(bystander) == before[bystander]
            assert (await _neo4j_state(bystander))[0] > 0
            assert any(_milvus_collections(bystander).values())

            # Clean up the bystander too
            assert (
                await purge_workspace(
                    bystander,
                    workspace_manager=manager,
                    registry=registry,
                    working_dir=working_dir,
                    input_dir=input_dir,
                )
            ).ok
        finally:
            await manager.shutdown()
            await registry.finalize()
            finalize_share_data()

    asyncio.run(scenario())
