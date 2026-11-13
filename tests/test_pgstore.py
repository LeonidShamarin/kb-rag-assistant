"""
pgvector-бекенд.

Перша половина тестів іде без бази (формат вектора, відбиток, делегування з
KnowledgeIndex через фейк). Друга половина потребує справжньої Postgres з
pgvector і запускається лише коли задано PGVECTOR_DSN: локально через
docker (див. README), у CI через service-контейнер pgvector/pgvector.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import numpy as np
import pytest

from src.pgstore import PgVectorStore, index_fingerprint, vector_literal
from src.retriever import Retriever

DSN = os.environ.get("PGVECTOR_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="PGVECTOR_DSN не задано: інтеграційні тести pgvector пропущено")


# --- без бази ---


def test_vector_literal_format():
    assert vector_literal(np.array([0.5, -1.0, 0.25])) == "[0.5,-1,0.25]"


def test_vector_literal_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        vector_literal(np.array([0.1, np.nan]))


def test_fingerprint_changes_with_index():
    meta = {"embedder": "e5", "built_at": "2026-11-13T10:00:00+00:00"}
    base = index_fingerprint(meta, 100, 384)
    assert base == index_fingerprint(dict(meta), 100, 384)
    assert base != index_fingerprint(meta, 101, 384)
    assert base != index_fingerprint({**meta, "embedder": "minilm"}, 100, 384)


def test_table_name_is_validated_before_connecting():
    # Назву таблиці не можна передати параметром SQL, тому вона перевіряється
    # ще до спроби з'єднання: ін'єкція через неї неможлива.
    with pytest.raises(ValueError, match="Недопустима"):
        PgVectorStore("postgresql://nowhere/x", table="kb; DROP TABLE users")


class FakePgStore:
    """Повторює контракт PgVectorStore, рахуючи пошук numpy, щоб перевірити підключення."""

    def __init__(self):
        self.synced = None
        self.calls = 0

    def sync(self, chunk_ids, vectors, meta):
        self.synced = (list(chunk_ids), vectors.copy())
        return True

    def search(self, query_vector, top_k):
        self.calls += 1
        _, vectors = self.synced
        scores = vectors @ query_vector
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top]


async def test_index_delegates_dense_search_to_pgvector(index, embedder, config):
    fake = FakePgStore()
    index.enable_pgvector("unused", store=fake)
    assert fake.synced[0] == [c.chunk_id for c in index.chunks]

    retriever = Retriever(index, embedder, replace(config, retriever="dense"))
    results = await retriever.retrieve("скільки днів відпустки")
    assert fake.calls == 1
    assert results and results[0].chunk.doc_id == "vidpustky"


# --- з базою ---


@pytest.fixture
def pg_table():
    """Окрема таблиця на тест, прибирається після нього."""
    table = "kb_test_" + uuid.uuid4().hex[:8]
    yield table
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("DELETE FROM kb_index_meta WHERE tbl = %s", (table,))


@needs_db
def test_exact_search_matches_numpy(index, pg_table):
    store = PgVectorStore(DSN, table=pg_table)
    try:
        assert store.sync([c.chunk_id for c in index.chunks], index.vectors, index.meta) is True
        # Порядок при рівних скорах у двох бекендів різний (у hashing-векторів
        # нічиї трапляються), тому порівнюємо скори позиція в позицію і те, що
        # кожен повернутий чанк справді має такий скор за numpy.
        for q in index.vectors:
            expected = index.search_dense(q, 3)
            got = store.search(q, 3)
            assert np.allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)
            exact = index.vectors @ q
            for i, s in got:
                assert abs(exact[i] - s) < 1e-5
    finally:
        store.close()


@needs_db
def test_sync_is_idempotent(index, pg_table):
    ids = [c.chunk_id for c in index.chunks]
    store = PgVectorStore(DSN, table=pg_table)
    try:
        assert store.sync(ids, index.vectors, index.meta) is True
        assert store.sync(ids, index.vectors, index.meta) is False
        changed = {**index.meta, "built_at": "2099-01-01T00:00:00+00:00"}
        assert store.sync(ids, index.vectors, changed) is True
    finally:
        store.close()


@needs_db
def test_hnsw_mode_returns_results(index, pg_table):
    store = PgVectorStore(DSN, table=pg_table, hnsw=True)
    try:
        store.sync([c.chunk_id for c in index.chunks], index.vectors, index.meta)
        got = store.search(index.vectors[0], 2)
        assert got[0][0] == 0
    finally:
        store.close()
