from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.retriever import Retriever
from src.store import KnowledgeIndex


async def test_dense_retrieval_returns_scores(index, embedder, config):
    retriever = Retriever(index, embedder, replace(config, retriever="dense"))
    results = await retriever.retrieve("скільки днів відпустки")
    assert results
    assert all(r.dense_score is not None for r in results)
    assert all(r.origin == "dense" for r in results)


async def test_bm25_mode_has_no_dense_score(index, embedder, config):
    retriever = Retriever(index, embedder, replace(config, retriever="bm25"))
    results = await retriever.retrieve("1Password пароль")
    assert results
    assert all(r.dense_score is None for r in results)
    assert results[0].chunk.doc_id == "bezpeka"


async def test_hybrid_merges_both_sources(index, embedder, config):
    retriever = Retriever(index, embedder, replace(config, retriever="hybrid"))
    results = await retriever.retrieve("добові в Україні")
    assert results
    assert {r.origin for r in results} & {"both", "dense", "bm25"}
    assert any(r.chunk.doc_id == "vidryadzhennya" for r in results)


def test_rrf_prefers_consensus():
    """
    Суть RRF: документ, який обидва ретривери поставили другим, має обійти той,
    який лише один поставив першим. Саме тому гібрид стійкіший за кожен окремо.
    """
    dense = [(1, 0.9), (2, 0.8)]
    lexical = [(3, 12.0), (2, 9.0)]
    fused = Retriever._fuse(dense, lexical, k=60)
    assert fused[0][0] == 2
    assert fused[0][3] == 1 and fused[0][4] == 1  # ранги в обох списках


def test_index_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = KnowledgeIndex.load(tmp_path)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.chunks[0].chunk_id == index.chunks[0].chunk_id
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.meta["chunks"] == len(index.chunks)


def test_index_rejects_foreign_embedder(index):
    """Найпідступніша помилка в RAG — і вона має падати гучно, а не тихо шукати сміття."""
    with pytest.raises(ValueError, match="побудований моделлю"):
        index.assert_compatible("some-other-model", index.vectors.shape[1])


def test_index_rejects_dimension_mismatch(index):
    with pytest.raises(ValueError, match="Розмірність"):
        index.assert_compatible(index.meta["embedder"], 1)


def test_index_rejects_length_mismatch(index):
    with pytest.raises(ValueError, match="Розбіжність"):
        KnowledgeIndex(index.chunks[:-1], index.vectors, index.meta)


def test_load_missing_index_explains_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        KnowledgeIndex.load(tmp_path / "nope")
