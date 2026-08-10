from __future__ import annotations

from dataclasses import replace

import pytest

from src.pipeline import RagPipeline
from src.schema import LLMAnswer, RerankItem, RerankResponse
from tests.conftest import FakeLLM


async def test_low_relevance_refuses_without_calling_llm(index, embedder, config):
    """
    Ключова властивість: питання не з бази не має коштувати жодного токена.
    Поріг виставлено недосяжно високим, щоб перевірка була детермінованою.
    """
    llm = FakeLLM([])
    pipeline = RagPipeline(index, embedder, replace(config, min_dense_score=1.01), llm)

    answer = await pipeline.ask("яка столиця Франції?")

    assert answer.answered is False
    assert llm.calls == []
    assert "теми немає в базі" in (answer.refusal_reason or "")
    assert answer.prompt_tokens == 0
    assert answer.latency_ms >= 0


async def test_full_flow_returns_citation(index, embedder, config):
    llm = FakeLLM([LLMAnswer(answered=True, answer="24 дні.", citation_ids=["C1"])])
    pipeline = RagPipeline(index, embedder, replace(config, min_dense_score=0.0), llm)

    answer = await pipeline.ask("скільки днів відпустки")

    assert answer.answered is True
    assert len(llm.calls) == 1
    assert answer.citations
    assert answer.retrieved


async def test_rerank_adds_one_extra_call_and_reorders(index, embedder, config):
    cfg = replace(config, min_dense_score=0.0, rerank=True, rerank_candidates=4, top_k=2)
    # перший виклик — переранжування, другий — генерація
    rerank_response = RerankResponse(
        items=[
            RerankItem(id="C1", relevance=1),
            RerankItem(id="C2", relevance=10),
            RerankItem(id="C3", relevance=0),
        ]
    )
    llm = FakeLLM([rerank_response, LLMAnswer(answered=True, answer="ok", citation_ids=["C1"])])
    pipeline = RagPipeline(index, embedder, cfg, llm)

    answer = await pipeline.ask("скільки днів відпустки")

    assert len(llm.calls) == 2
    assert answer.answered is True


async def test_search_only_never_calls_llm(index, embedder, config):
    llm = FakeLLM([])
    pipeline = RagPipeline(index, embedder, replace(config, min_dense_score=0.0), llm)

    results = await pipeline.search("добові")

    assert results
    assert llm.calls == []


async def test_ask_without_llm_raises(index, embedder, config):
    pipeline = RagPipeline(index, embedder, replace(config, min_dense_score=0.0), llm=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await pipeline.ask("скільки днів відпустки")


async def test_bm25_mode_skips_threshold_gate(index, embedder, config):
    """У BM25 немає порівнюваної шкали, тому поріг до нього не застосовується."""
    llm = FakeLLM([LLMAnswer(answered=True, answer="ok", citation_ids=["C1"])])
    cfg = replace(config, retriever="bm25", min_dense_score=1.01)
    pipeline = RagPipeline(index, embedder, cfg, llm)

    answer = await pipeline.ask("пароль 1Password")

    assert len(llm.calls) == 1
    assert answer.answered is True
