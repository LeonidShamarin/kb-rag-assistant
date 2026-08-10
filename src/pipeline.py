"""
Оркестрація RAG: інжест і відповідь на питання.

Порядок кроків у `ask()` не випадковий — він оптимізує вартість:

    пошук → поріг відмови → переранжування → генерація

Поріг стоїть **до** переранжування і до генерації, тому питання не з бази
(«яка столиця Франції?») коштує рівно один виклик embeddings і жодного виклику
LLM. Порахувати відмову — найдешевша операція в системі, і робити її треба
першою, а не після того, як модель уже спалила токени на порожньому контексті.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.chunking import chunk_documents
from src.config import RagConfig
from src.embeddings import BaseEmbedder
from src.generator import generate_answer, refusal
from src.llm import GeminiLLM
from src.loaders import load_corpus
from src.rerank import rerank
from src.retriever import Retriever
from src.schema import Answer
from src.store import KnowledgeIndex

logger = logging.getLogger(__name__)


async def ingest(
    corpus_dir: Path, index_dir: Path, config: RagConfig, embedder: BaseEmbedder
) -> KnowledgeIndex:
    docs = load_corpus(corpus_dir)
    if not docs:
        raise ValueError(f"У {corpus_dir} не знайдено жодного підтримуваного документа")

    chunks = chunk_documents(docs, config.chunking, config.chunk_chars, config.chunk_overlap)
    logger.info(
        "Завантажено %d документів → %d чанків (стратегія: %s)",
        len(docs),
        len(chunks),
        config.chunking,
    )

    vectors = await embedder.embed_documents([c.embed_text for c in chunks])
    index = KnowledgeIndex(
        chunks=chunks,
        vectors=vectors,
        meta={
            "embedder": embedder.name,
            "config": config.to_dict(),
            "documents": len(docs),
            "sources": {d.doc_id: d.source_path for d in docs},
        },
    )
    index.save(index_dir)
    logger.info("Індекс збережено в %s", index_dir)
    return index


class RagPipeline:
    def __init__(
        self,
        index: KnowledgeIndex,
        embedder: BaseEmbedder,
        config: RagConfig,
        llm: GeminiLLM | None = None,
    ):
        index.assert_compatible(embedder.name, embedder.dim)
        self.index = index
        self.config = config
        self.llm = llm
        self.retriever = Retriever(index, embedder, config)
        self.sources: dict[str, str] = index.meta.get("sources", {})

    async def search(self, question: str):
        """Тільки пошук — використовується eval-матрицею, де генерація не потрібна."""
        n = self.config.rerank_candidates if self.config.rerank else self.config.top_k
        candidates = await self.retriever.retrieve(question, top_k=n)
        if self.config.rerank and self.llm is not None:
            return await rerank(self.llm, question, candidates, self.config.top_k)
        return candidates[: self.config.top_k]

    async def ask(self, question: str) -> Answer:
        started = time.perf_counter()

        n = self.config.rerank_candidates if self.config.rerank else self.config.top_k
        candidates = await self.retriever.retrieve(question, top_k=n)

        if not candidates:
            return self._finish(refusal(question, "пошук не повернув жодного фрагмента"), started)

        gate = self._below_threshold(candidates)
        if gate is not None:
            return self._finish(
                refusal(question, gate, [c.chunk.chunk_id for c in candidates]), started
            )

        if self.config.rerank and self.llm is not None:
            candidates = await rerank(self.llm, question, candidates, self.config.top_k)
        else:
            candidates = candidates[: self.config.top_k]

        if self.llm is None:
            raise RuntimeError("Для відповіді потрібен LLM-клієнт (GEMINI_API_KEY)")

        answer = await generate_answer(self.llm, question, candidates, self.sources)
        return self._finish(answer, started)

    def _below_threshold(self, candidates) -> str | None:
        """
        Найкращий dense-скор нижче порогу означає, що в базі просто немає теми.
        У режимі `bm25` перевірка не виконується — у BM25 немає порівнюваної між
        корпусами шкали, і будь-який поріг для нього був би вигаданим числом.
        """
        scores = [c.dense_score for c in candidates if c.dense_score is not None]
        if not scores:
            return None
        best = max(scores)
        if best < self.config.min_dense_score:
            return (
                f"найкращий фрагмент має релевантність {best:.2f} "
                f"(поріг {self.config.min_dense_score:.2f}) — теми немає в базі"
            )
        return None

    @staticmethod
    def _finish(answer: Answer, started: float) -> Answer:
        answer.latency_ms = int((time.perf_counter() - started) * 1000)
        return answer
