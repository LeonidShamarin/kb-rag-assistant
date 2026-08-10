"""
Пошук: dense, BM25 і гібрид.

Гібрид зроблено через **Reciprocal Rank Fusion**, а не через зважену суму скорів.
Причина практична: косинус лежить у [0, 1] і на реальних даних майже завжди
в діапазоні 0.5–0.9, а BM25 — необмежений і залежить від довжини корпусу.
Зважена сума таких величин вимагає нормалізації, яка ламається щоразу, коли
корпус змінюється. RRF же дивиться лише на **ранг**, тому нічого нормалізувати
не треба і параметрів у нього рівно один.

    score(chunk) = Σ 1 / (k + rank_i)

`k = 60` — значення з оригінальної статті Cormack et al.; воно приглушує
надмірний вплив першого місця одного ретривера над консенсусом обох.
"""

from __future__ import annotations

import logging

from src.bm25 import BM25Index
from src.config import RagConfig
from src.embeddings import BaseEmbedder
from src.schema import ScoredChunk
from src.store import KnowledgeIndex

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, index: KnowledgeIndex, embedder: BaseEmbedder, config: RagConfig):
        self.index = index
        self.embedder = embedder
        self.config = config
        self.bm25: BM25Index | None = None
        if config.retriever in ("bm25", "hybrid"):
            # Індексуємо embed_text, а не text: breadcrumb заголовків має
            # брати участь і в лексичному пошуку теж.
            self.bm25 = BM25Index(
                [c.embed_text for c in index.chunks], stemming=config.stemming
            )

    async def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        cfg = self.config
        top_k = top_k or cfg.top_k
        n = cfg.candidates

        dense: list[tuple[int, float]] = []
        if cfg.retriever in ("dense", "hybrid"):
            qvec = await self.embedder.embed_query(query)
            dense = self.index.search_dense(qvec, n)

        lexical: list[tuple[int, float]] = []
        if self.bm25 is not None:
            lexical = self.bm25.search(query, n)

        if cfg.retriever == "dense":
            ranked = [(i, s, s, r, None) for r, (i, s) in enumerate(dense)]
        elif cfg.retriever == "bm25":
            ranked = [(i, s, None, None, r) for r, (i, s) in enumerate(lexical)]
        else:
            ranked = self._fuse(dense, lexical, cfg.rrf_k)

        results = [
            ScoredChunk(
                chunk=self.index.chunks[idx],
                score=score,
                dense_score=dense_score,
                dense_rank=d_rank,
                bm25_rank=b_rank,
            )
            for idx, score, dense_score, d_rank, b_rank in ranked[:top_k]
        ]
        return results

    @staticmethod
    def _fuse(
        dense: list[tuple[int, float]], lexical: list[tuple[int, float]], k: int
    ) -> list[tuple[int, float, float | None, int | None, int | None]]:
        fused: dict[int, float] = {}
        dense_rank: dict[int, int] = {}
        dense_score: dict[int, float] = {}
        bm25_rank: dict[int, int] = {}

        for rank, (idx, score) in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
            dense_rank[idx] = rank
            dense_score[idx] = score

        for rank, (idx, _score) in enumerate(lexical):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
            bm25_rank[idx] = rank

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (idx, score, dense_score.get(idx), dense_rank.get(idx), bm25_rank.get(idx))
            for idx, score in ordered
        ]
