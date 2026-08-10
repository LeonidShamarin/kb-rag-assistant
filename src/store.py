"""
Сховище індексу: чанки + вектори + метадані конфігурації.

Про вибір «просто numpy замість FAISS» — свідомий і заміряний. На корпусі цього
масштабу (сотні чанків) точний пошук множенням матриці займає одиниці
мілісекунд, тобто ~0.1% від часу одного виклику LLM. FAISS тут не пришвидшує
нічого, зате додає залежність, яка на Windows ставиться через раз.

Тому: numpy — дефолт, FAISS — опційний бекенд (`--faiss`) для демонстрації, що
інтерфейс до нього готовий. Поріг, за яким перехід дійсно потрібен, — приблизно
10^5 векторів; у README це записано як обмеження, а не як «зробимо колись».
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.schema import Chunk

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict):
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Розбіжність індексу: {len(chunks)} чанків проти {vectors.shape[0]} векторів"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.meta = meta
        self._faiss = None

    # --- пошук ---

    def search_dense(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Косинусна близькість. Вектори нормалізовані, тому це просто скалярний добуток."""
        if not self.chunks:
            return []
        if self._faiss is not None:
            scores, idx = self._faiss.search(query_vector.reshape(1, -1).astype("float32"), top_k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

        scores = self.vectors @ query_vector
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top]

    def enable_faiss(self) -> bool:
        """Вмикає FAISS-бекенд, якщо пакет доступний. Повертає факт успіху."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss не встановлено — лишаємось на numpy")
            return False
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors.astype("float32"))
        self._faiss = index
        return True

    # --- персистентність ---

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        np.save(directory / VECTORS_FILE, self.vectors)
        meta = {
            **self.meta,
            "chunks": len(self.chunks),
            "dim": int(self.vectors.shape[1]),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (directory / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeIndex":
        meta_path = directory / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Індекс не знайдено в {directory}. Спершу виконайте: python main.py ingest"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk.model_validate_json(line)
            for line in (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(directory / VECTORS_FILE)
        return cls(chunks, vectors, meta)

    def assert_compatible(self, embedder_name: str, dim: int) -> None:
        """
        Найпідступніша помилка в RAG: індекс побудований однією моделлю, а запит
        кодується іншою. Розмірність може навіть збігтись — і пошук просто тихо
        поверне сміття. Тому звіряємо явно і падаємо гучно.
        """
        built_with = self.meta.get("embedder")
        if built_with and built_with != embedder_name:
            raise ValueError(
                f"Індекс побудований моделлю '{built_with}', а запит кодується '{embedder_name}'. "
                "Перебудуйте індекс: python main.py ingest"
            )
        if dim != self.vectors.shape[1]:
            raise ValueError(
                f"Розмірність не збігається: індекс {self.vectors.shape[1]}, модель {dim}"
            )
