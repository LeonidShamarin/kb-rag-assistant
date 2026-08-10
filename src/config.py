"""
Конфігурація RAG-пайплайна.

Сенс окремого об'єкта: eval-матриця (`main.py matrix`) перебирає саме ці
параметри, а індекс зберігає свою копію конфігу поруч із векторами — щоб не
шукати відповідь векторами від однієї моделі в індексі, побудованому іншою.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

RetrieverMode = Literal["dense", "bm25", "hybrid"]
ChunkStrategy = Literal["structural", "fixed"]

# Модель генерації. flash-lite достатньо: завдання — переказати наданий контекст,
# а не міркувати. Денна квота безкоштовного тіру в неї помітно щедріша за flash.
DEFAULT_GEN_MODEL = "gemini-3.5-flash-lite"

# Embeddings. gemini-embedding-001 — багатомовна; для української це принципово
# (див. README, розділ про порівняння моделей).
DEFAULT_EMBED_MODEL = "gemini-embedding-001"

DEFAULT_RPM = 5


@dataclass
class RagConfig:
    # --- індексація ---
    chunking: ChunkStrategy = "structural"
    chunk_chars: int = 900
    chunk_overlap: int = 150
    embed_provider: str = "gemini"  # gemini | st | hash
    embed_model: str = DEFAULT_EMBED_MODEL

    # --- пошук ---
    retriever: RetrieverMode = "hybrid"
    top_k: int = 5
    candidates: int = 20  # скільки беремо з кожного ретривера до злиття
    rrf_k: int = 60
    stemming: bool = True
    rerank: bool = False
    rerank_candidates: int = 12

    # --- поріг відмови ---
    # Нижче цього косинуса найкращий чанк вважається нерелевантним, і система
    # відмовляється відповідати ще ДО виклику генератора — безкоштовно.
    min_dense_score: float = 0.55

    # --- генерація ---
    gen_model: str = DEFAULT_GEN_MODEL
    temperature: float = 0.0
    rpm: int = DEFAULT_RPM

    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def short_name(self) -> str:
        """Компактна назва конфігурації для рядка таблиці в eval-звіті."""
        parts = [self.retriever, self.chunking]
        if self.retriever in ("bm25", "hybrid"):
            parts.append("stem" if self.stemming else "nostem")
        if self.rerank:
            parts.append("rerank")
        if self.embed_provider != "gemini":
            parts.append(self.embed_model.split("/")[-1])
        return "+".join(parts)
