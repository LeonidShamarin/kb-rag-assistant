"""
Провайдери embeddings.

Три реалізації за спільним інтерфейсом — щоб eval-матриця могла їх порівняти
без змін у решті коду:

- `GeminiEmbeddings` — дефолт, багатомовна, без локальних моделей;
- `SentenceTransformerEmbeddings` — локальні моделі (e5, MiniLM). Саме тут
  вимірюється теза «англомовна модель провалюється на українській»;
- `HashingEmbeddings` — детермінований фейк для тестів. Не «майже як справжній»,
  а свідомо тупий: тести перевіряють механіку пайплайна, а не якість пошуку.

Дві деталі, які на українській даються взнаки і яких у туторіалах зазвичай немає:

1. **Асиметрія query/document.** Питання і документ кодуються різними режимами.
   У Gemini це `task_type` (RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT), у e5 —
   обов'язкові префікси `query:` / `passage:`. Без них e5 втрачає помітну частку
   якості, і це найчастіша мовчазна помилка при його використанні.
2. **Нормалізація.** `gemini-embedding-001` повертає ненормалізовані вектори для
   будь-якої розмірності, крім 3072. Порівнювати їх косинусом без нормалізації —
   тихо неправильно. Нормалізуємо завжди й самі.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

import numpy as np

from src.backoff import Breaker, RateLimiter, call_with_backoff

logger = logging.getLogger(__name__)

# Gemini приймає список у одному виклику; 32 — компроміс між кількістю запитів
# (квота рахує запити, не тексти) і розміром payload.
GEMINI_BATCH = 32


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    async def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashingEmbeddings(BaseEmbedder):
    """
    Детермінований bag-of-words hashing. Мережі не потребує — на ньому працює
    весь тестовий набір, тому тести запускаються без API-ключа.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        from src.textnorm import normalize

        vec = np.zeros(self.dim, dtype=np.float32)
        for token in normalize(text, stemming=True):
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        return vec

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.vstack([self._vector(t) for t in texts]))

    async def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vector(text).reshape(1, -1))[0]


class GeminiEmbeddings(BaseEmbedder):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        rpm: int = 100,
        breaker: Breaker | None = None,
    ):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dim = dim
        self.name = model
        self._limiter = RateLimiter(rpm)
        self._breaker = breaker or Breaker()

    async def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        out: list[list[float]] = []

        for start in range(0, len(texts), GEMINI_BATCH):
            batch = texts[start : start + GEMINI_BATCH]

            async def call(batch: list[str] = batch):
                return await self._client.aio.models.embed_content(
                    model=self._model, contents=batch, config=config
                )

            response = await call_with_backoff(
                call, self._limiter, self._breaker, what=f"embed[{start}]"
            )
            values = [e.values for e in response.embeddings]
            if len(values) != len(batch):
                raise ValueError(
                    f"Очікували {len(batch)} векторів, отримали {len(values)}"
                )
            out.extend(values)

        return _l2_normalize(np.array(out, dtype=np.float32))

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]


class SentenceTransformerEmbeddings(BaseEmbedder):
    """
    Локальні моделі. Ставиться окремо: `pip install -r requirements-local.txt`
    (тягне torch, ~2 ГБ — тому не в основних залежностях і не в Docker-образі).
    """

    def __init__(self, model: str = "intfloat/multilingual-e5-small"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Потрібен sentence-transformers: pip install -r requirements-local.txt"
            ) from exc

        self._model = SentenceTransformer(model)
        self.name = model
        # sentence-transformers 5.x перейменував метод; стара назва ще працює,
        # але сипле FutureWarning у кожному прогоні.
        get_dim = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()
        # Префікси e5 — не косметика: без них модель втрачає частину якості,
        # бо навчалась саме на такій асиметрії.
        self._needs_prefix = "e5" in model.lower()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(
            np.array(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)
        )

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._needs_prefix:
            texts = [f"passage: {t}" for t in texts]
        # Синхронна CPU-бібліотека — виносимо з event loop, щоб не блокувати сервер.
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> np.ndarray:
        query = f"query: {text}" if self._needs_prefix else text
        return (await asyncio.to_thread(self._encode, [query]))[0]


def build_embedder(
    provider: str, model: str, api_key: str | None = None, breaker: Breaker | None = None
) -> BaseEmbedder:
    if provider == "hash":
        return HashingEmbeddings()
    if provider == "st":
        return SentenceTransformerEmbeddings(model)
    if provider == "gemini":
        if not api_key:
            raise ValueError("GEMINI_API_KEY не заданий — потрібен для provider=gemini")
        return GeminiEmbeddings(api_key=api_key, model=model, breaker=breaker)
    raise ValueError(f"Невідомий провайдер embeddings: {provider}")
