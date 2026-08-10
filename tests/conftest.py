"""
Спільні фікстури.

Головне правило набору: **жоден тест не ходить у мережу**. Embeddings —
детермінований hashing-провайдер, LLM — фейк із заздалегідь заданою відповіддю.
Тому тести запускаються без GEMINI_API_KEY і не залежать від квоти.
"""

from __future__ import annotations

import asyncio

import pytest

from src.chunking import chunk_documents
from src.config import RagConfig
from src.embeddings import HashingEmbeddings
from src.llm import LLMResult
from src.schema import Document
from src.store import KnowledgeIndex

CORPUS = [
    Document(
        doc_id="vidpustky",
        title="Відпустки",
        source_path="data/corpus/vidpustky.md",
        ext=".md",
        text=(
            "# Відпустки\n\n"
            "## Щорічна відпустка\n\n"
            "Кожен працівник має 24 календарні дні щорічної оплачуваної відпустки.\n\n"
            "## Як подати заявку\n\n"
            "Заявка подається за 14 календарних днів до початку відпустки.\n"
        ),
    ),
    Document(
        doc_id="vidryadzhennya",
        title="Відрядження",
        source_path="data/corpus/vidryadzhennya.md",
        ext=".md",
        text=(
            "# Відрядження\n\n"
            "## Добові\n\n"
            "Добові в Україні становлять 400 грн на день, у країнах ЄС — 60 EUR.\n\n"
            "## Авансовий звіт\n\n"
            "Звіт подається протягом 5 робочих днів після повернення.\n"
        ),
    ),
    Document(
        doc_id="bezpeka",
        title="Інформаційна безпека",
        source_path="data/corpus/bezpeka.md",
        ext=".md",
        text=(
            "# Інформаційна безпека\n\n"
            "## Паролі\n\n"
            "Мінімальна довжина пароля — 14 символів. Паролі зберігаються в 1Password.\n"
        ),
    ),
]


@pytest.fixture
def instant_sleep(monkeypatch):
    """
    Прибирає паузи backoff, не ламаючи `asyncio.sleep`.

    Пастка, через яку це стало фікстурою, а не рядком у тесті:

        monkeypatch.setattr(asyncio, "sleep", lambda *_: asyncio.sleep(0))

    підміняє функцію саму на себе. Лямбда всередині шукає `asyncio.sleep` уже
    після підміни — тобто викликає себе, і так без кінця. Кожен рівень лишає по
    собі неочікувану корутину, RecursionError ловиться `except Exception` у
    `call_with_backoff` і цикл починається знову. На практиці процес виїдав
    десятки гігабайт і клав машину. Оригінал треба захопити ДО підміни.
    """
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_: real_sleep(0))


@pytest.fixture
def config() -> RagConfig:
    return RagConfig(embed_provider="hash", embed_model="hash-256", rpm=0)


@pytest.fixture
def embedder() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
async def index(config: RagConfig, embedder: HashingEmbeddings) -> KnowledgeIndex:
    chunks = chunk_documents(CORPUS, config.chunking, config.chunk_chars, config.chunk_overlap)
    vectors = await embedder.embed_documents([c.embed_text for c in chunks])
    return KnowledgeIndex(
        chunks=chunks,
        vectors=vectors,
        meta={
            "embedder": embedder.name,
            "documents": len(CORPUS),
            "sources": {d.doc_id: d.source_path for d in CORPUS},
        },
    )


class FakeLLM:
    """
    Підміна GeminiLLM. Віддає заздалегідь задані об'єкти по черзі й рахує виклики
    — саме на лічильнику тримається перевірка, що відмова не витрачає токенів.
    """

    def __init__(self, responses: list, model: str = "fake"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str, schema, max_repairs: int = 1):
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("FakeLLM: викликів більше, ніж заготовлених відповідей")
        parsed = self.responses.pop(0)
        return LLMResult(parsed=parsed, prompt_tokens=100, output_tokens=20, repairs=0)


@pytest.fixture
def fake_llm():
    return FakeLLM
