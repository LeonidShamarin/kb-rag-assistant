"""
Контракт даних. Єдине джерело правди: ці ж моделі йдуть у LLM як `response_schema`
і ними ж валідується сира відповідь.

Розділення навмисне:
- `LLMAnswer` — те, що *каже* модель. Їй не можна довіряти наосліп.
- `Answer` — те, що система *віддає* користувачу, уже після перевірки цитат.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Цілий файл бази знань до розбиття на чанки."""

    doc_id: str
    title: str
    source_path: str
    ext: str
    text: str


class Chunk(BaseModel):
    """
    Одиниця пошуку. `section` — «хлібні крихти» заголовків (H1 > H2), а не просто
    назва секції: без них чанк «до 200 EUR — керівник команди» не має сенсу ні для
    людини, ні для embedding-моделі.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    ordinal: int

    @property
    def embed_text(self) -> str:
        """
        Текст, який реально індексується. Контекст заголовків додається до тіла
        чанка — це найдешевший приріст якості і для dense, і для BM25.
        """
        head = f"{self.doc_title} > {self.section}" if self.section else self.doc_title
        return f"{head}\n{self.text}"

    @property
    def label(self) -> str:
        return f"{self.doc_title} > {self.section}" if self.section else self.doc_title


class ScoredChunk(BaseModel):
    """Чанк із скором конкретного ретривера + сліди, звідки він прийшов."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

    @property
    def origin(self) -> str:
        if self.dense_rank is not None and self.bm25_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.bm25_rank is not None:
            return "bm25"
        return "unknown"


class LLMAnswer(BaseModel):
    """Сира відповідь моделі. `citation_ids` тут ще не перевірені."""

    answered: bool = Field(
        description="true, якщо у наданому контексті достатньо інформації для відповіді"
    )
    answer: str = Field(description="Відповідь українською або пояснення, чому відповісти не можна")
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Ідентифікатори фрагментів контексту (C1, C2, ...), на які спирається відповідь",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RerankItem(BaseModel):
    id: str = Field(description="Ідентифікатор фрагмента, напр. C3")
    relevance: int = Field(ge=0, le=10, description="0 — не стосується питання, 10 — містить пряму відповідь")


class RerankResponse(BaseModel):
    items: list[RerankItem] = Field(default_factory=list)


class Citation(BaseModel):
    """Перевірене посилання на джерело — те, що показуємо користувачу."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    source_path: str
    quote: str


class Answer(BaseModel):
    """Фінальна відповідь системи."""

    question: str
    answered: bool
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    refusal_reason: str | None = None
    retrieved: list[str] = Field(default_factory=list, description="chunk_id, що потрапили в контекст")
    latency_ms: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    dropped_citations: list[str] = Field(
        default_factory=list, description="ID, які модель вигадала — відкинуті"
    )
