"""
HTTP-шар: FastAPI + одна самодостатня HTML-сторінка.

Чому не Gradio: він тягне ~200 МБ залежностей і власний фронтенд заради вікна
чату. Тут вікно чату — це 200 рядків HTML без жодної зовнішньої залежності, і
Docker-образ лишається легким. Для демо цього достатньо, а контроль над тим, як
показані цитати, у RAG важливіший за красиві дефолти.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import RagConfig
from src.embeddings import build_embedder
from src.llm import GeminiLLM
from src.pipeline import RagPipeline
from src.schema import Answer
from src.store import KnowledgeIndex

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

state: dict[str, object] = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


def build_pipeline(index_dir: Path, config: RagConfig) -> RagPipeline:
    index = KnowledgeIndex.load(index_dir)
    api_key = os.environ.get("GEMINI_API_KEY")
    embedder = build_embedder(config.embed_provider, config.embed_model, api_key)
    llm = (
        GeminiLLM(api_key, config.gen_model, config.temperature, config.rpm)
        if api_key
        else None
    )
    return RagPipeline(index, embedder, config, llm)


def create_app(index_dir: Path, config: RagConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Індекс вантажиться один раз на старті: перечитувати його на кожен
        # запит — це десятки мілісекунд і зайва робота диску.
        state["pipeline"] = build_pipeline(index_dir, config)
        logger.info("Індекс завантажено, сервіс готовий")
        yield
        state.clear()

    app = FastAPI(title="kb-rag-assistant", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        pipeline: RagPipeline | None = state.get("pipeline")  # type: ignore[assignment]
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Індекс не завантажено")
        return {
            "status": "ok",
            "chunks": len(pipeline.index.chunks),
            "documents": pipeline.index.meta.get("documents"),
            "embedder": pipeline.index.meta.get("embedder"),
            "retriever": pipeline.config.retriever,
            "rerank": pipeline.config.rerank,
            "generation_enabled": pipeline.llm is not None,
        }

    @app.post("/ask", response_model=Answer)
    async def ask(request: AskRequest) -> Answer:
        pipeline: RagPipeline | None = state.get("pipeline")  # type: ignore[assignment]
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Індекс не завантажено")
        if pipeline.llm is None:
            raise HTTPException(
                status_code=503, detail="GEMINI_API_KEY не заданий — генерація недоступна"
            )
        try:
            return await pipeline.ask(request.question)
        except Exception as exc:  # noqa: BLE001 — назовні віддаємо 502, не трейсбек
            logger.exception("Помилка обробки питання")
            raise HTTPException(status_code=502, detail=str(exc)[:300]) from exc

    @app.get("/")
    async def index_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app
