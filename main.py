"""
CLI асистента по базі знань.

    python main.py ingest                       # побудувати індекс
    python main.py ask "скільки днів відпустки?"
    python main.py serve                        # HTTP + веб-інтерфейс на :7860
    python main.py eval --full                  # прогнати golden-набір
    python main.py matrix                       # порівняти конфігурації пошуку

Env: GEMINI_API_KEY (обов'язково для gemini-embeddings і генерації).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from src.config import DEFAULT_EMBED_MODEL, DEFAULT_GEN_MODEL, DEFAULT_RPM, RagConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("kb-rag")

DEFAULT_CORPUS = Path("data/corpus")
DEFAULT_INDEX = Path("index")
DEFAULT_EVAL = Path("eval/questions.yaml")


def _use_system_trust_store() -> None:
    """
    httpx (усередині google-genai) довіряє лише бандлу `certifi`. У мережах з
    TLS-інспекцією корпоративний корінь є у сховищі ОС, але не в certifi — і всі
    виклики падають з CERTIFICATE_VERIFY_FAILED. truststore бере довіру звідти,
    де вона реально налаштована. Перевірку сертифікатів це не послаблює.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        logger.debug("truststore недоступний, лишаємось на certifi")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--chunking", choices=["structural", "fixed"], default="structural")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-stemming", action="store_true", help="Вимкнути стемер для BM25")
    parser.add_argument("--rerank", action="store_true", help="Переранжування кандидатів через LLM")
    parser.add_argument("--embed-provider", choices=["gemini", "st", "hash"], default="gemini")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--model", default=DEFAULT_GEN_MODEL, help="Модель генерації")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM, help="0 — без обмеження")
    parser.add_argument("--min-score", type=float, default=0.55, help="Поріг відмови (косинус)")
    parser.add_argument("--faiss", action="store_true", help="Використати FAISS замість numpy")


def config_from(args: argparse.Namespace) -> RagConfig:
    return RagConfig(
        chunking=args.chunking,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        embed_provider=args.embed_provider,
        embed_model=args.embed_model,
        retriever=args.retriever,
        top_k=args.top_k,
        stemming=not args.no_stemming,
        rerank=args.rerank,
        min_dense_score=args.min_score,
        gen_model=args.model,
        rpm=args.rpm,
    )


def _api_key(required: bool = True) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key and required:
        logger.error("GEMINI_API_KEY не знайдено. Додайте його в .env (див. .env.example)")
        sys.exit(1)
    return key


def _build(config: RagConfig, index_dir: Path, use_faiss: bool = False):
    """Складає пайплайн з готового індексу."""
    from src.embeddings import build_embedder
    from src.llm import GeminiLLM
    from src.pipeline import RagPipeline
    from src.store import KnowledgeIndex

    key = _api_key(required=config.embed_provider == "gemini")
    index = KnowledgeIndex.load(index_dir)
    if use_faiss:
        index.enable_faiss()
    embedder = build_embedder(config.embed_provider, config.embed_model, key)
    llm = GeminiLLM(key, config.gen_model, config.temperature, config.rpm) if key else None
    return RagPipeline(index, embedder, config, llm)


# --------------------------------------------------------------------------- #
# команди
# --------------------------------------------------------------------------- #


async def cmd_ingest(args: argparse.Namespace) -> None:
    from src.embeddings import build_embedder
    from src.pipeline import ingest

    config = config_from(args)
    key = _api_key(required=config.embed_provider == "gemini")
    embedder = build_embedder(config.embed_provider, config.embed_model, key)
    index = await ingest(args.corpus, args.index, config, embedder)
    print(
        f"Готово: {index.meta['documents']} документів → {len(index.chunks)} чанків, "
        f"вектори {index.vectors.shape}, модель {embedder.name}"
    )


async def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = _build(config_from(args), args.index, args.faiss)
    answer = await pipeline.ask(args.question)

    if args.json:
        print(json.dumps(answer.model_dump(), ensure_ascii=False, indent=2))
        return

    print()
    print(answer.answer)
    if answer.citations:
        print("\nДжерела:")
        for c in answer.citations:
            section = f" → {c.section}" if c.section else ""
            print(f"  • {c.doc_title}{section}  [{c.source_path}]")
    if not answer.answered and answer.refusal_reason:
        print(f"\n(відмова: {answer.refusal_reason})")
    if answer.dropped_citations:
        print(f"(відкинуто вигаданих цитат: {answer.dropped_citations})")
    print(
        f"\n{answer.latency_ms} мс · токенів: "
        f"{answer.prompt_tokens} вх / {answer.output_tokens} вих"
    )


async def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from src.app import create_app

    config = config_from(args)
    _api_key(required=config.embed_provider == "gemini")
    app = create_app(args.index, config)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
    await server.serve()


async def cmd_eval(args: argparse.Namespace) -> None:
    from src.evaluate import evaluate, failures_markdown, load_questions, summarize

    config = config_from(args)
    pipeline = _build(config, args.index, args.faiss)
    known = {c.doc_id for c in pipeline.index.chunks}
    questions = load_questions(args.questions, known)

    if args.types:
        wanted = set(args.types.split(","))
        questions = [q for q in questions if q.type in wanted]
    if args.limit:
        questions = questions[: args.limit]

    logger.info("Питань: %d, режим: %s", len(questions), "end-to-end" if args.full else "retrieval")
    results = await evaluate(pipeline, questions, full=args.full)
    summary = summarize(results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Результат eval",
        "",
        f"Конфігурація: `{config.short_name()}`, top_k={config.top_k}, "
        f"модель генерації `{config.gen_model}`, embeddings `{config.embed_model}`.",
        f"Режим: {'end-to-end (пошук + генерація)' if args.full else 'лише пошук'}.",
        f"Питань: {len(questions)}.",
        "",
        "## Підсумок",
        "",
        "| Метрика | Значення |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "## Провали",
        "",
        failures_markdown(results),
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Звіт: %s", args.out)

    if args.dump:
        payload = [
            {
                "id": r.question.id,
                "type": r.question.type,
                "question": r.question.q,
                "retrieved": r.retrieved_docs,
                "hit": r.hit,
                "answered": r.answered,
                "answer": r.answer,
                "facts_ok": r.facts_ok,
                "forbidden_found": r.forbidden_found,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ]
        args.dump.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Сирі результати: %s", args.dump)


# Матриця конфігурацій: саме ці рядки і є головним артефактом проєкту —
# вони показують, що дало приріст, а що ні, замість «ми використали hybrid search».
MATRIX = [
    ("dense + fixed chunking", dict(retriever="dense", chunking="fixed")),
    ("dense + structural", dict(retriever="dense", chunking="structural")),
    ("bm25 без стемера", dict(retriever="bm25", chunking="structural", stemming=False)),
    ("bm25 зі стемером", dict(retriever="bm25", chunking="structural", stemming=True)),
    ("hybrid без стемера", dict(retriever="hybrid", chunking="structural", stemming=False)),
    ("hybrid зі стемером", dict(retriever="hybrid", chunking="structural", stemming=True)),
]

RERANK_ROW = ("hybrid + rerank", dict(retriever="hybrid", chunking="structural", rerank=True))


async def cmd_matrix(args: argparse.Namespace) -> None:
    from src.embeddings import build_embedder
    from src.evaluate import evaluate, load_questions, summarize, summary_table
    from src.llm import GeminiLLM
    from src.pipeline import RagPipeline, ingest
    from src.store import KnowledgeIndex

    base = config_from(args)
    key = _api_key(required=base.embed_provider == "gemini")
    embedder = build_embedder(base.embed_provider, base.embed_model, key)
    llm = GeminiLLM(key, base.gen_model, base.temperature, base.rpm) if key else None

    rows = list(MATRIX)
    if args.with_rerank:
        rows.append(RERANK_ROW)

    # Індекс залежить від стратегії чанкінгу — тому будуємо по одному на
    # стратегію і перевикористовуємо для всіх конфігурацій пошуку.
    indexes: dict[str, KnowledgeIndex] = {}
    for chunking in {opts.get("chunking", base.chunking) for _, opts in rows}:
        index_dir = args.index.parent / f"{args.index.name}-{base.embed_provider}-{chunking}"
        cfg = replace(base, chunking=chunking)
        if index_dir.exists() and not args.rebuild:
            indexes[chunking] = KnowledgeIndex.load(index_dir)
            logger.info("Індекс %s узято з кешу (%s)", chunking, index_dir)
        else:
            indexes[chunking] = await ingest(args.corpus, index_dir, cfg, embedder)

    questions = load_questions(
        args.questions, {c.doc_id for c in next(iter(indexes.values())).chunks}
    )
    if args.limit:
        questions = questions[: args.limit]

    table_rows = []
    for name, opts in rows:
        cfg = replace(base, **opts)
        pipeline = RagPipeline(indexes[cfg.chunking], embedder, cfg, llm)
        logger.info("→ %s", name)
        results = await evaluate(pipeline, questions, full=False)
        table_rows.append((name, summarize(results)))

    columns = ["hit@k", "recall@k", "mrr", "latency_p50_ms"]
    table = summary_table(table_rows, columns)
    print("\n" + table + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(
            [
                "# Порівняння конфігурацій пошуку",
                "",
                f"Корпус: {args.corpus}. Питань: {len(questions)}. "
                f"top_k={base.top_k}, embeddings `{base.embed_model}`.",
                "Режим — лише пошук: генерація тут не потрібна і лише додала б шуму.",
                "",
                table,
                "",
                "hit@k — хоч один очікуваний документ у топ-k; recall@k — яка частка "
                "очікуваних документів знайдена; mrr — 1/позиція першого правильного.",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Матриця: %s", args.out)


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-асистент по внутрішній базі знань")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Побудувати індекс з документів")
    p_ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    add_config_args(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Поставити питання")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="Вивести повний JSON відповіді")
    add_config_args(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_serve = sub.add_parser("serve", help="Запустити HTTP-сервіс і веб-інтерфейс")
    p_serve.add_argument("--host", default="0.0.0.0")
    # 7860 — порт, який Hugging Face Spaces очікує від Docker-контейнера.
    # Береться з оточення, щоб той самий образ працював локально і в Space.
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    add_config_args(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="Прогнати golden-набір питань")
    p_eval.add_argument("--questions", type=Path, default=DEFAULT_EVAL)
    p_eval.add_argument("--full", action="store_true", help="З генерацією (дорожче, повільніше)")
    p_eval.add_argument("--types", help="Фільтр типів через кому, напр. out_of_scope,injection")
    p_eval.add_argument("--limit", type=int, help="Обмежити кількість питань")
    p_eval.add_argument("--out", type=Path, default=Path("output/eval.md"))
    p_eval.add_argument("--dump", type=Path, help="Записати сирі результати у JSON")
    add_config_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_matrix = sub.add_parser("matrix", help="Порівняти конфігурації пошуку між собою")
    p_matrix.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p_matrix.add_argument("--questions", type=Path, default=DEFAULT_EVAL)
    p_matrix.add_argument("--limit", type=int)
    p_matrix.add_argument("--rebuild", action="store_true", help="Перебудувати індекси")
    p_matrix.add_argument("--with-rerank", action="store_true", help="Додати рядок з reranker")
    p_matrix.add_argument("--out", type=Path, default=Path("output/eval-matrix.md"))
    add_config_args(p_matrix)
    p_matrix.set_defaults(func=cmd_matrix)

    return parser.parse_args()


def main() -> None:
    _use_system_trust_store()
    load_dotenv()
    args = parse_args()
    try:
        asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("\nПерервано")
        sys.exit(130)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
