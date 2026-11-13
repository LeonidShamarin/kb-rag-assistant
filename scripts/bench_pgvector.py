"""
Порівняння бекендів щільного пошуку на справжньому індексі: numpy, pgvector
(точний пошук) і pgvector з HNSW.

    python scripts/bench_pgvector.py --dsn postgresql://kb:kb@localhost:5434/kb

Що міряється, на тих самих питаннях golden-набору (eval/questions.yaml):
- hit@5 по документах, як у main.py eval: чи є серед 5 знайдених чанків чанк
  з очікуваного документа (лише питання, де джерело задане);
- збіг топ-5 з точним numpy-пошуком (recall@5 відносно numpy);
- затримка одного пошуку, p50 і p95, без часу кодування запиту моделлю.

Запити кодуються моделлю один раз і перевикористовуються, тож бекенди
отримують однакові вектори. Мережа не потрібна: модель локальна (st), база в Docker.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import build_embedder  # noqa: E402
from src.evaluate import load_questions  # noqa: E402
from src.pgstore import PgVectorStore  # noqa: E402
from src.store import KnowledgeIndex  # noqa: E402

TOP_K = 5
MAX_QUESTIONS = 500  # стеля на випадок, якщо golden-набір колись розростеться


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def run(name: str, search, qvecs, questions, index, exact_top) -> dict:
    times, hits, with_src, overlap = [], 0, 0, []
    for qi, (q, vec) in enumerate(zip(questions, qvecs)):
        t0 = time.perf_counter()
        res = search(vec, TOP_K)
        times.append((time.perf_counter() - t0) * 1000)
        ids = [i for i, _ in res]
        if exact_top is not None:
            overlap.append(len(set(ids) & set(exact_top[qi])) / TOP_K)
        if q.sources:
            with_src += 1
            docs = {index.chunks[i].doc_id for i in ids}
            hits += bool(docs & set(q.sources))
    return {
        "backend": name,
        "hit@5": f"{hits}/{with_src} ({100 * hits / with_src:.1f}%)" if with_src else "n/a",
        "recall@5 vs numpy": f"{100 * statistics.mean(overlap):.1f}%" if overlap else "100% (еталон)",
        "p50, мс": f"{statistics.median(times):.2f}",
        "p95, мс": f"{pct(times, 0.95):.2f}",
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--index", type=Path, default=ROOT / "index-e5s")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "pgvector-bench.md")
    args = ap.parse_args()

    index = KnowledgeIndex.load(args.index)
    known = {c.doc_id for c in index.chunks}
    questions = load_questions(args.questions, known)[:MAX_QUESTIONS]

    embedder = build_embedder("st", args.embed_model, None)
    index.assert_compatible(embedder.name, index.vectors.shape[1])
    qvecs = [await embedder.embed_query(q.q) for q in questions]

    exact_top = [[i for i, _ in index.search_dense(v, TOP_K)] for v in qvecs]
    rows = [run("numpy (точний)", index.search_dense, qvecs, questions, index, None)]

    for name, hnsw in (("pgvector, точний", False), ("pgvector, HNSW ef_search=40", True)):
        store = PgVectorStore(args.dsn, table="kb_bench", hnsw=hnsw)
        try:
            store.sync([c.chunk_id for c in index.chunks], index.vectors, index.meta)
            for v in qvecs[:10]:  # прогрів: перший запит платить за план і кеш
                store.search(v, TOP_K)
            rows.append(run(name, store.search, qvecs, questions, index, exact_top))
        finally:
            store.close()

    cols = ["backend", "hit@5", "recall@5 vs numpy", "p50, мс", "p95, мс"]
    lines = [
        "# pgvector проти numpy",
        "",
        f"Індекс `{args.index.name}`: {len(index.chunks)} чанків, вектори {index.vectors.shape[1]}-вимірні, "
        f"модель `{args.embed_model}`. Питань: {len(questions)} "
        f"(з очікуваним джерелом: {sum(1 for q in questions if q.sources)}). Пошук топ-{TOP_K}, "
        "лише щільна частина, без BM25 і без часу кодування запиту.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
        *["| " + " | ".join(r[c] for c in cols) + " |" for r in rows],
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    # Як і main.py: модель вантажиться з Hugging Face, а в мережі з TLS-інспекцією
    # корінь є лише в сховищі ОС. Перевірку сертифікатів це не вимикає.
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    asyncio.run(main())
