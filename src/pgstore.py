"""
Postgres + pgvector як бекенд щільного пошуку.

Навіщо, якщо numpy на цьому корпусі шукає за мілісекунди (див. src/store.py)?
Бо в реальному продукті індекс живе не у файлі поруч із кодом, а в базі, яку
ділять кілька процесів: API, фоновий ingest, адмінка. pgvector дає це без
окремого сервісу: вектори лежать у тій самій Postgres, що й решта даних, і
шукаються одним SQL-запитом. Тут він підключається тим самим способом, що й
FAISS: `KnowledgeIndex.enable_pgvector(dsn)`, а решта пайплайна не знає, звідки
прийшли кандидати.

Рішення, які варто знати:

- **Джерело правди лишається файловий індекс** (`index-*/`). Таблиця в Postgres
  це його копія. `sync()` заливає її, лише якщо відбиток індексу (модель,
  кількість чанків, розмірність, час збирання) змінився, тому повторний запуск
  нічого не переписує.
- **Оператор `<#>`** (від'ємний скалярний добуток), а не `<=>` (косинусна
  відстань): вектори вже нормалізовані, і скалярний добуток дає той самий
  скор, що й numpy-бекенд. Так результати двох бекендів можна порівнювати
  напряму, що й робить scripts/bench_pgvector.py.
- **Точний пошук за замовчуванням, HNSW за прапорцем.** На сотнях чанків HNSW
  нічого не пришвидшує, але показує, як це виглядає на більшому корпусі, і
  дозволяє заміряти, скільки він губить відносно точного пошуку.
- **Тільки параметризовані запити.** Назва таблиці перевіряється регуляркою,
  бо її не можна передати параметром.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def vector_literal(vec: np.ndarray) -> str:
    """Текстовий формат pgvector: '[0.1,0.2,...]'. Без залежності від пакета pgvector."""
    flat = np.asarray(vec, dtype="float32").ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("Вектор містить NaN або нескінченність")
    return "[" + ",".join(f"{x:.7g}" for x in flat) + "]"


def index_fingerprint(meta: dict, n_chunks: int, dim: int) -> str:
    """Відбиток індексу: змінився хоч один з цих параметрів, значить таблицю треба перезалити."""
    payload = json.dumps(
        {
            "embedder": meta.get("embedder"),
            "built_at": meta.get("built_at"),
            "chunks": n_chunks,
            "dim": dim,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class PgVectorStore:
    def __init__(self, dsn: str, table: str = "kb_chunks", hnsw: bool = False, ef_search: int = 40):
        if not _TABLE_RE.match(table):
            raise ValueError(f"Недопустима назва таблиці: {table!r}")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - залежить від оточення
            raise RuntimeError("Для pgvector потрібен пакет psycopg: pip install 'psycopg[binary]'") from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self.table = table
        self.hnsw = hnsw
        self.ef_search = int(ef_search)
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)

    # --- завантаження ---

    def sync(self, chunk_ids: list[str], vectors: np.ndarray, meta: dict) -> bool:
        """
        Заливає вектори в таблицю, якщо там інший індекс. Повертає True, якщо
        заливав, і False, якщо таблиця вже відповідала індексу.
        """
        n, dim = vectors.shape
        if len(chunk_ids) != n:
            raise ValueError(f"Розбіжність: {len(chunk_ids)} чанків проти {n} векторів")
        fp = index_fingerprint(meta, n, dim)
        t = self.table
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS kb_index_meta (tbl text PRIMARY KEY, fingerprint text NOT NULL)"
            )
            cur.execute("SELECT fingerprint FROM kb_index_meta WHERE tbl = %s", (t,))
            row = cur.fetchone()
            if row and row[0] == fp:
                logger.info("pgvector: таблиця %s уже відповідає індексу (%s)", t, fp)
                self._ensure_hnsw(cur)
                return False

            # Перезаливка в одній транзакції: читач не побачить напівпорожню таблицю.
            with self._conn.transaction():
                cur.execute(f"DROP TABLE IF EXISTS {t}")
                cur.execute(
                    f"CREATE TABLE {t} (idx integer PRIMARY KEY, chunk_id text NOT NULL, "
                    f"embedding vector({int(dim)}) NOT NULL)"
                )
                with cur.copy(f"COPY {t} (idx, chunk_id, embedding) FROM STDIN") as copy:
                    for i, (cid, vec) in enumerate(zip(chunk_ids, vectors)):
                        copy.write_row((i, cid, vector_literal(vec)))
                cur.execute(
                    "INSERT INTO kb_index_meta (tbl, fingerprint) VALUES (%s, %s) "
                    "ON CONFLICT (tbl) DO UPDATE SET fingerprint = EXCLUDED.fingerprint",
                    (t, fp),
                )
            self._ensure_hnsw(cur)
            cur.execute(f"ANALYZE {t}")
        logger.info("pgvector: залито %d векторів (dim=%d) у %s", n, dim, t)
        return True

    def _ensure_hnsw(self, cur) -> None:
        if self.hnsw:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
                "USING hnsw (embedding vector_ip_ops)"
            )

    # --- пошук ---

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Повертає [(позиція чанка в індексі, скалярний добуток)], найкращі спершу."""
        top_k = max(1, min(int(top_k), 1000))
        q = vector_literal(query_vector)
        with self._conn.cursor() as cur:
            with self._conn.transaction():
                # SET LOCAL діє лише в цій транзакції. Без HNSW вимикаємо індексні
                # скани, щоб пошук був точним навіть якщо індекс лишився з минулого запуску.
                if self.hnsw:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {self.ef_search}")
                else:
                    cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute(
                    f"SELECT idx, -(embedding <#> %s::vector) AS score FROM {self.table} "
                    f"ORDER BY embedding <#> %s::vector LIMIT %s",
                    (q, q, top_k),
                )
                rows = cur.fetchall()
        return [(int(i), float(s)) for i, s in rows]

    def close(self) -> None:
        self._conn.close()
