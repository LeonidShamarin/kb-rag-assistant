"""
Розбиття документів на чанки.

Дві стратегії — і це не «на смак», а предмет виміру в eval-матриці:

- **fixed** — вікна фіксованої довжини по символах. Базова лінія з туторіалів:
  ріже посеред речення й посеред таблиці, губить заголовок секції.
- **structural** — ріже по заголовках markdown, і лише завеликі секції додатково
  ділить по абзацах із перекриттям. Кожен чанк несе breadcrumb заголовків.

Різниця найпомітніша на регламентах: у них відповідь майже завжди — це один
розділ, а ключове слово («добові», «ліміт») стоїть у заголовку, а не в тілі.
"""

from __future__ import annotations

import re

from src.schema import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_HRULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _window_paragraphs(paras: list[str], max_chars: int, overlap: int) -> list[str]:
    """
    Складає абзаци у вікна до max_chars. Перекриття — теж абзацами, а не
    символами: обрізати абзац посередині заради «рівно 150 символів overlap»
    означає віддати моделі уривок без початку думки.
    """
    windows: list[str] = []
    current: list[str] = []
    size = 0

    for para in paras:
        # окремий абзац довший за вікно — ріжемо його по реченнях
        if len(para) > max_chars:
            if current:
                windows.append("\n\n".join(current))
                current, size = [], 0
            windows.extend(_split_long(para, max_chars))
            continue

        if size + len(para) > max_chars and current:
            windows.append("\n\n".join(current))
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev)
            current, size = tail, tail_size

        current.append(para)
        size += len(para) + 2

    if current:
        windows.append("\n\n".join(current))
    return windows


def _split_long(para: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?;])\s+", para)
    out: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > max_chars:
            out.append(buf.strip())
            buf = ""
        buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf.strip())
    return out or [para[:max_chars]]


def _sections(doc: Document) -> list[tuple[str, str]]:
    """
    Розкладає документ на (breadcrumb, текст) за заголовками markdown.

    Горизонтальна лінія (`---`) теж рахується межею — так коректно ріжеться
    експорт зі старого хелпдеску (.txt), де заголовків немає взагалі.
    """
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            crumbs = [t for lvl, t in stack if not (lvl == 1 and t == doc.title)]
            sections.append((" > ".join(crumbs), text))
        buf.clear()

    for line in doc.text.split("\n"):
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
            continue
        if _HRULE.match(line):
            flush()
            continue
        buf.append(line)

    flush()
    return sections


def structural_chunks(doc: Document, max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, text in _sections(doc):
        for window in _window_paragraphs(_split_paragraphs(text), max_chars, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section=section,
                    text=window,
                    ordinal=len(chunks),
                )
            )
    return chunks


def fixed_chunks(doc: Document, max_chars: int, overlap: int) -> list[Chunk]:
    """Базова лінія для порівняння: сліпі вікна по символах, без структури."""
    text = doc.text
    step = max(1, max_chars - overlap)
    chunks: list[Chunk] = []
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                section="",
                text=piece,
                ordinal=len(chunks),
            )
        )
    return chunks


def chunk_documents(
    docs: list[Document], strategy: str, max_chars: int, overlap: int
) -> list[Chunk]:
    fn = structural_chunks if strategy == "structural" else fixed_chunks
    out: list[Chunk] = []
    for doc in docs:
        out.extend(fn(doc, max_chars, overlap))
    return out
