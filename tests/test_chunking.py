from __future__ import annotations

from src.chunking import chunk_documents, fixed_chunks, structural_chunks
from src.schema import Document
from tests.conftest import CORPUS


def test_structural_splits_by_headings():
    chunks = structural_chunks(CORPUS[0], max_chars=900, overlap=150)
    sections = [c.section for c in chunks]
    assert "Щорічна відпустка" in sections
    assert "Як подати заявку" in sections
    # H1 дублює title документа і в breadcrumb не потрапляє
    assert all("Відпустки >" not in s for s in sections)


def test_embed_text_carries_breadcrumb():
    chunk = structural_chunks(CORPUS[1], 900, 150)[0]
    assert chunk.embed_text.startswith("Відрядження > Добові")
    assert "400 грн" in chunk.embed_text


def test_fixed_chunking_has_no_sections():
    chunks = fixed_chunks(CORPUS[0], max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(c.section == "" for c in chunks)


def test_fixed_chunks_overlap():
    doc = Document(
        doc_id="d", title="d", source_path="d.md", ext=".md", text="абвгдеєжзиійклмнопрст" * 10
    )
    chunks = fixed_chunks(doc, max_chars=100, overlap=30)
    # сусідні вікна мають перетинатись — інакше overlap не працює
    assert chunks[0].text[-30:] in chunks[1].text


def test_long_section_is_windowed():
    long_para = " ".join(f"Речення номер {i} про порядок погодження." for i in range(120))
    doc = Document(
        doc_id="long",
        title="Довгий",
        source_path="long.md",
        ext=".md",
        text=f"# Довгий\n\n## Розділ\n\n{long_para}",
    )
    chunks = structural_chunks(doc, max_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)
    assert all(c.section == "Розділ" for c in chunks)


def test_horizontal_rule_splits_plain_text():
    """.txt без заголовків має різатись хоча б по роздільниках, а не одним шматком."""
    doc = Document(
        doc_id="faq",
        title="FAQ",
        source_path="faq.txt",
        ext=".txt",
        text="Питання один.\n\n---\n\nПитання два.\n\n---\n\nПитання три.",
    )
    chunks = structural_chunks(doc, 900, 150)
    assert len(chunks) == 3


def test_chunk_ids_unique_across_corpus():
    chunks = chunk_documents(CORPUS, "structural", 900, 150)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
