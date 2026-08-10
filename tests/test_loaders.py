from __future__ import annotations

import pytest

from src.loaders import UnsupportedFormat, load_corpus, load_file


def test_markdown_title_from_h1(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Відпустки\n\nТекст.", encoding="utf-8")

    doc = load_file(path)
    assert doc.doc_id == "doc"
    assert doc.title == "Відпустки"
    assert doc.ext == ".md"


def test_plain_text_title_from_first_line(tmp_path):
    path = tmp_path / "faq.txt"
    path.write_text("FAQ підтримки\n\nПитання один.", encoding="utf-8")
    assert load_file(path).title == "FAQ підтримки"


def test_excess_blank_lines_collapsed(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# A\n\n\n\n\nБ", encoding="utf-8")
    assert "\n\n\n" not in load_file(path).text


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"...")
    with pytest.raises(UnsupportedFormat):
        load_file(path)


def test_docx_headings_become_markdown(tmp_path):
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_heading("Політика", level=1)
    document.add_heading("Ліміти", level=2)
    document.add_paragraph("До 200 EUR погоджує керівник команди.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Сума"
    table.rows[0].cells[1].text = "Хто погоджує"
    path = tmp_path / "polityka.docx"
    document.save(str(path))

    doc = load_file(path)
    assert "# Політика" in doc.text
    assert "## Ліміти" in doc.text
    assert "Сума | Хто погоджує" in doc.text


def test_corpus_skips_unsupported_and_broken(tmp_path):
    (tmp_path / "ok.md").write_text("# Ок\n\nтекст", encoding="utf-8")
    (tmp_path / "ignore.xlsx").write_bytes(b"binary")
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    docs = load_corpus(tmp_path)

    assert [d.doc_id for d in docs] == ["ok"]


def test_corpus_reads_nested_directories(tmp_path):
    nested = tmp_path / "hr" / "2026"
    nested.mkdir(parents=True)
    (nested / "vidpustky.md").write_text("# Відпустки\n\n24 дні", encoding="utf-8")

    docs = load_corpus(tmp_path)
    assert [d.doc_id for d in docs] == ["vidpustky"]


def test_real_corpus_loads():
    """Справжній корпус проєкту має читатись без винятків — і .md, і .txt."""
    from pathlib import Path

    docs = load_corpus(Path("data/corpus"))
    assert len(docs) >= 13
    assert {d.ext for d in docs} == {".md", ".txt"}
    assert all(d.title and d.text for d in docs)
