from __future__ import annotations

from src.bm25 import BM25Index
from src.textnorm import normalize, stem, tokenize

DOCS = [
    "Відпустки > Щорічна відпустка\nКожен працівник має 24 календарні дні відпустки.",
    "Відрядження > Добові\nДобові в Україні становлять 400 грн на день.",
    "Безпека > Паролі\nМінімальна довжина пароля — 14 символів, зберігати в 1Password.",
]


def test_tokenize_drops_punctuation_and_apostrophes():
    assert tokenize("Об'єкт, 24 дні — і все!") == ["обєкт", "24", "дні", "і", "все"]


def test_stem_matches_inflected_forms():
    # саме заради цього стемер і потрібен: питання і документ майже ніколи
    # не збігаються за відмінком
    assert stem("відпустки") == stem("відпустку") == stem("відпустка")
    assert stem("добові") == stem("добових")


def test_stem_keeps_numbers_and_short_words():
    assert stem("24") == "24"
    assert stem("дні") == "дні"  # корінь був би коротшим за MIN_STEM


def test_normalize_can_be_disabled():
    assert normalize("відпустки", stemming=False) == ["відпустки"]
    assert normalize("відпустки", stemming=True) != ["відпустки"]


def test_bm25_finds_exact_term():
    index = BM25Index(DOCS)
    top = index.search("1Password", top_k=3)
    assert top and top[0][0] == 2


def test_bm25_stemming_helps_inflected_query():
    """Запит в іншій відмінковій формі: без стемера збігу немає взагалі."""
    with_stem = BM25Index(DOCS, stemming=True).search("скільки днів відпустки", top_k=3)
    without = BM25Index(DOCS, stemming=False).search("скільки днів відпустки", top_k=3)

    assert with_stem and with_stem[0][0] == 0
    # без стемера "відпустки" збігається, але "днів" ≠ "дні" — скор помітно нижчий
    assert not without or without[0][1] < with_stem[0][1]


def test_bm25_empty_query_returns_nothing():
    assert BM25Index(DOCS).search("!!!", top_k=3) == []


def test_bm25_ignores_unknown_terms():
    assert BM25Index(DOCS).search("кубернетес інгрес", top_k=3) == []
