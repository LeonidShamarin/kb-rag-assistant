"""
BM25 (Okapi) власною реалізацією.

Чому не бібліотека: алгоритм — 40 рядків, а залежність `rank_bm25` тягне свою
токенізацію, яка для української не годиться (див. textnorm.py). Тут потрібен
контроль над нормалізацією, тому дешевше написати, ніж обходити.

Роль у пайплайні: лексичний ретривер поруч із векторним. Він виграє там, де
dense систематично програє — точні коди, назви систем, суми, ідентифікатори
(`PROC`, `KB-PWNED-7731`, `1Password`), яких embedding-модель не бачила.
"""

from __future__ import annotations

import math
from collections import Counter

from src.textnorm import normalize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, texts: list[str], stemming: bool = True):
        self.stemming = stemming
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        df: Counter[str] = Counter()

        for text in texts:
            tokens = normalize(text, stemming)
            counts = Counter(tokens)
            self.docs.append(counts)
            self.lengths.append(len(tokens))
            df.update(counts.keys())

        self.n = len(texts)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        # Класичний IDF з BM25+, зі згладжуванням: не дає від'ємних ваг для
        # токенів, присутніх більш ніж у половині документів.
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = normalize(query, self.stemming)
        if not terms or self.n == 0:
            return []

        scores = [0.0] * self.n
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.docs):
                tf = counts.get(term)
                if not tf:
                    continue
                norm = 1 - B + B * (self.lengths[i] / self.avgdl if self.avgdl else 0.0)
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0.0), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]
