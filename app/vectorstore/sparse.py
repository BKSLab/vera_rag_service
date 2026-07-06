import re
import zlib
from collections import Counter

from qdrant_client import models

# Нативные sparse-векторы Qdrant с IDF-модификатором (SEARCH-1/QD-3,
# AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) вместо клиентского BM25
# (`rank_bm25`, полный scroll+пересчёт на каждый запрос). Qdrant сам считает
# IDF по статистике индекса при поиске (`Modifier.IDF`, см.
# `app/vectorstore/qdrant_client.py::ensure_collection`) — нам нужно только
# превратить текст в sparse-вектор term-frequency и при индексации, и при
# запросе, одной и той же функцией.
#
# Self-hosted Qdrant OSS не вычисляет sparse-вектор из текста на сервере
# (в отличие от управляемого Qdrant Cloud Inference) — векторизация делается
# на стороне клиента. Индексы токенов — хеш токена (`zlib.crc32`), не позиция
# в словаре: словарь не нужно строить и синхронизировать между индексацией
# и поиском, коллизии хеша на корпусе этого масштаба пренебрежимо редки.
SPARSE_VECTOR_NAME = 'bm25'

_WORD_PATTERN = re.compile(r'\w+', re.UNICODE)
_LEGAL_NORMALIZATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\bст\.', re.IGNORECASE), 'статья'),
    (re.compile(r'\bтк\s+рф\b', re.IGNORECASE), 'трудовой кодекс'),
    (re.compile(r'\bфз\b', re.IGNORECASE), 'федеральный закон'),
)


def normalize_sparse_text(text: str) -> str:
    """Нормализует частые русские юридические формы перед sparse-токенизацией."""
    normalized = text.lower().replace('ё', 'е')
    for pattern, replacement in _LEGAL_NORMALIZATION_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(normalize_sparse_text(text))


def text_to_sparse_vector(text: str) -> models.SparseVector:
    """Term-frequency sparse-вектор текста для нативного BM25-поиска Qdrant.

    Args:
        text: Текст чанка (индексация) или запроса (поиск).

    Returns:
        `SparseVector` с индексами-хешами токенов и их частотами. Пустой
        (нулевые `indices`/`values`), если в тексте нет токенов.
    """
    term_counts = Counter(tokenize(text))
    indices = [zlib.crc32(token.encode('utf-8')) for token in term_counts]
    values = [float(count) for count in term_counts.values()]
    return models.SparseVector(indices=indices, values=values)
