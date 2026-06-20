import asyncio
import re
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models
from rank_bm25 import BM25Okapi

from app.core.config_logger import logger
from app.models.metadata import Audience
from app.models.schemas import SearchFilters
from app.search.fusion import rrf_fusion
from app.vectorstore.qdrant_client import CHUNK_VECTOR_NAME, TEXT_PAYLOAD_FIELD

DENSE_TOP_K = 20
SPARSE_TOP_K = 20

# Шаг постраничной выгрузки кандидатов для BM25 (Qdrant.scroll). Корпус
# небольшой (раздел 0.1 плана), поэтому держим всех отфильтрованных
# кандидатов в памяти, а не строим отдельный полнотекстовый индекс.
SCROLL_BATCH_SIZE = 256

_WORD_PATTERN = re.compile(r'\w+', re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def _audience_match_values(audience: Audience) -> list[str]:
    """`audience='both'` означает «подходит всем», поэтому фильтр по
    конкретной аудитории должен включать и её, и `both` (раздел 3 плана:
    «вопрос работодателя исключает чанки только для соискателей», но не
    общие чанки)."""
    if audience == 'both':
        return ['both']
    return [audience, 'both']


def build_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    """Строит Qdrant-фильтр по метаданным — применяется до векторного сравнения.

    Args:
        filters: Фильтры по audience/topic/source_type. None — без фильтра.

    Returns:
        `models.Filter` или None, если фильтров нет.
    """
    if filters is None:
        return None

    conditions: list[models.FieldCondition] = []
    if filters.audience is not None:
        conditions.append(
            models.FieldCondition(
                key='audience', match=models.MatchAny(any=_audience_match_values(filters.audience))
            )
        )
    if filters.topic is not None:
        conditions.append(models.FieldCondition(key='topic', match=models.MatchValue(value=filters.topic)))
    if filters.source_type is not None:
        conditions.append(
            models.FieldCondition(key='source_type', match=models.MatchValue(value=filters.source_type))
        )

    return models.Filter(must=conditions) if conditions else None


async def dense_search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    filters: SearchFilters | None = None,
    top_k: int = DENSE_TOP_K,
) -> list[tuple[str, float]]:
    """Dense-поиск (cosine) по основному вектору чанка (Этап 5).

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_vector: Эмбеддинг запроса (query-модель, не doc-модель).
        filters: Фильтры по метаданным, применяются до векторного сравнения.
        top_k: Сколько кандидатов вернуть.

    Returns:
        Список (chunk_id, score) в порядке убывания score.
    """
    result = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=CHUNK_VECTOR_NAME,
        query_filter=build_qdrant_filter(filters),
        limit=top_k,
        with_payload=False,
    )
    return [(str(point.id), point.score) for point in result.points]


async def _scroll_all_candidates(
    client: AsyncQdrantClient, collection_name: str, query_filter: models.Filter | None
) -> list[tuple[str, str]]:
    """Выгружает все точки, прошедшие фильтр по метаданным, с их текстом.

    Returns:
        Список (chunk_id, text).
    """
    candidates: list[tuple[str, str]] = []
    offset = None

    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=[TEXT_PAYLOAD_FIELD],
            with_vectors=False,
        )
        candidates.extend((str(point.id), point.payload.get(TEXT_PAYLOAD_FIELD, '')) for point in points)
        if offset is None:
            break

    return candidates


async def sparse_search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    filters: SearchFilters | None = None,
    top_k: int = SPARSE_TOP_K,
) -> list[tuple[str, float]]:
    """Sparse-поиск (BM25) по тексту чанка — закрывает точные термины
    ("статья 21", "квота 2%"), которые dense-поиск может смазать (Этап 5).

    BM25 считается в памяти над кандидатами, прошедшими фильтр по
    метаданным — без отдельного полнотекстового индекса, оправдано только
    для небольшого корпуса (раздел 0.1 плана). При росте корпуса до
    десятков тысяч чанков это нужно будет заменить на нативные
    sparse-векторы Qdrant.

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_text: Текст запроса.
        filters: Фильтры по метаданным, применяются до BM25.
        top_k: Сколько кандидатов вернуть.

    Returns:
        Список (chunk_id, score) в порядке убывания score. Пустой список,
        если после фильтра не осталось кандидатов.
    """
    candidates = await _scroll_all_candidates(client, collection_name, build_qdrant_filter(filters))
    if not candidates:
        return []

    chunk_ids = [chunk_id for chunk_id, _ in candidates]
    tokenized_corpus = [_tokenize(text) for _, text in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query_text))

    ranked = sorted(zip(chunk_ids, scores), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


@dataclass
class HybridSearchResult:
    """Результат гибридного поиска с промежуточными данными (Этап 5, 8).

    Промежуточные `dense`/`sparse` (до фьюжна) нужны не для самого поиска —
    он использует только `fused` — а для персистентного логирования запроса
    (Этап 8 плана, `SearchService._save_search_log`).
    """

    dense: list[tuple[str, float]]
    sparse: list[tuple[str, float]]
    fused: list[tuple[str, float]]


async def hybrid_search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    query_text: str,
    filters: SearchFilters | None = None,
) -> HybridSearchResult:
    """Запускает dense и sparse поиск параллельно и объединяет их через RRF (Этап 5).

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_vector: Эмбеддинг запроса (query-модель).
        query_text: Текст запроса (для BM25).
        filters: Фильтры по метаданным.

    Returns:
        `dense`/`sparse` — результаты до фьюжна, `fused` — объединённый
        список (chunk_id, rrf_score) в порядке убывания score, кандидаты
        для переранжирования на Этапе 6. RRF score переносится в финальный
        ответ API как относительный показатель уверенности (раздел 3 плана),
        реранжирование само по себе scores не даёт.
    """
    dense_results, sparse_results = await asyncio.gather(
        dense_search(client, collection_name, query_vector, filters),
        sparse_search(client, collection_name, query_text, filters),
    )
    logger.info(
        '🔍 Hybrid search: dense=%d, sparse=%d кандидатов.', len(dense_results), len(sparse_results)
    )

    dense_ids = [chunk_id for chunk_id, _ in dense_results]
    sparse_ids = [chunk_id for chunk_id, _ in sparse_results]
    fused = rrf_fusion([dense_ids, sparse_ids])
    return HybridSearchResult(dense=dense_results, sparse=sparse_results, fused=fused)


async def get_candidate_chunk_ids(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    query_text: str,
    filters: SearchFilters | None = None,
) -> list[tuple[str, float]]:
    """Обёртка над `hybrid_search`, возвращающая только объединённый список (без промежуточных данных)."""
    result = await hybrid_search(client, collection_name, query_vector, query_text, filters)
    return result.fused
