import asyncio
from dataclasses import dataclass
from typing import get_args

from qdrant_client import AsyncQdrantClient, models

from app.core.config_logger import logger
from app.core.settings import get_settings
from app.models.metadata import Audience, Category
from app.models.schemas import SearchFilters
from app.search.fusion import rrf_fusion
from app.vectorstore.qdrant_client import CHUNK_VECTOR_NAME
from app.vectorstore.sparse import SPARSE_VECTOR_NAME, text_to_sparse_vector

DENSE_TOP_K = 20
SPARSE_TOP_K = 20

ALL_CATEGORIES: tuple[Category, ...] = get_args(Category)


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
        filters: Фильтры по audience/topic/category. None — без фильтра.

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
    if filters.category is not None:
        conditions.append(
            models.FieldCondition(key='category', match=models.MatchValue(value=filters.category))
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


async def sparse_search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    filters: SearchFilters | None = None,
    top_k: int = SPARSE_TOP_K,
) -> list[tuple[str, float]]:
    """Sparse-поиск (BM25) по тексту чанка — закрывает точные термины
    ("статья 21", "квота 2%"), которые dense-поиск может смазать (Этап 5).

    Нативный sparse-вектор Qdrant с IDF-модификатором (SEARCH-1/QD-3,
    AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — обычный индексный
    запрос, как и `dense_search`, без полной выгрузки коллекции на каждый
    вызов (раньше — `scroll` + пересчёт `rank_bm25.BM25Okapi` с нуля).

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_text: Текст запроса.
        filters: Фильтры по метаданным, применяются до BM25.
        top_k: Сколько кандидатов вернуть.

    Returns:
        Список (chunk_id, score) в порядке убывания score. Пустой список,
        если в запросе нет токенов (пустая sparse-вектор).
    """
    query_sparse_vector = text_to_sparse_vector(query_text)
    if not query_sparse_vector.indices:
        return []

    result = await client.query_points(
        collection_name=collection_name,
        query=query_sparse_vector,
        using=SPARSE_VECTOR_NAME,
        query_filter=build_qdrant_filter(filters),
        limit=top_k,
        with_payload=False,
    )
    return [(str(point.id), point.score) for point in result.points]


async def _category_balanced_lanes(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    query_text: str,
    filters: SearchFilters | None,
) -> tuple[list[list[tuple[str, float]]], list[list[tuple[str, float]]]]:
    """Запускает dense и sparse поиск отдельно на каждую `category` (Этап 5.1 плана).

    Каждая категория — собственный ранжированный список, не общий пул кандидатов:
    при RRF-фьюжне (`app/search/fusion.py`) топ-1 малочисленной категории
    (`case_law`, `authorial`) получает тот же вклад 1/(k+1), что и топ-1
    крупной (`labor_code`) — иначе плоский top-K систематически вымывал бы
    редкие, но юридически значимые источники (раздел 4 плана, риски).
    Применяется только когда вызывающий не указал конкретную `category`
    явно — иначе балансировать нечего, ищем только в запрошенной категории
    с обычным top-K (см. `hybrid_search`).

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_vector: Эмбеддинг запроса.
        query_text: Текст запроса (для BM25).
        filters: Фильтры audience/topic (без category — она перебирается здесь).

    Returns:
        (dense_lanes, sparse_lanes) — по одному ранжированному списку
        (chunk_id, score) на каждую из 5 категорий, в порядке `ALL_CATEGORIES`.
    """
    per_category_filters = [
        SearchFilters(
            audience=filters.audience if filters else None,
            topic=filters.topic if filters else None,
            category=category,
        )
        for category in ALL_CATEGORIES
    ]

    # SEARCH-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — top-K на
    # категорию вынесен в Settings, не хардкод константой в коде: значение
    # не измерено на реальном корпусе (раздел 5.1 плана), должно быть
    # доступно для замера/правки без редеплоя кода.
    search_settings = get_settings().search
    dense_lanes, sparse_lanes = await asyncio.gather(
        asyncio.gather(
            *(
                dense_search(
                    client, collection_name, query_vector, f,
                    top_k=search_settings.dense_top_k_per_category,
                )
                for f in per_category_filters
            )
        ),
        asyncio.gather(
            *(
                sparse_search(
                    client, collection_name, query_text, f,
                    top_k=search_settings.sparse_top_k_per_category,
                )
                for f in per_category_filters
            )
        ),
    )
    return list(dense_lanes), list(sparse_lanes)


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
    """Запускает dense и sparse поиск и объединяет их через RRF (Этап 5).

    Если вызывающий не указал `filters.category` явно — поиск
    категорийно-сбалансированный (Этап 5.1 плана): dense и sparse
    запускаются отдельно на каждую из 5 категорий, каждая лента — отдельный
    ранжированный список во фьюжне, что гарантирует малочисленным категориям
    (`case_law`, `authorial`) представленность в кандидатах независимо от
    объёма корпуса `labor_code`. Если `category` указан явно — балансировать
    нечего, обычный плоский поиск с top-K (`DENSE_TOP_K`/`SPARSE_TOP_K`)
    внутри запрошенной категории.

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_vector: Эмбеддинг запроса (query-модель).
        query_text: Текст запроса (для BM25).
        filters: Фильтры по метаданным.

    Returns:
        `dense`/`sparse` — результаты до фьюжна (для логирования, Этап 8),
        `fused` — объединённый список (chunk_id, rrf_score) в порядке
        убывания score, кандидаты для переранжирования на Этапе 6. RRF score
        переносится в финальный ответ API как относительный показатель
        уверенности (раздел 3 плана), реранжирование само по себе scores не даёт.
    """
    if filters is not None and filters.category is not None:
        dense_results, sparse_results = await asyncio.gather(
            dense_search(client, collection_name, query_vector, filters),
            sparse_search(client, collection_name, query_text, filters),
        )
        fused = rrf_fusion(
            [[chunk_id for chunk_id, _ in dense_results], [chunk_id for chunk_id, _ in sparse_results]]
        )
    else:
        dense_lanes, sparse_lanes = await _category_balanced_lanes(
            client, collection_name, query_vector, query_text, filters
        )
        dense_results = [item for lane in dense_lanes for item in lane]
        sparse_results = [item for lane in sparse_lanes for item in lane]
        fused = rrf_fusion(
            [[chunk_id for chunk_id, _ in lane] for lane in dense_lanes]
            + [[chunk_id for chunk_id, _ in lane] for lane in sparse_lanes]
        )

    logger.info(
        '🔍 Hybrid search: dense=%d, sparse=%d кандидатов.', len(dense_results), len(sparse_results)
    )
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
