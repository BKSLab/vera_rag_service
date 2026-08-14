import asyncio
from dataclasses import dataclass, field
from typing import get_args

from qdrant_client import AsyncQdrantClient, models

from app.core.config_logger import logger
from app.core.settings import get_settings
from app.models.metadata import Audience, Category
from app.models.schemas import SearchFilters
from app.search.fusion import rrf_fusion
from app.vectorstore.qdrant_client import CHUNK_VECTOR_NAME, QUESTION_VECTOR_NAMES
from app.vectorstore.sparse import SPARSE_VECTOR_NAME, text_to_sparse_vector

DENSE_TOP_K = 20
SPARSE_TOP_K = 20

ALL_CATEGORIES: tuple[Category, ...] = get_args(Category)


def merge_question_lanes(
    lanes: list[list[tuple[str, float]]], limit: int
) -> list[tuple[str, float]]:
    """Сворачивает question_0..4 одной категории в одну ленту по лучшему рангу."""
    best_rank: dict[str, int] = {}
    score_by_id: dict[str, float] = {}
    for lane in lanes:
        for rank, (chunk_id, score) in enumerate(lane, start=1):
            if chunk_id not in best_rank or rank < best_rank[chunk_id]:
                best_rank[chunk_id] = rank
                score_by_id[chunk_id] = score

    ordered = sorted(best_rank, key=lambda chunk_id: best_rank[chunk_id])[:limit]
    return [(chunk_id, score_by_id[chunk_id]) for chunk_id in ordered]


def _audience_match_values(audience: Audience) -> list[str]:
    """`audience='both'` означает «подходит всем», поэтому фильтр по
    конкретной аудитории должен включать и её, и `both` (раздел 3 плана:
    «вопрос работодателя исключает чанки только для соискателей», но не
    общие чанки)."""
    if audience == 'both':
        return ['both']
    return [audience, 'both']


def build_qdrant_filter(filters: SearchFilters | None) -> models.Filter:
    """Строит Qdrant-фильтр по метаданным — применяется до векторного сравнения.

    `is_actual=True` добавляется всегда — скрывает исторические редакции
    статей (Этап 13 плана). Остальные условия опциональны.

    Args:
        filters: Фильтры по audience/topic/category. None — только is_actual.

    Returns:
        `models.Filter` (всегда, не None — минимум is_actual).
    """
    conditions: list[models.Condition] = [
        models.FieldCondition(key='is_actual', match=models.MatchValue(value=True))
    ]
    if filters is not None:
        if filters.audience is not None:
            conditions.append(
                models.FieldCondition(
                    key='audience', match=models.MatchAny(any=_audience_match_values(filters.audience))
                )
            )
        if filters.topic is not None:
            # Payload-поле `topics` — массив; `MatchValue` на массиве в
            # Qdrant проверяет вхождение значения в список, а не точное
            # равенство всего списка — фильтр остаётся "чанк содержит эту
            # тему", даже если у чанка их несколько.
            conditions.append(models.FieldCondition(key='topics', match=models.MatchValue(value=filters.topic)))
        if filters.category is not None:
            conditions.append(
                models.FieldCondition(key='category', match=models.MatchValue(value=filters.category))
            )

    return models.Filter(must=conditions)


async def dense_search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    filters: SearchFilters | None = None,
    top_k: int = DENSE_TOP_K,
    vector_name: str = CHUNK_VECTOR_NAME,
) -> list[tuple[str, float]]:
    """Dense-поиск (cosine) по основному вектору чанка (Этап 5).

    Args:
        client: Клиент Qdrant.
        collection_name: Имя коллекции.
        query_vector: Эмбеддинг запроса (query-модель, не doc-модель).
        filters: Фильтры по метаданным, применяются до векторного сравнения.
        top_k: Сколько кандидатов вернуть.
        vector_name: Named vector Qdrant, по которому искать (`chunk` или `question_N`).

    Returns:
        Список (chunk_id, score) в порядке убывания score.
    """
    result = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
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
) -> tuple[
    list[list[tuple[str, float]]],
    list[list[list[tuple[str, float]]]],
    list[list[tuple[str, float]]],
]:
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
        (chunk_dense_lanes, question_dense_lanes, sparse_lanes) — ранжированные
        списки (chunk_id, score): основной dense по каждой категории, пять
        question-лент, сгруппированных по категории, и sparse по категории.
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
    chunk_dense_lanes, question_dense_lanes_nested, sparse_lanes = await asyncio.gather(
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
                dense_search(
                    client, collection_name, query_vector, f,
                    top_k=search_settings.question_dense_top_k_per_category,
                    vector_name=question_vector_name,
                )
                for f in per_category_filters
                for question_vector_name in QUESTION_VECTOR_NAMES
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
    question_lanes_per_category = [
        list(question_dense_lanes_nested[index:index + len(QUESTION_VECTOR_NAMES)])
        for index in range(0, len(question_dense_lanes_nested), len(QUESTION_VECTOR_NAMES))
    ]
    return list(chunk_dense_lanes), question_lanes_per_category, list(sparse_lanes)


def _record_candidate_sources(
    target: dict[str, set[tuple[str, Category]]],
    lane: list[tuple[str, float]],
    lane_type: str,
    category: Category,
) -> None:
    for chunk_id, _ in lane:
        target.setdefault(chunk_id, set()).add((lane_type, category))


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
    candidate_sources: dict[str, set[tuple[str, Category]]] = field(default_factory=dict)


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
    candidate_sources: dict[str, set[tuple[str, Category]]] = {}
    if filters is not None and filters.category is not None:
        search_settings = get_settings().search
        dense_results, question_dense_results_nested, sparse_results = await asyncio.gather(
            dense_search(client, collection_name, query_vector, filters),
            asyncio.gather(
                *(
                    dense_search(
                        client, collection_name, query_vector, filters,
                        top_k=search_settings.question_dense_top_k,
                        vector_name=question_vector_name,
                    )
                    for question_vector_name in QUESTION_VECTOR_NAMES
                )
            ),
            sparse_search(client, collection_name, query_text, filters),
        )
        question_dense_results = [item for lane in question_dense_results_nested for item in lane]
        question_fusion_lanes = (
            list(question_dense_results_nested)
            if search_settings.question_lane_merge_limit is None
            else [
                merge_question_lanes(
                    list(question_dense_results_nested), search_settings.question_lane_merge_limit
                )
            ]
        )
        fused = rrf_fusion(
            [[chunk_id for chunk_id, _ in dense_results]]
            + [[chunk_id for chunk_id, _ in lane] for lane in question_fusion_lanes]
            + [[chunk_id for chunk_id, _ in sparse_results]]
        )
        _record_candidate_sources(candidate_sources, dense_results, 'chunk-dense', filters.category)
        for lane in question_dense_results_nested:
            _record_candidate_sources(candidate_sources, lane, 'question-dense', filters.category)
        _record_candidate_sources(candidate_sources, sparse_results, 'sparse', filters.category)
    else:
        dense_lanes, question_lanes_per_category, sparse_lanes = await _category_balanced_lanes(
            client, collection_name, query_vector, query_text, filters
        )
        dense_results = [item for lane in dense_lanes for item in lane]
        question_dense_results = [
            item
            for category_lanes in question_lanes_per_category
            for lane in category_lanes
            for item in lane
        ]
        sparse_results = [item for lane in sparse_lanes for item in lane]
        search_settings = get_settings().search
        question_fusion_lanes = (
            [lane for category_lanes in question_lanes_per_category for lane in category_lanes]
            if search_settings.question_lane_merge_limit is None
            else [
                merge_question_lanes(category_lanes, search_settings.question_lane_merge_limit)
                for category_lanes in question_lanes_per_category
            ]
        )
        fused = rrf_fusion(
            [[chunk_id for chunk_id, _ in lane] for lane in dense_lanes]
            + [[chunk_id for chunk_id, _ in lane] for lane in question_fusion_lanes]
            + [[chunk_id for chunk_id, _ in lane] for lane in sparse_lanes]
        )
        for category, dense_lane, question_lanes, sparse_lane in zip(
            ALL_CATEGORIES, dense_lanes, question_lanes_per_category, sparse_lanes, strict=True
        ):
            _record_candidate_sources(candidate_sources, dense_lane, 'chunk-dense', category)
            for lane in question_lanes:
                _record_candidate_sources(candidate_sources, lane, 'question-dense', category)
            _record_candidate_sources(candidate_sources, sparse_lane, 'sparse', category)

    logger.info(
        '🔍 Hybrid search: dense=%d, question_dense=%d, sparse=%d кандидатов.',
        len(dense_results), len(question_dense_results), len(sparse_results),
    )
    return HybridSearchResult(
        dense=[*dense_results, *question_dense_results],
        sparse=sparse_results,
        fused=fused,
        candidate_sources=candidate_sources,
    )


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
