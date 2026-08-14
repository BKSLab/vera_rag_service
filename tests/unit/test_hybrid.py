from types import SimpleNamespace
from unittest.mock import AsyncMock

import app.search.hybrid as hybrid_module
from app.core.settings import SearchSettings
from app.models.schemas import SearchFilters
from app.search.hybrid import (
    ALL_CATEGORIES,
    _category_balanced_lanes,
    build_qdrant_filter,
    hybrid_search,
    merge_question_lanes,
)
from app.vectorstore.qdrant_client import CHUNK_VECTOR_NAME, QUESTION_VECTOR_NAMES


def test_merge_question_lanes_uses_best_rank_and_limit():
    lanes = [
        [('chunk-a', 0.91), ('chunk-shared', 0.81), ('chunk-x', 0.71)],
        [('chunk-shared', 0.95), ('chunk-b', 0.80)],
        [('chunk-c', 0.89), ('chunk-a', 0.79)],
        [('chunk-d', 0.88)],
        [('chunk-e', 0.87)],
    ]

    merged = merge_question_lanes(lanes, limit=4)

    assert merged == [
        ('chunk-a', 0.91),
        ('chunk-shared', 0.95),
        ('chunk-c', 0.89),
        ('chunk-d', 0.88),
    ]


def test_search_settings_keep_optional_search_changes_disabled_by_default():
    settings = SearchSettings()

    assert settings.question_lane_merge_limit is None
    assert settings.rerank_candidate_limit is None


def test_audience_both_filter_matches_only_both_chunks():
    qdrant_filter = build_qdrant_filter(SearchFilters(audience='both'))
    audience_condition = next(condition for condition in qdrant_filter.must if condition.key == 'audience')

    assert audience_condition.match.any == ['both']


async def test_category_balanced_question_lanes_are_grouped_by_category(monkeypatch):
    async def fake_dense_search(
        client, collection_name, query_vector, filters, top_k, vector_name=CHUNK_VECTOR_NAME
    ):
        return [(f'{filters.category}:{vector_name}', 1.0)]

    async def fake_sparse_search(client, collection_name, query_text, filters, top_k):
        return [(f'{filters.category}:sparse', 1.0)]

    monkeypatch.setattr(hybrid_module, 'dense_search', fake_dense_search)
    monkeypatch.setattr(hybrid_module, 'sparse_search', fake_sparse_search)

    dense_lanes, question_groups, sparse_lanes = await _category_balanced_lanes(
        AsyncMock(), 'vera_kb', [0.1], 'запрос', SearchFilters()
    )

    assert len(dense_lanes) == len(ALL_CATEGORIES)
    assert len(question_groups) == len(ALL_CATEGORIES)
    assert len(sparse_lanes) == len(ALL_CATEGORIES)
    for category, group in zip(ALL_CATEGORIES, question_groups, strict=True):
        assert len(group) == len(QUESTION_VECTOR_NAMES)
        assert [lane[0][0] for lane in group] == [
            f'{category}:{vector_name}' for vector_name in QUESTION_VECTOR_NAMES
        ]


async def test_hybrid_search_records_lane_type_and_category_for_candidate():
    chunk_id = '11111111-1111-1111-1111-111111111111'
    client = AsyncMock()
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id=chunk_id, score=0.9)]
    )

    result = await hybrid_search(
        client,
        'vera_kb',
        [0.1],
        'квота',
        SearchFilters(category='federal_law'),
    )

    assert result.candidate_sources[chunk_id] == {
        ('chunk-dense', 'federal_law'),
        ('question-dense', 'federal_law'),
        ('sparse', 'federal_law'),
    }
