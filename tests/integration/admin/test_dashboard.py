from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from app.admin.dashboard import get_dashboard_stats
from app.db.models.search_log import SearchLog


def make_search_log(latency_embed_query_ms: float, created_at: datetime) -> SearchLog:
    return SearchLog(
        request_id=str(uuid4()),
        query='квота на инвалидов',
        dense_candidates=[],
        sparse_candidates=[],
        rrf_candidates=[],
        reranked_chunk_ids=[],
        final_response=[],
        latency_embed_query_ms=latency_embed_query_ms,
        latency_hybrid_search_ms=latency_embed_query_ms,
        latency_rerank_ms=latency_embed_query_ms,
        created_at=created_at,
    )


async def test_get_dashboard_stats_averages_only_recent_window(db_session):
    """LOG-3 — latency на дашборде усредняется по последним 24 часам, не по
    всей истории `search_logs` (иначе стоимость запроса растёт вместе с
    таблицей при каждом открытии страницы)."""
    now = datetime.now(UTC)
    db_session.add(make_search_log(latency_embed_query_ms=1000.0, created_at=now - timedelta(days=10)))
    db_session.add(make_search_log(latency_embed_query_ms=100.0, created_at=now - timedelta(hours=1)))
    await db_session.commit()

    fake_vector_store = AsyncMock()
    fake_vector_store.client.get_collection.side_effect = Exception('not relevant to this test')

    stats = await get_dashboard_stats(db_session, fake_vector_store)

    assert stats.avg_latency_embed_query_ms == 100.0
    assert stats.search_logs_total == 2
