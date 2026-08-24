from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.admin.dashboard import get_dashboard_stats


@pytest.mark.parametrize(
    'connection_error',
    [ConnectionRefusedError('[Errno 111] Connect call failed'), OSError('Multiple exceptions')],
)
async def test_dashboard_degrades_when_postgres_is_unreachable(connection_error: OSError):
    """При полностью недоступном Postgres asyncpg роняет сокет-ошибку, которую
    SQLAlchemy не заворачивает в `SQLAlchemyError` — без её перехвата дашборд
    отдавал 500 вместо страницы с `postgres_ok=False` (FASTAPI_PATTERNS.md,
    раздел 9 — деградация при частичном отказе)."""
    db_session = AsyncMock()
    db_session.execute.side_effect = connection_error
    vector_store = AsyncMock()
    vector_store.client.get_collection.return_value = SimpleNamespace(points_count=952, status='green')

    stats = await get_dashboard_stats(db_session, vector_store)

    assert stats.postgres_ok is False
    assert stats.documents_total == 0
    assert stats.search_logs_total == 0
    assert stats.avg_latency_rerank_ms is None


async def test_dashboard_keeps_qdrant_stats_when_postgres_is_unreachable():
    """Отказ Postgres не должен скрывать статистику Qdrant — зависимости
    опрашиваются независимо."""
    db_session = AsyncMock()
    db_session.execute.side_effect = ConnectionRefusedError('[Errno 111] Connect call failed')
    vector_store = AsyncMock()
    vector_store.client.get_collection.return_value = SimpleNamespace(points_count=952, status='green')

    stats = await get_dashboard_stats(db_session, vector_store)

    assert stats.qdrant_ok is True
    assert stats.qdrant_points_count == 952
    assert stats.qdrant_collection_status == 'green'
    assert stats.reconciliation_mismatches is None
