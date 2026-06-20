from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.db.models.search_log import SearchLog
from app.vectorstore.qdrant_client import QdrantVectorStore


@dataclass
class DashboardStats:
    """Сводка для мониторинга сервиса через `/admin` (расширение Этапа 11
    плана) — без неё единственный способ оценить состояние БЗ и поиска —
    листать сырые списки в `DocumentAdmin`/`SearchLogAdmin` по одной строке."""

    postgres_ok: bool
    documents_total: int
    documents_active: int
    distinct_documents: int
    qdrant_ok: bool
    qdrant_points_count: int | None
    qdrant_collection_status: str | None
    search_logs_total: int
    avg_latency_embed_query_ms: float | None
    avg_latency_hybrid_search_ms: float | None
    avg_latency_rerank_ms: float | None
    last_search_at: datetime | None


async def get_dashboard_stats(db_session: AsyncSession, vector_store: QdrantVectorStore) -> DashboardStats:
    """Собирает сводную статистику Postgres + Qdrant для дашборда админки.

    Каждая из двух зависимостей опрашивается независимо — недоступность
    одной (например, Qdrant) не должна скрывать статистику по другой
    (FASTAPI_PATTERNS.md, раздел 9 — деградация при частичном отказе).
    """
    postgres_ok = True
    documents_total = documents_active = distinct_documents = 0
    search_logs_total = 0
    avg_embed = avg_hybrid = avg_rerank = None
    last_search_at = None

    try:
        documents_total = (await db_session.execute(select(func.count()).select_from(Document))).scalar_one()
        documents_active = (
            await db_session.execute(select(func.count()).select_from(Document).where(Document.is_active.is_(True)))
        ).scalar_one()
        distinct_documents = (
            await db_session.execute(select(func.count(func.distinct(Document.document_id))))
        ).scalar_one()

        search_logs_total = (await db_session.execute(select(func.count()).select_from(SearchLog))).scalar_one()
        avg_embed, avg_hybrid, avg_rerank, last_search_at = (
            await db_session.execute(
                select(
                    func.avg(SearchLog.latency_embed_query_ms),
                    func.avg(SearchLog.latency_hybrid_search_ms),
                    func.avg(SearchLog.latency_rerank_ms),
                    func.max(SearchLog.created_at),
                )
            )
        ).one()
    except SQLAlchemyError:
        postgres_ok = False

    qdrant_ok = True
    qdrant_points_count = None
    qdrant_collection_status = None
    try:
        collection_info = await vector_store.client.get_collection(vector_store.collection_name)
        qdrant_points_count = collection_info.points_count
        qdrant_collection_status = collection_info.status
    except Exception:  # noqa: BLE001 — дашборд должен деградировать на любой сбой Qdrant-клиента, не только сетевой
        qdrant_ok = False

    return DashboardStats(
        postgres_ok=postgres_ok,
        documents_total=documents_total,
        documents_active=documents_active,
        distinct_documents=distinct_documents,
        qdrant_ok=qdrant_ok,
        qdrant_points_count=qdrant_points_count,
        qdrant_collection_status=qdrant_collection_status,
        search_logs_total=search_logs_total,
        avg_latency_embed_query_ms=avg_embed,
        avg_latency_hybrid_search_ms=avg_hybrid,
        avg_latency_rerank_ms=avg_rerank,
        last_search_at=last_search_at,
    )
