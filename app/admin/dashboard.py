from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.reconciliation import find_active_documents_missing_in_qdrant
from app.db.models.document import Document
from app.db.models.search_log import SearchLog
from app.vectorstore.qdrant_client import QdrantVectorStore

# ING-5 — сверка реестра и Qdrant стоит один scroll-запрос на активный
# документ; при большом количестве активных документов это сделало бы
# открытие дашборда слишком дорогим. Сверка пропускается (не падает) выше
# этого порога — тогда сверку нужно гонять отдельным офлайн-скриптом, не на
# каждый просмотр дашборда.
RECONCILIATION_MAX_ACTIVE_DOCUMENTS = 200

# LOG-3 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — без окна latency
# усреднялась бы по всей истории `search_logs`, и стоимость самого запроса
# дашборда росла бы вместе с таблицей при каждом открытии страницы. Полное
# партиционирование/retention для самой таблицы — отдельная задача
# (см. LOG-3 в плане), это инфраструктурное решение по объёму хранения, не
# чинится одним запросом; здесь — только ограничение стоимости конкретно
# дашборда.
DASHBOARD_RECENT_WINDOW = timedelta(hours=24)


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
    avg_latency_query_expansion_ms: float | None
    avg_latency_embed_query_ms: float | None
    avg_latency_hybrid_search_ms: float | None
    avg_latency_rerank_ms: float | None
    last_search_at: datetime | None
    reconciliation_mismatches: list[tuple[str, str]] | None
    """ING-5 — активные версии реестра без чанков в Qdrant. `None`, если
    сверка пропущена (Postgres/Qdrant недоступны или слишком много активных
    документов — см. `RECONCILIATION_MAX_ACTIVE_DOCUMENTS`)."""


async def get_dashboard_stats(db_session: AsyncSession, vector_store: QdrantVectorStore) -> DashboardStats:
    """Собирает сводную статистику Postgres + Qdrant для дашборда админки.

    Каждая из двух зависимостей опрашивается независимо — недоступность
    одной (например, Qdrant) не должна скрывать статистику по другой
    (FASTAPI_PATTERNS.md, раздел 9 — деградация при частичном отказе).
    """
    postgres_ok = True
    documents_total = documents_active = distinct_documents = 0
    search_logs_total = 0
    avg_expansion = avg_embed = avg_hybrid = avg_rerank = None
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
        recent_window_start = datetime.now(UTC) - DASHBOARD_RECENT_WINDOW
        avg_expansion, avg_embed, avg_hybrid, avg_rerank, last_search_at = (
            await db_session.execute(
                select(
                    func.avg(SearchLog.latency_query_expansion_ms),
                    func.avg(SearchLog.latency_embed_query_ms),
                    func.avg(SearchLog.latency_hybrid_search_ms),
                    func.avg(SearchLog.latency_rerank_ms),
                    func.max(SearchLog.created_at),
                ).where(SearchLog.created_at >= recent_window_start)
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

    reconciliation_mismatches: list[tuple[str, str]] | None = None
    if postgres_ok and qdrant_ok and documents_active <= RECONCILIATION_MAX_ACTIVE_DOCUMENTS:
        try:
            reconciliation_mismatches = await find_active_documents_missing_in_qdrant(db_session, vector_store)
        except Exception:  # noqa: BLE001 — сверка диагностическая, не должна ронять дашборд
            reconciliation_mismatches = None

    return DashboardStats(
        postgres_ok=postgres_ok,
        documents_total=documents_total,
        documents_active=documents_active,
        distinct_documents=distinct_documents,
        qdrant_ok=qdrant_ok,
        qdrant_points_count=qdrant_points_count,
        qdrant_collection_status=qdrant_collection_status,
        search_logs_total=search_logs_total,
        avg_latency_query_expansion_ms=avg_expansion,
        avg_latency_embed_query_ms=avg_embed,
        avg_latency_hybrid_search_ms=avg_hybrid,
        avg_latency_rerank_ms=avg_rerank,
        last_search_at=last_search_at,
        reconciliation_mismatches=reconciliation_mismatches,
    )
