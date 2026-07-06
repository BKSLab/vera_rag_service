from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.vectorstore.qdrant_client import QdrantVectorStore


async def find_active_documents_missing_in_qdrant(
    db_session: AsyncSession, vector_store: QdrantVectorStore
) -> list[tuple[str, str]]:
    """Находит версии документов, активные в Postgres, но без актуальных чанков в Qdrant (ING-5,
    AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md).

    Дешёвая часть `ingest_document` (запись реестра) и дорогая (upsert в
    Qdrant) выполняются раздельно, без распределённой транзакции — реестр
    может расходиться с фактическим содержимым Qdrant при частичном отказе
    (обрыв соединения на середине `upsert_chunks`, например). Минимальная
    сверка вместо полноценного outbox/saga — для текущего масштаба
    избыточен.

    Args:
        db_session: Сессия Postgres.
        vector_store: Клиент Qdrant.

    Returns:
        Список (document_id, version) для активных записей реестра, у
        которых нет ни одного поисково-доступного чанка (`is_actual=True`)
        в Qdrant. Пустой список, если расхождений нет.
    """
    active_documents = (
        await db_session.execute(select(Document.document_id, Document.version).where(Document.is_active.is_(True)))
    ).all()

    mismatches: list[tuple[str, str]] = []
    for document_id, version in active_documents:
        actual_chunks_count = await vector_store.count_actual_document_chunks(document_id, version)
        if actual_chunks_count == 0:
            mismatches.append((document_id, version))

    return mismatches
