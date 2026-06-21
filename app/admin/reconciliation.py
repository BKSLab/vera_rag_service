from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.vectorstore.qdrant_client import QdrantVectorStore


async def find_active_documents_missing_in_qdrant(
    db_session: AsyncSession, vector_store: QdrantVectorStore
) -> list[tuple[str, str]]:
    """Находит версии документов, активные в реестре Postgres, но без
    соответствующих чанков в Qdrant (ING-5,
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
        которых не нашлось ни одного чанка в Qdrant. Пустой список, если
        расхождений нет.
    """
    active_documents = (
        await db_session.execute(select(Document.document_id, Document.version).where(Document.is_active.is_(True)))
    ).all()

    mismatches: list[tuple[str, str]] = []
    for document_id, version in active_documents:
        versions_in_qdrant = await vector_store.get_document_versions(document_id)
        if version not in versions_in_qdrant:
            mismatches.append((document_id, version))

    return mismatches
