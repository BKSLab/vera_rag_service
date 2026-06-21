from app.core.config_logger import logger
from app.exceptions.document import DocumentRepositoryError
from app.repositories.document import DocumentRepository
from app.vectorstore.qdrant_client import QdrantVectorStore


class DocumentsService:
    """Управление документами в базе знаний на уровне всего документа (Этап 7).

    Единая точка удаления документа — общая для публичного API
    (`DELETE /document/{id}`) и админки (`DocumentAdmin.delete_model`), чтобы
    оба пути гарантированно давали одинаковый результат (ARCH-4,
    AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — раньше публичный API не
    трогал реестр `documents` в Postgres, только админка.
    """

    def __init__(self, vector_store: QdrantVectorStore, document_repository: DocumentRepository):
        self.vector_store = vector_store
        self.document_repository = document_repository

    async def delete_document(self, document_id: str, version: str | None = None) -> None:
        """Удаляет документ из Qdrant и реестра Postgres.

        Args:
            document_id: Идентификатор документа.
            version: Если задано — удаляется только эта версия (используется
                админкой при удалении одной строки реестра — одна строка
                реестра — одна версия). Если не задано — удаляются все версии
                (публичный API).
        """
        await self.vector_store.delete_document(document_id, version=version)
        try:
            await self.document_repository.delete_document(document_id, version=version)
        except DocumentRepositoryError as error:
            logger.warning(
                '⚠️ Чанки документа %s удалены из Qdrant, но не из реестра Postgres. Детали: %s',
                document_id, error,
            )
