from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.exceptions.document import DocumentRepositoryError


class DocumentRepository:
    """Реестр документов в БЗ (Этап 11.1 плана) — отдельно от самих чанков
    в Qdrant, нужен только для отображения списка/истории версий в админке."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def save_document(self, document: Document) -> None:
        """Записывает одну версию документа после успешного ingestion в Qdrant.

        Args:
            document: Запись реестра (без `id`/`created_at` — генерируются БД).

        Raises:
            DocumentRepositoryError: При ошибке записи в БД.
        """
        try:
            self.db_session.add(document)
            await self.db_session.commit()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise DocumentRepositoryError(error_details=str(error)) from error

    async def mark_versions_inactive(self, document_id: str, versions: list[str]) -> None:
        """Помечает версии документа неактивными после удаления их чанков из Qdrant.

        Строки не удаляются — `is_active=False` сохраняет историю версий
        для аудита (раздел 3 плана: "какая редакция была проиндексирована
        на момент конкретного ответа агента").

        Args:
            document_id: Идентификатор документа.
            versions: Версии, чьи чанки только что удалены из Qdrant.

        Raises:
            DocumentRepositoryError: При ошибке записи в БД.
        """
        if not versions:
            return

        try:
            await self.db_session.execute(
                update(Document)
                .where(Document.document_id == document_id, Document.version.in_(versions))
                .values(is_active=False)
            )
            await self.db_session.commit()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise DocumentRepositoryError(error_details=str(error)) from error
