from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document_change_log import DocumentChangeLog
from app.exceptions.document import DocumentChangeLogRepositoryError


class DocumentChangeLogRepository:
    """Журнал изменений документов базы знаний."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def add_entry(self, entry: DocumentChangeLog) -> None:
        """Добавляет запись в журнал изменений документа.

        Args:
            entry: Запись журнала (без `id`/`created_at` — генерируются БД).

        Raises:
            DocumentChangeLogRepositoryError: При ошибке записи в БД.
        """
        try:
            self.db_session.add(entry)
            await self.db_session.commit()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise DocumentChangeLogRepositoryError(str(error)) from error

    async def get_document_history(self, document_id: str, limit: int = 100) -> list[DocumentChangeLog]:
        """Возвращает историю изменений одного документа, новые записи первыми.

        Args:
            document_id: Идентификатор документа.
            limit: Максимальное количество записей.

        Returns:
            Записи журнала изменений.

        Raises:
            DocumentChangeLogRepositoryError: При ошибке чтения из БД.
        """
        statement = (
            select(DocumentChangeLog)
            .where(DocumentChangeLog.document_id == document_id)
            .order_by(DocumentChangeLog.created_at.desc(), DocumentChangeLog.id.desc())
            .limit(limit)
        )
        try:
            result = await self.db_session.execute(statement)
            return list(result.scalars().all())
        except SQLAlchemyError as error:
            raise DocumentChangeLogRepositoryError(str(error)) from error
