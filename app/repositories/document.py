from sqlalchemy import delete, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.exceptions.document import DocumentRepositoryError


class DocumentRepository:
    """Реестр документов в БЗ (Этап 11.1 плана) — отдельно от самих чанков
    в Qdrant, нужен только для отображения списка/истории версий в админке."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def acquire_document_lock(self, document_id: str) -> None:
        """Сессионный Postgres advisory lock на `document_id` (ING-2) —
        серилизует конкурентные `ingest_document` для одного и того же
        документа (двойной клик в форме загрузки, повторный retry клиента
        поверх ещё выполняющегося запроса). Сессионный, не транзакционный
        (`pg_advisory_lock`, не `pg_advisory_xact_lock`) — должен держаться
        на протяжении всего `ingest_document`, который коммитит несколько
        раз (`save_document`/`mark_versions_inactive`), а не одной
        транзакции. Снимается явно через `release_document_lock` —
        вызывающий код обязан сделать это в `finally`.
        """
        await self.db_session.execute(
            text('SELECT pg_advisory_lock(hashtext(:document_id))'), {'document_id': document_id}
        )

    async def release_document_lock(self, document_id: str) -> None:
        await self.db_session.execute(
            text('SELECT pg_advisory_unlock(hashtext(:document_id))'), {'document_id': document_id}
        )

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

    async def delete_document(self, document_id: str, version: str | None = None) -> None:
        """Удаляет строки реестра документа (опционально — только указанной версии).

        Используется `DocumentsService.delete_document` (ARCH-4,
        AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — единая точка
        удаления документа, общая для публичного API и админки, чтобы
        реестр не расходился с фактическим содержимым Qdrant.

        Args:
            document_id: Идентификатор документа.
            version: Если задано — удаляется только строка этой версии.

        Raises:
            DocumentRepositoryError: При ошибке записи в БД.
        """
        try:
            conditions = [Document.document_id == document_id]
            if version is not None:
                conditions.append(Document.version == version)
            await self.db_session.execute(delete(Document).where(*conditions))
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
