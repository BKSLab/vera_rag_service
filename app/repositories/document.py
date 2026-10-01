from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import delete, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.exceptions.document import DocumentRepositoryError


class DocumentRepository:
    """Реестр документов в БЗ (Этап 11.1 плана) — отдельно от самих чанков
    в Qdrant, нужен только для отображения списка/истории версий в админке."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    @asynccontextmanager
    async def document_lock(self, document_id: str) -> AsyncIterator[None]:
        """Сериализует полную загрузку и обновления статей одного документа.

        Отдельная транзакция удерживает одно соединение до выхода из блока.
        Коммиты реестра и журнала её не освобождают; завершение или обрыв
        соединения снимает блокировку без session-lock, оставшегося в пуле.
        """
        async with AsyncSession(bind=self.db_session.bind) as lock_session:
            async with lock_session.begin():
                await lock_session.execute(
                    text('SELECT pg_advisory_xact_lock(hashtext(:document_id))'),
                    {'document_id': document_id},
                )
                yield

    async def save_document(self, document: Document) -> None:
        """Записывает или обновляет одну версию документа после успешного ingestion в Qdrant.

        Args:
            document: Запись реестра (без `id`/`created_at` — генерируются БД).

        Raises:
            DocumentRepositoryError: При ошибке записи в БД.
        """
        try:
            values = {
                'document_id': document.document_id,
                'version': document.version,
                'category': document.category,
                'source_title': document.source_title,
                'audience': document.audience,
                'topics': document.topics,
                'act_type': document.act_type,
                'act_number': document.act_number,
                'act_date': document.act_date,
                'act_title': document.act_title,
                'act_authority': document.act_authority,
                'revision_date': document.revision_date,
                'is_active': document.is_active,
            }
            statement = (
                insert(Document)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=['document_id', 'version'],
                    set_={
                        'category': values['category'],
                        'source_title': values['source_title'],
                        'audience': values['audience'],
                        'topics': values['topics'],
                        'act_type': values['act_type'],
                        'act_number': values['act_number'],
                        'act_date': values['act_date'],
                        'act_title': values['act_title'],
                        'act_authority': values['act_authority'],
                        'revision_date': values['revision_date'],
                        'is_active': values['is_active'],
                    },
                )
            )
            await self.db_session.execute(statement)
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
