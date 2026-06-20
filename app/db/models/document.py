from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Document(Base):
    """Реестр документов в базе знаний (Этап 11.1 плана).

    Qdrant хранит чанки, а не документы как сущность — без этой таблицы
    нет способа показать в админке список документов и их версий, в том
    числе после удаления чанков старой версии из Qdrant (раздел "Обновление
    документа", Этап 7: старая версия удаляется из Qdrant, но должна
    остаться видна здесь для аудита — отсюда `is_active`, а не удаление
    строки). Одна строка — одна версия одного документа, пишется при
    каждом успешном `IngestionService.ingest_document`.
    """

    __tablename__ = 'documents'

    id: Mapped[int] = mapped_column(primary_key=True, comment='Уникальный идентификатор записи реестра.')
    document_id: Mapped[str] = mapped_column(String(length=255), nullable=False, index=True, comment='Идентификатор документа-источника.')
    version: Mapped[str] = mapped_column(String(length=20), nullable=False, comment='Версия документа, под которой проиндексирован.')
    category: Mapped[str] = mapped_column(String(length=20), nullable=False, comment='Категория источника (раздел 3 плана).')
    source_title: Mapped[str] = mapped_column(String(length=255), nullable=False, comment='Человекочитаемое название источника.')
    audience: Mapped[str] = mapped_column(String(length=20), nullable=False, comment='Целевая аудитория документа.')
    topic: Mapped[str] = mapped_column(String(length=100), nullable=False, comment='Тема документа.')
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, comment='Дата вступления редакции в силу.')
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment='Признак активной (актуальной) версии — неактивные хранятся для аудита.')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment='Момент успешного ingestion этой версии.'
    )

    def __repr__(self) -> str:
        return f"<Document(document_id='{self.document_id}', version='{self.version}', is_active={self.is_active})>"
