from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
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
    __table_args__ = (
        UniqueConstraint('document_id', 'version', name='uq_documents_document_id_version'),
    )

    id: Mapped[int] = mapped_column(primary_key=True, comment='Уникальный идентификатор записи реестра.')
    document_id: Mapped[str] = mapped_column(String(length=255), nullable=False, index=True, comment='Идентификатор документа-источника.')
    version: Mapped[str] = mapped_column(String(length=20), nullable=False, comment='Версия документа, под которой проиндексирован.')
    category: Mapped[str] = mapped_column(String(length=20), nullable=False, comment='Категория источника (раздел 3 плана).')
    source_title: Mapped[str] = mapped_column(Text, nullable=False, comment='Полное официальное наименование документа: вид акта, дата, номер и название.')
    audience: Mapped[str] = mapped_column(String(length=20), nullable=False, comment='Целевая аудитория документа.')
    topics: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, comment='Темы документа (раздел 3 плана) — пусто для labor_code/federal_law.',
    )
    act_type: Mapped[str | None] = mapped_column(
        String(length=120), nullable=True, comment='Вид акта; отсутствует у авторских материалов.',
    )
    act_number: Mapped[str | None] = mapped_column(
        String(length=100), nullable=True, comment='Номер акта; отсутствует у авторских материалов.',
    )
    act_date: Mapped[date] = mapped_column(
        Date, nullable=False, comment='Дата акта: подписания для правовых актов, публикации для авторских.',
    )
    act_title: Mapped[str] = mapped_column(
        Text, nullable=False, comment='Наименование акта без реквизитов.',
    )
    act_authority: Mapped[str | None] = mapped_column(
        String(length=255), nullable=True, comment='Принявший (подписавший) орган.',
    )
    revision_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        comment=(
            'Дата действующей редакции акта на момент этой загрузки. Постатейные обновления '
            'её не двигают: строка реестра описывает состоявшуюся загрузку, а что менялось '
            'потом — отвечает document_change_log. Не применяется к судебной практике и '
            'авторским материалам.'
        ),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment='Признак активной (актуальной) версии — неактивные хранятся для аудита.')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment='Момент успешного ingestion этой версии.'
    )

    def __repr__(self) -> str:
        return f"<Document(document_id='{self.document_id}', version='{self.version}', is_active={self.is_active})>"
