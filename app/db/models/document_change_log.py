from datetime import date, datetime

from sqlalchemy import Date, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base

# Исход попытки обновления статьи. `rejected` — данные отклонены проверками
# (повтор с тем же телом бессмыслен, нужен оператор), `failed` — сорвалась
# обработка: LLM или эмбеддинги недоступны, и повтор имеет смысл.
CHANGE_APPLIED = 'applied'
CHANGE_REJECTED = 'rejected'
CHANGE_FAILED = 'failed'


class DocumentChangeLog(Base):
    """Журнал изменений документов базы знаний.

    Таблица `documents` хранит версии документа как состояние, но не отвечает
    на вопрос «что и когда поменялось в этом документе». Для юридической базы
    знаний этот вопрос основной: по нему видно, какие статьи переиндексированы,
    какой редакцией и когда изменения вступили в силу.

    Запись добавляется при каждой попытке гранулярного обновления статьи и не
    перезаписывается: это журнал, а не текущее состояние. Отклонённые и
    упавшие попытки фиксируются наравне с применёнными — журнал, который
    показывает только удачи, для контроля бесполезен: молчание в нём нельзя
    отличить от того, что сервис синхронизации ничего не присылал.
    """

    __tablename__ = 'document_change_log'

    __table_args__ = (
        Index('ix_document_change_log_document_id_created_at', 'document_id', 'created_at'),
        Index('ix_document_change_log_status_created_at', 'status', 'created_at'),
    )

    id: Mapped[int] = mapped_column(primary_key=True, comment='Уникальный идентификатор записи журнала.')
    status: Mapped[str] = mapped_column(
        String(length=20),
        nullable=False,
        default=CHANGE_APPLIED,
        comment=f'Исход попытки: {CHANGE_APPLIED} | {CHANGE_REJECTED} | {CHANGE_FAILED}.',
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='Причина отказа; пусто для применённых изменений.',
    )
    document_id: Mapped[str] = mapped_column(
        String(length=255), nullable=False, index=True, comment='Идентификатор изменённого документа.',
    )
    section_number: Mapped[str] = mapped_column(
        String(length=100), nullable=False, comment='Номер переиндексированной статьи или пункта.',
    )
    section_title: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='Заголовок статьи на момент изменения.',
    )
    category: Mapped[str] = mapped_column(
        String(length=20), nullable=False, comment='Категория источника на момент изменения.',
    )
    version: Mapped[str] = mapped_column(
        String(length=20), nullable=False, comment='Версия, под которой проиндексирована новая редакция.',
    )
    revision_date: Mapped[date] = mapped_column(
        Date, nullable=False, comment='Дата редакции, из которой взят новый текст статьи.',
    )
    amending_act_type: Mapped[str | None] = mapped_column(
        String(length=120), nullable=True, comment='Вид акта, которым внесены изменения.',
    )
    amending_act_number: Mapped[str | None] = mapped_column(
        String(length=100), nullable=True, comment='Номер акта, которым внесены изменения.',
    )
    amending_act_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment='Дата акта, которым внесены изменения.',
    )
    chunks_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment='Сколько чанков создано для новой редакции.',
    )
    superseded_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment='Сколько чанков прошлой редакции помечено неактуальными.',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment='Момент, когда изменение было применено к базе знаний.',
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChangeLog(document_id='{self.document_id}', "
            f"section_number='{self.section_number}', version='{self.version}', "
            f"status='{self.status}')>"
        )
