from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Topic(Base):
    """Справочник тем документов (раздел 3 плана) — управляется через админку,
    не хардкодится в коде (в отличие от `Category`/`Audience`), потому что
    список тем должен пополняться контент-менеджером без деплоя кода.

    Осмыслен только для узких по предмету категорий (`other_npa`, `case_law`,
    `authorial`) — широкие кодексы/законы (`labor_code`, `federal_law`)
    регулируют десятки разных предметов, свести их к одной-двум темам
    означало бы соврать или обесценить фильтр (обсуждение с пользователем
    2026-07-08). Проверяется в `IngestionService`, не здесь — эта таблица
    только справочник разрешённых значений.
    """

    __tablename__ = 'topics'

    id: Mapped[int] = mapped_column(primary_key=True, comment='Уникальный идентификатор темы.')
    name: Mapped[str] = mapped_column(String(length=100), nullable=False, unique=True, comment='Название темы (например, "квотирование").')
    comment: Mapped[str | None] = mapped_column(
        String(length=255), nullable=True, comment='Пояснение для админки — не выводится в API/поиске.'
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment='Момент добавления темы в справочник.'
    )

    def __repr__(self) -> str:
        return f"<Topic(name='{self.name}')>"
