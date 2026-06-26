from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal['labor_code', 'case_law', 'federal_law', 'other_npa', 'authorial']
Audience = Literal['seeker', 'employer', 'both']

# Человекочитаемые подписи для выбора category в формах админки (раздел 3
# RAG_SERVICE_PLAN.md) — сами значения остаются техническими идентификаторами
# (используются как есть в фильтрах/payload Qdrant), это только подпись в UI.
CATEGORY_LABELS: dict[Category, str] = {
    'labor_code': 'ТК РФ',
    'case_law': 'Судебная практика, разъяснения Пленумов ВС РФ',
    'federal_law': 'Иные федеральные законы (не ТК РФ), например ФЗ-181',
    'other_npa': 'Подзаконные акты (постановления Правительства и т.п.)',
    'authorial': 'Авторские статьи и систематизации',
}


class ChunkMetadata(BaseModel):
    """Схема метаданных чанка — см. RAG_SERVICE_PLAN.md, раздел 3."""

    chunk_id: str = Field(..., description='Уникальный идентификатор чанка (uuid).')
    document_id: str = Field(..., description='Идентификатор документа-источника.', examples=['fz-181-art21'])
    parent_id: str = Field(
        ..., description='Единица обновления/удаления: f"{document_id}:{section_number}" или просто document_id (Этап 13 плана).',
    )
    category: Category = Field(..., description='Категория источника (раздел 3, Этап 5.1 плана).')
    source_title: str = Field(..., description='Человекочитаемое название источника.', examples=['ФЗ-181, Статья 21'])
    audience: Audience = Field(..., description='Целевая аудитория чанка.')
    topic: str = Field(..., description='Тема чанка.', examples=['quota'])
    date_added: date = Field(..., description='Дата добавления чанка в базу знаний.')
    chunk_index: int = Field(..., description='Порядковый номер чанка в пределах документа.')
    chunk_number_in_section: int = Field(
        ..., description='Локальный порядковый номер чанка внутри секции — основа детерминированного chunk_id (Этап 13).',
    )
    version: str = Field(..., description='Дата редакции нормативного акта или ревизии статьи.')
    effective_date: date = Field(..., description='Дата вступления редакции в силу.')
    effective_until: date | None = Field(
        None, description='Дата, когда эту редакцию сменила следующая; None — текущая действующая (Этап 13 плана).',
    )
    is_actual: bool = Field(
        True, description='True — действующая редакция (фильтр поиска по умолчанию); False — историческая (Этап 13 плана).',
    )
    section_number: str | None = Field(None, description='Номер статьи/пункта из структуры документа (например, "128").')
    section_title: str | None = Field(None, description='Заголовок статьи/пункта (например, "Отпуска без сохранения заработной платы").')
