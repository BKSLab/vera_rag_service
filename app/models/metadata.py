from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal['labor_code', 'case_law', 'federal_law', 'other_npa', 'authorial']
Audience = Literal['seeker', 'employer', 'both']


class ChunkMetadata(BaseModel):
    """Схема метаданных чанка — см. RAG_SERVICE_PLAN.md, раздел 3."""

    chunk_id: str = Field(..., description='Уникальный идентификатор чанка (uuid).')
    document_id: str = Field(..., description='Идентификатор документа-источника.', examples=['fz-181-art21'])
    category: Category = Field(..., description='Категория источника (раздел 3, Этап 5.1 плана).')
    source_title: str = Field(..., description='Человекочитаемое название источника.', examples=['ФЗ-181, Статья 21'])
    audience: Audience = Field(..., description='Целевая аудитория чанка.')
    topic: str = Field(..., description='Тема чанка.', examples=['quota'])
    date_added: date = Field(..., description='Дата добавления чанка в базу знаний.')
    chunk_index: int = Field(..., description='Порядковый номер чанка в пределах документа.')
    version: str = Field(..., description='Дата редакции нормативного акта или ревизии статьи.')
    effective_date: date = Field(..., description='Дата вступления редакции в силу.')
