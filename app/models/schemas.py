from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.metadata import Audience, Category

# Верхняя граница `raw_text` одного документа (API-3, RAG_SERVICE_PLAN.md,
# раздел 7). Без лимита один запрос мог бы запустить неограниченное число
# платных вызовов LLM/embedding API. Изначальная оценка (~840K символов,
# по статистике Word) была занижена: реальный ТК РФ, извлечённый из .docx
# через наш парсер (app/ingestion/extract.py — параграфы+таблицы+колонтитулы,
# без агрессивного схлопывания пробелов), дал 1 028 313 символов — с запасом
# на это и на будущие более крупные/обновлённые редакции взят порог 2M.
MAX_RAW_TEXT_LENGTH = 2_000_000

# Расширение запроса перед поиском (раздел 8 плана) — ограничивает веер
# параллельных hybrid_search на один запрос пользователя (decomposed
# sub_question × rephrasing): максимум MAX_SUB_QUESTIONS подвопросов,
# каждый — исходная формулировка + до MAX_REPHRASINGS_PER_SUB_QUESTION
# юридических переформулировок. Худший случай — 3 × 2 = 6 параллельных
# hybrid_search вместо одного, согласовано как приемлемый верхний предел
# latency/нагрузки на Qdrant.
MAX_SUB_QUESTIONS = 3
MAX_REPHRASINGS_PER_SUB_QUESTION = 1


class Chunk(BaseModel):
    """Чанк — единица индексации, результат Этапа 2 (иерархический чанкинг).

    Метаданные секции (`section_number`, `section_title`), из которой получен
    чанк, переносятся как контекст шире самого чанка — нужны на этапе
    генерации ответа LLM (см. RAG_SERVICE_PLAN.md, Этап 2).
    """

    chunk_id: str = Field(..., description='Уникальный идентификатор чанка (uuid).')
    chunk_index: int = Field(..., description='Сквозной порядковый номер чанка в пределах документа.')
    document_id: str = Field(..., description='Идентификатор документа-источника.')
    category: Category = Field(..., description='Категория источника (раздел 3 плана).')
    section_index: int = Field(..., description='Номер секции-источника в документе.')
    section_number: str | None = Field(None, description='Номер статьи/пункта (для law).')
    section_title: str = Field(..., description='Заголовок секции-источника.')
    text: str = Field(..., description='Текст чанка.')


class Section(BaseModel):
    """Секция документа — промежуточный результат препроцессинга (Этап 1).

    Документ разбивается на секции (статья закона / раздел статьи) до того,
    как секция дальше делится на чанки (Этап 2). Структурные метаданные
    секции (`section_number`, `section_title`) переносятся как контекст
    в метаданные каждого чанка, полученного из этой секции.
    """

    document_id: str = Field(..., description='Идентификатор документа-источника.')
    category: Category = Field(..., description='Категория источника (раздел 3 плана).')
    section_index: int = Field(..., description='Порядковый номер секции в документе.')
    section_number: str | None = Field(
        None, description='Номер статьи/пункта (для law) — извлекается из текста, если есть.'
    )
    section_title: str = Field(..., description='Заголовок секции (название статьи или заголовок раздела).')
    text: str = Field(..., description='Текст секции после очистки.')


class ChunkEnrichmentResult(BaseModel):
    """Structured output LLM на Этапе 3 (обогащение чанков).

    Валидируется клиентом сразу после извлечения контента ответа —
    ошибка валидации уходит в retry LlmClient, а не сразу в финальный отказ.
    """

    synthetic_title: str = Field(..., min_length=1, description='Синтетический заголовок чанка.')
    hypothetical_questions: list[str] = Field(
        ..., min_length=3, max_length=5, description='3–5 гипотетических вопросов к чанку.'
    )

    @field_validator('hypothetical_questions', mode='before')
    @classmethod
    def drop_empty_questions(cls, value: object) -> object:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return value


class EnrichedChunk(BaseModel):
    """Чанк, обогащённый синтетическим заголовком и гипотетическими вопросами."""

    chunk: Chunk = Field(..., description='Исходный чанк (Этап 2).')
    synthetic_title: str = Field(..., description='Синтетический заголовок чанка.')
    hypothetical_questions: list[str] = Field(..., description='Гипотетические вопросы к чанку.')


class EmbeddedChunk(BaseModel):
    """Обогащённый чанк с векторами эмбеддингов (Этап 4).

    `chunk_vector` — эмбеддинг текста "заголовок + текст чанка"
    (`build_embedding_text`). `question_vectors` — отдельные эмбеддинги
    каждого гипотетического вопроса, в том же порядке, что
    `enriched_chunk.hypothetical_questions` — индексируются как
    дополнительные векторы той же точки в Qdrant.
    """

    enriched_chunk: EnrichedChunk = Field(..., description='Обогащённый чанк (Этап 3).')
    chunk_vector: list[float] = Field(..., description='Эмбеддинг заголовка и текста чанка.')
    question_vectors: list[list[float]] = Field(
        ..., description='Эмбеддинги гипотетических вопросов, по порядку.'
    )


class DocumentMetadataInput(BaseModel):
    """Метаданные документа, поставляемые при ingestion (см. раздел 3 плана).

    Эти поля не вычисляются ни на одном из этапов ingestion-пайплайна —
    их задаёт вызывающая сторона (Expert/`/ingest`) на уровне документа,
    а не отдельного чанка, и они одинаковы для всех чанков документа.
    """

    source_title: str = Field(..., description='Человекочитаемое название источника.')
    audience: Audience = Field(..., description='Целевая аудитория.')
    topic: str = Field(..., description='Тема документа.')
    version: str = Field(..., description='Дата редакции документа.')
    effective_date: date = Field(..., description='Дата вступления редакции в силу.')


class SearchFilters(BaseModel):
    """Фильтры по метаданным, применяемые до векторного сравнения (Этап 5).

    `audience` — ключевое поле: вопрос работодателя исключает чанки только
    для соискателей (раздел 3 плана). Все поля опциональны — пустой фильтр
    означает поиск по всей базе знаний.
    """

    audience: Audience | None = Field(None, description='Фильтр по целевой аудитории.')
    topic: str | None = Field(None, description='Фильтр по теме.')
    category: Category | None = Field(None, description='Фильтр по категории источника.')


class SearchResultChunk(BaseModel):
    """Один чанк в результатах поиска — после RRF fusion (Этап 5)."""

    chunk_id: str = Field(..., description='Идентификатор чанка.')
    text: str = Field(..., description='Текст чанка.')
    synthetic_title: str = Field(..., description='Синтетический заголовок чанка.')
    source_title: str = Field(..., description='Человекочитаемое название источника.')
    audience: Audience = Field(..., description='Целевая аудитория чанка.')
    topic: str = Field(..., description='Тема чанка.')
    category: Category = Field(
        ..., description='Категория источника (Этап 5.1 плана) — нужна потребителю, чтобы '
        'выстроить финальный ответ в порядке "база → судебная практика → иные акты → комментарий".'
    )
    score: float = Field(..., description='Итоговый score после RRF fusion.')


class QueryVariant(BaseModel):
    """Один смысловой подвопрос исходного запроса и его юридические переформулировки.

    Для простого (не составного) запроса — единственный элемент
    `QueryExpansionResult.variants`, `sub_question` равен исходному
    тексту запроса. Для составного запроса — один элемент на каждый
    независимый подвопрос, полученный декомпозицией.
    """

    sub_question: str = Field(..., min_length=1, description='Подвопрос (или исходный запрос целиком).')
    rephrasings: list[str] = Field(
        default_factory=list, description='Юридические переформулировки этого подвопроса (раздел 8 плана).'
    )

    @field_validator('rephrasings', mode='before')
    @classmethod
    def cap_rephrasings(cls, value: object) -> object:
        if isinstance(value, list):
            valid = [item for item in value if isinstance(item, str) and item.strip()]
            return valid[:MAX_REPHRASINGS_PER_SUB_QUESTION]
        return value


class QueryExpansionResult(BaseModel):
    """Structured output LLM-расширения запроса перед поиском (раздел 8 плана).

    Решает две задачи одним вызовом: декомпозицию составного запроса на
    независимые подвопросы (`variants`, не более `MAX_SUB_QUESTIONS`) и
    переформулировку каждого подвопроса ближе к терминологии трудового
    права (`rephrasings` внутри каждого `QueryVariant`).
    """

    variants: list[QueryVariant] = Field(..., min_length=1, description='Подвопросы исходного запроса.')

    @field_validator('variants', mode='before')
    @classmethod
    def cap_variants(cls, value: object) -> object:
        if isinstance(value, list):
            return value[:MAX_SUB_QUESTIONS]
        return value


class RerankResult(BaseModel):
    """Structured output LLM-reranker'а (Этап 6).

    Модель получает кандидатов под номерами (не chunk_id — длинный UUID
    в выводе LLM рискует быть переврана на один символ и сломать маппинг
    обратно на чанк), возвращает номера в порядке релевантности.
    """

    ranked_indices: list[int] = Field(
        ..., min_length=1, description='Номера кандидатов (как в промпте) в порядке убывания релевантности.'
    )


class SearchRequest(BaseModel):
    """Тело запроса `POST /search` — контракт с MCP Tools Server (раздел 5 плана)."""

    model_config = ConfigDict(
        json_schema_extra={'example': {'query': 'Какая квота на трудоустройство инвалидов?', 'audience': 'employer', 'top_k': 5}}
    )

    query: str = Field(..., min_length=1, description='Текст поискового запроса.', examples=['Какая квота на трудоустройство инвалидов?'])
    audience: Audience | None = Field(None, description='Фильтр по целевой аудитории.')
    topic: str | None = Field(None, description='Фильтр по теме.')
    category: Category | None = Field(None, description='Фильтр по категории источника.')
    top_k: int = Field(5, ge=1, le=20, description='Сколько чанков вернуть после переранжирования.')


class SearchResponse(BaseModel):
    """Тело ответа `POST /search`."""

    chunks: list[SearchResultChunk] = Field(..., description='Найденные чанки, отсортированные по релевантности.')


class IngestRequest(BaseModel):
    """Тело запроса `POST /ingest` — запуск ingestion-пайплайна для одного документа."""

    document_id: str = Field(..., min_length=1, description='Идентификатор документа.', examples=['fz-181-art21'])
    category: Category = Field(..., description='Категория источника (раздел 3 плана).')
    raw_text: str = Field(
        ..., min_length=1, max_length=MAX_RAW_TEXT_LENGTH,
        description='Исходный текст документа (PDF/MD/TXT уже декодированы в строку).',
    )
    source_title: str = Field(..., description='Человекочитаемое название источника.')
    audience: Audience = Field(..., description='Целевая аудитория.')
    topic: str = Field(..., description='Тема документа.')
    version: str = Field(..., description='Дата редакции документа.')
    effective_date: date = Field(..., description='Дата вступления редакции в силу.')


class IngestResponse(BaseModel):
    """Тело ответа `POST /ingest`."""

    document_id: str = Field(..., description='Идентификатор документа.')
    version: str = Field(..., description='Версия, под которой документ проиндексирован.')
    chunks_count: int = Field(..., description='Количество созданных чанков.')
    replaced_versions: list[str] = Field(
        default_factory=list, description='Версии документа, удалённые после успешного upsert новой (раздел 3 плана).'
    )
    not_removed_versions: list[str] = Field(
        default_factory=list,
        description=(
            'Старые версии, удаление которых не удалось (ING-3) — новая версия уже '
            'успешно проиндексирована, эти версии нужно почистить вручную через админку.'
        ),
    )


class DocumentDeletedResponse(BaseModel):
    """Тело ответа `DELETE /document/{id}`."""

    document_id: str = Field(..., description='Идентификатор удалённого документа.')
