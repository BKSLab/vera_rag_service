from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.metadata import Audience, Category
from app.models.requisites import (
    ActType,
    allowed_act_types,
    build_act_reference,
    build_document_id,
    default_authority,
    requisites_required,
    revision_applicable,
)

# Обязательный суффикс номера акта по категории — простая проверка, которая
# ловит перепутанную категорию на вводе: у кодексов и федеральных законов
# номер всегда оканчивается на «-ФЗ» или «-ФКЗ». У подзаконных актов номера
# слишком разнородны (845, 1234-р, 33н), поэтому там проверки нет.
REQUIRED_NUMBER_SUFFIX_BY_CATEGORY: dict[str, str] = {
    'labor_code': 'ФЗ',
    'federal_law': 'ФЗ',
}

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


def _canonical_iso_date(value: str) -> str:
    return date.fromisoformat(value).isoformat()


class Chunk(BaseModel):
    """Чанк — единица индексации, результат Этапа 2 (иерархический чанкинг).

    Метаданные секции (`section_number`, `section_title`), из которой получен
    чанк, переносятся как контекст шире самого чанка — нужны на этапе
    генерации ответа LLM (см. RAG_SERVICE_PLAN.md, Этап 2).
    """

    chunk_id: str = Field(..., description='Уникальный идентификатор чанка (uuid).')
    chunk_index: int = Field(..., description='Сквозной порядковый номер чанка в пределах документа.')
    chunk_number_in_section: int = Field(
        ..., description='Локальный порядковый номер чанка внутри секции (Этап 13 плана).',
    )
    document_id: str = Field(
        ..., description='Идентификатор документа-источника.', examples=['fz-181-1995'],
    )
    parent_id: str = Field(
        ..., description='Единица обновления/удаления статьи: f"{document_id}:{section_number}" или document_id (Этап 13).',
    )
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

    document_id: str = Field(
        ..., description='Идентификатор документа-источника.', examples=['fz-181-1995'],
    )
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

    source_title: str = Field(
        ...,
        description=(
            'Полное официальное наименование документа: вид акта, дата принятия, номер и название. '
            'Возвращается потребителю как ссылка на источник, поэтому должно точно соответствовать документу.'
        ),
        examples=['Федеральный закон от 24.11.1995 № 181-ФЗ "О социальной защите инвалидов в Российской Федерации"'],
    )
    audience: Audience = Field(..., description='Целевая аудитория.')
    topics: list[str] = Field(
        default_factory=list,
        description='Темы документа (раздел 3 плана) — допустимы только для other_npa/case_law/authorial, '
        'для labor_code/federal_law список должен быть пустым.',
    )
    version: str = Field(..., description='Ключ ревизии документа в хранилище (ISO-дата содержимого).')
    amending_act: str | None = Field(
        None,
        description='Акт, которым внесены изменения в эту статью; задаётся только при гранулярном обновлении.',
        examples=['Федеральный закон от 26.07.2026 № 246-ФЗ'],
    )
    effective_date: date = Field(
        ...,
        description=(
            'Начало интервала действия этой редакции чанка; в паре с `effective_until` '
            'обслуживает запросы «какой текст действовал на дату X».'
        ),
    )

    @field_validator('version')
    @classmethod
    def canonicalize_version(cls, value: str) -> str:
        return _canonical_iso_date(value)


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
    source_title: str = Field(
        ...,
        description=(
            'Полное официальное наименование документа: вид акта, дата принятия, номер и название. '
            'Возвращается потребителю как ссылка на источник, поэтому должно точно соответствовать документу.'
        ),
        examples=['Федеральный закон от 24.11.1995 № 181-ФЗ "О социальной защите инвалидов в Российской Федерации"'],
    )
    audience: Audience = Field(..., description='Целевая аудитория чанка.')
    topics: list[str] = Field(default_factory=list, description='Темы чанка (раздел 3 плана) — пусто для labor_code/federal_law.')
    category: Category = Field(
        ..., description='Категория источника (Этап 5.1 плана) — нужна потребителю, чтобы '
        'выстроить финальный ответ в порядке "база → судебная практика → иные акты → комментарий".'
    )
    section_number: str | None = Field(None, description='Номер статьи/пункта из структуры документа (например, "128").')
    section_title: str | None = Field(None, description='Заголовок статьи/пункта (например, "Статья 128. Отпуска без сохранения заработной платы").')
    revision_date: date = Field(
        ...,
        description=(
            'Редакция, из которой взят этот текст. Норма без указания редакции — утверждение '
            'без даты, поэтому потребитель обязан иметь возможность сослаться на неё: '
            '«статья 128 в редакции от 01.03.2027».'
        ),
        examples=['2027-03-01'],
    )
    amending_act: str | None = Field(
        None,
        description=(
            'Акт, которым изменена именно эта статья; `None` — статья не менялась с момента '
            'загрузки документа. Задаётся только при гранулярном обновлении редакции.'
        ),
        examples=['Федеральный закон от 26.07.2026 № 246-ФЗ'],
    )
    score: float = Field(..., description='Итоговый score после RRF fusion.')
    rerank_rank: int = Field(..., ge=1, description='Позиция чанка после LLM-reranker, начиная с 1.')


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
        ..., description='Номера кандидатов (как в промпте) в порядке убывания релевантности. '
        'Пустой список означает "ни один кандидат не релевантен" (Этап 6.1 плана).'
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


class ActRequisitesMixin(BaseModel):
    """Реквизиты правового акта — общая часть карточки документа.

    Наименование хранится разобранным, а не одной строкой: собрать строку из
    реквизитов можно всегда, разобрать обратно — уже нет. `document_id` и
    `source_title` не вводятся, а выводятся отсюда, поэтому один и тот же акт
    нельзя завести под двумя разными наименованиями.
    """

    category: Category = Field(..., description='Категория источника (раздел 3 плана).')
    act_type: ActType | None = Field(
        None,
        description='Вид акта. Допустимые значения зависят от категории; для авторских материалов не задаётся.',
        examples=['Федеральный закон'],
    )
    act_number: str | None = Field(
        None, max_length=100,
        description='Номер акта. Обязателен для нормативных актов и судебной практики.',
        examples=['181-ФЗ'],
    )
    act_date: date = Field(
        ...,
        description='Дата акта: дата подписания для правовых актов, дата публикации для авторских материалов.',
        examples=['1995-11-24'],
    )
    act_title: str = Field(
        ..., min_length=1,
        description='Наименование акта без реквизитов.',
        examples=['О социальной защите инвалидов в Российской Федерации'],
    )
    act_authority: str | None = Field(
        None,
        description='Принявший (подписавший) орган. Подставляется по виду акта, если не задан.',
        examples=['Президент Российской Федерации'],
    )
    revision_date: date | None = Field(
        None,
        description=(
            'Дата действующей редакции акта — «редакция от». Обязательна для нормативных актов; '
            'к судебной практике и авторским материалам не применяется.'
        ),
        examples=['2026-05-25'],
    )
    document_id: str | None = Field(
        None, max_length=255,
        description='Идентификатор документа. Выводится из реквизитов; задавать вручную нужно только там, где номера акта нет.',
        examples=['fz-181-1995'],
    )
    source_title: str = Field(
        ...,
        min_length=1,
        description=(
            'Отображаемое название документа целиком — то, что видит пользователь как ссылку '
            'на источник. Задаётся оператором, а не собирается из реквизитов: принятое '
            'написание из полей не выводится (кавычки-ёлочки, дата словами, «N» вместо «№», '
            'наименование кодекса без реквизитов), а эта строка уходит в промпт обогащения, '
            "а значит и в эмбеддинги, в заголовок кандидата reranker'а и в абзац «Основание» "
            'ответа. Её формулировка — часть настроенного бенчмарками качества выдачи, и '
            'менять её из-за смены способа хранения реквизитов нельзя.'
        ),
        examples=['Федеральный закон «О социальной защите инвалидов в Российской Федерации» от 24 ноября 1995 N 181-ФЗ'],
    )

    @model_validator(mode='after')
    def validate_and_complete_requisites(self) -> 'ActRequisitesMixin':
        """Проверяет реквизиты по правилам категории и достраивает выводимые поля."""

        self._validate_act_type()
        self._validate_act_number()
        self._validate_revision_date()
        if self.act_authority is None and self.act_type is not None:
            self.act_authority = default_authority(self.act_type)
        if self.document_id is None:
            self.document_id = self._build_document_id()
        return self

    @property
    def content_date(self) -> date:
        """Дата, которой датируется содержимое документа.

        Для нормативных актов это дата действующей редакции, для остальных —
        дата самого документа: редакций у них не бывает.
        """

        return self.revision_date or self.act_date

    @property
    def version(self) -> str:
        """Ключ ревизии документа в хранилище — выводится из даты содержимого."""

        return self.content_date.isoformat()

    def _validate_act_type(self) -> None:
        allowed = allowed_act_types(self.category)
        if not requisites_required(self.category):
            return
        if self.act_type is None:
            raise ValueError(f'Для категории {self.category!r} обязателен вид акта.')
        if self.act_type not in allowed:
            raise ValueError(
                f'Вид акта {self.act_type!r} недопустим для категории {self.category!r}. '
                f'Допустимые: {list(allowed)}.'
            )

    def _validate_act_number(self) -> None:
        if not requisites_required(self.category):
            return
        if not self.act_number:
            raise ValueError(f'Для категории {self.category!r} обязателен номер акта.')
        expected_suffix = REQUIRED_NUMBER_SUFFIX_BY_CATEGORY.get(self.category)
        if expected_suffix and not self.act_number.upper().endswith(expected_suffix):
            raise ValueError(
                f'Номер акта {self.act_number!r} не оканчивается на {expected_suffix!r}: '
                f'это не документ категории {self.category!r}.'
            )

    def _validate_revision_date(self) -> None:
        if revision_applicable(self.category):
            if self.revision_date is None:
                raise ValueError(
                    f'Для категории {self.category!r} обязательна дата действующей редакции.'
                )
            if self.revision_date < self.act_date:
                raise ValueError('Дата редакции не может быть раньше даты самого акта.')
        elif self.revision_date is not None:
            raise ValueError(
                f'К категории {self.category!r} дата редакции не применяется: '
                'такие документы не изменяются, а заменяются новыми.'
            )

    def _build_document_id(self) -> str:
        if not self.act_number:
            raise ValueError(
                'Идентификатор документа не выводится без номера акта — задайте document_id явно.'
            )
        return build_document_id(
            category=self.category,
            act_number=self.act_number,
            act_date=self.act_date,
        )


class IngestRequest(ActRequisitesMixin):
    """Тело запроса `POST /ingest` — запуск ingestion-пайплайна для одного документа."""

    raw_text: str = Field(
        ..., min_length=1, max_length=MAX_RAW_TEXT_LENGTH,
        description='Исходный текст документа (PDF/MD/TXT уже декодированы в строку).',
    )
    audience: Audience = Field(..., description='Целевая аудитория.')
    topics: list[str] = Field(
        default_factory=list,
        description='Темы документа (раздел 3 плана) — допустимы только для other_npa/case_law/authorial.',
    )


class IngestionReceipt(BaseModel):
    """Подтверждение записи и проверки коллекции, общее для двух путей загрузки."""

    operation_id: str | None = None
    status: str = 'succeeded'
    warnings: list[str] = Field(default_factory=list)
    collection_name: str | None = None
    input_sha256: str | None = None
    integrity_verified: bool = False


class IngestResponse(IngestionReceipt):
    """Тело ответа `POST /ingest`."""

    document_id: str = Field(..., description='Идентификатор документа.', examples=['fz-181-1995'])
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

    document_id: str = Field(
        ..., description='Идентификатор удалённого документа.', examples=['fz-181-1995'],
    )


# Категории, поддерживающие гранулярное обновление одной статьи/пункта
# (Этап 13 плана). case_law и authorial обновляются только целым документом:
# у них нет устойчивой нумерации статей, за которую можно зацепиться.
# other_npa добавлена вместе с мониторингом постановлений Правительства РФ
# в Legal Sync Service — они меняются постатейно ровно так же, как законы.
SECTION_UPDATE_ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {'labor_code', 'federal_law', 'other_npa'}
)

# Категории, для которых осмысленны темы (раздел 3 плана, обсуждение с
# пользователем 2026-07-08) — узкие по предмету источники, документ обычно
# посвящён одному-двум конкретным вопросам. labor_code/federal_law — широкие
# кодексы/законы, регулирующие десятки разных тем одновременно: свести это
# к одной-двум темам на документ означало бы соврать или обесценить фильтр.
TOPICS_ALLOWED_CATEGORIES: frozenset[str] = frozenset({'other_npa', 'case_law', 'authorial'})


class SectionUpdateRequest(BaseModel):
    """Тело запроса `PUT /document/{id}/sections/{section_number}`.

    Принимает готовый (консолидированный) текст одной статьи/пункта — не
    текст закона-поправки с описанием дельты. Старые чанки этой секции
    не удаляются физически, а помечаются `is_actual=False`/`effective_until`
    (Этап 13 плана, историческое хранение редакций).
    """

    category: Category = Field(
        ..., description=f'Категория источника. Допустимые значения для гранулярного обновления: {sorted(SECTION_UPDATE_ALLOWED_CATEGORIES)}.',
    )
    raw_text: str = Field(
        ..., min_length=1, max_length=MAX_RAW_TEXT_LENGTH,
        description='Готовый текст только этой статьи/пункта (не закон-поправка с описанием дельты).',
    )
    section_title: str = Field(..., min_length=1, description='Заголовок статьи/пункта (например, "Отпуска без сохранения заработной платы").')
    revision_date: date = Field(
        ...,
        description='Дата редакции, из которой взят текст этой статьи.',
        examples=['2027-03-01'],
    )
    source_title: str = Field(
        ...,
        description=(
            'Полное официальное наименование документа: вид акта, дата принятия, номер и название. '
            'Возвращается потребителю как ссылка на источник, поэтому должно точно соответствовать документу.'
        ),
        examples=['"Трудовой кодекс Российской Федерации" от 30.12.2001 № 197-ФЗ'],
    )
    audience: Audience = Field(..., description='Целевая аудитория.')
    topics: list[str] = Field(
        default_factory=list,
        description='Темы должны быть пустыми для категорий с гранулярным обновлением.',
    )
    amending_act_type: str | None = Field(
        None,
        description='Вид акта, которым внесены изменения в эту статью.',
        examples=['Федеральный закон'],
    )
    amending_act_number: str | None = Field(
        None, max_length=100,
        description='Номер акта, которым внесены изменения в эту статью.',
        examples=['246-ФЗ'],
    )
    amending_act_date: date | None = Field(
        None,
        description='Дата акта, которым внесены изменения в эту статью.',
        examples=['2026-07-26'],
    )

    @property
    def version(self) -> str:
        """Ключ ревизии статьи в хранилище — выводится из даты редакции."""

        return self.revision_date.isoformat()

    @property
    def amending_act_reference(self) -> str | None:
        """Ссылка на акт, которым внесены изменения именно в эту статью.

        Хранится на уровне статьи, а не документа: один закон меняет лишь
        часть статей кодекса, и приписывать его остальным — значит утверждать
        неправду о происхождении их текста.
        """

        if not (self.amending_act_type and self.amending_act_number and self.amending_act_date):
            return None
        return build_act_reference(
            act_type=self.amending_act_type,
            act_number=self.amending_act_number,
            act_date=self.amending_act_date,
        )


class SectionUpdateResponse(IngestionReceipt):
    """Тело ответа `PUT /document/{id}/sections/{section_number}`."""

    document_id: str = Field(..., description='Идентификатор документа.', examples=['fz-181-1995'])
    section_number: str = Field(..., description='Номер обновлённой статьи/пункта.')
    parent_id: str = Field(..., description='Идентификатор секции: f"{document_id}:{section_number}".')
    version: str = Field(..., description='Версия, под которой проиндексирована новая редакция.')
    chunks_count: int = Field(..., description='Количество чанков новой редакции.')
    superseded_chunks: int = Field(
        0, description='Количество чанков предыдущей редакции, помеченных неактуальными.',
    )
