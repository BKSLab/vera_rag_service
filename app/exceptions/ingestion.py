from collections.abc import Iterable


class IngestionIntegrityError(Exception):
    """Подготовленные чанки и фактические точки Qdrant не совпали."""

    status_code = 500

    def __init__(
        self,
        document_id: str,
        version: str,
        scope: str,
        expected_count: int,
        actual_count: int,
        *,
        duplicate_chunk_ids: Iterable[str] = (),
        missing_chunk_ids: Iterable[str] = (),
        unexpected_chunk_ids: Iterable[str] = (),
    ):
        self.document_id = document_id
        self.version = version
        self.scope = scope
        self.expected_count = expected_count
        self.actual_count = actual_count
        self.duplicate_chunk_ids = sorted(set(duplicate_chunk_ids))
        self.missing_chunk_ids = sorted(set(missing_chunk_ids))
        self.unexpected_chunk_ids = sorted(set(unexpected_chunk_ids))
        super().__init__(self.__str__())

    def __str__(self) -> str:
        details = [
            f'expected_count={self.expected_count}',
            f'actual_count={self.actual_count}',
        ]
        if self.duplicate_chunk_ids:
            details.append(f'duplicate_chunk_ids={self.duplicate_chunk_ids!r}')
        if self.missing_chunk_ids:
            details.append(f'missing_chunk_ids={self.missing_chunk_ids!r}')
        if self.unexpected_chunk_ids:
            details.append(f'unexpected_chunk_ids={self.unexpected_chunk_ids!r}')
        return (
            f'Нарушен инвариант целостности ingestion для документа {self.document_id!r} '
            f'(version={self.version!r}, scope={self.scope!r}): ' + '; '.join(details)
        )

    @property
    def detail(self) -> str:
        return str(self)


class RawTextTooLargeError(Exception):
    """`raw_text` превышает допустимый размер (API-3).

    `IngestRequest.raw_text` уже ограничен `max_length` на уровне Pydantic
    (`app/models/schemas.py::MAX_RAW_TEXT_LENGTH`) — этот же лимит
    проверяется здесь, в `IngestionService`, потому что админка
    (`DocumentUploadView`) вызывает `ingest_document` напрямую, минуя
    `IngestRequest`/Pydantic-валидацию (читает файл и сама строит `raw_text`).
    """

    def __init__(self, document_id: str, length: int, max_length: int):
        self.document_id = document_id
        self.length = length
        self.max_length = max_length
        super().__init__(self.__str__())

    def __str__(self) -> str:
        return (
            f'Документ {self.document_id}: текст {self.length} символов — '
            f'превышен лимит {self.max_length}.'
        )


class TopicsNotAllowedForCategoryError(Exception):
    """Темы заданы для категории, где они не осмысленны (API-3, обсуждение
    с пользователем 2026-07-08).

    `labor_code`/`federal_law` — широкие кодексы/законы, регулирующие
    десятки разных тем одновременно: свести это к одной-двум темам на
    документ означало бы соврать или обесценить фильтр. Темы допустимы
    только для узких по предмету категорий (см. `TOPICS_ALLOWED_CATEGORIES`,
    `app/models/schemas.py`).
    """

    def __init__(self, document_id: str, category: str, topics: list[str]):
        self.document_id = document_id
        self.category = category
        self.topics = topics
        super().__init__(self.__str__())

    def __str__(self) -> str:
        return (
            f'Документ {self.document_id} (category={self.category!r}) не может иметь темы '
            f'{self.topics!r} — темы допустимы только для other_npa/case_law/authorial.'
        )


class TooManyChunksError(Exception):
    """Документ дал больше чанков, чем разумный верхний предел одного документа (API-3).

    Без этого предела один запрос мог бы запустить неограниченное число
    платных вызовов LLM-обогащения и эмбеддинга — явный отказ лучше тихой
    деградации (FASTAPI_PATTERNS.md, раздел 9).
    """

    def __init__(self, document_id: str, chunks_count: int, max_chunks: int):
        self.document_id = document_id
        self.chunks_count = chunks_count
        self.max_chunks = max_chunks
        super().__init__(self.__str__())

    def __str__(self) -> str:
        return (
            f'Документ {self.document_id} дал {self.chunks_count} чанков — '
            f'превышен лимит {self.max_chunks} на один документ.'
        )
