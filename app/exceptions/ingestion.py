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
