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
