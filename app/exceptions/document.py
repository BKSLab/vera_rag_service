class DocumentRepositoryError(Exception):
    """Ошибка записи в реестр документов (Этап 11.1 плана).

    Источник правды о содержимом БЗ — Qdrant, а не эта таблица (она только
    для отображения списка/истории версий в админке). Поэтому отказ записи
    сюда не должен ронять `IngestionService.ingest_document` — ingestion
    в Qdrant к этому моменту уже успешно завершён (FASTAPI_PATTERNS.md,
    раздел 9 — деградация при частичном отказе, по аналогии с `search_logs`).
    """

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'Ошибка записи в реестр документов. Подробности: {self.error_details}'
