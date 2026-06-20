class SearchLogRepositoryError(Exception):
    """Ошибка записи журнала поискового запроса в БД.

    Никогда не должна ронять сам поисковый запрос (FASTAPI_PATTERNS.md,
    раздел 9 — деградация при частичном отказе) — перехватывается и
    логируется в `SearchService`, ответ клиенту отдаётся независимо от
    успеха записи лога. Поэтому у неё намеренно нет status_code/detail.
    """

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'Ошибка записи журнала поискового запроса. Подробности: {self.error_details}'
