from fastapi import status


class DatabaseUnavailableError(Exception):
    """Подключение к Postgres недоступно."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = 'База данных недоступна.'

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'Ошибка подключения к БД. Подробности: {self.error_details}'


class QdrantUnavailableError(Exception):
    """Подключение к Qdrant недоступно."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = 'Векторное хранилище Qdrant недоступно.'

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'Ошибка подключения к Qdrant. Подробности: {self.error_details}'
