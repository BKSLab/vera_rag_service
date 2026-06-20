from fastapi import status


class EmbeddingClientRequestError(Exception):
    """Ошибка одной попытки запроса к Embedding API (сеть, HTTP, таймаут).

    Перехватывается и retry'ится внутри EmbeddingClient — никогда не
    пересекает границу клиента, поэтому у неё намеренно нет status_code/detail.
    """


class EmbeddingClientContentError(Exception):
    """Ошибка контента одной попытки: ответ без поля embedding или с пустым вектором."""


class EmbeddingApiRequestError(Exception):
    """Финальная ошибка: все попытки запроса к Embedding API исчерпаны.

    Единственное исключение модуля, которое пересекает границу клиента.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, error_details: str, request_url: str):
        self.error_details = error_details
        self.request_url = request_url
        super().__init__(self.error_details, self.request_url)

    def __str__(self) -> str:
        return (
            f'Ошибка запроса к Embedding API. URL: {self.request_url}. '
            f'Подробности: {self.error_details}'
        )

    @property
    def detail(self) -> str:
        return f'Ошибка при запросе к Embedding API. Подробности: {self.error_details}'
