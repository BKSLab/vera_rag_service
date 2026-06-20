from fastapi import status


class LlmClientRequestError(Exception):
    """Ошибка одной попытки запроса к LLM API (сеть, HTTP, таймаут).

    Перехватывается и retry'ится внутри LlmClient — никогда не пересекает
    границу клиента, поэтому у неё намеренно нет status_code/detail.
    """


class LlmClientContentError(Exception):
    """Ошибка контента одной попытки: пустой ответ, невалидный JSON,
    ответ не прошёл валидацию по Pydantic-схеме.
    """


class LlmApiRequestError(Exception):
    """Финальная ошибка: все попытки запроса к LLM исчерпаны.

    Единственное исключение модуля, которое пересекает границу клиента.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, error_details: str, request_url: str):
        self.error_details = error_details
        self.request_url = request_url
        super().__init__(self.error_details, self.request_url)

    def __str__(self) -> str:
        return (
            f'Ошибка запроса к LLM API. URL: {self.request_url}. '
            f'Подробности: {self.error_details}'
        )

    @property
    def detail(self) -> str:
        return f'Ошибка при запросе к LLM API. Подробности: {self.error_details}'
