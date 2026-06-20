import asyncio
import json
import random

import httpx

from app.core.config_logger import logger
from app.exceptions.embedding import (
    EmbeddingApiRequestError,
    EmbeddingClientContentError,
    EmbeddingClientRequestError,
)


class EmbeddingClient:
    """Клиент для Yandex Cloud Foundation Models — Text Embedding API.

    Используется и в ingestion (doc-модель, офлайн, не в hot path), и в
    search (query-модель, hot path — раздел 4 «Зависимости и риски» плана:
    сетевая зависимость влияет на SLA "первый токен ≤5 сек"). Поэтому
    таймаут и retry умеренные, без щедрых значений LlmClient.
    """

    DEFAULT_TIMEOUT_SECONDS: int = 30
    DEFAULT_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: float = 0.5
    DEFAULT_MAX_RETRY_DELAY: float = 10.0
    JITTER_RATIO: float = 0.1

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        url: str,
        headers: dict,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
        delay: float = DEFAULT_RETRY_DELAY,
        max_delay: float = DEFAULT_MAX_RETRY_DELAY,
    ):
        """Инициализирует клиент.

        Args:
            httpx_client: Готовый httpx.AsyncClient, жизненным циклом управляет
                вызывающий код через DI.
            url: URL эндпоинта Text Embedding API.
            headers: Заголовки запроса (Authorization, Content-Type и т.п.).
            timeout: Таймаут одного HTTP-запроса в секундах.
            retries: Максимальное количество попыток на один вызов.
            delay: Базовая задержка перед повтором (секунды).
            max_delay: Верхняя граница задержки между повторами.
        """
        self.httpx_client = httpx_client
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.max_delay = max_delay

    def _get_backoff_delay(self, attempt: int) -> float:
        base_delay = min(self.max_delay, self.delay * (2 ** (attempt - 1)))
        jitter = base_delay * self.JITTER_RATIO * random.random()
        return base_delay + jitter

    async def _send_request(self, model_uri: str, text: str) -> dict:
        """Отправляет один запрос к Embedding API и возвращает сырой ответ.

        Raises:
            EmbeddingClientRequestError: При HTTP-ошибке, таймауте или сетевой ошибке.
        """
        payload = {'modelUri': model_uri, 'text': text}
        data_json = json.dumps(payload, ensure_ascii=False)
        try:
            response = await self.httpx_client.post(
                url=self.url,
                headers=self.headers,
                content=data_json,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            logger.error(
                '🌐 HTTP %s от Embedding API: %s', error.response.status_code, error.response.text
            )
            raise EmbeddingClientRequestError(f'HTTP {error.response.status_code}') from error
        except httpx.TimeoutException as error:
            logger.error('⏱️ Таймаут при запросе к Embedding API (%ss): %s', self.timeout, error)
            raise EmbeddingClientRequestError('Таймаут запроса к Embedding API') from error
        except httpx.RequestError as error:
            logger.error(
                '🌐 Сетевая ошибка при запросе к Embedding API: %s: %s', type(error).__name__, error
            )
            raise EmbeddingClientRequestError(f'Сетевая ошибка: {type(error).__name__}') from error

    @staticmethod
    def _extract_embedding(response: dict) -> list[float]:
        """Извлекает вектор эмбеддинга из ответа.

        Raises:
            EmbeddingClientContentError: Если поле embedding отсутствует или пустое.
        """
        embedding = response.get('embedding')
        if not embedding:
            raise EmbeddingClientContentError('Embedding API вернул пустой вектор')
        return embedding

    async def get_embedding(self, text: str, model_uri: str) -> list[float]:
        """Получает вектор эмбеддинга текста с экспоненциальными повторами при ошибках.

        Args:
            text: Текст для эмбеддинга.
            model_uri: URI модели (doc-модель при индексации, query-модель при поиске).

        Returns:
            Вектор эмбеддинга.

        Raises:
            EmbeddingApiRequestError: Если все попытки запроса исчерпаны.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = await self._send_request(model_uri, text)
                embedding = self._extract_embedding(response)
                if attempt > 1:
                    logger.info('✅ Эмбеддинг получен с %s-й попытки', attempt)
                return embedding
            except EmbeddingClientContentError as error:
                last_error = error
                logger.warning(
                    '📭 Некорректный контент от Embedding API (попытка %d/%d): %s',
                    attempt, self.retries, error,
                )
            except EmbeddingClientRequestError as error:
                last_error = error
                logger.warning(
                    '⚠️ Ошибка запроса к Embedding API (попытка %d/%d): %s',
                    attempt, self.retries, error,
                )

            if attempt < self.retries:
                delay = self._get_backoff_delay(attempt)
                await asyncio.sleep(delay)

        logger.error(
            '❌ Не удалось получить эмбеддинг после %d попыток. Последняя ошибка: %s',
            self.retries, last_error,
        )
        raise EmbeddingApiRequestError(error_details=str(last_error), request_url=self.url)
