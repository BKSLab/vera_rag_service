import asyncio
import functools
import json
import random
import re
from collections.abc import Callable
from pprint import pformat
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config_logger import logger
from app.exceptions.llm import (
    LlmApiRequestError,
    LlmClientContentError,
    LlmClientRequestError,
)

PydanticModel = TypeVar('PydanticModel', bound=BaseModel)

# YandexGPT периодически добавляет markdown-эмфазис (_..._, **...**) внутрь
# JSON-ответа несмотря на явный запрет в промпте — портит и значения, и сами
# имена полей ("_hypothetical_questions" вместо "hypothetical_questions").
# Убираем подчёркивания/звёздочки, прилегающие непосредственно к кавычкам
# JSON-строки, перед валидацией по схеме — это артефакт форматирования
# модели, а не часть содержания.
_JSON_MARKDOWN_EMPHASIS_PATTERN = re.compile(r'(?<=")[_*]+|[_*]+(?=")')

# Модель почти всегда оборачивает JSON-ответ в markdown code fence (```json
# ... ``` или просто ``` ... ```) несмотря на просьбу не делать этого —
# снимаем обёртку перед парсингом, а не полагаемся на соблюдение промпта.
_MARKDOWN_CODE_FENCE_PATTERN = re.compile(r'^```[a-zA-Z]*\n?|```\s*$')


def _strip_json_markdown_emphasis(content: str) -> str:
    return _JSON_MARKDOWN_EMPHASIS_PATTERN.sub('', content)


def _strip_markdown_code_fence(content: str) -> str:
    return _MARKDOWN_CODE_FENCE_PATTERN.sub('', content.strip()).strip()


class LlmClient:
    """Клиент для LLM API, совместимых с контрактом OpenAI Chat Completions.

    Используется для офлайн-обогащения чанков на этапе ingestion (Этап 3
    плана) — не в hot path поиска, поэтому таймауты и количество retry
    выставлены щедрыми по умолчанию. Провайдер-специфичные расширения
    payload передаются через extra_payload в конструкторе.
    """

    DEFAULT_TIMEOUT_SECONDS: int = 90
    DEFAULT_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: float = 1.0
    DEFAULT_MAX_RETRY_DELAY: float = 30.0
    JITTER_RATIO: float = 0.1

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        model: str,
        url: str,
        headers: dict,
        temperature: float = 0.3,
        stream: bool = False,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
        delay: float = DEFAULT_RETRY_DELAY,
        max_delay: float = DEFAULT_MAX_RETRY_DELAY,
        extra_payload: dict | None = None,
    ):
        """Инициализирует клиент.

        Args:
            httpx_client: Готовый httpx.AsyncClient, жизненным циклом управляет
                вызывающий код через DI.
            model: Имя/URI модели по умолчанию для запросов.
            url: URL эндпоинта Chat Completions провайдера.
            headers: Заголовки запроса (Authorization, Content-Type и т.п.).
            temperature: Температура генерации по умолчанию.
            stream: Признак потокового ответа.
            timeout: Таймаут одного HTTP-запроса в секундах.
            retries: Максимальное количество попыток на один вызов.
            delay: Базовая задержка перед повтором (секунды).
            max_delay: Верхняя граница задержки между повторами.
            extra_payload: Провайдер-специфичные поля, добавляемые в payload
                каждого запроса. None — без расширений.
        """
        self.httpx_client = httpx_client
        self.model = model
        self.url = url
        self.headers = headers
        self.temperature = temperature
        self.stream = stream
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.max_delay = max_delay
        self.extra_payload = extra_payload or {}

    def _get_backoff_delay(self, attempt: int) -> float:
        """Вычисляет задержку по экспоненциальному алгоритму с джиттером."""
        base_delay = min(self.max_delay, self.delay * (2 ** (attempt - 1)))
        jitter = base_delay * self.JITTER_RATIO * random.random()
        return base_delay + jitter

    async def _send_request_to_llm(self, payload: dict) -> dict:
        """Отправляет один запрос к LLM и возвращает сырой ответ.

        Raises:
            LlmClientRequestError: При HTTP-ошибке, таймауте или сетевой ошибке.
        """
        data_json = json.dumps(payload, ensure_ascii=False)
        try:
            logger.info('📤 Отправка запроса к LLM, модель: %s', payload.get('model'))
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
                '🌐 HTTP %s от LLM: %s', error.response.status_code, error.response.text
            )
            raise LlmClientRequestError(f'HTTP {error.response.status_code}') from error
        except httpx.TimeoutException as error:
            logger.error('⏱️ Таймаут при запросе к LLM (%ss): %s', self.timeout, error)
            raise LlmClientRequestError('Таймаут запроса к LLM') from error
        except httpx.RequestError as error:
            logger.error(
                '🌐 Сетевая ошибка при запросе к LLM: %s: %s', type(error).__name__, error
            )
            raise LlmClientRequestError(f'Сетевая ошибка: {type(error).__name__}') from error

    def _extract_content(self, response: dict) -> str:
        """Извлекает и валидирует текстовый контент из ответа LLM.

        Raises:
            LlmClientContentError: Если структура ответа не соответствует
                ожидаемому контракту, или контент пустой.
        """
        try:
            content = response.get('choices')[0].get('message').get('content')
        except (KeyError, IndexError, TypeError) as error:
            logger.debug('Полный ответ LLM с невалидной структурой: %s', pformat(response))
            raise LlmClientContentError(
                f'Невалидная структура ответа LLM: {type(error).__name__}'
            ) from error

        if not content or not content.strip():
            raise LlmClientContentError('LLM вернул пустой ответ')

        return content

    def _extract_validated(self, response: dict, schema: type[PydanticModel]) -> PydanticModel:
        """Извлекает контент из ответа LLM и валидирует его по Pydantic-схеме.

        Raises:
            LlmClientContentError: Если контент не прошёл валидацию схемы.
        """
        content = _strip_json_markdown_emphasis(
            _strip_markdown_code_fence(self._extract_content(response))
        )
        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            logger.warning(
                '📋 Ответ LLM не прошёл валидацию схемы %s: %s', schema.__name__, error
            )
            raise LlmClientContentError(
                f'Ответ не соответствует схеме {schema.__name__}'
            ) from error

    async def _fetch_with_retries(
        self, payload: dict, extractor: Callable[[dict], Any]
    ) -> Any:
        """Выполняет запрос к LLM с экспоненциальными повторами при любых ошибках.

        Raises:
            LlmApiRequestError: Если все попытки исчерпаны без успеха.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = await self._send_request_to_llm(payload)
                content = extractor(response)
                if attempt > 1:
                    logger.info('✅ Ответ от LLM получен с %s-й попытки', attempt)
                return content
            except LlmClientContentError as error:
                last_error = error
                logger.warning(
                    '📭 Некорректный контент от LLM (попытка %d/%d): %s',
                    attempt, self.retries, error,
                )
            except LlmClientRequestError as error:
                last_error = error
                logger.warning(
                    '⚠️ Ошибка запроса к LLM (попытка %d/%d): %s',
                    attempt, self.retries, error,
                )

            if attempt < self.retries:
                delay = self._get_backoff_delay(attempt)
                logger.info(
                    '🔄 Повтор через %.1fс (следующая попытка: %d/%d)',
                    delay, attempt + 1, self.retries,
                )
                await asyncio.sleep(delay)

        logger.error(
            '❌ Не удалось получить ответ от LLM после %d попыток. Последняя ошибка: %s',
            self.retries, last_error,
        )
        raise LlmApiRequestError(error_details=str(last_error), request_url=self.url)

    async def get_llm_response(
        self,
        content: str,
        prompt: str,
        model: str | None = None,
        schema: type[PydanticModel] | None = None,
        max_completion_tokens: int = 6000,
    ) -> str | PydanticModel:
        """Получает ответ от LLM.

        Args:
            content: Пользовательское сообщение (роль "user").
            prompt: Системный промпт (роль "system").
            model: Модель для этого конкретного вызова, если отличается от
                модели по умолчанию, заданной в конструкторе.
            schema: Pydantic-схема для structured output. Если передана,
                клиент валидирует ответ по схеме (ожидая JSON, запрошенный
                через промпт). `response_format: json_object` намеренно не
                используется — у YandexGPT (`yandexgpt/rc`) принудительный
                JSON-режим ломает генерацию списков строк (наблюдались
                зацикленные числовые значения и потеря второго поля
                схемы), тогда как обычный chat-режим с JSON-инструкцией
                в промпте даёт стабильный валидный JSON, обёрнутый в
                markdown code fence — fence снимается в `_extract_validated`.
            max_completion_tokens: Лимит токенов ответа.

        Returns:
            Текст ответа (если schema не передана) или валидированный
            экземпляр Pydantic-модели (если передана).

        Raises:
            LlmApiRequestError: Если все попытки запроса исчерпаны.
        """
        payload = {
            'model': model or self.model,
            'messages': [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': content},
            ],
            'temperature': self.temperature,
            'max_completion_tokens': max_completion_tokens,
            'stream': self.stream,
        }
        if self.extra_payload:
            payload.update(self.extra_payload)

        extractor = (
            functools.partial(self._extract_validated, schema=schema)
            if schema
            else self._extract_content
        )
        return await self._fetch_with_retries(payload, extractor=extractor)
