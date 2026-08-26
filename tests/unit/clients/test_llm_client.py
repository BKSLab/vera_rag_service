import json

import httpx
import pytest
from pydantic import BaseModel

from app.clients.llm import LlmClient
from app.core.circuit_breaker import CircuitBreaker
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import RerankResult


class _EchoSchema(BaseModel):
    answer: str


def _make_client(handler, **overrides) -> LlmClient:
    httpx_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LlmClient(
        httpx_client=httpx_client,
        model='test-model',
        url='https://llm.example.com/v1/chat/completions',
        headers={'Authorization': 'Api-Key test'},
        retries=overrides.pop('retries', 3),
        delay=overrides.pop('delay', 0.001),
        max_delay=overrides.pop('max_delay', 0.001),
        **overrides,
    )


def _chat_completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={'choices': [{'message': {'content': content}}]},
    )


async def test_get_llm_response_returns_content_without_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('Привет')

    client = _make_client(handler)

    result = await client.get_llm_response(content='вопрос', prompt='system')

    assert result == 'Привет'


async def test_get_llm_response_omits_temperature_from_payload():
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return _chat_completion_response('Привет')

    client = _make_client(handler)

    await client.get_llm_response(content='вопрос', prompt='system')

    assert 'temperature' not in captured_payload


@pytest.mark.parametrize('temperature', [0.0, 0.3])
async def test_get_llm_response_includes_configured_temperature_in_payload(temperature):
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return _chat_completion_response('Привет')

    client = _make_client(handler, temperature=temperature)

    await client.get_llm_response(content='вопрос', prompt='system')

    assert captured_payload['temperature'] == temperature


async def test_get_llm_response_returns_validated_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response(json.dumps({'answer': 'да'}))

    client = _make_client(handler)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert isinstance(result, _EchoSchema)
    assert result.answer == 'да'


async def test_get_llm_response_retries_on_http_error_then_succeeds():
    attempts = {'count': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts['count'] += 1
        if attempts['count'] < 2:
            return httpx.Response(500, text='internal error')
        return _chat_completion_response('успех со второй попытки')

    client = _make_client(handler, retries=3)

    result = await client.get_llm_response(content='вопрос', prompt='system')

    assert result == 'успех со второй попытки'
    assert attempts['count'] == 2


async def test_get_llm_response_raises_after_exhausting_retries_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text='internal error')

    client = _make_client(handler, retries=2)

    with pytest.raises(LlmApiRequestError):
        await client.get_llm_response(content='вопрос', prompt='system')


async def test_get_llm_response_raises_after_exhausting_retries_on_invalid_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response(json.dumps({'wrong_field': 'значение'}))

    client = _make_client(handler, retries=2)

    with pytest.raises(LlmApiRequestError):
        await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)


async def test_get_llm_response_raises_on_empty_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('   ')

    client = _make_client(handler, retries=2)

    with pytest.raises(LlmApiRequestError):
        await client.get_llm_response(content='вопрос', prompt='system')


@pytest.mark.parametrize('fence_language', ['', 'json'])
async def test_get_llm_response_strips_markdown_code_fence_before_validation(
    fence_language,
):
    """Внешняя fence-оболочка structured output снимается для любого LLM."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response(
            f'```{fence_language}\n{{"answer": "да"}}\n```'
        )

    client = _make_client(handler)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert result.answer == 'да'


async def test_get_llm_response_accepts_gemini_fenced_rerank_result_by_default():
    """Регрессия N-01: Gemini через Polza оборачивает RerankResult в fence."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response(
            '```json\n{"ranked_indices": [11, 51]}\n```'
        )

    client = _make_client(handler)

    result = await client.get_llm_response(
        content='кандидаты', prompt='system', schema=RerankResult
    )

    assert result.ranked_indices == [11, 51]


async def test_get_llm_response_strips_markdown_emphasis_underscores_before_validation():
    """Регрессия: YandexGPT иногда добавляет markdown-эмфазис (_..._) вокруг
    значений строк прямо внутри JSON (см. RAG_SERVICE_PLAN.md, Этап 3)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('{"answer": "_да, можно_"}')

    client = _make_client(handler, strip_markdown_artifacts=True)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert result.answer == 'да, можно'


async def test_get_llm_response_does_not_strip_markdown_emphasis_by_default():
    """Без opt-in клиент не меняет легитимный эмфазис внутри JSON-строк."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('{"answer": "_да_"}')

    client = _make_client(handler)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert result.answer == '_да_'


async def test_get_llm_response_fails_fast_when_circuit_breaker_open():
    """LLM-2 — открытый breaker пропускает реальный HTTP-вызов целиком."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _chat_completion_response('{"answer": "да"}')

    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
    breaker.record_failure()  # уже открыт
    client = _make_client(handler, circuit_breaker=breaker)

    with pytest.raises(LlmApiRequestError):
        await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert call_count == 0


async def test_get_llm_response_opens_breaker_after_exhausted_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
    client = _make_client(handler, retries=1, circuit_breaker=breaker)

    with pytest.raises(LlmApiRequestError):
        await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert breaker.is_open() is True


async def test_get_llm_response_records_success_on_breaker():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('{"answer": "да"}')

    client = _make_client(handler, circuit_breaker=breaker)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert result.answer == 'да'
    assert breaker.is_open() is False
