import json

import httpx
import pytest
from pydantic import BaseModel

from app.clients.llm import LlmClient
from app.exceptions.llm import LlmApiRequestError


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


async def test_get_llm_response_strips_markdown_code_fence_before_validation():
    """Регрессия: YandexGPT почти всегда оборачивает JSON-ответ в ```...```
    (см. RAG_SERVICE_PLAN.md, Этап 3) — клиент должен снимать обёртку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('```json\n{"answer": "да"}\n```')

    client = _make_client(handler)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert result.answer == 'да'


async def test_get_llm_response_strips_markdown_emphasis_underscores_before_validation():
    """Регрессия: YandexGPT иногда добавляет markdown-эмфазис (_..._) вокруг
    значений строк прямо внутри JSON (см. RAG_SERVICE_PLAN.md, Этап 3)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response('{"answer": "_да, можно_"}')

    client = _make_client(handler)

    result = await client.get_llm_response(content='вопрос', prompt='system', schema=_EchoSchema)

    assert result.answer == 'да, можно'
