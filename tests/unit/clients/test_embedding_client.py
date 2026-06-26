import httpx
import pytest

from app.clients.embeddings import EmbeddingClient
from app.exceptions.embedding import EmbeddingApiRequestError


def _make_client(handler, **overrides) -> EmbeddingClient:
    httpx_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EmbeddingClient(
        httpx_client=httpx_client,
        url='https://embeddings.example.com/v1/textEmbedding',
        headers={'Authorization': 'Api-Key test'},
        retries=overrides.pop('retries', 3),
        delay=overrides.pop('delay', 0.001),
        max_delay=overrides.pop('max_delay', 0.001),
        **overrides,
    )


async def test_get_embedding_returns_vector_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'embedding': [0.1, 0.2, 0.3], 'numTokens': '3'})

    client = _make_client(handler)

    result = await client.get_embedding(text='текст', model_uri='emb://folder/model/latest')

    assert result == [0.1, 0.2, 0.3]


async def test_get_embedding_retries_on_http_error_then_succeeds():
    attempts = {'count': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts['count'] += 1
        if attempts['count'] < 2:
            return httpx.Response(500, text='internal error')
        return httpx.Response(200, json={'embedding': [0.4, 0.5]})

    client = _make_client(handler, retries=3)

    result = await client.get_embedding(text='текст', model_uri='emb://folder/model/latest')

    assert result == [0.4, 0.5]
    assert attempts['count'] == 2


async def test_get_embedding_raises_after_exhausting_retries_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text='internal error')

    client = _make_client(handler, retries=2)

    with pytest.raises(EmbeddingApiRequestError):
        await client.get_embedding(text='текст', model_uri='emb://folder/model/latest')


async def test_get_embedding_raises_on_empty_embedding_field():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'embedding': []})

    client = _make_client(handler, retries=2)

    with pytest.raises(EmbeddingApiRequestError):
        await client.get_embedding(text='текст', model_uri='emb://folder/model/latest')


async def test_get_embedding_sends_vector_dimension_when_set():
    import json as _json

    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(_json.loads(request.content))
        return httpx.Response(200, json={'embedding': [0.1, 0.2], 'numTokens': '2'})

    client = _make_client(handler, vector_dimension=768)

    await client.get_embedding(text='текст', model_uri='emb://folder/model/latest')

    assert received[0].get('vectorDimension') == 768


async def test_get_embedding_omits_vector_dimension_when_none():
    import json as _json

    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(_json.loads(request.content))
        return httpx.Response(200, json={'embedding': [0.1, 0.2], 'numTokens': '2'})

    client = _make_client(handler)

    await client.get_embedding(text='текст', model_uri='emb://folder/model/latest')

    assert 'vectorDimension' not in received[0]
